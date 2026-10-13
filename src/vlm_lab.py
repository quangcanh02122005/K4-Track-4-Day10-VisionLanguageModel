"""Lab Day 10: finetune Qwen3-VL-4B (QLoRA) cho grounding hư hại mặt đường.

Một file, một thí nghiệm mỗi lần chạy:
    python src/vlm_lab.py --name e1 --setup e1
setup: zeroshot (E0) | e1 (LoRA LLM) | e2 (+projector) | e3 (+LoRA ViT)
"""
import argparse
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
CLASSES = ("pothole", "crack", "manhole")

# ---------------------------------------------------------------- nhãn / prompt


def yolo_to_box(label, yolo, coord="norm1000", width=640, height=360):
    xc, yc, w, h = yolo
    x1, y1, x2, y2 = xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2
    if coord == "norm1000":
        sx, sy = 1000, 1000
    else:
        sx, sy = width, height
    box = [round(x1 * sx), round(y1 * sy), round(x2 * sx), round(y2 * sy)]
    box = [min(max(v, 0), sx if i % 2 == 0 else sy) for i, v in enumerate(box)]
    return {"label": label, "bbox_2d": box}


def objects_to_answer(objects, coord="norm1000", width=640, height=360):
    boxes = [yolo_to_box(o["label"], o["yolo"], coord, width, height) for o in objects]
    boxes.sort(key=lambda b: (b["bbox_2d"][1], b["bbox_2d"][0]))
    return json.dumps(boxes, separators=(",", ":"))


def make_prompt(coord="norm1000", width=640, height=360):
    if coord == "norm1000":
        scale = "integer coordinates on a 0-1000 scale (relative to image width and height)"
    else:
        scale = f"integer pixel coordinates (image is {width}x{height} pixels)"
    return (
        "Detect all road damage in this dashcam image. Classes: pothole, crack, manhole. "
        "Answer with ONLY a JSON list, no other text, in the form "
        '[{"label": "pothole", "bbox_2d": [x1, y1, x2, y2]}] using ' + scale +
        ". Answer [] if there is none."
    )


# ---------------------------------------------------------------- model


def load_model(setup):
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    q = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True, llm_int8_skip_modules=["model.visual", "visual", "lm_head"])
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, quantization_config=q, dtype=torch.float16, device_map={"": 0})
    bad = [n for n, m in model.named_modules()
           if "visual" in n and type(m).__name__ == "Linear4bit"]
    assert not bad, f"ViT vẫn bị lượng tử hoá 4-bit ({len(bad)} module, vd {bad[0]})"
    print("[load] ViT giữ fp16 (không Linear4bit)", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    processor.tokenizer.padding_side = "left"
    return model, processor


LLM_RE = r"model\.language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"
VIT_RE = r"model\.visual\.blocks\.\d+\.(attn\.(qkv|proj)|mlp\.(linear_fc1|linear_fc2))"


def apply_setup(model, setup, r=16):
    import torch
    from peft import LoraConfig, get_peft_model

    for p in model.parameters():
        p.requires_grad = False
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    target = LLM_RE if setup in ("e1", "e2") else f"{LLM_RE}|{VIT_RE}"
    cfg = LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.05, target_modules=target)
    pm = get_peft_model(model, cfg)
    n_lora = sum(1 for _, m in pm.named_modules() if hasattr(m, "lora_A") and len(m.lora_A) > 0)
    print(f"[setup {setup}] LoRA modules: {n_lora}", flush=True)
    if setup in ("e2", "e3"):
        n = 0
        for name, p in pm.named_parameters():
            if "visual.merger" in name or "visual.deepstack_merger_list" in name:
                p.data = p.data.float()
                p.requires_grad = True
                n += p.numel()
        print(f"[setup {setup}] projector params full-trained: {n / 1e6:.1f}M", flush=True)
    for p in pm.parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()  # GradScaler không unscale được gradient fp16
    trainable = sum(p.numel() for p in pm.parameters() if p.requires_grad)
    print(f"[setup {setup}] trainable params: {trainable:,}", flush=True)
    for n, m in pm.named_modules():
        if n.endswith(("visual.merger.linear_fc1", "visual.blocks.0.attn.qkv.base_layer",
                       "visual.deepstack_merger_list.0.linear_fc1")):
            w = getattr(m, "weight", None)
            print(f"[diag] {n}: {type(m).__name__} {tuple(w.shape) if w is not None else None} "
                  f"{w.dtype if w is not None else None}", flush=True)
    vis = sum(p.numel() for n, p in pm.named_parameters() if ".visual." in n and "lora_" not in n)
    print(f"[diag] visual params (numel): {vis:,}", flush=True)
    return pm, n_lora, trainable


# ---------------------------------------------------------------- dữ liệu huấn luyện


def user_msg(image, prompt):
    return {"role": "user", "content": [{"type": "image", "image": image},
                                        {"type": "text", "text": prompt}]}


class TrainSet:
    def __init__(self, entries, processor, data_dir, coord):
        self.entries, self.proc, self.dir, self.coord = entries, processor, Path(data_dir), coord

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        from PIL import Image
        e = self.entries[i]
        img = Image.open(self.dir / e["image_path"]).convert("RGB")
        prompt = make_prompt(self.coord, e["width"], e["height"])
        answer = objects_to_answer(e["objects"], self.coord, e["width"], e["height"])
        u = user_msg(img, prompt)
        a = {"role": "assistant", "content": [{"type": "text", "text": answer}]}
        kw = dict(tokenize=True, return_dict=True, return_tensors="pt")
        p_enc = self.proc.apply_chat_template([u], add_generation_prompt=True, **kw)
        f_enc = self.proc.apply_chat_template([u, a], add_generation_prompt=False, **kw)
        n = p_enc["input_ids"].shape[1]
        assert (f_enc["input_ids"][0, :n] == p_enc["input_ids"][0]).all(), "prompt không là tiền tố"
        labels = f_enc["input_ids"].clone()
        labels[:, :n] = -100
        out = {k: v for k, v in f_enc.items()}
        out["labels"] = labels
        return out


def collate(batch, pad_id):
    import torch
    out = {}
    seq_keys = [k for k in batch[0] if k not in ("pixel_values", "image_grid_thw")]
    maxlen = max(b["input_ids"].shape[1] for b in batch)
    for k in seq_keys:
        fill = pad_id if k == "input_ids" else (-100 if k == "labels" else 0)
        rows = []
        for b in batch:
            t = b[k][0]
            pad = torch.full((maxlen - t.shape[0],), fill, dtype=t.dtype)
            rows.append(torch.cat([t, pad]))
        out[k] = torch.stack(rows)
    for k in ("pixel_values", "image_grid_thw"):
        out[k] = torch.cat([b[k] for b in batch], dim=0)
    return out


def sanity_check(ds, processor):
    ex = ds[0]
    img_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    n_img = int((ex["input_ids"] == img_id).sum())
    sup = ex["labels"][0][ex["labels"][0] != -100]
    print(f"[check] image tokens = {n_img} (kỳ vọng 220)", flush=True)
    print(f"[check] supervised text = {processor.tokenizer.decode(sup)!r}", flush=True)


# ---------------------------------------------------------------- huấn luyện


def train(pm, ds, processor, steps, lr, bs, accum, seed):
    import torch
    from torch.utils.data import DataLoader

    pad_id = processor.tokenizer.pad_token_id
    g = torch.Generator().manual_seed(seed)
    dl = DataLoader(ds, batch_size=bs, shuffle=True, generator=g, num_workers=2,
                    collate_fn=lambda b: collate(b, pad_id), drop_last=True)
    params = [p for p in pm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    warm = max(1, int(0.05 * steps))

    def lam(s):
        if s < warm:
            return (s + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lam)
    scaler = torch.amp.GradScaler("cuda")
    pm.train()
    torch.cuda.reset_peak_memory_stats()
    it, losses = iter(dl), []
    t0 = time.time()
    for step in range(1, steps + 1):
        acc = 0.0
        for _ in range(accum):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(dl)
                batch = next(it)
            batch = {k: v.to("cuda") for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.float16):
                loss = pm(**batch).loss / accum
            if not torch.isfinite(loss):
                raise RuntimeError(f"loss không hữu hạn ở bước {step}: {loss.item()}")
            scaler.scale(loss).backward()
            acc += loss.item()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        sched.step()
        losses.append(acc)
        if step % 10 == 0 or step == 1:
            print(f"step {step}/{steps} loss={acc:.4f} lr={sched.get_last_lr()[0]:.2e} "
                  f"{(time.time() - t0) / 60:.1f}min", flush=True)
    minutes = (time.time() - t0) / 60
    peak = torch.cuda.max_memory_allocated() / 1024 ** 3
    return losses, minutes, peak


def save_trainable(pm, out_dir):
    import torch
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pm.save_pretrained(out_dir)
    extra = {n: p.detach().cpu() for n, p in pm.named_parameters()
             if p.requires_grad and "lora_" not in n}
    if extra:
        torch.save(extra, out_dir / "extra_trainable.pt")


# ---------------------------------------------------------------- suy luận


def infer(model, processor, entries, data_dir, coord, out_path, batch_size=4, max_new=256):
    import torch
    from PIL import Image
    model.eval()
    model.config.use_cache = True
    rows = []
    t0 = time.time()
    for i in range(0, len(entries), batch_size):
        chunk = entries[i:i + batch_size]
        convs = []
        for e in chunk:
            img = Image.open(Path(data_dir) / e["image_path"]).convert("RGB")
            convs.append([user_msg(img, make_prompt(coord, e["width"], e["height"]))])
        inputs = processor.apply_chat_template(
            convs, tokenize=True, add_generation_prompt=True, return_dict=True,
            return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
        texts = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                       skip_special_tokens=True)
        for e, t in zip(chunk, texts):
            rows.append({"image_id": e["image_id"], "raw": t})
        if (i // batch_size) % 10 == 0:
            print(f"infer {i + len(chunk)}/{len(entries)} {(time.time() - t0) / 60:.1f}min",
                  flush=True)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return rows


# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="tên file ra: work/<name>.jsonl, evidence/<name>.json")
    ap.add_argument("--setup", required=True, choices=["zeroshot", "e1", "e2", "e3"])
    ap.add_argument("--score-name", default=None)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--coord", default="norm1000", choices=["norm1000", "pixel"])
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--evidence-dir", default="report/evidence")
    ap.add_argument("--adapter-dir", default="/tmp/adapters")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    from scripts.labcommon import read_jsonl
    random.seed(a.seed)
    sd = Path(a.data_dir) / "splits"
    model, processor = load_model(a.setup)
    test = read_jsonl(sd / f"{a.split}.jsonl")
    if a.limit:
        test = test[:a.limit]
    meta_args = []
    run = {"name": a.name, "setup": a.setup, "r": a.r, "fraction": a.fraction,
           "coord": a.coord, "max_steps": a.max_steps}

    if a.setup == "zeroshot":
        engine = model
    else:
        import torch
        train_entries = read_jsonl(sd / "train.jsonl")
        if a.fraction < 1.0:
            k = max(1, int(len(train_entries) * a.fraction))
            train_entries = random.Random(a.seed).sample(train_entries, k)
        print(f"[train] {len(train_entries)} ảnh", flush=True)
        pm, n_lora, trainable = apply_setup(model, a.setup, a.r)
        ds = TrainSet(train_entries, processor, a.data_dir, a.coord)
        sanity_check(ds, processor)
        losses, minutes, peak = train(pm, ds, processor, a.max_steps, a.lr, 2, 4, a.seed)
        run.update(trainable_params=trainable, lora_modules=n_lora, peak_vram_gb=round(peak, 3),
                   train_minutes=round(minutes, 2), loss_first10=losses[:10], loss_last10=losses[-10:],
                   loss_curve=losses, train_images=len(train_entries))
        save_trainable(pm, Path(a.adapter_dir) / a.name)
        torch.cuda.empty_cache()
        pm.gradient_checkpointing_disable() if hasattr(pm, "gradient_checkpointing_disable") else None
        engine = pm
        meta_args = [f"trainable_params={trainable}", f"peak_vram_gb={peak:.3f}",
                     f"train_minutes={minutes:.2f}"]

    pred_path = Path("work") / f"{a.name}.jsonl"
    infer(engine, processor, test, a.data_dir, a.coord, pred_path)
    ev_dir = Path(a.evidence_dir)
    ev_dir.mkdir(parents=True, exist_ok=True)
    (ev_dir / f"{a.name}.run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    cmd = [sys.executable, "scripts/score.py", "--pred", str(pred_path), "--split", a.split,
           "--coord", a.coord, "--name", a.score_name or a.name.upper(),
           "--out", str(ev_dir / f"{a.name}.json"), "--splits-dir", str(sd)]
    for m in meta_args:
        cmd += ["--meta", m]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
