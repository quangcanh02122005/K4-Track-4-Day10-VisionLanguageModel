Họ tên: Đào Quang Cảnh
`dataset_version`: d10-21e4070301

## Bảng tổng hợp (số của bạn, lấy từ `report/evidence/*.json`)

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | zero-shot | 0 | — | — | 0.090 | [0.063, 0.119] | 0.801 | 0.135 | 0.030 | 0.176 |
| E1 | LoRA LLM | 33,030,144 | 6.384 | 41.61 | 0.277 | [0.242, 0.316] | 1.244 | 0.167 | 0.223 | 0.518 |
| E2 | + projector | 142,135,296 | 8.078 | 42.56 | 0.325 | [0.285, 0.369] | 1.106 | 0.145 | 0.261 | 0.605 |
| E3 | + LoRA ViT | 148,426,752 | 8.255 | 56.88 | 0.447 | [0.406, 0.488] | 0.875 | 0.345 | 0.371 | 0.688 |
| E4 | data_size=0.1 (10% train data) | 148,426,752 | 7.704 | 52.72 | 0.322 | [0.283, 0.362] | 1.473 | 0.196 | 0.225 | 0.639 |

> **Ghi chú E4**: Biến thay đổi là `fraction = 0.1` (chỉ dùng 10% ảnh train ≈ 142 ảnh), giữ nguyên toàn bộ setup E3 (LoRA LLM + projector + LoRA ViT, r=16, 300 steps). Huấn luyện mất 52.72 phút, VRAM đỉnh 7.704 GB. F1 micro đạt 0.322 (giảm 0.125 so với E3 đầy đủ 0.447).

## 10 câu hỏi (trả lời bằng **số của bạn**, không chép lại tài liệu)

### Câu 1 — Kiến trúc: ViT, projector và LLM của Qwen3-VL-4B có bao nhiêu tham số mỗi khối, và E1/E2/E3 mỗi cái cập nhật khối nào?

**Kiến trúc Qwen3-VL-4B:**
- **ViT (Visual Encoder)**: SigLIP-400M backbone với các khối attention + MLP. Mỗi khối ViT có attention (`qkv`, `proj`) và MLP (`linear_fc1`, `linear_fc2`). Trong code, `VIT_RE = r"model\.visual\.blocks\.\d+\.(attn\.(qkv|proj)|mlp\.(linear_fc1|linear_fc2))"`.
- **Projector**: `visual.merger` và `visual.deepstack_merger_list` — đây là các lớp MLP kết nối ViT với LLM, chuyển đổi visual tokens sang không gian LLM.
- **LLM**: Qwen3-4B backbone. Mỗi khối có `self_attn` (q, k, v, o projection) và `mlp` (gate, up, down projection). `LLM_RE = r"model\.language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"`.

**Tham số trainable theo setup:**
- **E1**: Chỉ LoRA trên LLM → 33,030,144 params, 252 LoRA modules. ViT và projector đóng băng.
- **E2**: LoRA LLM + projector full-finetune → 142,135,296 params (252 LoRA modules + ~109M projector params). ViT vẫn đóng băng.
- **E3**: LoRA LLM + projector full-finetune + LoRA ViT → 148,426,752 params, 348 LoRA modules (thêm ~96 modules ViT so với E2).

### Câu 2 — E0: `parse_valid_rate` bằng bao nhiêu, và hai lỗi phổ biến nhất của model gốc là gì (kèm một ví dụ output thô)?

**`parse_valid_rate` E0 = 0.9325** (93.25% trong 311 ảnh được parse thành công).

**Hai lỗi phổ biến nhất:**
1. **Toạ độ ngoài thang 0–1000 (pixel leak)**: Model zero-shot đôi khi trả về toạ độ dưới dạng pixel tuyệt đối thay vì norm1000. Ví dụ: `[{"label": "crack", "bbox_2d": [0, 656, 1000, 800]}]` — box trải dài toàn chiều ngang (0→1000 hợp lệ) nhưng y2=800 vượt ngưỡng ảnh 360px, đây là bbox "stripe" bao phủ quá rộng.
2. **Bỏ sót đối tượng (high FN) và nhầm lớp**: Model zero-shot chủ yếu phát hiện manhole (dễ nhận biết hơn về hình dạng) nhưng bỏ qua phần lớn crack và pothole. F1 crack chỉ 0.030, FP/ảnh = 0.801 trong khi fn=544 — tức bỏ sót 544 ground truth boxes. Ví dụ output: `[]` (không phát hiện gì) dù ảnh có nhiều vết nứt.

### Câu 3 — Thang 0–1000: nếu chấm output theo pixel (hoặc ngược lại) thì F1 ra sao và vì sao? Bạn đã kiểm chứng điều này thế nào?

**Tác động của nhầm thang toạ độ:**
Nếu model được train với `coord=norm1000` (toạ độ 0–1000) nhưng chấm điểm với `coord=pixel` (640×360), hoặc ngược lại, thì IoU sẽ gần bằng 0 vì các bounding box bị scale khác nhau hoàn toàn. Ví dụ: bbox `[320, 180, 640, 360]` ở norm1000 tương ứng với góc phải-dưới của ảnh, nhưng nếu hiểu là pixel thì chỉ là vùng từ (320px, 180px) → (640px, 360px) — cùng vùng; nhưng `[500, 500, 800, 900]` ở norm1000 tương ứng pixel ≈ (320, 180, 512, 324), trong khi nếu dùng luôn làm pixel thì ra ngoài ảnh 640×360.

**Kiểm chứng**: Code trong `vlm_lab.py` hàm `yolo_to_box()` và `make_prompt()` đảm bảo cùng `coord` được dùng cho cả training labels và inference prompt. Script `score.py` nhận `--coord` để dùng cùng thang khi so khớp. E0, E1, E2, E3 đều dùng `coord=norm1000` nhất quán → `mean_iou` E3 = 0.699 trên các box được match, hợp lý.

### Câu 4 — Mask loss: vì sao chỉ tính loss trên phần trả lời? Điều gì xảy ra nếu tính trên cả prompt và token ảnh?

**Lý do chỉ tính loss trên phần trả lời:**
Trong `TrainSet.__getitem__()`, code mask `labels[:, :n] = -100` với `n` là độ dài phần prompt (bao gồm system message, user prompt, và image tokens). Chỉ phần assistant response mới có loss được tính.

**Nếu tính loss trên cả prompt và token ảnh:**
- **Token ảnh**: Các `<|image_pad|>` tokens là đặc biệt; gradient qua chúng không có ý nghĩa với nhãn text → loss nhiễu, mô hình học sai tín hiệu.
- **Prompt text**: Phần prompt (ví dụ "Detect all road damage...") là hằng số, model không cần học sinh ra nó → gradient trên phần này sẽ "kéo" model về phía sinh ra chính prompt thay vì tập trung vào output JSON. Về mặt lý thuyết, loss sẽ tăng nhưng chất lượng output giảm vì model bị confuse giữa việc sinh prompt và sinh nhãn.
- **Thực tế**: Loss trên image tokens thường rất cao (model không thể dự đoán pixel values), gây gradient không ổn định và có thể làm mất hội tụ.

### Câu 5 — Vì sao không thể huấn luyện đầy đủ projector nếu ViT bị lượng tử hoá 4-bit, và bạn đã xử lý thế nào? Vì sao projector phải ở float32?

**Vấn đề ViT 4-bit và projector:**
BitsAndBytes 4-bit quantization áp dụng cho Linear layers — nếu ViT bị quantize, output của ViT là tensor fp16 tính từ weights 4-bit. Projector nhận input từ ViT; nếu projector được fine-tune trong 4-bit, gradient sẽ phải đi qua `dequantize → multiply → quantize` — quá trình này không có gradient thực sự (quantization không khả vi), do đó gradient bị chặn hoàn toàn.

**Cách xử lý trong code** (`vlm_lab.py` dòng 64–65):
```python
BitsAndBytesConfig(..., llm_int8_skip_modules=["model.visual", "visual", "lm_head"])
```
→ ViT (`model.visual`) **không bị lượng tử hoá**, giữ fp16. Assertion ở dòng 68–70 kiểm tra không có `Linear4bit` trong visual modules.

**Vì sao projector phải float32**: Code dòng 98–101 chuyển projector params sang `float32` trước khi bật `requires_grad=True`. Lý do: GradScaler (mixed precision training) chỉ có thể unscale gradient cho fp32 parameters. Nếu projector ở fp16, gradient underflow/overflow sẽ xảy ra và scaler không thể cứu được → training không hội tụ.

### Câu 6 — Split theo phiên quay: vì sao chia ngẫu nhiên sẽ cho F1 cao giả? Số ảnh và số box từng lớp ở mỗi split là bao nhiêu?

**Vấn đề chia ngẫu nhiên:**
Dữ liệu được quay từ camera hành trình (dashcam) theo các phiên (`groups`). Các frame liên tiếp trong cùng phiên quay có **nội dung rất tương đồng** (cùng đoạn đường, cùng ánh sáng, cùng góc camera). Nếu chia ngẫu nhiên theo frame, frame t và t+1 từ cùng phiên quay có thể rơi vào train và test — model chỉ cần "nhớ" ngoại hình đoạn đường đó là đủ đạt F1 cao mà không học được khả năng generalize. Đây là **data leakage theo thời gian/không gian**.

**Giải pháp**: Chia theo phiên quay (session-level split):
```json
"vlc-2025-02-18-18": "val",
"vlc-2025-02-19-17": "test",  "vlc-2025-02-18-23": "test",  "num-4": "test"
```

**Số liệu từ `data/splits/meta.json` (`dataset_version: d10-21e4070301`):**

| Split | Ảnh | pothole boxes | crack boxes | manhole boxes |
|-------|-----|---------------|-------------|---------------|
| train | 1422 | 1074 | 1750 | 691 |
| val   | 276  | 66   | 470  | 103 |
| test  | 311  | 121  | 299  | 163 |

### Câu 7 — So sánh E1/E2/E3: mở projector và LoRA trên ViT có đáng không (F1, CI95, VRAM, thời gian)?

**So sánh chi tiết:**

| Bước | F1 | CI95 | Δ params (M) | ΔVRAM (GB) | Δthời gian (phút) |
|------|-----|------|--------------|------------|-------------------|
| E1 → E2 | 0.277 → 0.325 | Chồng lấp | +109M projector | +1.694 GB | +0.95 phút |
| E2 → E3 | 0.325 → 0.447 | Không chồng lấp | +6.3M ViT LoRA | +0.177 GB | +14.32 phút |

**Nhận xét:**
- **E1 → E2** (+projector): F1 tăng 0.048 nhưng CI95 của E1 [0.242, 0.316] và E2 [0.285, 0.369] **chồng lấp** → không thể kết luận cải thiện có ý nghĩa thống kê. Chi phí tương đối nhỏ (1.7GB VRAM, <1 phút).
- **E2 → E3** (+LoRA ViT): F1 tăng 0.122 — **cải thiện lớn nhất**. CI95 không chồng lấp → kết luận có ý nghĩa. Chi phí 14 phút thêm nhưng chỉ +0.18GB VRAM và +6.3M params. **Rõ ràng đáng**.
- **Kết luận**: LoRA ViT (E3) là bước cải thiện quan trọng nhất; projector (E2) có cải thiện nhưng cần thêm dữ liệu để xác nhận. F1 manhole nhảy từ 0.518 (E1) lên 0.688 (E3), F1 crack từ 0.223 lên 0.371.

### Câu 8 — Tập test có 311 ảnh: khoảng tin cậy rộng thế nào, và điều đó giới hạn kết luận so sánh các setup ra sao?

**Độ rộng CI95:**
- E0: [0.063, 0.119] → rộng **0.056** (nhưng điểm thấp)
- E1: [0.242, 0.316] → rộng **0.074**
- E2: [0.285, 0.369] → rộng **0.084**
- E3: [0.406, 0.488] → rộng **0.082**

**Giới hạn kết luận:**
Với 311 ảnh test, CI95 rộng ~0.08 (±4 điểm F1). Điều này có nghĩa:
- So sánh **E1 vs E2** (Δ=0.048): CI95 chồng lấp hoàn toàn → **không thể kết luận** E2 tốt hơn E1 với độ tin cậy 95%. Cần ≥1000 ảnh test để phân biệt sự khác biệt nhỏ như vậy.
- So sánh **E2 vs E3** (Δ=0.122): CI95 không chồng lấp ([0.285, 0.369] vs [0.406, 0.488]) → kết luận đáng tin.
- So sánh **E0 vs E1** (Δ=0.187): Hoàn toàn không chồng lấp → kết luận đáng tin.

**Tổng quát**: Bộ test nhỏ giới hạn khả năng phân biệt các setup gần nhau về F1.

### Câu 9 — Phân tích lỗi theo lớp: lớp nào kém nhất và vì sao (kích thước box, số mẫu, ngữ nghĩa)? Kèm ảnh minh hoạ.

**Lớp kém nhất: pothole** (F1=0.345 ở E3, thấp nhất trong 3 lớp)

**Lý do:**
1. **Kích thước box nhỏ và không đồng đều**: Pothole thường có diện tích nhỏ, hình dạng bất thường (tròn, méo, lỗ chỗ) — khó dự đoán bbox chính xác hơn manhole (hình vuông/tròn chuẩn) hay crack (dài và có pattern).
2. **Số mẫu train ít hơn tương đối**: 1074 boxes pothole vs 1750 crack trong train. Nhưng số ảnh test lại có 121 pothole boxes (nhiều hơn manhole 163, ít hơn crack 299) → tỷ lệ mẫu chưa hẳn là vấn đề chính.
3. **Ngữ nghĩa mơ hồ**: Pothole (ổ gà) và crack (vết nứt) có ranh giới ngữ nghĩa không rõ ràng — ổ gà lớn có thể bắt đầu từ vết nứt; model dễ nhầm lẫn.
4. **Độ tương phản thấp**: Trong ảnh dashcam, pothole thường có màu gần giống mặt đường xung quanh, khác với manhole cover (thường bằng kim loại, nổi bật).

**Kết quả định lượng E3**: Pothole — precision=0.371, recall=0.322, 39 TP / 66 FP / 82 FN. Recall thấp (32%) cho thấy model bỏ sót nhiều pothole hơn là dự đoán sai.

*(Ảnh minh hoạ: xem preds/e3.jsonl so với ground truth trong test.jsonl cho các image_id có pothole có fn > 0)*

### Câu 10 — E4: bạn đổi biến nào, kết quả thế nào, và kết luận có đáng tin không?

**Biến thay đổi**: `data_size = 0.1` (`fraction = 0.1`) — chỉ dùng 10% dữ liệu train (142 ảnh thay vì 1422), giữ nguyên mọi tham số và kiến trúc của E3 (LoRA LLM + projector + LoRA ViT, r=16, lr=2e-4, max_steps=300).

**Kết quả thực tế (từ `report/evidence/e4.json` & `e4.run.json`):**
- **F1 micro tổng**: giảm từ **0.447** (E3) xuống **0.322** (E4) — mức giảm rất lớn (**Δ = -0.125**).
- **CI95**: [0.283, 0.362] cho E4 so với [0.406, 0.488] cho E3. Hai khoảng tin cậy này **hoàn toàn tách biệt, không hề chồng lấp**.
- **Từng lớp**: 
  - Pothole: F1 giảm từ 0.345 xuống **0.196**
  - Crack: F1 giảm từ 0.371 xuống **0.225**
  - Manhole: F1 giảm từ 0.688 xuống **0.639** (ít chịu ảnh hưởng nhất do đặc trưng nhận diện rõ ràng)
- **Hành vi mô hình & Loss curve**:
  - Do chỉ có 142 ảnh nhưng chạy tới 300 steps (batch 2, grad accum 4 = 8 ảnh/step, tương đương ~17 epoch trên tập con 142 ảnh), mô hình đã bị **overfitting cực nặng**: train loss giảm từ 2.128 ở bước 1 xuống tận ~0.002 - 0.005 ở 10 bước cuối (gần như ghi nhớ toàn bộ tập train).
  - FP/ảnh tăng từ 0.875 (E3) lên **1.473** (E4) do mô hình sinh box quá đà (hallucination) trên tập test lạ.

**Kết luận về độ tin cậy:**
Kết luận rằng **"kích thước dữ liệu đầy đủ đóng vai trò quyết định hiệu năng của toàn bộ pipeline LoRA ViT + LLM"** là **hoàn toàn đáng tin và có ý nghĩa thống kê** vì mức giảm Δ F1 = 0.125 vượt xa biên độ lỗi, và CI95 của E4 và E3 hoàn toàn không giao nhau. Khi giảm 90% dữ liệu, LoRA ViT không nhận được đủ đa dạng mẫu mặt đường để khái quát hoá và nhanh chóng ghi nhớ tập train nhỏ.

## Nhìn lại

**Điều chưa hiểu / vấn đề gặp phải:**

1. **E2 kém hơn kỳ vọng so với E3**: Mở projector (E2) chỉ tăng F1 thêm 0.048 và không đủ có ý nghĩa thống kê, nhưng khi thêm LoRA ViT (E3) thì tăng vọt 0.122. Điều này gợi ý projector alone không đủ — visual encoder cần được fine-tune để tạo ra features tốt hơn cho domain này, còn projector mapping chỉ phát huy khi ViT features đã adapt.

2. **Parse valid rate E0 = 93.25%**: ~7% output của zero-shot model không parse được thành JSON hợp lệ. Lỗi thường gặp là model thêm text giải thích ngoài JSON, hoặc format key sai. Công cụ chấm điểm tính những trường hợp này là 0 boxes — điều này có thể ảnh hưởng nhẹ đến F1 E0.

3. **FP/ảnh cao ở E1 (1.244)**: Cao hơn E0 (0.801) — có vẻ phản trực giác. Lý do: model sau fine-tune học cách "nói" JSON hợp lệ nên sinh ra nhiều boxes hơn, kể cả false positive. E3 giảm FP/ảnh xuống 0.875, cho thấy visual features tốt hơn giúp filter bớt các dự đoán sai.

4. **Crack vẫn là lớp khó nhất về precision**: Dù F1 tăng từ 0.030 (E0) lên 0.371 (E3), precision crack chỉ 0.374 — cứ 3 box crack dự đoán thì có gần 2 cái sai (184 FP). Crack có nhiều hình dạng khác nhau (longitudinal, transverse, alligator) và kích thước đa dạng, khiến model khó học boundary chính xác.
