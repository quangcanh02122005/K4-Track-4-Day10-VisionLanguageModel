"""Đóng gói scripts/ + src/ thành một notebook Kaggle tự chạy.

    python tools/build_kernel.py --out <file.ipynb> --run "name=e1 setup=e1" --run "name=e2 setup=e2"
Mỗi --run là một lần gọi src/vlm_lab.py (các cặp key=value thành --key value).
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ZIP_URL = ("https://www.kaggle.com/api/v1/datasets/download/lorenzoarcioni/"
           "road-damage-dataset-potholes-cracks-and-manholes")

TEMPLATE = r'''
import json, os, subprocess, sys, shutil, time
from pathlib import Path

FILES = __FILES__
RUNS = __RUNS__
ZIP_URL = "__ZIP_URL__"

ROOT = Path("/kaggle/working/lab")
for rel, text in FILES.items():
    p = ROOT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
os.chdir(ROOT)

def sh(cmd, check=True):
    print("$", " ".join(cmd), flush=True)
    r = subprocess.run(cmd)
    if check and r.returncode:
        raise SystemExit(f"lệnh lỗi ({r.returncode}): {cmd}")
    return r.returncode

sh(["nvidia-smi", "-L"], check=False)
sh([sys.executable, "-m", "pip", "-q", "install", "transformers==5.19.0", "peft==0.21.2",
    "accelerate==1.15.0", "bitsandbytes==0.50.2", "pyyaml", "pillow"])
sh([sys.executable, "scripts/prepare_data.py", "--zip", ZIP_URL])

status = {}
for run in RUNS:
    cmd = [sys.executable, "src/vlm_lab.py"]
    for kv in run.split():
        k, v = kv.split("=", 1)
        cmd += ["--" + k.replace("_", "-"), v]
    t0 = time.time()
    status[run] = sh(cmd, check=False)
    print(f"=== {run}: exit {status[run]} sau {(time.time()-t0)/60:.1f} phút", flush=True)

out = Path("/kaggle/working/out")
out.mkdir(exist_ok=True)
for src in ["report/evidence", "work", "data/splits"]:
    d = Path(src)
    if d.exists():
        shutil.copytree(d, out / d.name if src != "data/splits" else out / "splits", dirs_exist_ok=True)
(out / "status.json").write_text(json.dumps(status, indent=2))
shutil.rmtree("/kaggle/working/lab/data", ignore_errors=True)
print("XONG", status)
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--run", action="append", required=True)
    a = ap.parse_args()
    files = {}
    for rel in ["scripts/__init__.py", "scripts/labcommon.py", "scripts/splitting.py",
                "scripts/prepare_data.py", "scripts/score.py", "src/vlm_lab.py"]:
        files[rel] = (ROOT / rel).read_text(encoding="utf-8")
    code = (TEMPLATE.replace("__FILES__", repr(files)).replace("__RUNS__", repr(a.run))
            .replace("__ZIP_URL__", ZIP_URL))
    nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {},
                     "outputs": [], "source": code.splitlines(keepends=True)}],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                      "name": "python3"}},
          "nbformat": 4, "nbformat_minor": 5}
    Path(a.out).write_text(json.dumps(nb), encoding="utf-8")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
