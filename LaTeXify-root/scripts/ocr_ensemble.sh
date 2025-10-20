set -euo pipefail
IMG="$1"
OUTDIR="data/ocr_runs/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUTDIR"

# 1) NN2
source .venv-nn2/bin/activate
python scripts/bin/nn2_ocr.py "$IMG" > "$OUTDIR/nn2.txt"
deactivate

# 2) DoTS
source .venv-dots/bin/activate
python scripts/bin/dots_ocr.py "$IMG" > "$OUTDIR/dots.txt"
deactivate

# 3) Qwen2.5-VL-3B-Instruct (add a scripts/bin/qwen_ocr.py similar to nn2_ocr.py)
source .venv-qwen/bin/activate
python scripts/bin/qwen_ocr.py "$IMG" > "$OUTDIR/qwen.txt"
deactivate

# 4) Florence2 (create .venv-florence + scripts/bin/florence_ocr.py)
source .venv-florence/bin/activate
python scripts/bin/florence_ocr.py "$IMG" > "$OUTDIR/florence.txt"
deactivate

# 5) TrOCR/Nougat (create .venv-trocr + scripts/bin/trocr_ocr.py)
source .venv-trocr/bin/activate
python scripts/bin/trocr_ocr.py "$IMG" > "$OUTDIR/trocr.txt"
deactivate

jq -n --arg out "$OUTDIR" '{outputs_dir:$out, models: ["nn2","dots","qwen","florence","trocr"]}'
