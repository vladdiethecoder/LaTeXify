# Offline PDF → LaTeX (Docling + Nougat + pix2tex)

Fully local pipeline for Fedora Linux, aligned with the “Offline LaTeX PDF Conversion Pipeline (Fedora Linux)” blueprint. Uses Docling for layout + OCR, Nougat for page OCR, and pix2tex for math fallback. This integrates into the existing `LaTeXifyPipeline` via the `docling` ingestion backend.

## System setup (Fedora)

```bash
sudo dnf install -y tesseract tesseract-devel leptonica-devel tesseract-langpack-eng tesseract-osd \
  poppler-cpp texlive-scheme-full latexmk
# Optional: set Tesseract data path
export TESSDATA_PREFIX="/usr/share/tesseract/tessdata/"
```

Install Python deps (inside your venv):

```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
pip install "docling[vlm,tesserocr]" nougat-ocr pix2tex
# optional extras
pip install easyocr rapidocr onnxruntime
```

GPU: ensure `nvidia-smi` works and CUDA libraries match your PyTorch wheel.

## Usage

`config/pipeline.yaml` now defaults to `ingestion.backend: docling` with VLM + OCR. Run the pipeline as usual:

```bash
python run_latexify.py --pdf path/to/input.pdf
```

Behavior:
- Docling does layout + OCR; if Docling is missing or fails, the pipeline falls back to MinerU → PyMuPDF.
- Blocks are mapped to `DocumentState.layout_blocks`/`chunks` for downstream planning/synthesis.
- Math-heavy regions can be refined later by existing Nougat/pix2tex adapters in vision branches.

## Notes & troubleshooting
- If Docling is installed but you want the legacy path, set `pipeline.ingestion.backend: mineru` in `config/pipeline.yaml`.
- For low-confidence math from Docling, rely on the existing pix2tex/Nougat adapters during vision synthesis; Docling is only the ingestion step.
- Compile with `latexmk -pdf -interaction=nonstopmode output.tex` to verify output locally.
