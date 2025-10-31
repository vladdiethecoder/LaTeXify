import hashlib, json, shutil, subprocess, tempfile
from pathlib import Path
from datetime import datetime
import fitz  # PyMuPDF

KB_ROOT   = Path("kb/course/GEN_BASELINE")
INBOX     = Path("data/inbox")
MANIFEST  = KB_ROOT / "manifest.jsonl"

def sha256sum(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()

def pdf_is_born_digital(path: Path) -> bool:
    try:
        doc = fitz.open(path)
        text = doc[0].get_text("text")
        return bool(text.strip())
    except Exception:
        return False

def rasterize_pdf_to_images(pdf_path: Path, dpi=300) -> list[Path]:
    tmpdir = Path(tempfile.mkdtemp())
    outprefix = tmpdir / "page"
    cmd = ["pdftoppm", "-png", "-r", str(dpi), str(pdf_path), str(outprefix)]
    subprocess.run(cmd, check=True)
    return sorted(tmpdir.glob("page-*.png"))

def target_subdir(src: Path) -> Path:
    s = src.name.lower()
    if "syllabus" in s: return KB_ROOT / "syllabus"
    if "lecture" in s or s.startswith("l0"): return KB_ROOT / "lectures"
    if "assign" in s or s.startswith("a"): return KB_ROOT / "assignments"
    return KB_ROOT / "notes"

def normalize_to_markdown(src: Path, dst_md: Path) -> str:
    text = ""
    if src.suffix.lower() == ".pdf" and pdf_is_born_digital(src):
        doc = fitz.open(src)
        pages = [f"# [Page {i+1}]\n{page.get_text('text')}" for i, page in enumerate(doc)]
        text = "\n\n".join(pages).strip()
    else:
        from scripts.ocr_nanonets import ocr_page_to_markdown  # lazy import
        images = []
        if src.suffix.lower() == ".pdf":
            images = rasterize_pdf_to_images(src, dpi=300)
        else:
            images = [src]
        md_pages = [f"# [Page {i+1}]\n{ocr_page_to_markdown(str(img))}" for i, img in enumerate(images)]
        text = "\n\n".join(md_pages).strip()
    dst_md.parent.mkdir(parents=True, exist_ok=True)
    dst_md.write_text(text, encoding="utf-8")
    return text

def write_manifest_entry(meta: dict):
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("a", encoding="utf-8") as f:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")

def ingest_files():
    assert INBOX.exists(), f"Missing {INBOX}"
    processed = []
    for p in sorted(INBOX.iterdir()):
        if p.is_dir(): continue
        subdir = target_subdir(p)
        subdir.mkdir(parents=True, exist_ok=True)
        raw_dest = subdir / p.name
        shutil.copy2(p, raw_dest)
        md_dest = (subdir / p.stem).with_suffix(".md")
        _ = normalize_to_markdown(raw_dest, md_dest)
        write_manifest_entry({
            "source_id": raw_dest.stem,
            "title": raw_dest.name,
            "type": raw_dest.suffix.lower().lstrip("."),
            "file": str(raw_dest),
            "normalized_md": str(md_dest),
            "course_id": "GEN_BASELINE",
            "created": datetime.utcnow().isoformat(timespec="seconds")+"Z",
            "hash": sha256sum(raw_dest),
            "license": "personal",
        })
        processed.append(str(raw_dest))
    print(json.dumps({"ingested": processed}, indent=2))

if __name__ == "__main__":
    ingest_files()
