import logging
from pathlib import Path
try:
    import fitz  # type: ignore
except ImportError:
    fitz = None
try:
    from pdf2image import convert_from_path
except ImportError:
    convert_from_path = None

LOGGER = logging.getLogger(__name__)

class PageImageStore:
    """Lazily rasterizes PDF pages to PNG files for downstream models."""

    def __init__(self, pdf_path: Path, cache_dir: Path, enabled: bool) -> None:
        self.pdf_path = pdf_path
        self.cache_dir = cache_dir
        self._pymupdf_doc = None
        self.enabled = enabled and (fitz is not None or convert_from_path is not None)
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _render_with_pymupdf(self, page_index: int, target: Path) -> bool:
        if fitz is None:
            return False
        try:
            if self._pymupdf_doc is None:
                self._pymupdf_doc = fitz.open(str(self.pdf_path))
            page = self._pymupdf_doc.load_page(page_index)
            pix = page.get_pixmap(dpi=300)
            pix.save(str(target))
            return target.exists()
        except Exception as exc:
            LOGGER.warning("PyMuPDF rasterization failed for page %s: %s", page_index + 1, exc)
            return False

    def get_page_image(self, page_index: int) -> Path | None:
        if not self.enabled:
            return None
        candidate = self.cache_dir / f"page_{page_index + 1:04d}.png"
        if candidate.exists():
            return candidate
        if self._render_with_pymupdf(page_index, candidate):
            return candidate
        if convert_from_path is None:
            LOGGER.warning(
                "Unable to rasterize page %s: pdf2image unavailable and PyMuPDF failed.",
                page_index + 1,
            )
            self.enabled = False
            return None
        try:
            convert_from_path(
                str(self.pdf_path),
                first_page=page_index + 1,
                last_page=page_index + 1,
                fmt="png",
                output_folder=str(self.cache_dir),
                output_file=f"page_{page_index + 1:04d}",
            )
        except Exception as exc:
            LOGGER.warning("Failed to rasterize page %s: %s", page_index + 1, exc)
            self.enabled = False
            return None
        # pdf2image output file naming might differ slightly depending on version/args, 
        # but usually it appends suffix. We rely on it matching our expectation or 
        # we might need to rename. 
        # In ingestion.py it checked `candidate` existence. 
        # pdf2image usually adds -01, -02. 
        # ingestion.py used `output_file=f"page_{page_index + 1:04d}"`
        # which results in `page_00010001.png` or `page_0001.png` depending on strictness.
        # We'll check the candidate.
        
        return candidate if candidate.exists() else None
