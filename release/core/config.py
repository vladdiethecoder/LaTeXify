"""Structured configuration for ingestion/backend toggles."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Literal

OCRBackendChoice = Literal["florence", "ensemble", "mineru"]
MathOCRChoice = Literal["none", "pix2tex", "latex-ocr"]


@dataclass
class BackendToggleConfig:
    """Flags that describe which ingestion/math OCR backends should be active."""

    ocr_backend: OCRBackendChoice = "ensemble"
    mineru_enabled: bool = False
    marker_enabled: bool = False
    mcp_pdf_processor_enabled: bool = False
    math_ocr_backend: MathOCRChoice = "none"

    def resolve_ingestion_mode(self) -> str:
        """Map the high-level backend selection onto the ingestion OCR mode."""

        mapping = {
            "florence": "florence2",
            "ensemble": "auto",
            "mineru": "auto",  # placeholder until MinerU adapter lands
        }
        return mapping.get(self.ocr_backend, "auto")

    def as_dict(self) -> Dict[str, object]:
        """Materialize the toggle state for telemetry/manifest output."""

        return asdict(self)

    def wants_math_ocr(self) -> bool:
        return self.math_ocr_backend != "none"
