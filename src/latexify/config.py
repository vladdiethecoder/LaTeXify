from typing import Literal, List, Optional, Dict, Union
from pathlib import Path
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

OCRBackendChoice = Literal["auto", "nougat", "florence2", "internvl", "qwenvl", "mathvision", "mathocr", "pytesseract", "monkeyocr", "none"]
MathOCRChoice = Literal["none", "pix2tex", "latex-ocr"]
VisionPresetChoice = Literal["balanced", "fast", "quality"]

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LATEXIFY_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # --- Backend Config ---
    ocr_backend: OCRBackendChoice = Field("auto", description="Primary OCR backend")
    layout_backend: Literal["pymupdf", "surya"] = Field("pymupdf", description="Layout analysis backend")
    
    # --- Model Paths ---
    # Defaults pointing to relative paths, resolved at runtime if needed
    models_dir: Path = Field(Path("src/latexify/models"), description="Root directory for model weights")
    hf_home: Optional[Path] = Field(None, validation_alias="HF_HOME")

    # --- Feature Flags ---
    enable_vision_synthesis: bool = Field(True, description="Enable multi-view vision synthesis")
    enable_monkey_ocr: bool = Field(True, description="Enable MonkeyOCR adapter if available")
    enable_robust_compilation: bool = Field(True, description="Enable incremental compilation loop")
    
    # --- Resource Management ---
    ocr_vram_headroom_gb: float = Field(1.0, description="VRAM headroom to leave free (GB)")
    force_gpu_ocr: Union[bool, str] = Field(False, description="Force GPU usage or specific device")
    
    # --- Model Specifics ---
    internvl_model: str = "OpenGVLab/InternVL3_5-8B"
    qwenvl_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    
    # --- Compilation ---
    compilation_retry_count: int = 3
    layout_confidence_threshold: float = 0.0

    @field_validator("ocr_vram_headroom_gb", mode="before")
    @classmethod
    def parse_vram(cls, v):
        if v is None: return 1.0
        return float(v)

    def resolve_models_dir(self) -> Path:
        return self.models_dir.resolve()

# Global instance
settings = Settings()
