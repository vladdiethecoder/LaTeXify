from typing import Literal, List, Optional, Dict, Union
from pathlib import Path
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

OCRBackendChoice = Literal["auto", "nougat", "florence2", "internvl", "qwenvl", "mathvision", "mathocr", "pytesseract", "monkeyocr", "none", "ensemble"]
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

    # --- Backend Config (Optimized for 5090/32GB) ---
    ocr_backend: OCRBackendChoice = Field("ensemble", description="Primary OCR backend")
    layout_backend: Literal["pymupdf", "surya", "monkeyocr"] = Field("surya", description="Layout analysis backend")
    
    # --- Model Paths ---
    models_dir: Path = Field(Path("src/latexify/models"), description="Root directory for model weights")
    hf_home: Optional[Path] = Field(None, validation_alias="HF_HOME")

    # --- Feature Flags (Quality First) ---
    enable_vision_synthesis: bool = Field(True, description="Enable multi-view vision synthesis")
    enable_monkey_ocr: bool = Field(True, description="Enable MonkeyOCR adapter if available")
    enable_robust_compilation: bool = Field(True, description="Enable incremental compilation loop")
    
    # --- Vision & Refinement ---
    vision_preset: VisionPresetChoice = Field("quality", description="Vision synthesis preset")
    llm_mode: str = Field("auto", description="LLM refinement mode")
    
    # --- Resource Management (Safe for 32GB VRAM / 32GB RAM) ---
    ocr_vram_headroom_gb: float = Field(2.0, description="VRAM headroom to leave free (GB)")
    ocr_release_mode: Literal["run", "page"] = Field("page", description="Aggressively free VRAM after each page")
    force_gpu_ocr: Union[bool, str] = Field(False, description="Force GPU usage or specific device")
    
    # --- Model Specifics ---
    internvl_model: str = "OpenGVLab/InternVL3_5-8B"
    qwenvl_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    
    # --- Compilation ---
    compilation_retry_count: int = 5
    layout_confidence_threshold: float = 0.0
    
    # --- Orchestration ---
    max_attempts: int = Field(4, description="Maximum attempts for agent loops")
    
    # --- Neuro-Symbolic Flags ---
    verify_truth: bool = Field(True, description="Enable formal verification (Lean 4)")
    vectorize_diagrams: bool = Field(False, description="Enable VLM-to-TikZ vectorization")

    # --- Secrets (Read from Env Only) ---
    openai_api_key: Optional[str] = Field(None, description="OpenAI API Key")
    anthropic_api_key: Optional[str] = Field(None, description="Anthropic API Key")

    @field_validator("ocr_vram_headroom_gb", mode="before")
    @classmethod
    def parse_vram(cls, v):
        if v is None: return 2.0
        return float(v)

    def resolve_models_dir(self) -> Path:
        return self.models_dir.resolve()

# Global instance
settings = Settings()