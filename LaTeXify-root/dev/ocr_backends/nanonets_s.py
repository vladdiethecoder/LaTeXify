from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM

# ======================= GPU / DTYPE ENFORCEMENT =======================
if not torch.cuda.is_available():
    raise RuntimeError("CUDA required: Nanonets backends forbid CPU fallback.")
DEVICE = torch.device("cuda:0")
_DTYPE_MAP = {"auto": None, "fp16": torch.float16, "bf16": torch.bfloat16}

# ============================ MODEL CHOICE =============================
MODEL_ID = os.getenv("NANONETS_OCR_S_ID", "nanonets/nanonets-ocr-small")


@dataclass
class OCRResult:
    model: str
    page: str
    text_md: str
    blocks: Optional[List[Dict[str, Any]]] = None


class Backend:
    """
    Nanonets OCR (small) — faster, lower-VRAM variant for cross-check vs 3B.
    GPU-only; identical calling surface.
    """
    name = "nanonets-ocr-s"

    def __init__(self, dtype: str = "auto") -> None:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
        dtype_torch = _DTYPE_MAP.get(dtype, None)

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map="auto",
            torch_dtype=dtype_torch,
            attn_implementation="sdpa",
            trust_remote_code=True,
        ).eval()

    def _prompt(self, image_path: str) -> str:
        return (
            "OCR the given page image into clean Markdown. "
            "Preserve structure (headings, lists, formulas) with minimal hallucinations.\n"
            f"IMAGE_PATH: {image_path}\n"
            "OUTPUT:"
        )

    @torch.inference_mode()
    def recognize_page(self, image_path: str, page: int = 1) -> OCRResult:
        _ = Image.open(image_path).convert("RGB")

        prompt = self._prompt(image_path)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)
        out = self.model.generate(
            **inputs,
            max_new_tokens=1536,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        if "OUTPUT:" in text:
            text = text.split("OUTPUT:", 1)[-1].strip()
        return OCRResult(model=self.name, page=os.path.basename(image_path), text_md=text, blocks=None)
