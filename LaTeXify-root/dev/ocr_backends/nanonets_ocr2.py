from __future__ import annotations
import os, json
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
# Let env override; default keeps prior naming convention.
MODEL_ID = os.getenv("NANONETS_OCR2_3B_ID", "nanonets/nanonets-ocr2-3b")


@dataclass
class OCRResult:
    model: str
    page: str
    text_md: str
    blocks: Optional[List[Dict[str, Any]]] = None


class Backend:
    """
    Nanonets OCR2 3B — text-only causal LM that outputs Markdown transcript
    given an OCR prompt instruction. We keep it minimal and GPU-only.
    """
    name = "nanonets-ocr2-3b"

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
            "You are an OCR-to-Markdown converter. The user provides an image path. "
            "Return clean, readable Markdown preserving headings, math, bullet lists, "
            "tables, and inline formatting. Do NOT add explanations.\n"
            f"IMAGE_PATH: {image_path}\n"
            "OUTPUT (Markdown only):"
        )

    @torch.inference_mode()
    def recognize_page(self, image_path: str, page: int = 1) -> OCRResult:
        # sanity read
        _ = Image.open(image_path).convert("RGB")

        prompt = self._prompt(image_path)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)
        out = self.model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        # Strip the prompt echo if present
        if "OUTPUT (Markdown only):" in text:
            text = text.split("OUTPUT (Markdown only):", 1)[-1].strip()
        return OCRResult(model=self.name, page=os.path.basename(image_path), text_md=text, blocks=None)
