#!/usr/bin/env python3
"""
Nanonets OCR2 (TrOCR-base proxy)
- Prompts do not apply to TrOCR; deterministic beam decoding.
- Accepts image path or PDF path + page (via io_utils).
- Sanitizes output for consistency.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from dev.ocr_backends import sanitize
from dev.ocr_backends.io_utils import load_page_image

MODEL_ID = "microsoft/trocr-base-printed"
MODEL_LABEL = "Nanonets OCR2 (TrOCR-base)"
torch.manual_seed(42)


@dataclass
class Result:
    model: str
    page: int
    text_md: str


class Backend:
    def __init__(self, device: Optional[str] = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = TrOCRProcessor.from_pretrained(MODEL_ID)
        self.model = VisionEncoderDecoderModel.from_pretrained(MODEL_ID).to(self.device).eval()

    @torch.inference_mode()
    def recognize_page(self, image_or_pdf: str, page: int = 1) -> Result:
        image = load_page_image(image_or_pdf, page=page, dpi=200)
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(
            **inputs,
            num_beams=4,
            length_penalty=0.0,
            max_length=384,
            early_stopping=True,
        )
        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        text = sanitize.sanitize_output(text, remove_tikz=False)
        return Result(model=MODEL_LABEL, page=page, text_md=text or "")
