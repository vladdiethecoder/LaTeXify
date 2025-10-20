# dev/ocr_backends/nanonets_ocr2.py
from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import Any, Dict, List
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_ID = "nanonets/Nanonets-OCR2-3B"


@dataclass
class OCRResult:
    model: str
    page: int
    text_md: str
    blocks: List[Dict[str, Any]]


class Backend:
    """
    Nanonets OCR2: Image-Text-to-Text pipeline that emits Markdown + LaTeX.
    Matches the model card's recommended usage: AutoProcessor + apply_chat_template,
    pass images=[PIL.Image], generate, then trim prompt tokens before decode.
    """

    def __init__(self) -> None:
        self.name = "nanonets-ocr2-3b"
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            torch_dtype="auto",
            device_map="auto",
        ).eval()

    def recognize_page(self, image_path: str, page: int = 1) -> OCRResult:
        img = Image.open(image_path).convert("RGB")

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{image_path}"},
                {"type": "text", "text": (
                    "Transcribe this page faithfully. "
                    "Use Markdown for structure (headings, lists, tables). "
                    "Keep every formula in LaTeX (inline \\( ... \\) or display $$ ... $$). "
                    "Do not translate. Preserve reading order."
                )},
            ],
        }]

        # Build chat prompt, then encode with image
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            images=[img],
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=False,
            )

        # Trim the input/prompt tokens before decoding generated tokens
        trimmed = [gen[len(inp):] for inp, gen in zip(inputs.input_ids, output_ids)]
        md = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return OCRResult(model=self.name, page=page, text_md=md.strip(), blocks=[])
