from __future__ import annotations
from dataclasses import dataclass
from typing import List
from PIL import Image
import torch

from transformers import AutoProcessor, AutoModelForCausalLM
from .base import OCRBackend, OCRResult

MODEL_ID = "rednote-hilab/dots.ocr"

PROMPT = r"""Extract this page into clean Markdown with correct reading order.
- Equations: LaTeX (inline \( ... \), display \[ ... \] or $$ ... $$).
- Tables: HTML <table> with <thead>/<tbody>.
- Preserve headings, lists, captions, superscripts and subscripts."""

class Backend(OCRBackend):
    name = "dots-ocr"

    def __init__(self):
        self.processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype="auto", device_map="auto",
            attn_implementation="sdpa", trust_remote_code=True,
        ).eval()

    def recognize_page(self, image_path: str, page:int=1) -> OCRResult:
        img = Image.open(image_path).convert("RGB")
        # Keep chat content as a STRING to avoid chat template list-concat issues.
        messages = [
            {"role": "system", "content": "You are a precise document OCR+layout engine."},
            {"role": "user",   "content": PROMPT},
        ]
        chat = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = self.processor(
            text=[chat],
            images=[img],    # pass image via processor, not inside list content
            videos=[],
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            eos = getattr(self.model.generation_config, "eos_token_id", None)
            pad = getattr(self.model.generation_config, "pad_token_id", eos)
            out = self.model.generate(
                **inputs, max_new_tokens=3000, do_sample=False,
                eos_token_id=eos, pad_token_id=pad,
                return_dict_in_generate=True
            ).sequences

        in_len  = inputs.input_ids.size(1)
        gen_only = out[:, in_len:] if out.size(1) > in_len else out[:, -1:]
        decoded = self.processor.batch_decode(gen_only, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        md = (decoded[0] if decoded else "").strip()
        return OCRResult(model=self.name, page=page, text_md=md, blocks=[])
