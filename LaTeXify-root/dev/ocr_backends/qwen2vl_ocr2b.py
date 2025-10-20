# Qwen2-VL OCR backend (2B Instruct)
# Works with transformers >= 4.43 and qwen-vl-utils==0.0.8

from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
from PIL import Image
from transformers import (
    AutoProcessor,             # or Qwen2VLProcessor
    Qwen2VLForConditionalGeneration,
)

from dev.ocr_backends.base import OCRResult, OCRBackend  # your shared types


MODEL_ID = "prithivMLmods/Qwen2-VL-OCR-2B-Instruct"  # fine-tuned for OCR
# You can swap to the official base if you prefer:
# MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"


PROMPT = (
    "You are an OCR assistant specialized in math/science documents. "
    "Return clean Markdown with LaTeX for formulas ($ ... $ or $$ ... $$). "
    "Preserve headings, bullet lists, tables, and equation blocks. "
    "Do not add commentary beyond the text on the page."
)


@dataclass
class Backend(OCRBackend):
    name: str = "qwen2-vl-ocr-2b-instruct"

    def __post_init__(self):
        # Processor builds tokens + handles chat templates
        self.processor = AutoProcessor.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
        )
        # Correct model class for VL
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype="auto",
            device_map="auto",
            attn_implementation="sdpa",  # keeps us off Flash-Attn 2
        ).eval()

        # Small stability tweaks
        torch.set_grad_enabled(False)
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True

    def recognize_page(self, image_path: str, page: int = 1) -> OCRResult:
        # 1) Load page image
        img = Image.open(image_path).convert("RGB")

        # 2) Compose chat with explicit image item
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},    # IMPORTANT: real image object
                    {"type": "text",  "text": PROMPT},
                ],
            }
        ]

        # 3) Build inputs the Qwen2-VL way
        # NOTE: new-ish processors can inline vision, but qwen-vl-utils is robust
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)

        chat = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=[chat],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        # 4) Generate
        out = self.model.generate(
            **inputs,
            max_new_tokens=1200,
            do_sample=False,
        )

        # 5) Slice off the prompt tokens and decode
        prompt_len = inputs["input_ids"].shape[1]
        gen_tokens = out[:, prompt_len:]
        text = self.processor.batch_decode(
            gen_tokens, skip_special_tokens=True
        )[0].strip()

        return OCRResult(model=self.name, page=page, text_md=text, blocks=[])
