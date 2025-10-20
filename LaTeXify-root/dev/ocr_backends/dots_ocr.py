from __future__ import annotations

from PIL import Image
import torch

# --- Keep these imports at module scope; do not rebind these names later ---
from transformers import (
    AutoModelForCausalLM,
    AutoImageProcessor,
    AutoTokenizer,
    Qwen2_5_VLProcessor,
)
from transformers.video_processing_utils import BaseVideoProcessor

from .base import OCRBackend, OCRResult

MODEL_ID = "rednote-hilab/dots.ocr"

PROMPT = r"""Please output the layout information from the page image, including each layout element's bbox,
its category, and the corresponding text content within the bbox.
1) Bbox format: [x1, y1, x2, y2] (integers).
2) Categories: ['Caption','Footnote','Formula','List-item','Page-footer','Page-header','Picture','Section-header','Table','Text','Title'].
3) Rules:
   - Picture → do NOT OCR text inside the image region.
   - Formula → output LaTeX (inline \( ... \) or display \[ ... \] / $$ ... $$).
   - Table → output HTML <table>...</table>.
   - Others → output Markdown in reading order.
Return a single JSON object covering the whole page."""

class Backend(OCRBackend):
    name = "dots-ocr"

    def __init__(self):
        # Model
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
            trust_remote_code=True,
        ).eval()

        # Processor (Qwen2.5-VL style: requires image/tokenizer/video processors)
        image_proc = AutoImageProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        tokenizer  = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        video_proc = BaseVideoProcessor()  # satisfy required video processor

        self.processor = Qwen2_5_VLProcessor(
            image_processor=image_proc,
            tokenizer=tokenizer,
            video_processor=video_proc,
            chat_template=getattr(tokenizer, "chat_template", None),
        )


    def recognize_page(self, image_path: str, page: int = 1) -> OCRResult:
        # --- load image & compose messages (Qwen chat format) ---
        messages = [
            {"role": "system", "content": "You are a precise document OCR+layout engine."},
            {"role": "user", "content": [
                {"type": "image", "path": image_path},   # Qwen expects 'path'
                {"type": "text", "text": PROMPT},
            ]},
        ]
        chat = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = self.processor(
            text=[chat],
            images=[img],
            videos=[],
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            eos = getattr(self.model.generation_config, "eos_token_id", None)
            pad = getattr(self.model.generation_config, "pad_token_id", eos)
            out = self.model.generate(
                **inputs, max_new_tokens=3000, do_sample=False,
                eos_token_id=eos, pad_token_id=pad, return_dict_in_generate=True
            ).sequences

        in_len = inputs.input_ids.size(1)
        gen_only = out[:, in_len:] if out.size(1) > in_len else out[:, -1:]
        md = self.processor.batch_decode(gen_only, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0].strip()
        return OCRResult(model=self.name, page=page, text_md=md, blocks=[])
