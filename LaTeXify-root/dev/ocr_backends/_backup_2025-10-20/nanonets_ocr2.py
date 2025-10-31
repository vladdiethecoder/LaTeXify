from pathlib import Path
from PIL import Image
import torch
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    AutoModelForImageTextToText,
    GenerationConfig,
)
from .base import OCRBackend, OCRResult

# LaTeX-aware page OCR prompt (raw string preserves backslashes)
PROMPT = r"""Extract the page into clean Markdown with correct reading order.
- Equations: output LaTeX (inline \( ... \), display \[ ... \] or $$ ... $$).
- Tables: output as HTML <table> with <thead>/<tbody> where applicable.
- Keep figure captions with figures; keep section headers and lists.
- Preserve symbols, superscripts/subscripts, and math spacing."""

MODEL_ID = "nanonets/Nanonets-OCR2-3B"

class Backend(OCRBackend):
    name = "nanonets-ocr2-3b"

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)

        # If flash-attn isn't installed, you can remove attn_implementation arg.
        self.model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            torch_dtype="auto",
            device_map="auto",
            attn_implementation="sdpa",
        ).eval()

        # Explicitly load generation config to avoid rare dict->to_dict issues
        try:
            self.model.generation_config = GenerationConfig.from_pretrained(MODEL_ID)
        except Exception:
            pass  # safe fallback

        # Prompt recommended by the model card: structured markdown w/ math & tables
        self.prompt = (
            r"""Extract the text from the page as **complete, readable markdown**.
                - Equations → LaTeX (inline \( ... \), display \[ ... \] or $$ ... $$).
                - Tables → HTML <table>...</table> with rows/cols preserved.
                - Preserve headings, lists, figure captions, footnotes, page numbers.
                - Transcribe verbatim; do not summarize or omit content."""
        )

    def recognize_page(self, image_path: str, page: int = 1) -> OCRResult:
        # 1) load image and build messages
        img = Image.open(image_path).convert("RGB")
        messages = [
            {"role": "system", "content": "You are a precise OCR engine that outputs Markdown/LaTeX/HTML tables exactly as seen."},
            {"role": "user", "content": [
                {"type": "image", "path": image_path},
                {"type": "text", "text": PROMPT}  # make PROMPT a raw string r"""...""" to avoid \(-escape warnings
            ]}
        ]
        chat = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # 2) preprocess
        inputs = self.processor(
            text=[chat],
            images=[img],
            videos=[],              # explicit empty list for VL processors
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        # 3) generate (robust settings per HF docs)
        with torch.no_grad():
            eos = getattr(self.model.generation_config, "eos_token_id", None)
            pad = getattr(self.model.generation_config, "pad_token_id", eos)
            out = self.model.generate(
                **inputs,
                max_new_tokens=3000,
                do_sample=False,
                eos_token_id=eos,
                pad_token_id=pad,
                return_dict_in_generate=True,
            ).sequences

        # Robust slice: handle 0 new tokens
        in_len = inputs.input_ids.size(1)
        gen_only = out[:, in_len:] if out.size(1) > in_len else out[:, -1:]
        decoded = self.processor.batch_decode(gen_only, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        text = (decoded[0] if decoded else "").strip()
        return OCRResult(model=self.name, page=page, text_md=text, blocks=[])

