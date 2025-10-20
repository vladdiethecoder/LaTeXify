from PIL import Image
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, GenerationConfig
from .base import OCRBackend, OCRResult

MODEL_ID = "prithivMLmods/Qwen2-VL-OCR-2B-Instruct"

# Raw string avoids invalid-escape warnings for \(...\), \[...\], etc.
PROMPT = r"""Extract the page as complete, readable markdown.
- Equations → LaTeX (inline \( ... \), display \[ ... \] or $$ ... $$).
- Tables → HTML <table>…</table> with rows/cols preserved.
- Preserve headings, lists, figure captions, footnotes, and page numbers.
- Transcribe verbatim; do not summarize or omit content."""

class Backend(OCRBackend):
    name = "qwen2-vl-ocr-2b-instruct"

    def __init__(self):
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            torch_dtype="auto",
            device_map="auto",
            attn_implementation="sdpa",   # fast, no FA2 required
        ).eval()
        # Defensive: explicitly load generation config if available.
        try:
            self.model.generation_config = GenerationConfig.from_pretrained(MODEL_ID)
        except Exception:
            pass

    def recognize_page(self, image_path: str, page: int = 1) -> OCRResult:
        # 1) load and compose messages
        from qwen_vl_utils import process_vision_info

        messages = [
            {"role": "system", "content": "You are a precise OCR engine that outputs Markdown/LaTeX/HTML tables exactly as seen."},
            {"role": "user", "content": [
                {"type": "image", "path": image_path},  # Qwen expects 'path'
                {"type": "text", "text": PROMPT}
            ]}
        ]
        chat = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        image_inputs, video_inputs = process_vision_info(messages)  # builds lists properly
        inputs = self.processor(
            text=[chat],
            images=image_inputs,
            videos=video_inputs,          # empty list is fine
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        # 3) generate with explicit eos/pad + sane cap

        with torch.no_grad():
            eos = getattr(self.model.generation_config, "eos_token_id", None)
            pad = getattr(self.model.generation_config, "pad_token_id", eos)
            out = self.model.generate(
                **inputs, max_new_tokens=3000, do_sample=False,
                eos_token_id=eos, pad_token_id=pad, return_dict_in_generate=True
            ).sequences

        in_len = inputs.input_ids.size(1)
        gen_only = out[:, in_len:] if out.size(1) > in_len else out[:, -1:]
        decoded = self.processor.batch_decode(gen_only, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        text = (decoded[0] if decoded else "").strip()
        return OCRResult(model=self.name, page=page, text_md=text, blocks=[])
