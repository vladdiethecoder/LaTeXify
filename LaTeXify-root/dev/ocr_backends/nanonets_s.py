from .base import OCRBackend, OCRResult
from PIL import Image
import torch
from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText

MODEL_ID = "nanonets/Nanonets-OCR-s"

class Backend(OCRBackend):
    name = "nanonets-ocr-s"
    def __init__(self):
        self.tok = AutoTokenizer.from_pretrained(MODEL_ID)
        self.proc = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID, torch_dtype="auto", device_map="auto",
            attn_implementation="sdpa"
        ).eval()
        self.prompt = r"""Extract the text from the page as **complete, readable markdown**.
                        - Equations → LaTeX (inline \( ... \), display \[ ... \] or $$ ... $$).
                        - Tables → HTML <table>...</table> with rows/cols preserved.
                        - Preserve headings, lists, figure captions, footnotes, page numbers.
                        - Transcribe verbatim; do not summarize or omit content."""

    def recognize_page(self, image_path: str, page:int=1) -> OCRResult:
        img = Image.open(image_path).convert("RGB")
        messages = [{"role":"user","content":[{"type":"image","image":f"file://{image_path}"},
                                             {"type":"text","text": self.prompt}]}]
        chat = self.proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.proc(text=[chat], images=[img], padding=True, return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=3000, do_sample=False)
        gen = self.proc.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        return OCRResult(model=self.name, page=page, text_md=gen, blocks=[])
