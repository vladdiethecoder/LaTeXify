from PIL import Image
import torch
from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText
from .base import OCRBackend, OCRPage, Block

MODEL_ID = "nanonets/Nanonets-OCR2-3B"  # image→markdown, LaTeX eqs, HTML tables
PROMPT = ("Extract faithful structured markdown. Tables→HTML, Equations→LaTeX. "
          "Add <img> alt if no caption. Wrap watermarks <watermark>..</watermark> "
          "and page numbers <page_number>..</page_number>. Use ☐/☑ for checkboxes.")

# load once (module import)
_tok = AutoTokenizer.from_pretrained(MODEL_ID)
_proc = AutoProcessor.from_pretrained(MODEL_ID)
try:
    _mdl = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, torch_dtype="auto", device_map="auto", attn_implementation="flash_attention_2"
    )
except Exception:
    _mdl = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, torch_dtype="auto", device_map="auto", attn_implementation="sdpa"
    )
_mdl.eval()

class Backend(OCRBackend):
    name = "nanonets-ocr2-3b"
    def recognize_page(self, image_path: str, page_num: int = 1) -> OCRPage:
        img = Image.open(image_path).convert("RGB")
        messages = [
            {"role":"system","content":"You are a careful OCR assistant."},
            {"role":"user","content":[{"type":"image","image":f"file://{image_path}"},
                                      {"type":"text","text":PROMPT}]},
        ]
        chat = _proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = _proc(text=[chat], images=[img], padding=True, return_tensors="pt").to(_mdl.device)
        with torch.no_grad():
            out = _mdl.generate(**inputs, max_new_tokens=3500, do_sample=False)
        gen_ids = [o[len(i):] for i,o in zip(inputs.input_ids, out)]
        text = _proc.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
        return OCRPage(model=self.name, page=page_num, text_md=text, blocks=[])
