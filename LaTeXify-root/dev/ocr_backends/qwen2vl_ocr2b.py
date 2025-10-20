from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from .base import OCRBackend, OCRPage

MODEL_ID = "prithivMLmods/Qwen2-VL-OCR-2B-Instruct"
_proc = AutoProcessor.from_pretrained(MODEL_ID)
_mdl  = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto", device_map="auto")
_mdl.eval()

PROMPT = ("OCR the page into structured markdown. Use LaTeX for math when appropriate; "
          "keep reading order; include table structure.")

class Backend(OCRBackend):
    name = "qwen2vl-ocr-2b"
    def recognize_page(self, image_path: str, page_num: int = 1) -> OCRPage:
        img = Image.open(image_path).convert("RGB")
        messages = [{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":PROMPT}]}]
        chat = _proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = _proc(text=[chat], images=[img], return_tensors="pt").to(_mdl.device)
        with torch.no_grad():
            out = _mdl.generate(**inputs, max_new_tokens=2200, do_sample=False)
        resp = _proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
        return OCRPage(model=self.name, page=page_num, text_md=resp, blocks=[])
