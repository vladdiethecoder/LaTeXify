# Smaller sibling, same API
from .nanonets_ocr2_3b import OCRBackend, OCRPage, Block, Image, torch, AutoTokenizer, AutoProcessor, AutoModelForImageTextToText
MODEL_ID = "nanonets/Nanonets-OCR-s"
_tok = AutoTokenizer.from_pretrained(MODEL_ID)
_proc = AutoProcessor.from_pretrained(MODEL_ID)
try:
    _mdl = AutoModelForImageTextToText.from_pretrained(MODEL_ID, torch_dtype="auto", device_map="auto", attn_implementation="flash_attention_2")
except Exception:
    _mdl = AutoModelForImageTextToText.from_pretrained(MODEL_ID, torch_dtype="auto", device_map="auto", attn_implementation="sdpa")
_mdl.eval()

PROMPT = ("Extract structured markdown; prefer HTML tables and LaTeX for equations.")

class Backend(OCRBackend):
    name = "nanonets-ocr-s"
    def recognize_page(self, image_path: str, page_num: int = 1) -> OCRPage:
        img = Image.open(image_path).convert("RGB")
        messages = [{"role":"user","content":[{"type":"image","image":f"file://{image_path}"},
                                              {"type":"text","text":PROMPT}]}]
        chat = _proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = _proc(text=[chat], images=[img], padding=True, return_tensors="pt").to(_mdl.device)
        with torch.no_grad():
            out = _mdl.generate(**inputs, max_new_tokens=3000, do_sample=False)
        text = _proc.batch_decode([out[0][len(inputs.input_ids[0]):]], skip_special_tokens=True)[0].strip()
        return OCRPage(model=self.name, page=page_num, text_md=text, blocks=[])
