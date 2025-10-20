from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from .base import OCRBackend, OCRPage

from qwen_vl_utils import process_vision_info  # provided by dots.ocr / Qwen utils

MODEL_PATH = "third_party/dots.ocr/weights/DotsOCR"
proc = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH, trust_remote_code=True, torch_dtype="auto",
    attn_implementation="sdpa", device_map="auto"
)

_mdl.eval()


PROMPT = (
    "Read the document in natural order and return structured markdown. "
    "Use HTML tables, LaTeX equations, and include figure captions if present."
)

class Backend(OCRBackend):
    name = "dots-ocr"
    def recognize_page(self, image_path: str, page_num: int = 1) -> OCRPage:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text",  "text": PROMPT},
            ],
        }]
        text = _proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)  # ensures video fields are present
        inputs = _proc(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(_mdl.device)
        with torch.no_grad():
            out = _mdl.generate(**inputs, max_new_tokens=2500, do_sample=False)
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
        resp = _proc.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        return OCRPage(model=self.name, page=page_num, text_md=resp, blocks=[])
