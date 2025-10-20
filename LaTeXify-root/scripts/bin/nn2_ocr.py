import sys
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_ID = "nanonets/Nanonets-OCR2-3B"

img_path = sys.argv[1]
image = Image.open(img_path).convert("RGB")

proc = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
    device_map="auto",
    attn_implementation="sdpa",  # (works without flash-attn)
    trust_remote_code=True,
)

prompt = "Transcribe the text in this image."
messages = [{
    "role": "user",
    "content": [
        {"type": "image"},
        {"type": "text", "text": prompt},
    ],
}]

# 1) turn messages → chat templated string
text = proc.apply_chat_template(messages, add_generation_prompt=True)

# 2) feed (text, image) to the processor
inputs = proc(text=[text], images=[image], return_tensors="pt").to(model.device)

with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=2048)
print(proc.batch_decode(out, skip_special_tokens=True)[0].strip())
