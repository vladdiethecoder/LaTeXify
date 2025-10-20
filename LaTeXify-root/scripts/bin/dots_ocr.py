import sys, json
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from qwen_vl_utils import process_vision_info

# Path to the local weights folder you already have
MODEL_PATH = "third_party/dots.ocr/weights/DotsOCR"  # adjust if different

img_path = sys.argv[1]
image = Image.open(img_path).convert("RGB")

# Use the same TF version DoTS expects
proc = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, trust_remote_code=True, torch_dtype="auto", device_map="auto"
)

# Build chat-style messages per DoTS/Qwen2 examples
messages = [
    {"role": "user", "content": [{"type": "image"} , {"type": "text", "text": "Transcribe the text in this image."}]}
]
text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# Convert image to vision inputs
image_inputs, _ = process_vision_info([image])

# Tokenize
inputs = proc(text=[text], videos=image_inputs, return_tensors="pt").to(model.device)

with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=2048)
    resp = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]

print(json.dumps({"raw": resp}))
