from pathlib import Path
from PIL import Image
import torch
from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText

MODEL_ID = "nanonets/Nanonets-OCR2-3B"
_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
_processor = AutoProcessor.from_pretrained(MODEL_ID)

# try flash-attn2; fall back to SDPA/eager if unavailable
try:
    _model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="flash_attention_2"
    )
except Exception:
    _model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="sdpa"  # widely supported fallback
    )
_model.eval()
