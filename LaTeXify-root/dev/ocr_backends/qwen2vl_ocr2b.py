# dev/ocr_backends/qwen2vl_ocr2b.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os, torch
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
)

# Preferred → fallback repo IDs (override with QWEN_VL_MODEL_ID)
PREFERRED_MODELS = [
    "Qwen/Qwen2.5-VL-2B-Instruct",   # may be gated; requires login/accept
    "Qwen/Qwen2-VL-2B-Instruct",     # public fallback
]
MODEL_ID = os.getenv("QWEN_VL_MODEL_ID") or PREFERRED_MODELS[0]

# Tokens: transformers reads HUGGINGFACE_HUB_TOKEN; we also accept HF_TOKEN for convenience
HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")

def _best_dtype(user_dtype: str) -> torch.dtype:
    if user_dtype == "bf16":
        return torch.bfloat16
    if user_dtype == "fp16":
        return torch.float16
    # auto: Ampere loves fp16; allow override via env QWEN_DTYPE
    env = os.getenv("QWEN_DTYPE", "").lower()
    if env in ("fp16", "float16"): return torch.float16
    if env in ("bf16", "bfloat16"): return torch.bfloat16
    return torch.float16

@dataclass
class OCRResult:
    model: str
    page: str
    text_md: str
    blocks: list | None = None

@dataclass
class Backend:
    name: str = "qwen2-vl-ocr-2b-instruct"
    dtype: str = "auto"
    _ready: bool = False

    def __post_init__(self):
        self.torch_dtype = _best_dtype(self.dtype)
        self.device = "cuda"
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required: Qwen2-VL backend forbids CPU fallback.")

        self.model = None
        self.processor = None
        last_err = None

        # Try preferred → fallback
        for repo in [MODEL_ID] + [m for m in PREFERRED_MODELS if m != MODEL_ID]:
            try:
                self.processor = AutoProcessor.from_pretrained(
                    repo, trust_remote_code=True, token=HF_TOKEN
                )
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    repo,
                    trust_remote_code=True,
                    torch_dtype=self.torch_dtype,
                    device_map="auto",          # will bind within visible GPU
                    attn_implementation="sdpa", # avoid Flash-Attn dep
                    token=HF_TOKEN,
                ).eval()
                self.repo_id = repo
                self._ready = True
                break
            except Exception as e:
                last_err = e
                continue

        if not self._ready:
            raise RuntimeError(
                f"Failed to load any Qwen2 VL model. Last error: {last_err}\n"
                "If using Qwen2.5, ensure you are logged in and have access."
            )

        # Modest cap to prevent OOM while keeping quality
        self.max_new_tokens = int(os.getenv("QWEN_MAX_NEW_TOKENS", "1024"))

    def recognize_page(self, page_path: str, page: int = 1) -> OCRResult:
        # Qwen2-VL expects [(image, prompt)] within chat template
        from qwen_vl_utils import process_vision_info  # lightweight helper
        prompt = "Read this page as markdown. Preserve lists, headings, and math."
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": Path(page_path).as_posix()},
                {"type": "text", "text": prompt},
            ]}
        ]
        chat = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[chat],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt"
        ).to(self.model.device, dtype=self.torch_dtype)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=0.0,
            )

        text = self.processor.batch_decode(out, skip_special_tokens=True)[0]
        # Qwen chat formats often include the prompt again; strip prelude if present
        sep = "</s>"
        if sep in text:
            text = text.split(sep)[-1].strip()
        return OCRResult(model=self.name, page=Path(page_path).name, text_md=text, blocks=None)
