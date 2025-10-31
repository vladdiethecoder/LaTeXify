from __future__ import annotations
# Qwen2-VL Instruct OCR shim (v1.2.3)
# - dtype first, fallback to torch_dtype (Transformers compatibility)
# - explicit use_fast processor
# - PDF or image path; helpful path suggestions on failure
# - Greedy decoding, trims prompt, sanitizes output

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
from transformers import AutoProcessor
from transformers import Qwen2VLForConditionalGeneration as QwenVL

from dev.ocr_backends.prompts import GOLDEN_PROMPT
from dev.ocr_backends import sanitize
from dev.ocr_backends.io_utils import load_page_image


def _to_bool(v: Optional[str], default: bool = False) -> bool:
    if v is None:
        return default
    return v.lower() in {"1", "true", "yes", "y", "on"}


def _dtype_from_str(s: Optional[str]):
    s = (s or "fp16").lower()
    if s == "auto":
        return "auto"
    if s == "bf16":
        return torch.bfloat16
    return torch.float16


@dataclass
class Result:
    model: str
    page: int
    text_md: str


@dataclass
class Backend:
    name: str = "qwen2-vl-ocr-2b"
    model_id: Optional[str] = None
    device: str = "cuda:0"
    dtype: str = "fp16"  # "auto" | "fp16" | "bf16"

    max_new_tokens: int = int(os.getenv("QWEN_OCR_MAX_NEW_TOKENS", "2048"))
    temperature: float = float(os.getenv("QWEN_OCR_TEMPERATURE", "0.0"))
    top_p: float = float(os.getenv("QWEN_OCR_TOP_P", "1.0"))
    do_sample: bool = _to_bool(os.getenv("QWEN_OCR_DO_SAMPLE"), False)

    use_fast_processor: bool = _to_bool(os.getenv("QWEN_OCR_USE_FAST", "1"), True)
    pdf_dpi: int = int(os.getenv("QWEN_OCR_PDF_DPI", "200"))
    remove_tikz: bool = _to_bool(os.getenv("QWEN_OCR_REMOVE_TIKZ"), True)

    def __post_init__(self):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required: Qwen2-VL backend forbids CPU fallback.")

        self._torch_dtype = _dtype_from_str(self.dtype)
        attn_impl = "flash_attention_2" if _to_bool(os.getenv("USE_FA2"), False) else "sdpa"
        model_src = self.model_id or os.getenv("QWEN_VL_2B_DIR") or "Qwen/Qwen2-VL-2B-Instruct"

        # Prefer new `dtype=` kwarg, fall back to `torch_dtype=` for older transformers.
        common_kwargs: Dict[str, Any] = dict(
            device_map={"": self.device},
            attn_implementation=attn_impl,
        )
        try:
            self.model = QwenVL.from_pretrained(
                model_src,
                dtype=self._torch_dtype if self._torch_dtype != "auto" else "auto",
                **common_kwargs,  # type: ignore[arg-type]
            )
        except TypeError:
            self.model = QwenVL.from_pretrained(
                model_src,
                torch_dtype=self._torch_dtype if self._torch_dtype != "auto" else "auto",
                **common_kwargs,
            )
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(
            model_src,
            use_fast=self.use_fast_processor,  # quiets the warning, stable behavior
        )

    def _messages(self, image_path: str):
        return [
            {"role": "system", "content": [{"type": "text", "text": GOLDEN_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{image_path}"},
                    {"type": "text", "text": "Transcribe this page to Markdown per the rules."},
                ],
            },
        ]

    @torch.inference_mode()
    def recognize_page(self, image_or_pdf: str, page: int = 1) -> Result:
        # Load an RGB PIL.Image from an image path or from a PDF page.
        img = load_page_image(image_or_pdf, page=page, dpi=self.pdf_dpi)

        # We include the path in messages for the chat template, but pass the PIL image into the processor.
        messages = self._messages(image_or_pdf)
        chat_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = self.processor(text=[chat_text], images=[img], padding=True, return_tensors="pt")
        inputs = {k: (v.to(self.model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

        gen = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=self.do_sample,
        )

        # Only decode newly generated tokens (trim the prompt).
        prompt_len = inputs["input_ids"].shape[1]
        trimmed = gen[:, prompt_len:]

        out_text = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        out_text = sanitize.sanitize_output(out_text, remove_tikz=self.remove_tikz)
        return Result(model=self.name, page=page, text_md=out_text or "")
