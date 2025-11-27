from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LOGGER = logging.getLogger(__name__)


@dataclass
class LocalLLMConfig:
    model_path: str
    device: str = "auto"
    max_new_tokens: int = 512
    temperature: float = 0.4
    top_p: float = 0.9
    load_in_4bit: bool = True
    fallback_model_path: Optional[str] = None  # try this if primary fails


class LocalTextGenerator:
    """
    Minimal local text generator using Hugging Face causal LM.
    Falls back to deterministic stub JSON if loading or generation fails.
    """

    def __init__(self, config: LocalLLMConfig):
        self.config = config
        self._tokenizer = None
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        # Avoid attempting to load very large models on CPU-only environments.
        if (not torch.cuda.is_available()) or self.config.device == "cpu":
            if any(name in self.config.model_path.lower() for name in ["mixtral", "deepseek", "qwen2.5-72b", "deepseek-v3"]):
                LOGGER.warning("No CUDA detected; skipping large model load for %s. Using stub.", self.config.model_path)
                self._model = None
                return
            # Disable 4bit on CPU
            self.config.load_in_4bit = False
            kwargs = {"device_map": "cpu", "trust_remote_code": True}
        else:
            kwargs = {"trust_remote_code": True}
            if self.config.load_in_4bit:
                kwargs["load_in_4bit"] = True
                kwargs["device_map"] = "auto"
            elif self.config.device != "auto":
                kwargs["device_map"] = self.config.device
        kwargs = {"trust_remote_code": True}
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_path, trust_remote_code=True)
            self._model = AutoModelForCausalLM.from_pretrained(self.config.model_path, **kwargs)
        except Exception as exc:
            LOGGER.warning("Failed to load local model (%s).", exc)
            if self.config.fallback_model_path:
                LOGGER.info("Attempting fallback model: %s", self.config.fallback_model_path)
                try:
                    self._tokenizer = AutoTokenizer.from_pretrained(self.config.fallback_model_path, trust_remote_code=True)
                    self._model = AutoModelForCausalLM.from_pretrained(self.config.fallback_model_path, **kwargs)
                    return
                except Exception as exc2:
                    LOGGER.warning("Failed to load fallback model (%s). Using stub.", exc2)
            self._model = None

    def __call__(self, prompt: str) -> str:
        self._ensure_loaded()
        if self._model is None or self._tokenizer is None:
            return self._stub_response()
        try:
            # Prefer chat template if available to reduce preamble.
            if hasattr(self._tokenizer, "apply_chat_template"):
                messages = [
                    {"role": "system", "content": "You are an assistant that ONLY returns JSON. No prose."},
                    {"role": "user", "content": prompt},
                ]
                prompt_text = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                prompt_text = "Return JSON only. No prose.\n" + prompt

            inputs = self._tokenizer(prompt_text, return_tensors="pt").to(self._model.device)
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                pad_token_id=self._tokenizer.eos_token_id,
            )
            text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            return text
        except Exception as exc:
            LOGGER.warning("Local generation failed (%s). Using stub.", exc)
            return self._stub_response()

    def _stub_response(self) -> str:
        return """{"proposals": [{"candidate_id": "stub-child", "strategy": "VALIDATION", "rationale": "stub fallback", "ops": [], "target_tests": []}]}"""


# GGUF support using llama.cpp
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False


@dataclass
class GGUFLLMConfig:
    model_path: str
    n_ctx: int = 4096
    n_threads: Optional[int] = None
    max_new_tokens: int = 256


class GGUFTextGenerator:
    """
    Minimal GGUF generator via llama-cpp-python.
    """

    def __init__(self, config: GGUFLLMConfig):
        self.config = config
        self.llm: Optional[Any] = None
        if not LLAMA_CPP_AVAILABLE:
            LOGGER.warning("llama_cpp not available; GGUF generator will stub.")

    def _ensure_loaded(self) -> None:
        if self.llm or not LLAMA_CPP_AVAILABLE:
            return
        try:
            self.llm = Llama(
                model_path=self.config.model_path,
                n_ctx=self.config.n_ctx,
                n_threads=self.config.n_threads or None,
            )
        except Exception as exc:
            LOGGER.warning("Failed to load GGUF model (%s). Using stub.", exc)
            self.llm = None

    def __call__(self, prompt: str) -> str:
        self._ensure_loaded()
        if not self.llm:
            return """{"proposals": [{"candidate_id": "stub-child", "strategy": "VALIDATION", "rationale": "stub fallback", "ops": [], "target_tests": []}]}"""
        try:
            res = self.llm(
                prompt,
                max_tokens=self.config.max_new_tokens,
                temperature=0.0,
                top_p=1.0,
                echo=False,
            )
            return res["choices"][0]["text"]
        except Exception as exc:
            LOGGER.warning("GGUF generation failed (%s). Using stub.", exc)
            return """{"proposals": [{"candidate_id": "stub-child", "strategy": "VALIDATION", "rationale": "stub fallback", "ops": [], "target_tests": []}]}"""
