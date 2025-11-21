"""
vLLM Client Adapter.
Provides high-performance inference for Qwen 2.5 models.
"""
import os
from typing import List, Optional, Dict, Any
import logging

try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None

LOGGER = logging.getLogger(__name__)

class VLLMClient:
    def __init__(self, model_name: str, tensor_parallel_size: int = 1):
        if LLM is None:
            raise ImportError("vLLM not installed.")
        
        self.model_name = model_name
        LOGGER.info(f"Initializing vLLM with model: {model_name}")
        self.llm = LLM(model=model_name, tensor_parallel_size=tensor_parallel_size, trust_remote_code=True)

    def generate(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.7) -> str:
        params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        outputs = self.llm.generate([prompt], params)
        return outputs[0].outputs[0].text

# Singleton management
_CLIENT_INSTANCE = None

def get_vllm_client(model_name: str = "Qwen/Qwen2.5-Math-7B-Instruct") -> VLLMClient:
    global _CLIENT_INSTANCE
    if _CLIENT_INSTANCE is None:
        _CLIENT_INSTANCE = VLLMClient(model_name)
    return _CLIENT_INSTANCE
