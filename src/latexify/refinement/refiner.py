import torch
import re
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

class LLMRefiner:
    def __init__(self, model_path: str = "Qwen/Qwen2.5-Coder-14B-Instruct", device: str = "cuda", use_vllm: bool = True, load_in_8bit: bool = False, load_in_4bit: bool = False):
        self.model_path = model_path
        self.device = device
        self.use_vllm = use_vllm and VLLM_AVAILABLE
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        
        self.model = None
        self.tokenizer = None
        self.vllm_model = None
        self.mock_mode = False

    def load_model(self):
        """
        Load the model. Delayed loading to save VRAM if not used immediately.
        """
        print(f"Loading Refiner LLM: {self.model_path} (vLLM={self.use_vllm}, 8bit={self.load_in_8bit}, 4bit={self.load_in_4bit})...")
        
        if self.use_vllm:
            try:
                self.vllm_model = LLM(model=self.model_path, trust_remote_code=True, dtype="float16") 
                return
            except Exception as e:
                print(f"Failed to initialize vLLM: {e}. Falling back to Transformers.")
                self.use_vllm = False

        # Fallback or Standard Transformers
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            # For 8bit, avoid passing device_map="auto" if it causes dispatch errors.
            # Usually load_in_8bit handles device placement on GPU 0 by default.
            model_kwargs = {
                "torch_dtype": torch.float16,
                "trust_remote_code": True
            }
            
            if self.load_in_8bit:
                model_kwargs["load_in_8bit"] = True
                # Do not set device_map here to avoid .to() calls during dispatch
            elif self.load_in_4bit:
                model_kwargs["load_in_4bit"] = True
            else:
                model_kwargs["device_map"] = self.device
            
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **model_kwargs)
            
            if not self.load_in_8bit and not self.load_in_4bit and hasattr(torch, "compile"):
                try:
                    print("Compiling Refiner LLM with torch.compile...")
                    self.model = torch.compile(self.model, mode="max-autotune")
                except Exception as e:
                    print(f"torch.compile failed (continuing without it): {e}")

        except Exception as e:
            print(f"Failed to load LLM ({e}). Switching to MOCK mode.")
            import traceback
            traceback.print_exc()
            self.mock_mode = True

    def refine(self, raw_latex: str) -> str:
        """
        Refine the raw LaTeX string using the LLM.
        """
        if self.model is None and self.vllm_model is None and not self.mock_mode:
            self.load_model()
            
        if self.mock_mode:
            return raw_latex + "\n% Refined by Mock LLM"

        prompt = f"""
        You are a LaTeX expert. Your task is to fix syntax errors, close unclosed environments, 
        and correct OCR typos in the provided LaTeX code.
        
        Think step by step:
        1. Identify broken environments (e.g. \begin{{equation}} without \end{{equation}}).
        2. Identify OCR glitches (e.g. '1nt' instead of 'int').
        3. Fix them while preserving the original content and structure.
        
        RAW LATEX:
        {raw_latex}
        
        FIXED LATEX:
        """
        return self._generate(prompt)

    def fix_error(self, latex_code: str, error_log: str) -> str:
        """
        Fix LaTeX based on compiler error log.
        """
        if self.model is None and self.vllm_model is None and not self.mock_mode:
            self.load_model()
            
        if self.mock_mode:
            return latex_code + "\n% Fixed by Mock LLM"

        prompt = f"""
        The following LaTeX code failed to compile. 
        
        ERROR LOG:
        {error_log}
        
        LATEX CODE:
        {latex_code}
        
        Think step by step to resolve the specific error reported in the log.
        Return only the fixed LaTeX code.
        
        FIXED LATEX:
        """
        return self._generate(prompt)

    def _generate(self, prompt: str) -> str:
        if self.mock_mode:
            return "Mock Generation"

        if self.use_vllm and self.vllm_model:
            sampling_params = SamplingParams(temperature=0.7, max_tokens=4096)
            outputs = self.vllm_model.generate([prompt], sampling_params)
            result = outputs[0].outputs[0].text
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                outputs = self.model.generate(**inputs, max_new_tokens=4096)
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up output
        # Clean up output
        if "FIXED LATEX:" in result:
            result = result.split("FIXED LATEX:")[-1].strip()
            
        # Strip markdown code blocks
        code_block_pattern = re.compile(r"```(?:latex)?\s*(.*?)\s*```", re.DOTALL)
        match = code_block_pattern.search(result)
        if match:
            result = match.group(1).strip()
            
        return result
