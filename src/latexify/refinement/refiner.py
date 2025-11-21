import torch
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM

class LLMRefiner:
    def __init__(self, model_path: str = "Qwen/Qwen2.5-Coder-14B-Instruct", device: str = "cuda"):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """
        Load the model. Delayed loading to save VRAM if not used immediately.
        """
        print(f"Loading Refiner LLM: {self.model_path}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, 
                device_map=self.device, 
                torch_dtype=torch.float16
            )
            # Optimization: Compile model
            if hasattr(torch, "compile"):
                try:
                    print("Compiling Refiner LLM with torch.compile...")
                    self.model = torch.compile(self.model, mode="max-autotune")
                except Exception as e:
                    print(f"torch.compile failed (continuing without it): {e}")

        except Exception as e:
            print(f"Failed to load LLM: {e}")
            raise

    def refine(self, raw_latex: str) -> str:
        """
        Refine the raw LaTeX string using the LLM.
        """
        if self.model is None:
            self.load_model()
            
        prompt = f"""
        You are a LaTeX expert. Fix any syntax errors, close unclosed environments, 
        and correct OCR typos in the following LaTeX code. 
        Do not change the content/meaning.
        
        RAW LATEX:
        {raw_latex}
        
        FIXED LATEX:
        """
        return self._generate(prompt)

    def fix_error(self, latex_code: str, error_log: str) -> str:
        """
        Fix LaTeX based on compiler error log.
        """
        if self.model is None:
            self.load_model()
            
        prompt = f"""
        The following LaTeX code failed to compile. 
        
        ERROR LOG:
        {error_log}
        
        LATEX CODE:
        {latex_code}
        
        Fix the errors reported in the log. Return only the fixed LaTeX.
        
        FIXED LATEX:
        """
        return self._generate(prompt)

    def _generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, max_new_tokens=4096)
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "FIXED LATEX:" in result:
            return result.split("FIXED LATEX:")[-1].strip()
        return result
