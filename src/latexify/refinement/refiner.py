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
                model_kwargs["device_map"] = "auto"
            elif self.load_in_4bit:
                model_kwargs["load_in_4bit"] = True
                model_kwargs["device_map"] = "auto"
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

    def _generate(self, prompt: str, step_name: str = "generate") -> str:
        if self.mock_mode:
            return "Mock Generation"

        # Construct messages for Chat Model
        messages = [
            {"role": "system", "content": "You are a helpful assistant and a LaTeX expert."},
            {"role": "user", "content": prompt}
        ]

        if self.use_vllm and self.vllm_model:
            # vLLM usually handles chat templates if we pass prompt as string with special tokens, 
            # OR we use the chat entrypoint. But here we are using LLM class which is for completion.
            # We need to format it manually using tokenizer (if available) or just hope the model works?
            # Actually LLM class in vLLM 0.4 supports 'chat' via `chat` method in newer versions, 
            # or we format it. Let's try formatting if tokenizer is available, else raw prompt might fail.
            # The current `LLM` class usage suggests text completion.
            # Let's try to use the tokenizer to format it if we have one, or just use raw prompt if we can't.
            # Ideally we should use `tokenizer.apply_chat_template`.
            # Since we don't have the tokenizer loaded in vLLM mode easily (it's inside the engine),
            # let's assume we can load it or it's Qwen which needs <|im_start|>.
            
            # For safety, let's fallback to transformers logic for formatting if possible, 
            # or just disable vLLM if we can't format.
            # But wait, we didn't load tokenizer in vLLM mode.
            # Let's load tokenizer always.
            pass

        # Ensure tokenizer is loaded for template application
        if self.tokenizer is None:
             self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

        text_prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        if self.use_vllm and self.vllm_model:
            sampling_params = SamplingParams(temperature=0.4, max_tokens=8192, repetition_penalty=1.1)
            outputs = self.vllm_model.generate([text_prompt], sampling_params)
            result = outputs[0].outputs[0].text
        else:
            inputs = self.tokenizer(text_prompt, return_tensors="pt").to(self.device)
            input_len = inputs.input_ids.shape[1]
            with torch.inference_mode():
                outputs = self.model.generate(**inputs, max_new_tokens=8192, temperature=0.4, do_sample=True, repetition_penalty=1.1)
            
            # Decode only the new tokens
            generated_tokens = outputs[0][input_len:]
            result = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Debug: Log raw LLM output
        with open(f"debug_llm_{step_name}.txt", "w", encoding="utf-8") as f:
            f.write(result)

        # Clean up output
        if "FIXED LATEX:" in result:
            result = result.split("FIXED LATEX:")[-1].strip()
            
        # Strip markdown code blocks
        code_block_pattern = re.compile(r"```(?:latex)?\s*(.*?)\s*```", re.DOTALL)
        match = code_block_pattern.search(result)
        if match:
            result = match.group(1).strip()
        else:
            # Fallback: heuristic cleanup if no code blocks found
            # Look for \documentclass
            if "\\documentclass" in result:
                start = result.find("\\documentclass")
                result = result[start:]
            # Look for end document
            if "\\end{document}" in result:
                end = result.find("\\end{document}") + len("\\end{document}")
                result = result[:end]
            
        return result

    def refine(self, raw_latex: str) -> str:
        # ... (omitted init check) ...
        if self.model is None and self.vllm_model is None and not self.mock_mode:
            self.load_model()
            
        if self.mock_mode:
            return raw_latex + "\n% Refined by Mock LLM"

        prompt = f"""
        You are a professional LaTeX typesetter.
        
        **TASK:**
        Convert the following Raw OCR content into a complete, compilable LaTeX document.
        
        **RULES:**
        1. Output **ONLY** the LaTeX code inside a Markdown code block (```latex ... ```).
        2. Do NOT include conversational text, explanations, or preambles outside the code block.
        3. Use a standard `article` class.
        4. Fix all syntax errors and OCR typos.
        5. Infer sections and structure from the text.
        6. Ensure all math is correctly formatted.
        7. **STRICTLY FORBIDDEN:** Do NOT use Unicode mathematical symbols (e.g., −, ×, ≤, ≥, ’). You MUST use their LaTeX equivalents (e.g., -, \\times, \\leq, \\geq, ').
        8. Wrap all single-letter mathematical variables in math mode (e.g., $x$, $y$).
        
        **RAW INPUT:**
        {raw_latex}
        """
        return self._generate(prompt, step_name="refine_initial")

    def fix_error(self, latex_code: str, error_log: str) -> str:
        # ... (omitted init check) ...
        if self.model is None and self.vllm_model is None and not self.mock_mode:
            self.load_model()
            
        if self.mock_mode:
            return latex_code + "\n% Fixed by Mock LLM"

        prompt = f"""
        The following LaTeX code failed to compile.
        
        **ERROR LOG:**
        {error_log}
        
        **LATEX CODE:**
        {latex_code}
        
        **INSTRUCTIONS:**
        1. Fix the specific error reported in the log.
        2. Return **ONLY** the corrected LaTeX code inside a Markdown code block (```latex ... ```).
        3. Do NOT add explanations.
        4. Ensure no Unicode characters remain in the fixed code. Use LaTeX macros instead.
        """
        return self._generate(prompt, step_name="refine_fix")
