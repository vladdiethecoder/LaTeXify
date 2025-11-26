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
        if not self.model_path or str(self.model_path).lower() == "none":
            print("Loading Refiner LLM: <disabled> -> MOCK mode")
            self.mock_mode = True
            return

        print(f"Loading Refiner LLM: {self.model_path} (vLLM={self.use_vllm}, 8bit={self.load_in_8bit}, 4bit={self.load_in_4bit})...")
        
        if self.use_vllm:
            try:
                self.vllm_model = LLM(model=self.model_path, trust_remote_code=True, dtype="float16") 
                return
            except Exception as e:
                print(f"Failed to initialize vLLM: {e}. Falling back to Transformers.")
                self.use_vllm = False

        if self.device == "cpu":
            if self.load_in_8bit or self.load_in_4bit:
                print("Warning: Disabling 4-bit/8-bit quantization on CPU (not supported).")
                self.load_in_8bit = False
                self.load_in_4bit = False

        # Fallback or Standard Transformers
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            # For 8bit, avoid passing device_map="auto" if it causes dispatch errors.
            # Usually load_in_8bit handles device placement on GPU 0 by default.
            dtype = torch.float32 if self.device == "cpu" else torch.float16
            model_kwargs = {
                "torch_dtype": dtype,
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
            
            # if not self.load_in_8bit and not self.load_in_4bit and hasattr(torch, "compile"):
            #     try:
            #         print("Compiling Refiner LLM with torch.compile...")
            #         self.model = torch.compile(self.model, mode="max-autotune")
            #     except Exception as e:
            #         print(f"torch.compile failed (continuing without it): {e}")

        except Exception as e:
            print(f"Failed to load LLM ({e}). Switching to MOCK mode.")
            import traceback
            traceback.print_exc()
            self.mock_mode = True

    def _post_process_latex(self, text: str) -> str:
        """
        Deterministic cleanup of common LLM LaTeX mistakes.
        """
        # 1. Unicode to LaTeX Map (Wrapped in ensuremath for safety)
        replacements = {
            "≠": "\\ensuremath{\\neq}",
            "≤": "\\ensuremath{\\leq}",
            "≥": "\\ensuremath{\\geq}",
            "×": "\\ensuremath{\\times}",
            "−": "-", 
            "’": "'",
            "“": "``",
            "”": "''",
            "…": "\\dots",
            "̸=": "\\ensuremath{\\neq}",
            "≈": "\\ensuremath{\\approx}",
            "←": "\\ensuremath{\\leftarrow}",
            "→": "\\ensuremath{\\rightarrow}",
            "⇒": "\\ensuremath{\\Rightarrow}",
            "⇐": "\\ensuremath{\\Leftarrow}",
            "⇔": "\\ensuremath{\\Leftrightarrow}",
            "±": "\\ensuremath{\\pm}",
            "∞": "\\ensuremath{\\infty}",
            "°": "\\ensuremath{^{\\circ}}",
            "µ": "\\ensuremath{\\mu}"
        }
        for char, repl in replacements.items():
            text = text.replace(char, repl)

        # 1.5 Fix missing space after specific macros (e.g., \RightarrowN -> \Rightarrow N)
        # This ensures they are parsed as command + text, not a new unknown command.
        # We only target \Rightarrow for now as it's a common hallucination artifact.
        text = re.sub(r"\\Rightarrow(?=[a-zA-Z0-9])", r"\\Rightarrow ", text)

        # 2. Safety wrapper for common commands often used in text mode by mistake
        # We use simple string replacement because \ensuremath works in math mode too.
        # We avoid replacing if it's already wrapped (simple check) to keep source clean, 
        # but strictly speaking \ensuremath{\ensuremath{...}} is valid/safe.
        # List of commands that MUST be in math mode:
        math_commands = [
            "\\Rightarrow", "\\rightarrow", "\\leftarrow", "\\leftrightarrow",
            "\\Leftarrow", "\\Leftrightarrow", "\\neq", "\\leq", "\\geq",
            "\\approx", "\\pm", "\\infty", "\\times"
        ]
        
        for cmd in math_commands:
            # Negative lookbehind to avoid double wrapping if we run this multiple times 
            # or if it was replaced above. 
            # Actually, the replacements above already produce \ensuremath{...}.
            # We need to catch EXPLICIT \Rightarrow usage by the LLM.
            
            # We perform a replacement of "\cmd" with "\ensuremath{\cmd}"
            # BUT we must ensure we don't match "\ensuremath{\cmd}" itself.
            # Regex: (?<!\\ensuremath\{)\\\bcmd\b
            # Note: standard macros usually don't have \b at start (it's \)
            
            escaped_cmd = re.escape(cmd)
            # Pattern: Not preceded by ensuremath{, match cmd, word boundary or non-letter
            # Using a simple string replace might loop if we are not careful.
            # Let's use a unique marker approach or just precise regex.
            
            pattern = r"(?<!\\ensuremath\{)" + escaped_cmd + r"(?![a-zA-Z])"
            # Use lambda to avoid re.sub interpreting backslashes in the replacement string
            text = re.sub(pattern, lambda m: f"\\ensuremath{{{cmd}}}", text)

        # 3. Fix common math environment issues
        text = re.sub(r"\\begin\{align\*?\}\s*\\\[", r"\\begin{align}\n", text)
        text = re.sub(r"\\\]\s*\\end\{align\*?\}", r"\n\\end{align}", text)
        
        # 4. Auto-wrap naked math lines
        lines = text.splitlines()
        processed_lines = []
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("%") or stripped.startswith("\\"):
                processed_lines.append(line)
                continue
            
            # Check for math indicators
            has_math_chars = any(c in stripped for c in ["=", "^", "_"])
            # Don't count ensuremath as a full-line delimiter. 
            # If a line has ^ or _ but only ensuremath, it likely needs full wrapping.
            has_delimiters = "$" in stripped or "\\(" in stripped or "\\[" in stripped
            
            if has_math_chars and not has_delimiters:
                processed_lines.append(f"\\[ {stripped} \\]")
            else:
                processed_lines.append(line)
        
        return "\n".join(processed_lines)

    def _generate(self, prompt: str, step_name: str = "generate") -> str:
        if self.mock_mode:
            return "Mock Generation"

        # Construct messages for Chat Model
        messages = [
            {"role": "system", "content": "You are a helpful assistant and a LaTeX expert."},
            {"role": "user", "content": prompt}
        ]

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
        
        # Apply Deterministic Sanitization
        result = self._post_process_latex(result)
            
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
        7. **CRITICAL:** Convert all Unicode math symbols (e.g. →, ≠, ≤) to LaTeX commands inside math mode (e.g. $\\rightarrow$, $\\neq$, $\\leq$).
        8. **CRITICAL:** Arrows like \\Rightarrow, \\rightarrow MUST be inside math delimiters (e.g. $\\Rightarrow$).
        9. Wrap all single-letter mathematical variables in math mode (e.g., $x$, $y$).
        
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
        5. Check for "Missing $" errors: ensure all math symbols (\\Rightarrow, \\pm, etc.) are inside $...$.
        """
        return self._generate(prompt, step_name="refine_fix")
