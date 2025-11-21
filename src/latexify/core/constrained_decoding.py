"""
Constrained Decoding for LaTeX using Outlines.
Enforces Context-Free Grammar (CFG) constraints on LLM generation to ensure valid LaTeX syntax.
"""
import outlines
import torch
from enum import Enum
from typing import List, Optional

class LatexGrammar(str, Enum):
    MATH_ONLY = "math_only"
    DOCUMENT = "document"

class ConstrainedGenerator:
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.2", device: str = "cuda"):
        self.model = outlines.models.transformers(model_name, device=device)
        
        # Define simplified LaTeX grammar
        # This ensures balanced environments and valid commands
        self.latex_grammar = r"""
            ?start: text
            text: (content | environment | math)*
            content: /[^{}\\n]+/  # Corrected: Added \\n to match newline characters within content
            environment: "\\begin{" name "}" text "\\end{" name "}"
            math: "$" /[^$]+/ "$" | "$$" /[^$]+/ "$$"
            name: /[a-zA-Z*]+/ 
        """
        
        self.generator = outlines.generate.cfg(self.model, self.latex_grammar)

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """
        Generate LaTeX content strictly adhering to the CFG.
        """
        return self.generator(prompt, max_tokens=max_tokens)

# Singleton instance for reuse
_instance: Optional[ConstrainedGenerator] = None

def get_generator(device="cuda"):
    global _instance
    if _instance is None:
        try:
            if not torch.cuda.is_available():
                device = "cpu"
            _instance = ConstrainedGenerator(device=device)
        except Exception as e:
            print(f"Failed to initialize ConstrainedGenerator: {e}")
            return None
    return _instance
