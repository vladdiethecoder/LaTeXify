from collections import Counter
from typing import Set

class HallucinationChecker:
    def __init__(self):
        pass

    def get_ngrams(self, text: str, n: int) -> Set[str]:
        words = text.lower().split()
        return set(" ".join(words[i:i+n]) for i in range(len(words)-n+1))

    def check_overlap(self, source_text: str, generated_latex: str, n: int = 3) -> float:
        """
        Check n-gram overlap between source OCR text and generated LaTeX.
        This is a heuristic to detect massive hallucinations.
        
        Args:
            source_text: Raw text from simple OCR (e.g. Tesseract) on the same region.
            generated_latex: The LLM/Model generated LaTeX.
            
        Returns:
            float: Overlap score (0.0 to 1.0).
        """
        # Normalize latex: remove commands roughly
        import re
        clean_latex = re.sub(r'\\ [a-zA-Z]+', ' ', generated_latex)
        clean_latex = re.sub(r'[\{\}]', ' ', clean_latex)
        
        source_ngrams = self.get_ngrams(source_text, n)
        latex_ngrams = self.get_ngrams(clean_latex, n)
        
        if not source_ngrams:
            return 1.0 # If no source text, assume okay or indeterminate
            
        intersection = source_ngrams.intersection(latex_ngrams)
        return len(intersection) / len(source_ngrams)
