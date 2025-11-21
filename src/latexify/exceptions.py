class LatexifyError(Exception):
    """Base exception for all LaTeXify errors."""
    pass

class ModelLoadError(LatexifyError):
    """Raised when a model fails to load or weights are missing."""
    pass

class LayoutAnalysisError(LatexifyError):
    """Raised when layout analysis fails or produces invalid results."""
    pass

class OCRError(LatexifyError):
    """Raised when an OCR backend fails."""
    pass

class CompilationError(LatexifyError):
    """Raised when LaTeX compilation fails."""
    pass

class PipelineConfigurationError(LatexifyError):
    """Raised when the pipeline is misconfigured."""
    pass

class HallucinationError(LatexifyError):
    """Raised when the model output is determined to be hallucinated or invalid."""
    pass
