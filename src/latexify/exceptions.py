class LatexifyError(Exception):
    """Base exception for all LaTeXify errors."""
    pass

class ModelLoadError(LatexifyError):
    """Raised when a model fails to load or weights are missing."""

class LayoutDetectionError(LaTeXifyError):
    """Raised when layout detection fails or produces no results."""
    pass


class ModelLoadingError(LaTeXifyError):
    """Raised when a model fails to load (CUDA OOM, missing weights, etc.)."""
    pass


class CompilationError(LaTeXifyError):
    """Raised when generated LaTeX fails to compile."""
    
    def __init__(self, message: str, latex_code: str = "", error_log: str = ""):
        super().__init__(message)
        self.latex_code = latex_code
        self.error_log = error_log


class LowConfidenceError(LaTeXifyError):
    """Raised when OCR/extraction confidence is below acceptable threshold."""
    
    def __init__(self, message: str, confidence: float, threshold: float):
        super().__init__(message)
        self.confidence = confidence
        self.threshold = threshold


class ExtractionError(LaTeXifyError):
    """Raised when content extraction fails for a specific region."""
    
    def __init__(self, message: str, region_id: str = "", extractor: str = ""):
        super().__init__(message)
        self.region_id = region_id
        self.extractor = extractor


class ReadingOrderError(LaTeXifyError):
    """Raised when reading order reconstruction fails."""
    pass


class EnvironmentError(LaTeXifyError):
    """Raised when required environment/dependencies are missing."""

class PipelineConfigurationError(LaTeXifyError):
    """Raised when the pipeline is misconfigured."""
    pass

class HallucinationError(LaTeXifyError):
    """Raised when the model output is determined to be hallucinated or invalid."""
    pass
