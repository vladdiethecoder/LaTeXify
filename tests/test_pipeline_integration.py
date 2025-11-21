import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from omegaconf import OmegaConf
from latexify.core.pipeline import LaTeXifyPipeline

@pytest.fixture
def mock_config():
    return OmegaConf.create({
        "pipeline": {
            "ingestion": {"dpi": 72},
            "layout": {"model": "yolov10n.pt"},
            "ocr": {"lang": "en"},
            "refinement": {"enabled": True, "use_vllm": False},
            "metadata_extraction": False
        }
    })

@patch("latexify.core.pipeline.PyMuPDFIngestor")
@patch("latexify.core.pipeline.YOLOLayoutEngine")
@patch("latexify.core.pipeline.cv2")
@patch("latexify.core.pipeline.Image")
def test_pipeline_process(mock_pil_image, mock_cv2, mock_layout_cls, mock_ingestor_cls, mock_config):
    # Setup Mocks
    mock_ingestor = mock_ingestor_cls.return_value
    mock_ingestor.ingest.return_value = [Path("page1.png")]
    
    mock_layout = mock_layout_cls.return_value
    # Mock detection: 1 title, 1 equation
    mock_layout.detect.return_value = [
        {'bbox': [0, 0, 100, 50], 'class': 'Title', 'confidence': 0.9},
        {'bbox': [0, 60, 100, 100], 'class': 'Equation_Display', 'confidence': 0.8}
    ]
    
    # Mock Image loading
    mock_img = MagicMock()
    mock_img.shape = (1000, 1000, 3)
    # Slicing returns another mock
    mock_img.__getitem__.return_value = mock_img 
    mock_cv2.imread.return_value = mock_img
    mock_cv2.cvtColor.return_value = mock_img
    
    # Mock PIL
    mock_pil_image.fromarray.return_value = MagicMock()
    
    pipeline = LaTeXifyPipeline(mock_config)
    
    # Force mock components where dependency injection isn't fully clean yet
    pipeline.math_ocr.predict_batch = MagicMock(return_value=["E=mc^2"])
    pipeline.text_ocr.recognize = MagicMock(return_value="Recognized Title")
    pipeline.refiner.refine = MagicMock(return_value="Refined LaTeX")
    pipeline.compiler.compile = MagicMock(return_value=(True, ""))
    
    result = pipeline.process(Path("dummy.pdf"))
    
    assert result == "Refined LaTeX"
    pipeline.ingestor.ingest.assert_called_once()
    pipeline.layout.detect.assert_called_once()
    pipeline.math_ocr.predict_batch.assert_called_once()
