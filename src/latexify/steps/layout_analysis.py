from latexify.core.pipeline import PipelineStep, DocumentState
from latexify.core.layout_engine import LayoutAnalyzer
from latexify.core.state import TextBlock

class LayoutAnalysisStep(PipelineStep):
    def __init__(self):
        super().__init__("LayoutAnalysis")

    async def process(self, state: DocumentState) -> DocumentState:
        analyzer = LayoutAnalyzer(state.file_path)
        
        if not analyzer.enabled:
            state.add_log("LayoutAnalysis skipped: fitz (PyMuPDF) not available.")
            return state

        # Get page count safely
        try:
            import fitz
            doc = fitz.open(str(state.file_path))
            page_count = doc.page_count
            doc.close()
        except ImportError:
            state.add_log("LayoutAnalysis failed: fitz not installed.")
            return state
        except Exception as e:
            state.add_log(f"LayoutAnalysis failed to open PDF: {e}")
            return state
        
        regions_map = analyzer.analyze_document(page_count)
        analyzer.close()
        
        # Convert LayoutRegions to TextBlocks
        for page_idx, regions in regions_map.items():
            for region in regions:
                # Map LayoutRegion to TextBlock
                block = TextBlock(
                    text=region.text,
                    confidence=region.extras.get("layout_confidence", 1.0),
                    bbox=region.bbox,
                    page_index=page_idx,
                    tag=region.tag,
                    metadata=region.extras
                )
                state.ocr_blocks.append(block)
                
        state.add_log(f"Layout analysis completed. Found {len(state.ocr_blocks)} regions.")
        return state
