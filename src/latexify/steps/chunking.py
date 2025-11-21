from typing import List, Dict, Any
from dataclasses import asdict
from latexify.core.pipeline import PipelineStep, DocumentState
from latexify.core.chunking_engine import chunk_text
from latexify.core.layout_engine import LayoutRegion
try:
    from latexify.pipeline.semantic_chunking import SemanticChunker
except ImportError:
    SemanticChunker = None

class ChunkingStep(PipelineStep):
    def __init__(self, chunk_chars: int = 1200):
        super().__init__("Chunking")
        self.chunk_chars = chunk_chars
        self.semantic_chunker = SemanticChunker() if SemanticChunker else None

    async def process(self, state: DocumentState) -> DocumentState:
        # Reconstruct layout_segments from ocr_blocks
        layout_segments: Dict[int, List[LayoutRegion]] = {}
        for block in state.ocr_blocks:
            idx = block.page_index
            if idx not in layout_segments:
                layout_segments[idx] = []
            
            region = LayoutRegion(
                text=block.text,
                tag=block.tag,
                bbox=block.bbox,
                column=block.metadata.get("column", 1),
                order=block.metadata.get("order", 0),
                font_size=block.metadata.get("font_size", 0.0),
                extras=block.metadata
            )
            layout_segments[idx].append(region)

        # Prepare page images map (empty for now as we don't pass them through state yet)
        page_images: Dict[int, List[str]] = {} 
        page_image_paths: Dict[int, str] = {} 
        
        chunks = chunk_text(
            pages=state.pages,
            page_images=page_images,
            chunk_chars=self.chunk_chars,
            ocr_content=state.ocr_content,
            page_image_paths=page_image_paths,
            semantic_chunker=self.semantic_chunker,
            layout_segments=layout_segments
        )
        
        state.chunks = [asdict(c) for c in chunks]
        
        state.add_log(f"Created {len(chunks)} chunks.")
        return state
