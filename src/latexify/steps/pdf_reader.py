from pypdf import PdfReader
from latexify.core.pipeline import PipelineStep, DocumentState

class PDFReaderStep(PipelineStep):
    def __init__(self):
        super().__init__("PDFReader")

    async def process(self, state: DocumentState) -> DocumentState:
        if not state.file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {state.file_path}")

        try:
            reader = PdfReader(str(state.file_path))
            text_parts = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                text_parts.append(text)
            
            state.pages = text_parts
            state.raw_text = "\n\n".join(text_parts)
            state.add_log(f"Extracted {len(text_parts)} pages of text.")
        except Exception as e:
            state.add_log(f"PDF extract error: {e}")
            # We don't fail the pipeline if raw text extraction fails, 
            # as OCR might pick it up later, but we log it.
            # If critical, we could raise.
            
        return state
