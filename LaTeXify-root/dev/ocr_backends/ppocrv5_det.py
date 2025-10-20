from .base import OCRBackend, OCRPage
class Backend(OCRBackend):
    name = "pp-ocrv5-det"
    def recognize_page(self, image_path: str, page_num: int = 1) -> OCRPage:
        # Placeholder: a real impl would run PaddleOCR det and return blocks with bboxes.
        return OCRPage(model=self.name, page=page_num, text_md="", blocks=[])
