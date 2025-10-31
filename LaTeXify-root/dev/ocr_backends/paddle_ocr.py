from paddleocr import PaddleOCR

class Backend:
    name = "paddleocr"
    def __init__(self, dtype="auto"):
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True)

    def recognize_page(self, image_path, page=1):
        result = self.ocr.ocr(image_path, cls=True)
        # Convert PaddleOCR result to Markdown (simple lines)
        lines = []
        for res in result:
            for line in res:
                lines.append(line[1][0])
        md_text = "\n".join(lines)
        return OCRResult(model=self.name, page=os.path.basename(image_path), text_md=md_text, blocks=None)
