import fitz
import sys
from pathlib import Path

pdf_path = Path("src/latexify/inputs/sample.pdf")
print(f"Opening {pdf_path}")
doc = fitz.open(pdf_path)
print(f"Page count: {len(doc)}")

prev_text = ""
for i, page in enumerate(doc):
    text = page.get_text()
    start = text[:50].replace("\n", " ")
    is_dup = text == prev_text
    print(f"Page {i}: {start} (Duplicate: {is_dup})")
    prev_text = text

