import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from latexify.pipeline.ingestion_mineru import MinerUIngestion

def test_semantic_block_processing():
    # Setup mock
    adapter = MinerUIngestion()
    
    # Mock MinerU layout blocks
    # Scenario:
    # 1. Title
    # 2. Text intro
    # 3. Question 1
    # 4. Equation
    # 5. Image (Figure)
    
    blocks = [
        {"type": "title", "text": "Sample Exam", "bbox": [0,0,100,20], "page_idx": 0},
        {"type": "text", "text": "This is the intro text.", "bbox": [0,30,100,50], "page_idx": 0},
        {"type": "text", "text": "Question 1. Solve for x.", "bbox": [0,60,100,80], "page_idx": 0},
        {"type": "interline_equation", "text": "x^2 + y^2 = z^2", "bbox": [0,90,100,110], "page_idx": 0},
        {"type": "image", "text": "", "bbox": [0,120,100,200], "page_idx": 0, "img_path": "fig1.jpg"},
        {"type": "text", "text": "End of section.", "bbox": [0,210,100,230], "page_idx": 0},
    ]
    
    chunk_chars = 1000
    
    # Run processing
    chunks = adapter._semantic_block_processing(blocks, chunk_chars)
    
    # Assertions
    assert len(chunks) >= 4 # Title, Intro, Question+Eq, Figure, End... logic depends on flushing
    
    # 1. Title -> Heading
    assert chunks[0].metadata["tag"] == "heading"
    assert chunks[0].text == "Sample Exam"
    
    # 2. Intro -> Text
    # It might be merged if not forced break, but title usually forces break if tag changes?
    # _refine_tag: title -> heading. text -> text.
    # refined_tag changed from heading to text -> is_tag_change=True -> Flush.
    assert chunks[1].metadata["tag"] == "text"
    assert chunks[1].text == "This is the intro text."
    
    # 3. Question -> Question
    # "Question 1..." matches QUESTION_RE -> refined_tag="question".
    # current_tag was "text". refined="question" -> is_semantic_break=True -> Flush previous buffer (Intro).
    # New buffer starts with Question. tag="question".
    # Next block: Equation. refined="equation".
    # refined="equation" vs current="question".
    # is_semantic_break=False (equation not in list).
    # is_tag_change=True.
    # Wait, does equation force flush?
    # refined_tag in ["question", "figure", "table", "heading"] -> Equation is NOT in this list in the code I wrote?
    # Code: is_semantic_break = refined_tag in ["question", "figure", "table", "heading"]
    # "equation" is NOT there.
    # So Equation is appended to Question buffer?
    # But is_tag_change = refined_tag != current_tag (equation != question) -> True.
    # So it flushes Question.
    
    # So chunk 2 is Question.
    assert chunks[2].metadata["tag"] == "question"
    assert "Question 1" in chunks[2].text
    
    # 4. Equation
    # chunk 3 is Equation.
    assert chunks[3].metadata["tag"] == "equation"
    assert "x^2" in chunks[3].text
    assert chunks[3].metadata["contains_equations"] is True
    
    # 5. Figure
    # chunk 4 is Figure.
    assert chunks[4].metadata["tag"] == "figure"
    assert chunks[4].metadata["contains_images"] is True
    
    # 6. End text
    # chunk 5 is text.
    assert chunks[5].metadata["tag"] == "text"
    assert "End of section" in chunks[5].text

if __name__ == "__main__":
    test_semantic_block_processing()
