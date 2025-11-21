import pytest
from latexify.core.reading_order import ReadingOrder, LayoutBlock
from latexify.core.assembler import Assembler

@pytest.fixture
def sample_blocks():
    return [
        LayoutBlock(id="1", bbox=[100, 200, 200, 250], category="Text_Block", confidence=0.9, page_num=0, content="World"),
        LayoutBlock(id="2", bbox=[100, 100, 200, 150], category="Title", confidence=0.95, page_num=0, content="Hello"),
        LayoutBlock(id="3", bbox=[100, 100, 200, 150], category="Text_Block", confidence=0.9, page_num=1, content="Page 2"),
    ]

def test_reading_order_sort(sample_blocks):
    ro = ReadingOrder()
    sorted_blocks = ro.sort(sample_blocks)
    
    # Expecting: Page 0 Top (Title "Hello") -> Page 0 Bottom (Text "World") -> Page 1
    assert sorted_blocks[0].content == "Hello"
    assert sorted_blocks[1].content == "World"
    assert sorted_blocks[2].content == "Page 2"

def test_assembler_basic():
    assembler = Assembler()
    blocks = [
        LayoutBlock(id="1", bbox=[0,0,0,0], category="Title", confidence=1.0, page_num=0, content="My Title"),
        LayoutBlock(id="2", bbox=[0,0,0,0], category="Text_Block", confidence=1.0, page_num=0, content="Some text."),
        LayoutBlock(id="3", bbox=[0,0,0,0], category="Equation_Display", confidence=1.0, page_num=0, content="E=mc^2"),
    ]
    
    latex = assembler.assemble(blocks)
    
    assert r"\title{My Title}" in latex
    assert "Some text." in latex
    assert r"\[" in latex
    assert "E=mc^2" in latex
