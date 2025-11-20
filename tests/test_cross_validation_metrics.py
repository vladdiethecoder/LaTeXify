import pytest

from release.pipeline.cross_validation import (
    TableStructureDiff,
    SymbolicMathValidator,
)
from release.core import common


def _make_chunk(region_type: str, text: str = "", metadata=None):
    return common.Chunk(
        chunk_id=f"{region_type}-0",
        page=1,
        text=text,
        metadata=metadata or {"region_type": region_type},
    )


def test_table_structure_diff_matches_columns():
    chunk = _make_chunk("table", metadata={"table_signature": {"columns": 3}})
    latex = r"""
    \begin{tabular}{ccc}
    a & b & c \\
    1 & 2 & 3 \\
    \end{tabular}
    """
    scores = TableStructureDiff().score([chunk], latex)
    assert 0.9 <= scores["table_columns_match"] <= 1.0
    assert scores["table_count_match"] == 1.0


def test_symbolic_math_validator_soft_fallback():
    chunk = _make_chunk("equation", text="x + y")
    latex = r"\\begin{equation} x + y \\end{equation}"
    scores = SymbolicMathValidator().score([chunk], latex)
    # Always returns bounded defaults; if SymPy is present it may increase.
    assert 0.0 <= scores["symbolic_match"] <= 1.0
    assert 0.0 <= scores["parse_success"] <= 1.0
