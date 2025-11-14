from pathlib import Path

from release.core import common
from release.pipeline import rag, specialists


def test_table_agent_uses_rag_reference():
    entry_payload = {
        "entry_id": "ref-table-001",
        "doc_id": "reference.tex",
        "type": "table",
        "text": "\\begin{table}\\centering\\begin{tabular}{lcc}\\toprule A & B \\\\\\bottomrule\\end{tabular}\\caption{Ref}\\end{table}",
        "packages": ["booktabs"],
    }
    entry = rag.RAGEntry.from_json(entry_payload)
    preamble = specialists.PreambleAgent()
    chunk = common.Chunk(
        chunk_id="chunk-1",
        page=1,
        text="A | B",
        images=[],
        metadata={"region_type": "table", "table_signature": {"columns": 2}},
    )
    result = specialists.dispatch_specialist("table", chunk, preamble, [entry])
    assert "RAG reference" in result.latex
    assert "booktabs" in [pkg["package"] for pkg in preamble.packages()]


def test_extract_environments_infers_domain(tmp_path):
    source_dir = tmp_path / "reference_tex"
    domain_dir = source_dir / "math"
    domain_dir.mkdir(parents=True)
    tex_path = domain_dir / "sample.tex"
    tex_path.write_text(
        """\\begin{equation}E=mc^2\\end{equation}""",
        encoding="utf-8",
    )
    entries = rag.extract_environments(tex_path, source_dir)
    assert entries, "expected at least one entry"
    assert entries[0].domain == "math"


def test_rag_search_prefers_requested_domain():
    entries = [
        rag.RAGEntry(
            entry_id="doc-math-000",
            doc_id="math.tex",
            snippet_type="equation",
            text="\\begin{equation}a=b\\end{equation}",
            packages=[],
            embedding=[0.0] * rag.EMBED_DIM,
            domain="math",
        ),
        rag.RAGEntry(
            entry_id="doc-finance-000",
            doc_id="finance.tex",
            snippet_type="equation",
            text="\\begin{equation}x=y\\end{equation}",
            packages=[],
            embedding=[0.0] * rag.EMBED_DIM,
            domain="finance",
        ),
    ]
    # Give the math entry a slightly lower cosine score so the domain boost must take effect.
    entries[0].embedding[0] = 0.4
    entries[1].embedding[0] = 0.5
    index = rag.RAGIndex(entries)
    results = index.search("formula", "equation", k=1, domain="math")
    assert results and results[0].domain == "math"


def test_dispatch_specialist_injects_context_comment():
    preamble = specialists.PreambleAgent()
    chunk = common.Chunk(chunk_id="c3", page=1, text="Paragraph text", images=[], metadata={})
    result = specialists.dispatch_specialist(
        "text",
        chunk,
        preamble,
        [],
        context={"section_title": "Biology", "section_summary": "Cells and tissues"},
    )
    assert "% context:" in result.latex
