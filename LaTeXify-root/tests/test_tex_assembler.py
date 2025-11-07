from __future__ import annotations

import json
from types import SimpleNamespace

from latexify.pipeline import tex_assembler


def _write_plan(tmp_path):
    plan = {
        "doc_class": "lix_textbook",
        "frontmatter": {"title": "Sample", "author": "Tester", "course": "STAT 101"},
        "content_flags": {"has_figures": True, "has_tables": True, "has_math": True, "has_code": False},
        "tasks": [
            {"id": "PREAMBLE", "kind": "preamble", "order": 0},
            {"id": "TITLE", "kind": "title", "order": 1},
            {"id": "Q1", "title": "Linear Models", "kind": "section", "block_id": "block-1", "order": 2},
            {"id": "FIG1", "title": "Trend Plot", "kind": "figure", "block_id": "block-2", "asset_path": "assets/plot.png", "order": 3},
        ],
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    return path


def _write_consensus(tmp_path):
    entries = [
        {
            "block_id": "block-1",
            "text": "Intro paragraph with multiple sentences.\nIt explains the concept of slope.",
            "block_type": "text",
            "page_index": 0,
            "ocr_outputs": {"internvl": "Intro paragraph"},
        },
        {
            "block_id": "block-2",
            "text": "figure placeholder",
            "block_type": "figure",
            "page_index": 0,
            "ocr_outputs": {},
        },
    ]
    path = tmp_path / "blocks_refined.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    return path


def test_tex_assembler_generates_snippets(tmp_path):
    plan_path = _write_plan(tmp_path)
    consensus_path = _write_consensus(tmp_path)
    snippets_dir = tmp_path / "snippets"
    golden_dir = tmp_path / "golden"
    golden_dir.mkdir()
    (golden_dir / "Q1.tex").write_text("% golden snippet\n\\section*{Linear Models}\n", encoding="utf-8")
    args = SimpleNamespace(
        plan=plan_path,
        consensus=consensus_path,
        layout_plan=None,
        snippets_dir=snippets_dir,
        golden_dir=golden_dir,
        model_path=None,
        n_ctx=4096,
        temperature=0.2,
        max_tokens=600,
        aggregate=False,
        build_dir=None,
    )
    tex_assembler.assemble(args)
    q1 = (snippets_dir / "Q1.tex").read_text(encoding="utf-8")
    assert q1.startswith("% golden snippet")
    fig_snip = (snippets_dir / "FIG1.tex").read_text(encoding="utf-8")
    assert "\\includegraphics" in fig_snip
    meta = json.loads((snippets_dir / "Q1.meta.json").read_text(encoding="utf-8"))
    assert meta["task_id"] == "Q1"
    preamble = (snippets_dir / "PREAMBLE.tex").read_text(encoding="utf-8")
    assert "\\usepackage{graphicx}" in preamble
    assert "\\usepackage{booktabs}" in preamble
