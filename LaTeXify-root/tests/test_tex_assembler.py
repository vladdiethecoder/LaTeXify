from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from latexify.pipeline import tex_assembler


def _write_plan(tmp_path, *, figure_asset: str | None = "assets/plot.png"):
    plan = {
        "doc_class": "lix_textbook",
        "frontmatter": {"title": "Sample", "author": "Tester", "course": "STAT 101"},
        "content_flags": {"has_figures": True, "has_tables": True, "has_math": True, "has_code": False},
        "tasks": [
            {"id": "PREAMBLE", "kind": "preamble", "order": 0},
            {"id": "TITLE", "kind": "title", "order": 1},
            {"id": "Q1", "title": "Linear Models", "kind": "section", "block_id": "block-1", "order": 2},
            {
                "id": "FIG1",
                "title": "Trend Plot",
                "kind": "figure",
                "block_id": "block-2",
                **({"asset_path": figure_asset} if figure_asset else {}),
                "order": 3,
            },
        ],
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    return path


def _write_layout_plan(tmp_path):
    blueprint = {
        "version": "layout-planner/v0.1",
        "model_name": "unit-test",
        "created_at": "2025-01-01T00:00:00Z",
        "plan": {
            "page_layout": {"columns": "two-column", "title_area": "Full width"},
            "sections": [
                {
                    "id": "Q1",
                    "title": "Linear Models",
                    "layout": "Two-column narrative with callouts",
                    "kind": "subsection",
                    "notes": ["Wrap content in multicol"],
                }
            ],
            "content_flags": {"two_column": True},
        }
    }
    path = tmp_path / "layout_plan.json"
    path.write_text(json.dumps(blueprint, indent=2), encoding="utf-8")
    return path


def _write_consensus(tmp_path, *, figure_assets: list[str] | None = None):
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
    if figure_assets:
        entries[1]["assets"] = figure_assets
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


def test_section_snippet_prefers_block_assets_for_figures(tmp_path):
    plan_path = _write_plan(tmp_path, figure_asset=None)
    consensus_path = _write_consensus(tmp_path, figure_assets=["assets/page0001-img01.png"])
    snippets_dir = tmp_path / "snippets"
    args = SimpleNamespace(
        plan=plan_path,
        consensus=consensus_path,
        layout_plan=None,
        snippets_dir=snippets_dir,
        golden_dir=None,
        model_path=None,
        n_ctx=4096,
        temperature=0.2,
        max_tokens=600,
        aggregate=False,
        build_dir=None,
    )
    tex_assembler.assemble(args)
    fig_snip = (snippets_dir / "FIG1.tex").read_text(encoding="utf-8")
    assert "assets/page0001-img01.png" in fig_snip


def test_table_parser_handles_multiline_alignment():
    assembler = tex_assembler.TexAssembler(llm=None)
    task = tex_assembler.PlanTask(
        task_id="TAB1",
        title="Results Table",
        kind="table",
        content_type="table",
        block_id="block-table",
        order=2,
        asset_path=None,
        notes={},
    )
    table_text = (
        "Item | Value\n"
        "Alpha | 10\n"
        "Beta | 12.5\n"
        "Gamma details\n"
        "continue | 7\n"
    )
    block = tex_assembler.ConsensusBlock(
        block_id="block-table",
        text=table_text,
        block_type="table",
        page_index=0,
        flagged=False,
        ocr_outputs={},
        assets=[],
    )
    snippet = assembler.section_snippet(task, block)
    assert "\\begin{tabular}{lr}" in snippet
    assert "Gamma details continue" in snippet


def test_layout_plan_influences_sections(tmp_path):
    plan_path = _write_plan(tmp_path)
    consensus_path = _write_consensus(tmp_path)
    snippets_dir = tmp_path / "snippets"
    layout_plan = _write_layout_plan(tmp_path)
    args = SimpleNamespace(
        plan=plan_path,
        consensus=consensus_path,
        layout_plan=layout_plan,
        snippets_dir=snippets_dir,
        golden_dir=None,
        model_path=None,
        n_ctx=4096,
        temperature=0.2,
        max_tokens=600,
        aggregate=False,
        build_dir=None,
    )
    tex_assembler.assemble(args)
    q1 = (snippets_dir / "Q1.tex").read_text(encoding="utf-8")
    assert q1.startswith("\\subsection")
    assert "\\begin{multicols}{2}" in q1
    preamble = (snippets_dir / "PREAMBLE.tex").read_text(encoding="utf-8")
    assert "\\usepackage{multicol}" in preamble


def test_load_plan_validation_error(tmp_path):
    bad_plan = {
        "frontmatter": {},
        "tasks": [
            {"title": "Missing identifier"},
        ],
    }
    path = tmp_path / "bad_plan.json"
    path.write_text(json.dumps(bad_plan), encoding="utf-8")
    with pytest.raises(ValueError):
        tex_assembler.load_plan(path)


def test_load_consensus_validation_error(tmp_path):
    path = tmp_path / "blocks_refined.jsonl"
    path.write_text(json.dumps({"text": "no id"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        tex_assembler.load_consensus(path)


def test_layout_plan_validation_error(tmp_path):
    plan_path = _write_plan(tmp_path)
    consensus_path = _write_consensus(tmp_path)
    bad_layout = tmp_path / "layout_plan.json"
    bad_layout.write_text(json.dumps({"version": "v1"}), encoding="utf-8")
    args = SimpleNamespace(
        plan=plan_path,
        consensus=consensus_path,
        layout_plan=bad_layout,
        snippets_dir=tmp_path / "snippets",
        golden_dir=None,
        model_path=None,
        n_ctx=4096,
        temperature=0.2,
        max_tokens=600,
        aggregate=False,
        build_dir=None,
    )
    with pytest.raises(ValueError):
        tex_assembler.assemble(args)
