#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run-level evaluation harness for LaTeXify outputs.

This script resolves pipeline outputs against golden fixtures under
``dev/eval/fixtures``.  It compares the compiled ``build/main.tex`` and
``build/main.pdf`` artefacts against the reference ``main.tex`` / ``reference.pdf``
for the matched fixture, computing text metrics (WER, CER, BLEU-ish, etc.)
and page-level dSSIM scores.  A JSON report is emitted summarising the
results along with any compile or pipeline metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import sys

SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dev.eval.pdf_reference import (
    compute_dssim_for_pdfs,
    extract_pdf_reference,
)
from scripts.metrics_text import TextScores


@dataclass
class Fixture:
    identifier: str
    title: str
    tex_path: Path
    pdf_path: Path
    tex_sha256: str

    @classmethod
    def from_manifest(cls, entry: Dict[str, Any]) -> "Fixture":
        return cls(
            identifier=entry["id"],
            title=entry.get("title", entry["id"]),
            tex_path=Path(entry["tex_path"]).resolve(),
            pdf_path=Path(entry["pdf_path"]).resolve(),
            tex_sha256=entry.get("pipeline_input", {}).get("build_main_tex_sha256", ""),
        )


def _load_manifest(fixtures_root: Path) -> List[Fixture]:
    manifest_path = fixtures_root / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Missing fixtures manifest at {manifest_path}")
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    return [Fixture.from_manifest(entry) for entry in raw]


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_fixture(
    fixtures: List[Fixture],
    main_tex: Path,
    explicit_id: Optional[str] = None,
) -> Fixture:
    if explicit_id:
        for fx in fixtures:
            if fx.identifier == explicit_id:
                return fx
        raise SystemExit(f"Fixture '{explicit_id}' not found in manifest")
    if not main_tex.exists():
        raise SystemExit(f"Cannot resolve fixture because {main_tex} does not exist")
    main_hash = _hash_file(main_tex)
    for fx in fixtures:
        if fx.tex_sha256 and fx.tex_sha256 == main_hash:
            return fx
    raise SystemExit(
        "Unable to match build/main.tex to any fixture. "
        "Provide --fixture-id explicitly or ensure the manifest is up to date."
    )


def _text_scores(hyp: str, ref: str) -> Dict[str, Any]:
    scores = TextScores.compute(hyp, ref)
    return asdict(scores)


def _evaluate_pdf_text(run_pdf: Path, ref_pdf: Path) -> Dict[str, Any]:
    """Compute text-level metrics for each page by extracting text from PDFs."""
    if not run_pdf.exists() or not ref_pdf.exists():
        status = "missing-run" if not run_pdf.exists() else "missing-reference"
        return {"status": status, "per_page": [], "macro_average": {}}
    hyp_pages = extract_pdf_reference(str(run_pdf))
    ref_pages = extract_pdf_reference(str(ref_pdf))
    per_page: List[Dict[str, Any]] = []
    page_count = max(len(hyp_pages), len(ref_pages))
    for idx in range(page_count):
        page_no = idx + 1
        hyp_text = hyp_pages[idx].text if idx < len(hyp_pages) else ""
        ref_text = ref_pages[idx].text if idx < len(ref_pages) else ""
        scores = TextScores.compute(hyp_text, ref_text)
        per_page.append({"page": page_no, "scores": asdict(scores)})
    # macro average across numeric metrics
    metric_names = [
        "wer",
        "cer",
        "lev_dist",
        "jaccard_unigram",
        "bleuish",
    ]
    macro: Dict[str, Optional[float]] = {}
    for name in metric_names:
        values = [entry["scores"].get(name) for entry in per_page]
        numeric = [float(v) for v in values if isinstance(v, (int, float))]
        if not numeric:
            macro[name] = None
        else:
            macro[name] = float(statistics.fmean(numeric))
    macro["pages"] = page_count
    return {"status": "ok", "per_page": per_page, "macro_average": macro}


def _evaluate_dssim(run_pdf: Path, ref_pdf: Path) -> Dict[str, Any]:
    """Compute structural similarity between PDFs page by page."""
    if not run_pdf.exists() or not ref_pdf.exists():
        status = "missing-run" if not run_pdf.exists() else "missing-reference"
        return {"status": status, "per_page": [], "summary": {}}
    per_page = compute_dssim_for_pdfs(str(run_pdf), str(ref_pdf))
    numeric = [entry["dssim"] for entry in per_page if entry.get("dssim") is not None]
    summary = {
        "mean_dssim": float(statistics.fmean(numeric)) if numeric else None,
        "max_dssim": float(max(numeric)) if numeric else None,
        "pages_compared": len(numeric),
    }
    return {"status": "ok", "per_page": per_page, "summary": summary}


def _load_json_if_exists(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a LaTeXify run against reference fixtures")
    ap.add_argument("--run-dir", type=str, default=None, help="Run folder (defaults to latest in dev/runs)")
    ap.add_argument("--build-dir", type=str, default="build", help="Build directory containing pipeline outputs")
    ap.add_argument("--fixtures-root", type=str, default="dev/eval/fixtures", help="Fixtures root")
    ap.add_argument("--fixture-id", type=str, default=None, help="Override fixture identifier")
    ap.add_argument("--output", type=str, default=None, help="Optional output path for the aggregated JSON report")
    args = ap.parse_args()
    fixtures_root = Path(args.fixtures_root).resolve()
    fixtures = _load_manifest(fixtures_root)
    build_dir = Path(args.build_dir).resolve()
    main_tex = build_dir / "main.tex"
    run_pdf = build_dir / "main.pdf"
    # Determine the run directory: explicit or most recent under dev/runs
    run_dir: Optional[Path]
    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
    else:
        runs_root = Path("dev") / "runs"
        candidates = sorted(runs_root.glob("*"))
        run_dir = candidates[-1].resolve() if candidates else None
    # Resolve which fixture to compare against
    fixture = _resolve_fixture(fixtures, main_tex, args.fixture_id)
    # Compute metrics
    run_tex_text = main_tex.read_text(encoding="utf-8") if main_tex.exists() else ""
    ref_tex_text = fixture.tex_path.read_text(encoding="utf-8") if fixture.tex_path.exists() else ""
    tex_metrics = _text_scores(run_tex_text, ref_tex_text)
    pdf_text_eval = _evaluate_pdf_text(run_pdf, fixture.pdf_path)
    dssim_eval = _evaluate_dssim(run_pdf, fixture.pdf_path)
    compile_metrics = _load_json_if_exists(build_dir / "compile_metrics.json")
    pipeline_metrics = _load_json_if_exists(build_dir / "run_metrics.json")
    report = {
        "run_dir": str(run_dir) if run_dir else None,
        "build_dir": str(build_dir),
        "fixture": {
            "id": fixture.identifier,
            "title": fixture.title,
            "tex_path": str(fixture.tex_path),
            "pdf_path": str(fixture.pdf_path),
            "tex_sha256": fixture.tex_sha256,
        },
        "tex_comparison": {"scores": tex_metrics},
        "pdf_text": pdf_text_eval,
        "pdf_dssim": dssim_eval,
        "compile_metrics": compile_metrics,
        "pipeline_metrics": pipeline_metrics,
    }
    if args.output:
        output_path = Path(args.output)
    elif run_dir:
        output_path = Path(run_dir) / "evaluation.json"
    else:
        output_path = build_dir / "evaluation.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote evaluation report to {output_path}")


if __name__ == "__main__":
    main()