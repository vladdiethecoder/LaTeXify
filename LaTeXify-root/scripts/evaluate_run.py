#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run-level evaluation harness for LaTeXify outputs.

This script resolves pipeline outputs against golden fixtures under
``dev/eval/fixtures``. It compares the compiled ``build/main.tex`` and
``build/main.pdf`` artefacts against the reference ``main.tex`` / ``reference.pdf``
for the matched fixture, computing text metrics (BLEU-ish, WER, CER, etc.) and
page-level dSSIM scores.
"""

from __future__ import annotations
<<<<<<< ours

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
=======

import argparse
import hashlib
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import sys
>>>>>>> theirs

SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dev.eval.pdf_reference import (
    compute_dssim_for_pdfs,
    extract_pdf_reference,
)
from scripts.metrics_text import TextScores
<<<<<<< ours
from dev.eval.pdf_reference import compare_pdf_images, PDFImageComparison
=======


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
    payload = asdict(scores)
    return payload

>>>>>>> theirs

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUILD = REPO_ROOT / "build"
DEFAULT_GOLDEN = REPO_ROOT / "dev" / "eval" / "golden"
DEFAULT_SUMMARY = DEFAULT_BUILD / "pipeline_summary.json"
DEFAULT_SNIPPETS = DEFAULT_BUILD / "snippets"


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _discover_golden(golden_dir: Path) -> List[Dict[str, Path]]:
    entries: List[Dict[str, Path]] = []
    if not golden_dir.exists():
        return entries
    for item in sorted(golden_dir.iterdir()):
        if not item.is_dir():
            continue
        ref_tex = item / "reference.tex"
        ref_pdf = item / "reference.pdf"
        if ref_tex.exists() and ref_pdf.exists():
            entries.append({
                "id": item.name,
                "tex": ref_tex,
                "pdf": ref_pdf,
            })
    return entries


<<<<<<< ours
def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
=======

def _evaluate_pdf_text(run_pdf: Path, ref_pdf: Path) -> Dict[str, Any]:
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
        elif name == "lev_dist":
            macro[name] = float(statistics.fmean(numeric))
        else:
            macro[name] = float(statistics.fmean(numeric))
    macro["pages"] = page_count

    return {"status": "ok", "per_page": per_page, "macro_average": macro}


def _evaluate_dssim(run_pdf: Path, ref_pdf: Path) -> Dict[str, Any]:
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
>>>>>>> theirs
    except Exception:
        return None

<<<<<<< ours

def _aggregate_numeric(records: List[Dict[str, Any]], key: str) -> Optional[float]:
    values = [r["metrics"][key] for r in records if r.get("status") == "ok" and r.get("metrics") and r["metrics"].get(key) is not None]
    if not values:
        return None
    return sum(values) / len(values)


def evaluate_text(snippets_dir: Path, golden: List[Dict[str, Path]]) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    for entry in golden:
        snippet_tex = snippets_dir / f"{entry['id']}.tex"
        record: Dict[str, Any] = {
            "id": entry["id"],
            "reference_tex": str(entry["tex"]),
            "generated_tex": str(snippet_tex),
        }
        if not snippet_tex.exists():
            record.update({
                "status": "missing",
                "metrics": None,
            })
        else:
            hyp = _read_text(snippet_tex)
            ref = _read_text(entry["tex"])
            scores = TextScores.compute(hyp, ref)
            record.update({
                "status": "ok",
                "metrics": scores.as_dict(),
            })
        results.append(record)

    aggregates = {
        "wer": _aggregate_numeric(results, "wer"),
        "cer": _aggregate_numeric(results, "cer"),
        "lev_dist": _aggregate_numeric(results, "lev_dist"),
        "jaccard_unigram": _aggregate_numeric(results, "jaccard_unigram"),
        "bleuish": _aggregate_numeric(results, "bleuish"),
        "meteorish": _aggregate_numeric(results, "meteorish"),
        "evaluated": len([r for r in results if r.get("status") == "ok"]),
        "total": len(results),
    }
    return {"per_snippet": results, "aggregates": aggregates}


def evaluate_images(snippets_dir: Path, golden: List[Dict[str, Path]], dpi: int) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    for entry in golden:
        snippet_pdf = snippets_dir / f"{entry['id']}.pdf"
        record: Dict[str, Any] = {
            "id": entry["id"],
            "reference_pdf": str(entry["pdf"]),
            "generated_pdf": str(snippet_pdf),
        }
        if not snippet_pdf.exists():
            record.update({
                "status": "missing",
                "metrics": None,
            })
        else:
            try:
                comparison: PDFImageComparison = compare_pdf_images(entry["pdf"], snippet_pdf, dpi=dpi)
                record.update({
                    "status": "ok",
                    "metrics": {
                        "mean_dssim": comparison.mean_dssim,
                        "page_dssim": comparison.page_dssim,
                    },
                })
            except Exception as exc:
                record.update({
                    "status": "error",
                    "error": str(exc),
                    "metrics": None,
                })
        results.append(record)

    values = [r["metrics"]["mean_dssim"] for r in results if r.get("status") == "ok" and r.get("metrics")]
    aggregates = {
        "mean_dssim": (sum(values) / len(values)) if values else None,
        "evaluated": len([r for r in results if r.get("status") == "ok"]),
        "total": len(results),
    }
    return {"per_snippet": results, "aggregates": aggregates}


def _resolve_output_paths(summary: Dict[str, Any], build_dir: Path, out_json: Optional[Path]) -> List[Path]:
    targets = []
    if out_json:
        targets.append(out_json)
    else:
        targets.append(build_dir / "evaluation_summary.json")
    run_dir = summary.get("run_dir")
    if run_dir:
        run_path = Path(run_dir)
        targets.append(run_path / "evaluation_summary.json")
    return targets


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate generated snippets against golden references")
    ap.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD)
    ap.add_argument("--golden-dir", type=Path, default=DEFAULT_GOLDEN)
    ap.add_argument("--snippets-dir", type=Path, default=None)
    ap.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY)
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--dpi", type=int, default=150, help="Rendering DPI for PDF comparisons")
    args = ap.parse_args()

    build_dir = args.build_dir.resolve()
    summary = _load_json(args.summary_path.resolve()) if args.summary_path else {}
    snippets_dir = (args.snippets_dir or Path(summary.get("snippets_dir", DEFAULT_SNIPPETS))).resolve()
    golden_dir = args.golden_dir.resolve()

    golden_entries = _discover_golden(golden_dir)
    text_eval = evaluate_text(snippets_dir, golden_entries)
    image_eval = evaluate_images(snippets_dir, golden_entries, dpi=args.dpi)

    evaluation = {
        "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "build_dir": str(build_dir),
        "snippets_dir": str(snippets_dir),
        "golden_dir": str(golden_dir),
        "compile": summary.get("compile"),
        "text": text_eval,
        "images": image_eval,
    }

    targets = _resolve_output_paths(summary, build_dir, args.out_json.resolve() if args.out_json else None)
    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(evaluation, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[evaluate_run] wrote {target}")

    if summary:
        summary["metrics"] = {
            "text": text_eval["aggregates"],
            "images": image_eval["aggregates"],
        }
        args.summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[evaluate_run] updated summary → {args.summary_path}")
=======

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

    run_dir: Optional[Path]
    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
    else:
        runs_root = Path("dev") / "runs"
        candidates = sorted(runs_root.glob("*"))
        run_dir = candidates[-1].resolve() if candidates else None

    fixture = _resolve_fixture(fixtures, main_tex, args.fixture_id)

    run_tex_text = _read_text(main_tex)
    ref_tex_text = _read_text(fixture.tex_path)
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
        "tex_comparison": {
            "scores": tex_metrics,
        },
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
>>>>>>> theirs


if __name__ == "__main__":
    main()
