#!/usr/bin/env python3
"""CI helper that enforces the visual regression diff gate."""

from __future__ import annotations

import argparse
from importlib import util
from pathlib import Path
import sys
from typing import Dict

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Run directory (build/runs/<run_id>)")
    parser.add_argument(
        "--tex",
        type=Path,
        default=None,
        help="LaTeX file used to locate the compiled PDF (default: <run-dir>/main.tex)",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="Directory of reference page renders (default: <run-dir>/artifacts/page_rasters)",
    )
    parser.add_argument("--threshold", type=float, default=None, help="Diff threshold override.")
    parser.add_argument("--page-limit", type=int, default=None, help="Maximum number of pages to compare.")
    parser.add_argument("--dpi", type=int, default=None, help="Rasterization DPI override.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path (default: <run-dir>/reports/visual_regression_ci.json)",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Exit successfully when reference images are unavailable.",
    )
    return parser.parse_args()


def _resolve_paths(args: argparse.Namespace) -> Dict[str, Path]:
    run_dir = args.run_dir.resolve()
    if not run_dir.exists():
        raise SystemExit(f"Run directory not found: {run_dir}")
    tex_path = args.tex or run_dir / "main.tex"
    reference_dir = args.reference or run_dir / "artifacts" / "page_rasters"
    output_path = args.output or run_dir / "reports" / "visual_regression_ci.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return {"run_dir": run_dir, "tex": tex_path, "reference": reference_dir, "output": output_path}


def main() -> None:
    args = parse_args()
    spec = util.spec_from_file_location(
        "visual_regression_module", REPO_ROOT / "release" / "pipeline" / "visual_regression.py"
    )
    if spec is None or spec.loader is None:
        raise SystemExit("Unable to load visual_regression module")
    module = util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    run_visual_regression = getattr(module, "run_visual_regression")
    paths = _resolve_paths(args)
    tex_path = paths["tex"]
    reference_dir = paths["reference"]
    output_path = paths["output"]
    if not tex_path.exists():
        raise SystemExit(f"LaTeX file not found: {tex_path}")
    report = run_visual_regression(
        tex_path,
        reference_dir,
        output_path,
        page_limit=args.page_limit,
        threshold=args.threshold,
        dpi=args.dpi,
    )
    if not report.get("available"):
        message = report.get("reason", "visual regression unavailable")
        print(f"[visual-ci] skipped: {message}")
        if args.allow_missing:
            raise SystemExit(0)
        raise SystemExit(2)
    flagged = int(report.get("flagged_pages", 0))
    evaluated = int(report.get("pages_evaluated", 0))
    if flagged:
        print(f"[visual-ci] blocked: {flagged} of {evaluated} page(s) exceed the diff threshold.")
        raise SystemExit(1)
    print(f"[visual-ci] clean: {evaluated} page(s) checked; diff threshold passed.")


if __name__ == "__main__":
    main()
