#!/usr/bin/env python3
"""
End-to-end test (OCR ensemble → chunks → FAISS → Planner → Synthesis → Aggregate)

Updates:
- Prefer role-aware chunking via scripts/role_chunker.py when blocks_refined.jsonl exists
- Fall back to build_chunks.py (rolling) with a robust prefer order
- --clobber (default True) to overwrite previous runs & indexes

Usage:
python scripts/test_e2e.py \
  --assignment_pdf dev/inputs/...Assignment.pdf \
  --assessment_pdf dev/inputs/...Assessment.pdf \
  --rubric_pdf     dev/inputs/...Rubric.pdf \
  --user_pdf       dev/inputs/...USER.pdf \
  --only_backends "qwen2-vl-ocr-2b,nanonets-ocr-s,nanonets-ocr2-3b" \
  --title "Assignment 1" --author "Your Name" --course "CSC-XXX" --date "2025-10-29"
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS = REPO_ROOT / "dev" / "runs"
INDEXES = REPO_ROOT / "indexes"


def _env_with_repo_path() -> dict:
    env = os.environ.copy()
    rp = str(REPO_ROOT)
    if env.get("PYTHONPATH"):
        if rp not in env["PYTHONPATH"].split(os.pathsep):
            env["PYTHONPATH"] = env["PYTHONPATH"] + os.pathsep + rp
    else:
        env["PYTHONPATH"] = rp
    return env


def run(cmd: list[str], cwd: Optional[Path] = None) -> int:
    cwd = cwd or REPO_ROOT
    print("$", " ".join(cmd))
    res = subprocess.run(cmd, cwd=str(cwd), check=False, env=_env_with_repo_path())
    if res.returncode != 0:
        print(f"[warn] exited {res.returncode}: {' '.join(cmd)}", file=sys.stderr)
    return res.returncode


def _rm_rf(p: Path) -> None:
    if p.exists():
        shutil.rmtree(p, ignore_errors=True)


def ensure_ocr(role: str, pdf: Path, run_dir: Path, only: str = "") -> bool:
    outputs = run_dir / "outputs"
    if outputs.exists() and any(outputs.rglob("page-*.md")):
        print(f"[{role}] OCR outputs already present.")
        return True
    cmd = ["python", "scripts/ocr_ensemble_driver.py", "--pdf", str(pdf), "--run_dir", str(run_dir)]
    if only:
        cmd += ["--only", only]
    run(cmd)
    ok = outputs.exists() and any(outputs.rglob("page-*.md"))
    print(f"[{role}] OCR outputs ok: {ok}")
    return ok


def fallback_ingest(role: str, pdf: Path, run_dir: Path) -> bool:
    rc = run(["python", "scripts/ingest_pdf.py", "--pdf", str(pdf), "--run_dir", str(run_dir)])
    ok = (run_dir / "outputs").exists() and any((run_dir / "outputs").rglob("page-*.md"))
    print(f"[{role}] fallback outputs ok: {ok}")
    return rc == 0 and ok


def _role_chunk(role: str, run_dir: Path, pdf: Path) -> Path:
    """Prefer role_chunker if blocks are present, else build_chunks."""
    chunks = run_dir / "chunks.jsonl"
    blocks_file = run_dir / "blocks_refined.jsonl"
    if blocks_file.exists():
        # role-aware chunking
        params = {
            "user":        dict(max_chars=1100, overlap=150, min_par_len=40),
            "assessment":  dict(max_chars=2400, overlap=200, min_par_len=20),  # virtually atomic
            "rubric":      dict(max_chars=1100, overlap=150, min_par_len=20),
            "assignment":  dict(max_chars=900,  overlap=140, min_par_len=40),
        }[role]
        cmd = [
            "python", "scripts/role_chunker.py",
            "--run_dir", str(run_dir),
            "--pdf", str(pdf),
            "--role", role,
            "--max_chars", str(params["max_chars"]),
            "--overlap", str(params["overlap"]),
            "--min_par_len", str(params["min_par_len"]),
        ]
        run(cmd)
    else:
        # fallback to durable rolling chunker
        prefer = "qwen2-vl-ocr-2b,nanonets-ocr-2b,nanonets-ocr2-3b,nanonets-ocr-s"
        cmd = [
            "python", "scripts/build_chunks.py",
            "--run_dir", str(run_dir),
            "--pdf", str(pdf),
            "--max_chars", "1100" if role != "assignment" else "900",
            "--overlap", "150" if role != "assignment" else "140",
            "--min_par_len", "40",
            "--prefer", prefer,
        ]
        run(cmd)
        if not chunks.exists() or chunks.stat().st_size == 0:
            print(f"[{role}] chunks missing/empty. Retrying with Qwen-only prefer…")
            cmd[-1] = "qwen2-vl-ocr-2b"
            run(cmd)
    if not chunks.exists() or chunks.stat().st_size == 0:
        raise SystemExit(f"[{role}] chunks.jsonl not created in {run_dir}")
    return chunks


def build_faiss(role: str, run_dir: Path, out_dir: Path) -> Tuple[Path, Path]:
    idx, meta = run_dir / "faiss.index", run_dir / "faiss.meta.json"
    if not (idx.exists() and meta.exists()):
        run(["python", "scripts/build_index.py", "--run_dir", str(run_dir)])
        if not idx.exists() or not meta.exists():
            raise SystemExit(f"[{role}] FAISS not written in {run_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in ("faiss.index", "faiss.meta.json"):
        dst = out_dir / f
        if dst.exists():
            dst.unlink()
    shutil.copy2(str(idx), str(out_dir / "faiss.index"))
    shutil.copy2(str(meta), str(out_dir / "faiss.meta.json"))
    print(f"[{role}] -> {out_dir/'faiss.index'} ; {out_dir/'faiss.meta.json'}")
    return out_dir / "faiss.index", out_dir / "faiss.meta.json"


def main() -> None:
    ap = argparse.ArgumentParser(description="End-to-end test (OCR → chunks → FAISS → plan → synth → aggregate)")
    ap.add_argument("--assignment_pdf", type=Path, required=True)
    ap.add_argument("--assessment_pdf", type=Path, required=True)
    ap.add_argument("--rubric_pdf", type=Path, required=True)
    ap.add_argument("--user_pdf", type=Path, required=True)
    ap.add_argument("--only_backends", type=str, default="")
    ap.add_argument("--fallback_if_empty", action="store_true")
    ap.add_argument("--title", type=str, default="Assignment 1")
    ap.add_argument("--author", type=str, default="Test User")
    ap.add_argument("--course", type=str, default="CSC-XXX")
    ap.add_argument("--date", type=str, default="")
    ap.add_argument("--clobber", action="store_true", default=True)
    args = ap.parse_args()

    # Phase 0 KB
    run([
        "python", "scripts/build_latex_kb.py",
        "--seed_jsonl", "tasks/seed_latex_kb_min.jsonl",
        "--out_jsonl", "data/latex_docs.jsonl",
        "--out_index", "indexes/latex_docs.index",
        "--out_meta", "indexes/latex_docs.meta.json",
    ])

    roles = {
        "assignment": args.assignment_pdf,
        "assessment": args.assessment_pdf,
        "rubric": args.rubric_pdf,
        "user": args.user_pdf,
    }
    run_dirs = {r: (RUNS / f"{r}_e2e") for r in roles}

    if args.clobber:
        for r in roles:
            _rm_rf(run_dirs[r])
            _rm_rf(INDEXES / r)

    # OCR (or fallback raster ingestion)
    for r, pdf in roles.items():
        ok = ensure_ocr(r, pdf, run_dirs[r], only=args.only_backends)
        if not ok and args.fallback_if_empty:
            print(f"[{r}] trying fallback ingestion …")
            ok = fallback_ingest(r, pdf, run_dirs[r])
        if not ok:
            raise SystemExit(f"[{r}] No page-*.md outputs. Fix OCR backend or re-run with --fallback_if_empty.")

    # Role-aware chunking
    for r, pdf in roles.items():
        _role_chunk(r, run_dirs[r], pdf)

    # FAISS indexes
    for r in roles:
        build_faiss(r, run_dirs[r], INDEXES / r)

    # Planner → Synth → Aggregate
    run([
        "python", "scripts/planner_scaffold.py",
        "--assignment", "indexes/assignment",
        "--assessment", "indexes/assessment",
        "--rubric", "indexes/rubric",
        "--user", "indexes/user",
        "--doc_class", "lix_textbook",
        "--title", args.title, "--author", args.author, "--course", args.course, "--date", args.date,
        "--out", "plan.json",
    ])

    run([
        "python", "scripts/synth_latex.py",
        "--plan", "plan.json",
        "--assignment", "indexes/assignment",
        "--assessment", "indexes/assessment",
        "--rubric", "indexes/rubric",
        "--user", "indexes/user",
        "--latex_kb", "indexes/latex_docs.index",
        "--out_dir", "snippets",
    ])

    run([
        "python", "scripts/aggregate_tex.py",
        "--plan", "plan.json",
        "--snippets_dir", "snippets",
        "--out_dir", "build",
    ])

    rep = Path("build/report.json")
    print("\n[MSC] Completion report:" if rep.exists() else "\n[MSC] Report missing — check build/")
    if rep.exists():
        print(rep.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
