# latexify/pipeline/phase2_run_task.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

# Local modules (installed by you in-repo)
from latexify.pipeline.retrieval_bundle import build_context_bundle  # type: ignore
from latexify.pipeline.synth_latex import synthesize_snippet  # type: ignore


def _load_plan(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser(
        description="Orchestrate Phase-2 for a single task: Retrieval → Synthesis → snippet"
    )
    ap.add_argument("--plan", type=Path, default=Path("plan.json"))
    ap.add_argument("--task_id", type=str, required=True)
    ap.add_argument("--assignment", type=Path, default=Path("kb/latex"))
    ap.add_argument("--assessment", type=Path, default=Path("kb/latex"))
    ap.add_argument("--rubric", type=Path, default=Path("kb/latex"))
    ap.add_argument("--user", type=Path, default=Path("kb/latex"))

    ap.add_argument("--k_user", type=int, default=6)
    ap.add_argument("--k_rubric", type=int, default=6)
    ap.add_argument("--k_assignment", type=int, default=6)
    ap.add_argument("--k_assessment", type=int, default=6)

    ap.add_argument("--snippets_dir", type=Path, default=Path("snippets"))
    ap.add_argument("--bundle_out", type=Path, default=None, help="Optional path to save bundle JSON")
    args = ap.parse_args()

    plan = _load_plan(args.plan)
    tasks = {t["id"]: t for t in plan.get("tasks", [])}
    task = tasks.get(args.task_id)
    if not task:
        raise SystemExit(f"Task id {args.task_id} not found in plan {args.plan}")

    indices = {
        "assignment": args.assignment,
        "assessment": args.assessment,
        "rubric": args.rubric,
        "user": args.user,
    }

    # Build retrieval bundle (graceful even if indexes are missing)
    bundle = build_context_bundle(
        task=task,
        indices=indices,
        k_user=args.k_user,
        k_rubric=args.k_rubric,
        k_assignment=args.k_assignment,
        k_assessment=args.k_assessment,
        evidence_dir=Path("evidence"),
    )

    # Optionally persist the bundle (operator inspectability)
    if args.bundle_out:
        js = {
            "task_id": bundle.task_id,
            "question": bundle.question,
            "rubric": [c.__dict__ for c in bundle.rubric],
            "assignment_rules": [c.__dict__ for c in bundle.assignment_rules],
            "assessment": [c.__dict__ for c in bundle.assessment],
            "user_answer": {
                "chunks": [c.__dict__ for c in bundle.user_answer.chunks],
                "flags": bundle.user_answer.flags,
            },
        }
        args.bundle_out.parent.mkdir(parents=True, exist_ok=True)
        args.bundle_out.write_text(json.dumps(js, ensure_ascii=False, indent=2), encoding="utf-8")

    # Synthesize deterministic LiX snippet
    snippet = synthesize_snippet(json.loads(json.dumps({
        "task_id": bundle.task_id,
        "question": bundle.question,
        "rubric": [c.__dict__ for c in bundle.rubric],
        "assignment_rules": [c.__dict__ for c in bundle.assignment_rules],
        "assessment": [c.__dict__ for c in bundle.assessment],
        "user_answer": {
            "chunks": [c.__dict__ for c in bundle.user_answer.chunks],
            "flags": bundle.user_answer.flags,
        },
        "task_meta": bundle.task_meta,
    })))

    # Write snippet
    args.snippets_dir.mkdir(parents=True, exist_ok=True)
    out_tex = args.snippets_dir / f"{bundle.task_id}.tex"
    out_tex.write_text(snippet, encoding="utf-8")

    # Minimal JSONL operator trace
    ev = Path("evidence") / f"{bundle.task_id}.orchestrate.jsonl"
    ev.parent.mkdir(parents=True, exist_ok=True)
    ev.write_text(json.dumps({"event": "phase2_orchestrate_done", "task_id": bundle.task_id, "snippet": str(out_tex)}, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps({"snippet": str(out_tex)}, indent=2))


if __name__ == "__main__":
    main()
