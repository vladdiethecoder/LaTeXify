#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

# Preamble allowlist (unchanged)
ALLOWLIST_PACKAGES = [
    "amsmath", "amssymb", "amsthm", "mathtools", "thmtools",
    "graphicx", "booktabs", "caption", "siunitx", "microtype",
    "enumitem", "geometry", "hyperref", "cleveref",
]
SEED = 42  # determinism

# LiX classes generally want XeLaTeX/LuaLaTeX because they load modern fonts via fontspec
DEFAULT_LIX_CLASSES = {
    "textbook", "thesis", "paper", "novel", "novella", "news", "poem", "ieee_modern"
}

def _log_event(log_path: Path, event: str, **details) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"event": event, **details}, ensure_ascii=False) + "\n")

def _read_plan(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))

def _kpsewhich_exists() -> bool:
    return shutil.which("kpsewhich") is not None

def _class_exists(doc_class: str) -> bool:
    if not _kpsewhich_exists():
        return False
    try:
        p = subprocess.run(["kpsewhich", f"{doc_class}.cls"],
                           check=False, capture_output=True, text=True)
        return p.returncode == 0 and p.stdout.strip() != ""
    except Exception:
        return False

def _resolve_doc_class(raw_class: str) -> Tuple[str, bool]:
    force = os.environ.get("AGG_FORCE_DOCCLASS_FALLBACK") == "1"
    if force and raw_class.startswith("lix_"):
        return "scrartcl", True
    if _class_exists(raw_class):
        return raw_class, False
    return "scrartcl", True

# ---------- NEW: engine selection ----------
def choose_engine(doc_class: str, override: str | None = None) -> str:
    """
    Decide latex engine: 'pdflatex' | 'xelatex' | 'lualatex'.
    Precedence:
      1) env AGG_LATEX_ENGINE if in {pdflatex,xelatex,lualatex}
      2) --engine CLI override (passed here by caller)
      3) If doc_class is LiX-like -> xelatex
      4) default -> pdflatex
    """
    env_engine = os.environ.get("AGG_LATEX_ENGINE", "").strip().lower()
    if env_engine in {"pdflatex", "xelatex", "lualatex"}:
        return env_engine
    if override in {"pdflatex", "xelatex", "lualatex"}:
        return override
    if doc_class in DEFAULT_LIX_CLASSES:
        return "xelatex"
    return "pdflatex"

def latexmk_flags_for_engine(engine: str) -> List[str]:
    # From latexmk manual: -pdf (pdflatex), -pdfxe (xelatex), -pdflua (lualatex)
    # We'll also add -g to force a make if previous run errored/out-of-date.
    if engine == "xelatex":
        return ["-pdfxe", "-g"]
    if engine == "lualatex":
        return ["-pdflua", "-g"]
    return ["-pdf", "-g"]

# ------------------------------------------

@dataclass
class AssembleResult:
    tex: str
    used_bib: bool
    doc_class: str

def _assemble_document(plan: Dict, snippets_dir: Path, evidence_log: Path) -> AssembleResult:
    tid_to_title = {t["id"]: t["title"] for t in plan.get("tasks", [])}
    doc_class_raw = plan.get("doc_class", "lix_article")
    doc_class, fell_back = _resolve_doc_class(doc_class_raw)

    _log_event(evidence_log, "aggregate_start",
               plan="plan.json", resolved_plan="plan.json",
               snippets=str(snippets_dir), out_dir="build")
    if fell_back:
        _log_event(evidence_log, "doc_class_fallback", requested=doc_class_raw, used=doc_class)

    preamble = [f"\\documentclass{{{doc_class}}}"]
    for pkg in ALLOWLIST_PACKAGES:
        preamble.append(f"\\usepackage{{{pkg}}}")
    preamble.append("\\hypersetup{hidelinks}\n")

    body_lines: List[str] = []
    body_lines.append("\\begin{document}")
    body_lines.append("% aggregator: \\maketitle after frontmatter")
    body_lines.append("\\maketitle\n")

    used_bib = False
    for task in sorted(plan.get("tasks", []), key=lambda t: (t.get("order", 0), t["id"])):
        tid = task["id"]
        title = tid_to_title.get(tid, tid)
        sp = snippets_dir / f"{tid}.tex"
        if not sp.exists():
            _log_event(evidence_log, "snippet_missing_placeholder_injected",
                       task_id=tid, path=str(sp))
            placeholder = (
                f"% Placeholder for {tid}\n"
                f"\\section{{{title}}}\n"
                f"\\label{{sec:{tid}-placeholder}}\n"
                f"\\todo{{Write content.}}\n\n"
            )
            body_lines.append(f"% --- {tid} ---\n{placeholder}")
        else:
            _log_event(evidence_log, "snippet_found", task_id=tid, path=str(sp))
            body = sp.read_text(encoding="utf-8")
            if (r"\cite{" in body) or (r"\addbibresource" in body):
                used_bib = True
            body_lines.append(f"% --- {tid} ---\n{body.strip()}\n\n")

    _log_event(evidence_log, "bib_detection", use_biblatex=used_bib)
    body_lines.append("\\end{document}\n")

    tex = "\n".join(preamble) + "\n" + "\n".join(body_lines)
    return AssembleResult(tex=tex, used_bib=used_bib, doc_class=doc_class)

def _write_main(tex: str, out_dir: Path, evidence_log: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    main_tex = out_dir / "main.tex"
    main_tex.write_text(tex, encoding="utf-8")
    _log_event(evidence_log, "main_written", path=str(main_tex), bytes=main_tex.stat().st_size)
    return main_tex

def _run_latexmk(main_tex: Path, engine: str, verbose: bool) -> Tuple[bool, str, str]:
    latexmk = shutil.which("latexmk")
    if not latexmk:
        return False, "", "latexmk not found"

    flags = latexmk_flags_for_engine(engine)
    cmd = [latexmk] + flags + ["-interaction=nonstopmode"]
    if not verbose:
        cmd.append("-quiet")
    cmd.append(main_tex.name)

    proc = subprocess.run(cmd, cwd=str(main_tex.parent), capture_output=True, text=True)
    ok = proc.returncode == 0
    return ok, proc.stdout, proc.stderr

def run_aggregator(plan_path: str, snippets_dir: str, out_dir: str,
                   no_compile: bool, simulate: bool,
                   engine_override: str | None = None, verbose: bool = False) -> Dict:
    plan = _read_plan(Path(plan_path))
    evidence_log = Path("build") / "aggregate.log.jsonl"
    res = _assemble_document(plan, Path(snippets_dir), evidence_log)
    main_tex = _write_main(res.tex, Path(out_dir), evidence_log)

    compile_attempted = False
    compile_ok = False
    stdout = ""
    stderr = ""
    engine = choose_engine(res.doc_class, override=engine_override)
    _log_event(evidence_log, "engine_selected", engine=engine, doc_class=res.doc_class)

    if not no_compile and not simulate:
        compile_attempted = True
        compile_ok, stdout, stderr = _run_latexmk(main_tex, engine, verbose)
        if not compile_ok:
            # Last resort: full cleanup then forced build (latexmk -C, then run again)
            # Per latexmk manual: -C cleans all regeneratable files; -g forces make.
            subprocess.run(["latexmk", "-C", main_tex.name],
                           cwd=str(main_tex.parent), capture_output=True, text=True)
            compile_ok, stdout, stderr = _run_latexmk(main_tex, engine, verbose)

    _log_event(evidence_log, "aggregate_done",
               compile_attempted=compile_attempted, compile_ok=compile_ok,
               used_bib=res.used_bib, out=out_dir, engine=engine)

    print(json.dumps({
        "main_tex": str(main_tex),
        "doc_class": res.doc_class,
        "engine": engine,
        "compile_attempted": compile_attempted,
        "compile_ok": compile_ok,
        "stdout": stdout,
        "stderr": stderr,
    }, ensure_ascii=False))
    return {
        "main_tex": str(main_tex),
        "doc_class": res.doc_class,
        "engine": engine,
        "compile_attempted": compile_attempted,
        "compile_ok": compile_ok,
        "stdout": stdout,
        "stderr": stderr,
    }

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--snippets_dir", required=True)
    ap.add_argument("--out_dir", default="build")
    ap.add_argument("--no_compile", action="store_true")
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--engine", choices=["pdflatex", "xelatex", "lualatex"])
    ap.add_argument("--verbose", action="store_true", help="Show latexmk output (no -quiet)")
    args = ap.parse_args()
    run_aggregator(args.plan, args.snippets_dir, args.out_dir,
                   args.no_compile, args.simulate,
                   engine_override=args.engine, verbose=args.verbose)

if __name__ == "__main__":
    main()
