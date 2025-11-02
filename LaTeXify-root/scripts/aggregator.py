#!/usr/bin/env python3
"""
scripts/aggregator.py – Assemble LaTeX documents from snippets and assets.

This script reads a plan.json and a directory of LaTeX snippets and produces a
complete build directory with preamble, sections, main document and optional
assets.  It bundles any referenced assets into build/assets, writes a
manifest for those assets, and optionally runs latexmk to compile the
document.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

# -----------------------------------------------------------------------------
#  Basic configuration and capability mapping
# -----------------------------------------------------------------------------

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent

# Packages that we are willing to include in the preamble dynamically.
CAPABILITY_TO_PACKAGE = {
    "amsmath": r"\\usepackage{amsmath}",
    "amssymb": r"\\usepackage{amssymb}",
    "mathtools": r"\\usepackage{mathtools}",
    "thmtools": r"\\usepackage{thmtools}",
    "graphicx": r"\\usepackage{graphicx}",
    "booktabs": r"\\usepackage{booktabs}",
    "caption": r"\\usepackage{caption}",
    "siunitx": r"\\usepackage{siunitx}",
    "microtype": r"\\usepackage{microtype}",
    "enumitem": r"\\usepackage{enumitem}",
    "geometry": r"\\usepackage{geometry}",
    "hyperref": r"\\usepackage{hyperref}",
    "cleveref": r"\\usepackage{cleveref}",
    "bm": r"\\usepackage{bm}",
    "physics": r"\\usepackage{physics}",
    "cancel": r"\\usepackage{cancel}",
}

BASE_PREAMBLE_LINES = [
    r"\\usepackage{microtype}",
    r"\\usepackage{geometry}",
    r"\\usepackage{hyperref}",
    r"\\hypersetup{hidelinks}",
]
SEED = 42  # determinism

# LiX classes generally want XeLaTeX/LuaLaTeX because they load modern fonts via fontspec
DEFAULT_LIX_CLASSES = {
    "textbook", "thesis", "paper", "novel", "novella", "news", "poem", "ieee_modern"
}


def _log_event(log_path: Path, event: str, **details) -> None:
    """Append a JSON event to the evidence log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {}
    for key, value in details.items():
        if isinstance(value, Path):
            serializable[key] = str(value)
        else:
            serializable[key] = value
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"event": event, **serializable}, ensure_ascii=False) + "\n")


def _read_plan(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _kpsewhich_exists() -> bool:
    return shutil.which("kpsewhich") is not None


def _class_exists(doc_class: str) -> bool:
    if not _kpsewhich_exists():
        return False
    try:
        p = subprocess.run([
            "kpsewhich",
            f"{doc_class}.cls",
        ], check=False, capture_output=True, text=True)
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
        return override  # type: ignore[return-value]
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


@dataclass
class SectionEntry:
    task_id: str
    title: str
    filename: Path
    content: str
    is_placeholder: bool
    is_frontmatter: bool
    source_path: Path | None


@dataclass
class SourceMapEntry:
    task_id: str
    main_start: int
    main_end: int
    section_file: str
    snippet_file: str | None
    snippet_start: int
    snippet_end: int


@dataclass
class AssembleResult:
    doc_class: str
    used_bib: bool
    preamble_lines: List[str]
    sections: List[SectionEntry]


def _resolve_asset_source(asset_path: str, out_dir: Path) -> Path | None:
    """Resolve an asset path relative to out_dir or repository roots."""
    candidate = Path(asset_path)
    search: List[Path] = []
    if candidate.is_absolute():
        search.append(candidate)
    else:
        search.extend([
            out_dir / candidate,
            REPO_ROOT / asset_path,
            REPO_ROOT / "build" / candidate,
        ])
        if candidate.parts and candidate.parts[0] != "assets":
            search.append((out_dir / "assets" / candidate.name))
    for path in search:
        if path.exists():
            return path
    return candidate if candidate.exists() else None


def _bundle_plan_assets(plan: Dict, out_dir: Path, evidence_log: Path) -> List[dict]:
    """Copy assets referenced in the plan into out_dir/assets and write a manifest."""
    assets: List[dict] = []
    assets_dir = out_dir / "assets"
    seen: set[Path] = set()
    for task in plan.get("tasks", []):
        asset_path = task.get("asset_path")
        if not asset_path:
            continue
        source = _resolve_asset_source(asset_path, out_dir)
        if not source or not source.exists():
            raise FileNotFoundError(f"Asset referenced in plan not found: {asset_path}")
        rel = Path(asset_path)
        if rel.is_absolute():
            dest = assets_dir / rel.name
            rel_out = Path("assets") / rel.name
        else:
            dest = (out_dir / rel).resolve()
            try:
                rel_out = dest.relative_to(out_dir)
            except ValueError:
                dest = assets_dir / rel.name
                rel_out = Path("assets") / rel.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest not in seen:
            if source.resolve() != dest.resolve():
                shutil.copy2(source, dest)
            seen.add(dest)
        record = {
            "task_id": task.get("id"),
            "source": str(source),
            "bundled_path": rel_out.as_posix(),
        }
        assets.append(record)
        _log_event(evidence_log, "asset_bundled", **record)
    if assets:
        manifest_path = assets_dir / "manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(assets, ensure_ascii=False, indent=2), encoding="utf-8")
        _log_event(evidence_log, "asset_manifest_written", path=manifest_path, count=len(assets))
    return assets


def _slugify(value: str, fallback: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
    return value or fallback.lower()


def _collect_capabilities(snippets_dir: Path) -> List[str]:
    caps: List[str] = []
    if not snippets_dir.exists():
        return caps
    for meta in snippets_dir.rglob("*.meta.json"):
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
        except Exception:
            continue
        for cap in data.get("capabilities", []) or []:
            if isinstance(cap, str):
                caps.append(cap.strip())
    # Preserve deterministic order
    seen = set()
    ordered: List[str] = []
    for cap in caps:
        if cap in CAPABILITY_TO_PACKAGE and cap not in seen:
            seen.add(cap)
            ordered.append(cap)
    return ordered


def _preamble_from_capabilities(capabilities: Iterable[str]) -> List[str]:
    lines: List[str] = []
    for cap in capabilities:
        pkg_line = CAPABILITY_TO_PACKAGE.get(cap)
        if pkg_line and pkg_line not in lines:
            lines.append(pkg_line)
    # Ensure baseline packages are present exactly once
    out: List[str] = []
    seen = set()
    for base in BASE_PREAMBLE_LINES:
        if base not in seen:
            out.append(base)
            seen.add(base)
    for line in lines:
        if line not in seen:
            out.append(line)
            seen.add(line)
    return out


def _prepare_assets(asset_src: Path | None, out_dir: Path, evidence_log: Path) -> Path | None:
    if asset_src is None:
        return None
    try:
        resolved_src = asset_src.resolve()
    except FileNotFoundError:
        resolved_src = asset_src
    if not resolved_src.exists():
        _log_event(evidence_log, "assets_missing", src=str(asset_src))
        return None
    dest = out_dir / resolved_src.name
    dest_resolved = dest.resolve()
    if dest_resolved == resolved_src:
        file_count = sum(1 for p in dest.rglob("*") if p.is_file()) if dest.exists() else 0
        _log_event(evidence_log, "assets_available", src=str(resolved_src), dest=str(dest_resolved), files=file_count)
        return dest
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(resolved_src, dest)
    file_count = sum(1 for p in dest.rglob("*") if p.is_file())
    _log_event(evidence_log, "assets_copied", src=str(resolved_src), dest=str(dest), files=file_count)
    return dest


def _assemble_document(plan: Dict, snippets_dir: Path, evidence_log: Path) -> AssembleResult:
    tid_to_title = {t["id"]: t["title"] for t in plan.get("tasks", [])}
    doc_class_raw = plan.get("doc_class", "lix_article")
    doc_class, fell_back = _resolve_doc_class(doc_class_raw)
    _log_event(
        evidence_log,
        "aggregate_start",
        plan="plan.json",
        resolved_plan="plan.json",
        snippets=str(snippets_dir),
        out_dir="build",
    )
    if fell_back:
        _log_event(evidence_log, "doc_class_fallback", requested=doc_class_raw, used=doc_class)
    caps = _collect_capabilities(snippets_dir)
    preamble_lines = _preamble_from_capabilities(caps)
    sections: List[SectionEntry] = []
    used_bib = False
    ordered_tasks = sorted(plan.get("tasks", []), key=lambda t: (t.get("order", 0), t["id"]))
    for idx, task in enumerate(ordered_tasks):
        tid = task["id"]
        title = tid_to_title.get(tid, tid)
        anchor = task.get("anchor", "") or ""
        is_frontmatter = anchor.startswith("frontmatter")
        slug_source = task.get("title") or anchor or tid
        filename = Path("sections") / f"{idx:02d}_{_slugify(slug_source, tid)}.tex"
        sp = snippets_dir / f"{tid}.tex"
        if not sp.exists():
            _log_event(evidence_log, "snippet_missing_placeholder_injected", task_id=tid, path=str(sp))
            content = (
                f"% Placeholder for {tid}\n"
                f"\\section{{{title}}}\n"
                f"\\label{{sec:{tid}-placeholder}}\n"
                f"\\todo{{Write content.}}\n"
            )
            is_placeholder = True
            source_path = None
        else:
            _log_event(evidence_log, "snippet_found", task_id=tid, path=str(sp))
            body = sp.read_text(encoding="utf-8").rstrip()
            if (r"\\cite{" in body) or (r"\\addbibresource" in body):
                used_bib = True
            content = body
            is_placeholder = False
            source_path = sp
        sections.append(
            SectionEntry(
                task_id=tid,
                title=title,
                filename=filename,
                content=content,
                is_placeholder=is_placeholder,
                is_frontmatter=is_frontmatter,
                source_path=source_path,
            )
        )
    _log_event(evidence_log, "bib_detection", use_biblatex=used_bib)
    return AssembleResult(
        doc_class=doc_class,
        used_bib=used_bib,
        preamble_lines=preamble_lines,
        sections=sections,
    )


def _write_preamble(lines: List[str], out_dir: Path, evidence_log: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    preamble = out_dir / "preamble.tex"
    header = [
        "% Auto-generated by scripts.aggregator",
        "% This file is intentionally a fragment to be \input{} after \documentclass",
        "",
    ]
    text = "\n".join(header + lines) + "\n"
    preamble.write_text(text, encoding="utf-8")
    _log_event(evidence_log, "preamble_written", path=preamble, bytes=preamble.stat().st_size)
    return preamble


def _write_sections(sections: List[SectionEntry], out_dir: Path, evidence_log: Path) -> Dict[str, Dict[str, int | str | None]]:
    section_map: Dict[str, Dict[str, int | str | None]] = {}
    for entry in sections:
        target = out_dir / entry.filename
        target.parent.mkdir(parents=True, exist_ok=True)
        text = entry.content.rstrip() + "\n"
        target.write_text(text, encoding="utf-8")
        _log_event(
            evidence_log,
            "section_written",
            task_id=entry.task_id,
            path=target,
            placeholder=entry.is_placeholder,
        )
        if entry.source_path and not entry.is_placeholder:
            rel_section = entry.filename.as_posix()
            rel_snippet = entry.source_path.as_posix()
            section_map[rel_section] = {
                "task_id": entry.task_id,
                "snippet": rel_snippet,
                "line_count": len(text.splitlines()),
            }
    return section_map


def _build_main(doc_class: str, sections: List[SectionEntry], used_bib: bool) -> Tuple[str, List[SourceMapEntry]]:
    mapping: List[SourceMapEntry] = []
    frontmatter: List[Tuple[List[str], SourceMapEntry]] = []
    body: List[Tuple[List[str], SourceMapEntry]] = []
    for entry in sections:
        block = [f"% --- {entry.task_id} ---", f"\\input{{{entry.filename.as_posix()}}}", ""]
        snippet_lines = entry.content.count("\n") + 1 if entry.content else 0
        map_entry = SourceMapEntry(
            task_id=entry.task_id,
            main_start=0,
            main_end=0,
            section_file=entry.filename.as_posix(),
            snippet_file=entry.source_path.as_posix() if entry.source_path else None,
            snippet_start=1,
            snippet_end=snippet_lines,
        )
        if entry.is_frontmatter:
            frontmatter.append((block, map_entry))
        else:
            body.append((block, map_entry))
    lines: List[str] = [
        f"\\documentclass{{{doc_class}}}",
        "\\input{preamble.tex}",
        "",
        "\\begin{document}",
    ]
    current_line = len(lines)
    for block, map_entry in frontmatter:
        block_start = current_line + 1
        lines.extend(block)
        current_line += len(block)
        map_entry.main_start = block_start
        map_entry.main_end = current_line
        mapping.append(map_entry)
    if frontmatter:
        lines.append("% aggregator: \\maketitle after frontmatter")
        lines.append("\\maketitle")
        lines.append("")
        current_line = len(lines)
    for block, map_entry in body:
        block_start = current_line + 1
        lines.extend(block)
        current_line += len(block)
        map_entry.main_start = block_start
        map_entry.main_end = current_line
        mapping.append(map_entry)
    if used_bib:
        lines.append("\\printbibliography")
        lines.append("")
        current_line = len(lines)
    lines.append("\\end{document}")
    return "\n".join(lines) + "\n", mapping


def _write_main(doc_class: str, sections: List[SectionEntry], used_bib: bool,
                out_dir: Path, evidence_log: Path) -> Tuple[Path, List[SourceMapEntry]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    main_tex = out_dir / "main.tex"
    tex, mapping = _build_main(doc_class, sections, used_bib)
    main_tex.write_text(tex, encoding="utf-8")
    _log_event(evidence_log, "main_written", path=main_tex, bytes=main_tex.stat().st_size)
    return main_tex, mapping


def _write_source_map(out_dir: Path,
                      mapping: List[SourceMapEntry],
                      section_map: Dict[str, Dict[str, int | str | None]],
                      evidence_log: Path) -> Path:
    source_map_path = out_dir / "main.sourcemap.json"
    payload = {
        "version": 1,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "entries": [
            {
                "task_id": m.task_id,
                "main": {"start": m.main_start, "end": m.main_end},
                "section_file": m.section_file,
                "snippet_file": m.snippet_file,
                "snippet": {"start": m.snippet_start, "end": m.snippet_end},
            }
            for m in mapping
        ],
        "sections": section_map,
    }
    source_map_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _log_event(evidence_log, "source_map_written", path=source_map_path, entries=len(mapping))
    return source_map_path


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
                   assets_dir: str | None = None,
                   engine_override: str | None = None, verbose: bool = False) -> Dict:
    """Execute the aggregation pipeline and optionally compile the document."""
    plan_file = Path(plan_path)
    if not plan_file.exists():
        raise SystemExit(f"Plan not found: {plan_file}")
    plan = _read_plan(plan_file)
    out_dir_path = Path(out_dir)
    evidence_log = out_dir_path / "aggregate.log.jsonl"
    out_dir_path.mkdir(parents=True, exist_ok=True)
    # Copy assets from external directory if provided
    assets_dst = _prepare_assets(Path(assets_dir) if assets_dir else None, out_dir_path, evidence_log)
    # Bundle any assets referenced directly in the plan into out_dir/assets
    bundled_assets = _bundle_plan_assets(plan, out_dir_path, evidence_log)
    # Assemble the document structure
    res = _assemble_document(plan, Path(snippets_dir), evidence_log)
    _write_preamble(res.preamble_lines, out_dir_path, evidence_log)
    section_map = _write_sections(res.sections, out_dir_path, evidence_log)
    main_tex, mapping = _write_main(res.doc_class, res.sections, res.used_bib, out_dir_path, evidence_log)
    source_map_path = _write_source_map(out_dir_path, mapping, section_map, evidence_log)
    # Determine compile engine and optionally run latexmk
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
            subprocess.run([
                "latexmk", "-C", main_tex.name
            ], cwd=str(main_tex.parent), capture_output=True, text=True)
            compile_ok, stdout, stderr = _run_latexmk(main_tex, engine, verbose)
    _log_event(
        evidence_log,
        "aggregate_done",
        compile_attempted=compile_attempted,
        compile_ok=compile_ok,
        used_bib=res.used_bib,
        out=out_dir,
        engine=engine,
    )
    asset_manifest_path = out_dir_path / "assets" / "manifest.json"
    # Print summary to stdout
    print(
        json.dumps(
            {
                "main_tex": str(main_tex),
                "doc_class": res.doc_class,
                "engine": engine,
                "compile_attempted": compile_attempted,
                "compile_ok": compile_ok,
                "stdout": stdout,
                "stderr": stderr,
                "source_map": str(source_map_path),
                "assets_dir": str(assets_dst) if assets_dst else None,
                "asset_manifest": str(asset_manifest_path) if asset_manifest_path.exists() else None,
                "bundled_assets": bundled_assets,
            },
            ensure_ascii=False,
        )
    )
    # Return structured result
    return {
        "main_tex": str(main_tex),
        "doc_class": res.doc_class,
        "engine": engine,
        "compile_attempted": compile_attempted,
        "compile_ok": compile_ok,
        "stdout": stdout,
        "stderr": stderr,
        "source_map": str(source_map_path),
        "assets_dir": str(assets_dst) if assets_dst else None,
        "asset_manifest": str(asset_manifest_path) if asset_manifest_path.exists() else None,
        "bundled_assets": bundled_assets,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--snippets_dir", required=True)
    ap.add_argument("--out_dir", default="build")
    ap.add_argument(
        "--assets_dir", type=str, default=None, help="Optional directory of assets to copy alongside build outputs"
    )
    ap.add_argument("--no_compile", action="store_true")
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--engine", choices=["pdflatex", "xelatex", "lualatex"])
    ap.add_argument("--verbose", action="store_true", help="Show latexmk output (no -quiet)")
    args = ap.parse_args()
    run_aggregator(
        args.plan,
        args.snippets_dir,
        args.out_dir,
        args.no_compile,
        args.simulate,
        assets_dir=args.assets_dir,
        engine_override=args.engine,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()