#!/usr/bin/env python3
"""Builds SFT and preference datasets from gold-standard LaTeX files."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, List, Tuple

import random

LATEX_COMMAND_RE = re.compile(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{[^}]*\})?")


def latex_to_text(latex: str) -> str:
    text = re.sub(r"%.*", " ", latex)
    text = re.sub(r"\\begin\{[^\}]+\}|\\end\{[^\}]+\}", " ", text)
    text = re.sub(LATEX_COMMAND_RE, " ", text)
    text = re.sub(r"[\{\}]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def mutate_latex(latex: str) -> str:
    mutated = latex.replace("\\toprule", "\\hline").replace("\\midrule", "\\hline")
    mutated = mutated.replace("\\begin{align}", "\\begin{eqnarray}")
    mutated = mutated.replace("\\end{align}", "\\end{eqnarray}")
    return mutated


def collect_pairs(source_dir: Path) -> Iterable[Tuple[str, str]]:
    for tex_file in source_dir.rglob("*.tex"):
        latex = tex_file.read_text(encoding="utf-8", errors="ignore")
        text = latex_to_text(latex)
        if len(text.split()) < 32:
            continue
        yield text, latex


def build_datasets(source_dir: Path, output_dir: Path, limit: int | None = None) -> None:
    pairs = list(collect_pairs(source_dir))
    if not pairs:
        raise SystemExit(f"No LaTeX files found under {source_dir}")
    random.shuffle(pairs)
    if limit:
        pairs = pairs[:limit]
    sft_path = output_dir / "sft_dataset.jsonl"
    pref_path = output_dir / "preference_dataset.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    with sft_path.open("w", encoding="utf-8") as sft_handle, pref_path.open("w", encoding="utf-8") as pref_handle:
        for idx, (prompt, response) in enumerate(pairs):
            sft_handle.write(json.dumps({"prompt": prompt, "response": response}, ensure_ascii=False) + "\n")
            rejected = mutate_latex(response)
            pref_handle.write(
                json.dumps(
                    {
                        "prompt": prompt,
                        "chosen": response,
                        "rejected": rejected,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"SFT dataset written to {sft_path}")
    print(f"Preference dataset written to {pref_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SFT + preference datasets from reference_tex.")
    parser.add_argument("--source", type=Path, default=Path("reference_tex"), help="Directory of gold .tex files")
    parser.add_argument("--output", type=Path, default=Path("datasets"), help="Output directory for JSONL files")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of documents")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_datasets(args.source, args.output, args.limit)


if __name__ == "__main__":
    main()
