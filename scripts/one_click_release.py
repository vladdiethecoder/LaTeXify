#!/usr/bin/env python3
"""Single-command helper to register layout datasets, ensure LayoutLM assets, and run the latexify.pipeline."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = REPO_ROOT / "training_data"
RELEASE_VENV_PY = REPO_ROOT / "src" / "latexify" / ".venv" / "bin" / "python"
DEFAULT_LAYOUT_SLUGS: Sequence[str] = tuple(
    sorted(
        {
            "cdsse",
            "diachronic",
            "docbank",
            "doclaynet",
            "funsd",
            "grotoap2",
            "m6doc",
            "newspaper-navigator",
            "publaynet",
            "rvlcdip",
            "tqa",
        }
    )
)
SPLITS = ("train", "val", "test")
SOURCE_PATTERNS = (
    "{split}.jsonl",
    "*.source.jsonl",
    "*.real.jsonl",
    "*.jsonl",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", required=True, help="PDF fed into run_release.py.")
    parser.add_argument(
        "--layout-model",
        default="training_runs/layoutlmv3-doclaynet",
        help="Fine-tuned LayoutLM checkpoint directory.",
    )
    parser.add_argument(
        "--layout-base",
        default="microsoft/layoutlmv3-base",
        help="Fallback processor source if the fine-tuned run is missing preprocessor assets.",
    )
    parser.add_argument(
        "--slugs",
        help="Comma-separated slug list. Defaults to all known layout datasets.",
    )
    parser.add_argument(
        "--no-register",
        action="store_true",
        help="Skip auto-registration of layout splits.",
    )
    parser.add_argument(
        "--force-register",
        action="store_true",
        help="Overwrite data.jsonl even when it already contains non-placeholder samples.",
    )
    parser.add_argument(
        "--skip-release",
        action="store_true",
        help="Perform registration/asset checks only (do not run run_release.py).",
    )
    parser.add_argument(
        "--layout-device",
        default="cpu",
        help="Value assigned to LATEXIFY_LAYOUTLM_DEVICE (default: cpu).",
    )
    parser.add_argument(
        "--clip-device",
        default="cpu",
        help="Value assigned to LATEXIFY_CLIP_DEVICE (default: cpu).",
    )
    parser.add_argument(
        "--ocr-max-heavy",
        default="1",
        help="Value assigned to LATEXIFY_OCR_MAX_HEAVY (default: 1).",
    )
    parser.add_argument(
        "--extra-env",
        action="append",
        help="Additional KEY=VALUE entries exported before running the latexify.pipeline.",
    )
    return parser.parse_args()


def resolve_slugs(raw: str | None) -> List[str]:
    if not raw:
        return list(DEFAULT_LAYOUT_SLUGS)
    if raw.strip().lower() in {"all", "auto"}:
        return list(DEFAULT_LAYOUT_SLUGS)
    return [slug.strip() for slug in raw.split(",") if slug.strip()]


def find_source_file(split_dir: Path, split: str) -> Path | None:
    for pattern in SOURCE_PATTERNS:
        resolved = pattern.format(split=split)
        for candidate in sorted(split_dir.glob(resolved)):
            if candidate.name == "data.jsonl":
                continue
            return candidate
    return None


def load_first_json_record(path: Path) -> dict | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                return json.loads(line)
    except Exception:
        return None
    return None


def is_placeholder(path: Path) -> bool:
    record = load_first_json_record(path)
    if not record:
        return False
    ident = str(record.get("id", "")).lower()
    if "placeholder" in ident:
        return True
    tokens = record.get("tokens")
    if isinstance(tokens, list) and len(tokens) <= 2:
        normalized = " ".join(str(tok).lower() for tok in tokens)
        if "placeholder" in normalized:
            return True
    return False


def register_layout_splits(slugs: Iterable[str], force: bool) -> int:
    updates = 0
    missing: List[str] = []
    for slug in slugs:
        split_root = TRAINING_DIR / "processed" / slug / "splits"
        if not split_root.exists():
            print(f"[register] skipping '{slug}': {split_root} missing.")
            continue
        for split in SPLITS:
            split_dir = split_root / split
            if not split_dir.exists():
                split_dir.mkdir(parents=True, exist_ok=True)
            dest = split_dir / "data.jsonl"
            if dest.exists() and not force and not is_placeholder(dest):
                continue
            source = find_source_file(split_dir, split)
            if source is None:
                missing.append(f"{slug}:{split} ({split_dir})")
                continue
            shutil.copy2(source, dest)
            updates += 1
            print(f"[register] '{slug}' ({split}) <- {source.name}")
    if missing:
        formatted = "\n  - ".join(missing)
        raise SystemExit(
            "[register] Missing real JSONL splits; drop the actual files next to data.jsonl and rerun:\n"
            f"  - {formatted}\n"
            "Hint: stage your exports as `<split>.jsonl` (train/val/test) or pass explicit paths via "
            "`scripts/register_layout_dataset.py`."
        )
    return updates


def ensure_layoutlm_processor(model_dir: Path, base_source: str) -> bool:
    try:
        from transformers import LayoutLMv3Processor  # type: ignore
    except Exception:
        print("[layoutlm] transformers unavailable; skipping processor validation.")
        return False

    try:
        LayoutLMv3Processor.from_pretrained(str(model_dir))
        return False
    except Exception:
        pass

    try:
        processor = LayoutLMv3Processor.from_pretrained(base_source)
    except Exception as exc:  # pragma: no cover - depends on local HF cache
        raise SystemExit(f"[layoutlm] Failed to load processor from '{base_source}': {exc}") from exc
    processor.save_pretrained(str(model_dir))
    print(f"[layoutlm] Saved processor assets from '{base_source}' into {model_dir}.")
    return True


def run_release(pdf: Path, layout_model: Path, args: argparse.Namespace) -> None:
    env = os.environ.copy()
    env.setdefault("LATEXIFY_OCR_MAX_HEAVY", args.ocr_max_heavy)
    env.setdefault("LATEXIFY_CLIP_DEVICE", args.clip_device)
    env.setdefault("LATEXIFY_LAYOUTLM_DEVICE", args.layout_device)
    env["LATEXIFY_LAYOUTLM_MODEL"] = str(layout_model)
    if args.extra_env:
        for pair in args.extra_env:
            if "=" not in pair:
                raise SystemExit(f"Invalid --extra-env entry '{pair}'. Expected KEY=VALUE.")
            key, value = pair.split("=", 1)
            env[key.strip()] = value
    python_bin = RELEASE_VENV_PY if RELEASE_VENV_PY.exists() else Path(sys.executable)
    cmd = [str(python_bin), str((REPO_ROOT / "run_release.py").resolve()), "--pdf", str(pdf)]
    print("[release] Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)


def main() -> None:
    args = parse_args()
    pdf_path = (REPO_ROOT / args.pdf).resolve() if not os.path.isabs(args.pdf) else Path(args.pdf)
    if not pdf_path.exists():
        raise SystemExit(f"PDF '{pdf_path}' not found.")
    layout_model = (REPO_ROOT / args.layout_model).resolve() if not os.path.isabs(args.layout_model) else Path(args.layout_model)
    if not layout_model.exists():
        raise SystemExit(f"LayoutLM checkpoint '{layout_model}' not found.")
    if not args.no_register:
        slugs = resolve_slugs(args.slugs)
        updated = register_layout_splits(slugs, force=args.force_register)
        if updated:
            print(f"[register] Updated {updated} split file(s).")
    ensure_layoutlm_processor(layout_model, args.layout_base)
    if args.skip_release:
        print("[release] Skipped by --skip-release.")
        return
    run_release(pdf_path, layout_model, args)


if __name__ == "__main__":
    main()
