#!/usr/bin/env python3
"""Fine-tune a causal language model on local LaTeX/text corpora."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence

try:  # pragma: no cover - optional dependency until requirements installed
    from datasets import Dataset
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "The 'datasets' package is required for fine-tuning. Install it via "
        "`pip install datasets>=2.19.0` or `pip install -r release/requirements.txt`."
    ) from exc

from transformers import (  # type: ignore
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_RAW_ROOT = REPO_ROOT / "training_data" / "raw"
MANIFESTS_DIR = REPO_ROOT / "training_data" / "manifests"
DEFAULT_EXTENSIONS = [".tex", ".txt", ".md", ".jsonl", ".json", ".parquet"]

PROMPT_TEMPLATE = """### Instruction:
Convert the input below into high-quality LaTeX that mirrors the source.

### Input:
{input}

### Output:
{target}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-slug", required=True, help="Slug under training_data/raw/.")
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Hugging Face model id to fine-tune.",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory to store checkpoints (default: training_runs/<slug>-<model>/).",
    )
    parser.add_argument(
        "--extensions",
        nargs="*",
        default=DEFAULT_EXTENSIONS,
        help="File extensions considered as training text.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        help="Optional limit on how many files to read per dataset.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        help="Optional limit on the number of training records.",
    )
    parser.add_argument(
        "--json-field",
        help="Single field to extract from JSON/JSONL payloads (dot notation supported).",
    )
    parser.add_argument(
        "--json-input-field",
        help="Dot-notated field used as the prompt when json-target-field is provided.",
    )
    parser.add_argument(
        "--json-target-field",
        help="Dot-notated field used as the completion when training instruction-style data.",
    )
    parser.add_argument(
        "--prompt-template",
        default=PROMPT_TEMPLATE,
        help="Template applied when both input and target JSON fields are present.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=1024,
        help="Maximum tokenized sequence length.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Per-device batch size.",
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=8,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--epochs",
        type=float,
        default=1.0,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Base learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Weight decay factor.",
    )
    parser.add_argument(
        "--eval-split",
        type=float,
        default=0.05,
        help="Fraction reserved for evaluation (0 disables eval).",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed.")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Enable fp16 training if CUDA is available.",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Enable bf16 training (takes precedence over fp16).",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable transformer gradient checkpointing.",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="Steps between Trainer logs.",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=100,
        help="Steps between checkpoint saves.",
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=2,
        help="Maximum checkpoints to keep on disk.",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=4,
        help="Parallel workers for dataset map/group operations.",
    )
    parser.add_argument(
        "--max-file-size-mb",
        type=float,
        default=25.0,
        help="Skip files larger than this threshold when scanning raw data.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        help="Resume training from the provided checkpoint path.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the dataset and print stats without launching Trainer.",
    )
    return parser.parse_args()


def slugify_join(values: Sequence[str]) -> str:
    safe = "-".join(values)
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in safe)


def manifest_for_slug(slug: str) -> Dict | None:
    manifest_path = MANIFESTS_DIR / f"{slug}.json"
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text())


def dotted_lookup(payload: Dict, path: str | None) -> str | None:
    if not path:
        return None
    value = payload
    for key in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(key)
        if value is None:
            return None
    if isinstance(value, (list, tuple)):
        return "\n".join(str(item) for item in value)
    return str(value)


def normalize_extensions(extensions: Iterable[str]) -> List[str]:
    normalized = []
    for ext in extensions:
        ext = ext.strip().lower()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = "." + ext
        normalized.append(ext)
    return normalized or DEFAULT_EXTENSIONS


def iter_candidate_files(root: Path, extensions: Sequence[str]) -> Iterator[Path]:
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() in extensions:
            yield path


def read_text_file(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None


def iter_structured_records(path: Path) -> Iterator[Dict]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
    elif suffix == ".json":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    yield item
        elif isinstance(data, dict):
            yield data
    elif suffix == ".parquet":
        try:
            parquet_ds = Dataset.from_parquet(str(path))
        except Exception:
            return
        for row in parquet_ds:
            yield dict(row)


def gather_text_corpus(
    dataset_slug: str,
    extensions: Sequence[str],
    max_files: int | None,
    max_records: int | None,
    max_file_size_mb: float,
    json_field: str | None,
    json_input_field: str | None,
    json_target_field: str | None,
    prompt_template: str,
) -> List[str]:
    dataset_root = TRAINING_RAW_ROOT / dataset_slug
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset slug '{dataset_slug}' not found under {TRAINING_RAW_ROOT}")
    texts: List[str] = []
    file_count = 0
    size_limit_bytes = max_file_size_mb * 1024**2
    for candidate in iter_candidate_files(dataset_root, extensions):
        if candidate.stat().st_size > size_limit_bytes:
            continue
        file_count += 1
        if max_files and file_count >= max_files:
            break
        suffix = candidate.suffix.lower()
        if suffix in {".json", ".jsonl", ".parquet"}:
            for record in iter_structured_records(candidate):
                text = None
                if json_target_field:
                    target = dotted_lookup(record, json_target_field)
                    if not target:
                        continue
                    prompt = dotted_lookup(record, json_input_field) or ""
                    text = prompt_template.format(input=prompt.strip(), target=target.strip()).strip()
                else:
                    field = json_field or json_input_field or json_target_field
                    if field:
                        extracted = dotted_lookup(record, field)
                        text = extracted.strip() if extracted else None
                    else:
                        text = json.dumps(record, ensure_ascii=False)
                if text:
                    texts.append(text)
                    if max_records and len(texts) >= max_records:
                        return texts
        else:
            payload = read_text_file(candidate)
            if payload:
                payload = payload.strip()
                if payload:
                    texts.append(payload)
                    if max_records and len(texts) >= max_records:
                        return texts
    return texts


def build_dataset(texts: Sequence[str], seed: int, eval_split: float) -> Dict[str, Dataset]:
    dataset = Dataset.from_dict({"text": list(texts)})
    dataset = dataset.shuffle(seed=seed)
    if eval_split > 0.0 and len(dataset) > 1:
        split = dataset.train_test_split(test_size=eval_split, seed=seed)
        return {"train": split["train"], "eval": split["test"]}
    return {"train": dataset, "eval": None}


def tokenize_dataset(dataset: Dataset, tokenizer, block_size: int, num_proc: int) -> Dataset:
    def tokenize_function(batch: Dict[str, List[str]]) -> Dict:
        return tokenizer(batch["text"], truncation=True, max_length=block_size)

    return dataset.map(tokenize_function, batched=True, remove_columns=["text"], num_proc=num_proc)


def write_metadata(
    output_dir: Path,
    dataset_slug: str,
    model_name: str,
    sample_count: int,
    eval_count: int,
    manifest: Dict | None,
    args: argparse.Namespace,
) -> None:
    metadata = {
        "dataset_slug": dataset_slug,
        "model_name": model_name,
        "samples": sample_count,
        "eval_samples": eval_count,
        "output_dir": str(output_dir),
        "manifest": manifest or {},
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "training_args": {
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "block_size": args.block_size,
            "eval_split": args.eval_split,
            "fp16": args.fp16,
            "bf16": args.bf16,
        },
    }
    (output_dir / "finetune_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    extensions = normalize_extensions(args.extensions)
    texts = gather_text_corpus(
        dataset_slug=args.dataset_slug,
        extensions=extensions,
        max_files=args.max_files,
        max_records=args.max_records,
        max_file_size_mb=args.max_file_size_mb,
        json_field=args.json_field,
        json_input_field=args.json_input_field,
        json_target_field=args.json_target_field,
        prompt_template=args.prompt_template,
    )
    if not texts:
        raise RuntimeError(
            f"No training samples discovered under {TRAINING_RAW_ROOT / args.dataset_slug}. "
            "Confirm that raw assets are staged and extensions/json fields are correct."
        )
    split = build_dataset(texts, seed=args.seed, eval_split=args.eval_split)
    train_dataset = split["train"]
    eval_dataset = split["eval"]
    manifest = manifest_for_slug(args.dataset_slug)
    model_slug = slugify_join([args.dataset_slug, Path(args.model_name).name])
    output_dir = Path(args.output_dir) if args.output_dir else (REPO_ROOT / "training_runs" / model_slug)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[finetune] dataset={args.dataset_slug} samples={len(train_dataset)} eval={len(eval_dataset) if eval_dataset else 0}")
    print(f"[finetune] writing checkpoints to {output_dir}")
    if args.dry_run:
        write_metadata(
            output_dir,
            args.dataset_slug,
            args.model_name,
            len(train_dataset),
            len(eval_dataset) if eval_dataset else 0,
            manifest,
            args,
        )
        print("[finetune] dry run complete (no training launched).")
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenized_train = tokenize_dataset(train_dataset, tokenizer, args.block_size, args.num_proc)
    tokenized_eval = tokenize_dataset(eval_dataset, tokenizer, args.block_size, args.num_proc) if eval_dataset else None

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        evaluation_strategy="steps" if tokenized_eval is not None else "no",
        save_strategy="steps",
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        fp16=bool(args.fp16 and not args.bf16),
        bf16=bool(args.bf16),
        report_to="none",
        gradient_checkpointing=args.gradient_checkpointing,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_state()
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    write_metadata(
        output_dir,
        args.dataset_slug,
        args.model_name,
        len(train_dataset),
        len(eval_dataset) if eval_dataset else 0,
        manifest,
        args,
    )
    print("[finetune] completed training.")


if __name__ == "__main__":
    main()
