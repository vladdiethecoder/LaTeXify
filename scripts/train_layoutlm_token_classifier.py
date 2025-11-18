#!/usr/bin/env python3
"""Minimal LayoutLM token-classification trainer for local DocLayNet-style datasets."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Sequence

try:  # pragma: no cover - optional dependency
    import datasets
    from datasets import DatasetDict
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "The 'datasets' package is required for LayoutLM fine-tuning. Install it via "
        "`pip install datasets>=2.19.0` or `pip install -r release/requirements.txt`."
    ) from exc
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

try:  # pragma: no cover - optional convenience
    from transformers import LayoutLMv3Processor
except Exception:  # pragma: no cover
    LayoutLMv3Processor = None  # type: ignore

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", required=True, help="Path to JSON/JSONL annotations for training.")
    parser.add_argument("--validation-file", required=True, help="Path to JSON/JSONL annotations for validation.")
    parser.add_argument(
        "--model-name-or-path",
        "--model_name_or_path",
        default="microsoft/layoutlmv3-base",
        help="Hugging Face checkpoint to fine-tune.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory to store checkpoints.")
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--max-train-samples", type=int, help="Optional cap on train samples.")
    parser.add_argument("--max-eval-samples", type=int, help="Optional cap on eval samples.")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument(
        "--label-all-tokens",
        action="store_true",
        help="Propagate word labels to all subword pieces.",
    )
    return parser.parse_args()


def load_json_dataset(train_file: str, validation_file: str) -> DatasetDict:
    data_files = {"train": train_file, "validation": validation_file}
    dataset = datasets.load_dataset("json", data_files=data_files)
    return dataset


def build_labels(dataset: DatasetDict) -> List[str]:
    features = dataset["train"].features
    ner_feature = features.get("ner_tags")
    if ner_feature and isinstance(ner_feature.feature, datasets.features.ClassLabel):
        return ner_feature.feature.names  # type: ignore[return-value]
    labels = set()
    for split in ("train", "validation"):
        for sample in dataset[split]["ner_tags"]:
            for tag in sample:
                labels.add(str(tag))
    labels_list = sorted(labels)
    LOGGER.info("Discovered %s labels: %s", len(labels_list), labels_list)
    return labels_list


def align_labels(
    examples,
    tokenizer,
    label_to_id: Dict[str, int],
    label_all_tokens: bool,
):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        boxes=examples.get("bboxes"),
        truncation=True,
        padding="max_length",
        return_offsets_mapping=False,
    )
    all_labels = []
    all_bboxes = []
    for batch_index, labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=batch_index)
        previous_word_idx = None
        label_ids: List[int] = []
        bbox_sequence: List[List[int]] = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
                bbox_sequence.append([0, 0, 0, 0])
                continue
            if word_idx != previous_word_idx:
                label = labels[word_idx]
                label_ids.append(label_to_id[str(label)])
            else:
                label_ids.append(label_to_id[str(labels[word_idx])] if label_all_tokens else -100)
            if examples.get("bboxes"):
                bbox_sequence.append(list(map(int, examples["bboxes"][batch_index][word_idx])))
            else:
                bbox_sequence.append([0, 0, 0, 0])
            previous_word_idx = word_idx
        all_labels.append(label_ids)
        all_bboxes.append(bbox_sequence)
    tokenized_inputs["labels"] = all_labels
    tokenized_inputs["bbox"] = all_bboxes
    return tokenized_inputs


def main() -> None:
    args = parse_args()
    dataset = load_json_dataset(args.train_file, args.validation_file)
    if args.max_train_samples:
        dataset["train"] = dataset["train"].select(range(min(args.max_train_samples, len(dataset["train"]))))
    if args.max_eval_samples:
        dataset["validation"] = dataset["validation"].select(range(min(args.max_eval_samples, len(dataset["validation"]))))

    label_list = build_labels(dataset)
    label_to_id = {label: idx for idx, label in enumerate(label_list)}

    processor = None
    if LayoutLMv3Processor is not None:
        try:
            processor = LayoutLMv3Processor.from_pretrained(args.model_name_or_path)
            tokenizer = processor.tokenizer
        except Exception as exc:  # pragma: no cover - optional dependency
            LOGGER.warning(
                "LayoutLMv3Processor unavailable at %s (%s); falling back to AutoTokenizer.",
                args.model_name_or_path,
                exc,
            )
            processor = None
            tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=len(label_list),
        id2label={i: label for i, label in enumerate(label_list)},
        label2id=label_to_id,
    )

    def preprocess(batch):
        return align_labels(
            batch,
            tokenizer=tokenizer,
            label_to_id=label_to_id,
            label_all_tokens=args.label_all_tokens,
        )

    tokenized = dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing",
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="steps",
        fp16=args.fp16 and not args.bf16,
        bf16=args.bf16,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    if processor is not None:
        try:
            processor.tokenizer = tokenizer
        except Exception:
            pass
        processor.save_pretrained(args.output_dir)
    metadata = {
        "train_file": args.train_file,
        "validation_file": args.validation_file,
        "model_name": args.model_name_or_path,
        "output_dir": args.output_dir,
        "labels": label_list,
        "epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
    }
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir) / "training_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    LOGGER.info("Training complete. Checkpoints saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
