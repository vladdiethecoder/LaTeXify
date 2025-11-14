#!/usr/bin/env python3
"""Runs SFT + DPO training using datasets built from reference_tex."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
# Note: Install trl when running this script.
try:  # pragma: no cover
    from trl import DPOTrainer
except Exception as exc:  # pragma: no cover
    raise SystemExit("trl is required to run this script. pip install trl") from exc


def load_jsonl(path: Path):
    return load_dataset("json", data_files=str(path))


def run_sft(model_name: str, dataset_path: Path, output_dir: Path, epochs: int = 1) -> Path:
    data = load_jsonl(dataset_path)["train"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    def tokenize(batch):
        return tokenizer(batch["prompt"], text_target=batch["response"], truncation=True)
    tokenized = data.map(tokenize, batched=True)
    training_args = TrainingArguments(
        output_dir=str(output_dir / "sft"),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=epochs,
        logging_steps=10,
        save_strategy="no",
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized, tokenizer=tokenizer)
    trainer.train()
    output_path = output_dir / "sft_model"
    trainer.save_model(output_path)
    tokenizer.save_pretrained(output_path)
    return output_path


def run_dpo(base_model_dir: Path, preference_path: Path, output_dir: Path, epochs: int = 1) -> Path:
    dataset = load_jsonl(preference_path)["train"]
    tokenizer = AutoTokenizer.from_pretrained(base_model_dir)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_model_dir)
    training_args = TrainingArguments(
        output_dir=str(output_dir / "dpo"),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=epochs,
        logging_steps=10,
        save_strategy="no",
    )
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        beta=0.1,
        train_dataset=dataset,
        tokenizer=tokenizer,
        max_length=1024,
    )
    trainer.train()
    output_path = output_dir / "dpo_model"
    trainer.model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SFT + DPO training.")
    parser.add_argument("--model", default="tiiuae/falcon-7b-instruct", help="Base model for fine-tuning")
    parser.add_argument("--sft_dataset", type=Path, default=Path("datasets/sft_dataset.jsonl"))
    parser.add_argument("--preference_dataset", type=Path, default=Path("datasets/preference_dataset.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("models/alignment"), help="Output directory")
    parser.add_argument("--sft_epochs", type=int, default=1)
    parser.add_argument("--dpo_epochs", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sft_model = run_sft(args.model, args.sft_dataset, args.output, epochs=args.sft_epochs)
    run_dpo(sft_model, args.preference_dataset, args.output, epochs=args.dpo_epochs)
    print(f"DPO-tuned model saved under {args.output}")


if __name__ == "__main__":
    main()
