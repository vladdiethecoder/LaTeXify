#!/usr/bin/env python3
"""
Direct Preference Optimization (DPO) Training Script.
Fine-tunes a local model on (Accepted, Rejected) LaTeX pairs.
"""
import argparse
import json
import logging
from pathlib import Path
import sys

try:
    from datasets import load_dataset
    from trl import DPOTrainer
    from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer
    import torch
except ImportError:
    print("Install trl, datasets, transformers, torch first.")
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

def train_dpo(model_name: str, data_dir: str, output_dir: str):
    data_path = Path(data_dir)
    if not data_path.exists():
        LOGGER.error(f"Data directory {data_dir} not found.")
        return

    # 1. Load Data
    # Consolidate JSONs into a dataset
    data_files = list(data_path.glob("*.json"))
    if not data_files:
        LOGGER.error("No annotation files found.")
        return

    json_list = []
    for f in data_files:
        with open(f, 'r') as jf:
            json_list.append(json.load(jf))
    
    # Format for DPO: prompt, chosen, rejected
    # In our case, prompt is the image or the rough OCR text. 
    # For text-only DPO, we assume 'prompt' is derived from the chunk context.
    # We'll use a placeholder or extracted text if available in the annotation.
    dpo_data = []
    for item in json_list:
        dpo_data.append({
            "prompt": f"Convert the following image region to LaTeX: [IMAGE_CONTEXT]", # Simplified
            "chosen": item["chosen"],
            "rejected": item["rejected"]
        })

    # Convert to HF Dataset
    # In a real script we'd split train/val
    # Here strictly scaffolding
    
    # 2. Initialize Model
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    ref_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # 3. Train
    training_args = TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        max_steps=100,
        learning_rate=1e-5,
        fp16=True,
        logging_steps=10,
        output_dir=output_dir,
    )

    trainer = DPOTrainer(
        model,
        ref_model,
        args=training_args,
        train_dataset=dpo_data, # Needs proper HF dataset object
        tokenizer=tokenizer,
    )
    
    trainer.train()
    trainer.save_model(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deepseek-ai/deepseek-coder-1.3b-instruct")
    parser.add_argument("--data", default="training_data/human_feedback")
    parser.add_argument("--output", default="models/latexify-dpo-adapter")
    args = parser.parse_args()
    
    train_dpo(args.model, args.data, args.output)
