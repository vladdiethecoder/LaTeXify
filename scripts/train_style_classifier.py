#!/usr/bin/env python3
"""
Training script for the Latexify Style Classifier.
Parses the training_database-catalog.json to find relevant datasets (DocLayNet, NeurIPS, ICLR),
loads samples, and fine-tunes the StyleClassifier.
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure src is in python path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from latexify.ml.style_classifier import StyleClassifier

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

CATALOG_PATH = Path(__file__).resolve().parents[1] / "training_database-catalog.json"

def parse_catalog(catalog_path: Path) -> Dict[str, Dict]:
    """
    Reads the JSONL catalog and returns a dict of dataset metadata keyed by name.
    """
    if not catalog_path.exists():
        raise FileNotFoundError(f"Catalog not found at {catalog_path}")
    
    datasets = {}
    with open(catalog_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    entry = json.loads(line)
                    datasets[entry.get("name")] = entry
                except json.JSONDecodeError:
                    continue
    return datasets

def mock_load_dataset(name: str, split: str) -> List[Tuple[str, str]]:
    """
    Simulates loading data for the purpose of this exercise since we cannot download GBs of data.
    In a real scenario, this would use `datasets.load_dataset` or read local files defined in the catalog.
    
    Returns: List of (text, label) tuples.
    """
    LOGGER.info(f"Loading mock data for {name} ({split})...")
    
    # Synthetic data generation based on dataset characteristics
    data = []
    count = 100 if split == 'train' else 20
    
    if name == "DocLayNet":
        # DocLayNet covers: scientific, financial, legal, patents, manuals
        # We map these to "report" or "article" mostly, or "textbook" for manuals
        for _ in range(count):
            if random.random() > 0.5:
                text = "This financial report details the quarterly earnings. The assets significantly outweigh liabilities."
                label = "report"
            else:
                text = "In this scientific article, we explore the properties of quantum entanglement in isolated systems."
                label = "article"
            data.append((text, label))
            
    elif name == "NeurIPS Proceedings":
        for _ in range(count):
            text = "We propose a novel deep learning architecture. Extensive experiments show state-of-the-art performance on ImageNet."
            label = "neurips"
            data.append((text, label))
            
    elif name == "ICLR Proceedings":
        for _ in range(count):
            text = "Representation learning is crucial for downstream tasks. We introduce a self-supervised method for graph neural networks."
            label = "iclr"
            data.append((text, label))
            
    elif name == "Textbook Question Answering (TQA)":
        for _ in range(count):
            text = "Chapter 5: Photosynthesis. In this chapter, we will learn how plants convert sunlight into energy. Figure 5.1 shows the chloroplast."
            label = "textbook"
            data.append((text, label))
            
    else:
        # Generic fallback
        for _ in range(count):
            text = "This is a generic document text containing sections, figures, and tables."
            label = "article"
            data.append((text, label))
            
    return data

def prepare_training_data() -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Aggregates data from targeted datasets.
    """
    catalog = parse_catalog(CATALOG_PATH)
    
    # Target Datasets
    targets = [
        "DocLayNet",
        "NeurIPS Proceedings", 
        "ICLR Proceedings",
        "Textbook Question Answering (TQA)"
    ]
    
    all_train_texts = []
    all_train_labels = []
    all_val_texts = []
    all_val_labels = []
    
    for target in targets:
        if target not in catalog:
            LOGGER.warning(f"Target dataset {target} not found in catalog. Skipping.")
            continue
            
        # In production: Use catalog['url'] or catalog['download_urls'] to fetch real data
        # Here: Use mock loader
        train_data = mock_load_dataset(target, "train")
        val_data = mock_load_dataset(target, "validation")
        
        for text, label in train_data:
            all_train_texts.append(text)
            all_train_labels.append(label)
            
        for text, label in val_data:
            all_val_texts.append(text)
            all_val_labels.append(label)
            
    return all_train_texts, all_train_labels, all_val_texts, all_val_labels

def main():
    parser = argparse.ArgumentParser(description="Train Style Classifier")
    parser.add_argument("--output-dir", default="src/latexify/models/style_classifier", help="Path to save the model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    
    train_texts, train_labels, val_texts, val_labels = prepare_training_data()
    
    LOGGER.info(f"Training on {len(train_texts)} samples, validating on {len(val_texts)} samples.")
    LOGGER.info(f"Labels: {set(train_labels)}")
    
    classifier = StyleClassifier(device="cuda" if 0 else "cpu") # Auto-detect inside class, but passing explicit here implies preference
    
    classifier.train(
        train_texts=train_texts,
        train_labels=train_labels,
        val_texts=val_texts,
        val_labels=val_labels,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    main()
