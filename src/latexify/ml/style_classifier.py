"""
Transformer-based Style Classifier for LaTeXify.
Wraps a DistilBERT model to classify document styles (NeurIPS, ICLR, Textbook, etc.) based on textual content.
"""
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
)
from pydantic import BaseModel

LOGGER = logging.getLogger(__name__)

class StylePrediction(BaseModel):
    style_label: str
    confidence: float
    logits: Dict[str, float]

class StyleClassifier:
    """
    A production-grade wrapper for a Transformer-based document style classifier.
    """
    
    DEFAULT_MODEL_NAME = "distilbert-base-uncased"
    DEFAULT_LABELS = ["neurips", "iclr", "textbook", "article", "report"]
    
    def __init__(self, model_dir: Optional[Union[str, Path]] = None, device: str = "cpu"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_dir = Path(model_dir) if model_dir else None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.model: Optional[PreTrainedModel] = None
        self.labels: List[str] = self.DEFAULT_LABELS
        self.id2label: Dict[int, str] = {}
        self.label2id: Dict[str, int] = {}

    def load(self) -> bool:
        """
        Attempts to load a pre-trained model from the model_dir.
        Returns True if successful, False otherwise.
        """
        if not self.model_dir or not self.model_dir.exists():
            LOGGER.warning(f"Model directory {self.model_dir} does not exist. Classifier not loaded.")
            return False

        try:
            LOGGER.info(f"Loading StyleClassifier from {self.model_dir}...")
            config = AutoConfig.from_pretrained(self.model_dir)
            self.id2label = config.id2label
            self.label2id = config.label2id
            self.labels = list(self.id2label.values())
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
            self.model.to(self.device)
            self.model.eval()
            
            LOGGER.info("StyleClassifier loaded successfully.")
            return True
        except Exception as e:
            LOGGER.error(f"Failed to load StyleClassifier: {e}")
            return False

    def train(
        self,
        train_texts: List[str],
        train_labels: List[str],
        val_texts: List[str],
        val_labels: List[str],
        output_dir: Union[str, Path],
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
    ):
        """
        Fine-tunes a base DistilBERT model on the provided text/label pairs.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Prepare labels
        unique_labels = sorted(list(set(train_labels + val_labels)))
        self.labels = unique_labels
        self.label2id = {l: i for i, l in enumerate(unique_labels)}
        self.id2label = {i: l for i, l in enumerate(unique_labels)}

        # Initialize Base Model
        LOGGER.info(f"Initializing base model {self.DEFAULT_MODEL_NAME} for training...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.DEFAULT_MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.DEFAULT_MODEL_NAME,
            num_labels=len(unique_labels),
            id2label=self.id2label,
            label2id=self.label2id
        )
        self.model.to(self.device)

        # Dataset Helper
        class TextDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels

            def __getitem__(self, idx):
                item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
                item['labels'] = torch.tensor(self.labels[idx])
                return item

            def __len__(self):
                return len(self.labels)

        # Tokenize
        train_encodings = self.tokenizer(train_texts, truncation=True, padding=True, max_length=512)
        val_encodings = self.tokenizer(val_texts, truncation=True, padding=True, max_length=512)

        train_dataset = TextDataset(train_encodings, [self.label2id[l] for l in train_labels])
        val_dataset = TextDataset(val_encodings, [self.label2id[l] for l in val_labels])

        # Training Arguments
        training_args = TrainingArguments(
            output_dir=str(output_path),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=str(output_path / "logs"),
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            learning_rate=learning_rate,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )

        LOGGER.info("Starting training...")
        trainer.train()
        
        LOGGER.info(f"Saving model to {output_path}...")
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        LOGGER.info("Training complete.")

    def predict(self, text: str) -> StylePrediction:
        """
        Predicts the style of a given text snippet.
        Fails gracefully if model is not loaded.
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("StyleClassifier model is not loaded. Call load() or train() first.")

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits
        
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        confidence, predicted_class_id = torch.max(probabilities, dim=-1)
        
        predicted_label = self.id2label[predicted_class_id.item()]
        confidence_score = confidence.item()
        
        logits_dict = {
            self.id2label[i]: probabilities[0][i].item()
            for i in range(len(self.labels))
        }

        return StylePrediction(
            style_label=predicted_label,
            confidence=confidence_score,
            logits=logits_dict
        )
