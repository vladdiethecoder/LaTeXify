"""
Streamlit Annotation Callback.
Handles user feedback from the UI to create training data.
"""
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict

LOGGER = logging.getLogger(__name__)
DATASET_DIR = Path("training_data/human_feedback")

def save_annotation(
    chunk_id: str,
    original_image_path: str,
    bad_latex: str,
    corrected_latex: str,
    user_comments: str = ""
):
    """
    Save a (Image, Bad Latex, Good Latex) tuple for DPO/SFT.
    """
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{chunk_id}.json"
    
    entry = {
        "chunk_id": chunk_id,
        "image_path": original_image_path,
        "rejected": bad_latex,
        "chosen": corrected_latex,
        "comments": user_comments,
        "timestamp": timestamp
    }
    
    try:
        with open(DATASET_DIR / filename, "w") as f:
            json.dump(entry, f, indent=2)
        LOGGER.info(f"Annotation saved: {filename}")
        return True
    except Exception as e:
        LOGGER.error(f"Failed to save annotation: {e}")
        return False
