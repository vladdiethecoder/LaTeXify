import sys
import os
from pathlib import Path
import logging

# Add repo root to path
sys.path.insert(0, os.getcwd())

from latexify.models.llm_refiner import LLMRefiner, LLMRefinerConfig, LLMRefinerError

logging.basicConfig(level=logging.INFO)

def main():
    repo_id = "Qwen/Qwen2.5-1.5B-Instruct"
    print(f"Attempting to load LLMRefiner with repo_id={repo_id}")
    
    config = LLMRefinerConfig(
        repo_id=repo_id,
        backend="hf",
        style_domain="default"
    )
    
    try:
        refiner = LLMRefiner(config)
        print("SUCCESS: Model loaded.")
        print(f"Device: {refiner._model_device}")
    except Exception as e:
        print("\nFAILURE: Could not load model.")
        print(f"Error type: {type(e)}")
        print(f"Error message: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

