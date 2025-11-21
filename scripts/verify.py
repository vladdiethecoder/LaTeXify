import argparse
from pathlib import Path
from nltk.translate.bleu_score import sentence_bleu
import sys

# Add src to sys.path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root / "src"))

# Mocking Hydra config for verify script usage without full setup
from omegaconf import OmegaConf
from latexify.core.pipeline import LaTeXifyPipeline

def calculate_bleu(reference: str, candidate: str) -> float:
    """
    Calculate BLEU score between reference and candidate LaTeX strings.
    Simple tokenization by space.
    """
    ref_tokens = reference.split()
    cand_tokens = candidate.split()
    return sentence_bleu([ref_tokens], cand_tokens)

def main():
    parser = argparse.ArgumentParser(description="Verify LaTeXify against Golden Set")
    parser.add_argument("--golden-dir", type=Path, default=Path("data/golden_set"), help="Path to Golden Set")
    parser.add_argument("--run-pipeline", action="store_true", help="Actually run the pipeline (requires models)")
    args = parser.parse_args()
    
    if not args.golden_dir.exists():
        print(f"Golden Set directory not found: {args.golden_dir}")
        return

    pdfs = list(args.golden_dir.glob("*.pdf"))
    if not pdfs:
        print("No PDFs found in Golden Set.")
        return
    
    # Init pipeline if requested
    pipeline = None
    if args.run_pipeline:
        cfg = OmegaConf.create({
            "pipeline": {
                "ingestion": {"dpi": 300},
                "layout": {"model": "yolov10n.pt"},
                "ocr": {"lang": "en"},
                "refinement": {"enabled": True}
            }
        })
        pipeline = LaTeXifyPipeline(cfg)
        
    print(f"Found {len(pdfs)} samples. Starting verification...")
    
    scores = []
    for pdf in pdfs:
        tex = pdf.with_suffix(".tex")
        if not tex.exists():
            print(f"Missing GT LaTeX for {pdf.name}, skipping.")
            continue
            
        if pipeline:
            try:
                print(f"Processing {pdf.name}...")
                candidate_tex = pipeline.process(pdf)
            except Exception as e:
                print(f"Failed to process {pdf.name}: {e}")
                continue
        else:
            print(f"Mock Verify {pdf.name}: [Pipeline Skipped]")
            candidate_tex = "Mock LaTeX Output" 
        
        gt_tex = tex.read_text(encoding="utf-8")
        score = calculate_bleu(gt_tex, candidate_tex)
        print(f"  BLEU: {score:.4f}")
        scores.append(score)
        
    if scores:
        avg_bleu = sum(scores) / len(scores)
        print(f"Average BLEU: {avg_bleu:.4f}")
    else:
        print("No valid pairs found.")

if __name__ == "__main__":
    main()