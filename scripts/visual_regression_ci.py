#!/usr/bin/env python3
"""
Visual Regression CI Script.
Compares two PDF files using diff-pdf and returns non-zero exit code if they differ.
"""
import argparse
import sys
import subprocess
import shutil
from pathlib import Path

def run_visual_regression(input_pdf: Path, reference_pdf: Path, output_diff: Path):
    if not shutil.which("diff-pdf"):
        print("Error: diff-pdf not found. Please install it (e.g., sudo apt-get install diff-pdf).")
        sys.exit(1)

    if not input_pdf.exists():
        print(f"Error: Input PDF {input_pdf} not found.")
        sys.exit(1)
    
    if not reference_pdf.exists():
        print(f"Error: Reference PDF {reference_pdf} not found.")
        sys.exit(1)

    # Run diff-pdf
    # --output-diff=<file> generates a visual diff PDF
    cmd = ["diff-pdf", "--output-diff", str(output_diff), str(reference_pdf), str(input_pdf)]
    
    try:
        # diff-pdf returns non-zero if files differ
        subprocess.run(cmd, check=True)
        print("Visual regression check passed: PDFs match.")
    except subprocess.CalledProcessError:
        print(f"Visual regression failed: Differences detected. See {output_diff}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_pdf", type=Path)
    parser.add_argument("reference_pdf", type=Path)
    parser.add_argument("--output-diff", type=Path, default=Path("diff.pdf"))
    args = parser.parse_args()
    
    run_visual_regression(args.input_pdf, args.reference_pdf, args.output_diff)