
import re
from pathlib import Path

def _ensure_math_macros(tex: str) -> str:
    # Fix common OCR typos first. Use ensuremath to be safe.
    tex = tex.replace(r"\inR", r"\ensuremath{\in \mathbb{R}}")
    tex = tex.replace(r"\inZ", r"\ensuremath{\in \mathbb{Z}}")
    tex = tex.replace(r"\inN", r"\ensuremath{\in \mathbb{N}}")
    
    replacements = [
        "pm", "times", "neq", "leq", "geq", "le", "ge",
        "in", "infty", "Rightarrow", "rightarrow", "leftarrow", "Leftrightarrow",
        "cdot", "approx", "equiv", "propto", "angle", "nabla", "partial",
        "alpha", "beta", "gamma", "theta", "pi", "mu", "sigma", "omega", "Delta"
    ]
    
    for macro in replacements:
        tex = re.sub(rf"\\{macro}(?![a-zA-Z])", rf"\\ensuremath{{\\{macro}}}", tex)
        
    return tex

def main():
    path = Path("release/output/sample.tex")
    if not path.exists():
        print("File not found")
        return
    
    content = path.read_text(encoding="utf-8")
    print(f"Original len: {len(content)}")
    if r"\inR" in content:
        print("Found \inR")
    else:
        print("Did NOT find \inR")
        # find what looks like it
        match = re.search(r"\\in.?R", content)
        if match:
            print(f"Found close match: '{match.group(0)}'")
            print(f"Hex: {match.group(0).encode('utf-8').hex()}")

    fixed = _ensure_math_macros(content)
    print(f"Fixed len: {len(fixed)}")
    if r"\inR" in fixed:
        print("STILL Found \inR")
    else:
        print("Fixed \inR")
        
    if r"\ensuremath{\in \mathbb{R}}" in fixed:
        print("Found replacement")

if __name__ == "__main__":
    main()

