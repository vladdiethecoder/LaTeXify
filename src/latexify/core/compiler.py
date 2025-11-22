import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Tuple, Optional

class LatexCompiler:
    def __init__(self, engine: str = "tectonic"):
        """
        Initialize the compiler.
        Args:
            engine: 'tectonic' or 'latexmk'
        """
        self.engine = engine
        
    def compile(self, latex_code: str, output_pdf_path: Optional[Path] = None) -> Tuple[bool, str]:
        """
        Compile LaTeX code.
        Args:
            latex_code: The LaTeX source.
            output_pdf_path: Optional path to save the compiled PDF.
        Returns:
            (success: bool, log: str)
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            tex_file = tmp_path / "main.tex"
            tex_file.write_text(latex_code, encoding="utf-8")
            
            cmd = []
            if self.engine == "tectonic":
                if shutil.which("tectonic"):
                    cmd = ["tectonic", str(tex_file)]
                elif shutil.which("latexmk"):
                    # Fallback to latexmk when tectonic is unavailable.
                    cmd = [
                        "latexmk",
                        "-pdf",
                        "-interaction=nonstopmode",
                        "-output-directory=" + str(tmp_path),
                        str(tex_file),
                    ]
                else:
                    return False, "tectonic not found in PATH and latexmk fallback unavailable"
            elif self.engine == "latexmk":
                if not shutil.which("latexmk"):
                    return False, "latexmk not found in PATH"
                cmd = ["latexmk", "-pdf", "-interaction=nonstopmode", "-output-directory=" + str(tmp_path), str(tex_file)]
            else:
                return False, f"Unknown engine: {self.engine}"
            
            try:
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True, 
                    check=False,
                    timeout=60
                )
                
                if result.returncode == 0:
                    generated_pdf = tmp_path / "main.pdf"
                    if generated_pdf.exists() and output_pdf_path:
                        shutil.copy(generated_pdf, output_pdf_path)
                    return True, result.stdout
                else:
                    return False, result.stdout + "\n" + result.stderr
            except subprocess.TimeoutExpired:
                return False, "Compilation timed out"
            except Exception as e:
                return False, str(e)
