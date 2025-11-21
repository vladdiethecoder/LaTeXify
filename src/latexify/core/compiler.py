import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Tuple

class LatexCompiler:
    def __init__(self, engine: str = "tectonic"):
        """
        Initialize the compiler.
        Args:
            engine: 'tectonic' or 'latexmk'
        """
        self.engine = engine
        
    def compile(self, latex_code: str) -> Tuple[bool, str]:
        """
        Compile LaTeX code.
        Returns:
            (success: bool, log: str)
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            tex_file = tmp_path / "main.tex"
            tex_file.write_text(latex_code, encoding="utf-8")
            
            cmd = []
            if self.engine == "tectonic":
                if not shutil.which("tectonic"):
                    return False, "tectonic not found in PATH"
                cmd = ["tectonic", str(tex_file)]
            elif self.engine == "latexmk":
                if not shutil.which("latexmk"):
                    return False, "latexmk not found in PATH"
                # -interaction=nonstopmode to prevent hanging on errors
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
                    return True, result.stdout
                else:
                    # Capture combined stdout/stderr as log
                    return False, result.stdout + "\n" + result.stderr
            except subprocess.TimeoutExpired:
                return False, "Compilation timed out"
            except Exception as e:
                return False, str(e)
