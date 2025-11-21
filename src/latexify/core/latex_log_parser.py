"""
Self-Correction via Compiler Log Parsing.
Parses pdflatex.log to extract structured error information for the RepairAgent.
"""
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

@dataclass
class LatexError:
    line: int
    message: str
    context: str
    error_type: str # undefined_control_sequence, missing_bracket, etc.

class LogParser:
    ERROR_PATTERNS = [
        (r"! Undefined control sequence.", "undefined_control_sequence"),
        (r"! Missing \"}\" inserted.", "missing_closing_brace"),
        (r"! Missing \"{\" inserted.", "missing_opening_brace"),
        (r"! LaTeX Error: Environment .* undefined.", "undefined_environment"),
        (r"! LaTeX Error: File `.*' not found.", "missing_package"),
    ]

    def parse(self, log_path: Path) -> List[LatexError]:
        if not log_path.exists():
            return []

        content = log_path.read_text(errors="replace")
        errors = []
        lines = content.splitlines()
        
        for i, line in enumerate(lines):
            if line.startswith("!"):
                # Found error
                error_type = "unknown"
                for pattern, type_name in self.ERROR_PATTERNS:
                    if re.match(pattern, line):
                        error_type = type_name
                        break
                
                # Try to find line number in subsequent lines
                # Typical format: l.10 \command
                line_num = -1
                context = ""
                for j in range(1, 5): # Look ahead a few lines
                    if i + j >= len(lines): break
                    next_line = lines[i+j]
                    m = re.match(r"l\. (\d+)(.*)", next_line)
                    if m:
                        line_num = int(m.group(1))
                        context = m.group(2).strip()
                        break
                
                errors.append(LatexError(
                    line=line_num,
                    message=line,
                    context=context,
                    error_type=error_type
                ))
        
        return errors

def get_fix_prompt(errors: List[LatexError], source_code: str) -> str:
    """Generate a prompt for the LLM based on parsed errors."""
    source_lines = source_code.splitlines()
    prompt = "Fix the following LaTeX compilation errors:\n\n"
    
    for err in errors[:5]: # Limit to top 5 to avoid context overflow
        context_line = source_lines[err.line - 1] if 0 < err.line <= len(source_lines) else "???"
        prompt += f"Error: {err.message}\nType: {err.error_type}\nLine {err.line}: {context_line}\n\n"
        
    return prompt
