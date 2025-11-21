from typing import List
from .reading_order import LayoutBlock

class Assembler:
    def assemble(self, blocks: List[LayoutBlock]) -> str:
        """
        Assemble blocks into a raw LaTeX string.
        """
        latex_output = []
        
        for block in blocks:
            content = block.content or ""
            
            if block.category == "Title":
                latex_output.append(f"\\title{{{content}}}")
            elif block.category == "Header":
                # Often ignored or handled separately
                pass 
            elif block.category == "Footer":
                pass
            elif block.category == "Equation_Display":
                latex_output.append(f"\n\[
{content}
\]
")
            elif block.category == "Equation_Inline":
                # Usually inline equations are embedded in text, 
                # but if detected separately (which is rare for inline), 
                # we might wrap it.
                latex_output.append(f"${content}$")
            elif block.category == "Table":
                latex_output.append(content) # Tables often come pre-formatted
            elif block.category == "Image":
                latex_output.append(f"\\begin{{figure}}[h]
\\centering
\\includegraphics[width=0.8\\linewidth]{{{content}}}
\\end{{figure}}")
            else: # Text_Block or fallback
                latex_output.append(f"\n{content}
")
                
        return "\n".join(latex_output)
