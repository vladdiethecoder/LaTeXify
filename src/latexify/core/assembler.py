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
                latex_output.append("\\title{" + content + "}")
            elif block.category == "Header":
                pass 
            elif block.category == "Footer":
                pass
            elif block.category == "Equation_Display":
                latex_output.append("\\\[\n" + content + "\n\\\\]\n")
            elif block.category == "Equation_Inline":
                latex_output.append("$" + content + "$")
            elif block.category == "Table":
                latex_output.append(content) 
            elif block.category == "Image":
                latex_output.append("\\begin{figure}[h]\n\\centering\n\\includegraphics[width=0.8\\linewidth]{" + content + "}\n\\end{figure}")
            else: 
                latex_output.append("\n" + content + "\n")
                
        return "\n".join(latex_output)