from typing import List, Dict, Any
from .reading_order import LayoutBlock

class Assembler:
    def assemble(self, blocks: List[LayoutBlock], metadata: Dict[str, Any] = None) -> str:
        """
        Assemble blocks into a raw LaTeX string.
        """
        latex_output = []
        metadata = metadata or {}
        
        # Preamble
        latex_output.append(r"\documentclass{article}")
        latex_output.append(r"\usepackage{amsmath}")
        latex_output.append(r"\usepackage{graphicx}")
        latex_output.append(r"\usepackage{booktabs}")
        
        # Title/Author in Preamble
        if metadata.get("title"):
            latex_output.append(f"\\title{{{metadata.get('title')}}}")
        if metadata.get("authors"):
            latex_output.append(f"\\author{{{metadata.get('authors')}}}")
            
        latex_output.append(r"\begin{document}")
        
        if metadata.get("title"):
            latex_output.append(r"\maketitle")
            
        if metadata.get("abstract"):
            latex_output.append(r"\begin{abstract}")
            latex_output.append(metadata.get("abstract"))
            latex_output.append(r"\end{abstract}")
        
        for block in blocks:
            content = block.content or ""
            
            if block.category == "Title":
                # If title extracted from metadata, we might skip detected title to avoid duplicate?
                # For now keep it but maybe as section?
                latex_output.append(r"\section*{ " + content + "}")
            elif block.category == "Header":
                pass 
            elif block.category == "Footer":
                pass
            elif block.category == "Equation_Display":
                latex_output.append(r"\\[" + "\n" + content + "\n" + r"\\]")
            elif block.category == "Equation_Inline":
                latex_output.append("$" + content + "$")
            elif block.category == "Table":
                latex_output.append(content) 
            elif block.category == "Image":
                latex_output.append(r"\begin{figure}[h]")
                latex_output.append(r"\centering")
                latex_output.append(r"\includegraphics[width=0.8\linewidth]{ " + content + "}")
                latex_output.append(r"\end{figure}")
            else: 
                latex_output.append("\n" + content + "\n")
        
        latex_output.append(r"\end{document}")
        
        return "\n".join(latex_output)