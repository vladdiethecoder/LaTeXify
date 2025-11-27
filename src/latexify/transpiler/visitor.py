import ast
from typing import Optional, Union
from .precedence import Precedence
from .mapper import IdentifierMapper
from .config import TranspilerConfig
from .mixins import ArithmeticMixin, ControlFlowMixin, DataStructureMixin, FunctionMixin

class LatexifyVisitor(ast.NodeVisitor, ArithmeticMixin, ControlFlowMixin, DataStructureMixin, FunctionMixin):
    def __init__(self, config: Optional[TranspilerConfig] = None):
        self.config = config or TranspilerConfig()
        self.mapper = IdentifierMapper(self.config.identifiers)

    def visit(self, node: ast.AST, parent_precedence: Precedence = Precedence.LOWEST) -> str:
        """
        Override standard visit to accept parent_precedence.
        """
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        
        # If the visitor method accepts parent_precedence, pass it.
        # Otherwise, assume it consumes the node context fully (like statement visitors).
        # We can inspect signature, or just default to trying.
        # Since I defined mixins with it, most will take it.
        # For standard NodeVisitor methods (if any left), they don't take it.
        # But I am overriding the dispatch mechanism.
        
        # Python's ast.NodeVisitor.visit doesn't take extra args easily without strict overriding.
        # So I have to implement the dispatch logic manually here for my custom signature.
        
        try:
            return visitor(node, parent_precedence)
        except TypeError:
            # Fallback for methods that don't accept precedence (e.g. generic_visit or statements)
            return visitor(node)

    def generic_visit(self, node: ast.AST) -> str:
        raise NotImplementedError(f"No visitor implemented for {type(node).__name__}")

    def visit_Module(self, node: ast.Module) -> str:
        return "\n".join([self.visit(stmt) for stmt in node.body])

    def visit_Expr(self, node: ast.Expr) -> str:
        return self.visit(node.value)

    def visit_Name(self, node: ast.Name, parent_precedence: Precedence = Precedence.LOWEST) -> str:
        return self.mapper.map_identifier(node.id)

    def visit_Constant(self, node: ast.Constant, parent_precedence: Precedence = Precedence.LOWEST) -> str:
        if node.value is None:
            return "\\text{None}"
        if isinstance(node.value, bool):
            return "\\text{True}" if node.value else "\\text{False}"
        if isinstance(node.value, str):
            return f"\\text{{{node.value}}}" # Simple string literal
        return str(node.value)

    def visit_Assign(self, node: ast.Assign) -> str:
        targets = [self.visit(t) for t in node.targets]
        value = self.visit(node.value)
        
        op = "=" if self.config.equation_mode else "\\leftarrow"
        
        target_str = " = ".join(targets) # a = b = ...
        return f"{target_str} {op} {value}"

    def visit_AnnAssign(self, node: ast.AnnAssign) -> str:
        target = self.visit(node.target)
        if node.value:
            value = self.visit(node.value)
            op = "=" if self.config.equation_mode else "\\leftarrow"
            return f"{target} {op} {value}"
        return target # Just declaration?

    def visit_Return(self, node: ast.Return) -> str:
        if node.value:
            return f"\\textbf{{return }} {self.visit(node.value)}"
        return "\\textbf{return}"

    # Add support for Attributes (obj.attr)
    def visit_Attribute(self, node: ast.Attribute, parent_precedence: Precedence = Precedence.LOWEST) -> str:
        # obj.attr -> obj_{attr} or obj.attr? 
        # Let's assume obj.attr style for now or delegate to mapper.
        obj = self.visit(node.value, Precedence.ATOM)
        return f"{obj}.{self.mapper.map_identifier(node.attr)}"

def latexify(code: str, config: Optional[TranspilerConfig] = None) -> str:
    tree = ast.parse(code)
    visitor = LatexifyVisitor(config)
    return visitor.visit(tree)
