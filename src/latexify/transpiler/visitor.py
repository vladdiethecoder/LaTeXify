import ast
from typing import Optional, Any
from .precedence import Precedence
from .mapper import IdentifierMapper
from .config import TranspilerConfig
from .mixins import ArithmeticMixin, CalculusMixin, LinearAlgebraMixin, AlgorithmicMixin

class LatexifyVisitor(
    ast.NodeVisitor,
    ArithmeticMixin,
    CalculusMixin,
    LinearAlgebraMixin,
    AlgorithmicMixin
):
    """
    The SOTA v1.0 Visitor.
    Composes specialized Mixins to handle specific domains of math/logic.
    """
    def __init__(self, config: Optional[TranspilerConfig] = None):
        self.config = config or TranspilerConfig()
        self.mapper = IdentifierMapper(self.config.identifiers)

    def visit(self, node: ast.AST, parent_precedence: Precedence = Precedence.LOWEST) -> str:
        """
        Dispatch method.
        Python's built-in visit() doesn't accept arguments, so we implement our own recursive dispatcher
        that Mixins rely on.
        """
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        
        # Check if the visitor accepts precedence
        # Mixin methods do. Standard ones might not if not overridden.
        try:
            return visitor(node, parent_precedence)
        except TypeError:
            return visitor(node)

    def generic_visit(self, node: ast.AST) -> str:
        raise NotImplementedError(f"No visitor implemented for {type(node).__name__}")

    # --- Core Literals & Identifiers ---

    def visit_Module(self, node: ast.Module) -> str:
        return "\n".join([self.visit(stmt) for stmt in node.body])

    def visit_Expr(self, node: ast.Expr) -> str:
        return self.visit(node.value)

    def visit_Name(self, node: ast.Name, parent_precedence: Precedence = Precedence.LOWEST) -> str:
        return self.mapper.map_identifier(node.id)

    def visit_Constant(self, node: ast.Constant, parent_precedence: Precedence = Precedence.LOWEST) -> str:
        if node.value is None: return "\\text{None}"
        if node.value is True: return "\\text{True}"
        if node.value is False: return "\\text{False}"
        if isinstance(node.value, str): return f"\\text{{{node.value}}}"
        return str(node.value)

    def visit_Attribute(self, node: ast.Attribute, parent_precedence: Precedence = Precedence.LOWEST) -> str:
        # obj.attr -> obj_{attr} seems reasonable for math context
        obj = self.visit(node.value, Precedence.ATOM)
        attr = self.mapper.map_identifier(node.attr)
        return f"{obj}__{{{attr}}}"

    # --- Statements ---

    def visit_Assign(self, node: ast.Assign) -> str:
        targets = [self.visit(t) for t in node.targets]
        value = self.visit(node.value)
        op = "=" if self.config.equation_mode else "\\leftarrow"
        lhs = " = ".join(targets)
        return f"{lhs} {op} {value}"

    def visit_AnnAssign(self, node: ast.AnnAssign) -> str:
        target = self.visit(node.target)
        if node.value:
            value = self.visit(node.value)
            op = "=" if self.config.equation_mode else "\\leftarrow"
            return f"{target} {op} {value}"
        return target

    def visit_Return(self, node: ast.Return) -> str:
        val = self.visit(node.value) if node.value else ""
        return f"\\mathbf{{return}}~{val}"

    # --- Data Structures Fallback ---
    
    def visit_Tuple(self, node: ast.Tuple, parent_precedence: Precedence = Precedence.LOWEST) -> str:
        elts = [self.visit(e, Precedence.LOWEST) for e in node.elts]
        return f"\\left({', '.join(elts)}\right)"

def latexify(code: str, config: Optional[TranspilerConfig] = None) -> str:
    tree = ast.parse(code)
    visitor = LatexifyVisitor(config)
    return visitor.visit(tree)
