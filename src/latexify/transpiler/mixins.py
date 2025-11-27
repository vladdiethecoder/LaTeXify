import ast
from typing import Any, List, Optional, cast
from .precedence import Precedence

class BaseMixin:
    """Type hinting helper for the main visitor."""
    def visit(self, node: ast.AST, precedence: Precedence = Precedence.LOWEST) -> str:
        raise NotImplementedError

class ArithmeticMixin:
    def visit_BinOp(self: Any, node: ast.BinOp, parent_precedence: Precedence = Precedence.LOWEST) -> str:
        # Determine current operator precedence
        op_type = type(node.op)
        if op_type == ast.Pow:
            current_precedence = Precedence.POW
            op_str = "^{{{}}}"
        elif op_type in (ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.MatMult):
            current_precedence = Precedence.MULT
        elif op_type in (ast.Add, ast.Sub):
            current_precedence = Precedence.ADD
        elif op_type in (ast.BitOr,):
            current_precedence = Precedence.BIT_OR
        elif op_type in (ast.BitXor,):
            current_precedence = Precedence.BIT_XOR
        elif op_type in (ast.BitAnd,):
            current_precedence = Precedence.BIT_AND
        elif op_type in (ast.LShift, ast.RShift):
            current_precedence = Precedence.SHIFT
        else:
            current_precedence = Precedence.LOWEST # Default fallback

        # Get operands
        left = self.visit(node.left, current_precedence)
        # For non-associative or specific ops, right side might need tighter binding?
        # e.g. a - (b - c). - is left associative. a - b - c is (a-b)-c.
        # If we have (a-b)-c, tree is BinOp(left=BinOp(a-b), right=c).
        # left visit: prec=ADD. node.left is ADD. ADD >= ADD?
        # If strictly >, (a-b) -> a-b.
        # If >=, (a-b) -> a-b.
        # Right side: if a-(b-c). node.right is SUB. ADD >= ADD?
        # If we don't parens, a-b-c != a-(b-c). So right side usually needs > prec if same op is non-associative?
        # Python is left-associative for +,-,*,/. ** is right-associative.
        
        right_precedence = current_precedence
        if op_type == ast.Pow:
            # Right associative: 2**3**4 is 2**(3**4).
            # Left side needs strict check if same op? (2**3)**4
            # If left is Pow, Precedence.POW >= Precedence.POW is True. 
            # We need parens on left if it's Pow.
            pass 
        else:
            # Left associative: 1-2-3 is (1-2)-3.
            # If right child is same op, we MUST parens it. 1-(2-3).
            # So right child context needs slightly higher precedence to force parens?
            # Or explicitly check structure.
            # A common trick: pass current_precedence + 1 (or epsilon) to right child for left-associative ops.
            pass

        # Refined logic:
        if op_type == ast.Pow:
             left = self.visit(node.left, Precedence.POW + 1) # Force parens if left is also POW
             right = self.visit(node.right, Precedence.POW)
             expr = f"{left}^{{{right}}}"
        elif isinstance(node.op, ast.Div):
            # Fraction style usually resets precedence inside numerator/denominator?
            # Or we wrap in \frac{...}{...}.
            left = self.visit(node.left, Precedence.LOWEST)
            right = self.visit(node.right, Precedence.LOWEST)
            expr = f"\\frac{{{left}}}{{{right}}}"
        elif isinstance(node.op, ast.FloorDiv):
            left = self.visit(node.left, Precedence.LOWEST)
            right = self.visit(node.right, Precedence.LOWEST)
            expr = f"\\left\\lfloor\\frac{{{left}}}{{{right}}}\\right\\rfloor"
        elif isinstance(node.op, ast.Mult):
            # Handle implicit multiplication? Explicit \cdot or nothing?
            # Standard: \cdot or space.
            right = self.visit(node.right, current_precedence)
            expr = f"{left} \\cdot {right}"
        elif isinstance(node.op, ast.MatMult):
            right = self.visit(node.right, current_precedence)
            expr = f"{left} @ {right}" # Or \times? Usually @ is preserved or specific matrix mult symbol.
        else:
            # +, -, etc.
            op_symbol = self._get_op_symbol(node.op)
            # Right side precedence bump for left-associativity
            right = self.visit(node.right, current_precedence + 1 if self._is_left_associative(node.op) else current_precedence)
            expr = f"{left} {op_symbol} {right}"

        # Wrap if needed
        if current_precedence < parent_precedence:
            return f"\\left({expr}\\right)"
        return expr

    def visit_UnaryOp(self: Any, node: ast.UnaryOp, parent_precedence: Precedence = Precedence.LOWEST) -> str:
        current_precedence = Precedence.UNARY
        operand = self.visit(node.operand, current_precedence)
        
        if isinstance(node.op, ast.UAdd):
            expr = f"+{operand}"
        elif isinstance(node.op, ast.USub):
            expr = f"-{operand}"
        elif isinstance(node.op, ast.Not):
            current_precedence = Precedence.BOOL_NOT
            expr = f"\\neg {operand}"
        elif isinstance(node.op, ast.Invert):
            expr = f"\\sim {operand}"
        else:
            expr = f"?{operand}"

        if current_precedence < parent_precedence:
            return f"\\left({expr}\\right)"
        return expr

    def _get_op_symbol(self, op: ast.operator) -> str:
        if isinstance(op, ast.Add): return "+"
        if isinstance(op, ast.Sub): return "-"
        if isinstance(op, ast.Mod): return "\\%"
        if isinstance(op, ast.BitOr): return "|"
        if isinstance(op, ast.BitXor): return "\\oplus"
        if isinstance(op, ast.BitAnd): return "\\&"
        if isinstance(op, ast.LShift): return "\\ll"
        if isinstance(op, ast.RShift): return "\\gg"
        return "?"

    def _is_left_associative(self, op: ast.operator) -> bool:
        return not isinstance(op, ast.Pow)


class ControlFlowMixin:
    def visit_Compare(self: Any, node: ast.Compare, parent_precedence: Precedence = Precedence.LOWEST) -> str:
        current_precedence = Precedence.COMPARE
        left = self.visit(node.left, current_precedence)
        
        parts = [left]
        for op, comparator in zip(node.ops, node.comparators):
            op_symbol = self._get_cmp_symbol(op)
            right = self.visit(comparator, current_precedence + 1) # Chained comps are effectively left-associative-ish logic
            parts.append(f"{op_symbol} {right}")
            
        expr = " ".join(parts)
        if current_precedence < parent_precedence:
            return f"\\left({expr}\\right)"
        return expr

    def visit_IfExp(self: Any, node: ast.IfExp, parent_precedence: Precedence = Precedence.LOWEST) -> str:
        # Python: body if test else orelse
        # LaTeX cases: \begin{cases} body & \text{if } test \\ orelse & \text{otherwise} \end{cases}
        # Or inline: body \text{ if } test \text{ else } orelse (less mathy)
        # Let's use cases for robustness.
        # Note: Cases environment usually needs to be top-level or inside math.
        # But IfExp is an expression.
        # Standard inline notation: \begin{cases} ... \end{cases} works.
        
        body = self.visit(node.body, Precedence.LOWEST)
        test = self.visit(node.test, Precedence.LOWEST)
        orelse = self.visit(node.orelse, Precedence.LOWEST)
        
        return f"\\begin{{cases}} {body} & \\text{{if }} {test} \\ \\ {orelse} & \\text{{otherwise}} \\end{{cases}}"

    def _get_cmp_symbol(self, op: ast.cmpop) -> str:
        if isinstance(op, ast.Eq): return "="
        if isinstance(op, ast.NotEq): return "\\neq"
        if isinstance(op, ast.Lt): return "<"
        if isinstance(op, ast.LtE): return "\\leq"
        if isinstance(op, ast.Gt): return ">"
        if isinstance(op, ast.GtE): return "\\geq"
        if isinstance(op, ast.Is): return "\\equiv" # Approximation
        if isinstance(op, ast.IsNot): return "\\not\\equiv"
        if isinstance(op, ast.In): return "\\in"
        if isinstance(op, ast.NotIn): return "\\notin"
        return "?"

class DataStructureMixin:
    def visit_List(self: Any, node: ast.List, parent_precedence: Precedence = Precedence.LOWEST) -> str:
        # Check if it's a matrix (list of lists of equal length)
        if self._is_matrix(node):
            return self._render_matrix(node)
        
        # Standard list
        elts = [self.visit(e, Precedence.LOWEST) for e in node.elts]
        return f"\\left[{', '.join(elts)}\\right]"

    def visit_Tuple(self: Any, node: ast.Tuple, parent_precedence: Precedence = Precedence.LOWEST) -> str:
        elts = [self.visit(e, Precedence.LOWEST) for e in node.elts]
        return f"\\left({', '.join(elts)}\\right)"

    def _is_matrix(self, node: ast.List) -> bool:
        if not node.elts:
            return False
        first = node.elts[0]
        if not isinstance(first, ast.List):
            return False
        width = len(first.elts)
        for elt in node.elts[1:]:
            if not isinstance(elt, ast.List) or len(elt.elts) != width:
                return False
        return True

    def _render_matrix(self: Any, node: ast.List) -> str:
        rows = []
        for row_node in node.elts:
            row_node = cast(ast.List, row_node) # Safe per _is_matrix
            cols = [self.visit(c, Precedence.LOWEST) for c in row_node.elts]
            rows.append(" & ".join(cols))
        
        body = " \\ ".join(rows)
        return f"\\begin{{bmatrix}} {body} \\end{{bmatrix}}"

class FunctionMixin:
    def visit_Call(self: Any, node: ast.Call, parent_precedence: Precedence = Precedence.LOWEST) -> str:
        func_name = ""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr
        
        # Smart Functions
        if func_name == "sqrt":
            if len(node.args) == 1:
                arg = self.visit(node.args[0], Precedence.LOWEST)
                return f"\\sqrt{{{arg}}}"
        elif func_name == "sum":
            # Heuristic: sum(iterable) -> \sum. Hard to map range/iterator without semantics.
            # If args has 1, just \sum(arg).
            args_str = ", ".join([self.visit(a, Precedence.LOWEST) for a in node.args])
            return f"\\sum\\left({args_str}\right)"
        elif func_name == "prod":
            args_str = ", ".join([self.visit(a, Precedence.LOWEST) for a in node.args])
            return f"\\prod\\left({args_str}\right)"
        
        # Default Call
        # Render func name as \text{name} or \mathrm? or just identifier logic
        # self.visit(node.func) will trigger identifier mapper.
        
        func_latex = self.visit(node.func, Precedence.ATOM)
        args_str = ", ".join([self.visit(a, Precedence.LOWEST) for a in node.args])
        return f"{func_latex}\\left({args_str}\right)"
