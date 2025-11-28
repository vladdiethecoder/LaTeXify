import ast
from typing import Any, List, Optional, cast, Union
from .precedence import Precedence

class BaseMixin:
    """
    Abstract base for all mixins.
    """
    def visit(self, node: ast.AST, precedence: Precedence = Precedence.LOWEST) -> str:
        raise NotImplementedError

class LinearAlgebraMixin(BaseMixin):
    """
    Mutation B: Handles Matrix Inference.
    """
    def visit_List(self: Any, node: ast.List, parent_precedence: Precedence = Precedence.LOWEST) -> str:
        # Matrix Inference: Check if it's a list of lists with uniform length
        if self._is_matrix(node):
            return self._render_matrix(node)
        
        # Standard Vector/List
        elts = [self.visit(e, Precedence.LOWEST) for e in node.elts]
        return f"\left[{', '.join(elts)}\right]"

    def visit_Subscript(self: Any, node: ast.Subscript, parent_precedence: Precedence = Precedence.LOWEST) -> str:
        # Mutation A: Handling Slices and Indexing
        # x[i] -> x_{i}
        # x[i, j] -> x_{i, j}
        
        value = self.visit(node.value, Precedence.ATOM)
        
        # Handle Index
        if isinstance(node.slice, ast.Constant) or isinstance(node.slice, ast.Name):
            # Simple index
            sub = self.visit(node.slice, Precedence.LOWEST)
            return f"{value}_{{{sub}}}"
        elif isinstance(node.slice, ast.Tuple):
            # Multi-index x[i, j]
            dims = [self.visit(d, Precedence.LOWEST) for d in node.slice.elts]
            return f"{value}_{{{', '.join(dims)}}}"
        elif isinstance(node.slice, ast.Slice):
            # Slice x[0:N] -> x_{0 \dots N} or similar?
            # Heuristic: start:stop -> start \dots stop
            lower = self.visit(node.slice.lower, Precedence.LOWEST) if node.slice.lower else ""
            upper = self.visit(node.slice.upper, Precedence.LOWEST) if node.slice.upper else ""
            return f"{value}_{{{lower} : {upper}}}"
            
        # Fallback
        sub = self.visit(node.slice, Precedence.LOWEST)
        return f"{value}_{{{sub}}}"

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
            row_node = cast(ast.List, row_node)
            cols = [self.visit(c, Precedence.LOWEST) for c in row_node.elts]
            rows.append(" & ".join(cols))
        body = " \\ ".join(rows)
        return f"\begin{{bmatrix}} {body} \end{{bmatrix}}"

class CalculusMixin(BaseMixin):
    """
    Mutation B: Handles Smart Functions and Integrals.
    """
    def visit_Call(self: Any, node: ast.Call, parent_precedence: Precedence = Precedence.LOWEST) -> str:
        func_name = ""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr

        # Smart Integral: sum(expr * dx) -> \int expr \, dx
        if func_name == "sum" and len(node.args) == 1:
            arg = node.args[0]
            if isinstance(arg, ast.BinOp) and isinstance(arg.op, ast.Mult):
                # Check for dx / dt / d_something
                right_id = getattr(arg.right, "id", "")
                if right_id.startswith("d") and len(right_id) > 1:
                    integrand = self.visit(arg.left, Precedence.MULT) # Tighter binding than + 
                    differential = self.visit(arg.right, Precedence.ATOM)
                    return f"\int {integrand} \, {differential}"

        # Standard Mappings
        if func_name == "sqrt":
            arg = self.visit(node.args[0], Precedence.LOWEST)
            return f"\sqrt{{{arg}}}"
        if func_name == "sum":
            args_str = ", ".join([self.visit(a, Precedence.LOWEST) for a in node.args])
            return f"\sum\left({args_str}\right)"
        if func_name == "prod":
            args_str = ", ".join([self.visit(a, Precedence.LOWEST) for a in node.args])
            return f"\prod\left({args_str}\right)"
        if func_name == "len":
            arg = self.visit(node.args[0], Precedence.LOWEST)
            return f"|{arg}|"
        if func_name == "abs":
            arg = self.visit(node.args[0], Precedence.LOWEST)
            return f"\left|{arg}\right|"

        # Default Call
        func_latex = self.visit(node.func, Precedence.ATOM)
        args_str = ", ".join([self.visit(a, Precedence.LOWEST) for a in node.args])
        return f"{func_latex}\left({args_str}\right)"

class AlgorithmicMixin(BaseMixin):
    """
    Mutation A: Handles Control Flow (Cases).
    """
    def visit_IfExp(self: Any, node: ast.IfExp, parent_precedence: Precedence = Precedence.LOWEST) -> str:
        # body if test else orelse -> cases
        body = self.visit(node.body, Precedence.LOWEST)
        test = self.visit(node.test, Precedence.LOWEST)
        orelse = self.visit(node.orelse, Precedence.LOWEST)
        return f"\begin{{cases}} {body} & \text{{if }} {test} \\ {orelse} & \text{{otherwise}} \end{{cases}}"

    def visit_BoolOp(self: Any, node: ast.BoolOp, parent_precedence: Precedence = Precedence.LOWEST) -> str:
        op_symbol = "\\lor" if isinstance(node.op, ast.Or) else "\\land"
        current_precedence = Precedence.BOOL_OR if isinstance(node.op, ast.Or) else Precedence.BOOL_AND
        
        values = [self.visit(v, current_precedence) for v in node.values]
        expr = f" {op_symbol} ".join(values)
        
        if current_precedence < parent_precedence:
            return f"\left({expr}\right)"
        return expr

class ArithmeticMixin(BaseMixin):
    def visit_BinOp(self: Any, node: ast.BinOp, parent_precedence: Precedence = Precedence.LOWEST) -> str:
        op_type = type(node.op)
        
        # Precedence Mapping
        if op_type == ast.Pow:
            current = Precedence.POW
        elif op_type in (ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.MatMult):
            current = Precedence.MULT
        elif op_type in (ast.Add, ast.Sub):
            current = Precedence.ADD
        elif op_type in (ast.BitOr,):
            current = Precedence.BIT_OR
        elif op_type in (ast.BitXor,):
            current = Precedence.BIT_XOR
        elif op_type in (ast.BitAnd,):
            current = Precedence.BIT_AND
        else:
            current = Precedence.LOWEST

        # Special Handling for Division (Fraction)
        if isinstance(node.op, ast.Div):
            left = self.visit(node.left, Precedence.LOWEST)
            right = self.visit(node.right, Precedence.LOWEST)
            return f"\\frac{{{left}}}{{{right}}}"
        
        # Special Handling for Power
        if isinstance(node.op, ast.Pow):
            left = self.visit(node.left, Precedence.POW + 1) # Force parens if base is also pow (though rare)
            right = self.visit(node.right, Precedence.LOWEST) # Exponent can be loose
            return f"{left}^{{{right}}}"

        # General Binary Op
        symbol_map = {
            ast.Add: "+", ast.Sub: "-", ast.Mult: "\\cdot", ast.MatMult: "@",
            ast.Mod: "\\%", ast.BitOr: "|", ast.BitXor: "\\oplus", ast.BitAnd: "\\&"
        }
        symbol = symbol_map.get(op_type, "?")
        
        # Associativity check: Python arithmetic is Left Associative
        # If right child has SAME precedence, it effectively binds looser than this op in the chain (a-b)-c.
        # Wait, (a-b)-c. Root is -. Left is -. Right is c.
        # a-(b-c). Root is -. Left is a. Right is -.
        # So if right child is same precedence, we need parens.
        
        left = self.visit(node.left, current)
        right = self.visit(node.right, current + 1) # Tighten right requirement
        
        expr = f"{left} {symbol} {right}"
        if current < parent_precedence:
            return f"\left({expr}\right)"
        return expr

    def visit_UnaryOp(self: Any, node: ast.UnaryOp, parent_precedence: Precedence = Precedence.LOWEST) -> str:
        current = Precedence.UNARY
        operand = self.visit(node.operand, current)
        
        if isinstance(node.op, ast.USub):
            expr = f"-{operand}"
        elif isinstance(node.op, ast.UAdd):
            expr = f"{operand}"
        elif isinstance(node.op, ast.Not):
            current = Precedence.BOOL_NOT
            expr = f"\\neg {operand}"
        else:
            expr = f"?{operand}"
            
        if current < parent_precedence:
            return f"\left({expr}\right)"
        return expr

    def visit_Compare(self: Any, node: ast.Compare, parent_precedence: Precedence = Precedence.LOWEST) -> str:
        current = Precedence.COMPARE
        left = self.visit(node.left, current)
        
        parts = [left]
        for op, comparator in zip(node.ops, node.comparators):
            op_symbol = self._get_cmp_symbol(op)
            right = self.visit(comparator, current + 1)
            parts.append(f"{op_symbol} {right}")
            
        expr = " ".join(parts)
        if current < parent_precedence:
            return f"\left({expr}\right)"
        return expr

    def _get_cmp_symbol(self, op: ast.cmpop) -> str:
        mapping = {
            ast.Eq: "=", ast.NotEq: "\\neq", ast.Lt: "<", ast.LtE: "\\leq",
            ast.Gt: ">", ast.GtE: "\\geq", ast.Is: "\\equiv", ast.IsNot: "\\not\\equiv",
            ast.In: "\\in", ast.NotIn: "\\notin"
        }
        return mapping.get(type(op), "?")
