from enum import IntEnum, auto

class Precedence(IntEnum):
    """
    Operator precedence levels mirroring Python 3.13 grammar.
    Higher value means tighter binding.
    """
    LOWEST = 0
    YIELD = auto()
    LAMBDA = auto()
    IF_EXP = auto()  # if-else expression
    BOOL_OR = auto()
    BOOL_AND = auto()
    BOOL_NOT = auto()
    COMPARE = auto() # in, not in, is, is not, <, <=, >, >=, !=, ==
    BIT_OR = auto()
    BIT_XOR = auto()
    BIT_AND = auto()
    SHIFT = auto()
    ADD = auto()     # +, -
    MULT = auto()    # *, @, /, //, %
    UNARY = auto()   # +x, -x, ~x
    POW = auto()     # **
    ATOM = auto()    # literals, identifiers, parenthesized, calls, slicing
