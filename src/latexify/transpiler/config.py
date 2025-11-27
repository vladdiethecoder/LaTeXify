from dataclasses import dataclass, field
from typing import Dict

@dataclass
class TranspilerConfig:
    equation_mode: bool = False # If False, use algorithmic assignment (leftarrow)
    identifiers: Dict[str, str] = field(default_factory=dict)
    reduce_assignments: bool = False
