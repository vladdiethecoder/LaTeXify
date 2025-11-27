# LaTeXify Migration Guide (2025 Refactoring)

## Overview
This update consolidates LaTeXify into a production-grade library supporting both PDF-to-LaTeX conversion (legacy pipeline) and a robust Python-to-LaTeX transpiler (new architecture).

## Architectural Changes

### 1. Modular Visitor Architecture
The monolithic `LatexifyVisitor` has been decomposed into mixins located in `src/latexify/transpiler/mixins.py`:
- `ArithmeticMixin`: Handles mathematical operators (`+`, `-`, `*`, `/`, `**`) with strict precedence support.
- `ControlFlowMixin`: Handles comparisons and conditionals (`if-else` -> `cases`).
- `DataStructureMixin`: Handles lists, tuples, and **matrices** (lists of lists).
- `FunctionMixin`: Handles smart function mapping (`sqrt`, `sum`, `prod`).

### 2. Strict Operator Precedence
A new `Precedence` enum (`src/latexify/transpiler/precedence.py`) mirrors the Python 3.13 grammar. The visitor now passes `parent_precedence` down the tree to conditionally render parentheses `\left( ... \right)` only when strictly necessary.

### 3. Symbol Disambiguation
The `IdentifierMapper` (`src/latexify/transpiler/mapper.py`) automatically handles:
- Greek letters (e.g., `alpha` -> `\alpha`).
- Subscripts (e.g., `x_i` -> `x_{i}`).

## New Features

### Matrix Support
Lists of lists are now automatically detected and rendered as `\begin{bmatrix}` environments.
```python
[[1, 2], [3, 4]]  # Renders as a 2x2 matrix
```

### Algorithmic vs. Equation Mode
Configure the output style via `TranspilerConfig`:
```python
config = TranspilerConfig(equation_mode=True) # Uses "=" for assignments
config = TranspilerConfig(equation_mode=False) # Uses "\\leftarrow" (Algorithmic)
```

## Developer Workflow

### Dependency Management (uv)
We have migrated to `uv` for fast dependency resolution.
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv
uv pip install -e ".[dev]"
```

### Linting & Typing
- **Ruff**: Enforced for linting (replacing Flake8/Isort).
- **MyPy**: Strict mode enabled for type safety.

## Build System
The project now uses `hatchling` backend defined in `pyproject.toml`.
```bash
uv run python -m build
```

```