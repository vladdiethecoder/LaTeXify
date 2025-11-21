# LaTeXify Agents & Neuro-Symbolic 2.0

## Agents

*   **ParsingAgent**: Ingestion and OCR.
*   **PlannerAgent**: Semantic structure planning.
*   **LayoutAgent**: Page layout analysis.
*   **StructureGraphAgent**: Dependency modeling.
*   **RetrievalAgent**: RAG context.
*   **LaTeXSynthAgent**: Core synthesis.
*   **AssemblyAgent**: Compilation.
*   **RefinementAgent**: Iterative fixing.

## Neuro-Symbolic 2.0 Features

### 1. Neuro-Symbolic Autoformalization ("Truth" Loop)
*   **Module**: `formal_verification.py`
*   **Logic**: Translates LaTeX math to Lean 4 to verify logical consistency. Flags "hallucinated math" that compiles but is mathematically invalid.

### 2. Direct Generative Vectorization
*   **Module**: `vectorization.py`
*   **Logic**: Uses VLMs (Qwen-VL) to generate TikZ code directly from raster crops, replacing brittle bitmap embedding.

### 3. Inverse-Graphics Layout Optimization
*   **Module**: `visual_regression.py` (Visual Critic)
*   **Logic**: Compares generated PDF with source PDF visually to suggest layout fixes (spacing, alignment) rather than just text diffs.

### 4. Semantic Graph
*   **Module**: `structure_graph.py`
*   **Logic**: Models logical dependencies (Definition -> Usage) to ensure notation consistency across the document.

### 5. Active Gaze Alignment
*   **Module**: `ingestion.py` / `chunking_engine.py`
*   **Logic**: Preserves bounding box metadata throughout the pipeline to enable reverse-mapping from LaTeX to source pixels.