from __future__ import annotations

from textwrap import dedent
from typing import Optional

from .archive import Archive
from .knowledge_graph import KnowledgeGraph


def render_report(archive: Archive, graph: Optional[KnowledgeGraph] = None) -> str:
    """
    Produce a minimal LaTeX report summarizing archive contents.
    """
    rows = []
    for agent in archive.get_top(k=len(archive)):
        rows.append(f"{agent.version_id} & {agent.parent_id or '-'} & {agent.metrics.score:.3f} \\\\")

    edges = ""
    if graph:
        edges = "\\\\\n".join(
            f"{e['src']} \\rightarrow {e['dst']} ({e['relation']})" for e in graph.edges
        )

    best_score = archive.get_top(1)[0].metrics.score if len(archive) else 0.0
    archive_rows = "\n".join(rows)
    return (
        dedent(
            f"""
            \\documentclass{{article}}
            \\usepackage{{booktabs}}
            \\begin{document}
            \\section*{{Self-Improvement Summary}}
            Best score: {best_score:.3f}

            \\subsection*{{Archive}}
            \\begin{tabular}{{lll}}
            \\toprule
            Agent & Parent & Score\\\\
            \\midrule
            {archive_rows}
            \\bottomrule
            \\end{tabular}

            \\subsection*{{Lineage}}
            {edges or 'No edges recorded.'}
            \\end{document}
            """
        )
        .strip()
    )
