from __future__ import annotations

from ..utils import escape_tex, parse_table_rows, text_to_paragraphs


def figure_snippet(asset_path: str, title: str, label: str) -> str:
    asset = asset_path or "assets/figure-placeholder.pdf"
    return "\n".join([
        "\\begin{figure}[ht]",
        "  \\centering",
        f"  \\includegraphics[width=0.85\\linewidth]{{{asset}}}",
        f"  \\caption{{{escape_tex(title)}}}",
        f"  \\label{{fig:{label}}}",
        "\\end{figure}",
        "",
    ])


def table_snippet(raw_text: str, title: str, label: str) -> str:
    rows = parse_table_rows(raw_text)
    if not rows:
        return "% TODO: add table content\n"
    width = max(len(r) for r in rows)
    fmt = " ".join(["l"] * width)
    header = rows[0]
    body = rows[1:] or [[""] * width]
    lines = [
        "\\begin{table}[ht]",
        "  \\centering",
        f"  \\caption{{{escape_tex(title)}}}",
        f"  \\label{{tab:{label}}}",
        f"  \\begin{{tabular}}{{{fmt}}}",
        "    \\toprule",
        "    " + " & ".join(header) + r" \\",
        "    \\midrule",
    ]
    for row in body:
        padded = row + [""] * (width - len(row))
        lines.append("    " + " & ".join(escape_tex(cell) for cell in padded) + r" \\")
    lines += [
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
        "",
    ]
    return "\n".join(lines)


def math_snippet(raw_text: str, title: str, label: str) -> str:
    math_text = (raw_text or "").strip()
    if not math_text.startswith("\\") and not math_text.startswith("$"):
        math_text = "$" + math_text + "$"
    return "\n".join([
        f"% {escape_tex(title)}",
        math_text,
        f"% label: eq:{label}",
        "",
    ])


def body_snippet(
    section_command: str,
    title: str,
    label: str,
    source_text: str,
    *,
    multicolumn: bool = False,
) -> str:
    body = text_to_paragraphs(source_text) or "% TODO: add content\n"
    lines = [
        f"{section_command}{{{escape_tex(title)}}}",
        f"\\label{{sec:{label}}}",
    ]
    if multicolumn:
        lines.append("\\begin{multicols}{2}")
        lines.append(body.rstrip("\n"))
        lines.append("\\end{multicols}")
    else:
        lines.append(body.rstrip("\n"))
    lines.append("")
    return "\n".join(lines)
