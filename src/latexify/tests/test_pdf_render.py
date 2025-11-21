from latexify.scripts import pdf_render


def test_format_prompt_with_sections_and_bullets():
    prompt = "# Title\n\nIntro paragraph.\n- item one\n- item two"
    latex = pdf_render.format_prompt_to_latex(prompt)
    assert "\\section*" in latex
    assert "\\begin{itemize}" in latex
    assert "item one" in latex


def test_format_prompt_converts_markdown_table():
    prompt = (
        "| Col A | Col B |\n"
        "|-------|-------|\n"
        "| 1     | 2     |\n"
        "| 3     | 4     |"
    )
    latex = pdf_render.format_prompt_to_latex(prompt)
    assert "\\begin{table}" in latex
    assert "1 & 2" in latex
