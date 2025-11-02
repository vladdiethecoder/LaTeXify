# tests/test_build_latex_kb.py
from __future__ import annotations
from latexify.kb.build_latex_kb import (
    parse_texse_html,
    parse_overleaf_html,
    parse_wikibooks_html,
    KBRecord,
)

TEXSE_HTML = """
<html><head><title>How to change the font size in LaTeX - TeX</title></head>
<body>
<h1>How to change the font size in LaTeX</h1>
<div id="question"><div class="js-post-body">I need to change font size globally.</div></div>
<div class="answer accepted-answer"><div class="js-post-body">
Use <code>\\documentclass[12pt]{article}</code> or packages like <code>anyfontsize</code>.
</div></div>
<div class="post-taglist"><a class="post-tag">fontsize</a><a class="post-tag">documentclass</a></div>
</body></html>
"""

OVERLEAF_HTML = """
<html><head><title>Tables - Overleaf, Online LaTeX Editor</title></head>
<body>
<article>
<h1>Tables</h1>
<p>Use the <code>tabular</code> environment. For booktabs rules, use \\toprule, \\midrule, \\bottomrule.</p>
<pre><code>\\begin{tabular}{ll}</code></pre>
</article>
</body></html>
"""

WIKIBOOKS_HTML = """
<html>
<head><title>LaTeX/Mathematics - Wikibooks</title></head>
<body>
<h1 id="firstHeading">LaTeX/Mathematics</h1>
<div id="mw-content-text">
  <p>Inline math: <code>\\( a^2 + b^2 = c^2 \\)</code>; display math uses <code>\\[ ... \\]</code>.</p>
</div>
</body></html>
"""

def _assert_record(rec: KBRecord):
    assert rec.id.startswith("sha1:")
    assert rec.title
    assert rec.question
    assert isinstance(rec.answer, str)
    assert isinstance(rec.code_blocks, list)

def test_texse_parser_minimal():
    rec = parse_texse_html("https://tex.stackexchange.com/q/123", TEXSE_HTML)
    _assert_record(rec)
    assert "fontsize" in " ".join(rec.tags).lower()

def test_overleaf_parser_minimal():
    rec = parse_overleaf_html("https://www.overleaf.com/learn/latex/Tables", OVERLEAF_HTML)
    _assert_record(rec)
    assert "tables" in rec.title.lower()
    assert any("tabular" in c.lower() for c in rec.code_blocks)

def test_wikibooks_parser_minimal():
    rec = parse_wikibooks_html("https://en.wikibooks.org/wiki/LaTeX/Mathematics", WIKIBOOKS_HTML)
    _assert_record(rec)
    assert "mathematics" in rec.title.lower()
