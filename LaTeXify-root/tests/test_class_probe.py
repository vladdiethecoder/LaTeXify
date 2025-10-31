# -*- coding: utf-8 -*-
import json
from pathlib import Path

from scripts.class_probe import parse_issues, suggest_fixes, ClassProfile

def test_parse_missing_class_and_microtype():
    text = "LaTeX Error: File `lix_article.cls' not found.\n" \
           "LaTeX Error: File `microtype.sty' not found.\n"
    issues = parse_issues(text)
    assert {"type": "missing_class", "name": "lix_article.cls"} in issues
    assert {"type": "missing_package", "name": "microtype.sty"} in issues

def test_parse_tcrm_and_mf():
    text = "Font TS1/cmr/m/n/10.95=tcrm1095 not loadable: Metric (TFM) file not found.\n" \
           "/usr/bin/mktextfm: line 126: mf: command not found\n"
    issues = parse_issues(text)
    kinds = {i["type"] for i in issues}
    assert "missing_ec_fonts" in kinds
    assert "missing_metafont" in kinds

def test_suggest_fedora_hints():
    profile = ClassProfile(name="lix_article", fallback="scrartcl", packages=["enumitem","geometry","microtype"])
    issues = [
        {"type": "missing_class", "name": "lix_article.cls"},
        {"type": "missing_package", "name": "microtype.sty"},
        {"type": "missing_ec_fonts", "name": "tcrm/TS1"},
    ]
    hints = suggest_fixes(profile, issues)
    fed = "\n".join(hints["fedora"])
    assert "texlive-microtype" in fed
    assert "texlive-ec" in fed
    # ensure fallback advice is present
    notes = json.dumps(hints["notes"])
    assert "scrartcl" in notes
