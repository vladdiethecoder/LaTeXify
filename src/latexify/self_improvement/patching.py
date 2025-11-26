from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional

LOGGER = logging.getLogger(__name__)


@dataclass
class PatchOperation:
    """A targeted search-and-replace operation."""

    file_path: Path
    search: str
    replace: str


@dataclass
class PatchProposal:
    candidate_id: str
    strategy: str
    rationale: str
    operations: List[PatchOperation] = field(default_factory=list)
    target_tests: List[str] = field(default_factory=list)


@dataclass
class PatchApplyResult:
    success: bool
    modified_files: List[Path]
    errors: List[str]


def apply_patch_proposal(proposal: PatchProposal, root: Path) -> PatchApplyResult:
    """
    Apply a set of targeted replacements. This intentionally avoids whole-file rewrites.
    """
    modified: List[Path] = []
    errors: List[str] = []
    for op in proposal.operations:
        target = (root / op.file_path).resolve()
        if not target.exists():
            errors.append(f"{target} does not exist")
            continue
        try:
            content = target.read_text(encoding="utf-8")
            if op.search not in content:
                errors.append(f"Search text not found in {target}: {op.search[:80]}")
                continue
            new_content = content.replace(op.search, op.replace, 1)
            target.write_text(new_content, encoding="utf-8")
            modified.append(target)
        except Exception as exc:
            errors.append(f"Failed to patch {target}: {exc}")
    return PatchApplyResult(success=len(errors) == 0, modified_files=modified, errors=errors)
