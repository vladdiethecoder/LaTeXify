"""Helper entrypoints exposed under ``latexify.scripts``.

Tests and training utilities import ``latexify.scripts.pdf_render``; this
package module provides a thin shim over the top-level ``scripts`` directory
shipped with the repository.
"""

from . import pdf_render

__all__ = ["pdf_render"]

