"""
Generalized dependency installer and verifier for release runs.

This module enforces that both Python and system dependencies are present
before the LaTeXify pipeline executes. Python dependencies can be installed
directly inside the release virtualenv, while system tools emit precise
instructions so users can remediate quickly on their platform.
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
RELEASE_DIR = REPO_ROOT / "src" / "latexify"
DEFAULT_VENV = RELEASE_DIR / ".venv"


class DependencyInstallError(RuntimeError):
    """Raised when required dependencies are missing."""

    def __init__(self, failures: Sequence["DependencyCheckResult"]) -> None:
        self.failures = list(failures)
        lines = ["Required dependencies are missing:"]
        for result in self.failures:
            lines.append(f"- {result.name}: {result.details}")
            for hint in result.hints:
                lines.append(f"    * {hint}")
        super().__init__("\n".join(lines))


@dataclass
class DependencyCheckResult:
    name: str
    ok: bool
    details: str
    hints: List[str] = field(default_factory=list)


@dataclass
class DependencyContext:
    repo_root: Path
    release_dir: Path
    venv_path: Path
    auto_install: bool = True
    _pip_bin: Path | None = None
    _python_bin: Path | None = None

    def pip_bin(self) -> Path:
        if self._pip_bin:
            return self._pip_bin
        candidate = self._resolve_from_venv("pip")
        self._pip_bin = candidate
        return candidate

    def python_bin(self) -> Path:
        if self._python_bin:
            return self._python_bin
        candidate = self._resolve_from_venv("python")
        self._python_bin = candidate
        return candidate

    def _resolve_from_venv(self, binary: str) -> Path:
        if os.name == "nt":
            path = self.venv_path / "Scripts" / (binary + ("" if binary.endswith(".exe") else ".exe"))
        else:
            path = self.venv_path / "bin" / binary
        return path

    def install_python_package(self, requirement: str) -> None:
        pip_bin = self.pip_bin()
        if not pip_bin.exists():
            raise RuntimeError(
                f"Cannot auto-install '{requirement}' because {pip_bin} is missing. "
                "Create the virtualenv (release/.venv) first."
            )
        cmd = [str(pip_bin), "install", requirement]
        subprocess.run(cmd, check=True)

    def pip_hint(self, requirement: str) -> str:
        pip_bin = self.pip_bin()
        if pip_bin.exists():
            return f"{pip_bin} install '{requirement}'"
        return f"python -m pip install '{requirement}'  # (activate {self.venv_path})"


@dataclass
class DependencySpec:
    name: str
    description: str
    check: Callable[[DependencyContext], DependencyCheckResult]


def _resolve_venv_path() -> Path:
    env = os.environ.get("VIRTUAL_ENV")
    if env:
        return Path(env)
    return DEFAULT_VENV


def _python_module_spec(module_name: str, requirement: str, description: str) -> DependencySpec:
    def check(ctx: DependencyContext) -> DependencyCheckResult:
        if importlib.util.find_spec(module_name):
            return DependencyCheckResult(
                name=spec_name,
                ok=True,
                details=f"{module_name} importable",
            )
        if ctx.auto_install:
            try:
                ctx.install_python_package(requirement)
            except subprocess.CalledProcessError as exc:
                return DependencyCheckResult(
                    name=spec_name,
                    ok=False,
                    details=f"pip install failed for '{requirement}': {exc}",
                    hints=[ctx.pip_hint(requirement)],
                )
            if importlib.util.find_spec(module_name):
                return DependencyCheckResult(
                    name=spec_name,
                    ok=True,
                    details=f"{module_name} installed via pip",
                )
        return DependencyCheckResult(
            name=spec_name,
            ok=False,
            details=f"Python module '{module_name}' missing",
            hints=[ctx.pip_hint(requirement)],
        )

    spec_name = f"Python:{module_name}"
    return DependencySpec(name=spec_name, description=description, check=check)


def _binary_spec(binary: str, description: str, hints: Iterable[str]) -> DependencySpec:
    hints_list = list(hints)

    def check(ctx: DependencyContext) -> DependencyCheckResult:
        path = shutil.which(binary)
        if path:
            return DependencyCheckResult(
                name=spec_name,
                ok=True,
                details=f"{binary} found at {path}",
            )
        return DependencyCheckResult(
            name=spec_name,
            ok=False,
            details=f"{binary} not found on PATH",
            hints=hints_list,
        )

    spec_name = f"Binary:{binary}"
    return DependencySpec(name=spec_name, description=description, check=check)


def _tex_engine_spec() -> DependencySpec:
    binaries = ("tectonic", "latexmk", "pdflatex")
    hints = [
        "Fedora: sudo dnf install texlive-scheme-small latexmk",
        "Ubuntu/Debian: sudo apt-get install texlive-latex-extra latexmk",
        "macOS (Homebrew): brew install mactex-no-gui",
    ]

    def check(_: DependencyContext) -> DependencyCheckResult:
        available = [binary for binary in binaries if shutil.which(binary)]
        if available:
            return DependencyCheckResult(
                name="Binary:tex-engine",
                ok=True,
                details=f"Available engines: {', '.join(available)}",
            )
        return DependencyCheckResult(
            name="Binary:tex-engine",
            ok=False,
            details="No TeX engine (tectonic/latexmk/pdflatex) detected",
            hints=hints,
        )

    return DependencySpec(
        name="Binary:tex-engine",
        description="At least one TeX compiler must be installed.",
        check=check,
    )


def _poppler_hints() -> List[str]:
    return [
        "Fedora: sudo dnf install poppler-utils",
        "Ubuntu/Debian: sudo apt-get install poppler-utils",
        "macOS (Homebrew): brew install poppler",
    ]


def _optional_python_spec(module_name: str, requirement: str, description: str) -> DependencySpec:
    """
    Track optional dependencies (warn + hint, but never fail).
    """

    def check(ctx: DependencyContext) -> DependencyCheckResult:
        if importlib.util.find_spec(module_name):
            return DependencyCheckResult(
                name=spec_name,
                ok=True,
                details=f"Optional module '{module_name}' available",
            )
        return DependencyCheckResult(
            name=spec_name,
            ok=True,
            details=f"Optional module '{module_name}' missing",
            hints=[ctx.pip_hint(requirement)],
        )

    spec_name = f"OptionalPython:{module_name}"
    return DependencySpec(name=spec_name, description=description, check=check)


DEFAULT_DEPENDENCIES: List[DependencySpec] = [
    _python_module_spec("torch", "torch>=2.1.0", "GPU/CPU tensor runtime"),
    _python_module_spec("transformers", "transformers>=4.37.0", "Hugging Face model orchestration"),
    _python_module_spec("fitz", "PyMuPDF>=1.24.0", "PDF rasterization"),
    _python_module_spec("pdf2image", "pdf2image>=1.16.3", "PDF rasterization fallback"),
    _python_module_spec("PIL", "pillow>=10.0.0", "Image loading and preprocessing"),
    _python_module_spec("sympy", "sympy>=1.12", "Symbolic math checks for validation metrics"),
    _python_module_spec("psutil", "psutil>=5.9.0", "Process telemetry + watchdogs"),
    _python_module_spec("llama_cpp", "llama-cpp-python==0.2.79", "Kimi GGUF adapter"),
    _python_module_spec("duckduckgo_search", "duckduckgo-search>=5.3.0", "ResearchAgent query client"),
    _tex_engine_spec(),
    _binary_spec("kpsewhich", "TeX font helper required by typography_engine", hints=(
        "Fedora: sudo dnf install texlive-kpathsea",
        "Ubuntu/Debian: sudo apt-get install texlive-binaries",
        "macOS (Homebrew): install mactex-no-gui or BasicTeX",
    )),
    _binary_spec("gs", "Ghostscript for PDF post-processing", hints=(
        "Fedora: sudo dnf install ghostscript",
        "Ubuntu/Debian: sudo apt-get install ghostscript",
        "macOS (Homebrew): brew install ghostscript",
    )),
    _binary_spec("pdftoppm", "Poppler rasterizer used by pdf2image", hints=_poppler_hints()),
    _binary_spec("pdfinfo", "Poppler metadata extractor", hints=_poppler_hints()),
    _optional_python_spec(
        "flash_attn",
        "flash-attn --no-build-isolation",
        "Optional FlashAttention2 kernels (reduces Florence2/InternVL VRAM usage)",
    ),
]


def ensure_release_dependencies(
    *,
    specs: Sequence[DependencySpec] | None = None,
    auto_install_python: bool = True,
) -> List[DependencyCheckResult]:
    """
    Verify (and optionally install) runtime dependencies.

    Returns a list of DependencyCheckResult entries for logging. Raises
    DependencyInstallError if any dependency is still missing after remediation.
    """

    ctx = DependencyContext(
        repo_root=REPO_ROOT,
        release_dir=RELEASE_DIR,
        venv_path=_resolve_venv_path(),
        auto_install=auto_install_python,
    )
    results: List[DependencyCheckResult] = []
    failures: List[DependencyCheckResult] = []

    for spec in specs or DEFAULT_DEPENDENCIES:
        result = spec.check(ctx)
        results.append(result)
        if not result.ok:
            failures.append(result)

    if failures:
        raise DependencyInstallError(failures)
    return results
