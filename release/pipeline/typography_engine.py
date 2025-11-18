"""Rule-driven typography helpers for LaTeX preambles."""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Sequence


PackageSpec = Dict[str, str | None]


FONT_LIBRARY: Dict[str, List[PackageSpec]] = {
    "newtx": [
        {"package": "fontenc", "options": "T1"},
        {"package": "newtxtext", "options": None},
        {"package": "newtxmath", "options": None},
    ],
    "mathpazo": [
        {"package": "fontenc", "options": "T1"},
        {"package": "mathpazo", "options": None},
        {"package": "eulervm", "options": None},
    ],
    "libertine": [
        {"package": "fontenc", "options": "T1"},
        {"package": "libertine", "options": None},
        {"package": "newtxmath", "options": None},
    ],
}


EXTRA_PACKAGE_LIBRARY: Dict[str, PackageSpec] = {
    "csquotes": {"package": "csquotes", "options": None},
    "ragged2e": {"package": "ragged2e", "options": None},
}

LOGGER = logging.getLogger(__name__)
ENABLE_MICROTYPE = os.environ.get("LATEXIFY_ENABLE_MICROTYPE", "0") == "1"
ENABLE_SETSPACE = os.environ.get("LATEXIFY_ENABLE_SETSPACE", "0") == "1"
_PACKAGE_AVAILABILITY_CACHE: Dict[str, bool] = {}


def _package_available(name: str) -> bool:
    if not name:
        return False
    cached = _PACKAGE_AVAILABILITY_CACHE.get(name)
    if cached is not None:
        return cached
    kpse = shutil.which("kpsewhich")
    if not kpse:
        _PACKAGE_AVAILABILITY_CACHE[name] = False
        return False
    try:
        result = subprocess.run([kpse, f"{name}.sty"], capture_output=True, text=True, check=False)
        available = bool(result.stdout.strip())
    except Exception:
        available = False
    _PACKAGE_AVAILABILITY_CACHE[name] = available
    return available


@dataclass(frozen=True)
class TypographicProfile:
    """Capture the stylistic intent for a group of document classes."""

    name: str
    description: str
    preferred_classes: Sequence[str] = field(default_factory=tuple)
    font_stack: Sequence[str] = field(default_factory=lambda: ("newtx",))
    hyphenation_language: str = "english"
    line_stretch: float = 1.05
    parskip: str = "0.55em"
    parindent: str = "1.2em"
    emergency_stretch: str = "3em"
    tolerance: int = 1200
    hyphen_penalty: int = 300
    just_commands: Sequence[str] = field(default_factory=lambda: ("\\frenchspacing",))
    extras: Sequence[str] = field(default_factory=tuple)
    microtype_options: str = "final,protrusion=true,expansion=true"
    tracking_preset: str | None = "tracking=true"
    left_hyphen_min: int = 2
    right_hyphen_min: int = 3


@dataclass
class TypographyDirectives:
    """Accumulated preamble changes decided by a set of rules."""

    profile_name: str
    packages: List[PackageSpec] = field(default_factory=list)
    pre_document_commands: List[str] = field(default_factory=list)
    _command_seen: set[str] = field(default_factory=set, init=False, repr=False)

    def add_package(self, name: str, *, options: str | None = None) -> None:
        if not name:
            return
        self.packages.append({"package": name, "options": options})

    def add_packages(self, packages: Sequence[PackageSpec]) -> None:
        for pkg in packages:
            name = pkg.get("package")
            if not name:
                continue
            self.add_package(name, options=pkg.get("options"))

    def add_command(self, command: str) -> None:
        normalized = (command or "").strip()
        if not normalized or normalized in self._command_seen:
            return
        self._command_seen.add(normalized)
        self.pre_document_commands.append(normalized)


@dataclass(frozen=True)
class RuleContext:
    document_class: str
    profile: TypographicProfile
    options: Dict[str, object]


class BaseTypographyRule:
    """Base helper for declarative rules."""

    name = "base"

    def applies(self, _: RuleContext) -> bool:  # pragma: no cover - configurable hook
        return True

    def apply(self, ctx: RuleContext, directives: TypographyDirectives) -> None:  # pragma: no cover - interface only
        raise NotImplementedError


def _format_float(value: float) -> str:
    formatted = f"{value:.3f}".rstrip("0").rstrip(".")
    return formatted or "1"


def _coerce_float(value: object, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive fallback
        return fallback


def _coerce_int(value: object, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive fallback
        return fallback


class FontStackRule(BaseTypographyRule):
    name = "font-stack"

    def apply(self, ctx: RuleContext, directives: TypographyDirectives) -> None:
        stack = ctx.profile.font_stack or ("newtx",)
        for key in stack:
            packages = FONT_LIBRARY.get(key)
            if not packages:
                continue
            directives.add_packages(packages)


class ExtraPackageRule(BaseTypographyRule):
    name = "extras"

    def apply(self, ctx: RuleContext, directives: TypographyDirectives) -> None:
        for extra in ctx.profile.extras:
            spec = EXTRA_PACKAGE_LIBRARY.get(extra)
            if not spec:
                continue
            pkg_name = spec.get("package")
            if pkg_name and not _package_available(pkg_name):
                LOGGER.info("Skipping LaTeX package %s (not found via kpsewhich).", pkg_name)
                continue
            directives.add_package(pkg_name, options=spec.get("options"))


class MicrotypeRule(BaseTypographyRule):
    name = "microtype"

    def apply(self, ctx: RuleContext, directives: TypographyDirectives) -> None:
        if not ENABLE_MICROTYPE:
            return
        if not _package_available("microtype"):
            LOGGER.info("Skipping LaTeX package microtype (not found via kpsewhich).")
            return
        options = ctx.options.get("microtype_options", ctx.profile.microtype_options)
        directives.add_package("microtype", options=str(options))
        setup = ctx.options.get("microtype_setup") or ctx.profile.tracking_preset
        if setup:
            directives.add_command(f"\\microtypesetup{{{setup}}}")


class HyphenationRule(BaseTypographyRule):
    name = "hyphenation"

    def apply(self, ctx: RuleContext, directives: TypographyDirectives) -> None:
        language = str(ctx.options.get("language", ctx.profile.hyphenation_language))
        directives.add_package("babel", options=language)
        directives.add_command(f"\\lefthyphenmin={ctx.profile.left_hyphen_min}")
        directives.add_command(f"\\righthyphenmin={ctx.profile.right_hyphen_min}")
        tolerance = _coerce_int(ctx.options.get("tolerance", ctx.profile.tolerance), ctx.profile.tolerance)
        hyphen_penalty = _coerce_int(
            ctx.options.get("hyphen_penalty", ctx.profile.hyphen_penalty), ctx.profile.hyphen_penalty
        )
        directives.add_command(f"\\tolerance={tolerance}")
        directives.add_command(f"\\hyphenpenalty={hyphen_penalty}")
        directives.add_command(f"\\exhyphenpenalty={hyphen_penalty}")


class SpacingRule(BaseTypographyRule):
    name = "spacing"

    def apply(self, ctx: RuleContext, directives: TypographyDirectives) -> None:
        stretch = _coerce_float(ctx.options.get("line_stretch", ctx.profile.line_stretch), ctx.profile.line_stretch)
        if ENABLE_SETSPACE and _package_available("setspace"):
            directives.add_package("setspace")
            directives.add_command(f"\\setstretch{{{_format_float(stretch)}}}")
        else:
            directives.add_command(f"\\linespread{{{_format_float(stretch)}}}")


class ParagraphRule(BaseTypographyRule):
    name = "paragraph"

    def apply(self, ctx: RuleContext, directives: TypographyDirectives) -> None:
        parskip = str(ctx.options.get("parskip", ctx.profile.parskip))
        parindent = str(ctx.options.get("parindent", ctx.profile.parindent))
        emergency = str(ctx.options.get("emergency_stretch", ctx.profile.emergency_stretch))
        directives.add_command(f"\\setlength{{\\parskip}}{{{parskip}}}")
        directives.add_command(f"\\setlength{{\\parindent}}{{{parindent}}}")
        directives.add_command("\\setlength{\\parfillskip}{0pt plus 1fil}")
        directives.add_command(f"\\setlength{{\\emergencystretch}}{{{emergency}}}")
        extra_cmds = list(ctx.profile.just_commands)
        payload = ctx.options.get("extra_paragraph_commands", [])
        if isinstance(payload, (list, tuple, set)):
            extra_cmds.extend(str(command) for command in payload if command)
        elif payload:
            extra_cmds.append(str(payload))
        for command in extra_cmds:
            directives.add_command(command)


class WidowOrphanRule(BaseTypographyRule):
    name = "widow-orphan"

    def apply(self, _: RuleContext, directives: TypographyDirectives) -> None:
        directives.add_command("\\clubpenalty=10000")
        directives.add_command("\\widowpenalty=10000")
        directives.add_command("\\displaywidowpenalty=10000")
        directives.add_command("\\brokenpenalty=10000")


PROFILE_LIBRARY: Dict[str, TypographicProfile] = {
    "scholarly-serif": TypographicProfile(
        name="scholarly-serif",
        description="Classic serif palette for books and reports.",
        preferred_classes=("book", "report", "memoir"),
        font_stack=("mathpazo",),
        line_stretch=1.08,
        parskip="0.7em",
        parindent="1.5em",
        emergency_stretch="4em",
        tolerance=900,
        hyphen_penalty=200,
        just_commands=("\\frenchspacing", "\\raggedbottom"),
        extras=(),
        microtype_options="final,protrusion=true,expansion=true,tracking=true",
        tracking_preset="tracking=true,spacing=nonfrench",
    ),
    "technical-report": TypographicProfile(
        name="technical-report",
        description="Tighter typesetting for research articles and memos.",
        preferred_classes=("article", "elsarticle", "siamart", "ieeetran"),
        font_stack=("newtx",),
        line_stretch=1.04,
        parskip="0.5em",
        parindent="1.25em",
        emergency_stretch="3em",
        tolerance=1400,
        hyphen_penalty=400,
        extras=(),
        microtype_options="final,protrusion=true,expansion=true,tracking=true",
        tracking_preset="tracking=true",
    ),
    "presentation-sans": TypographicProfile(
        name="presentation-sans",
        description="Open forms with extra breathing room for slides.",
        preferred_classes=("beamer", "metropolis"),
        font_stack=("libertine",),
        hyphenation_language="english",
        line_stretch=1.12,
        parskip="0.8em",
        parindent="1em",
        emergency_stretch="2em",
        tolerance=600,
        hyphen_penalty=100,
        just_commands=("\\frenchspacing", "\\raggedbottom", "\\justifying"),
        extras=("ragged2e",),
        microtype_options="final,expansion=true,protrusion=true",
        tracking_preset=None,
    ),
}

DEFAULT_PROFILE = "scholarly-serif"


class TypographyEngine:
    """Evaluate a set of typographic rules for a document."""

    def __init__(self, config: Dict[str, object] | None = None) -> None:
        self.config: Dict[str, object] = dict(config or {})
        self._rules: List[BaseTypographyRule] = [
            FontStackRule(),
            ExtraPackageRule(),
            MicrotypeRule(),
            SpacingRule(),
            HyphenationRule(),
            ParagraphRule(),
            WidowOrphanRule(),
        ]

    def directives_for(self, document_class: str) -> TypographyDirectives:
        context = self._build_context(document_class)
        directives = TypographyDirectives(profile_name=context.profile.name)
        for rule in self._rules:
            if rule.applies(context):
                rule.apply(context, directives)
        return directives

    def describe_profile(self, document_class: str) -> str:
        profile = self._resolve_profile(document_class)
        return profile.description

    def _build_context(self, document_class: str) -> RuleContext:
        profile = self._resolve_profile(document_class)
        options: Dict[str, object] = {
            "language": self.config.get("language", profile.hyphenation_language),
            "line_stretch": self.config.get("line_stretch", profile.line_stretch),
            "parskip": self.config.get("parskip", profile.parskip),
            "parindent": self.config.get("parindent", profile.parindent),
            "emergency_stretch": self.config.get("emergency_stretch", profile.emergency_stretch),
            "tolerance": self.config.get("tolerance", profile.tolerance),
            "hyphen_penalty": self.config.get("hyphen_penalty", profile.hyphen_penalty),
            "microtype_options": self.config.get("microtype_options", profile.microtype_options),
            "microtype_setup": self.config.get("microtype_setup", profile.tracking_preset),
            "extra_paragraph_commands": self.config.get("extra_paragraph_commands", []),
        }
        return RuleContext(document_class=document_class, profile=profile, options=options)

    def _resolve_profile(self, document_class: str) -> TypographicProfile:
        requested = str(self.config.get("profile", "")).strip().lower()
        if requested and requested in PROFILE_LIBRARY:
            return PROFILE_LIBRARY[requested]
        normalized_class = (document_class or "").strip().lower()
        for profile in PROFILE_LIBRARY.values():
            if normalized_class in {class_name.lower() for class_name in profile.preferred_classes}:
                return profile
        return PROFILE_LIBRARY[DEFAULT_PROFILE]


__all__ = ["TypographyEngine", "TypographicProfile", "TypographyDirectives"]
