"""Refactor tools: apply scoped, AST-aware migrations with ``libcst``.

This module implements the scoped refactor engine. Given a migration plan for
one of the two supported demo packages, it applies the **mechanical** parts of
the upgrade using ``libcst`` (a concrete-syntax tree that preserves formatting)
rather than blind string replacement (R3.1), bumps the pinned version in the
manifest (R3.2), and — crucially — never returns broken code: every transformed
source is **re-parsed** to confirm it is still valid Python before it is
accepted (R3.4), and any unsafe transform degrades to a ``guided_pr`` version
bump plus flagged notes instead of committing a broken change (R3.5).

Scope is deliberately narrow (per requirements.md → Scope boundaries):
general-purpose auto-refactor is out of scope, so this engine handles only a
small, **pre-defined pattern registry** for exactly two demo packages:

- ``pydantic`` (1 → 2): a handful of well-known, mechanical renames — the
  ``BaseSettings`` import move, the ``@validator`` → ``@field_validator``
  decorator rename, and the ``.dict()`` → ``.model_dump()`` method rename — plus
  the non-mechanical constructs (class-based ``Config``) which are **flagged**
  for human review rather than transformed (R3.3).
- ``requests``: a **version bump only** — no code transform.

Public entrypoints:

- :func:`validate_python_source` AST-parses candidate source and returns whether
  it is syntactically valid (the gate used everywhere, and the R8.4 guard for
  model-generated code).
- :func:`apply_source_transforms` runs a package's registered ``libcst``
  transforms over one source string, re-parses the result, and reports which
  patterns were auto-fixed vs left untouched/flagged.
- :func:`bump_manifest_version` updates a pinned version in a
  ``requirements.txt`` / ``pyproject.toml`` manifest string (R3.2).
- :func:`apply_migration` is the executor-facing boundary: it consumes a
  migration plan, applies the auto-fixable transforms to the provided source
  files, bumps the manifest, records flagged breaking changes, and falls back to
  ``guided_pr`` when a fix cannot be applied safely (R3.3, R3.5). GitHub PR
  creation is task 8, so this returns the transformed files / diff / flagged
  changes rather than opening a PR.

Design references:
- design.md → Tools layer → ``refactor_tools.py``: "AST-aware transforms
  (``libcst``). Pattern registry only; validate AST after."
- design.md → Error handling → "Auto-fix unsafe / invalid AST | Fall back to
  ``guided_pr``, never commit broken code (R3.5)".
- design.md → Data flow: "Executor branches, applies scoped ``libcst`` fix (or
  version bump only)".
- requirements.md → R3.1: apply auto-fixable changes with an AST-aware
  transformation (``libcst``), not a blind string replacement.
- requirements.md → R3.2: when applying a version bump, update the pinned
  version in the manifest.
- requirements.md → R3.3: leave non-auto-fixable changes unchanged and record
  them as flagged breaking changes for human review.
- requirements.md → R3.4: any transformed file remains syntactically valid
  Python (verified by parsing) before it is committed.
- requirements.md → R3.5: if an auto-fix cannot be applied safely, fall back to
  ``guided_pr`` (version bump + notes) rather than committing a broken change.
- requirements.md → R8.4: validate model-generated code by AST parsing before
  using it.
"""

from __future__ import annotations

import ast
import difflib
import re
from dataclasses import dataclass, field
from typing import Callable

import libcst as cst


# --- Distribution-name normalization (mirrors the other tools) ------------

# PEP 503 name normalization, kept local so the registry lookup is forgiving of
# ``Pydantic`` / ``Requests`` / etc. without cross-importing from package_tools.
_NAME_SEP_RE = re.compile(r"[-_.]+")


def normalize_name(name: str) -> str:
    """Normalize a distribution name per PEP 503 (lowercase, runs of ``-_.`` → ``-``)."""
    return _NAME_SEP_RE.sub("-", name).lower()


# --- Migration strategy constants -----------------------------------------

# The three strategies a migration plan can carry (design.md → Data flow;
# requirements.md → R2.3). Kept as constants so the fallback contract (R3.5) is
# expressed once and referenced everywhere.
STRATEGY_AUTO_FIX = "auto_fix"
STRATEGY_GUIDED_PR = "guided_pr"
STRATEGY_HUMAN_REQUIRED = "human_required"


# --- AST validation gate (R3.4, R8.4) -------------------------------------


def validate_python_source(source: str) -> bool:
    """Return ``True`` if ``source`` parses as valid Python, else ``False``.

    This is the single syntactic gate used throughout the engine. It backs two
    requirements at once:

    - R3.4: after any ``libcst`` transform, the resulting source is re-parsed
      here to confirm it is still valid Python before being accepted.
    - R8.4: model-generated code is validated here before use; code that does
      not parse is rejected.

    We use the stdlib :mod:`ast` (a genuinely independent parser) so validation
    does not merely trust the same ``libcst`` round-trip that produced the code.

    Args:
        source: Candidate Python source text.

    Returns:
        ``True`` when :func:`ast.parse` succeeds, ``False`` on any
        :class:`SyntaxError` (including ``ValueError`` from embedded null bytes).
    """
    try:
        ast.parse(source)
    except (SyntaxError, ValueError):
        return False
    return True


# --- libcst transforms for the pydantic 1 → 2 pattern registry ------------


class _PydanticImportMover(cst.CSTTransformer):
    """Move ``BaseSettings`` out of ``pydantic`` into ``pydantic_settings``.

    In pydantic v2, ``BaseSettings`` was extracted into the separate
    ``pydantic-settings`` package, so ``from pydantic import BaseSettings`` must
    become ``from pydantic_settings import BaseSettings``. This is a mechanical,
    well-known 1 → 2 move and is therefore auto-fixable (R3.1).

    The transformer handles the common shapes:

    - ``from pydantic import BaseSettings`` → rewrites the module to
      ``pydantic_settings``.
    - ``from pydantic import BaseModel, BaseSettings`` → splits into two
      ``ImportFrom`` statements so the remaining names keep importing from
      ``pydantic``.

    It records whether it changed anything via :attr:`applied`.
    """

    def __init__(self) -> None:
        super().__init__()
        self.applied = False

    @staticmethod
    def _is_pydantic_module(node: cst.ImportFrom) -> bool:
        module = node.module
        return isinstance(module, cst.Name) and module.value == "pydantic"

    @staticmethod
    def _import_alias_name(alias: cst.ImportAlias) -> str | None:
        name = alias.name
        if isinstance(name, cst.Name):
            return name.value
        return None

    def leave_ImportFrom(  # noqa: N802 - libcst visitor naming
        self, original_node: cst.ImportFrom, updated_node: cst.ImportFrom
    ) -> cst.BaseSmallStatement | cst.FlattenSentinel[cst.BaseSmallStatement]:
        if not self._is_pydantic_module(original_node):
            return updated_node
        names = updated_node.names
        # ``from pydantic import *`` (ImportStar) is not a name list we split.
        if not isinstance(names, (list, tuple)):
            return updated_node

        settings = [a for a in names if self._import_alias_name(a) == "BaseSettings"]
        if not settings:
            return updated_node

        others = [a for a in names if self._import_alias_name(a) != "BaseSettings"]
        self.applied = True

        settings_import = updated_node.with_changes(
            module=cst.Name("pydantic_settings"),
            names=_clean_alias_commas(settings),
        )
        if not others:
            # Whole import was just BaseSettings: rewrite the module in place.
            return settings_import

        # Split: keep the other names on ``pydantic``, add a second import for
        # ``BaseSettings`` from ``pydantic_settings``.
        pydantic_import = updated_node.with_changes(names=_clean_alias_commas(others))
        return cst.FlattenSentinel([pydantic_import, settings_import])


def _clean_alias_commas(aliases: list[cst.ImportAlias]) -> list[cst.ImportAlias]:
    """Return import aliases with a trailing comma stripped from the last one.

    When we slice an import's name list, the last surviving alias may still
    carry a trailing comma from the original source, which would render invalid.
    This normalizes the comma on the final alias.
    """
    if not aliases:
        return aliases
    fixed = list(aliases)
    fixed[-1] = fixed[-1].with_changes(comma=cst.MaybeSentinel.DEFAULT)
    return fixed


class _NameRenamer(cst.CSTTransformer):
    """Rename bare ``Name`` references matching a fixed old → new mapping.

    Used for the ``@validator`` → ``@field_validator`` decorator rename. It only
    rewrites standalone ``Name`` nodes (e.g., a decorator ``@validator(...)`` or
    an imported name ``validator``), which is exactly the mechanical rename we
    want; attribute access like ``pydantic.validator`` is left alone because the
    demo scope imports the names directly.
    """

    def __init__(self, renames: dict[str, str]) -> None:
        super().__init__()
        self.renames = renames
        self.applied = False

    def leave_Name(  # noqa: N802 - libcst visitor naming
        self, original_node: cst.Name, updated_node: cst.Name
    ) -> cst.BaseExpression:
        new = self.renames.get(original_node.value)
        if new is not None:
            self.applied = True
            return updated_node.with_changes(value=new)
        return updated_node


class _MethodCallRenamer(cst.CSTTransformer):
    """Rename zero-argument method calls like ``x.dict()`` → ``x.model_dump()``.

    Targets the pydantic v2 ``.dict()`` → ``.model_dump()`` rename. To stay
    mechanical and safe within the demo scope, it only rewrites calls of the
    shape ``<expr>.<old>()`` with **no arguments** — the common serialization
    call — leaving argument-bearing calls untouched (those may need judgment and
    are better surfaced as flagged rather than silently rewritten).
    """

    def __init__(self, method_renames: dict[str, str]) -> None:
        super().__init__()
        self.method_renames = method_renames
        self.applied = False

    def leave_Call(  # noqa: N802 - libcst visitor naming
        self, original_node: cst.Call, updated_node: cst.Call
    ) -> cst.BaseExpression:
        func = updated_node.func
        if not isinstance(func, cst.Attribute):
            return updated_node
        if updated_node.args:
            # Only the no-argument serialization call is mechanically safe.
            return updated_node
        old = func.attr.value
        new = self.method_renames.get(old)
        if new is None:
            return updated_node
        self.applied = True
        return updated_node.with_changes(
            func=func.with_changes(attr=cst.Name(new))
        )


# --- Pattern registry -----------------------------------------------------


@dataclass(frozen=True)
class TransformPattern:
    """One registered pattern in a package's scoped transform registry.

    ``pattern_id`` is a stable machine-readable id; ``description`` is
    human-facing; ``build`` produces a fresh ``libcst`` transformer each run
    (transformers are stateful, so we never reuse an instance). ``auto_fixable``
    records whether the pattern is applied mechanically (``True``) or merely
    flagged for human review (``False``). ``detect`` is a regex used only to
    *detect* a flagged pattern in the source so it can be reported (R3.3).
    """

    pattern_id: str
    description: str
    auto_fixable: bool
    # ``build`` is ``None`` for flagged-only patterns (nothing is transformed).
    build: Callable[[], cst.CSTTransformer] | None = None
    detect: str | None = None


# The pydantic 1 → 2 registry. The first three are mechanical, well-known
# renames (auto-fixable, R3.1). The class-based ``Config`` → ``model_config``
# change requires restructuring the class body and is NOT mechanically safe, so
# it is registered as flagged-only (auto_fixable=False) and surfaced for human
# review (R3.3) rather than transformed.
_PYDANTIC_PATTERNS: list[TransformPattern] = [
    TransformPattern(
        pattern_id="pydantic.basesettings_import_move",
        description=(
            "Move BaseSettings import: `from pydantic import BaseSettings` -> "
            "`from pydantic_settings import BaseSettings` (extracted to the "
            "pydantic-settings package in v2)."
        ),
        auto_fixable=True,
        build=_PydanticImportMover,
    ),
    TransformPattern(
        pattern_id="pydantic.validator_rename",
        description="Rename `@validator` decorator to `@field_validator`.",
        auto_fixable=True,
        build=lambda: _NameRenamer({"validator": "field_validator"}),
    ),
    TransformPattern(
        pattern_id="pydantic.dict_to_model_dump",
        description="Rename model `.dict()` calls to `.model_dump()`.",
        auto_fixable=True,
        build=lambda: _MethodCallRenamer({"dict": "model_dump"}),
    ),
    TransformPattern(
        pattern_id="pydantic.config_class_to_model_config",
        description=(
            "Class-based `class Config:` must become a `model_config = "
            "ConfigDict(...)` assignment in v2. This restructures the class "
            "body and is left for human review rather than auto-applied."
        ),
        auto_fixable=False,
        # Detect an inner ``class Config`` block to flag it (R3.3).
        detect=r"(^|\n)\s+class\s+Config\b",
    ),
]

# The requests registry is intentionally empty of code transforms: the requests
# demo is a **version bump only** (design.md → Data flow; task 7.1). Any code
# change is out of scope, so there are no patterns to apply.
_REQUESTS_PATTERNS: list[TransformPattern] = []

# The scoped registry: normalized package name → its ordered patterns. Only the
# two demo packages are present; anything else is "unregistered" and yields no
# transforms (the executor then treats it as version-bump-only / guided).
PATTERN_REGISTRY: dict[str, list[TransformPattern]] = {
    "pydantic": _PYDANTIC_PATTERNS,
    "requests": _REQUESTS_PATTERNS,
}


def get_patterns(package: str) -> list[TransformPattern]:
    """Return the registered transform patterns for ``package`` (may be empty)."""
    return PATTERN_REGISTRY.get(normalize_name(package), [])


# --- Result shapes --------------------------------------------------------


@dataclass
class TransformResult:
    """Outcome of running one package's transforms over a single source string.

    ``ok`` is ``False`` only when the transformed source failed AST validation
    (R3.4) — in that case ``source`` is left as the *original* (never broken
    code) and the caller falls back to ``guided_pr`` (R3.5). ``applied`` lists
    the auto-fixable ``pattern_id``s that actually changed the source; ``flagged``
    lists non-auto-fixable patterns detected in the source (R3.3); ``changed`` is
    ``True`` when the source text differs from the input.
    """

    ok: bool
    source: str
    applied: list[str] = field(default_factory=list)
    flagged: list[dict[str, str]] = field(default_factory=list)
    changed: bool = False
    error: str | None = None


# A flagged breaking change surfaced for human review (R3.3): a stable
# ``pattern_id``, the human-facing ``description``, and the ``file`` it was found
# in (filled in by :func:`apply_migration`).
FlaggedChange = dict[str, str]


def _detect_flagged(
    patterns: list[TransformPattern], source: str
) -> list[FlaggedChange]:
    """Detect non-auto-fixable patterns present in ``source`` (R3.3)."""
    found: list[FlaggedChange] = []
    for pattern in patterns:
        if pattern.auto_fixable or pattern.detect is None:
            continue
        if re.search(pattern.detect, source, flags=re.MULTILINE):
            found.append(
                {
                    "pattern_id": pattern.pattern_id,
                    "description": pattern.description,
                }
            )
    return found


def apply_source_transforms(package: str, source: str) -> TransformResult:
    """Apply a package's auto-fixable transforms to one source string (R3.1/3.4).

    Runs each registered auto-fixable ``libcst`` transform in order, then
    **re-parses** the resulting source with :func:`validate_python_source` to
    confirm it is still valid Python before accepting it (R3.4). Non-auto-fixable
    patterns are never transformed; instead they are detected and reported in
    ``flagged`` for human review (R3.3).

    If the transformed source fails validation, the result is marked ``ok =
    False`` and ``source`` is the *original, untouched* text — the caller must
    not commit it and should fall back to ``guided_pr`` (R3.5). We never return
    broken code.

    Args:
        package: Distribution name of a supported demo package.
        source: The Python source to transform.

    Returns:
        A :class:`TransformResult`. On success, ``source`` is the transformed
        (and validated) text; on failure, it is the original source with ``ok =
        False``.
    """
    patterns = get_patterns(package)
    flagged = _detect_flagged(patterns, source)

    # Nothing to auto-fix (e.g., requests, or pydantic source with only flagged
    # constructs): return the source unchanged, still reporting any flags.
    auto_patterns = [p for p in patterns if p.auto_fixable and p.build is not None]
    if not auto_patterns:
        return TransformResult(ok=True, source=source, flagged=flagged, changed=False)

    # Parse once; a source that doesn't parse to begin with can't be transformed
    # safely, so we bail out to guided_pr rather than guessing (R3.5).
    try:
        module = cst.parse_module(source)
    except cst.ParserSyntaxError as exc:
        return TransformResult(
            ok=False,
            source=source,
            flagged=flagged,
            changed=False,
            error=f"could not parse source before transform: {exc}",
        )

    applied: list[str] = []
    current = module
    for pattern in auto_patterns:
        transformer = pattern.build()  # fresh, since transformers are stateful
        current = current.visit(transformer)
        if getattr(transformer, "applied", False):
            applied.append(pattern.pattern_id)

    new_source = current.code

    # R3.4: re-parse the transformed source with an independent parser. If it is
    # not valid Python, discard the transform and signal a safe fallback (R3.5).
    if not validate_python_source(new_source):
        return TransformResult(
            ok=False,
            source=source,  # keep the ORIGINAL, never broken code
            applied=[],
            flagged=flagged,
            changed=False,
            error="transformed source failed AST validation",
        )

    return TransformResult(
        ok=True,
        source=new_source,
        applied=applied,
        flagged=flagged,
        changed=new_source != source,
    )


# --- Manifest version bump (R3.2) -----------------------------------------


def _pin_pattern(package: str) -> re.Pattern[str]:
    """Build a regex matching a ``==`` pin for ``package`` (separator-insensitive).

    Matches a pinned dependency in either a ``requirements.txt``
    (``pydantic==1.10.13``) or a ``pyproject.toml`` PEP 621 / poetry string
    (``"pydantic==1.10.13"``). The name portion tolerates the interchangeable
    ``-``, ``_`` and ``.`` separators PyPI treats as equivalent, and an optional
    extras group (``pydantic[dotenv]``). It captures ``(prefix)(old)`` so we can
    rewrite only the version while preserving surrounding text.
    """
    # Turn each separator in the name into a class matching any separator.
    parts = re.split(r"[-_.]+", normalize_name(package))
    name_re = r"[-_.]".join(re.escape(p) for p in parts if p)
    return re.compile(
        r"(?P<prefix>\b" + name_re + r"\b\s*(?:\[[^\]]*\])?\s*==\s*)"
        r"(?P<old>[0-9][0-9A-Za-z.\-+!]*)",
        re.IGNORECASE,
    )


@dataclass
class ManifestBumpResult:
    """Outcome of a manifest version bump (R3.2).

    ``changed`` is ``True`` when at least one pin was rewritten. ``old_versions``
    lists the versions that were replaced (there may be more than one occurrence,
    e.g. in both ``[project]`` and a constraints block). ``content`` is the
    updated manifest text (unchanged when no pin matched).
    """

    changed: bool
    content: str
    old_versions: list[str] = field(default_factory=list)


def bump_manifest_version(
    manifest_content: str, package: str, new_version: str
) -> ManifestBumpResult:
    """Update the pinned version of ``package`` in a manifest string (R3.2).

    Rewrites every ``package==<old>`` pin to ``package==<new_version>`` in a
    ``requirements.txt`` or ``pyproject.toml`` body, preserving surrounding
    whitespace, extras, and quoting. Matching is PEP 503 separator-insensitive
    so ``Pydantic`` / ``pydantic`` names line up.

    This operates on manifest **text** (not code), so it does not go through the
    ``libcst`` path; the version bump is the one safe change that always applies
    even when a code transform is not possible (the ``guided_pr`` fallback,
    R3.5).

    Args:
        manifest_content: The manifest file's text.
        package: Distribution name whose pin should be updated.
        new_version: The version string to pin to.

    Returns:
        A :class:`ManifestBumpResult` with the updated ``content`` and the list
        of ``old_versions`` that were replaced (empty when nothing matched).
    """
    pattern = _pin_pattern(package)
    old_versions: list[str] = []

    def _replace(match: re.Match[str]) -> str:
        old_versions.append(match.group("old"))
        return f"{match.group('prefix')}{new_version}"

    new_content = pattern.sub(_replace, manifest_content)
    return ManifestBumpResult(
        changed=bool(old_versions),
        content=new_content,
        old_versions=old_versions,
    )


# --- Executor-facing migration application (R3.3, R3.5) -------------------

# The transformed set of source files: filename → new source text. Only files
# that actually changed are included.
TransformedFiles = dict[str, str]


def _unified_diff(filename: str, before: str, after: str) -> str:
    """Return a unified diff for a single file (empty string if unchanged)."""
    if before == after:
        return ""
    diff = difflib.unified_diff(
        before.splitlines(keepends=True),
        after.splitlines(keepends=True),
        fromfile=f"a/{filename}",
        tofile=f"b/{filename}",
    )
    return "".join(diff)


def apply_migration(
    plan: dict,
    *,
    source_files: dict[str, str] | None = None,
    manifest: tuple[str, str] | None = None,
) -> dict:
    """Apply a migration plan's scoped changes, never returning broken code.

    This is the executor-facing boundary (design.md → ExecutorAgent). It consumes
    a migration plan (``package``, ``current``, ``target``, ``strategy``,
    ``breaking_changes``; see requirements.md → R2.3) and produces the concrete
    file changes for the branch/PR step (task 8 opens the PR — this function
    stops at transformed files + diff + flagged changes).

    Behavior:

    - **Version bump (R3.2).** When a ``manifest`` is provided, its pinned
      version for the package is bumped to ``target``. This is always safe and
      always applied.
    - **Auto-fix transforms (R3.1, R3.4).** For each provided source file, the
      package's auto-fixable ``libcst`` patterns are applied and the result is
      re-parsed for validity. Validated changes are collected into ``changes``.
    - **Flagged changes (R3.3).** Non-auto-fixable patterns detected in the
      sources (e.g., pydantic class-based ``Config``) are recorded as flagged
      breaking changes and the code for them is left untouched. Any
      ``breaking_changes`` carried on the plan are merged in.
    - **Safe fallback (R3.5).** If any file's transform fails AST validation, or
      the plan's strategy is ``guided_pr`` / ``human_required``, no code changes
      are applied; the result strategy becomes ``guided_pr`` (version bump +
      notes) so a broken change is never committed.

    Args:
        plan: The migration plan dict. Recognized keys: ``package`` (required),
            ``current``, ``target``, ``strategy``, ``breaking_changes``.
        source_files: Optional mapping of ``filename -> source text`` to
            transform. Omit for a version-bump-only migration (e.g. requests).
        manifest: Optional ``(filename, content)`` tuple for the manifest whose
            pin should be bumped to ``target``.

    Returns:
        A dict with:
        ``package``, ``strategy`` (the *resolved* strategy after any fallback),
        ``changes`` (``filename -> new source``, transformed + validated files
        and the bumped manifest), ``flagged`` (list of flagged breaking changes),
        ``diff`` (a combined unified diff of all changes), and ``applied`` (the
        auto-fix ``pattern_id``s that fired).

    Raises:
        KeyError: If ``plan`` has no ``package`` key.
    """
    package = normalize_name(plan["package"])
    target = plan.get("target")
    requested_strategy = plan.get("strategy", STRATEGY_AUTO_FIX)

    changes: TransformedFiles = {}
    flagged: list[FlaggedChange] = []
    applied: list[str] = []
    diffs: list[str] = []
    manifest_name = manifest[0] if manifest is not None else None

    # Carry over any breaking changes the planner already identified (R3.3).
    for bc in plan.get("breaking_changes", []) or []:
        if isinstance(bc, dict):
            flagged.append(
                {
                    "pattern_id": str(bc.get("id", "plan.breaking_change")),
                    "description": str(bc.get("description", bc)),
                    "file": str(bc.get("file", "")),
                }
            )
        else:
            flagged.append(
                {
                    "pattern_id": "plan.breaking_change",
                    "description": str(bc),
                    "file": "",
                }
            )

    # --- Manifest version bump (always safe, always applied). -------------
    if manifest is not None and target:
        _, manifest_content = manifest
        bump = bump_manifest_version(manifest_content, package, str(target))
        if bump.changed:
            changes[manifest_name] = bump.content
            diffs.append(_unified_diff(manifest_name, manifest_content, bump.content))

    # A plan that already asks for guided_pr / human_required does not attempt
    # code transforms — the version bump + flagged notes are the deliverable
    # (R3.5 / R5.2). We still surface any flags detected in the sources.
    force_fallback = requested_strategy in (STRATEGY_GUIDED_PR, STRATEGY_HUMAN_REQUIRED)

    resolved_strategy = requested_strategy

    # --- Auto-fix source transforms (R3.1, R3.4) with safe fallback (R3.5). -
    if source_files:
        for filename, source in source_files.items():
            result = apply_source_transforms(package, source)

            # Record flags regardless of whether a fix applied (R3.3).
            for flag in result.flagged:
                flagged.append({**flag, "file": filename})

            if not result.ok:
                # A transform produced invalid code: never commit it. Drop ALL
                # code changes and fall back to guided_pr (version bump + notes).
                resolved_strategy = STRATEGY_GUIDED_PR
                # Keep only the (safe) manifest bump among the changes.
                changes = {
                    k: v for k, v in changes.items() if k == manifest_name
                }
                # Recompute diffs from the surviving (manifest-only) change.
                diffs = (
                    [_unified_diff(manifest_name, manifest[1], changes[manifest_name])]
                    if manifest_name in changes
                    else []
                )
                applied = []
                break

            if force_fallback:
                # Guided/human plans: leave code untouched (R5.2), only flag.
                continue

            if result.changed:
                changes[filename] = result.source
                diffs.append(_unified_diff(filename, source, result.source))
                applied.extend(result.applied)

    combined_diff = "\n".join(d for d in diffs if d)

    return {
        "package": package,
        "strategy": resolved_strategy,
        "changes": changes,
        "flagged": flagged,
        "diff": combined_diff,
        "applied": applied,
    }
