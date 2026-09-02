"""Changelog tools: fetch and summarize release notes for the demo packages.

This module implements the changelog fetcher for the migration planner. Its job
is to answer, for one of the two supported demo packages, "what changed between
the version I have and the version I'd upgrade to?" and to distill that into a
short, planner-facing summary of **breaking changes, deprecations, and
migration steps**.

Scope is deliberately narrow (per requirements.md → Scope boundaries): general
changelog discovery is out of scope, so the package→GitHub-repo mapping is
**hardcoded for the two demo packages** — ``requests`` and ``pydantic``. Any
other package is treated as unmapped and forces ``low`` confidence downstream.

Public entrypoints:

- :func:`fetch_release_notes` fetches the GitHub release notes for every release
  in the ``(current, target]`` range of a supported package.
- :func:`summarize_release_notes` condenses raw notes into a structured summary
  focused on breaking changes / deprecations / migration steps, using an
  injectable Nova Lite client (see :class:`ChangelogModel`).
- :func:`fetch_and_summarize_changelog` composes the two behind a single
  planner-facing boundary and, crucially, returns a :data:`ChangelogResult`
  whose ``confidence`` is forced to ``"low"`` whenever notes cannot be fetched
  (R2.4).

Design references:
- design.md → Tools layer → ``changelog_tools.py``: "Fetch + summarize notes.
  Hardcoded package→repo map for the 2 demos."
- design.md → Models layer: Nova Lite for changelog summarization; the Bedrock
  client (``bedrock_client.py``, task 5) is not built yet, so summarization
  accepts an injected client interface and keeps the Nova invocation isolated.
- design.md → Error handling → "Changelog unavailable | Force ``low`` confidence
  → human review (R2.4)".
- requirements.md → R2.1: fetch release notes for the version range of a
  supported demo package.
- requirements.md → R2.2: summarize focusing on breaking changes, deprecations,
  and migration steps.
- requirements.md → R2.4: if a changelog cannot be fetched, classify the
  migration as ``low`` confidence and require human review.
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

import requests

try:  # ``packaging`` gives PEP 440-correct comparisons when available.
    from packaging.version import InvalidVersion, Version

    _HAS_PACKAGING = True
except ImportError:  # pragma: no cover - fallback path when packaging is absent
    _HAS_PACKAGING = False


# --- Hardcoded package → GitHub repo map (the two demo packages only) -----

# Per the scope boundary, changelog/repo mapping is hardcoded for exactly the
# two demo packages. Keys are PEP 503-normalized distribution names; values are
# ``owner/repo`` slugs whose GitHub Releases carry the changelog. Anything not
# in this map is "unmapped" and forces ``low`` confidence downstream (R2.4).
PACKAGE_REPO_MAP: dict[str, str] = {
    "requests": "psf/requests",
    "pydantic": "pydantic/pydantic",
}

# GitHub REST API base for listing a repository's releases.
_GITHUB_RELEASES_URL = "https://api.github.com/repos/{repo}/releases"

# Network timeout (seconds) for GitHub requests: (connect, read).
_GITHUB_TIMEOUT = (5.0, 10.0)

# Per-request page size when listing releases. 100 is GitHub's max and is more
# than enough to span any range across the two demo packages.
_RELEASES_PER_PAGE = 100

# Confidence levels the planner understands. Kept as constants so the "force
# low on failure" contract (R2.4) is expressed once and referenced everywhere.
CONFIDENCE_LOW = "low"
CONFIDENCE_HIGH = "high"

# PEP 503 name normalization: collapse runs of ``-_.`` into a single dash.
_NAME_SEP_RE = re.compile(r"[-_.]+")

# Fallback numeric-segment tokenizer for the no-``packaging`` path.
_NUMERIC_SEGMENT_RE = re.compile(r"\d+")


def normalize_name(name: str) -> str:
    """Normalize a distribution name per PEP 503 (lowercase, runs of ``-_.`` → ``-``).

    Kept local (mirroring ``package_tools.normalize_name``) so the map lookup is
    forgiving of ``Requests`` / ``PyDantic`` / etc. without cross-importing.
    """
    return _NAME_SEP_RE.sub("-", name).lower()


# --- Result shapes --------------------------------------------------------

# A single release's notes within the upgrade range: the release's version tag
# and its raw body (the human-written changelog text from GitHub).
ReleaseNote = dict[str, str]

# The structured, planner-facing outcome of the changelog step. On success it
# carries the summary text and the raw notes it was built from. On any fetch
# failure it carries ``confidence == "low"`` plus a machine-readable ``error``
# so the planner routes the migration to human review (R2.4). ``confidence`` is
# ``None`` on the happy path — the planner (Nova Pro, task 6) sets the real
# confidence — but is pinned to ``"low"`` here whenever the changelog is
# unavailable, which is the whole point of this contract.
ChangelogResult = dict[str, object]


# --- Injectable Nova Lite client interface (task 5 not built yet) ---------


@runtime_checkable
class ChangelogModel(Protocol):
    """Minimal interface for the model used to summarize changelog notes.

    The concrete Bedrock/Nova client lands in ``bedrock_client.py`` (task 5).
    Until then — and to keep summarization unit-testable without AWS — callers
    inject any object satisfying this Protocol. The real Nova Lite wrapper will
    satisfy it by exposing a ``summarize`` method, keeping the Bedrock
    invocation isolated behind this seam.
    """

    def summarize(self, prompt: str) -> str:
        """Return a summary string for the given prompt."""
        ...


# The instruction wrapped around raw notes before handing them to Nova Lite.
# Focus is pinned to exactly the three things the planner cares about (R2.2).
_SUMMARY_PROMPT_TEMPLATE = (
    "You are summarizing the release notes for upgrading the Python package "
    "'{package}' from version {current} to {target}. Summarize concisely, "
    "focusing ONLY on: (1) breaking changes, (2) deprecations, and "
    "(3) concrete migration steps a developer must take. Omit unrelated "
    "features and bug fixes. Release notes follow:\n\n{notes}"
)


# --- Version range selection ----------------------------------------------


def _version_key(version: str) -> tuple[int, ...]:
    """Return leading integer segments for a coarse compare (fallback path)."""
    return tuple(int(seg) for seg in _NUMERIC_SEGMENT_RE.findall(version))


def _parse_version(version: str) -> object | None:
    """Best-effort PEP 440 parse; ``None`` when the tag isn't a clean version."""
    if _HAS_PACKAGING:
        try:
            return Version(version)
        except InvalidVersion:
            return None
    key = _version_key(version)
    return key or None


def _strip_tag(tag: str) -> str:
    """Normalize a GitHub tag to a bare version (drop a leading ``v``).

    Both demo repos tag releases as plain versions (``2.32.3``) or with a ``v``
    prefix on older tags; stripping a single leading ``v`` covers both without
    guessing at exotic tag schemes outside the demo scope.
    """
    tag = tag.strip()
    if tag[:1] in ("v", "V") and tag[1:2].isdigit():
        return tag[1:]
    return tag


def _in_upgrade_range(tag: str, current: str, target: str) -> bool:
    """Return ``True`` if ``tag`` falls in the ``(current, target]`` window.

    We want the notes for versions strictly newer than what's installed, up to
    and including the target — that's exactly what changed by upgrading. Tags
    that don't parse as versions are excluded (conservative: we'd rather drop a
    weird tag than mis-scope the range).
    """
    tag_v = _parse_version(_strip_tag(tag))
    current_v = _parse_version(current)
    target_v = _parse_version(target)
    if tag_v is None or current_v is None or target_v is None:
        return False
    return current_v < tag_v <= target_v


# --- Fetching release notes (R2.1) ----------------------------------------


def fetch_release_notes(
    package: str,
    current: str,
    target: str,
    *,
    session: requests.Session | None = None,
) -> list[ReleaseNote]:
    """Fetch GitHub release notes for the ``(current, target]`` range (R2.1).

    Resolves ``package`` through the hardcoded :data:`PACKAGE_REPO_MAP` and
    pulls the repository's releases from the GitHub Releases API, keeping only
    those whose version tag falls in the upgrade window. Notes are returned
    newest-first (GitHub's default ordering).

    Args:
        package: Distribution name (normalized or not) of a supported demo
            package. Must be present in :data:`PACKAGE_REPO_MAP`.
        current: The currently pinned version (exclusive lower bound).
        target: The version being upgraded to (inclusive upper bound).
        session: Optional ``requests.Session`` to reuse a connection pool.

    Returns:
        A list of ``{"version", "body"}`` dicts, one per in-range release.

    Raises:
        KeyError: If ``package`` is not one of the supported demo packages.
            Callers (see :func:`fetch_and_summarize_changelog`) translate this
            into a ``low``-confidence result rather than crashing (R2.4).
        requests.RequestException: On network/HTTP failures reaching GitHub.
            Likewise translated to ``low`` confidence upstream (R2.4).
    """
    repo = PACKAGE_REPO_MAP[normalize_name(package)]
    url = _GITHUB_RELEASES_URL.format(repo=repo)
    getter = session.get if session is not None else requests.get
    response = getter(
        url,
        params={"per_page": _RELEASES_PER_PAGE},
        timeout=_GITHUB_TIMEOUT,
    )
    response.raise_for_status()
    payload = response.json()

    notes: list[ReleaseNote] = []
    if isinstance(payload, list):
        for release in payload:
            if not isinstance(release, dict):
                continue
            tag = release.get("tag_name") or release.get("name") or ""
            if not isinstance(tag, str) or not tag:
                continue
            if not _in_upgrade_range(tag, current, target):
                continue
            body = release.get("body")
            notes.append(
                {
                    "version": _strip_tag(tag),
                    "body": body if isinstance(body, str) else "",
                }
            )
    return notes


# --- Summarizing notes with Nova Lite (R2.2) ------------------------------


def _render_notes(notes: list[ReleaseNote]) -> str:
    """Flatten release notes into a single text block for summarization."""
    blocks: list[str] = []
    for note in notes:
        version = note.get("version", "")
        body = note.get("body", "")
        blocks.append(f"## {version}\n{body}".strip())
    return "\n\n".join(blocks)


def summarize_release_notes(
    package: str,
    current: str,
    target: str,
    notes: list[ReleaseNote],
    *,
    model: ChangelogModel,
) -> str:
    """Summarize release notes via Nova Lite, focused on migration impact (R2.2).

    Builds a focused prompt (breaking changes, deprecations, migration steps
    only) and delegates the actual generation to the injected ``model``. The
    Bedrock/Nova invocation lives entirely behind :class:`ChangelogModel`, so
    this function has no AWS dependency and is fully unit-testable with a fake
    client.

    Args:
        package: The distribution name (for prompt context).
        current: The current version (for prompt context).
        target: The target version (for prompt context).
        notes: Release notes from :func:`fetch_release_notes`.
        model: An injected client satisfying :class:`ChangelogModel`. The real
            Nova Lite wrapper (task 5) will satisfy this Protocol.

    Returns:
        The model's summary text. Whitespace is stripped for a clean result.
    """
    prompt = _SUMMARY_PROMPT_TEMPLATE.format(
        package=package,
        current=current,
        target=target,
        notes=_render_notes(notes),
    )
    return model.summarize(prompt).strip()


# --- Composed, planner-facing entrypoint (R2.1, R2.2, R2.4) ---------------


def _low_confidence_result(
    package: str,
    current: str,
    target: str,
    error_kind: str,
    message: str,
) -> ChangelogResult:
    """Build a ``low``-confidence changelog result for the planner (R2.4).

    Per design.md's error table, a changelog we can't fetch must force the
    migration to ``low`` confidence and human review. This packages that
    outcome uniformly: no summary, an empty notes list, ``confidence == "low"``,
    and a structured ``error`` describing why.
    """
    return {
        "package": normalize_name(package),
        "current": current,
        "target": target,
        "summary": None,
        "notes": [],
        "confidence": CONFIDENCE_LOW,
        "error": {"kind": error_kind, "message": message},
    }


def fetch_and_summarize_changelog(
    package: str,
    current: str,
    target: str,
    *,
    model: ChangelogModel,
    session: requests.Session | None = None,
) -> ChangelogResult:
    """Fetch + summarize a demo package's changelog for the planner (R2.1/2.2/2.4).

    This is the single boundary the planner calls. On the happy path it fetches
    the in-range GitHub release notes and returns a Nova Lite summary focused on
    breaking changes / deprecations / migration steps. On **any** failure —
    an unmapped package, a network/HTTP error reaching GitHub, or a
    summarization error — it returns a :data:`ChangelogResult` with
    ``confidence`` pinned to ``"low"`` so the migration is routed to human
    review (R2.4), rather than propagating the exception.

    Design reference:
    - design.md → Error handling → "Changelog unavailable | Force ``low``
      confidence → human review (R2.4)".

    Args:
        package: Distribution name of a supported demo package.
        current: Currently pinned version (exclusive lower bound).
        target: Version being upgraded to (inclusive upper bound).
        model: Injected Nova Lite client satisfying :class:`ChangelogModel`.
        session: Optional ``requests.Session`` reused for the GitHub call.

    Returns:
        A :data:`ChangelogResult`. On success: ``{"package", "current",
        "target", "summary", "notes", "confidence": None, "error": None}`` with
        ``confidence`` left for the planner to set. On failure: the same shape
        with ``summary``/``notes`` empty, ``confidence == "low"``, and a
        populated ``error``.
    """
    # Unmapped package: out of the supported demo scope → low confidence (R2.4).
    if normalize_name(package) not in PACKAGE_REPO_MAP:
        return _low_confidence_result(
            package,
            current,
            target,
            "unmapped_package",
            f"no changelog mapping for package '{package}' (supported: "
            f"{', '.join(sorted(PACKAGE_REPO_MAP))})",
        )

    # Fetch notes; a network/HTTP failure forces low confidence (R2.4).
    try:
        notes = fetch_release_notes(package, current, target, session=session)
    except requests.RequestException as exc:
        return _low_confidence_result(
            package, current, target, "fetch_failed", f"changelog fetch failed: {exc}"
        )

    # No in-range releases means we have nothing to reason about → low
    # confidence and human review, same as an unavailable changelog (R2.4).
    if not notes:
        return _low_confidence_result(
            package,
            current,
            target,
            "no_release_notes",
            f"no release notes found for {package} in range ({current}, {target}]",
        )

    # Summarize; if the model call fails, still degrade to low confidence (R2.4)
    # rather than crashing the pipeline.
    try:
        summary = summarize_release_notes(package, current, target, notes, model=model)
    except Exception as exc:  # noqa: BLE001 - any model failure degrades safely
        return _low_confidence_result(
            package,
            current,
            target,
            "summarize_failed",
            f"changelog summarization failed: {exc}",
        )

    return {
        "package": normalize_name(package),
        "current": current,
        "target": target,
        "summary": summary,
        "notes": notes,
        "confidence": None,  # planner (Nova Pro) sets real confidence on success
        "error": None,
    }
