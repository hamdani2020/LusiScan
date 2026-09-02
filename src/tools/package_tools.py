"""Package tools: parse declared dependencies and resolve latest versions.

This module implements the package monitor. It reads *declared* dependencies
from a repository's manifest — either a ``requirements.txt`` or a
``pyproject.toml`` — **without installing anything** (task 3.1), and resolves
each package's latest available version against **PyPI** (task 3.2), never the
local environment.

Public entrypoints:

- :func:`parse_dependencies` returns a normalized list of dicts shaped like
  ``{"name": <str>, "current": <str | None>}`` where ``current`` is the pinned
  version when the manifest pins one (``==``), or ``None`` when the declaration
  only expresses a lower/other bound or no version at all.
- :func:`fetch_latest_version` queries the PyPI JSON API for a single package's
  latest release.
- :func:`compute_outdated_packages` combines parsed dependencies with PyPI
  lookups and returns the outdated ones as ``[{"name", "current", "latest"}]``.

Design references:
- design.md → Tools layer → ``package_tools.py``: "Parse manifest, query PyPI
  for latest. No install; registry API, not ``pip list``."
- requirements.md → R1.1: read declared dependencies from the manifest without
  installing them.
- requirements.md → R1.2: determine the latest version by querying the package
  registry (PyPI), not the local environment.
- requirements.md → R1.3: record a newer-than-declared package as outdated with
  its name, current version, and latest version.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import requests

try:  # ``packaging`` gives PEP 440-correct comparisons when available.
    from packaging.version import InvalidVersion, Version

    _HAS_PACKAGING = True
except ImportError:  # pragma: no cover - fallback path when packaging is absent
    _HAS_PACKAGING = False

# A declared dependency, normalized. ``current`` is the exact pinned version
# (from a ``==`` specifier) when available, otherwise ``None``.
Dependency = dict[str, str | None]

# An outdated package: a declared dependency whose PyPI latest is newer than the
# pinned ``current`` version. ``latest`` is always a concrete version string.
OutdatedPackage = dict[str, str]

# Base URL for the PyPI JSON API. ``{name}`` is the (normalized) project name.
_PYPI_JSON_URL = "https://pypi.org/pypi/{name}/json"

# Network timeout (seconds) for PyPI requests: (connect, read).
_PYPI_TIMEOUT = (5.0, 10.0)

# PEP 508 requirement names: letters, digits, ``.``, ``-``, ``_`` (not starting
# with a separator). We capture the name, an optional extras group, and the
# trailing specifier so we can pull out an exact pin if present.
_REQUIREMENT_RE = re.compile(
    r"""
    ^\s*
    (?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)   # distribution name
    \s*
    (?:\[[^\]]*\])?                          # optional extras, e.g. [security]
    \s*
    (?P<spec>[^;#]*)                         # version specifier(s), up to marker/comment
    """,
    re.VERBOSE,
)

# Matches an exact pin (``==1.2.3``) inside a specifier string, ignoring
# ``===`` arbitrary-equality and wildcard pins like ``==1.2.*``.
_EXACT_PIN_RE = re.compile(r"==\s*(?P<version>[A-Za-z0-9][A-Za-z0-9.\-+!]*)")


def normalize_name(name: str) -> str:
    """Normalize a distribution name per PEP 503 (lowercase, runs of ``-_.`` → ``-``)."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _extract_pin(spec: str) -> str | None:
    """Return the exact pinned version from a specifier, or ``None``.

    Only a true ``==x.y.z`` pin yields a version. Wildcard pins (``==1.2.*``)
    and arbitrary equality (``===foo``) are treated as unpinned because they do
    not name a single concrete version.
    """
    if not spec:
        return None
    for match in _EXACT_PIN_RE.finditer(spec):
        # Guard against ``===`` (arbitrary equality): the char before ``==``
        # must not be another ``=``.
        start = match.start()
        if start > 0 and spec[start - 1] == "=":
            continue
        version = match.group("version")
        if version.endswith("*"):
            continue
        return version
    return None


def _parse_requirement_line(line: str) -> Dependency | None:
    """Parse a single ``requirements.txt`` line into a Dependency, or ``None``.

    Returns ``None`` for blank lines, comments, and non-package directives
    (``-r``, ``-e``, ``--hash``, URLs, etc.) which do not declare a named,
    version-pinnable dependency in scope for this build.
    """
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    # Skip pip options/directives and direct URL / VCS installs.
    if stripped.startswith("-") or "://" in stripped.split("#", 1)[0]:
        return None

    match = _REQUIREMENT_RE.match(stripped)
    if not match:
        return None
    name = match.group("name")
    spec = (match.group("spec") or "").strip()
    return {"name": normalize_name(name), "current": _extract_pin(spec)}


def parse_requirements_txt(text: str) -> list[Dependency]:
    """Parse ``requirements.txt`` content into a list of Dependencies.

    Handles line continuations (trailing ``\\``), comments, and blank lines.
    Non-package directives are skipped.
    """
    deps: list[Dependency] = []
    buffer = ""
    for raw_line in text.splitlines():
        # Support backslash line continuations.
        if raw_line.rstrip().endswith("\\"):
            buffer += raw_line.rstrip()[:-1] + " "
            continue
        line = buffer + raw_line
        buffer = ""
        dep = _parse_requirement_line(line)
        if dep is not None:
            deps.append(dep)
    if buffer:  # trailing continuation with no final newline
        dep = _parse_requirement_line(buffer)
        if dep is not None:
            deps.append(dep)
    return deps


def _parse_poetry_dependency(name: str, spec: object) -> Dependency | None:
    """Parse a poetry-style dependency entry into a Dependency, or ``None``.

    Poetry declares versions either as a string (``"^1.2.3"``, ``"1.10.13"``)
    or a table (``{version = "1.2.3", ...}``). The implicit ``python`` entry is
    not a real distribution and is skipped by the caller.
    """
    version: str | None = None
    if isinstance(spec, str):
        version = spec
    elif isinstance(spec, dict):
        raw = spec.get("version")
        version = raw if isinstance(raw, str) else None
    else:
        return None

    current = _extract_pin(f"=={version.strip()}") if version else None
    # Poetry's bare ``"1.10.13"`` means an exact pin, unlike PEP 508 where a
    # bare version is invalid; treat a plain version string as a pin.
    if current is None and version and re.fullmatch(r"[0-9][0-9A-Za-z.\-+!]*", version.strip()):
        current = version.strip()
    return {"name": normalize_name(name), "current": current}


def parse_pyproject_toml(text: str) -> list[Dependency]:
    """Parse ``pyproject.toml`` content into a list of Dependencies.

    Supports both PEP 621 (``[project].dependencies`` as PEP 508 strings) and
    Poetry (``[tool.poetry.dependencies]`` tables). If both are present, PEP
    621 is preferred and Poetry entries are ignored to avoid duplicates.
    """
    data = tomllib.loads(text)
    deps: list[Dependency] = []

    project = data.get("project")
    if isinstance(project, dict) and isinstance(project.get("dependencies"), list):
        for entry in project["dependencies"]:
            if isinstance(entry, str):
                dep = _parse_requirement_line(entry)
                if dep is not None:
                    deps.append(dep)
        return deps

    # Fall back to Poetry layout.
    poetry = data.get("tool", {})
    poetry = poetry.get("poetry", {}) if isinstance(poetry, dict) else {}
    poetry_deps = poetry.get("dependencies") if isinstance(poetry, dict) else None
    if isinstance(poetry_deps, dict):
        for name, spec in poetry_deps.items():
            if name.lower() == "python":
                continue
            dep = _parse_poetry_dependency(name, spec)
            if dep is not None:
                deps.append(dep)
    return deps


def parse_dependencies(repo_path: str | Path) -> list[Dependency]:
    """Parse declared dependencies from a repository's manifest (no install).

    Prefers ``pyproject.toml`` when present, otherwise falls back to
    ``requirements.txt``. Reads the manifest only — nothing is installed and
    the local environment is never consulted (R1.1).

    Args:
        repo_path: Path to the repository root (or directly to a manifest file).

    Returns:
        A normalized list of ``{"name", "current"}`` dependency dicts. Empty if
        no supported manifest is found.

    Raises:
        FileNotFoundError: If ``repo_path`` does not exist.
        tomllib.TOMLDecodeError: If a ``pyproject.toml`` is malformed. (Callers
            in task 3.3 record this as a per-repo error and continue.)
    """
    path = Path(repo_path)
    if not path.exists():
        raise FileNotFoundError(f"repo path does not exist: {path}")

    # Allow pointing directly at a manifest file.
    if path.is_file():
        if path.name == "pyproject.toml":
            return parse_pyproject_toml(path.read_text(encoding="utf-8"))
        return parse_requirements_txt(path.read_text(encoding="utf-8"))

    pyproject = path / "pyproject.toml"
    if pyproject.is_file():
        return parse_pyproject_toml(pyproject.read_text(encoding="utf-8"))

    requirements = path / "requirements.txt"
    if requirements.is_file():
        return parse_requirements_txt(requirements.read_text(encoding="utf-8"))

    return []


# --- Latest-version resolution against PyPI (task 3.2) --------------------

# Fallback version tokenizer: split a version into leading numeric components so
# we can compare releases when ``packaging`` is unavailable. Only used on the
# no-``packaging`` path; it intentionally ignores pre-release/local segments.
_NUMERIC_SEGMENT_RE = re.compile(r"\d+")


def _version_key(version: str) -> tuple[int, ...]:
    """Return a tuple of leading integer segments for a coarse version compare.

    This is the fallback used only when ``packaging`` is not importable. It maps
    ``"1.10.13"`` → ``(1, 10, 13)`` so numeric ordering is respected (crucially,
    ``10 > 9``, which a naive string compare gets wrong). Non-numeric suffixes
    (``rc1``, ``+local``) are dropped, so it is deliberately conservative.
    """
    return tuple(int(seg) for seg in _NUMERIC_SEGMENT_RE.findall(version))


def is_newer(latest: str, current: str) -> bool:
    """Return ``True`` if ``latest`` is a strictly newer version than ``current``.

    Uses PEP 440-correct comparison via ``packaging`` when available, falling
    back to a numeric-segment comparison otherwise. On any parse failure the
    versions are compared as strings, and equal strings are never "newer".
    """
    if latest == current:
        return False
    if _HAS_PACKAGING:
        try:
            return Version(latest) > Version(current)
        except InvalidVersion:
            pass  # fall through to the conservative comparison below
    latest_key, current_key = _version_key(latest), _version_key(current)
    if latest_key and current_key:
        return latest_key > current_key
    return latest > current


def fetch_latest_version(
    name: str, *, session: requests.Session | None = None
) -> str | None:
    """Query the PyPI JSON API for a package's latest released version (R1.2).

    Resolves the latest version from the **registry**, never the local
    environment. Uses PyPI's ``info.version`` field, which reflects the latest
    non-yanked release.

    Args:
        name: The distribution name (normalized or not; PyPI is case- and
            separator-insensitive for lookups).
        session: Optional ``requests.Session`` to reuse a connection pool across
            many lookups. A one-off request is made when not provided.

    Returns:
        The latest version string, or ``None`` if the package is unknown on
        PyPI (HTTP 404) or the response lacks a usable version field.

    Raises:
        requests.RequestException: On network/HTTP errors other than a 404
            (timeouts, connection errors, 5xx). Callers decide how to surface
            these; task 3.3 records them per-repo and continues.
    """
    url = _PYPI_JSON_URL.format(name=normalize_name(name))
    getter = session.get if session is not None else requests.get
    response = getter(url, timeout=_PYPI_TIMEOUT)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    payload = response.json()
    info = payload.get("info") if isinstance(payload, dict) else None
    if isinstance(info, dict):
        version = info.get("version")
        if isinstance(version, str) and version:
            return version
    return None


def compute_outdated_packages(
    dependencies: list[Dependency], *, session: requests.Session | None = None
) -> list[OutdatedPackage]:
    """Compute outdated packages from parsed dependencies via PyPI (R1.2, R1.3).

    For each dependency with a known pinned ``current`` version, queries PyPI
    for the latest version and records it as outdated when the latest is
    strictly newer. Dependencies without a pinned ``current`` are skipped —
    there is no concrete version to compare against.

    Args:
        dependencies: Parsed dependencies from :func:`parse_dependencies`.
        session: Optional ``requests.Session`` reused across all lookups.

    Returns:
        A list of ``{"name", "current", "latest"}`` dicts, one per outdated
        package (R1.3), preserving the order of the input dependencies.

    Raises:
        requests.RequestException: Propagated from :func:`fetch_latest_version`
            on network/HTTP failures (see its docstring).
    """
    owns_session = session is None
    session = session or requests.Session()
    outdated: list[OutdatedPackage] = []
    try:
        for dep in dependencies:
            current = dep.get("current")
            name = dep.get("name")
            if not name or not current:
                continue
            latest = fetch_latest_version(name, session=session)
            if latest is not None and is_newer(latest, current):
                outdated.append({"name": name, "current": current, "latest": latest})
    finally:
        if owns_session:
            session.close()
    return outdated


# --- Monitor-level safe scan (task 3.3) ----------------------------------

# A structured, per-repository error recorded when a manifest cannot be parsed
# or a lookup fails. ``kind`` is a stable machine-readable category; ``message``
# is a human-readable explanation. Recording one of these lets the monitor keep
# going instead of crashing the whole run (R1.4).
MonitorError = dict[str, str]

# The result of scanning a single repository: the outdated packages found (may
# be empty) plus any errors recorded along the way (empty on the happy path).
# Shaped so callers can act on ``outdated`` while still surfacing ``errors``.
ScanResult = dict[str, object]


def _record_error(kind: str, message: str, repo_path: str | Path) -> MonitorError:
    """Build a structured monitor error for a repository (R1.4)."""
    return {"kind": kind, "message": message, "repo_path": str(repo_path)}


def scan_packages(
    repo_path: str | Path, *, session: requests.Session | None = None
) -> ScanResult:
    """Scan a repository for outdated packages, recording errors instead of crashing.

    This is the monitor-level entrypoint that composes :func:`parse_dependencies`
    and :func:`compute_outdated_packages` behind a safety boundary. When the
    manifest cannot be parsed — a malformed ``pyproject.toml``, a missing path,
    or any other read/parse failure — the failure is captured as a structured
    per-repository error and the scan returns normally instead of propagating
    the exception (R1.4). PyPI lookup failures during the outdated computation
    are likewise recorded rather than crashing the run.

    Design reference:
    - design.md → Error handling → "Manifest unparseable | Record error, skip
      repo, don't crash (R1.4)".
    - design.md → Agent layer → ``scan_packages(repo_path)`` monitor tool.

    Args:
        repo_path: Path to the repository root (or directly to a manifest file).
        session: Optional ``requests.Session`` reused across PyPI lookups.

    Returns:
        A :data:`ScanResult` dict ``{"repo_path", "outdated", "errors"}`` where
        ``outdated`` is the list of outdated packages (possibly empty) and
        ``errors`` is a list of :data:`MonitorError` dicts (empty when the scan
        succeeded cleanly).
    """
    errors: list[MonitorError] = []

    # --- Parse the manifest behind a safety boundary (R1.4). -------------
    try:
        dependencies = parse_dependencies(repo_path)
    except FileNotFoundError as exc:
        errors.append(_record_error("manifest_not_found", str(exc), repo_path))
        return {"repo_path": str(repo_path), "outdated": [], "errors": errors}
    except tomllib.TOMLDecodeError as exc:
        errors.append(
            _record_error("manifest_unparseable", f"invalid pyproject.toml: {exc}", repo_path)
        )
        return {"repo_path": str(repo_path), "outdated": [], "errors": errors}
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        # Unreadable file, bad encoding, or other manifest read/parse failure —
        # still a per-repo error, still keep going (R1.4).
        errors.append(
            _record_error("manifest_unparseable", f"could not read manifest: {exc}", repo_path)
        )
        return {"repo_path": str(repo_path), "outdated": [], "errors": errors}

    # --- Resolve outdated packages; PyPI failures are recorded, not fatal. -
    try:
        outdated = compute_outdated_packages(dependencies, session=session)
    except requests.RequestException as exc:
        errors.append(_record_error("registry_unavailable", f"PyPI lookup failed: {exc}", repo_path))
        return {"repo_path": str(repo_path), "outdated": [], "errors": errors}

    return {"repo_path": str(repo_path), "outdated": outdated, "errors": errors}
