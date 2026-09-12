"""AgentCore Runtime entrypoint: wrap the orchestrator loop (task 11.1).

This module is the thin host boundary the deploy flow points at. LusiScan runs
its agent loop on **Amazon Bedrock AgentCore Runtime** (R6.1) — meaning we write
the Monitor→Planner→Executor→Validator loop ourselves (in
:class:`src.main.DepGuardOrchestrator`) and wrap it with a
``BedrockAgentCoreApp`` so the runtime can host and invoke it. This is
deliberately **AgentCore Runtime**, not the older ``AWS::Bedrock::Agent``
action-group product (design.md → API correction #2: "AgentCore ≠ Bedrock
Agents").

## The entrypoint contract (design.md → AgentCore entrypoint)

The runtime invokes the decorated ``@app.entrypoint`` with a JSON ``payload``.
The handler pulls ``repo_name`` out of the payload, runs one full orchestrator
cycle for that repo, and returns a structured result. The deploy flow that
targets this file (subtasks 11.2/11.3) is::

    agentcore configure --entrypoint src/agentcore_app.py
    agentcore launch
    agentcore invoke '{"repo_name": "owner/repo"}'

Running this module directly (``python -m src.agentcore_app`` /
``python src/agentcore_app.py``) starts AgentCore's local HTTP server on
``:8080`` so the entrypoint can be exercised before deploying.

## Degrading gracefully without the SDK (task 11.1)

The ``bedrock_agentcore`` SDK is not importable in every environment (it isn't
in unit tests). To keep this module importable — and the pure entrypoint logic
(``payload`` → orchestrator → result dict) directly unit-testable — the
``BedrockAgentCoreApp`` import is resolved behind a safe import guard with a
minimal fallback ``app`` object that still provides ``entrypoint`` (an identity
decorator) and ``run`` (raising a clear error if the real server is asked for
without the SDK). This mirrors the import-guard / lazy-dependency pattern
already used across the codebase (the ``strands`` guard in ``src/main.py`` and
lazy ``boto3`` in ``store.py`` / ``bedrock_client.py``): either way the module
loads and :func:`invoke` stays callable.

## Config + secrets (design.md → Security; R6.5)

This module never hardcodes a GitHub token, AWS credentials, or a table name.
Runtime config is read from the environment (which AgentCore populates from
Secrets Manager / task config at runtime) and passed through to the
orchestrator: the DynamoDB table name from ``DEPGUARD_STATE_TABLE`` (falling
back to ``STATE_TABLE_NAME``) and the GitHub token from ``GITHUB_TOKEN``. When
neither table env var is set, the table name is simply omitted and the
orchestrator/``StateStore`` resolve it themselves — no secret material is read,
logged, or committed here.

Design references:
- design.md → API correction #2: deploy to AgentCore *Runtime* and wrap the loop
  with ``BedrockAgentCoreApp`` (not the ``AWS::Bedrock::Agent`` product).
- design.md → AgentCore entrypoint: ``payload["repo_name"]`` →
  ``DepGuardOrchestrator(repo_name=repo).run()`` → structured result;
  ``app.run()`` for the local ``:8080`` server.
- requirements.md → R6.1 (run the agent loop on AgentCore Runtime),
  R6.2 (invoke per repo), R6.5 (secrets from Secrets Manager at runtime, never
  committed).
"""

from __future__ import annotations

import os
from typing import Any, Callable, Optional

from src.main import DepGuardOrchestrator
from src.models.bedrock_client import BedrockClient
from src.tools import github_tools


# --- BedrockAgentCoreApp import guard (task 11.1) -------------------------
#
# The real SDK wraps our loop for the AgentCore Runtime host. When it is absent
# (e.g. in unit tests), we fall back to a minimal stand-in that still exposes the
# two members this module uses — ``entrypoint`` (an identity decorator, so the
# handler stays a plain callable) and ``run`` (which raises a clear error rather
# than pretending to serve). This keeps the module importable and the pure
# ``invoke`` logic testable without the SDK, mirroring the ``strands`` guard in
# ``src/main.py``.
try:  # pragma: no cover - both branches are trivial; exercised via import
    from bedrock_agentcore.runtime import BedrockAgentCoreApp as _BedrockAgentCoreApp

    _HAS_AGENTCORE = True
except Exception:  # noqa: BLE001 - any import failure degrades to the fallback app

    class _BedrockAgentCoreApp:  # type: ignore[no-redef]
        """Minimal stand-in for ``BedrockAgentCoreApp`` when the SDK is absent.

        Provides just enough surface for this module to import and be tested:

        - :meth:`entrypoint` is an identity decorator, so the handler it wraps
          stays a plain, directly-callable function (exactly how the real
          decorated entrypoint is invoked in tests).
        - :meth:`run` raises, because there is no local ``:8080`` server to start
          without the real SDK — surfacing a clear, actionable error instead of
          silently doing nothing.
        """

        def entrypoint(self, func: Callable[..., Any]) -> Callable[..., Any]:
            return func

        def run(self) -> None:  # pragma: no cover - only hit if run without SDK
            raise RuntimeError(
                "bedrock_agentcore is not installed; install it to run the "
                "local AgentCore server on :8080 (see task 11 deploy flow)."
            )

    _HAS_AGENTCORE = False


BedrockAgentCoreApp = _BedrockAgentCoreApp


# The AgentCore app instance the runtime hosts and invokes.
app = BedrockAgentCoreApp()


# --- Runtime config env-var names (design.md → Security; R6.5) ------------
#
# Read at invoke time so the entrypoint picks up whatever AgentCore injects from
# Secrets Manager / task config, without any secret ever living in source.
ENV_STATE_TABLE = "DEPGUARD_STATE_TABLE"
ENV_STATE_TABLE_FALLBACK = "STATE_TABLE_NAME"
ENV_GITHUB_TOKEN = "GITHUB_TOKEN"

# --- Runtime secrets from AWS Secrets Manager (design.md → Security; R6.5) --
#
# The GitHub token is read from Secrets Manager **at invoke time** so nothing
# sensitive is baked into the image or the runtime config. The secret id is
# overridable (kept in sync with the IAM grant on ``lusiscan/github-token-*``).
ENV_GITHUB_TOKEN_SECRET_ID = "GITHUB_TOKEN_SECRET_ID"
DEFAULT_GITHUB_TOKEN_SECRET_ID = "lusiscan/github-token"


def _read_secret(secret_id: str) -> Optional[str]:
    """Return the string value of a Secrets Manager secret, or ``None``.

    Reads ``secret_id`` via ``secretsmanager:GetSecretValue`` using the runtime
    role's credentials (the IAM policy scopes this to ``lusiscan/github-token-*``
    / ``lusiscan/slack-webhook-*``; R6.5). ``boto3`` is imported lazily so the
    module stays importable — and unit-testable — without AWS installed.

    The secret may be stored either as a raw token string or as a small JSON
    object (``{"token": ...}`` / ``{"GITHUB_TOKEN": ...}``); both are supported.
    Any failure (missing secret, no access, no boto3) degrades to ``None`` so a
    misconfigured secret never crashes the runtime host — and the secret value
    is **never** logged.
    """
    try:
        import json

        import boto3  # noqa: PLC0415 - intentional lazy import (see docstring)

        client = boto3.client(
            "secretsmanager", region_name=os.environ.get("AWS_REGION")
        )
        secret_string = client.get_secret_value(SecretId=secret_id).get(
            "SecretString"
        )
        if not secret_string:
            return None

        # Accept either a raw string or a JSON object with a token field.
        try:
            parsed = json.loads(secret_string)
        except (ValueError, TypeError):
            return secret_string.strip()
        if isinstance(parsed, dict):
            for key in ("token", "GITHUB_TOKEN", "github_token", "value"):
                if parsed.get(key):
                    return str(parsed[key]).strip()
            return None
        return str(parsed).strip()
    except Exception:  # noqa: BLE001 - any failure degrades to no token (R6.2/6.5)
        return None


def _resolve_github_token() -> Optional[str]:
    """Resolve the GitHub token, preferring Secrets Manager at runtime (R6.5).

    Resolution order, most-explicit first:

    1. ``GITHUB_TOKEN`` env var — an explicit override for local runs (and the
       legacy ``agentcore launch --env`` path); handy for development.
    2. AWS Secrets Manager ``lusiscan/github-token`` (id overridable via
       ``GITHUB_TOKEN_SECRET_ID``) — the runtime path, so no secret is stored on
       the runtime config or in the image.

    Returns ``None`` when neither yields a token; GitHub calls then surface a
    clear credential error rather than the entrypoint crashing. No secret value
    is ever logged.
    """
    env_token = os.environ.get(ENV_GITHUB_TOKEN)
    if env_token:
        return env_token
    secret_id = os.environ.get(
        ENV_GITHUB_TOKEN_SECRET_ID, DEFAULT_GITHUB_TOKEN_SECRET_ID
    )
    return _read_secret(secret_id)


def _fetch_repo_files(repo: str, github_token: Optional[str]) -> tuple[
    Optional[str],
    Optional[Callable[[dict], tuple[str, str]]],
    Optional[Callable[[dict], dict[str, str]]],
]:
    """Materialize the target repo's manifest + source so the loop runs live.

    The Monitor (:func:`src.main.scan_packages`) reads a *filesystem* manifest,
    and the Executor transforms *source files* passed via provider seams. On the
    AgentCore runtime the repo isn't checked out, so we fetch what the loop needs
    directly from GitHub (PyGithub, authenticated with the runtime token) and:

    - write the fetched ``pyproject.toml`` into a temp dir returned as
      ``repo_path`` (so the Monitor detects the outdated pins), and
    - wire a ``manifest_provider`` (version-bump the same manifest) and a
      ``source_provider`` (the Python sources the auto-fix transforms rewrite).

    Only ``pyproject.toml`` and any ``demo_app/*.py`` sources are fetched — the
    demo scope (design.md → "controlled demo repo"). Any fetch failure degrades
    to ``(None, None, None)`` so the caller falls back to scanning ``repo`` as a
    local path rather than crashing (R1.4). No secret material is logged here.

    Returns:
        ``(repo_path, manifest_provider, source_provider)``; each element is
        ``None`` when unavailable.
    """
    import tempfile
    from pathlib import Path

    try:
        client = github_tools.get_client(token=github_token)
        gh_repo = client.get_repo(repo)

        manifest_content = gh_repo.get_contents("pyproject.toml").decoded_content.decode(
            "utf-8"
        )

        # Fetch the Python sources under demo_app/ (the auto-fix transform scope).
        source_files: dict[str, str] = {}
        try:
            for entry in gh_repo.get_contents("demo_app"):
                if entry.type == "file" and entry.path.endswith(".py"):
                    source_files[entry.path] = entry.decoded_content.decode("utf-8")
        except Exception:  # noqa: BLE001 - no demo_app/ is fine (version-bump-only)
            pass
    except Exception:  # noqa: BLE001 - any GitHub failure: fall back to local scan
        return None, None, None

    # Write the manifest into a temp dir for the Monitor to scan.
    tmp_dir = tempfile.mkdtemp(prefix="lusiscan-repo-")
    (Path(tmp_dir) / "pyproject.toml").write_text(manifest_content, encoding="utf-8")

    def manifest_provider(_plan: dict) -> tuple[str, str]:
        return ("pyproject.toml", manifest_content)

    def source_provider(_plan: dict) -> dict[str, str]:
        return dict(source_files)

    return tmp_dir, manifest_provider, source_provider


def _build_orchestrator(repo: str, **overrides: Any) -> DepGuardOrchestrator:
    """Construct the orchestrator for ``repo``, wiring runtime config from env.

    Reads the DynamoDB table name from the environment and resolves the GitHub
    token from AWS Secrets Manager (falling back to ``GITHUB_TOKEN`` for local
    runs) — never hardcoding or logging secret material (design.md → Security;
    R6.5). Any keyword in ``overrides`` wins over the env-derived defaults, which
    is the seam tests use to inject a fake orchestrator/config without touching
    the environment.
    """
    table_name = os.environ.get(ENV_STATE_TABLE) or os.environ.get(
        ENV_STATE_TABLE_FALLBACK
    )
    github_token = _resolve_github_token()

    kwargs: dict[str, Any] = {
        "table_name": table_name,
        "github_token": github_token,
        # Wire the real Nova reasoning client so the loop can actually plan and
        # execute migrations on the live runtime. ``BedrockClient`` satisfies
        # both PlannerModel (``converse_json``) and ChangelogModel
        # (``summarize``); it builds its ``bedrock-runtime`` client lazily using
        # the runtime role's credentials (R8.1/8.2) — no secret material here.
        # Tests override this via ``overrides`` to inject a fake model.
        "planner_model": BedrockClient(region_name=os.environ.get("AWS_REGION")),
    }

    # Materialize the target repo's manifest + source from GitHub so the Monitor
    # detects the outdated pins and the Executor has the sources to transform.
    # Degrades to a plain local scan of ``repo`` if the fetch fails.
    repo_path, manifest_provider, source_provider = _fetch_repo_files(
        repo, github_token
    )
    if repo_path is not None:
        kwargs["repo_path"] = repo_path
        kwargs["manifest_provider"] = manifest_provider
        kwargs["source_provider"] = source_provider

    kwargs.update(overrides)
    return DepGuardOrchestrator(repo_name=repo, **kwargs)


def run_for_repo(
    payload: dict,
    *,
    orchestrator_factory: Optional[Callable[..., DepGuardOrchestrator]] = None,
) -> dict:
    """Pure entrypoint logic: ``payload`` → orchestrator cycle → result dict.

    Kept separate from the ``@app.entrypoint``-decorated :func:`invoke` so the
    wiring (payload parsing → orchestrator construction → summary) is directly
    unit-testable without the AgentCore SDK. ``orchestrator_factory`` is the DI
    seam tests use to inject a fake orchestrator; when omitted it is resolved at
    call time to :func:`_build_orchestrator` (which reads config from the
    environment) — resolving lazily rather than binding a def-time default keeps
    the production path monkeypatch-friendly.

    A missing or empty ``repo_name`` is returned as a structured error rather
    than raising, so a bad payload never crashes the runtime host (R6.2).

    Args:
        payload: The invocation payload; must include a non-empty ``repo_name``.
        orchestrator_factory: Optional callable ``(repo, **overrides) ->
            orchestrator``. Defaults (when ``None``) to :func:`_build_orchestrator`.

    Returns:
        On success, the orchestrator's structured summary
        (``{"repo", "decisions", "migrations", "errors", "outcome"}``) augmented
        with ``{"status": "completed"}``. On a missing ``repo_name``,
        ``{"status": "error", "error": "..."}``.
    """
    repo = (payload or {}).get("repo_name")
    if not repo:
        return {
            "status": "error",
            "error": "payload must include a non-empty 'repo_name'",
        }

    factory = orchestrator_factory or _build_orchestrator
    orchestrator = factory(repo)
    summary = orchestrator.run()
    return {"status": "completed", **summary}


@app.entrypoint
def invoke(payload: dict) -> dict:
    """AgentCore Runtime entrypoint (design.md → AgentCore entrypoint).

    Thin wrapper the runtime calls with the invocation ``payload``. Delegates to
    :func:`run_for_repo` for the testable payload→orchestrator→result logic.

    Args:
        payload: The JSON payload the runtime forwards (e.g.
            ``{"repo_name": "owner/repo"}``).

    Returns:
        The structured result dict from :func:`run_for_repo`.
    """
    return run_for_repo(payload)


if __name__ == "__main__":
    # Start AgentCore's local HTTP server on :8080 so the entrypoint can be
    # exercised before deploying (design.md → AgentCore entrypoint).
    app.run()
