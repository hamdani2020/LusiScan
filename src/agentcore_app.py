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


def _build_orchestrator(repo: str, **overrides: Any) -> DepGuardOrchestrator:
    """Construct the orchestrator for ``repo``, wiring runtime config from env.

    Reads the DynamoDB table name and GitHub token from the environment and
    forwards them to :class:`~src.main.DepGuardOrchestrator` (design.md →
    Security; R6.5) — never hardcoding or logging secret material. Any keyword
    in ``overrides`` wins over the env-derived defaults, which is the seam tests
    use to inject a fake orchestrator/config without touching the environment.
    """
    table_name = os.environ.get(ENV_STATE_TABLE) or os.environ.get(
        ENV_STATE_TABLE_FALLBACK
    )
    github_token = os.environ.get(ENV_GITHUB_TOKEN)

    kwargs: dict[str, Any] = {
        "table_name": table_name,
        "github_token": github_token,
    }
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
