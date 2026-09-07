"""Unit tests for the AgentCore Runtime entrypoint (task 11.1).

Exercises the entrypoint's *pure* logic without the ``bedrock_agentcore`` SDK
(which isn't installed here), proving the module imports and stays testable via
its import guard, and that it:

- wires ``payload["repo_name"]`` through to the orchestrator and returns the
  orchestrator's structured summary (design.md → AgentCore entrypoint),
- returns a structured error — rather than crashing — when ``repo_name`` is
  missing or empty (R6.2),
- exposes ``invoke`` as a plain callable through the fallback ``app`` (so the
  decorated entrypoint is directly testable without the SDK).

Every boundary is faked/injected, mirroring the DI conventions the existing
tests use (``test_orchestrator.py`` fakes the store/github/model; here we inject
a fake orchestrator via the ``orchestrator_factory`` seam). All config values in
these tests are inert placeholders — never real credentials (R6.5).

_Requirements: 6.1 (run the agent loop on AgentCore Runtime), 6.2 (invoke per
repo, handle bad payloads), 6.5 (config/secrets from the environment, never
committed)._
"""

from __future__ import annotations

import pytest

from src import agentcore_app


# --- Fakes ----------------------------------------------------------------


class FakeOrchestrator:
    """Records the repo it was built for and returns a canned run summary."""

    def __init__(self, repo_name: str, summary: dict) -> None:
        self.repo_name = repo_name
        self._summary = summary
        self.run_calls = 0

    def run(self) -> dict:
        self.run_calls += 1
        return self._summary


def _summary(repo: str) -> dict:
    """A representative orchestrator summary (the shape ``run()`` returns)."""
    return {
        "repo": repo,
        "decisions": [],
        "migrations": [{"package": "requests", "target": "2.32.0"}],
        "errors": [],
        "outcome": "completed",
    }


# --- Module stays importable without the SDK (import guard, task 11.1) ----


def test_module_imports_without_sdk_and_exposes_callable_entrypoint():
    """Without the SDK, the fallback app yields a plain, callable ``invoke``."""
    # If the SDK is absent, the guard must have selected the fallback app.
    if not agentcore_app._HAS_AGENTCORE:
        assert agentcore_app._HAS_AGENTCORE is False
    # Either way, ``invoke`` is a directly callable function (identity decorator
    # under the fallback; the real decorator also keeps it callable).
    assert callable(agentcore_app.invoke)


# --- Happy path: wire repo_name through, return the summary ---------------


def test_run_for_repo_wires_repo_name_and_returns_summary():
    """``repo_name`` reaches the orchestrator; its summary is returned + status."""
    built: dict = {}

    def fake_factory(repo: str, **overrides):
        orch = FakeOrchestrator(repo, _summary(repo))
        built["orchestrator"] = orch
        return orch

    result = agentcore_app.run_for_repo(
        {"repo_name": "octocat/hello-world"},
        orchestrator_factory=fake_factory,
    )

    # The orchestrator was constructed for the payload's repo and run once.
    assert built["orchestrator"].repo_name == "octocat/hello-world"
    assert built["orchestrator"].run_calls == 1

    # The response carries the orchestrator's summary plus a completed status.
    assert result["status"] == "completed"
    assert result["repo"] == "octocat/hello-world"
    assert result["outcome"] == "completed"
    assert result["migrations"] == [{"package": "requests", "target": "2.32.0"}]
    assert result["decisions"] == []
    assert result["errors"] == []


def test_invoke_delegates_to_run_for_repo(monkeypatch):
    """The decorated ``invoke`` handler delegates to the pure ``run_for_repo``."""
    fake = FakeOrchestrator("owner/repo", _summary("owner/repo"))
    monkeypatch.setattr(
        agentcore_app, "_build_orchestrator", lambda repo, **kw: fake
    )

    result = agentcore_app.invoke({"repo_name": "owner/repo"})

    assert fake.run_calls == 1
    assert result["status"] == "completed"
    assert result["repo"] == "owner/repo"


# --- Error path: missing / empty repo_name must not crash (R6.2) ----------


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"repo_name": ""},
        {"repo_name": None},
        None,
    ],
)
def test_run_for_repo_missing_repo_returns_structured_error(payload):
    """A missing/empty/None ``repo_name`` yields a structured error, no crash."""
    calls = {"n": 0}

    def factory_should_not_run(repo: str, **overrides):
        calls["n"] += 1
        raise AssertionError("orchestrator must not be built without a repo")

    result = agentcore_app.run_for_repo(
        payload, orchestrator_factory=factory_should_not_run
    )

    assert result["status"] == "error"
    assert "repo_name" in result["error"]
    # The orchestrator factory was never invoked for a bad payload.
    assert calls["n"] == 0


def test_invoke_missing_repo_returns_structured_error():
    """The decorated entrypoint also surfaces the structured error path."""
    result = agentcore_app.invoke({})
    assert result["status"] == "error"
    assert "repo_name" in result["error"]


# --- Config wiring from the environment (R6.5, no committed secrets) ------


def test_build_orchestrator_reads_config_from_env(monkeypatch):
    """``_build_orchestrator`` passes env-derived table/token to the orchestrator."""
    captured: dict = {}

    class RecordingOrchestrator:
        def __init__(self, repo_name, **kwargs):
            captured["repo_name"] = repo_name
            captured.update(kwargs)

    monkeypatch.setattr(agentcore_app, "DepGuardOrchestrator", RecordingOrchestrator)
    monkeypatch.setenv(agentcore_app.ENV_STATE_TABLE, "lusiscan-state")
    monkeypatch.setenv(agentcore_app.ENV_GITHUB_TOKEN, "placeholder-token")

    agentcore_app._build_orchestrator("owner/repo")

    assert captured["repo_name"] == "owner/repo"
    assert captured["table_name"] == "lusiscan-state"
    assert captured["github_token"] == "placeholder-token"


def test_build_orchestrator_omits_missing_config(monkeypatch):
    """With no env config set, table/token pass through as ``None`` (no defaults)."""
    captured: dict = {}

    class RecordingOrchestrator:
        def __init__(self, repo_name, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(agentcore_app, "DepGuardOrchestrator", RecordingOrchestrator)
    monkeypatch.delenv(agentcore_app.ENV_STATE_TABLE, raising=False)
    monkeypatch.delenv(agentcore_app.ENV_STATE_TABLE_FALLBACK, raising=False)
    monkeypatch.delenv(agentcore_app.ENV_GITHUB_TOKEN, raising=False)

    agentcore_app._build_orchestrator("owner/repo")

    assert captured["table_name"] is None
    assert captured["github_token"] is None
