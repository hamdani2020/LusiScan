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
    # Keep the test hermetic: with GITHUB_TOKEN unset the resolver would fall
    # through to Secrets Manager (a live AWS call), so stub the SM read to None.
    monkeypatch.setattr(agentcore_app, "_read_secret", lambda secret_id: None)

    agentcore_app._build_orchestrator("owner/repo")

    assert captured["table_name"] is None
    assert captured["github_token"] is None


# --- GitHub token from Secrets Manager at runtime (R6.5) ------------------


class _FakeSecretsClient:
    """Minimal stand-in for a boto3 ``secretsmanager`` client."""

    def __init__(self, *, secret_string=None, raises=None) -> None:
        self._secret_string = secret_string
        self._raises = raises
        self.requested_ids: list[str] = []

    def get_secret_value(self, *, SecretId):
        self.requested_ids.append(SecretId)
        if self._raises is not None:
            raise self._raises
        return {"SecretString": self._secret_string}


def _patch_boto3(monkeypatch, fake_client):
    """Patch ``boto3.client`` (imported lazily inside ``_read_secret``)."""
    import boto3

    monkeypatch.setattr(boto3, "client", lambda service, **kw: fake_client)


def test_resolve_github_token_prefers_env(monkeypatch):
    """An explicit ``GITHUB_TOKEN`` env var wins over Secrets Manager."""
    monkeypatch.setenv(agentcore_app.ENV_GITHUB_TOKEN, "env-token")
    # If the resolver reached Secrets Manager this would blow up the test.
    monkeypatch.setattr(
        agentcore_app,
        "_read_secret",
        lambda secret_id: (_ for _ in ()).throw(AssertionError("should not read SM")),
    )

    assert agentcore_app._resolve_github_token() == "env-token"


def test_resolve_github_token_reads_secrets_manager_when_env_absent(monkeypatch):
    """With no env var, the token is read from the default SM secret id."""
    monkeypatch.delenv(agentcore_app.ENV_GITHUB_TOKEN, raising=False)
    monkeypatch.delenv(agentcore_app.ENV_GITHUB_TOKEN_SECRET_ID, raising=False)
    seen: dict = {}

    def fake_read(secret_id):
        seen["secret_id"] = secret_id
        return "sm-token"

    monkeypatch.setattr(agentcore_app, "_read_secret", fake_read)

    assert agentcore_app._resolve_github_token() == "sm-token"
    assert seen["secret_id"] == agentcore_app.DEFAULT_GITHUB_TOKEN_SECRET_ID


def test_resolve_github_token_honors_custom_secret_id(monkeypatch):
    """``GITHUB_TOKEN_SECRET_ID`` overrides which secret is read."""
    monkeypatch.delenv(agentcore_app.ENV_GITHUB_TOKEN, raising=False)
    monkeypatch.setenv(agentcore_app.ENV_GITHUB_TOKEN_SECRET_ID, "custom/token")
    seen: dict = {}
    monkeypatch.setattr(
        agentcore_app, "_read_secret", lambda secret_id: seen.setdefault("id", secret_id)
    )

    agentcore_app._resolve_github_token()
    assert seen["id"] == "custom/token"


def test_read_secret_returns_raw_string(monkeypatch):
    """A raw (non-JSON) secret string is returned verbatim (trimmed)."""
    _patch_boto3(monkeypatch, _FakeSecretsClient(secret_string="  ghp_raw123  "))
    assert agentcore_app._read_secret("lusiscan/github-token") == "ghp_raw123"


def test_read_secret_parses_json_token_field(monkeypatch):
    """A JSON secret object is parsed for a recognized token field."""
    _patch_boto3(
        monkeypatch, _FakeSecretsClient(secret_string='{"token": "ghp_json456"}')
    )
    assert agentcore_app._read_secret("lusiscan/github-token") == "ghp_json456"


def test_read_secret_parses_github_token_json_key(monkeypatch):
    """The ``GITHUB_TOKEN`` JSON key is also recognized."""
    _patch_boto3(
        monkeypatch,
        _FakeSecretsClient(secret_string='{"GITHUB_TOKEN": "ghp_key789"}'),
    )
    assert agentcore_app._read_secret("lusiscan/github-token") == "ghp_key789"


def test_read_secret_returns_none_on_client_error(monkeypatch):
    """Any Secrets Manager failure degrades to ``None`` (never crashes)."""
    _patch_boto3(
        monkeypatch, _FakeSecretsClient(raises=RuntimeError("AccessDenied"))
    )
    assert agentcore_app._read_secret("lusiscan/github-token") is None


def test_read_secret_returns_none_on_empty(monkeypatch):
    """An empty ``SecretString`` yields ``None``."""
    _patch_boto3(monkeypatch, _FakeSecretsClient(secret_string=""))
    assert agentcore_app._read_secret("lusiscan/github-token") is None
