"""Unit tests for the ValidatorAgent (task 12): GitHub Actions + local sandbox.

These tests exercise the ``branch -> {status, details}`` validator contract in
isolation — no network, no real GitHub, no real pytest — by injecting fakes for
the PyGithub workflow API, the clock (``sleep`` / ``monotonic``), and the
subprocess runner. They cover:

- the ``conclusion`` → pass/fail mapping (R4.4),
- the Actions poller: pending → completed (success and failure), and the
  head_branch/latest-run selection (R4.1/4.2/4.3/4.4),
- timeout → escalate to human (R4.5),
- best-effort ``workflow_dispatch`` (R4.1) and graceful error handling,
- the local-sandbox pytest fallback returning the same shape (R4.6),
- ``make_validator`` falling back to the sandbox when Actions errors.

_Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6._
"""

from __future__ import annotations

import subprocess

import pytest

from src.tools import validation_tools as v


# --- Fake PyGithub workflow object graph ----------------------------------


class FakeWorkflowRun:
    """A workflow run whose ``status`` advances through a scripted sequence.

    Each call to :meth:`update` pops the next ``(status, conclusion)`` from the
    script (staying on the last one), mimicking the Actions API transitioning a
    run from ``queued`` → ``in_progress`` → ``completed``.
    """

    def __init__(self, script, *, head_branch="lusiscan/x", run_id=1, created_at=1):
        self._script = list(script)
        self._idx = 0
        self.head_branch = head_branch
        self.id = run_id
        self.created_at = created_at
        self.html_url = f"https://github.com/acme/demo/actions/runs/{run_id}"
        self.status, self.conclusion = self._script[0]

    def update(self):
        # Advance one step per refresh (clamp at the last scripted state).
        if self._idx < len(self._script) - 1:
            self._idx += 1
        self.status, self.conclusion = self._script[self._idx]


class FakeWorkflow:
    def __init__(self, path="tests.yml"):
        self.path = f".github/workflows/{path}"
        self.dispatch_calls = []

    def create_dispatch(self, ref):
        self.dispatch_calls.append(ref)
        return True


class FakeRepo:
    """A minimal stand-in for a PyGithub ``Repository`` (Actions surface)."""

    def __init__(self, runs=None, workflows=None, *, accept_branch_kwarg=True):
        self._runs = runs or []
        self._workflows = workflows if workflows is not None else [FakeWorkflow()]
        self._accept_branch_kwarg = accept_branch_kwarg
        self.get_workflow_runs_calls = []

    def get_workflows(self):
        return list(self._workflows)

    def get_workflow_runs(self, *, branch=None, **kwargs):
        if not self._accept_branch_kwarg and branch is not None:
            raise TypeError("this fake does not accept branch kwarg")
        self.get_workflow_runs_calls.append(branch)
        return list(self._runs)


class FakeClient:
    """A fake PyGithub client that returns a preconfigured repo."""

    def __init__(self, repo):
        self._repo = repo
        self.requested = None

    def get_repo(self, name):
        self.requested = name
        return self._repo


class _Clock:
    """A fake monotonic clock advanced explicitly by ``sleep``."""

    def __init__(self):
        self.t = 0.0

    def monotonic(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds


def _validator(repo, **kwargs):
    """Build a GitHubActionsValidator wired to a fake client + instant clock."""
    clock = _Clock()
    kwargs.setdefault("dispatch", False)
    return v.GitHubActionsValidator(
        "acme/demo",
        client=FakeClient(repo),
        sleep=clock.sleep,
        monotonic=clock.monotonic,
        **kwargs,
    )


# =====================================================================
# conclusion -> status mapping (R4.4)
# =====================================================================


@pytest.mark.parametrize(
    "conclusion,expected",
    [
        ("success", v.STATUS_PASSED),
        ("failure", v.STATUS_FAILED),
        ("cancelled", v.STATUS_FAILED),
        ("timed_out", v.STATUS_FAILED),
        ("action_required", v.STATUS_FAILED),
        ("neutral", v.STATUS_FAILED),
        ("skipped", v.STATUS_FAILED),
        (None, v.STATUS_FAILED),
    ],
)
def test_conclusion_to_status(conclusion, expected):
    # Only "success" is passed; everything else (incl. None) is failed (R4.4).
    assert v.conclusion_to_status(conclusion) == expected


# =====================================================================
# Actions poller (R4.1/4.2/4.3/4.4)
# =====================================================================


def test_poller_pending_then_completed_success():
    # A run that starts queued, goes in_progress, then completes green -> passed.
    run = FakeWorkflowRun(
        [("queued", None), ("in_progress", None), ("completed", "success")]
    )
    result = _validator(FakeRepo([run]), poll_interval=5, timeout=300)("lusiscan/x")

    assert result["status"] == v.STATUS_PASSED
    assert "completed with conclusion 'success'" in result["details"]


def test_poller_completed_failure_maps_to_failed():
    run = FakeWorkflowRun([("in_progress", None), ("completed", "failure")])
    result = _validator(FakeRepo([run]), poll_interval=5, timeout=300)("lusiscan/x")

    assert result["status"] == v.STATUS_FAILED
    assert "failure" in result["details"]


def test_poller_reads_actions_api_with_branch_filter():
    # R4.2: the poller must query the Actions API filtered by branch.
    run = FakeWorkflowRun([("completed", "success")], head_branch="lusiscan/x")
    repo = FakeRepo([run])
    _validator(repo, poll_interval=1, timeout=60)("lusiscan/x")

    assert repo.get_workflow_runs_calls  # was queried
    assert repo.get_workflow_runs_calls[0] == "lusiscan/x"


def test_poller_selects_most_recent_run_for_branch():
    # Given multiple runs, the newest (by created_at) is polled.
    old = FakeWorkflowRun([("completed", "failure")], run_id=1, created_at=1)
    new = FakeWorkflowRun([("completed", "success")], run_id=2, created_at=5)
    result = _validator(FakeRepo([old, new]), poll_interval=1, timeout=60)(
        "lusiscan/x"
    )
    # The newer run concluded success -> passed.
    assert result["status"] == v.STATUS_PASSED


def test_poller_ignores_runs_for_other_branches():
    other = FakeWorkflowRun([("completed", "success")], head_branch="main")
    repo = FakeRepo([other], accept_branch_kwarg=False)  # forces client-side filter
    result = _validator(repo, poll_interval=1, timeout=60)("lusiscan/x")
    # No run matches the branch -> never completes -> timeout escalation (R4.5).
    assert result["status"] == v.STATUS_TIMEOUT


# =====================================================================
# Timeout -> escalate to human (R4.5)
# =====================================================================


def test_poller_times_out_when_run_never_completes():
    # A run stuck in_progress forever -> timeout + escalation message (R4.5).
    run = FakeWorkflowRun([("in_progress", None)])
    result = _validator(FakeRepo([run]), poll_interval=10, timeout=30)("lusiscan/x")

    assert result["status"] == v.STATUS_TIMEOUT
    assert "escalating to human review" in result["details"]


def test_poller_times_out_when_no_run_found():
    # No runs at all -> timeout escalation with the "no run reached completion".
    result = _validator(FakeRepo([]), poll_interval=10, timeout=20)("lusiscan/x")

    assert result["status"] == v.STATUS_TIMEOUT
    assert "escalating to human review" in result["details"]


def test_empty_branch_returns_error_without_calling_github():
    result = v.GitHubActionsValidator("acme/demo", client=FakeClient(FakeRepo([])))("")
    assert result["status"] == v.STATUS_ERROR
    assert "no branch" in result["details"]


def test_repo_resolution_failure_returns_error_not_crash():
    class BoomClient:
        def get_repo(self, name):
            raise RuntimeError("boom")

    result = v.GitHubActionsValidator("acme/demo", client=BoomClient())("lusiscan/x")
    assert result["status"] == v.STATUS_ERROR
    assert "could not resolve repo" in result["details"]


# =====================================================================
# workflow_dispatch (R4.1) is best-effort
# =====================================================================


def test_dispatch_triggers_workflow_on_branch():
    wf = FakeWorkflow("tests.yml")
    run = FakeWorkflowRun([("completed", "success")])
    validator = _validator(
        FakeRepo([run], workflows=[wf]), dispatch=True, poll_interval=1, timeout=60
    )
    validator("lusiscan/x")
    assert wf.dispatch_calls == ["lusiscan/x"]


def test_dispatch_failure_is_swallowed_and_polling_continues():
    class BoomWorkflow(FakeWorkflow):
        def create_dispatch(self, ref):
            raise RuntimeError("no dispatch trigger")

    run = FakeWorkflowRun([("completed", "success")])
    repo = FakeRepo([run], workflows=[BoomWorkflow()])
    result = _validator(repo, dispatch=True, poll_interval=1, timeout=60)("lusiscan/x")
    # Dispatch blew up, but the push-triggered run still resolves to passed.
    assert result["status"] == v.STATUS_PASSED


# =====================================================================
# Local-sandbox pytest fallback (R4.6)
# =====================================================================


class _CompletedProcess:
    def __init__(self, returncode, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_sandbox_passes_on_zero_exit():
    runner = lambda *a, **k: _CompletedProcess(0, stdout="3 passed in 0.1s")
    result = v.LocalSandboxValidator(runner=runner)("lusiscan/x")
    assert result["status"] == v.STATUS_PASSED
    assert "3 passed" in result["details"]


def test_sandbox_fails_on_nonzero_exit():
    runner = lambda *a, **k: _CompletedProcess(1, stdout="1 failed, 2 passed")
    result = v.LocalSandboxValidator(runner=runner)("lusiscan/x")
    assert result["status"] == v.STATUS_FAILED
    assert "1 failed" in result["details"]


def test_sandbox_timeout_maps_to_timeout_status():
    def runner(*a, **k):
        raise subprocess.TimeoutExpired(cmd="pytest", timeout=5)

    result = v.LocalSandboxValidator(timeout=5, runner=runner)("lusiscan/x")
    assert result["status"] == v.STATUS_TIMEOUT
    assert "human review" in result["details"]


def test_sandbox_missing_pytest_maps_to_error():
    def runner(*a, **k):
        raise FileNotFoundError("pytest not found")

    result = v.LocalSandboxValidator(runner=runner)("lusiscan/x")
    assert result["status"] == v.STATUS_ERROR
    assert "could not run" in result["details"]


def test_sandbox_uses_configured_working_dir_and_command():
    seen = {}

    def runner(command, *, cwd, capture_output, text, timeout):
        seen["command"] = command
        seen["cwd"] = cwd
        return _CompletedProcess(0)

    v.LocalSandboxValidator(
        working_dir="/tmp/checkout", command=["pytest", "-x"], runner=runner
    )("lusiscan/x")

    assert seen["command"] == ["pytest", "-x"]
    assert seen["cwd"] == "/tmp/checkout"


# =====================================================================
# make_validator: default + local fallback wiring (R4.6)
# =====================================================================


def test_make_validator_returns_actions_validator_without_fallback():
    validator = v.make_validator("acme/demo", client=FakeClient(FakeRepo([])))
    assert isinstance(validator, v.GitHubActionsValidator)


def test_make_validator_falls_back_to_sandbox_on_actions_error(tmp_path, monkeypatch):
    # Actions path errors (repo resolution fails) -> sandbox fallback runs.
    class BoomClient:
        def get_repo(self, name):
            raise RuntimeError("no network")

    # Stub the sandbox's runner via subprocess.run so no real pytest is spawned.
    monkeypatch.setattr(
        v.subprocess, "run", lambda *a, **k: _CompletedProcess(0, stdout="ok")
    )

    validator = v.make_validator(
        "acme/demo", client=BoomClient(), local_fallback_dir=str(tmp_path)
    )
    result = validator("lusiscan/x")
    assert result["status"] == v.STATUS_PASSED


def test_make_validator_no_fallback_surfaces_actions_error():
    class BoomClient:
        def get_repo(self, name):
            raise RuntimeError("no network")

    validator = v.make_validator("acme/demo", client=BoomClient())
    result = validator("lusiscan/x")
    assert result["status"] == v.STATUS_ERROR
