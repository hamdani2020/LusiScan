"""Test validation via GitHub Actions, with a local-sandbox pytest fallback.

This module implements the **ValidatorAgent** boundary (design.md → Agent layer:
``branch → {status, details}``). After the ExecutorAgent has pushed a migration
commit to a temporary branch, the Validator answers one question: *do the
target repository's tests pass on that branch?*

Scope (requirements.md → R4):

- **R4.1 / R4.2** — trigger the target repo's GitHub Actions workflow on the
  branch and read the workflow-run ``status`` / ``conclusion`` via the **Actions
  API** (``get_workflow_runs``), never the legacy commit-status API.
- **R4.3** — while a run has not ``completed``, treat the result as pending and
  keep polling up to a configured timeout.
- **R4.4** — when a run completes, classify it as ``passed`` (conclusion
  ``success``) or ``failed`` (any other conclusion) — see
  :func:`conclusion_to_status`.
- **R4.5** — if CI does not complete within the timeout, report a ``timeout``
  and escalate to human review.
- **R4.6** — if the GitHub Actions round-trip is unavailable, run the repo's
  tests locally in the sandbox and return the **same** ``{status, details}``
  shape (:class:`LocalSandboxValidator`).

Every validator here is a plain callable ``(branch) -> {"status", "details"}``,
matching the injectable ``Validator`` seam the orchestrator already exposes
(``src/main.py`` → ``validate_branch(branch, *, validator=...)``). That means
task 12 plugs in without reworking the loop: the orchestrator just passes
``validator=GitHubActionsValidator(...)``.

Secrets handling (design.md → Security; R6.5): the GitHub validator never
hardcodes a token. It reuses :func:`github_tools.get_client`, which accepts an
injected client, a token string, or reads ``GITHUB_TOKEN`` (at runtime, injected
from Secrets Manager). No secret is read, logged, or committed here.

Design references:
- design.md → API correction #3: read GitHub Actions workflow-run ``conclusion``
  via the Actions API, not the legacy commit-status API.
- design.md → Error handling: "Model/CI pending → poll to timeout, then
  escalate".
- requirements.md → R4.1–R4.6.
"""

from __future__ import annotations

import subprocess
import time
from typing import Any, Callable, Optional

from src.tools import github_tools


# --- Result-shape contract (design.md → ValidatorAgent) -------------------
#
# Every validator returns ``{"status": <one of below>, "details": <str>}`` so
# the orchestrator can persist it verbatim as a migration's ``test_summary``
# (src/main.py → _persist_migration) and the Streamlit panel can render it.
STATUS_PASSED = "passed"
STATUS_FAILED = "failed"
STATUS_TIMEOUT = "timeout"
STATUS_ERROR = "error"

# GitHub Actions run lifecycle values we care about (Actions API).
_RUN_COMPLETED = "completed"
_CONCLUSION_SUCCESS = "success"

# Polling defaults (R4.3): how long to wait for a run to complete, and how often
# to re-check. Kept small so the demo's fast test suite resolves in seconds.
DEFAULT_TIMEOUT_SECONDS = 300
DEFAULT_POLL_INTERVAL_SECONDS = 10

# The validator callable contract: a branch name in, a result dict out.
Validator = Callable[[str], dict]


def conclusion_to_status(conclusion: Optional[str]) -> str:
    """Map a GitHub Actions run ``conclusion`` to ``passed`` / ``failed`` (R4.4).

    Per R4.4 the classification is deliberately binary and conservative: only a
    ``success`` conclusion counts as ``passed``; **any** other conclusion
    (``failure``, ``cancelled``, ``timed_out``, ``action_required``, ``neutral``,
    ``skipped``, ``stale``, or ``None``) is treated as ``failed`` so a migration
    is never reported as tested-green unless CI actually went green.

    Args:
        conclusion: The workflow run's ``conclusion`` field (may be ``None`` for
            a run that has not reached a conclusion).

    Returns:
        :data:`STATUS_PASSED` when ``conclusion == "success"``, else
        :data:`STATUS_FAILED`.
    """
    if conclusion == _CONCLUSION_SUCCESS:
        return STATUS_PASSED
    return STATUS_FAILED


class GitHubActionsValidator:
    """Poll a branch's GitHub Actions run and classify the result (R4.1–R4.5).

    A callable ``(branch) -> {"status", "details"}``. When invoked it:

    1. Resolves the target repo via :func:`github_tools.get_client` (R6.5).
    2. Optionally triggers the workflow on the branch via ``workflow_dispatch``
       (R4.1). The executor's push to the branch usually already triggers a
       ``push`` run, so dispatch is best-effort and never fatal.
    3. Finds the most recent workflow run for the branch and polls its
       ``status`` via the Actions API (R4.2) while it is not ``completed``,
       sleeping ``poll_interval`` seconds between checks, up to ``timeout``
       (R4.3).
    4. On completion, maps ``conclusion`` → ``passed`` / ``failed``
       (:func:`conclusion_to_status`, R4.4).
    5. On non-completion within ``timeout``, returns ``timeout`` and escalates
       to human review (R4.5).

    All time is injected via ``sleep`` / ``monotonic`` so tests run instantly and
    deterministically without real waiting.
    """

    def __init__(
        self,
        repo_name: str,
        *,
        client: Any | None = None,
        token: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        poll_interval: float = DEFAULT_POLL_INTERVAL_SECONDS,
        dispatch: bool = True,
        workflow_file: Optional[str] = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        """Configure the validator.

        Args:
            repo_name: ``owner/repo`` slug of the target repository.
            client: Optional pre-authenticated PyGithub client (injected in
                tests / at runtime).
            token: Optional token string (falls back to ``GITHUB_TOKEN``).
            timeout: Max seconds to wait for a run to complete (R4.3/4.5).
            poll_interval: Seconds to sleep between status checks (R4.3).
            dispatch: Whether to best-effort ``workflow_dispatch`` the workflow
                on the branch before polling (R4.1).
            workflow_file: Optional workflow filename (e.g. ``tests.yml``) to
                dispatch; when omitted the first repo workflow is used.
            sleep: Injectable sleep (defaults to ``time.sleep``); tests pass a
                no-op.
            monotonic: Injectable monotonic clock; tests pass a fake to advance
                time deterministically.
        """
        self.repo_name = repo_name
        self._client = client
        self._token = token
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.dispatch = dispatch
        self.workflow_file = workflow_file
        self._sleep = sleep
        self._monotonic = monotonic

    def __call__(self, branch: str) -> dict:
        """Validate ``branch`` and return the ``{status, details}`` result."""
        if not branch:
            return {
                "status": STATUS_ERROR,
                "details": "no branch provided to validate",
            }
        try:
            repo = self._resolve_repo()
        except Exception as exc:  # noqa: BLE001 - surface as data, never crash the loop
            return {
                "status": STATUS_ERROR,
                "details": f"could not resolve repo '{self.repo_name}': {exc}",
            }

        if self.dispatch:
            self._try_dispatch(repo, branch)

        return self._poll_for_result(repo, branch)

    # -- internals ---------------------------------------------------------

    def _resolve_repo(self) -> Any:
        """Resolve the target repository via the shared client helper (R6.5)."""
        client = github_tools.get_client(client=self._client, token=self._token)
        return client.get_repo(self.repo_name)

    def _try_dispatch(self, repo: Any, branch: str) -> None:
        """Best-effort ``workflow_dispatch`` on ``branch`` (R4.1).

        The executor's push to the branch typically already triggers a ``push``
        run, so an explicit dispatch is a belt-and-suspenders trigger. Any
        failure (no dispatch trigger configured, transient error) is swallowed —
        the subsequent poll still finds the push-triggered run.
        """
        try:
            workflows = list(repo.get_workflows())
            if not workflows:
                return
            workflow = workflows[0]
            if self.workflow_file is not None:
                for wf in workflows:
                    if str(getattr(wf, "path", "")).endswith(self.workflow_file):
                        workflow = wf
                        break
            workflow.create_dispatch(branch)
        except Exception:  # noqa: BLE001 - dispatch is best-effort (see docstring)
            return

    def _latest_run_for_branch(self, repo: Any, branch: str) -> Any | None:
        """Return the most recent workflow run for ``branch`` (Actions API, R4.2)."""
        try:
            runs = repo.get_workflow_runs(branch=branch)
        except TypeError:
            # Some fakes/clients don't accept the branch kwarg; fall back to all.
            runs = repo.get_workflow_runs()
        latest = None
        for run in runs:
            # ``get_workflow_runs(branch=...)`` already filters, but guard fakes
            # that ignore the filter by matching head_branch when present.
            head = getattr(run, "head_branch", branch)
            if head not in (branch, None):
                continue
            if latest is None:
                latest = run
                continue
            if _run_sort_key(run) > _run_sort_key(latest):
                latest = run
        return latest

    def _poll_for_result(self, repo: Any, branch: str) -> dict:
        """Poll the branch's run to completion or timeout (R4.3/4.4/4.5)."""
        deadline = self._monotonic() + self.timeout
        last_run = None

        while True:
            run = self._latest_run_for_branch(repo, branch)
            if run is not None:
                last_run = run
                # Refresh the run's fields from the API before reading them.
                _refresh(run)
                if getattr(run, "status", None) == _RUN_COMPLETED:
                    conclusion = getattr(run, "conclusion", None)
                    status = conclusion_to_status(conclusion)
                    return {
                        "status": status,
                        "details": _completed_details(run, branch, conclusion),
                    }

            if self._monotonic() >= deadline:
                # R4.5: CI did not complete in time — escalate to human review.
                return {
                    "status": STATUS_TIMEOUT,
                    "details": _timeout_details(last_run, branch, self.timeout),
                }

            self._sleep(self.poll_interval)


class LocalSandboxValidator:
    """Run the repo's tests locally in the sandbox (R4.6 fallback).

    A callable ``(branch) -> {"status", "details"}`` that shells out to
    ``pytest`` in a working directory and returns the **same** result shape as
    :class:`GitHubActionsValidator`. This is the R4.6 fallback for when the
    GitHub Actions round-trip is unavailable (no token, no network, workflow
    disabled): LusiScan can still produce an equivalent pass/fail signal.

    The subprocess call is injectable (``runner``) so tests exercise the mapping
    without actually spawning pytest.
    """

    def __init__(
        self,
        *,
        working_dir: Optional[str] = None,
        command: Optional[list[str]] = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        runner: Optional[Callable[..., Any]] = None,
    ) -> None:
        """Configure the sandbox validator.

        Args:
            working_dir: Directory to run pytest in (the repo checkout). When
                ``None``, pytest runs in the current working directory.
            command: The pytest command to run (defaults to ``["pytest", "-q"]``).
            timeout: Max seconds for the pytest process (R4.3-equivalent).
            runner: Injectable process runner with ``subprocess.run`` semantics
                (defaults to :func:`subprocess.run`); tests pass a fake.
        """
        self.working_dir = working_dir
        self.command = command or ["pytest", "-q"]
        self.timeout = timeout
        self._runner = runner or subprocess.run

    def __call__(self, branch: str) -> dict:
        """Run the tests locally and return the ``{status, details}`` result."""
        try:
            completed = self._runner(
                self.command,
                cwd=self.working_dir,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            return {
                "status": STATUS_TIMEOUT,
                "details": (
                    f"local pytest exceeded {self.timeout}s for branch "
                    f"'{branch}'; escalating to human review"
                ),
            }
        except Exception as exc:  # noqa: BLE001 - missing pytest / bad cwd, etc.
            return {
                "status": STATUS_ERROR,
                "details": f"local pytest could not run for branch '{branch}': {exc}",
            }

        return_code = getattr(completed, "returncode", 1)
        status = STATUS_PASSED if return_code == 0 else STATUS_FAILED
        tail = _tail(getattr(completed, "stdout", "") or getattr(completed, "stderr", ""))
        return {
            "status": status,
            "details": (
                f"local pytest for branch '{branch}' exited {return_code} "
                f"({status}). {tail}"
            ).strip(),
        }


def make_validator(
    repo_name: str,
    *,
    client: Any | None = None,
    token: Optional[str] = None,
    local_fallback_dir: Optional[str] = None,
    **kwargs: Any,
) -> Validator:
    """Build the default Validator: GitHub Actions with a local-sandbox fallback.

    Returns a callable ``(branch) -> {status, details}`` that first tries the
    :class:`GitHubActionsValidator`. If that returns an ``error`` status (e.g.
    the Actions round-trip is unavailable — no token/network, R4.6) **and** a
    ``local_fallback_dir`` is provided, it falls back to
    :class:`LocalSandboxValidator` over that checkout, returning the same shape.

    Args:
        repo_name: ``owner/repo`` slug of the target repository.
        client: Optional pre-authenticated PyGithub client.
        token: Optional token string (falls back to ``GITHUB_TOKEN``).
        local_fallback_dir: Optional local checkout dir enabling the R4.6 pytest
            fallback when the Actions path is unavailable.
        **kwargs: Forwarded to :class:`GitHubActionsValidator` (``timeout``,
            ``poll_interval``, ``dispatch``, ``workflow_file``, clock injectors).

    Returns:
        A ``Validator`` callable.
    """
    actions = GitHubActionsValidator(
        repo_name, client=client, token=token, **kwargs
    )

    if local_fallback_dir is None:
        return actions

    sandbox = LocalSandboxValidator(working_dir=local_fallback_dir)

    def validator(branch: str) -> dict:
        result = actions(branch)
        if result.get("status") == STATUS_ERROR:
            return sandbox(branch)
        return result

    return validator


# --- small helpers --------------------------------------------------------


def _run_sort_key(run: Any) -> Any:
    """Sort key preferring ``created_at`` then ``id`` (most recent last)."""
    created = getattr(run, "created_at", None)
    if created is not None:
        return (1, created)
    return (0, getattr(run, "id", 0))


def _refresh(run: Any) -> None:
    """Best-effort refresh of a run's fields from the API (``run.update()``)."""
    update = getattr(run, "update", None)
    if callable(update):
        try:
            update()
        except Exception:  # noqa: BLE001 - a stale read is fine; next poll retries
            return


def _completed_details(run: Any, branch: str, conclusion: Optional[str]) -> str:
    """Human-readable detail line for a completed run."""
    url = getattr(run, "html_url", "") or ""
    suffix = f" ({url})" if url else ""
    return (
        f"GitHub Actions run for branch '{branch}' completed with conclusion "
        f"'{conclusion}'{suffix}"
    )


def _timeout_details(run: Any, branch: str, timeout: float) -> str:
    """Human-readable detail line for a timeout escalation (R4.5)."""
    if run is None:
        return (
            f"no GitHub Actions run reached completion for branch '{branch}' "
            f"within {timeout}s; escalating to human review"
        )
    status = getattr(run, "status", "unknown")
    url = getattr(run, "html_url", "") or ""
    suffix = f" ({url})" if url else ""
    return (
        f"GitHub Actions run for branch '{branch}' did not complete within "
        f"{timeout}s (last status '{status}'){suffix}; escalating to human review"
    )


def _tail(text: str, *, max_chars: int = 500) -> str:
    """Return the trailing ``max_chars`` of ``text`` (for compact log details)."""
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return "…" + text[-max_chars:]
