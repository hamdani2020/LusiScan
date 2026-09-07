"""Integration test for the orchestrator loop (task 10.3).

Runs the full **Monitor → Planner → Executor → Validator** loop end-to-end
**locally** — no AWS, no network, no real GitHub — for both demo scenarios:

- ``requests`` 2.31.0 → latest: a safe patch/minor bump the (fake) planner rates
  ``high`` confidence / ``auto_fix``. Asserts a migration is written as
  ``pending_review`` and a PR is opened.
- ``pydantic`` 1.10.13 → 2.x: the human-in-the-loop major upgrade the (fake)
  planner rates ``low`` confidence / ``guided_pr`` with flagged breaking
  changes. Asserts a migration is written as ``pending_review`` and a guided PR
  is opened.

It then asserts the human-in-the-loop contract end-to-end: a recorded
``approved`` decision is honored on the **next** cycle — the PR is merged and the
migration transitions ``pending_review → approved → merged`` (R5.4/R5.5) — while
an unapproved migration is never merged.

Every external boundary is faked, reusing the exact conventions the existing
unit tests established:

- a fake ``StateStore`` table (the in-memory ``FakeTable`` from
  ``tests/test_state_store.py``),
- a fake PyGithub client (mirroring ``tests/test_github_tools.py``),
- a fake Bedrock/Nova model returning fixture JSON plans and summaries (the
  ``PlannerModel`` + ``ChangelogModel`` seams),
- a stubbed changelog network fetch (monkeypatching ``requests.get``).

_Requirements: 9.4 (demonstrate both demo scenarios end-to-end), 5.4/5.5
(human-in-the-loop: never merge without approval; honor the recorded decision
on the next cycle), 6.3 (persist state)._
"""

from __future__ import annotations

import pytest

from src import main
from src.main import DepGuardOrchestrator
from src.state import store as st
from src.state.store import StateStore


# --- Fake DynamoDB table (reused convention from test_state_store.py) ------


class FakeTable:
    """In-memory stand-in for a boto3 DynamoDB ``Table`` (see test_state_store)."""

    def __init__(self) -> None:
        self.items: dict[tuple[str, str], dict] = {}

    def put_item(self, *, Item: dict) -> None:
        self.items[(Item["pk"], Item["sk"])] = dict(Item)

    def get_item(self, *, Key: dict) -> dict:
        item = self.items.get((Key["pk"], Key["sk"]))
        return {"Item": dict(item)} if item is not None else {}

    def update_item(
        self,
        *,
        Key: dict,
        UpdateExpression: str,
        ExpressionAttributeNames: dict,
        ExpressionAttributeValues: dict,
        ReturnValues: str = "NONE",
    ) -> dict:
        item = self.items[(Key["pk"], Key["sk"])]
        attr = ExpressionAttributeNames["#s"]
        item[attr] = ExpressionAttributeValues[":to"]
        if ReturnValues == "ALL_NEW":
            return {"Attributes": dict(item)}
        return {}

    def query(self, *, pk: str, sk_prefix: str) -> dict:
        matched = [
            dict(item)
            for (ipk, isk), item in self.items.items()
            if ipk == pk and isk.startswith(sk_prefix)
        ]
        return {"Items": matched}


# --- Fake PyGithub object graph (mirrors test_github_tools.py) -------------


class FakeGitObject:
    def __init__(self, sha: str) -> None:
        self.sha = sha


class FakeGitRef:
    def __init__(self, sha: str) -> None:
        self.object = FakeGitObject(sha)
        self.edited_to: str | None = None

    def edit(self, sha: str) -> None:
        self.edited_to = sha
        self.object = FakeGitObject(sha)


class FakeCommit:
    def __init__(self, sha: str, tree: object | None = None) -> None:
        self.sha = sha
        self.tree = tree or object()


class FakeBranch:
    def __init__(self, sha: str) -> None:
        self.commit = FakeCommit(sha)


class FakePullRequest:
    def __init__(self, number: int) -> None:
        self.number = number
        self.html_url = f"https://github.com/acme/demo/pull/{number}"
        self.merged = False
        self.state = "open"

    def merge(self, commit_message: str | None = None) -> None:
        self.merged = True

    def edit(self, state: str | None = None) -> None:
        if state is not None:
            self.state = state


class FakeRepository:
    """A fake PyGithub ``Repository`` recording branch/commit/PR calls."""

    def __init__(self, *, base_sha: str = "basesha") -> None:
        self.base_sha = base_sha
        self.created_refs: list[dict] = []
        self.created_pulls: list[dict] = []
        self.branch_ref = FakeGitRef(base_sha)
        self._pulls: dict[int, FakePullRequest] = {}
        self._next_pr_number = 100

    def get_branch(self, name: str) -> FakeBranch:
        return FakeBranch(self.base_sha)

    def create_git_ref(self, ref: str, sha: str):
        self.created_refs.append({"ref": ref, "sha": sha})
        return FakeGitRef(sha)

    def get_git_ref(self, ref: str) -> FakeGitRef:
        return self.branch_ref

    def get_git_commit(self, sha: str) -> FakeCommit:
        return FakeCommit(sha, tree=object())

    def create_git_tree(self, elements, base_tree):
        return object()

    def create_git_commit(self, message, tree, parents):
        return FakeCommit("newcommitsha", tree=tree)

    def create_pull(self, *, title: str, body: str, head: str, base: str):
        number = self._next_pr_number
        self._next_pr_number += 1
        pull = FakePullRequest(number)
        self._pulls[number] = pull
        self.created_pulls.append(
            {"title": title, "body": body, "head": head, "base": base, "pr": pull}
        )
        return pull

    def get_pull(self, number: int) -> FakePullRequest:
        pull = self._pulls.get(number)
        if pull is None:
            pull = FakePullRequest(number)
            self._pulls[number] = pull
        return pull


class FakeGithub:
    def __init__(self, repo: FakeRepository) -> None:
        self._repo = repo

    def get_repo(self, name: str) -> FakeRepository:
        return self._repo


# --- Fake Nova model: fixture JSON plans + summaries -----------------------

# Fixture plans keyed by package. requests → high/auto_fix (safe bump);
# pydantic → low/guided_pr with flagged breaking changes (human-in-the-loop).
_FIXTURE_PLANS: dict[str, dict] = {
    "requests": {
        "confidence": "high",
        "strategy": "auto_fix",
        "estimated_risk": "low",
        "breaking_changes": [],
        "reasoning": "Safe patch bump with no breaking changes.",
    },
    "pydantic": {
        "confidence": "low",
        "strategy": "guided_pr",
        "estimated_risk": "high",
        "breaking_changes": [
            "BaseSettings moved to pydantic-settings",
            "class-based Config must become model_config",
        ],
        "reasoning": "Major 1->2 upgrade with breaking changes; needs judgment.",
    },
}


class FakeNovaModel:
    """A fake Bedrock/Nova client satisfying both PlannerModel and ChangelogModel.

    ``converse_json`` returns the fixture plan for whichever package the prompt
    names (no real Bedrock call); ``summarize`` returns a canned changelog
    summary. This keeps the whole planner + changelog path AWS-free.
    """

    def summarize(self, prompt: str) -> str:
        return "Summary: breaking changes and migration steps for the upgrade."

    def converse_json(self, prompt: str, **kwargs) -> dict:
        for package, plan in _FIXTURE_PLANS.items():
            if package in prompt:
                return dict(plan)
        # Default to a conservative plan if the package can't be identified.
        return dict(_FIXTURE_PLANS["pydantic"])


# --- Stub the changelog network fetch (no real GitHub Releases call) -------


def _fake_releases_payload(package_repo: str):
    """Return a canned GitHub Releases payload for the two demo repos."""
    if package_repo == "psf/requests":
        return [{"tag_name": "2.32.3", "body": "Patch fixes; no breaking changes."}]
    if package_repo == "pydantic/pydantic":
        return [
            {"tag_name": "2.0.0", "body": "Major release: BaseSettings moved; Config -> model_config."},
        ]
    return []


class _FakeResponse:
    def __init__(self, payload) -> None:
        self._payload = payload
        self.status_code = 200

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._payload


@pytest.fixture
def stub_changelog_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Monkeypatch ``changelog_tools.requests.get`` to avoid any network call."""

    def fake_get(url, params=None, timeout=None):
        # URL is https://api.github.com/repos/<owner>/<repo>/releases
        repo = url.split("/repos/", 1)[1].rsplit("/releases", 1)[0]
        return _FakeResponse(_fake_releases_payload(repo))

    from src.tools import changelog_tools

    monkeypatch.setattr(changelog_tools.requests, "get", fake_get)


# --- Source / manifest providers the Executor uses -------------------------

_PYDANTIC_SOURCE = (
    "from pydantic import BaseModel, validator\n\n\n"
    "class User(BaseModel):\n"
    "    email: str\n\n"
    "    class Config:\n"
    "        anystr_strip_whitespace = True\n\n"
    "    @validator('email')\n"
    "    def check(cls, v):\n"
    "        return v\n\n\n"
    "def summary(u):\n"
    "    return u.dict()\n"
)

_MANIFEST = (
    "[project]\n"
    'dependencies = [\n'
    '    "requests==2.31.0",\n'
    '    "pydantic==1.10.13",\n'
    "]\n"
)


def _source_provider(plan: dict) -> dict[str, str]:
    """Give the Executor the pydantic source to transform (requests bumps only)."""
    if plan.get("package") == "pydantic":
        return {"demo_app/models.py": _PYDANTIC_SOURCE}
    return {}


def _manifest_provider(plan: dict) -> tuple[str, str]:
    """Give the Executor the manifest to version-bump for each package."""
    return ("pyproject.toml", _MANIFEST)


# --- The controlled demo repo scan (no real filesystem / PyPI) -------------

# Fixed "outdated" result standing in for Monitor's PyPI-backed scan so the test
# is deterministic and offline. Shaped exactly like package_tools.scan_packages.
_SCAN_RESULT = {
    "repo_path": "acme/demo",
    "outdated": [
        {"name": "requests", "current": "2.31.0", "latest": "2.32.3"},
        {"name": "pydantic", "current": "1.10.13", "latest": "2.0.0"},
    ],
    "errors": [],
}


@pytest.fixture
def stub_scan(monkeypatch: pytest.MonkeyPatch) -> None:
    """Monkeypatch ``package_tools.scan_packages`` to a fixed offline scan."""
    from src.tools import package_tools

    monkeypatch.setattr(
        package_tools, "scan_packages", lambda repo_path, **kw: dict(_SCAN_RESULT)
    )


REPO = "acme/demo"


def _make_orchestrator(table: FakeTable, repo: FakeRepository) -> DepGuardOrchestrator:
    """Build an orchestrator wired to all the fakes."""
    return DepGuardOrchestrator(
        REPO,
        state=StateStore(table=table),
        planner_model=FakeNovaModel(),
        github_client=FakeGithub(repo),
        source_provider=_source_provider,
        manifest_provider=_manifest_provider,
    )


# --- The full-loop integration test ---------------------------------------


@pytest.mark.usefixtures("stub_changelog_network", "stub_scan")
class TestFullLoopBothDemoPackages:
    def test_first_cycle_opens_prs_and_writes_pending_migrations(self) -> None:
        table = FakeTable()
        repo = FakeRepository()
        orch = _make_orchestrator(table, repo)

        result = orch.run()

        # Both demo packages produced a migration this cycle.
        packages = {m["package"] for m in result["migrations"]}
        assert packages == {"requests", "pydantic"}
        assert result["outcome"] == "completed"

        store = StateStore(table=table)

        # requests: safe bump → high confidence / auto_fix, pending_review.
        req = store.get_migration(REPO, "requests", "2.32.3")
        assert req is not None
        assert req["status"] == st.STATUS_PENDING_REVIEW
        assert req["confidence"] == "high"
        assert req["strategy"] == "auto_fix"
        # A version bump was applied to the manifest → there are changes → a PR.
        assert req["pr_number"] is not None
        # The "safe auto-fix" side of R9.4, end-to-end: the manifest pin was
        # actually bumped in the diff (R3.2) — the loop produced a concrete,
        # reviewable change, not just a plan.
        assert "requests==2.31.0" in req["diff"]
        assert "requests==2.32.3" in req["diff"]

        # pydantic: major upgrade → low confidence / guided_pr, pending_review,
        # with flagged breaking changes surfaced for human judgment (R5.2).
        pyd = store.get_migration(REPO, "pydantic", "2.0.0")
        assert pyd is not None
        assert pyd["status"] == st.STATUS_PENDING_REVIEW
        assert pyd["confidence"] == "low"
        assert pyd["strategy"] == "guided_pr"
        assert pyd["flagged"]  # at least one flagged breaking change
        assert pyd["pr_number"] is not None
        # The "human-in-the-loop major upgrade" side of R9.4, end-to-end. A
        # guided_pr still carries the safe manifest version bump (R3.2)...
        assert "pydantic==1.10.13" in pyd["diff"]
        assert "pydantic==2.0.0" in pyd["diff"]
        # ...but the risky source code is left UNTOUCHED — no auto-fix transform
        # fired and the source file never appears in the diff (R3.3 / R5.2): the
        # class-based Config restructure is surfaced as a flag for a human, not
        # rewritten automatically.
        assert pyd["applied"] == []
        assert "demo_app/models.py" not in pyd["diff"]
        flagged_descriptions = " ".join(f["description"] for f in pyd["flagged"])
        assert "Config" in flagged_descriptions

        # A PR was opened for BOTH packages (R5.1 / R5.2).
        assert len(repo.created_pulls) == 2
        # Nothing was merged on the discovery cycle (R5.4).
        assert all(p["pr"].merged is False for p in repo.created_pulls)

        # A run log was written recording both packages (R6.3).
        logs = store.list_run_logs(REPO)
        assert len(logs) == 1
        assert set(logs[0]["packages_processed"]) == {"requests", "pydantic"}

    def test_recorded_approval_is_honored_on_next_cycle(self) -> None:
        # Cycle 1: discover + open PRs.
        table = FakeTable()
        repo = FakeRepository()
        orch = _make_orchestrator(table, repo)
        orch.run()

        store = StateStore(table=table)
        req = store.get_migration(REPO, "requests", "2.32.3")
        req_pr_number = req["pr_number"]

        # A human approves the requests migration in the (fake) control panel.
        store.put_decision(
            REPO, st.migration_id("requests", "2.32.3"), st.STATUS_APPROVED
        )

        # Cycle 2: the orchestrator reads the decision and acts on it.
        orch2 = _make_orchestrator(table, repo)
        result2 = orch2.run()

        # The approved requests migration was merged and transitioned to merged.
        actions = {a["package"]: a for a in result2["decisions"]}
        assert "requests" in actions
        assert actions["requests"]["merged"] is True
        assert actions["requests"]["status"] == st.STATUS_MERGED

        merged = store.get_migration(REPO, "requests", "2.32.3")
        assert merged["status"] == st.STATUS_MERGED

        # The actual PR object was merged (R5.4: only on explicit approval).
        assert repo.get_pull(req_pr_number).merged is True

        # The un-approved pydantic migration was NEVER merged and stays pending.
        pyd = store.get_migration(REPO, "pydantic", "2.0.0")
        assert pyd["status"] == st.STATUS_PENDING_REVIEW

    def test_ignored_decision_closes_pr_and_transitions_closed(self) -> None:
        table = FakeTable()
        repo = FakeRepository()
        orch = _make_orchestrator(table, repo)
        orch.run()

        store = StateStore(table=table)
        pyd_pr_number = store.get_migration(REPO, "pydantic", "2.0.0")["pr_number"]

        # A human ignores the pydantic migration.
        store.put_decision(
            REPO, st.migration_id("pydantic", "2.0.0"), st.STATUS_IGNORED
        )

        orch2 = _make_orchestrator(table, repo)
        orch2.run()

        pyd = store.get_migration(REPO, "pydantic", "2.0.0")
        assert pyd["status"] == st.STATUS_CLOSED
        # The PR was closed (skipped), never merged (R5.5 / R5.4).
        pull = repo.get_pull(pyd_pr_number)
        assert pull.state == "closed"
        assert pull.merged is False

    def test_no_decision_leaves_pr_open_and_migration_pending(self) -> None:
        # The "review / none" lane (R5.5): when a human records no decision (or
        # asks for further review), the orchestrator leaves the PR open and the
        # migration untouched at ``pending_review`` on the next cycle — nothing
        # is merged or closed.
        table = FakeTable()
        repo = FakeRepository()
        orch = _make_orchestrator(table, repo)
        orch.run()

        store = StateStore(table=table)
        req_pr_number = store.get_migration(REPO, "requests", "2.32.3")["pr_number"]
        pyd_pr_number = store.get_migration(REPO, "pydantic", "2.0.0")["pr_number"]

        # No decision is recorded for either migration.
        # Cycle 2: nothing actionable, so no decision-actions are produced and
        # both migrations stay pending with their PRs open.
        orch2 = _make_orchestrator(table, repo)
        result2 = orch2.run()

        assert result2["decisions"] == []

        req = store.get_migration(REPO, "requests", "2.32.3")
        pyd = store.get_migration(REPO, "pydantic", "2.0.0")
        assert req["status"] == st.STATUS_PENDING_REVIEW
        assert pyd["status"] == st.STATUS_PENDING_REVIEW

        # Neither PR was merged or closed — both remain open (R5.4).
        for number in (req_pr_number, pyd_pr_number):
            pull = repo.get_pull(number)
            assert pull.merged is False
            assert pull.state == "open"

    def test_explicit_review_decision_leaves_pr_open(self) -> None:
        # The ``_act_on_decision`` "leave open" guard (R5.4): if a decision that
        # is neither ``approved`` nor ``ignored`` is present (e.g. a ``review``
        # value written directly, since put_decision only accepts approve/ignore
        # from the control panel), the orchestrator must NOT merge or close —
        # it records a ``left_open`` action and leaves the migration pending.
        table = FakeTable()
        repo = FakeRepository()
        orch = _make_orchestrator(table, repo)
        orch.run()

        store = StateStore(table=table)
        req = store.get_migration(REPO, "requests", "2.32.3")
        req_pr_number = req["pr_number"]

        # Write a raw "review" decision item directly (bypassing put_decision's
        # approve/ignore validation) to exercise the orchestrator's leave-open
        # branch for a non-terminal decision value.
        mig_id = st.migration_id("requests", "2.32.3")
        table.put_item(
            Item={
                "pk": st.repo_pk(REPO),
                "sk": st.decision_sk(mig_id),
                "entity": "decision",
                "migration_id": mig_id,
                "decision": "review",
            }
        )

        orch2 = _make_orchestrator(table, repo)
        result2 = orch2.run()

        actions = {a["package"]: a for a in result2["decisions"]}
        assert "requests" in actions
        assert actions["requests"]["decision"] == "review"
        assert actions["requests"]["merged"] is False
        assert actions["requests"]["pr_state"] == "left_open"
        assert actions["requests"]["status"] == st.STATUS_PENDING_REVIEW

        # The migration stays pending and the PR is neither merged nor closed.
        assert (
            store.get_migration(REPO, "requests", "2.32.3")["status"]
            == st.STATUS_PENDING_REVIEW
        )
        pull = repo.get_pull(req_pr_number)
        assert pull.merged is False
        assert pull.state == "open"

    def test_a_run_log_is_persisted_on_every_cycle(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # R6.3: each cycle persists a run log. After two cycles there are two
        # run-log entries recording the run outcome, so the state table is a
        # durable audit trail of every pass. Run-log sort keys are timestamped
        # at seconds precision, so we feed distinct, monotonically increasing
        # timestamps to keep the two entries from colliding on the same key.
        clock = iter(
            ["2024-01-01T00:00:00Z", "2024-01-01T00:00:01Z", "2024-01-01T00:00:02Z"]
        )
        monkeypatch.setattr(st, "_now_iso", lambda: next(clock))

        table = FakeTable()
        repo = FakeRepository()

        _make_orchestrator(table, repo).run()
        _make_orchestrator(table, repo).run()

        store = StateStore(table=table)
        logs = store.list_run_logs(REPO)
        assert len(logs) == 2
        assert all(log["outcome"] == "completed" for log in logs)


# --- Strands import-guard sanity (task 10.1) -------------------------------


def test_tool_decorator_is_importable_without_strands() -> None:
    # The stage functions stay directly callable whether or not the SDK is
    # present; in this environment the identity-decorator fallback is exercised.
    assert callable(main.scan_packages)
    assert callable(main.fetch_changelog)
    assert callable(main.plan_migration)
    assert callable(main.apply_migration)
    assert callable(main.validate_branch)


def test_validate_branch_uses_injected_validator_seam() -> None:
    # The task-12 seam: an injected validator is used verbatim; the default is
    # the stub returning the {status, details} shape.
    default = main.validate_branch("some-branch")
    assert set(default) == {"status", "details"}
    assert default["status"] == main.VALIDATION_SKIPPED

    def fake_validator(branch: str) -> dict:
        return {"status": main.VALIDATION_PASSED, "details": f"ci green on {branch}"}

    injected = main.validate_branch("b", validator=fake_validator)
    assert injected["status"] == main.VALIDATION_PASSED
    assert "ci green on b" in injected["details"]
