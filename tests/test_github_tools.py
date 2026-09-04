"""Unit tests for the GitHub integration tools (tasks 8.1 and 8.2).

Covers the executor-facing PR creation path — create a branch off the base,
commit the migration's changed files in one commit, and open a pull request
returning ``{branch, pr_url, pr_number, diff, changes}`` (R5.1, R5.2) — and,
critically, the merge-on-approval boundary that must **never** merge without an
explicit recorded human approval (R5.4) and must honor a recorded decision on
the next cycle (R5.5).

No real GitHub API call is made: a small fake PyGithub client records the calls
it receives and replays canned objects, so the tests assert behavior (branch
created, single commit built, PR opened, merge refused/permitted) without any
network access.

_Requirements: 5.1 (high-confidence → open PR + ready-to-approve), 5.2
(low-confidence → guided PR with version bump / auto-fixes / flagged changes),
5.4 (never merge without explicit human approval), 5.5 (act on the recorded
decision: merge / leave open / close)._
"""

from __future__ import annotations

import pytest

from src.tools import github_tools as gt


# --- Fake PyGithub object graph -------------------------------------------


class FakeGitObject:
    """Stand-in for a git ref's ``object`` (carries the target sha)."""

    def __init__(self, sha: str) -> None:
        self.sha = sha


class FakeGitRef:
    """A fake git ref that records ``edit(sha=...)`` calls (branch moves)."""

    def __init__(self, sha: str) -> None:
        self.object = FakeGitObject(sha)
        self.edited_to: str | None = None

    def edit(self, sha: str) -> None:
        self.edited_to = sha
        self.object = FakeGitObject(sha)


class FakeCommit:
    """A fake commit carrying its sha and tree."""

    def __init__(self, sha: str, tree: object | None = None) -> None:
        self.sha = sha
        self.tree = tree or object()


class FakeBranch:
    """A fake branch whose ``commit`` exposes a sha (for ``get_branch``)."""

    def __init__(self, sha: str) -> None:
        self.commit = FakeCommit(sha)


class FakePullRequest:
    """A fake pull request recording merge/edit calls (the R5.4/R5.5 surface)."""

    def __init__(self, number: int) -> None:
        self.number = number
        self.html_url = f"https://github.com/acme/demo/pull/{number}"
        self.merged = False
        self.merge_message: str | None = None
        self.state = "open"
        self.edit_calls: list[str] = []

    def merge(self, commit_message: str | None = None) -> None:
        self.merged = True
        self.merge_message = commit_message

    def edit(self, state: str | None = None) -> None:
        if state is not None:
            self.state = state
            self.edit_calls.append(state)


class FakeRepository:
    """A fake PyGithub ``Repository`` recording the calls the tools make.

    It replays a base branch sha, hands back fake git refs/commits/trees for the
    Git Data API commit path, and produces / stores fake pull requests so tests
    can assert what was created and later merged/closed.
    """

    def __init__(self, *, base_sha: str = "basesha", existing_branches=None) -> None:
        self.base_sha = base_sha
        self.created_refs: list[dict] = []
        self.created_trees: list[dict] = []
        self.created_commits: list[dict] = []
        self.created_pulls: list[dict] = []
        self.branch_ref = FakeGitRef(base_sha)
        self._pulls: dict[int, FakePullRequest] = {}
        self._next_pr_number = 42
        # Names for which create_git_ref should raise a 422 (already exists).
        self._existing_branches = set(existing_branches or [])
        self.create_ref_raise: gt.GithubException | None = None

    # -- refs / branches --
    def get_branch(self, name: str) -> FakeBranch:
        return FakeBranch(self.base_sha)

    def create_git_ref(self, ref: str, sha: str):
        if self.create_ref_raise is not None:
            raise self.create_ref_raise
        branch = ref.replace("refs/heads/", "")
        if branch in self._existing_branches:
            raise gt.GithubException(422, {"message": "Reference already exists"}, None)
        self.created_refs.append({"ref": ref, "sha": sha})
        return FakeGitRef(sha)

    def get_git_ref(self, ref: str) -> FakeGitRef:
        return self.branch_ref

    # -- git data API (commit) --
    def get_git_commit(self, sha: str) -> FakeCommit:
        return FakeCommit(sha, tree=object())

    def create_git_tree(self, elements, base_tree):
        self.created_trees.append({"elements": elements, "base_tree": base_tree})
        return object()

    def create_git_commit(self, message, tree, parents):
        self.created_commits.append(
            {"message": message, "tree": tree, "parents": parents}
        )
        return FakeCommit("newcommitsha", tree=tree)

    # -- pull requests --
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
    """A fake PyGithub ``Github`` client returning a single fake repository."""

    def __init__(self, repo: FakeRepository) -> None:
        self._repo = repo
        self.requested_repos: list[str] = []

    def get_repo(self, name: str) -> FakeRepository:
        self.requested_repos.append(name)
        return self._repo


# --- get_client / credential resolution (R6.5, no committed secrets) ------


class TestGetClient:
    def test_injected_client_is_used_as_is(self) -> None:
        fake = FakeGithub(FakeRepository())
        assert gt.get_client(client=fake) is fake

    def test_missing_credentials_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        with pytest.raises(ValueError):
            gt.get_client()

    def test_env_token_is_read_when_no_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A placeholder value only; never a real secret. Building a client from a
        # token must not require network access.
        monkeypatch.setenv("GITHUB_TOKEN", "x-not-a-real-token")
        client = gt.get_client()
        assert isinstance(client, gt.Github)


# --- create_branch (task 8.1) ---------------------------------------------


class TestCreateBranch:
    def test_creates_branch_off_base_head(self) -> None:
        repo = FakeRepository(base_sha="abc123")
        name = gt.create_branch(repo, "lusiscan/pydantic-2.9.2", base_branch="main")
        assert name == "lusiscan/pydantic-2.9.2"
        assert repo.created_refs == [
            {"ref": "refs/heads/lusiscan/pydantic-2.9.2", "sha": "abc123"}
        ]

    def test_existing_branch_is_reused_not_failed(self) -> None:
        # A 422 (already exists) is swallowed so a re-run does not crash.
        repo = FakeRepository(existing_branches={"lusiscan/pydantic-2.9.2"})
        name = gt.create_branch(repo, "lusiscan/pydantic-2.9.2")
        assert name == "lusiscan/pydantic-2.9.2"

    def test_other_github_errors_propagate(self) -> None:
        repo = FakeRepository()
        repo.create_ref_raise = gt.GithubException(500, {"message": "boom"}, None)
        with pytest.raises(gt.GithubException):
            gt.create_branch(repo, "lusiscan/x")


# --- commit_changes (task 8.1) --------------------------------------------


class TestCommitChanges:
    def test_commits_all_files_in_a_single_commit(self) -> None:
        repo = FakeRepository()
        changes = {
            "pyproject.toml": 'deps = ["pydantic==2.9.2"]\n',
            "models.py": "from pydantic_settings import BaseSettings\n",
        }
        sha = gt.commit_changes(repo, "branch-x", changes, "upgrade pydantic")

        assert sha == "newcommitsha"
        # Exactly one commit was created (atomic multi-file commit).
        assert len(repo.created_commits) == 1
        # The tree carried a tree element per changed file.
        assert len(repo.created_trees) == 1
        assert len(repo.created_trees[0]["elements"]) == 2
        # The branch ref was moved to the new commit.
        assert repo.branch_ref.edited_to == "newcommitsha"

    def test_empty_changes_is_a_noop(self) -> None:
        repo = FakeRepository()
        sha = gt.commit_changes(repo, "branch-x", {}, "nothing")
        assert sha is None
        assert repo.created_commits == []
        assert repo.branch_ref.edited_to is None


# --- build_pr_content: confidence tiers (R5.1, R5.2) ----------------------


class TestBuildPrContent:
    def test_high_confidence_is_ready_to_approve(self) -> None:
        migration = {
            "package": "requests",
            "current": "2.31.0",
            "target": "2.32.3",
            "confidence": "high",
            "applied": [],
        }
        title, body = gt.build_pr_content(migration)
        assert "requests" in title and "2.32.3" in title
        assert "ready for a quick approval" in body
        assert "high" in body.lower()

    def test_low_confidence_is_guided_with_flags(self) -> None:
        # R5.2: guided PR lists version bump, applied auto-fixes, flagged changes.
        migration = {
            "package": "pydantic",
            "current": "1.10.13",
            "target": "2.9.2",
            "confidence": "low",
            "applied": ["pydantic.validator_rename"],
            "flagged": [
                {
                    "pattern_id": "pydantic.config_class_to_model_config",
                    "description": "Config class must become model_config",
                    "file": "models.py",
                }
            ],
        }
        title, body = gt.build_pr_content(migration)
        assert "guided" in body.lower()
        assert "1.10.13" in body and "2.9.2" in body
        assert "pydantic.validator_rename" in body
        assert "model_config" in body
        assert "models.py" in body

    def test_unknown_confidence_defaults_to_guided(self) -> None:
        # Conservative default: without a confidence, ask for judgment.
        title, body = gt.build_pr_content({"package": "x", "current": "1", "target": "2"})
        assert "guided" in body.lower()


# --- create_migration_pr: the executor boundary (R5.1, R5.2) --------------


class TestCreateMigrationPr:
    def _migration(self) -> dict:
        return {
            "package": "pydantic",
            "current": "1.10.13",
            "target": "2.9.2",
            "confidence": "low",
            "changes": {
                "pyproject.toml": 'deps = ["pydantic==2.9.2"]\n',
                "models.py": "field_validator\n",
            },
            "diff": "--- a/models.py\n+++ b/models.py\n",
            "applied": ["pydantic.validator_rename"],
            "flagged": [{"pattern_id": "x", "description": "d", "file": "models.py"}],
        }

    def test_happy_path_branch_commit_pr(self) -> None:
        repo = FakeRepository()
        client = FakeGithub(repo)
        result = gt.create_migration_pr("acme/demo", self._migration(), client=client)

        # Returned the executor result shape.
        assert result["branch"] == "lusiscan/pydantic-2.9.2"
        assert result["pr_number"] == 42
        assert result["pr_url"].endswith("/pull/42")
        assert result["diff"].startswith("--- a/models.py")
        assert set(result["changes"]) == {"pyproject.toml", "models.py"}

        # Branch was created off the base.
        assert repo.created_refs[0]["ref"] == "refs/heads/lusiscan/pydantic-2.9.2"
        # One commit carrying both changed files.
        assert len(repo.created_commits) == 1
        assert len(repo.created_trees[0]["elements"]) == 2
        # PR opened into main with a guided (low-confidence) body.
        assert repo.created_pulls[0]["base"] == "main"
        assert repo.created_pulls[0]["head"] == "lusiscan/pydantic-2.9.2"
        assert "guided" in repo.created_pulls[0]["body"].lower()

    def test_custom_branch_and_base_are_used(self) -> None:
        repo = FakeRepository()
        client = FakeGithub(repo)
        result = gt.create_migration_pr(
            "acme/demo",
            self._migration(),
            branch="feature/pyd",
            base_branch="develop",
            client=client,
        )
        assert result["branch"] == "feature/pyd"
        assert repo.created_refs[0]["ref"] == "refs/heads/feature/pyd"
        assert repo.created_pulls[0]["base"] == "develop"

    def test_missing_package_key_raises(self) -> None:
        client = FakeGithub(FakeRepository())
        with pytest.raises(KeyError):
            gt.create_migration_pr("acme/demo", {"target": "2.0.0"}, client=client)


# --- apply_decision: NEVER merge without explicit approval (R5.4, R5.5) ---


class TestApplyDecisionNeverMergesWithoutApproval:
    """The core safety guarantee: no merge without an explicit approval (R5.4)."""

    @pytest.mark.parametrize(
        "decision",
        [
            None,
            "review",
            "ignored",
            "REVIEW",
            "",
            "approve",  # not the exact recorded value
            "yes",
            {"decision": "review"},
            {"decision": "ignored"},
            {},  # no decision recorded at all
            {"approved": True},  # truthy but not an explicit "approved" decision
            123,
        ],
    )
    def test_merge_is_refused_for_any_non_approval(self, decision: object) -> None:
        repo = FakeRepository()
        client = FakeGithub(repo)
        result = gt.apply_decision("acme/demo", 42, decision, client=client)

        # The PR was never merged (R5.4).
        pull = repo.get_pull(42)
        assert pull.merged is False
        assert result["merged"] is False
        assert result["pr_state"] in (gt.PR_STATE_LEFT_OPEN, gt.PR_STATE_CLOSED)

    def test_review_leaves_pr_open(self) -> None:
        # R5.5: a "review" decision leaves the PR open (no merge, no close).
        repo = FakeRepository()
        client = FakeGithub(repo)
        result = gt.apply_decision("acme/demo", 42, "review", client=client)

        pull = repo.get_pull(42)
        assert pull.merged is False
        assert pull.state == "open"
        assert result["pr_state"] == gt.PR_STATE_LEFT_OPEN
        assert result["merged"] is False

    def test_no_recorded_decision_leaves_pr_open(self) -> None:
        # No decision yet (None): nothing merged, PR left open (R5.4/R5.5).
        repo = FakeRepository()
        client = FakeGithub(repo)
        result = gt.apply_decision("acme/demo", 42, None, client=client)
        assert result["pr_state"] == gt.PR_STATE_LEFT_OPEN
        assert result["merged"] is False

    def test_ignored_closes_pr_without_merging(self) -> None:
        # R5.5: an "ignored" decision closes the PR (skip) — but never merges.
        repo = FakeRepository()
        client = FakeGithub(repo)
        result = gt.apply_decision("acme/demo", 42, "ignored", client=client)

        pull = repo.get_pull(42)
        assert pull.merged is False
        assert pull.state == "closed"
        assert result["pr_state"] == gt.PR_STATE_CLOSED
        assert result["merged"] is False

    def test_ignored_via_decision_dict_closes_pr(self) -> None:
        repo = FakeRepository()
        client = FakeGithub(repo)
        result = gt.apply_decision(
            "acme/demo", 42, {"decision": "ignored"}, client=client
        )
        assert repo.get_pull(42).state == "closed"
        assert result["pr_state"] == gt.PR_STATE_CLOSED


class TestApplyDecisionMergesOnApproval:
    """The permitted path: an explicit recorded approval merges (R5.4, R5.5)."""

    def test_explicit_approval_string_merges(self) -> None:
        repo = FakeRepository()
        client = FakeGithub(repo)
        result = gt.apply_decision("acme/demo", 42, "approved", client=client)

        pull = repo.get_pull(42)
        assert pull.merged is True
        assert result["merged"] is True
        assert result["pr_state"] == gt.PR_STATE_MERGED
        assert result["decision"] == gt.DECISION_APPROVED

    def test_explicit_approval_decision_dict_merges(self) -> None:
        # R5.5: the decision as stored (a dict) is honored and merges.
        repo = FakeRepository()
        client = FakeGithub(repo)
        result = gt.apply_decision(
            "acme/demo",
            42,
            {"decision": "approved", "timestamp": "2024-01-01T00:00:00Z"},
            client=client,
        )
        assert repo.get_pull(42).merged is True
        assert result["merged"] is True

    def test_approval_is_case_insensitive(self) -> None:
        repo = FakeRepository()
        client = FakeGithub(repo)
        result = gt.apply_decision("acme/demo", 42, "Approved", client=client)
        assert result["merged"] is True

    def test_custom_merge_message_is_forwarded(self) -> None:
        repo = FakeRepository()
        client = FakeGithub(repo)
        gt.apply_decision(
            "acme/demo", 42, "approved", merge_message="ship it", client=client
        )
        assert repo.get_pull(42).merge_message == "ship it"
