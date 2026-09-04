"""GitHub tools: create branches, commit changes, open PRs, merge on approval.

This module implements the GitHub integration used by the ExecutorAgent. It
turns a completed migration (transformed files + diff + flagged changes, from
``refactor_tools.apply_migration``) into a real pull request, and — critically —
it never merges that pull request without an **explicit, recorded human
approval** (R5.4). The Streamlit control panel records the human's decision to
DynamoDB (task 9); on its next cycle the orchestrator hands that decision here
and this module acts on it: merge on ``approved``, leave open on ``review``,
close on ``ignored`` (R5.5).

Scope (per design.md → Tools layer → ``github_tools.py``): branch, commit, PR,
and the merge-on-approval path. Reading CI / GitHub Actions status is a separate
concern handled in task 12, so it is intentionally **not** implemented here.

Public entrypoints:

- :func:`create_branch` creates a new branch off a base branch's current head.
- :func:`commit_changes` commits a mapping of ``path -> content`` onto a branch
  as a single commit (via the Git Data API so multiple files land atomically).
- :func:`open_pull_request` opens a PR from the head branch into the base branch.
- :func:`create_migration_pr` composes the three above into the executor-facing
  ``{branch, pr_url, pr_number, diff, changes}`` result shape (R5.1, R5.2).
- :func:`apply_decision` is the merge-on-approval boundary: it merges **only**
  when handed an explicit ``approved`` decision (R5.4), leaves the PR open on
  ``review``, and closes it on ``ignored`` (R5.5).

Secrets handling (design.md → Security; R6.5): this module never hardcodes a
token. A caller passes an already-authenticated PyGithub ``Github`` client, or a
token string, or lets :func:`get_client` read ``GITHUB_TOKEN`` from the
environment. At runtime on AgentCore the token comes from Secrets Manager and is
injected here — no secret is ever committed.

Design references:
- design.md → Agent layer → ``ExecutorAgent``: plan → ``{branch, pr_url, diff,
  changes}``.
- design.md → Data flow: "Executor branches, applies scoped fix ... opens PR";
  "Next agent cycle reads the decision, merges the PR, transitions status."
- design.md → Security: "LusiScan never merges without explicit human approval
  (R5.4)"; "Secrets from Secrets Manager at runtime only; never committed."
- requirements.md → R5.1, R5.2, R5.4, R5.5.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from github import Github, GithubException
from github.Auth import Token
from github.InputGitTreeElement import InputGitTreeElement

if TYPE_CHECKING:  # pragma: no cover - typing only
    from github.PullRequest import PullRequest
    from github.Repository import Repository


# --- Decision constants (the human-in-the-loop contract, R5.4 / R5.5) -----

# The three decisions a human can record in the Streamlit control panel. These
# are the only values that drive an action in :func:`apply_decision`. Anything
# else is treated as "no explicit approval" and never merges (R5.4).
DECISION_APPROVED = "approved"
DECISION_REVIEW = "review"
DECISION_IGNORED = "ignored"

# The resulting PR states after a decision is applied (R5.5). ``left_open`` is
# the default no-op outcome so that, absent an approval, nothing is merged.
PR_STATE_MERGED = "merged"
PR_STATE_LEFT_OPEN = "left_open"
PR_STATE_CLOSED = "closed"

# Default base branch when a repository's default is not otherwise specified.
_DEFAULT_BASE_BRANCH = "main"

# Git reference prefix for branch heads (used with the Git refs API).
_HEADS_REF_PREFIX = "refs/heads/"


# --- Result shapes --------------------------------------------------------

# The executor-facing PR result (design.md → ExecutorAgent): the branch that was
# created, the opened PR's URL and number, the combined unified diff, and the
# mapping of changed files that were committed.
MigrationPRResult = dict[str, object]

# The outcome of applying a recorded human decision to a PR (R5.5): the decision
# that was acted on, the resulting PR state, whether a merge actually happened,
# and a human-readable message.
DecisionResult = dict[str, object]


# --- Client / auth (no committed secrets, R6.5) ---------------------------


def get_client(
    *, client: Github | None = None, token: str | None = None
) -> Github:
    """Return an authenticated PyGithub client without hardcoding any secret.

    Resolution order, most-explicit first (R6.5):

    1. ``client`` — an already-authenticated ``Github`` instance (what the
       orchestrator injects at runtime after reading the token from Secrets
       Manager; also what tests inject as a fake).
    2. ``token`` — a raw token string passed by the caller.
    3. ``GITHUB_TOKEN`` — read from the environment as a last resort for local
       runs.

    No token value is ever written to source or logged here.

    Args:
        client: An optional pre-built PyGithub client to use as-is.
        token: An optional token string to authenticate a new client.

    Returns:
        A ready-to-use ``Github`` client.

    Raises:
        ValueError: If no client, token, or ``GITHUB_TOKEN`` is available.
    """
    if client is not None:
        return client
    resolved = token or os.environ.get("GITHUB_TOKEN")
    if not resolved:
        raise ValueError(
            "no GitHub credentials: pass a client, a token, or set GITHUB_TOKEN"
        )
    return Github(auth=Token(resolved))


def _get_repo(client: Github, repo_name: str) -> "Repository":
    """Resolve a ``owner/repo`` slug to a PyGithub ``Repository``."""
    return client.get_repo(repo_name)


# --- Branch creation (R5.1 / R5.2 groundwork) -----------------------------


def create_branch(
    repo: "Repository", branch: str, *, base_branch: str = _DEFAULT_BASE_BRANCH
) -> str:
    """Create ``branch`` off the current head of ``base_branch``.

    The new branch points at whatever commit ``base_branch`` currently
    references, so subsequent commits build on top of the base's latest state.
    If the branch already exists it is reused (idempotent), which keeps a
    re-run of the same migration cycle from crashing.

    Args:
        repo: The target PyGithub repository.
        branch: The new branch name (e.g. ``lusiscan/pydantic-2.9.2``).
        base_branch: The branch to fork from (defaults to ``main``).

    Returns:
        The created (or existing) branch name.

    Raises:
        GithubException: On GitHub API errors other than an
            already-exists (422) conflict, which is handled as reuse.
    """
    base_ref = repo.get_branch(base_branch)
    base_sha = base_ref.commit.sha
    try:
        repo.create_git_ref(ref=f"{_HEADS_REF_PREFIX}{branch}", sha=base_sha)
    except GithubException as exc:
        # 422 = reference already exists; reuse it rather than failing the run.
        if exc.status == 422:
            return branch
        raise
    return branch


# --- Committing changes (Git Data API for an atomic multi-file commit) ----


def commit_changes(
    repo: "Repository",
    branch: str,
    changes: dict[str, str],
    message: str,
) -> str | None:
    """Commit ``changes`` (``path -> content``) onto ``branch`` as one commit.

    Uses the Git Data API (build a tree, then a commit, then move the branch ref)
    so that every changed file lands in a **single** atomic commit rather than
    one commit per file. The tree is based on the branch's current commit, so
    unchanged files are preserved.

    Args:
        repo: The target repository.
        branch: The branch to commit onto (must already exist).
        changes: Mapping of file path to its full new text content.
        message: The commit message.

    Returns:
        The new commit SHA, or ``None`` when ``changes`` is empty (nothing to
        commit — a no-op rather than an empty commit).

    Raises:
        GithubException: On GitHub API errors building the tree/commit or
            updating the ref.
    """
    if not changes:
        return None

    branch_ref = repo.get_git_ref(f"heads/{branch}")
    base_commit = repo.get_git_commit(branch_ref.object.sha)
    base_tree = base_commit.tree

    elements = [
        InputGitTreeElement(
            path=path, mode="100644", type="blob", content=content
        )
        for path, content in changes.items()
    ]
    new_tree = repo.create_git_tree(elements, base_tree)
    new_commit = repo.create_git_commit(message, new_tree, [base_commit])
    branch_ref.edit(sha=new_commit.sha)
    return new_commit.sha


# --- Opening a pull request (R5.1, R5.2) ----------------------------------


def open_pull_request(
    repo: "Repository",
    *,
    title: str,
    body: str,
    head: str,
    base: str = _DEFAULT_BASE_BRANCH,
) -> "PullRequest":
    """Open a pull request from ``head`` into ``base``.

    Args:
        repo: The target repository.
        title: The PR title.
        body: The PR description (Markdown).
        head: The source branch containing the migration commit.
        base: The branch to merge into (defaults to ``main``).

    Returns:
        The created ``PullRequest``.

    Raises:
        GithubException: On GitHub API errors opening the PR.
    """
    return repo.create_pull(title=title, body=body, head=head, base=base)


# --- PR body composition for the two confidence tiers (R5.1, R5.2) --------


def _render_flagged(flagged: list[dict] | None) -> str:
    """Render flagged breaking changes as a Markdown list (or a placeholder)."""
    if not flagged:
        return "_None._"
    lines: list[str] = []
    for flag in flagged:
        description = str(flag.get("description", flag))
        file = str(flag.get("file", "")) if isinstance(flag, dict) else ""
        suffix = f" (`{file}`)" if file else ""
        lines.append(f"- {description}{suffix}")
    return "\n".join(lines)


def _render_applied(applied: list[str] | None) -> str:
    """Render applied auto-fix pattern ids as a Markdown list (or placeholder)."""
    if not applied:
        return "_None (version bump only)._"
    return "\n".join(f"- `{pattern_id}`" for pattern_id in applied)


def build_pr_content(migration: dict) -> tuple[str, str]:
    """Build a ``(title, body)`` PR pair from a migration result.

    High-confidence migrations get a concise "ready to approve" PR (R5.1);
    low-confidence / guided migrations get a **guided** PR that spells out the
    version bump, the auto-fixes that were applied, and the flagged breaking
    changes still needing human judgment (R5.2). The confidence tier is taken
    from ``migration["confidence"]`` (``"high"``/``"low"``), defaulting to a
    guided body when confidence is unknown (conservative — asks for judgment).

    Args:
        migration: A dict carrying at least ``package``, ``current``, ``target``;
            optionally ``confidence``, ``applied``, ``flagged``, ``strategy``.

    Returns:
        A ``(title, body)`` tuple ready for :func:`open_pull_request`.
    """
    package = migration.get("package", "dependency")
    current = migration.get("current", "?")
    target = migration.get("target", "?")
    confidence = str(migration.get("confidence", "low")).lower()

    title = f"Upgrade {package} {current} → {target}"

    if confidence == "high":
        body = (
            f"### LusiScan migration: `{package}` {current} → {target}\n\n"
            f"**Confidence:** high — tests pass and the changes are mechanical.\n\n"
            f"**Applied auto-fixes:**\n{_render_applied(migration.get('applied'))}\n\n"
            "This migration is ready for a quick approval. Approve in the "
            "LusiScan control panel to merge."
        )
        return title, body

    # Low-confidence / guided PR (R5.2): version bump + auto-fixes + flags.
    body = (
        f"### LusiScan guided migration: `{package}` {current} → {target}\n\n"
        f"**Confidence:** low — this upgrade needs human judgment.\n\n"
        f"**Version bump:** `{package}` {current} → {target}\n\n"
        f"**Applied auto-fixes:**\n{_render_applied(migration.get('applied'))}\n\n"
        f"**Flagged breaking changes (need your judgment):**\n"
        f"{_render_flagged(migration.get('flagged'))}\n\n"
        "Review the flagged changes, then approve, review, or ignore this "
        "migration in the LusiScan control panel."
    )
    return title, body


# --- Executor-facing PR creation (R5.1, R5.2) -----------------------------


def create_migration_pr(
    repo_name: str,
    migration: dict,
    *,
    branch: str | None = None,
    base_branch: str = _DEFAULT_BASE_BRANCH,
    commit_message: str | None = None,
    client: Github | None = None,
    token: str | None = None,
) -> MigrationPRResult:
    """Branch, commit the migration's changes, and open a PR (R5.1, R5.2).

    This is the ExecutorAgent boundary. Given a migration result (as produced by
    ``refactor_tools.apply_migration`` — ``package``, ``target``, ``changes``,
    ``diff``, ``applied``, ``flagged``, plus a ``confidence`` set by the
    planner), it creates a dedicated branch, commits every changed file in a
    single commit, and opens a pull request whose body matches the confidence
    tier (concise "ready to approve" for high confidence, guided for low).

    It does **not** merge anything — merging happens only later, and only on an
    explicit recorded approval (see :func:`apply_decision`, R5.4).

    Args:
        repo_name: The ``owner/repo`` slug of the target repository.
        migration: The migration result to turn into a PR. Recognized keys:
            ``package`` (required), ``current``, ``target``, ``changes``
            (``path -> content``), ``diff``, ``applied``, ``flagged``,
            ``confidence``, ``strategy``.
        branch: Optional branch name; a stable default is derived from the
            package and target version when omitted.
        base_branch: The branch to fork from and target (defaults to ``main``).
        commit_message: Optional commit message; a sensible default is derived.
        client: Optional pre-authenticated PyGithub client (injected at runtime
            / in tests).
        token: Optional token string (falls back to ``GITHUB_TOKEN``).

    Returns:
        A :data:`MigrationPRResult` ``{branch, pr_url, pr_number, diff,
        changes}``.

    Raises:
        KeyError: If ``migration`` has no ``package`` key.
        ValueError: If no GitHub credentials are available.
        GithubException: On GitHub API failures.
    """
    package = migration["package"]
    target = migration.get("target", "latest")
    changes = migration.get("changes", {}) or {}
    diff = migration.get("diff", "")

    gh = get_client(client=client, token=token)
    repo = _get_repo(gh, repo_name)

    branch_name = branch or f"lusiscan/{package}-{target}"
    message = commit_message or f"LusiScan: upgrade {package} to {target}"

    create_branch(repo, branch_name, base_branch=base_branch)
    commit_changes(repo, branch_name, changes, message)

    title, body = build_pr_content(migration)
    pull = open_pull_request(
        repo, title=title, body=body, head=branch_name, base=base_branch
    )

    return {
        "branch": branch_name,
        "pr_url": pull.html_url,
        "pr_number": pull.number,
        "diff": diff,
        "changes": changes,
    }


# --- Merge-on-approval boundary (R5.4, R5.5) ------------------------------


def _is_explicit_approval(decision: object) -> bool:
    """Return ``True`` only for an explicit, recorded ``approved`` decision.

    This is the single gate protecting R5.4: LusiScan merges nothing without an
    explicit human approval. A decision qualifies only when it clearly carries
    the ``approved`` value — either the bare string ``"approved"`` or a decision
    dict ``{"decision": "approved", ...}``. Everything else — ``None``, an empty
    dict, ``review``, ``ignored``, an unknown value, or a truthy-but-unrelated
    object — is treated as *not approved* and must never merge.
    """
    if isinstance(decision, str):
        return decision.strip().lower() == DECISION_APPROVED
    if isinstance(decision, dict):
        value = decision.get("decision")
        return isinstance(value, str) and value.strip().lower() == DECISION_APPROVED
    return False


def _decision_value(decision: object) -> str | None:
    """Extract the normalized decision string from a decision object."""
    if isinstance(decision, str):
        return decision.strip().lower()
    if isinstance(decision, dict):
        value = decision.get("decision")
        if isinstance(value, str):
            return value.strip().lower()
    return None


def apply_decision(
    repo_name: str,
    pr_number: int,
    decision: object,
    *,
    merge_message: str | None = None,
    client: Github | None = None,
    token: str | None = None,
) -> DecisionResult:
    """Act on a recorded human decision for a PR — never merging without one.

    This is the merge-on-approval boundary read on the orchestrator's next cycle
    (design.md → Data flow step 8). It enforces the core human-in-the-loop
    guarantee:

    - **Merge only on explicit approval (R5.4).** The PR is merged **iff**
      ``decision`` is an explicit recorded ``approved`` (bare ``"approved"`` or
      ``{"decision": "approved"}``). For every other input — ``None``, missing,
      ``review``, ``ignored``, or anything unrecognized — the PR is **not**
      merged. There is no path that merges without an approval.
    - **Honor the recorded decision (R5.5).** ``approved`` → merge; ``ignored``
      → close the PR (skip the migration); ``review`` (or any non-approval) →
      leave the PR open for continued human review.

    Args:
        repo_name: The ``owner/repo`` slug of the target repository.
        pr_number: The pull request number returned by :func:`create_migration_pr`.
        decision: The recorded human decision. Accepts a bare string
            (``"approved"``/``"review"``/``"ignored"``) or a decision dict with
            a ``"decision"`` key (as stored in DynamoDB, task 9).
        merge_message: Optional merge commit message used when merging.
        client: Optional pre-authenticated PyGithub client.
        token: Optional token string (falls back to ``GITHUB_TOKEN``).

    Returns:
        A :data:`DecisionResult` ``{decision, pr_state, merged, message}``.

    Raises:
        ValueError: If no GitHub credentials are available.
        GithubException: On GitHub API failures interacting with the PR.
    """
    value = _decision_value(decision)

    # R5.4 gate: without an explicit recorded approval, we never merge. We do
    # not even need to touch GitHub for the pure no-op / close paths beyond what
    # the decision demands.
    if not _is_explicit_approval(decision):
        gh = get_client(client=client, token=token)
        repo = _get_repo(gh, repo_name)
        pull = repo.get_pull(pr_number)

        if value == DECISION_IGNORED:
            # R5.5: an ignored migration closes the PR (skip) without merging.
            pull.edit(state="closed")
            return {
                "decision": DECISION_IGNORED,
                "pr_state": PR_STATE_CLOSED,
                "merged": False,
                "message": f"PR #{pr_number} closed (migration ignored).",
            }

        # ``review`` or anything unrecognized: leave the PR open, merge nothing.
        return {
            "decision": value if value is not None else "none",
            "pr_state": PR_STATE_LEFT_OPEN,
            "merged": False,
            "message": (
                f"PR #{pr_number} left open; no explicit approval recorded, "
                "so nothing was merged."
            ),
        }

    # Explicit approval: merge the PR (R5.4 satisfied, R5.5 acted on).
    gh = get_client(client=client, token=token)
    repo = _get_repo(gh, repo_name)
    pull = repo.get_pull(pr_number)
    message = merge_message or f"LusiScan: merge approved migration (PR #{pr_number})"
    pull.merge(commit_message=message)

    return {
        "decision": DECISION_APPROVED,
        "pr_state": PR_STATE_MERGED,
        "merged": True,
        "message": f"PR #{pr_number} merged after explicit human approval.",
    }
