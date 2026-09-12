"""Human notifications: a plain Slack webhook or a PR comment (task 13.3).

This module is the **NotifierAgent** boundary. It sends a single, plain,
human-facing alert when LusiScan needs a person to look at a migration — either
to approve a ready PR (R5.1), to make a judgement call on a guided PR (R5.2), or
to handle an architectural change LusiScan deliberately did **not** touch (R5.3,
``human_required``).

Scope (design.md → Tools layer → ``notify_tools.py``; requirements.md → scope
boundaries): a **plain** Slack webhook message or a PR comment is sufficient —
Slack interactive buttons / webhooks-in are explicitly out of scope. The actual
decision (approve / review / ignore) is recorded in the Streamlit control panel,
not here.

Delivery order (most direct first):

1. **Slack webhook** — when a webhook URL is available, ``POST`` the message
   text as ``{"text": ...}`` (the classic incoming-webhook shape).
2. **PR comment** — otherwise, if the migration has an open PR, drop the message
   as a PR comment via PyGithub (``PullRequest.create_issue_comment``).
3. **No-op** — if neither channel is available, return a ``skipped`` result
   rather than raising, so the loop never crashes just because notification was
   not configured.

Secrets handling (design.md → Security; R6.5): the Slack webhook URL is a secret
resolved at runtime from Secrets Manager and **injected** into
:func:`notify` / :class:`Notifier`. This module never hardcodes it, and never
logs the URL. GitHub auth is delegated to :func:`github_tools.get_client`.

Design references:
- design.md → Tools layer: "``notify_tools.py`` | Send human alert | Plain Slack
  webhook or PR comment".
- design.md → Data flow step 6: "Orchestrator writes a ``pending_review``
  migration ... and notifies."
- requirements.md → R5.1, R5.2, R5.3 (surface a decision), scope boundary
  ("a plain PR comment or Slack message is sufficient").
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from src.tools import github_tools


# --- Notification channels + result shape ---------------------------------

CHANNEL_SLACK = "slack"
CHANNEL_PR_COMMENT = "pr_comment"
CHANNEL_NONE = "none"

# The notice "tier" — what kind of human attention a migration needs. Mirrors
# the R5.1/5.2/5.3 lanes so the message text (and, later, any routing) can vary.
TIER_READY = "ready_to_approve"      # R5.1: high confidence + tests pass
TIER_GUIDED = "guided_review"        # R5.2: low confidence, guided PR
TIER_HUMAN_REQUIRED = "human_required"  # R5.3: architectural, code untouched

# The notifier result (never raises): which channel delivered (or ``none``),
# whether it was sent, and a short human-readable message.
NotifyResult = dict[str, Any]


def build_notice(migration: dict) -> str:
    """Compose the plain human-facing notification text for a migration.

    The message states what LusiScan did and what it needs from the human,
    tailored to the migration's tier (R5.1/5.2/5.3). It is intentionally plain
    text (with light Markdown) so it renders fine in both Slack and a PR comment.

    Args:
        migration: A migration view/record carrying at least ``package``,
            ``current``/``from``, ``target``/``to``; optionally ``tier``,
            ``confidence``, ``strategy``, ``pr_url``, ``test_summary``,
            ``flagged``.

    Returns:
        The notification message text.
    """
    package = migration.get("package", "a dependency")
    current = migration.get("current") or migration.get("from") or "?"
    target = migration.get("target") or migration.get("to") or "?"
    tier = migration.get("tier") or _infer_tier(migration)
    pr_url = migration.get("pr_url")

    header = f"LusiScan: {package} {current} → {target}"

    if tier == TIER_HUMAN_REQUIRED:
        body = (
            "This upgrade needs an architectural decision, so LusiScan left the "
            "code untouched. Please review the breaking changes and decide how "
            "to proceed."
        )
    elif tier == TIER_READY:
        body = (
            "High confidence and tests pass — a pull request is ready for a "
            "quick approval."
        )
    else:  # TIER_GUIDED
        body = (
            "Low confidence — LusiScan opened a guided pull request with the "
            "version bump, applied auto-fixes, and flagged breaking changes for "
            "your judgement."
        )

    lines = [f"*{header}*", body]

    flagged = migration.get("flagged") or []
    if flagged:
        lines.append(f"Flagged for review: {len(flagged)} change(s).")

    summary = migration.get("test_summary") or {}
    if isinstance(summary, dict) and summary.get("status"):
        lines.append(f"Tests: {summary.get('status')}.")

    if pr_url:
        lines.append(f"PR: {pr_url}")

    return "\n".join(lines)


def _infer_tier(migration: dict) -> str:
    """Infer the notice tier from a migration's strategy / confidence / tests."""
    strategy = str(migration.get("strategy", "")).lower()
    if strategy == "human_required":
        return TIER_HUMAN_REQUIRED

    confidence = str(migration.get("confidence", "")).lower()
    summary = migration.get("test_summary") or {}
    tests_passed = isinstance(summary, dict) and summary.get("status") == "passed"
    if confidence == "high" and tests_passed:
        return TIER_READY
    return TIER_GUIDED


def notify(
    migration: dict,
    *,
    repo_name: Optional[str] = None,
    slack_webhook: Optional[str] = None,
    message: Optional[str] = None,
    github_client: Any | None = None,
    github_token: Optional[str] = None,
    session: Any | None = None,
) -> NotifyResult:
    """Send one human notification for ``migration`` (R5.1/5.2/5.3, task 13.3).

    Chooses the channel by availability: Slack webhook first, then a PR comment,
    then a no-op. Never raises — any delivery failure is captured in the returned
    result so the agent loop is never brought down by a notification problem.

    Args:
        migration: The migration view/record (see :func:`build_notice`). Used to
            build the message and to find the PR (``pr_number``/``pr_url``).
        repo_name: ``owner/repo`` slug, required only for the PR-comment channel.
        slack_webhook: The Slack incoming-webhook URL (a secret injected at
            runtime; never hardcoded/logged). When falsy, Slack is skipped.
        message: Optional pre-built message; when omitted, :func:`build_notice`
            composes it from ``migration``.
        github_client: Optional PyGithub client for the PR-comment fallback.
        github_token: Optional token for the PR-comment fallback (falls back to
            ``GITHUB_TOKEN``).
        session: Optional ``requests``-like session for the Slack ``POST``
            (injected in tests); defaults to the ``requests`` module.

    Returns:
        A :data:`NotifyResult` ``{"channel", "sent", "message"|"detail"}``.
    """
    text = message if message is not None else build_notice(migration)

    # 1) Slack webhook (most direct).
    if slack_webhook:
        result = _send_slack(slack_webhook, text, session=session)
        if result["sent"]:
            return result
        # Slack failed: fall through to try a PR comment.

    # 2) PR comment fallback.
    pr_number = migration.get("pr_number")
    if repo_name and pr_number is not None:
        result = _comment_on_pr(
            repo_name,
            int(pr_number),
            text,
            client=github_client,
            token=github_token,
        )
        if result["sent"]:
            return result

    # 3) Nothing configured / everything failed: a benign no-op.
    return {
        "channel": CHANNEL_NONE,
        "sent": False,
        "detail": "no notification channel available (no Slack webhook, no PR)",
        "message": text,
    }


def _send_slack(webhook: str, text: str, *, session: Any | None = None) -> NotifyResult:
    """POST ``text`` to a Slack incoming webhook; never raise, never log the URL."""
    try:
        http = session
        if http is None:
            import requests  # noqa: PLC0415 - lazy import; keeps module import light

            http = requests
        response = http.post(webhook, json={"text": text}, timeout=10)
        status_code = getattr(response, "status_code", None)
        ok = status_code is not None and 200 <= int(status_code) < 300
        return {
            "channel": CHANNEL_SLACK,
            "sent": bool(ok),
            "detail": f"slack webhook responded {status_code}",
            "message": text,
        }
    except Exception as exc:  # noqa: BLE001 - delivery failure is data, not fatal
        return {
            "channel": CHANNEL_SLACK,
            "sent": False,
            "detail": f"slack webhook post failed: {exc}",
            "message": text,
        }


def _comment_on_pr(
    repo_name: str,
    pr_number: int,
    text: str,
    *,
    client: Any | None = None,
    token: Optional[str] = None,
) -> NotifyResult:
    """Post ``text`` as a comment on the PR; never raise."""
    try:
        gh = github_tools.get_client(client=client, token=token)
        repo = gh.get_repo(repo_name)
        pull = repo.get_pull(pr_number)
        pull.create_issue_comment(text)
        return {
            "channel": CHANNEL_PR_COMMENT,
            "sent": True,
            "detail": f"commented on {repo_name}#{pr_number}",
            "message": text,
        }
    except Exception as exc:  # noqa: BLE001 - delivery failure is data, not fatal
        return {
            "channel": CHANNEL_PR_COMMENT,
            "sent": False,
            "detail": f"pr comment failed on {repo_name}#{pr_number}: {exc}",
            "message": text,
        }


class Notifier:
    """A pre-configured, callable notifier the orchestrator can inject.

    Binds the repo, Slack webhook, and GitHub credentials once so the
    orchestrator can call ``notifier(migration)`` per migration without threading
    config through every call. Delegates to :func:`notify`.
    """

    def __init__(
        self,
        repo_name: str,
        *,
        slack_webhook: Optional[str] = None,
        github_client: Any | None = None,
        github_token: Optional[str] = None,
        session: Any | None = None,
    ) -> None:
        self.repo_name = repo_name
        self.slack_webhook = slack_webhook
        self.github_client = github_client
        self.github_token = github_token
        self.session = session

    def __call__(self, migration: dict, *, message: Optional[str] = None) -> NotifyResult:
        return notify(
            migration,
            repo_name=self.repo_name,
            slack_webhook=self.slack_webhook,
            message=message,
            github_client=self.github_client,
            github_token=self.github_token,
            session=self.session,
        )
