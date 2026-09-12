"""Unit tests for the notifier (task 13.3): Slack webhook or PR comment.

Exercises the ``notify_tools`` module in isolation — no network, no real Slack,
no real GitHub — by injecting a fake ``requests``-like session and a fake
PyGithub client. Covers:

- the message tiers (:func:`build_notice`) for R5.1/5.2/5.3,
- Slack delivery success / failure,
- the PR-comment fallback when no webhook is configured,
- the no-op result when neither channel is available (never raises),
- the webhook URL is never surfaced in the result,
- the :class:`Notifier` callable seam the orchestrator injects.

_Requirements: 5.1, 5.2, 5.3 (surface the decision via a plain Slack message or
PR comment)._
"""

from __future__ import annotations

from src.tools import notify_tools as n


# --- Fakes ----------------------------------------------------------------


class _FakeResponse:
    def __init__(self, status_code):
        self.status_code = status_code


class _FakeSession:
    """A ``requests``-like session recording the last POST."""

    def __init__(self, status_code=200, raises=None):
        self._status_code = status_code
        self._raises = raises
        self.calls = []

    def post(self, url, *, json=None, timeout=None):
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        if self._raises is not None:
            raise self._raises
        return _FakeResponse(self._status_code)


class _FakePull:
    def __init__(self):
        self.comments = []

    def create_issue_comment(self, body):
        self.comments.append(body)
        return {"id": 1}


class _FakeRepo:
    def __init__(self, pull=None):
        self._pull = pull or _FakePull()
        self.requested_pull = None

    def get_pull(self, number):
        self.requested_pull = number
        return self._pull


class _FakeClient:
    def __init__(self, repo):
        self._repo = repo

    def get_repo(self, name):
        return self._repo


# --- build_notice tiers (R5.1/5.2/5.3) ------------------------------------


def test_build_notice_human_required_tier():
    text = n.build_notice(
        {
            "package": "pydantic",
            "from": "1.10.13",
            "to": "2.13.5",
            "strategy": "human_required",
            "flagged": [1, 2, 3, 4],
        }
    )
    assert "pydantic 1.10.13 → 2.13.5" in text
    assert "left the code untouched" in text
    assert "4 change(s)" in text


def test_build_notice_ready_tier_high_confidence_tests_passed():
    text = n.build_notice(
        {
            "package": "requests",
            "current": "2.31.0",
            "target": "2.34.2",
            "confidence": "high",
            "test_summary": {"status": "passed"},
            "pr_url": "https://github.com/acme/demo/pull/1",
        }
    )
    assert "ready for a quick approval" in text
    assert "Tests: passed" in text
    assert "https://github.com/acme/demo/pull/1" in text


def test_build_notice_guided_tier_low_confidence():
    text = n.build_notice(
        {
            "package": "foo",
            "current": "1",
            "target": "2",
            "confidence": "low",
            "test_summary": {"status": "failed"},
        }
    )
    assert "guided pull request" in text


# --- Slack delivery -------------------------------------------------------


def test_notify_sends_to_slack_webhook_on_success():
    session = _FakeSession(status_code=200)
    result = n.notify(
        {"package": "requests", "current": "2.31.0", "target": "2.34.2",
         "confidence": "high", "test_summary": {"status": "passed"}},
        slack_webhook="https://hooks.slack.test/abc",
        session=session,
    )
    assert result["channel"] == n.CHANNEL_SLACK
    assert result["sent"] is True
    # The message was posted as {"text": ...} to the webhook.
    assert session.calls[0]["url"] == "https://hooks.slack.test/abc"
    assert "text" in session.calls[0]["json"]


def test_notify_slack_non_2xx_is_not_sent_and_falls_back_to_pr_comment():
    session = _FakeSession(status_code=500)
    pull = _FakePull()
    result = n.notify(
        {"package": "requests", "current": "2.31.0", "target": "2.34.2",
         "pr_number": 7},
        repo_name="acme/demo",
        slack_webhook="https://hooks.slack.test/abc",
        github_client=_FakeClient(_FakeRepo(pull)),
        session=session,
    )
    # Slack returned 500 → not sent → PR comment fallback delivered it.
    assert result["channel"] == n.CHANNEL_PR_COMMENT
    assert result["sent"] is True
    assert len(pull.comments) == 1


def test_notify_slack_exception_does_not_raise():
    session = _FakeSession(raises=RuntimeError("network down"))
    result = n.notify(
        {"package": "requests", "current": "2.31.0", "target": "2.34.2"},
        slack_webhook="https://hooks.slack.test/abc",
        session=session,
    )
    # No PR to fall back to → benign no-op, never raises.
    assert result["channel"] == n.CHANNEL_NONE
    assert result["sent"] is False


# --- PR-comment fallback (no webhook) -------------------------------------


def test_notify_uses_pr_comment_when_no_webhook():
    pull = _FakePull()
    repo = _FakeRepo(pull)
    result = n.notify(
        {"package": "pydantic", "from": "1.10.13", "to": "2.13.5",
         "strategy": "human_required", "pr_number": 2},
        repo_name="acme/demo",
        github_client=_FakeClient(repo),
    )
    assert result["channel"] == n.CHANNEL_PR_COMMENT
    assert result["sent"] is True
    assert repo.requested_pull == 2
    assert "left the code untouched" in pull.comments[0]


def test_notify_pr_comment_failure_is_captured_not_raised():
    class BoomClient:
        def get_repo(self, name):
            raise RuntimeError("no access")

    result = n.notify(
        {"package": "x", "current": "1", "target": "2", "pr_number": 3},
        repo_name="acme/demo",
        github_client=BoomClient(),
    )
    assert result["channel"] == n.CHANNEL_NONE
    assert result["sent"] is False


# --- No channel available -------------------------------------------------


def test_notify_no_channel_is_benign_noop():
    result = n.notify(
        {"package": "x", "current": "1", "target": "2"},  # no webhook, no PR
    )
    assert result["channel"] == n.CHANNEL_NONE
    assert result["sent"] is False
    # The composed message is still returned for observability.
    assert "x 1 → 2" in result["message"]


def test_notify_never_leaks_webhook_url_in_result():
    session = _FakeSession(status_code=200)
    webhook = "https://hooks.slack.test/SECRET-TOKEN"
    result = n.notify(
        {"package": "x", "current": "1", "target": "2"},
        slack_webhook=webhook,
        session=session,
    )
    # The secret webhook URL must not appear anywhere in the returned result.
    assert "SECRET-TOKEN" not in str(result)


# --- Notifier callable seam -----------------------------------------------


def test_notifier_is_callable_and_delegates():
    session = _FakeSession(status_code=200)
    notifier = n.Notifier(
        "acme/demo", slack_webhook="https://hooks.slack.test/abc", session=session
    )
    result = notifier(
        {"package": "requests", "current": "2.31.0", "target": "2.34.2"}
    )
    assert result["channel"] == n.CHANNEL_SLACK
    assert result["sent"] is True


def test_notifier_accepts_prebuilt_message():
    session = _FakeSession(status_code=200)
    notifier = n.Notifier(
        "acme/demo", slack_webhook="https://hooks.slack.test/abc", session=session
    )
    notifier({"package": "x", "current": "1", "target": "2"}, message="custom text")
    assert session.calls[0]["json"]["text"] == "custom text"
