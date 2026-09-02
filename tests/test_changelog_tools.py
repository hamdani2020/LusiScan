"""Unit tests for the changelog fetcher (task 4).

Covers the hardcoded package→GitHub-repo mapping and version-range fetch
(task 4.1), Nova Lite summarization via an injected model client (task 4.2),
and the "force ``low`` confidence on any fetch failure" contract (task 4.3).

GitHub network access is stubbed with a fake ``requests.Session`` so no real
API calls are made, and the Nova Lite client is a fake satisfying the
``ChangelogModel`` Protocol — no AWS/Bedrock dependency in these tests.

_Requirements: 2.1 (fetch release notes for the version range of a supported
demo package), 2.2 (summarize focusing on breaking changes / deprecations /
migration steps), 2.4 (force ``low`` confidence + human review on fetch
failure)._
"""

from __future__ import annotations

import pytest
import requests

from src.tools import changelog_tools as ct


# --- Fakes: GitHub session + Nova Lite model ------------------------------


class _FakeResponse:
    """Minimal stand-in for ``requests.Response``."""

    def __init__(self, status_code: int, payload: object | None = None) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> object:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")


class FakeGitHubSession:
    """A fake ``requests.Session`` serving canned GitHub Releases responses.

    Maps ``owner/repo`` → a list of release dicts (as GitHub returns them).
    Records requested URLs so tests can assert the hardcoded map was used.
    """

    def __init__(self, releases_by_repo: dict[str, list[dict]]) -> None:
        self.releases_by_repo = releases_by_repo
        self.requested_urls: list[str] = []

    def get(self, url: str, params=None, timeout=None) -> _FakeResponse:  # noqa: ANN001
        self.requested_urls.append(url)
        repo = url.rsplit("/repos/", 1)[-1].rsplit("/releases", 1)[0]
        if repo in self.releases_by_repo:
            return _FakeResponse(200, self.releases_by_repo[repo])
        return _FakeResponse(404, None)


class FakeRaisingSession:
    """A fake session whose ``get`` always raises a network error."""

    def __init__(self, exc: Exception | None = None) -> None:
        self.exc = exc or requests.ConnectionError("boom")

    def get(self, url: str, params=None, timeout=None):  # noqa: ANN001
        raise self.exc


class FakeModel:
    """A fake Nova Lite client satisfying the ``ChangelogModel`` Protocol.

    Records the prompt it was handed so tests can assert the summarization
    prompt focuses on breaking changes / deprecations / migration steps.
    """

    def __init__(self, reply: str = "SUMMARY") -> None:
        self.reply = reply
        self.prompts: list[str] = []

    def summarize(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply


class FakeRaisingModel:
    """A fake model whose ``summarize`` always raises."""

    def summarize(self, prompt: str) -> str:  # noqa: ARG002
        raise RuntimeError("bedrock unavailable")


def _release(tag: str, body: str = "") -> dict:
    """Build a minimal GitHub release payload entry."""
    return {"tag_name": tag, "name": tag, "body": body}


# --- normalize_name (PEP 503) --------------------------------------------


class TestNormalizeName:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("Requests", "requests"),
            ("PyDantic", "pydantic"),
            ("already-normalized", "already-normalized"),
        ],
    )
    def test_normalization(self, raw: str, expected: str) -> None:
        assert ct.normalize_name(raw) == expected


# --- Hardcoded map (task 4.1) --------------------------------------------


class TestPackageRepoMap:
    def test_only_the_two_demo_packages_are_mapped(self) -> None:
        assert set(ct.PACKAGE_REPO_MAP) == {"requests", "pydantic"}

    def test_maps_to_expected_github_repos(self) -> None:
        assert ct.PACKAGE_REPO_MAP["requests"] == "psf/requests"
        assert ct.PACKAGE_REPO_MAP["pydantic"] == "pydantic/pydantic"


# --- Version range selection (task 4.1) ----------------------------------


class TestInUpgradeRange:
    def test_target_is_inclusive(self) -> None:
        assert ct._in_upgrade_range("2.32.3", "2.31.0", "2.32.3") is True

    def test_current_is_exclusive(self) -> None:
        assert ct._in_upgrade_range("2.31.0", "2.31.0", "2.32.3") is False

    def test_intermediate_release_is_included(self) -> None:
        assert ct._in_upgrade_range("2.32.0", "2.31.0", "2.32.3") is True

    def test_above_target_excluded(self) -> None:
        assert ct._in_upgrade_range("2.33.0", "2.31.0", "2.32.3") is False

    def test_below_current_excluded(self) -> None:
        assert ct._in_upgrade_range("2.30.0", "2.31.0", "2.32.3") is False

    def test_v_prefixed_tag_is_normalized(self) -> None:
        assert ct._in_upgrade_range("v2.0.0", "1.10.13", "2.0.0") is True

    def test_unparseable_tag_excluded(self) -> None:
        assert ct._in_upgrade_range("nightly", "2.31.0", "2.32.3") is False


# --- fetch_release_notes (task 4.1, R2.1) --------------------------------


class TestFetchReleaseNotes:
    def test_uses_hardcoded_repo_url_and_filters_range(self) -> None:
        session = FakeGitHubSession(
            {
                "psf/requests": [
                    _release("v2.33.0", "too new"),
                    _release("v2.32.3", "patch fixes"),
                    _release("v2.32.0", "minor changes"),
                    _release("v2.31.0", "current, excluded"),
                    _release("v2.30.0", "older, excluded"),
                ]
            }
        )
        notes = ct.fetch_release_notes("requests", "2.31.0", "2.32.3", session=session)
        # R2.1: hit the hardcoded psf/requests releases endpoint.
        assert session.requested_urls == [
            "https://api.github.com/repos/psf/requests/releases"
        ]
        # Only (2.31.0, 2.32.3] survives, newest-first.
        assert [n["version"] for n in notes] == ["2.32.3", "2.32.0"]
        assert notes[0]["body"] == "patch fixes"

    def test_package_name_normalized_before_lookup(self) -> None:
        session = FakeGitHubSession({"pydantic/pydantic": [_release("2.0", "v2 rewrite")]})
        notes = ct.fetch_release_notes("PyDantic", "1.10.13", "2.0", session=session)
        assert session.requested_urls == [
            "https://api.github.com/repos/pydantic/pydantic/releases"
        ]
        assert [n["version"] for n in notes] == ["2.0"]

    def test_unmapped_package_raises_keyerror(self) -> None:
        session = FakeGitHubSession({})
        with pytest.raises(KeyError):
            ct.fetch_release_notes("flask", "1.0.0", "2.0.0", session=session)

    def test_missing_body_becomes_empty_string(self) -> None:
        session = FakeGitHubSession({"psf/requests": [{"tag_name": "2.32.3"}]})
        notes = ct.fetch_release_notes("requests", "2.31.0", "2.32.3", session=session)
        assert notes == [{"version": "2.32.3", "body": ""}]


# --- summarize_release_notes (task 4.2, R2.2) ----------------------------


class TestSummarizeReleaseNotes:
    def test_delegates_to_injected_model_and_returns_summary(self) -> None:
        model = FakeModel(reply="  breaking: X; deprecated: Y  ")
        notes = [{"version": "2.0", "body": "removed foo()"}]
        summary = ct.summarize_release_notes(
            "pydantic", "1.10.13", "2.0", notes, model=model
        )
        # Stripped model output is returned verbatim.
        assert summary == "breaking: X; deprecated: Y"
        assert len(model.prompts) == 1

    def test_prompt_focuses_on_breaking_deprecations_migration(self) -> None:
        model = FakeModel()
        notes = [{"version": "2.0", "body": "removed foo()"}]
        ct.summarize_release_notes("pydantic", "1.10.13", "2.0", notes, model=model)
        prompt = model.prompts[0].lower()
        # R2.2: the summary is explicitly focused on the three concerns.
        assert "breaking change" in prompt
        assert "deprecation" in prompt
        assert "migration step" in prompt
        # Raw notes are embedded so the model has material to summarize.
        assert "removed foo()" in model.prompts[0]


# --- fetch_and_summarize_changelog: happy path (R2.1 + R2.2) -------------


class TestFetchAndSummarizeChangelogHappyPath:
    def test_returns_summary_and_leaves_confidence_for_planner(self) -> None:
        session = FakeGitHubSession(
            {"psf/requests": [_release("2.32.3", "security patch")]}
        )
        model = FakeModel(reply="no breaking changes; safe patch bump")
        result = ct.fetch_and_summarize_changelog(
            "requests", "2.31.0", "2.32.3", model=model, session=session
        )
        assert result["error"] is None
        assert result["summary"] == "no breaking changes; safe patch bump"
        assert [n["version"] for n in result["notes"]] == ["2.32.3"]
        # On success the planner (Nova Pro) sets real confidence, so it's None.
        assert result["confidence"] is None
        assert result["package"] == "requests"


# --- fetch_and_summarize_changelog: failure → low confidence (task 4.3) --


class TestFetchAndSummarizeChangelogForcesLowConfidence:
    def test_unmapped_package_is_low_confidence(self) -> None:
        model = FakeModel()
        result = ct.fetch_and_summarize_changelog(
            "flask", "1.0.0", "2.0.0", model=model, session=FakeGitHubSession({})
        )
        # R2.4: unmapped → low confidence, no summary, structured error.
        assert result["confidence"] == ct.CONFIDENCE_LOW
        assert result["summary"] is None
        assert result["notes"] == []
        assert result["error"]["kind"] == "unmapped_package"
        # The model is never consulted when we can't even map the package.
        assert model.prompts == []

    def test_network_failure_is_low_confidence(self) -> None:
        result = ct.fetch_and_summarize_changelog(
            "requests",
            "2.31.0",
            "2.32.3",
            model=FakeModel(),
            session=FakeRaisingSession(),
        )
        # R2.4: a fetch failure must force low confidence, not crash.
        assert result["confidence"] == ct.CONFIDENCE_LOW
        assert result["error"]["kind"] == "fetch_failed"

    def test_http_error_is_low_confidence(self) -> None:
        # 404/500 etc. raise via raise_for_status → RequestException path.
        session = FakeGitHubSession({})  # psf/requests absent → 404
        result = ct.fetch_and_summarize_changelog(
            "requests", "2.31.0", "2.32.3", model=FakeModel(), session=session
        )
        assert result["confidence"] == ct.CONFIDENCE_LOW
        assert result["error"]["kind"] == "fetch_failed"

    def test_no_release_notes_in_range_is_low_confidence(self) -> None:
        # Repo resolves, but nothing falls in the (current, target] window.
        session = FakeGitHubSession({"psf/requests": [_release("2.30.0", "old")]})
        result = ct.fetch_and_summarize_changelog(
            "requests", "2.31.0", "2.32.3", model=FakeModel(), session=session
        )
        assert result["confidence"] == ct.CONFIDENCE_LOW
        assert result["error"]["kind"] == "no_release_notes"

    def test_summarization_failure_is_low_confidence(self) -> None:
        session = FakeGitHubSession(
            {"psf/requests": [_release("2.32.3", "security patch")]}
        )
        result = ct.fetch_and_summarize_changelog(
            "requests",
            "2.31.0",
            "2.32.3",
            model=FakeRaisingModel(),
            session=session,
        )
        # R2.4: even a model failure degrades safely to low confidence.
        assert result["confidence"] == ct.CONFIDENCE_LOW
        assert result["error"]["kind"] == "summarize_failed"
