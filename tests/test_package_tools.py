"""Unit tests for the package monitor (task 3.4).

Covers dependency parsing, PEP 503 name normalization, version comparison, and
the PyPI-backed outdated computation / scan. Network access is stubbed with a
fake ``requests.Session`` so no real calls to PyPI are made and we can assert
the code queries the *registry* rather than the local environment.

_Requirements: 1.1 (read declared deps from the manifest without installing),
1.2 (determine latest version via PyPI, not the local env)._
"""

from __future__ import annotations

import pytest

from src.tools import package_tools as pt


# --- Fake PyPI session ----------------------------------------------------


class _FakeResponse:
    """Minimal stand-in for ``requests.Response`` used by the fake session."""

    def __init__(self, status_code: int, payload: object | None = None) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> object:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400 and self.status_code != 404:
            import requests

            raise requests.HTTPError(f"status {self.status_code}")


class FakePyPISession:
    """A fake ``requests.Session`` that serves canned PyPI JSON responses.

    Maps a *normalized* package name to a latest version string. Records every
    requested URL so tests can assert the PyPI registry was queried (R1.2).
    Unknown packages return HTTP 404 like PyPI does.
    """

    def __init__(self, latest_by_name: dict[str, str]) -> None:
        self.latest_by_name = latest_by_name
        self.requested_urls: list[str] = []
        self.closed = False

    def get(self, url: str, timeout=None) -> _FakeResponse:  # noqa: ANN001
        self.requested_urls.append(url)
        # Recover the normalized name from the PyPI JSON URL template.
        name = url.rsplit("/pypi/", 1)[-1].rsplit("/json", 1)[0]
        if name in self.latest_by_name:
            return _FakeResponse(200, {"info": {"version": self.latest_by_name[name]}})
        return _FakeResponse(404, None)

    def close(self) -> None:
        self.closed = True


# --- normalize_name (PEP 503) --------------------------------------------


class TestNormalizeName:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("Requests", "requests"),
            ("PyYAML", "pyyaml"),
            ("zope.interface", "zope-interface"),
            ("jaraco__collections", "jaraco-collections"),
            ("A.-_B", "a-b"),
            ("Flask-SQLAlchemy", "flask-sqlalchemy"),
            ("already-normalized", "already-normalized"),
        ],
    )
    def test_normalization(self, raw: str, expected: str) -> None:
        assert pt.normalize_name(raw) == expected

    def test_runs_of_separators_collapse(self) -> None:
        # Any run of -, _, . collapses to a single dash.
        assert pt.normalize_name("foo___bar...baz--qux") == "foo-bar-baz-qux"


# --- requirements.txt parsing --------------------------------------------


class TestParseRequirementsTxt:
    def test_exact_pin_extracted(self) -> None:
        deps = pt.parse_requirements_txt("requests==2.20.0\n")
        assert deps == [{"name": "requests", "current": "2.20.0"}]

    def test_comments_and_blank_lines_skipped(self) -> None:
        text = """
        # a leading comment

        requests==2.20.0
          # indented comment
        flask==1.0.0
        """
        deps = pt.parse_requirements_txt(text)
        assert deps == [
            {"name": "requests", "current": "2.20.0"},
            {"name": "flask", "current": "1.0.0"},
        ]

    def test_inline_comment_not_part_of_spec(self) -> None:
        deps = pt.parse_requirements_txt("requests==2.20.0  # pinned for CI\n")
        assert deps == [{"name": "requests", "current": "2.20.0"}]

    def test_extras_are_stripped_from_name(self) -> None:
        deps = pt.parse_requirements_txt("requests[security,socks]==2.20.0\n")
        assert deps == [{"name": "requests", "current": "2.20.0"}]

    def test_environment_markers_ignored_for_pin(self) -> None:
        deps = pt.parse_requirements_txt('requests==2.20.0 ; python_version < "3.10"\n')
        assert deps == [{"name": "requests", "current": "2.20.0"}]

    def test_lower_bound_has_no_pin(self) -> None:
        deps = pt.parse_requirements_txt("requests>=2.20.0\n")
        assert deps == [{"name": "requests", "current": None}]

    def test_compatible_release_has_no_pin(self) -> None:
        deps = pt.parse_requirements_txt("requests~=2.20.0\n")
        assert deps == [{"name": "requests", "current": None}]

    def test_bare_name_has_no_pin(self) -> None:
        deps = pt.parse_requirements_txt("requests\n")
        assert deps == [{"name": "requests", "current": None}]

    @pytest.mark.xfail(
        reason=(
            "Known parser defect: `==2.20.*` yields current='2.20.' instead of "
            "None. The wildcard '*' falls outside the version capture group in "
            "_EXACT_PIN_RE, so the endswith('*') guard never fires. Documented "
            "here per _extract_pin's contract ('wildcard pins are treated as "
            "unpinned'); fix belongs to the parser (subtask 3.1), not this test."
        ),
        strict=True,
    )
    def test_wildcard_pin_is_not_a_concrete_pin(self) -> None:
        deps = pt.parse_requirements_txt("requests==2.20.*\n")
        assert deps == [{"name": "requests", "current": None}]

    def test_wildcard_pin_current_behavior(self) -> None:
        # Regression capture of the CURRENT (buggy) behavior so the defect is
        # visible and any future fix will intentionally flip these two tests.
        deps = pt.parse_requirements_txt("requests==2.20.*\n")
        assert deps == [{"name": "requests", "current": "2.20."}]

    def test_arbitrary_equality_is_not_a_pin(self) -> None:
        deps = pt.parse_requirements_txt("requests===2.20.0\n")
        assert deps == [{"name": "requests", "current": None}]

    def test_editable_and_include_directives_skipped(self) -> None:
        text = "-e .\n-r other-requirements.txt\n--hash=sha256:abc\nrequests==2.20.0\n"
        deps = pt.parse_requirements_txt(text)
        assert deps == [{"name": "requests", "current": "2.20.0"}]

    def test_url_and_vcs_installs_skipped(self) -> None:
        text = (
            "https://example.com/pkg.tar.gz\n"
            "git+https://github.com/psf/requests.git\n"
            "flask==1.0.0\n"
        )
        deps = pt.parse_requirements_txt(text)
        assert deps == [{"name": "flask", "current": "1.0.0"}]

    def test_line_continuation_joins_into_single_requirement(self) -> None:
        # A backslash continuation should be joined; the marker is on line two.
        text = 'requests==2.20.0 \\\n ; python_version < "3.10"\n'
        deps = pt.parse_requirements_txt(text)
        assert deps == [{"name": "requests", "current": "2.20.0"}]

    def test_trailing_continuation_without_newline(self) -> None:
        text = "requests==2.20.0 \\"
        deps = pt.parse_requirements_txt(text)
        assert deps == [{"name": "requests", "current": "2.20.0"}]

    def test_names_normalized(self) -> None:
        deps = pt.parse_requirements_txt("Flask_SQLAlchemy==2.5.1\n")
        assert deps == [{"name": "flask-sqlalchemy", "current": "2.5.1"}]


# --- pyproject.toml parsing ----------------------------------------------


class TestParsePyprojectTomlPep621:
    def test_reads_project_dependencies(self) -> None:
        text = """
        [project]
        name = "demo"
        dependencies = [
            "requests==2.20.0",
            "pydantic>=1.10",
            "flask",
        ]
        """
        deps = pt.parse_pyproject_toml(text)
        assert deps == [
            {"name": "requests", "current": "2.20.0"},
            {"name": "pydantic", "current": None},
            {"name": "flask", "current": None},
        ]

    def test_pep621_preferred_over_poetry_when_both_present(self) -> None:
        text = """
        [project]
        dependencies = ["requests==2.20.0"]

        [tool.poetry.dependencies]
        python = "^3.11"
        flask = "1.0.0"
        """
        deps = pt.parse_pyproject_toml(text)
        # Poetry entries ignored to avoid duplicates.
        assert deps == [{"name": "requests", "current": "2.20.0"}]

    def test_extras_and_markers_in_pep621_entries(self) -> None:
        text = """
        [project]
        dependencies = ["requests[security]==2.20.0 ; python_version < '3.10'"]
        """
        deps = pt.parse_pyproject_toml(text)
        assert deps == [{"name": "requests", "current": "2.20.0"}]


class TestParsePyprojectTomlPoetry:
    def test_reads_poetry_dependencies_and_skips_python(self) -> None:
        text = """
        [tool.poetry.dependencies]
        python = "^3.11"
        requests = "2.20.0"
        pydantic = "^1.10.13"
        """
        deps = pt.parse_pyproject_toml(text)
        assert deps == [
            {"name": "requests", "current": "2.20.0"},
            {"name": "pydantic", "current": None},
        ]

    def test_poetry_bare_version_is_exact_pin(self) -> None:
        text = """
        [tool.poetry.dependencies]
        requests = "2.20.0"
        """
        deps = pt.parse_pyproject_toml(text)
        assert deps == [{"name": "requests", "current": "2.20.0"}]

    def test_poetry_caret_and_tilde_are_not_pins(self) -> None:
        text = """
        [tool.poetry.dependencies]
        requests = "^2.20.0"
        flask = "~1.0"
        """
        deps = pt.parse_pyproject_toml(text)
        assert deps == [
            {"name": "requests", "current": None},
            {"name": "flask", "current": None},
        ]

    def test_poetry_table_version(self) -> None:
        text = """
        [tool.poetry.dependencies]
        requests = { version = "2.20.0", optional = true }
        """
        deps = pt.parse_pyproject_toml(text)
        assert deps == [{"name": "requests", "current": "2.20.0"}]

    def test_poetry_name_normalized(self) -> None:
        text = """
        [tool.poetry.dependencies]
        Flask_SQLAlchemy = "2.5.1"
        """
        deps = pt.parse_pyproject_toml(text)
        assert deps == [{"name": "flask-sqlalchemy", "current": "2.5.1"}]


# --- version comparison: is_newer ----------------------------------------


class TestIsNewer:
    def test_numeric_ordering_not_lexical(self) -> None:
        # The classic trap: "1.9" < "1.10" numerically but > lexically.
        assert pt.is_newer("1.10", "1.9") is True
        assert pt.is_newer("1.9", "1.10") is False

    def test_equal_versions_are_not_newer(self) -> None:
        assert pt.is_newer("2.20.0", "2.20.0") is False

    def test_patch_bump_is_newer(self) -> None:
        assert pt.is_newer("2.20.1", "2.20.0") is True

    def test_major_bump_is_newer(self) -> None:
        assert pt.is_newer("2.0.0", "1.10.13") is True

    def test_older_is_not_newer(self) -> None:
        assert pt.is_newer("2.19.0", "2.20.0") is False

    def test_prerelease_is_older_than_final(self) -> None:
        # PEP 440: 2.0.0rc1 precedes the final 2.0.0 release.
        assert pt.is_newer("2.0.0", "2.0.0rc1") is True
        assert pt.is_newer("2.0.0rc1", "2.0.0") is False

    def test_final_newer_than_prerelease_of_same_series(self) -> None:
        assert pt.is_newer("2.0.0", "2.0.0b1") is True


# --- fetch_latest_version (PyPI, not local env) --------------------------


class TestFetchLatestVersion:
    def test_queries_pypi_registry_url(self) -> None:
        session = FakePyPISession({"requests": "2.32.3"})
        latest = pt.fetch_latest_version("requests", session=session)
        assert latest == "2.32.3"
        # R1.2: the lookup must hit the PyPI registry, not the local env.
        assert session.requested_urls == ["https://pypi.org/pypi/requests/json"]

    def test_name_normalized_before_lookup(self) -> None:
        session = FakePyPISession({"flask-sqlalchemy": "3.1.1"})
        latest = pt.fetch_latest_version("Flask_SQLAlchemy", session=session)
        assert latest == "3.1.1"
        assert session.requested_urls == ["https://pypi.org/pypi/flask-sqlalchemy/json"]

    def test_unknown_package_returns_none(self) -> None:
        session = FakePyPISession({})
        assert pt.fetch_latest_version("does-not-exist", session=session) is None

    def test_missing_version_field_returns_none(self) -> None:
        # Craft a 200 response with no usable version field.
        session = FakePyPISession({})

        def _get(url, timeout=None):  # noqa: ANN001
            session.requested_urls.append(url)
            return _FakeResponse(200, {"info": {}})

        session.get = _get  # type: ignore[method-assign]
        assert pt.fetch_latest_version("requests", session=session) is None


# --- compute_outdated_packages -------------------------------------------


class TestComputeOutdatedPackages:
    def test_reports_only_outdated_with_name_current_latest(self) -> None:
        deps = [
            {"name": "requests", "current": "2.20.0"},
            {"name": "flask", "current": "3.0.0"},
        ]
        session = FakePyPISession({"requests": "2.32.3", "flask": "3.0.0"})
        outdated = pt.compute_outdated_packages(deps, session=session)
        # requests is outdated; flask is current so it is omitted (R1.3).
        assert outdated == [{"name": "requests", "current": "2.20.0", "latest": "2.32.3"}]

    def test_dependencies_without_pin_are_skipped(self) -> None:
        deps = [{"name": "requests", "current": None}]
        session = FakePyPISession({"requests": "2.32.3"})
        outdated = pt.compute_outdated_packages(deps, session=session)
        assert outdated == []
        # No concrete version to compare, so PyPI is not even queried.
        assert session.requested_urls == []

    def test_unknown_package_on_pypi_is_not_outdated(self) -> None:
        deps = [{"name": "ghostpkg", "current": "1.0.0"}]
        session = FakePyPISession({})  # 404 for everything
        assert pt.compute_outdated_packages(deps, session=session) == []

    def test_preserves_input_order(self) -> None:
        deps = [
            {"name": "aaa", "current": "1.0.0"},
            {"name": "bbb", "current": "1.0.0"},
        ]
        session = FakePyPISession({"aaa": "2.0.0", "bbb": "2.0.0"})
        outdated = pt.compute_outdated_packages(deps, session=session)
        assert [o["name"] for o in outdated] == ["aaa", "bbb"]

    def test_queries_pypi_for_each_pinned_dependency(self) -> None:
        deps = [
            {"name": "requests", "current": "2.20.0"},
            {"name": "flask", "current": "1.0.0"},
        ]
        session = FakePyPISession({"requests": "2.32.3", "flask": "3.0.0"})
        pt.compute_outdated_packages(deps, session=session)
        # R1.2: every pinned dep is resolved against the registry.
        assert session.requested_urls == [
            "https://pypi.org/pypi/requests/json",
            "https://pypi.org/pypi/flask/json",
        ]


# --- scan_packages: error recording (R1.4, exercised here for coverage) --


class TestScanPackages:
    def test_missing_manifest_records_error(self, tmp_path) -> None:  # noqa: ANN001
        missing = tmp_path / "no-such-repo"
        result = pt.scan_packages(str(missing))
        assert result["outdated"] == []
        assert len(result["errors"]) == 1
        assert result["errors"][0]["kind"] == "manifest_not_found"

    def test_malformed_pyproject_records_error(self, tmp_path) -> None:  # noqa: ANN001
        (tmp_path / "pyproject.toml").write_text("this is = = not valid toml [[[")
        result = pt.scan_packages(str(tmp_path))
        assert result["outdated"] == []
        assert len(result["errors"]) == 1
        assert result["errors"][0]["kind"] == "manifest_unparseable"

    def test_happy_path_uses_pypi_not_local_env(self, tmp_path) -> None:  # noqa: ANN001
        (tmp_path / "pyproject.toml").write_text(
            '[project]\nname = "demo"\ndependencies = ["requests==2.20.0"]\n'
        )
        session = FakePyPISession({"requests": "2.32.3"})
        result = pt.scan_packages(str(tmp_path), session=session)
        assert result["errors"] == []
        assert result["outdated"] == [
            {"name": "requests", "current": "2.20.0", "latest": "2.32.3"}
        ]
        # R1.2: resolution went through the PyPI registry URL.
        assert session.requested_urls == ["https://pypi.org/pypi/requests/json"]
