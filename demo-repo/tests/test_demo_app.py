"""Tiny, fast test suite for the demo app.

These tests are intentionally quick and offline so GitHub Actions finishes in
seconds. They pass on the pinned versions (requests 2.31.0, pydantic 1.10.x).
"""

import pytest

from demo_app.http_client import build_url, default_headers
from demo_app.models import User, user_summary


def test_build_url_joins_with_single_slash():
    assert build_url("https://api.example.com/", "/users") == "https://api.example.com/users"


def test_default_headers_report_requests_version():
    headers = default_headers()
    assert headers["Accept"] == "application/json"
    assert headers["User-Agent"].startswith("lusiscan-demo/0.1 (requests/")


def test_user_model_validates_and_dumps():
    user = User(id=1, name="Ada", email="ada@example.com")
    summary = user_summary(user)
    assert summary == {"id": 1, "name": "Ada", "email": "ada@example.com"}


def test_user_email_validator_rejects_bad_email():
    with pytest.raises(ValueError):
        User(id=2, name="Bad", email="not-an-email")
