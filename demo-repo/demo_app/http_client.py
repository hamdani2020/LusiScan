"""A trivial requests-related helper.

Kept tiny and offline-friendly so the fast test suite never makes a network
call. The `requests` dependency is pinned a patch behind latest, which is the
safe auto-fix patch-bump demo for LusiScan.
"""

import requests


def build_url(base: str, path: str) -> str:
    """Join a base URL and a path with exactly one separating slash."""
    return f"{base.rstrip('/')}/{path.lstrip('/')}"


def default_headers() -> dict:
    """Return default headers, tagging the requests version we build against."""
    return {
        "User-Agent": f"lusiscan-demo/0.1 (requests/{requests.__version__})",
        "Accept": "application/json",
    }
