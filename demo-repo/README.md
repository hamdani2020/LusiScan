# LusiScan Demo Repo

A small, self-contained Python project used as the **controlled target repo**
for [LusiScan](../README.md). It intentionally pins outdated dependencies so the
agent has real, scoped migrations to detect, plan, apply, and validate.

## Intentionally outdated pins

| Package    | Pinned     | Demo scenario                                   |
|------------|------------|-------------------------------------------------|
| `requests` | `2.31.0`   | Safe auto-fix **patch bump** (Requirement 9.4)  |
| `pydantic` | `1.10.13`  | Human-in-the-loop **1 → 2 major upgrade** (9.4) |

## Layout

```
demo-repo/
├── demo_app/
│   ├── __init__.py
│   ├── http_client.py   # trivial requests helper (offline)
│   └── models.py        # pydantic 1.x-style model (validator, Config, .dict())
├── tests/
│   └── test_demo_app.py # tiny, fast suite
├── .github/workflows/
│   └── tests.yml        # installs deps + runs pytest (Requirement 4.1)
└── pyproject.toml
```

## Run the tests locally

```bash
cd demo-repo
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

The suite is offline and finishes in seconds, so the GitHub Actions run that
LusiScan triggers on its upgrade branch completes quickly.

## Pushing to a separate GitHub repo

This directory is self-contained. To use it as the live target, push its
contents to a dedicated GitHub repository (so the `.github/workflows/tests.yml`
workflow is active there) and point LusiScan at that repo.
