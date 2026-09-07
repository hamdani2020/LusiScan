"""Unit tests for the five orchestrator ``@tool`` stage functions (task 10.1).

Task 10.1 pins two things about the Strands Agent layer that the full-loop
integration test (``tests/test_orchestrator.py``) only exercises indirectly:

1. Each of the five stages (``scan_packages``, ``fetch_changelog``,
   ``plan_migration``, ``apply_migration``, ``validate_branch``) is a plain
   function decorated with ``@tool`` that is **invoked by calling it directly**
   (design.md → API correction #1: no ``Tool(...)`` wrapper, no ``.run()``, no
   ``AgentExecutor``) and returns a **structured dict** (design.md → Agent
   layer: "Tools return structured dicts").
2. The stages **compose in order** by ordinary function calls
   (Monitor → Planner → Executor → Validator), and the whole thing stays
   importable/callable whether or not the Strands SDK is installed (the import
   guard falls back to an identity decorator).

These tests target the stage functions in isolation — no orchestrator, no state
store, no GitHub — so a regression in a single stage's shape is caught directly
rather than only surfacing as a full-loop failure. No mocking of the code under
test: the underlying tool modules run for real, with only injected fake Nova
models (the documented ``PlannerModel`` / ``ChangelogModel`` seams) and a fixed
offline scan standing in for the PyPI-backed monitor.

_Requirements: 1.x (detect), 2.x (analyze/plan), 3.x (refactor)._
"""

from __future__ import annotations

from src import main


# --- Fake Nova model (PlannerModel + ChangelogModel seams) -----------------


class FakeNovaModel:
    """A fake Bedrock/Nova client satisfying both PlannerModel and ChangelogModel."""

    def summarize(self, prompt: str) -> str:
        return "Summary: breaking changes and migration steps for the upgrade."

    def converse_json(self, prompt: str, **kwargs) -> dict:
        return {
            "confidence": "high",
            "strategy": "auto_fix",
            "estimated_risk": "low",
            "breaking_changes": [],
            "reasoning": "Safe bump.",
        }


# =====================================================================
# Each stage is directly callable and returns a structured dict (10.1)
# =====================================================================


def test_all_five_stages_are_directly_callable_tool_functions() -> None:
    # The Strands convention: a @tool is a plain function called directly.
    # With the SDK absent the identity-decorator fallback keeps them callable;
    # with the SDK present the decorated function is still callable. Either way
    # these are ordinary callables — no Tool(...) wrapper / .run() / executor.
    for stage in (
        main.scan_packages,
        main.fetch_changelog,
        main.plan_migration,
        main.apply_migration,
        main.validate_branch,
    ):
        assert callable(stage)


def test_scan_packages_returns_structured_scan_dict(tmp_path) -> None:
    # Monitor stage over a real (empty) directory: no manifest to parse, so the
    # underlying tool records the situation as data rather than raising (R1.4).
    result = main.scan_packages(str(tmp_path))

    assert isinstance(result, dict)
    assert set(result) >= {"repo_path", "outdated", "errors"}
    assert isinstance(result["outdated"], list)
    assert isinstance(result["errors"], list)


def test_fetch_changelog_degrades_to_low_confidence_dict_on_unmapped_package() -> None:
    # Planner part 1: an unmapped package needs no network — the stage returns a
    # ChangelogResult dict with confidence pinned to "low" (R2.4), never raises.
    result = main.fetch_changelog(
        "definitely-not-a-real-package", "1.0.0", "2.0.0", model=FakeNovaModel()
    )

    assert isinstance(result, dict)
    assert set(result) >= {"package", "current", "target", "confidence"}
    assert result["confidence"] == "low"


def test_plan_migration_returns_normalized_plan_dict() -> None:
    # Planner part 2: a well-formed changelog + fake Nova Pro yields a normalized
    # plan dict with the R2.3 fields.
    changelog = {
        "package": "requests",
        "current": "2.31.0",
        "target": "2.32.3",
        "summary": "Patch fixes; no breaking changes.",
        "notes": "…",
        "confidence": None,
        "error": None,
    }

    plan = main.plan_migration(changelog, model=FakeNovaModel())

    assert isinstance(plan, dict)
    assert set(plan) >= {"confidence", "strategy", "estimated_risk", "breaking_changes"}
    assert plan["confidence"] == "high"
    assert plan["strategy"] == "auto_fix"


def test_plan_migration_forces_low_plan_on_low_confidence_changelog() -> None:
    # A changelog that already signalled low confidence routes to human review
    # without a model call — still a structured plan dict (R2.4).
    low_changelog = {
        "package": "requests",
        "current": "2.31.0",
        "target": "2.32.3",
        "summary": "",
        "notes": "",
        "confidence": "low",
        "error": "changelog fetch failed",
    }

    plan = main.plan_migration(low_changelog, model=FakeNovaModel())

    assert isinstance(plan, dict)
    assert plan["confidence"] == "low"


def test_apply_migration_version_bump_returns_structured_refactor_dict() -> None:
    # Executor stage: a version-bump-only migration over a real manifest returns
    # the refactor result shape, with the manifest change applied and re-parsed
    # valid (R3.1/R3.4). requests is a version-bump-only package (no code xform).
    plan = {
        "package": "requests",
        "current": "2.31.0",
        "target": "2.32.3",
        "strategy": "auto_fix",
        "confidence": "high",
    }
    manifest = (
        "[project]\n"
        "dependencies = [\n"
        '    "requests==2.31.0",\n'
        "]\n"
    )

    result = main.apply_migration(plan, manifest=("pyproject.toml", manifest))

    assert isinstance(result, dict)
    assert set(result) >= {"package", "strategy", "changes", "flagged", "diff", "applied"}
    assert isinstance(result["changes"], (list, dict))
    # The bump was applied: the new version appears in the produced diff.
    assert "2.32.3" in result["diff"]


def test_validate_branch_returns_status_details_dict() -> None:
    # Validator stage: default stub returns the {status, details} contract every
    # downstream consumer relies on, with a "skipped" status until task 12.
    result = main.validate_branch("some-branch")

    assert isinstance(result, dict)
    assert set(result) == {"status", "details"}
    assert result["status"] == main.VALIDATION_SKIPPED


# =====================================================================
# The stages compose in order by direct calls (Monitor→…→Validator, 10.1)
# =====================================================================


def test_stages_compose_in_order_by_direct_calls() -> None:
    # Planner → Executor → Validator, wired by ordinary function calls (no
    # AgentExecutor). Start from a changelog, plan it, apply it, validate it —
    # each hand-off is a plain dict, proving the composition seam.
    model = FakeNovaModel()

    changelog = main.fetch_changelog(
        "definitely-not-a-real-package", "1.0.0", "2.0.0", model=model
    )
    assert isinstance(changelog, dict)

    plan = main.plan_migration(changelog, model=model)
    assert isinstance(plan, dict) and "confidence" in plan

    refactor = main.apply_migration(plan)
    assert isinstance(refactor, dict) and "changes" in refactor

    validation = main.validate_branch("feature-branch")
    assert isinstance(validation, dict) and set(validation) == {"status", "details"}
