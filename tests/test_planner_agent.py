"""Unit tests for the migration planner (task 6).

Covers building the Nova Pro planning prompt and parsing a structured plan into
the R2.3 shape (task 6.1), the "retry once on non-JSON, else ``low`` confidence"
fallback (task 6.2), and plan parsing/normalization against fixture model
responses (task 6.3).

No AWS/Bedrock calls are made: the planner's model is injected. Two flavors of
fake are used:

- :class:`FakePlannerModel` returns canned parsed JSON directly (mirroring what
  ``converse_json`` yields) so tests can assert normalization without depending
  on extraction details.
- :class:`RealParsingModel` wraps the *actual* ``BedrockClient.converse_json``
  extraction over a fake ``bedrock-runtime`` client, so we can prove the planner
  copes with JSON wrapped in prose/a code fence and with a real
  :class:`JSONResponseError` after the stricter retry — the exact contract
  ``converse_json(retries=1)`` provides.

_Requirements: 2.3 (structured migration plan: confidence / strategy /
estimated_risk / breaking_changes), 2.4 (unavailable changelog or non-JSON →
``low`` confidence + human review), 8.1 (Nova Pro for planning), 8.3 (structured
JSON output parsed deterministically)._
"""

from __future__ import annotations

from typing import Any

import pytest

from src.agents import planner_agent as pa
from src.models.bedrock_client import BedrockClient, JSONResponseError


# --- Fakes ----------------------------------------------------------------


class FakePlannerModel:
    """A fake planner model returning a canned *already-parsed* JSON value.

    Records the calls so tests can assert the planner routes to Nova Pro
    (``model="pro"``) and asks for the single stricter retry (``retries=1``).
    """

    def __init__(self, reply: Any) -> None:
        self.reply = reply
        self.calls: list[dict[str, Any]] = []

    def converse_json(self, prompt: str, **kwargs: Any) -> Any:
        self.calls.append({"prompt": prompt, **kwargs})
        return self.reply


class RaisingPlannerModel:
    """A fake model whose ``converse_json`` raises ``JSONResponseError``.

    Simulates the hard failure ``converse_json`` raises when even the stricter
    retry did not yield parseable JSON.
    """

    def __init__(self, raw: str = "not json at all") -> None:
        self.raw = raw
        self.calls: list[dict[str, Any]] = []

    def converse_json(self, prompt: str, **kwargs: Any) -> Any:
        self.calls.append({"prompt": prompt, **kwargs})
        raise JSONResponseError("no JSON after retries", raw=self.raw)


def _converse_response(text: str) -> dict:
    """Build a minimal ``bedrock-runtime.converse`` response with one text block."""
    return {"output": {"message": {"role": "assistant", "content": [{"text": text}]}}}


class _FakeBedrockRuntime:
    """A fake ``bedrock-runtime`` client replaying canned ``converse`` replies.

    Lets us drive the *real* ``BedrockClient.converse_json`` extraction/retry
    path from fixture model text, so the planner is exercised end-to-end without
    AWS.
    """

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls: list[dict] = []

    def converse(self, **kwargs: Any) -> dict:
        self.calls.append(kwargs)
        idx = min(len(self.calls) - 1, len(self.replies) - 1)
        return _converse_response(self.replies[idx])


def RealParsingModel(replies: list[str]) -> BedrockClient:
    """A real ``BedrockClient`` over a fake runtime replaying ``replies``."""
    return BedrockClient(client=_FakeBedrockRuntime(replies))


def _changelog(
    *,
    package: str = "requests",
    current: str = "2.31.0",
    target: str = "2.32.3",
    summary: str | None = "no breaking changes; safe patch bump",
    confidence: str | None = None,
    error: dict | None = None,
) -> dict[str, Any]:
    """Build a ``ChangelogResult``-shaped dict for the planner under test."""
    return {
        "package": package,
        "current": current,
        "target": target,
        "summary": summary,
        "notes": [],
        "confidence": confidence,
        "error": error,
    }


# --- Enum constants match R2.3 -------------------------------------------


class TestPlanEnums:
    def test_confidence_levels(self) -> None:
        assert pa.CONFIDENCE_LEVELS == {"high", "low"}

    def test_strategies(self) -> None:
        assert pa.STRATEGIES == {"auto_fix", "guided_pr", "human_required"}

    def test_risk_levels(self) -> None:
        assert pa.RISK_LEVELS == {"low", "medium", "high"}


# --- Prompt building (task 6.1) ------------------------------------------


class TestBuildPlanPrompt:
    def test_prompt_names_package_range_and_schema(self) -> None:
        prompt = pa._build_plan_prompt("pydantic", "1.10.13", "2.0", "v2 rewrite")
        assert "pydantic" in prompt
        assert "1.10.13" in prompt
        assert "2.0" in prompt
        # R2.3: the four mandated fields are requested by name.
        assert "confidence" in prompt
        assert "strategy" in prompt
        assert "estimated_risk" in prompt
        assert "breaking_changes" in prompt
        # The changelog material is embedded for the model to reason over.
        assert "v2 rewrite" in prompt


# --- Valid plan parses + normalizes (task 6.1, 6.3, R2.3) ----------------


class TestValidPlanParsing:
    def test_valid_plan_parses_and_normalizes(self) -> None:
        model = FakePlannerModel(
            {
                "confidence": "high",
                "strategy": "auto_fix",
                "estimated_risk": "low",
                "breaking_changes": [],
                "reasoning": "safe patch bump",
            }
        )
        plan = pa.plan_migration(_changelog(), model=model)
        assert plan["confidence"] == "high"
        assert plan["strategy"] == "auto_fix"
        assert plan["estimated_risk"] == "low"
        assert plan["breaking_changes"] == []
        assert plan["reasoning"] == "safe patch bump"
        assert plan["error"] is None
        # Package/version context is carried through from the changelog result.
        assert plan["package"] == "requests"
        assert plan["current"] == "2.31.0"
        assert plan["target"] == "2.32.3"

    def test_routes_to_nova_pro_with_single_retry(self) -> None:
        # R8.1 + R8.3/design.md: planning uses Nova Pro and asks for retries=1.
        model = FakePlannerModel(
            {
                "confidence": "low",
                "strategy": "guided_pr",
                "estimated_risk": "medium",
                "breaking_changes": ["x"],
            }
        )
        pa.plan_migration(_changelog(), model=model)
        assert model.calls[0]["model"] == "pro"
        assert model.calls[0]["retries"] == 1

    def test_enum_values_are_case_and_whitespace_normalized(self) -> None:
        model = FakePlannerModel(
            {
                "confidence": " HIGH ",
                "strategy": "Auto_Fix",
                "estimated_risk": "LOW",
                "breaking_changes": [],
            }
        )
        plan = pa.plan_migration(_changelog(), model=model)
        assert plan["confidence"] == "high"
        assert plan["strategy"] == "auto_fix"
        assert plan["estimated_risk"] == "low"
        assert plan["error"] is None

    def test_breaking_changes_scalar_is_coerced_to_list(self) -> None:
        model = FakePlannerModel(
            {
                "confidence": "low",
                "strategy": "guided_pr",
                "estimated_risk": "high",
                "breaking_changes": "removed foo()",
            }
        )
        plan = pa.plan_migration(_changelog(), model=model)
        assert plan["breaking_changes"] == ["removed foo()"]

    def test_breaking_changes_missing_defaults_to_empty_list(self) -> None:
        model = FakePlannerModel(
            {
                "confidence": "high",
                "strategy": "auto_fix",
                "estimated_risk": "low",
            }
        )
        plan = pa.plan_migration(_changelog(), model=model)
        assert plan["breaking_changes"] == []

    def test_breaking_changes_list_items_stringified_and_cleaned(self) -> None:
        model = FakePlannerModel(
            {
                "confidence": "low",
                "strategy": "guided_pr",
                "estimated_risk": "medium",
                "breaking_changes": ["  drop py36  ", "", 42],
            }
        )
        plan = pa.plan_migration(_changelog(), model=model)
        assert plan["breaking_changes"] == ["drop py36", "42"]


# --- Extraction from prose / fenced JSON via real converse_json (R8.3) ---


class TestRealJsonExtraction:
    def test_plan_wrapped_in_prose_is_parsed(self) -> None:
        reply = (
            'The plan is {"confidence": "low", "strategy": "guided_pr", '
            '"estimated_risk": "medium", "breaking_changes": ["config change"]} '
            "and that's my recommendation."
        )
        plan = pa.plan_migration(_changelog(), model=RealParsingModel([reply]))
        assert plan["confidence"] == "low"
        assert plan["strategy"] == "guided_pr"
        assert plan["estimated_risk"] == "medium"
        assert plan["breaking_changes"] == ["config change"]
        assert plan["error"] is None

    def test_plan_in_code_fence_is_parsed(self) -> None:
        reply = (
            "Here is the plan:\n```json\n"
            '{"confidence": "high", "strategy": "auto_fix", '
            '"estimated_risk": "low", "breaking_changes": []}\n'
            "```\n"
        )
        plan = pa.plan_migration(_changelog(), model=RealParsingModel([reply]))
        assert plan["confidence"] == "high"
        assert plan["strategy"] == "auto_fix"
        assert plan["error"] is None

    def test_non_json_then_valid_json_on_retry(self) -> None:
        # First reply is prose (no JSON); the stricter retry returns a valid plan.
        replies = [
            "Sorry, I can't produce JSON right now.",
            '{"confidence": "low", "strategy": "human_required", '
            '"estimated_risk": "high", "breaking_changes": ["v1 API removed"]}',
        ]
        model = RealParsingModel(replies)
        plan = pa.plan_migration(_changelog(), model=model)
        assert plan["confidence"] == "low"
        assert plan["strategy"] == "human_required"
        assert plan["error"] is None
        # Confirms the retry actually happened (two underlying converse calls).
        assert len(model.client.calls) == 2


# --- Invalid / missing fields default to low confidence (task 6.2, 6.3) --


class TestInvalidPlanDefaultsToLowConfidence:
    def test_missing_required_field_defaults_low(self) -> None:
        # No confidence field at all → degrade to the safe fallback.
        model = FakePlannerModel({"strategy": "auto_fix", "estimated_risk": "low"})
        plan = pa.plan_migration(_changelog(), model=model)
        assert plan["confidence"] == "low"
        assert plan["strategy"] == "human_required"
        assert plan["error"]["kind"] == "invalid_plan"

    def test_invalid_enum_value_defaults_low(self) -> None:
        model = FakePlannerModel(
            {
                "confidence": "maybe",  # not a valid confidence level
                "strategy": "auto_fix",
                "estimated_risk": "low",
                "breaking_changes": [],
            }
        )
        plan = pa.plan_migration(_changelog(), model=model)
        assert plan["confidence"] == "low"
        assert plan["strategy"] == "human_required"
        assert plan["error"]["kind"] == "invalid_plan"

    def test_invalid_strategy_defaults_low(self) -> None:
        model = FakePlannerModel(
            {
                "confidence": "high",
                "strategy": "yolo_fix",  # not a valid strategy
                "estimated_risk": "low",
                "breaking_changes": [],
            }
        )
        plan = pa.plan_migration(_changelog(), model=model)
        assert plan["confidence"] == "low"
        assert plan["strategy"] == "human_required"
        assert plan["error"]["kind"] == "invalid_plan"

    def test_non_object_plan_defaults_low(self) -> None:
        # The model returned a JSON array instead of an object.
        model = FakePlannerModel(["not", "an", "object"])
        plan = pa.plan_migration(_changelog(), model=model)
        assert plan["confidence"] == "low"
        assert plan["strategy"] == "human_required"
        assert plan["error"]["kind"] == "invalid_plan"

    def test_invalid_plan_preserves_identified_breaking_changes(self) -> None:
        # Even when the plan is otherwise invalid, keep breaking changes so a
        # human reviewer isn't flying blind.
        model = FakePlannerModel(
            {
                "confidence": "bogus",
                "strategy": "auto_fix",
                "estimated_risk": "low",
                "breaking_changes": ["dropped Python 3.7 support"],
            }
        )
        plan = pa.plan_migration(_changelog(), model=model)
        assert plan["confidence"] == "low"
        assert plan["breaking_changes"] == ["dropped Python 3.7 support"]


# --- Non-JSON after retry → low confidence fallback (task 6.2, R2.4) -----


class TestJsonResponseErrorFallback:
    def test_json_response_error_yields_low_confidence(self) -> None:
        model = RaisingPlannerModel(raw="still not json")
        plan = pa.plan_migration(_changelog(), model=model)
        assert plan["confidence"] == "low"
        assert plan["strategy"] == "human_required"
        assert plan["estimated_risk"] == "high"
        assert plan["error"]["kind"] == "non_json_response"
        # The planner still consulted the model (once) before falling back.
        assert len(model.calls) == 1

    def test_real_client_exhausts_retries_then_low_confidence(self) -> None:
        # Both the first attempt and the stricter retry return non-JSON, so the
        # real converse_json raises JSONResponseError and the planner degrades.
        model = RealParsingModel(["no json here", "still no json"])
        plan = pa.plan_migration(_changelog(), model=model)
        assert plan["confidence"] == "low"
        assert plan["error"]["kind"] == "non_json_response"
        # Confirms one initial attempt + one stricter retry.
        assert len(model.client.calls) == 2


# --- Already-low changelog short-circuits without calling the model ------


class TestAlreadyLowChangelogShortCircuits:
    def test_low_changelog_returns_low_plan_without_model_call(self) -> None:
        model = FakePlannerModel(
            {
                "confidence": "high",
                "strategy": "auto_fix",
                "estimated_risk": "low",
                "breaking_changes": [],
            }
        )
        low_changelog = _changelog(
            summary=None,
            confidence="low",
            error={"kind": "fetch_failed", "message": "changelog fetch failed: boom"},
        )
        plan = pa.plan_migration(low_changelog, model=model)
        # R2.4: routed to a human, and the model was NEVER consulted.
        assert plan["confidence"] == "low"
        assert plan["strategy"] == "human_required"
        assert plan["error"]["kind"] == "changelog_low_confidence"
        assert model.calls == []
        # The upstream changelog error message is surfaced in the reasoning.
        assert "boom" in plan["reasoning"]

    def test_short_circuit_carries_version_context(self) -> None:
        model = FakePlannerModel({})
        low_changelog = _changelog(
            package="pydantic",
            current="1.10.13",
            target="2.0",
            summary=None,
            confidence="low",
            error={"kind": "unmapped_package", "message": "no mapping"},
        )
        plan = pa.plan_migration(low_changelog, model=model)
        assert plan["package"] == "pydantic"
        assert plan["current"] == "1.10.13"
        assert plan["target"] == "2.0"
        assert model.calls == []
