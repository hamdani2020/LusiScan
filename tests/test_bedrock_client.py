"""Unit tests for the Bedrock/Nova client wrapper (task 5).

Covers model routing across the three Nova tiers (R8.1/R8.2), the
low-temperature-by-default determinism contract (design.md), and the structured
JSON request/parse path with a single stricter retry on non-JSON output
(R8.3 + design.md's "Model returns non-JSON | Retry once with stricter prompt;
else ``low`` confidence").

No real AWS calls are made: a fake ``bedrock-runtime`` client that records
``converse(**kwargs)`` invocations and returns canned ``converse`` responses is
injected into :class:`BedrockClient`.

_Requirements: 8.1 (Nova Pro for planning/reasoning), 8.2 (Nova Lite for
summarization / Nova Micro for classification), 8.3 (request structured JSON
output the pipeline can parse deterministically)._
"""

from __future__ import annotations

import pytest

from src.models import bedrock_client as bc
from src.models.bedrock_client import BedrockClient, JSONResponseError


# --- Fake bedrock-runtime client ------------------------------------------


def _converse_response(text: str) -> dict:
    """Build a minimal ``bedrock-runtime.converse`` response with one text block."""
    return {"output": {"message": {"role": "assistant", "content": [{"text": text}]}}}


class FakeBedrockRuntime:
    """A fake ``bedrock-runtime`` client that records calls and replays replies.

    ``replies`` is consumed one-per-``converse``-call (FIFO); if exhausted, the
    last reply is repeated. Each call's kwargs are recorded on ``calls`` so
    tests can assert model routing and inference config.
    """

    def __init__(self, replies: list[str] | None = None) -> None:
        self.replies = list(replies) if replies else ["{}"]
        self.calls: list[dict] = []

    def converse(self, **kwargs) -> dict:  # noqa: ANN003
        self.calls.append(kwargs)
        idx = min(len(self.calls) - 1, len(self.replies) - 1)
        return _converse_response(self.replies[idx])


# --- Model id resolution / routing (R8.1, R8.2) ---------------------------


class TestResolveModelId:
    @pytest.mark.parametrize(
        "alias,expected",
        [
            ("pro", bc.NOVA_PRO_MODEL_ID),
            ("lite", bc.NOVA_LITE_MODEL_ID),
            ("micro", bc.NOVA_MICRO_MODEL_ID),
            ("planner", bc.NOVA_PRO_MODEL_ID),
            ("reasoning", bc.NOVA_PRO_MODEL_ID),
            ("summary", bc.NOVA_LITE_MODEL_ID),
            ("changelog", bc.NOVA_LITE_MODEL_ID),
            ("classify", bc.NOVA_MICRO_MODEL_ID),
            ("confidence", bc.NOVA_MICRO_MODEL_ID),
            ("PRO", bc.NOVA_PRO_MODEL_ID),  # case-insensitive
        ],
    )
    def test_roles_and_tiers_resolve(self, alias: str, expected: str) -> None:
        assert bc.resolve_model_id(alias) == expected

    def test_concrete_model_id_passes_through(self) -> None:
        assert bc.resolve_model_id("amazon.nova-pro-v1:0") == "amazon.nova-pro-v1:0"

    def test_unknown_role_raises(self) -> None:
        with pytest.raises(ValueError):
            bc.resolve_model_id("gpt-9000")

    def test_three_nova_tiers_are_registered(self) -> None:
        # R8.1/R8.2: exactly Pro, Lite, Micro are the supported tiers.
        assert set(bc.NOVA_MODELS) == {"pro", "lite", "micro"}


# --- converse: routing + low-temperature default -------------------------


class TestConverse:
    def test_defaults_to_nova_pro(self) -> None:
        # R8.1: planning/reasoning default routes to Nova Pro.
        fake = FakeBedrockRuntime(["hello"])
        client = BedrockClient(client=fake)
        out = client.converse("hi")
        assert out == "hello"
        assert fake.calls[0]["modelId"] == bc.NOVA_PRO_MODEL_ID

    def test_low_temperature_is_the_default(self) -> None:
        # design.md: low temperature for determinism.
        fake = FakeBedrockRuntime(["ok"])
        client = BedrockClient(client=fake)
        client.converse("hi")
        assert fake.calls[0]["inferenceConfig"]["temperature"] == 0.0

    def test_summarize_routes_to_nova_lite(self) -> None:
        # R8.2: changelog summarization uses Nova Lite.
        fake = FakeBedrockRuntime(["a summary"])
        client = BedrockClient(client=fake)
        out = client.summarize("summarize these notes")
        assert out == "a summary"
        assert fake.calls[0]["modelId"] == bc.NOVA_LITE_MODEL_ID

    def test_classify_routes_to_nova_micro(self) -> None:
        # R8.2: confidence classification uses Nova Micro.
        fake = FakeBedrockRuntime(["low"])
        client = BedrockClient(client=fake)
        client.converse("classify", model="classify")
        assert fake.calls[0]["modelId"] == bc.NOVA_MICRO_MODEL_ID

    def test_temperature_override_is_respected(self) -> None:
        fake = FakeBedrockRuntime(["ok"])
        client = BedrockClient(client=fake)
        client.converse("hi", temperature=0.7)
        assert fake.calls[0]["inferenceConfig"]["temperature"] == 0.7

    def test_system_prompt_is_forwarded(self) -> None:
        fake = FakeBedrockRuntime(["ok"])
        client = BedrockClient(client=fake)
        client.converse("hi", system="be terse")
        assert fake.calls[0]["system"] == [{"text": "be terse"}]

    def test_no_system_key_when_absent(self) -> None:
        fake = FakeBedrockRuntime(["ok"])
        client = BedrockClient(client=fake)
        client.converse("hi")
        assert "system" not in fake.calls[0]

    def test_multi_block_text_is_concatenated(self) -> None:
        fake = FakeBedrockRuntime()
        # Override converse to return a multi-part content list.
        fake.converse = lambda **kw: {  # noqa: ANN003
            "output": {"message": {"content": [{"text": "foo"}, {"text": "bar"}]}}
        }
        client = BedrockClient(client=fake)
        assert client.converse("hi") == "foobar"


# --- converse_json: structured output + retry (R8.3) ---------------------


class TestConverseJson:
    def test_parses_clean_json_object(self) -> None:
        fake = FakeBedrockRuntime(['{"confidence": "high", "strategy": "auto_fix"}'])
        client = BedrockClient(client=fake)
        result = client.converse_json("plan this")
        assert result == {"confidence": "high", "strategy": "auto_fix"}
        # Only one call needed when the first reply is valid JSON.
        assert len(fake.calls) == 1

    def test_json_defaults_to_nova_pro(self) -> None:
        # R8.1: code-related reasoning (planning) uses Nova Pro by default.
        fake = FakeBedrockRuntime(['{"ok": true}'])
        client = BedrockClient(client=fake)
        client.converse_json("plan")
        assert fake.calls[0]["modelId"] == bc.NOVA_PRO_MODEL_ID

    def test_json_system_instruction_is_included(self) -> None:
        fake = FakeBedrockRuntime(['{"ok": true}'])
        client = BedrockClient(client=fake)
        client.converse_json("plan")
        sys_text = fake.calls[0]["system"][0]["text"]
        assert "JSON" in sys_text

    def test_extracts_json_from_code_fence(self) -> None:
        reply = 'Here you go:\n```json\n{"risk": "low"}\n```\nThanks!'
        fake = FakeBedrockRuntime([reply])
        client = BedrockClient(client=fake)
        assert client.converse_json("plan") == {"risk": "low"}

    def test_extracts_json_embedded_in_prose(self) -> None:
        reply = 'The plan is {"strategy": "guided_pr"} and that is final.'
        fake = FakeBedrockRuntime([reply])
        client = BedrockClient(client=fake)
        assert client.converse_json("plan") == {"strategy": "guided_pr"}

    def test_ignores_braces_inside_strings(self) -> None:
        reply = '{"note": "use {curly} braces", "n": 1}'
        fake = FakeBedrockRuntime([reply])
        client = BedrockClient(client=fake)
        assert client.converse_json("plan") == {"note": "use {curly} braces", "n": 1}

    def test_retries_once_then_succeeds(self) -> None:
        # First reply is non-JSON; the stricter retry returns valid JSON.
        fake = FakeBedrockRuntime(["sorry, no json here", '{"confidence": "low"}'])
        client = BedrockClient(client=fake)
        result = client.converse_json("plan")
        assert result == {"confidence": "low"}
        assert len(fake.calls) == 2
        # The retry carries the stricter instruction.
        assert "not valid JSON" in fake.calls[1]["system"][0]["text"]

    def test_raises_after_exhausting_retries(self) -> None:
        # Both attempts fail to produce JSON → hard failure the planner handles.
        fake = FakeBedrockRuntime(["nope", "still nope"])
        client = BedrockClient(client=fake)
        with pytest.raises(JSONResponseError) as excinfo:
            client.converse_json("plan")
        # The last raw reply is attached for logging / the low-confidence path.
        assert excinfo.value.raw == "still nope"
        assert len(fake.calls) == 2

    def test_retries_zero_means_single_attempt(self) -> None:
        fake = FakeBedrockRuntime(["not json"])
        client = BedrockClient(client=fake)
        with pytest.raises(JSONResponseError):
            client.converse_json("plan", retries=0)
        assert len(fake.calls) == 1

    def test_negative_retries_rejected(self) -> None:
        client = BedrockClient(client=FakeBedrockRuntime())
        with pytest.raises(ValueError):
            client.converse_json("plan", retries=-1)


# --- No AWS/boto3 dependency for injected client -------------------------


class TestNoAwsDependency:
    def test_injected_client_never_touches_boto3(self) -> None:
        # With an injected fake, ``.client`` returns it without importing boto3.
        fake = FakeBedrockRuntime(["ok"])
        client = BedrockClient(client=fake)
        assert client.client is fake
        client.converse("hi")  # exercises the path end-to-end, still no boto3
