"""Bedrock/Nova client: a thin wrapper over ``bedrock-runtime.converse``.

This module is the single boundary between the pipeline and Amazon Bedrock. It
wraps the ``bedrock-runtime`` ``converse`` API for the three Amazon Nova models
the build uses — **Nova Pro** for migration planning/reasoning (R8.1), **Nova
Lite** for changelog summarization, and **Nova Micro** for confidence
classification (R8.2) — behind one small, injectable interface.

Two design decisions are baked in here:

- **Low temperature by default.** Every call defaults to a near-zero
  temperature so the reasoning is as deterministic as a sampled model allows.
  Planning and code reasoning want repeatability, not creativity.
- **Structured (JSON) output for code reasoning (R8.3).** Callers that need a
  machine-parseable result (the planner especially) use :meth:`BedrockClient.converse_json`,
  which instructs the model to emit JSON, extracts the JSON payload from the
  reply, and parses it deterministically. Because models occasionally wrap JSON
  in prose or a ```` ```json ```` fence, extraction is tolerant, and a single
  stricter retry is attempted before giving up — mirroring design.md's "Model
  returns non-JSON | Retry once with stricter prompt; else ``low`` confidence"
  contract. This module raises :class:`JSONResponseError` on a hard failure;
  the *planner* (task 6) owns the "else ``low`` confidence" fallback.

Scope note: validating model-*generated code* by AST parsing (R8.4) belongs to
the refactor engine (task 7), not here. This module only produces text/JSON.

Design references:
- design.md → Models layer: "``bedrock_client.py`` wraps
  ``bedrock-runtime.converse`` for Nova Pro/Lite/Micro. Planner requests JSON
  output; low temperature for determinism."
- design.md → Error handling: "Model returns non-JSON | Retry once with
  stricter prompt; else ``low`` confidence".
- requirements.md → R8.1 (Nova Pro for planning/reasoning), R8.2 (MAY use Nova
  Lite for summarization / Nova Micro for classification), R8.3 (request
  structured/JSON output for code-related reasoning that the pipeline can parse
  deterministically).
"""

from __future__ import annotations

import json
import re
from typing import Any


# --- Nova model identifiers (Bedrock model IDs) ---------------------------

# Bedrock model IDs for the three Nova tiers this build uses. Kept as constants
# (and mapped by role below) so callers ask for a *role* — "planner", "summary",
# "classify" — rather than hardcoding a model string at every call site.
NOVA_PRO_MODEL_ID = "amazon.nova-pro-v1:0"
NOVA_LITE_MODEL_ID = "amazon.nova-lite-v1:0"
NOVA_MICRO_MODEL_ID = "amazon.nova-micro-v1:0"

# Logical roles → model IDs. R8.1 pins planning/reasoning to Nova Pro; R8.2 lets
# summarization use Nova Lite and classification use Nova Micro.
NOVA_MODELS: dict[str, str] = {
    "pro": NOVA_PRO_MODEL_ID,
    "lite": NOVA_LITE_MODEL_ID,
    "micro": NOVA_MICRO_MODEL_ID,
}

# Task-oriented aliases so callers can express intent instead of a tier.
_ROLE_ALIASES: dict[str, str] = {
    "planner": "pro",
    "plan": "pro",
    "reasoning": "pro",
    "summary": "lite",
    "summarize": "lite",
    "changelog": "lite",
    "classify": "micro",
    "classification": "micro",
    "confidence": "micro",
}

# Determinism defaults. A near-zero temperature keeps planning/code reasoning
# as repeatable as a sampled model allows (design.md → "low temperature for
# determinism"). ``top_p`` is pinned high (nucleus disabled) for the same goal.
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_MAX_TOKENS = 2048

# The instruction appended when a caller requests JSON output (R8.3). Kept
# explicit so the "parse deterministically" contract is visible at the seam.
_JSON_SYSTEM_INSTRUCTION = (
    "You are a precise assistant. Respond with a single valid JSON object and "
    "nothing else. Do not include explanations, prose, or markdown code fences."
)

# A stricter reminder used on the one allowed retry when the first reply did not
# contain parseable JSON (design.md → "Retry once with stricter prompt").
_JSON_RETRY_INSTRUCTION = (
    "Your previous response was not valid JSON. Reply with ONLY a single valid "
    "JSON object. No markdown, no code fences, no commentary."
)

# Extracts a JSON object/array from a reply, tolerating a ```` ```json ```` (or
# bare ```` ``` ````) fence the model may wrap it in.
_JSON_FENCE_RE = re.compile(
    r"```(?:json)?\s*(?P<body>[\[{].*?[\]}])\s*```",
    re.DOTALL | re.IGNORECASE,
)


class BedrockClientError(RuntimeError):
    """Base error for the Bedrock client wrapper."""


class JSONResponseError(BedrockClientError):
    """Raised when a model reply cannot be parsed as JSON after the retry.

    Carries the last raw reply on :attr:`raw` so callers (the planner, task 6)
    can log it while applying their own "else ``low`` confidence" fallback.
    """

    def __init__(self, message: str, *, raw: str | None = None) -> None:
        super().__init__(message)
        self.raw = raw


def resolve_model_id(model: str) -> str:
    """Resolve a role/alias/tier/explicit id to a concrete Bedrock model id.

    Accepts, case-insensitively: a tier (``"pro"``/``"lite"``/``"micro"``), a
    task alias (``"planner"``, ``"summary"``, ``"classify"``, …), or an already
    concrete Bedrock model id (returned unchanged).

    Raises:
        ValueError: If ``model`` is neither a known role nor a plausible model
            id.
    """
    key = model.strip().lower()
    if key in NOVA_MODELS:
        return NOVA_MODELS[key]
    if key in _ROLE_ALIASES:
        return NOVA_MODELS[_ROLE_ALIASES[key]]
    # Allow passing a concrete model id straight through (e.g. a future Nova
    # revision) without forcing it into the role table.
    if "." in model or ":" in model:
        return model
    raise ValueError(
        f"unknown model '{model}'; expected one of {sorted(NOVA_MODELS)}, an "
        f"alias {sorted(_ROLE_ALIASES)}, or a concrete Bedrock model id"
    )


def _extract_text(response: dict[str, Any]) -> str:
    """Pull the assistant's text out of a ``converse`` response.

    The ``converse`` API returns
    ``{"output": {"message": {"content": [{"text": ...}, ...]}}}``. We
    concatenate every ``text`` block so multi-part replies aren't truncated.
    """
    output = response.get("output") if isinstance(response, dict) else None
    message = output.get("message") if isinstance(output, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict):
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts).strip()


def _extract_json(text: str) -> Any:
    """Extract and parse a JSON value from a (possibly noisy) model reply.

    Tolerant in three escalating steps so a model that wraps JSON in a fence or
    a sentence still parses deterministically:

    1. Try the whole reply as JSON.
    2. Try the contents of a ```` ```json ```` / ```` ``` ```` fence.
    3. Fall back to the first balanced ``{...}`` / ``[...]`` span in the reply.

    Raises:
        ValueError: If no parseable JSON can be found.
    """
    stripped = text.strip()
    if not stripped:
        raise ValueError("empty model response")

    # 1) Whole reply is JSON.
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # 2) JSON inside a fenced code block.
    fence = _JSON_FENCE_RE.search(stripped)
    if fence:
        try:
            return json.loads(fence.group("body"))
        except json.JSONDecodeError:
            pass

    # 3) First balanced object/array span anywhere in the text.
    span = _first_json_span(stripped)
    if span is not None:
        try:
            return json.loads(span)
        except json.JSONDecodeError:
            pass

    raise ValueError("no parseable JSON object found in model response")


def _first_json_span(text: str) -> str | None:
    """Return the first balanced ``{...}`` or ``[...]`` substring, or ``None``.

    Scans for the earliest opening bracket and walks forward tracking depth,
    respecting string literals and escapes so braces inside strings don't throw
    off the balance count.
    """
    start = None
    opener = None
    for i, ch in enumerate(text):
        if ch in "{[":
            start = i
            opener = ch
            break
    if start is None:
        return None
    closer = "}" if opener == "{" else "]"
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


class BedrockClient:
    """A thin, injectable wrapper over ``bedrock-runtime.converse`` for Nova.

    One instance can talk to any Nova tier; the target model is chosen per call
    by role (``"planner"``/``"summary"``/``"classify"``) or tier
    (``"pro"``/``"lite"``/``"micro"``). All calls default to a low temperature
    for determinism (design.md), and :meth:`converse_json` adds the request/parse
    of structured JSON output for code-related reasoning (R8.3).

    The underlying ``boto3`` ``bedrock-runtime`` client is **injected** so the
    wrapper is unit-testable without AWS: tests pass a fake exposing a
    ``converse(**kwargs)`` method. When no client is provided a real one is
    created lazily via ``boto3.client("bedrock-runtime")`` on first use.
    """

    def __init__(
        self,
        *,
        client: Any | None = None,
        region_name: str | None = None,
        default_temperature: float = DEFAULT_TEMPERATURE,
        default_max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        """Build a client.

        Args:
            client: A pre-built ``bedrock-runtime`` client (or any object with a
                compatible ``converse`` method). Primarily for tests/DI. When
                omitted, a real client is created lazily on first call.
            region_name: AWS region used only when lazily creating a real
                client.
            default_temperature: Sampling temperature applied when a call does
                not override it. Defaults to ``0.0`` for determinism.
            default_max_tokens: Default response token cap.
        """
        self._client = client
        self._region_name = region_name
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens

    # --- lazy real-client construction -----------------------------------

    @property
    def client(self) -> Any:
        """Return the underlying ``bedrock-runtime`` client, creating it lazily.

        Importing ``boto3`` is deferred to here so importing this module (and
        running unit tests with an injected fake) never requires ``boto3`` or
        AWS credentials. Real credentials are resolved by boto3's default chain
        (env/instance/role) — never read or stored by this module (R6.5).
        """
        if self._client is None:
            import boto3  # local import: only needed for real AWS calls

            self._client = boto3.client(
                "bedrock-runtime", region_name=self._region_name
            )
        return self._client

    # --- core converse call ----------------------------------------------

    def converse(
        self,
        prompt: str,
        *,
        model: str = "pro",
        system: str | None = None,
        temperature: float | None = None,
        top_p: float = DEFAULT_TOP_P,
        max_tokens: int | None = None,
    ) -> str:
        """Send a single-user-turn prompt to a Nova model and return its text.

        Args:
            prompt: The user message content.
            model: Role/alias/tier/id selecting the Nova model (see
                :func:`resolve_model_id`). Defaults to Nova Pro (R8.1).
            system: Optional system instruction.
            temperature: Sampling temperature; defaults to
                :attr:`default_temperature` (``0.0``) for determinism.
            top_p: Nucleus sampling parameter.
            max_tokens: Response token cap; defaults to
                :attr:`default_max_tokens`.

        Returns:
            The assistant's concatenated text reply (stripped).
        """
        model_id = resolve_model_id(model)
        temp = self.default_temperature if temperature is None else temperature
        tokens = self.default_max_tokens if max_tokens is None else max_tokens

        kwargs: dict[str, Any] = {
            "modelId": model_id,
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {
                "temperature": temp,
                "topP": top_p,
                "maxTokens": tokens,
            },
        }
        if system:
            kwargs["system"] = [{"text": system}]

        response = self.client.converse(**kwargs)
        return _extract_text(response)

    # --- structured JSON output (R8.3) ------------------------------------

    def converse_json(
        self,
        prompt: str,
        *,
        model: str = "pro",
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        retries: int = 1,
    ) -> Any:
        """Request JSON output and parse it deterministically (R8.3).

        Wraps :meth:`converse` with a JSON-only system instruction, then extracts
        and parses a JSON value from the reply. If the first reply is not
        parseable, retries up to ``retries`` more times with a stricter
        instruction (design.md → "Retry once with stricter prompt"). The default
        of ``retries=1`` yields exactly one initial attempt plus one stricter
        retry.

        Args:
            prompt: The user message content.
            model: Role/alias/tier/id selecting the Nova model. Defaults to Nova
                Pro for planning/reasoning (R8.1).
            system: Optional extra system instruction, combined with the JSON
                directive.
            temperature: Sampling temperature; defaults to ``0.0``.
            max_tokens: Response token cap.
            retries: Number of *additional* stricter attempts after the first.
                Must be ``>= 0``.

        Returns:
            The parsed JSON value (typically a ``dict`` for planning).

        Raises:
            JSONResponseError: If no attempt produced parseable JSON. The last
                raw reply is attached on ``.raw``; the planner (task 6) applies
                the "else ``low`` confidence" fallback.
        """
        if retries < 0:
            raise ValueError("retries must be >= 0")

        base_system = (
            f"{system}\n\n{_JSON_SYSTEM_INSTRUCTION}"
            if system
            else _JSON_SYSTEM_INSTRUCTION
        )

        last_raw = ""
        for attempt in range(retries + 1):
            sys_prompt = base_system
            if attempt > 0:
                # Stricter reminder on each retry.
                sys_prompt = f"{base_system}\n\n{_JSON_RETRY_INSTRUCTION}"
            raw = self.converse(
                prompt,
                model=model,
                system=sys_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            last_raw = raw
            try:
                return _extract_json(raw)
            except ValueError:
                continue

        raise JSONResponseError(
            f"model did not return parseable JSON after {retries + 1} attempt(s)",
            raw=last_raw,
        )

    # --- convenience: satisfy changelog_tools.ChangelogModel --------------

    def summarize(self, prompt: str) -> str:
        """Summarize via Nova Lite; satisfies ``changelog_tools.ChangelogModel``.

        The changelog fetcher (task 4) injects any object exposing
        ``summarize(prompt) -> str``. This method routes to Nova Lite (R8.2) so
        a :class:`BedrockClient` can be passed directly as that model.
        """
        return self.converse(prompt, model="lite")
