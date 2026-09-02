"""Migration planner (Nova Pro): turn a changelog into a structured plan.

This module implements the **PlannerAgent** stage of the pipeline. Given an
outdated package (with its version range) and the changelog summary produced by
the changelog fetcher (``changelog_tools.py``, task 4), it asks **Nova Pro**
(R8.1) to reason about the upgrade and return a structured, machine-parseable
migration plan (R8.3) that the rest of the loop can act on deterministically.

The plan shape is fixed by requirements.md → R2.3, which mandates at minimum:

- ``confidence``: ``high`` | ``low``
- ``strategy``: ``auto_fix`` | ``guided_pr`` | ``human_required``
- ``estimated_risk``: a risk level (``low`` | ``medium`` | ``high``)
- ``breaking_changes``: a list of identified breaking changes

Three behaviors are baked in here, mirroring design.md's error-handling table:

- **Short-circuit an already-``low`` changelog.** If the changelog step could
  not fetch/summarize notes it pins ``confidence == "low"`` and populates
  ``error`` (R2.4). In that case the planner does **not** call the model at all:
  there's nothing to reason about, so it returns a ``low`` /
  ``human_required`` plan directly (design.md → "Changelog unavailable | Force
  ``low`` confidence → human review").
- **Retry once on non-JSON, else ``low`` confidence.** The model call goes
  through :meth:`BedrockClient.converse_json` with ``retries=1``, which handles
  the single stricter retry (R8.3 / design.md → "Model returns non-JSON | Retry
  once with stricter prompt; else ``low`` confidence"). When even the retry
  fails, ``converse_json`` raises :class:`JSONResponseError`; the planner
  catches it and applies the "else ``low`` confidence" fallback (R2.4).
- **Validate + normalize the model's JSON.** Model output is untrusted: enum
  fields are coerced to the allowed values, ``breaking_changes`` is coerced to a
  list, and any missing/invalid field degrades the whole plan to a conservative
  ``low`` / ``human_required`` outcome rather than trusting a malformed plan.

The Nova Pro invocation is isolated behind an injected client interface
(:class:`PlannerModel`, satisfied by ``bedrock_client.BedrockClient``), so the
planner is fully unit-testable without AWS — mirroring the DI pattern in
``changelog_tools.py`` and ``bedrock_client.py``.

Design references:
- design.md → Agent layer: "``PlannerAgent`` (Nova Pro) | Decide strategy |
  changelog + code → migration plan (JSON)".
- design.md → Models layer: "Planner requests JSON output; low temperature for
  determinism."
- design.md → Error handling: "Changelog unavailable | Force ``low`` confidence
  → human review (R2.4)" and "Model returns non-JSON | Retry once with stricter
  prompt; else ``low`` confidence".
- requirements.md → R2.3 (structured migration plan shape), R2.4 (unavailable
  changelog → ``low`` confidence + human review), R8.1 (Nova Pro for planning),
  R8.3 (structured JSON output parsed deterministically).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from src.models.bedrock_client import JSONResponseError

# --- Allowed enum values (R2.3) -------------------------------------------

# The two confidence levels R2.3 mandates. Kept as constants so the "degrade to
# low" contract is expressed once and referenced everywhere.
CONFIDENCE_HIGH = "high"
CONFIDENCE_LOW = "low"
CONFIDENCE_LEVELS: frozenset[str] = frozenset({CONFIDENCE_HIGH, CONFIDENCE_LOW})

# The three migration strategies R2.3 mandates.
STRATEGY_AUTO_FIX = "auto_fix"
STRATEGY_GUIDED_PR = "guided_pr"
STRATEGY_HUMAN_REQUIRED = "human_required"
STRATEGIES: frozenset[str] = frozenset(
    {STRATEGY_AUTO_FIX, STRATEGY_GUIDED_PR, STRATEGY_HUMAN_REQUIRED}
)

# Risk levels the planner recognizes. R2.3 only requires "a risk level"; we
# constrain it to a small, ordered vocabulary for deterministic downstream use.
RISK_LOW = "low"
RISK_MEDIUM = "medium"
RISK_HIGH = "high"
RISK_LEVELS: frozenset[str] = frozenset({RISK_LOW, RISK_MEDIUM, RISK_HIGH})

# The conservative fallback applied whenever we can't trust a real plan: an
# unavailable changelog (R2.4), a hard non-JSON model failure, or model output
# that fails validation. Low confidence + human_required + high risk routes the
# migration to a human, which is always the safe default.
_FALLBACK_STRATEGY = STRATEGY_HUMAN_REQUIRED
_FALLBACK_RISK = RISK_HIGH


# --- Injectable Nova Pro client interface ---------------------------------


@runtime_checkable
class PlannerModel(Protocol):
    """Minimal interface for the model used to produce a migration plan.

    ``bedrock_client.BedrockClient`` satisfies this Protocol via its
    :meth:`converse_json` method. Injecting it (rather than importing a concrete
    client) keeps the Nova Pro invocation behind one seam and lets tests pass a
    fake that returns fixture JSON — no AWS/Bedrock dependency in unit tests.
    """

    def converse_json(
        self,
        prompt: str,
        *,
        model: str = ...,
        system: str | None = ...,
        temperature: float | None = ...,
        max_tokens: int | None = ...,
        retries: int = ...,
    ) -> Any:
        """Request JSON output from the model and return the parsed value."""
        ...


# --- Planner prompt -------------------------------------------------------

# The instruction handed to Nova Pro. Kept inline (mirroring changelog_tools'
# ``_SUMMARY_PROMPT_TEMPLATE``) so the plan contract is visible at the seam. The
# JSON-only directive itself is added by ``converse_json``; here we pin the
# *schema* and the allowed enum values so the model's output normalizes cleanly.
_PLAN_PROMPT_TEMPLATE = (
    "You are planning the migration of the Python package '{package}' from "
    "version {current} to {target}.\n\n"
    "Using the changelog summary below, decide how risky this upgrade is and "
    "how it should be handled. Respond with a JSON object with EXACTLY these "
    "keys:\n"
    '  - "confidence": one of "high" or "low"\n'
    '  - "strategy": one of "auto_fix", "guided_pr", or "human_required"\n'
    '  - "estimated_risk": one of "low", "medium", or "high"\n'
    '  - "breaking_changes": a JSON array of short strings, each naming one '
    "identified breaking change (use an empty array if there are none)\n"
    '  - "reasoning": a short string explaining the decision\n\n'
    "Guidance: choose \"auto_fix\" only for mechanical, low-risk changes (for "
    "example a safe patch bump with no breaking changes); choose \"guided_pr\" "
    "when a version bump plus migration notes are needed; choose "
    "\"human_required\" for changes needing architectural judgment. If there "
    "are breaking changes, confidence is usually \"low\".\n\n"
    "Changelog summary:\n{summary}"
)


def _build_plan_prompt(package: str, current: str, target: str, summary: str) -> str:
    """Render the Nova Pro planning prompt for a single upgrade (task 6.1)."""
    return _PLAN_PROMPT_TEMPLATE.format(
        package=package,
        current=current,
        target=target,
        summary=summary,
    )


# --- Plan normalization / validation --------------------------------------


def _coerce_breaking_changes(value: Any) -> list[str]:
    """Coerce the model's ``breaking_changes`` field into a list of strings.

    R2.3 requires a *list* of identified breaking changes. Models sometimes emit
    a single string, ``null``, or a list of non-strings; we normalize all of
    these to a clean ``list[str]`` (dropping empties) rather than trusting the
    raw shape.
    """
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, (list, tuple)):
        items: list[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                items.append(text)
        return items
    # Any other scalar: stringify it so no information is silently lost.
    text = str(value).strip()
    return [text] if text else []


def _fallback_plan(
    package: str,
    current: str,
    target: str,
    error_kind: str,
    message: str,
    *,
    breaking_changes: list[str] | None = None,
) -> dict[str, Any]:
    """Build the conservative ``low`` / ``human_required`` fallback plan.

    Applied whenever a trustworthy plan can't be produced — an already-``low``
    changelog (R2.4), a hard non-JSON model failure, or model output that fails
    validation. Always routes to a human, which is the safe default.
    """
    return {
        "package": package,
        "current": current,
        "target": target,
        "confidence": CONFIDENCE_LOW,
        "strategy": _FALLBACK_STRATEGY,
        "estimated_risk": _FALLBACK_RISK,
        "breaking_changes": list(breaking_changes or []),
        "reasoning": message,
        "error": {"kind": error_kind, "message": message},
    }


def _normalize_plan(
    raw: Any,
    package: str,
    current: str,
    target: str,
) -> dict[str, Any]:
    """Validate + normalize a model-produced plan into the R2.3 shape.

    The model's JSON is untrusted. This enforces the allowed enum values for
    ``confidence``/``strategy``/``estimated_risk`` and coerces
    ``breaking_changes`` to a list. If the payload isn't a JSON object, or any
    required enum field is missing/invalid, the whole plan degrades to the
    conservative ``low`` / ``human_required`` fallback rather than passing a
    malformed plan downstream.

    Returns:
        A normalized plan dict with ``error`` set to ``None`` on success, or the
        fallback plan (with a populated ``error``) when validation fails.
    """
    if not isinstance(raw, dict):
        return _fallback_plan(
            package,
            current,
            target,
            "invalid_plan",
            f"model returned a non-object plan of type {type(raw).__name__}",
        )

    breaking_changes = _coerce_breaking_changes(raw.get("breaking_changes"))

    confidence = raw.get("confidence")
    strategy = raw.get("strategy")
    risk = raw.get("estimated_risk")

    confidence = confidence.strip().lower() if isinstance(confidence, str) else None
    strategy = strategy.strip().lower() if isinstance(strategy, str) else None
    risk = risk.strip().lower() if isinstance(risk, str) else None

    # Any missing/invalid required field → degrade to the safe fallback, but
    # preserve whatever breaking changes the model did manage to identify so a
    # human isn't flying blind.
    invalid: list[str] = []
    if confidence not in CONFIDENCE_LEVELS:
        invalid.append("confidence")
    if strategy not in STRATEGIES:
        invalid.append("strategy")
    if risk not in RISK_LEVELS:
        invalid.append("estimated_risk")

    if invalid:
        return _fallback_plan(
            package,
            current,
            target,
            "invalid_plan",
            f"model plan had missing/invalid field(s): {', '.join(invalid)}",
            breaking_changes=breaking_changes,
        )

    reasoning = raw.get("reasoning")
    reasoning = reasoning.strip() if isinstance(reasoning, str) else ""

    return {
        "package": package,
        "current": current,
        "target": target,
        "confidence": confidence,
        "strategy": strategy,
        "estimated_risk": risk,
        "breaking_changes": breaking_changes,
        "reasoning": reasoning,
        "error": None,
    }


# --- Planner entrypoint (R2.3, R2.4, R8.1, R8.3) --------------------------


def plan_migration(
    changelog: dict[str, Any],
    *,
    model: PlannerModel,
    code: str | None = None,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    """Produce a structured migration plan from a changelog result (task 6).

    This is the single boundary the orchestrator calls for planning. It takes a
    :data:`changelog_tools.ChangelogResult` (which carries ``package`` /
    ``current`` / ``target`` and either a ``summary`` or an already-``low``
    ``confidence`` + ``error``) and returns a normalized plan matching R2.3.

    Behavior:

    - **Already-``low`` changelog → short-circuit (R2.4).** If the changelog
      step pinned ``confidence == "low"`` (unavailable / unmapped / fetch or
      summarize failure), the planner does not call the model and returns a
      ``low`` / ``human_required`` plan directly.
    - **Otherwise ask Nova Pro for JSON (R8.1, R8.3).** The call goes through
      ``converse_json(..., retries=1)``, which performs the single stricter
      retry on non-JSON output (design.md). The parsed plan is validated and
      normalized to the R2.3 shape.
    - **Non-JSON after retry → ``low`` confidence (R2.4).** If ``converse_json``
      raises :class:`JSONResponseError`, the planner catches it and returns the
      conservative fallback plan.

    Args:
        changelog: A ``ChangelogResult`` from
            :func:`changelog_tools.fetch_and_summarize_changelog`.
        model: An injected client satisfying :class:`PlannerModel` (the real
            Nova Pro wrapper, ``bedrock_client.BedrockClient``).
        code: Optional relevant source code to give the model more context.
            Appended to the prompt when provided.
        max_tokens: Optional response token cap forwarded to the model.

    Returns:
        A normalized migration plan dict with keys ``package``, ``current``,
        ``target``, ``confidence``, ``strategy``, ``estimated_risk``,
        ``breaking_changes``, ``reasoning``, and ``error`` (``None`` on success).
    """
    package = str(changelog.get("package", ""))
    current = str(changelog.get("current", ""))
    target = str(changelog.get("target", ""))

    # Short-circuit: an already-low changelog result has nothing to reason about
    # (R2.4). Route straight to a human without spending a model call.
    if changelog.get("confidence") == CONFIDENCE_LOW:
        cl_error = changelog.get("error")
        message = "changelog unavailable or low confidence; routing to human review"
        if isinstance(cl_error, dict) and cl_error.get("message"):
            message = f"{message}: {cl_error['message']}"
        return _fallback_plan(
            package, current, target, "changelog_low_confidence", message
        )

    summary = changelog.get("summary")
    summary_text = summary.strip() if isinstance(summary, str) else ""
    prompt = _build_plan_prompt(package, current, target, summary_text)
    if code:
        prompt = f"{prompt}\n\nRelevant code:\n{code}"

    # Ask Nova Pro for a JSON plan. ``converse_json(retries=1)`` owns the single
    # stricter retry on non-JSON output (R8.3 / design.md).
    try:
        raw_plan = model.converse_json(
            prompt, model="pro", retries=1, max_tokens=max_tokens
        )
    except JSONResponseError as exc:
        # Non-JSON even after the stricter retry → else "low" confidence (R2.4).
        return _fallback_plan(
            package,
            current,
            target,
            "non_json_response",
            f"model did not return parseable JSON: {exc}",
        )

    return _normalize_plan(raw_plan, package, current, target)
