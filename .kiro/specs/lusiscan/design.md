# Design — LusiScan

## Overview

LusiScan is a Strands-based agent pipeline that detects outdated Python
dependencies, plans and applies scoped migrations, validates them with tests,
and surfaces decisions to a human. The agent loop is deployed on **Amazon
Bedrock AgentCore Runtime**; reasoning runs on **Amazon Nova** models via
Bedrock; state lives in **DynamoDB**; and a **Streamlit** app provides a live
control panel over that state.

This design deliberately scopes to a hackathon-winnable slice: **Python only**,
**two demo packages**, against a **controlled demo repo**. It also corrects
three API mistakes found in the original build doc:

1. **Strands API** — tools are plain functions decorated with `@tool`; agents are
   `Agent(model=, tools=[...])` invoked by calling them directly. There is no
   `Tool(...)` wrapper, no `.run()`, and no `AgentExecutor`.
2. **AgentCore ≠ Bedrock Agents** — we deploy to AgentCore *Runtime* (write the
   loop ourselves, wrap with `BedrockAgentCoreApp`), not the older
   `AWS::Bedrock::Agent` action-group product.
3. **CI status** — read GitHub Actions workflow-run `conclusion` via the Actions
   API, not the legacy commit-status API.

> Verification note: the Strands and AgentCore API shapes were confirmed against
> official docs but not executed in this environment. IAM wiring and the
> Scheduler→`InvokeAgentRuntime` target are the parts most likely to need
> real-environment adjustment during Day 6-7.

## Architecture

```
                        Amazon EventBridge Scheduler
                                   │ (invoke on schedule)
                                   ▼
        ┌──────────────────────────────────────────────────┐
        │        Bedrock AgentCore Runtime (host)           │
        │   BedrockAgentCoreApp  →  DepGuardOrchestrator     │
        │                                                    │
        │   Monitor → Planner → Executor → Validator         │
        │   (Strands @tool functions + Nova via Bedrock)     │
        └───────────────┬───────────────────┬───────────────┘
                        │                    │
              reads/writes state       GitHub REST + Actions API
                        │                    │
                        ▼                    ▼
                 ┌────────────┐        ┌───────────────┐
                 │  DynamoDB   │        │    GitHub     │
                 │  (state)    │        │ branch/PR/CI  │
                 └─────┬───────┘        └───────────────┘
                       │  reads state / writes decisions
                       ▼
                 ┌──────────────────────────┐
                 │  Streamlit control panel  │  ← live demo link
                 │  (viewer + approvals)     │
                 └──────────────────────────┘

  Secrets (GitHub token, Slack webhook) ← AWS Secrets Manager (read at runtime)
```

### Why this shape

- **Agent on AgentCore, UI separate.** Keeping the loop on AgentCore Runtime and
  Streamlit as a pure viewer preserves the "autonomous background agent" theme —
  the human's click is a recorded decision the agent picks up, not a synchronous
  web request that runs the agent.
- **State as the contract.** DynamoDB is the single integration point between the
  agent and the UI. Neither calls the other directly, which makes the Streamlit
  deploy (public, low-trust) safe with read-mostly credentials.

## Components and Interfaces

### Agent layer (Strands)

Each stage is a Strands `@tool` function; the orchestrator composes them. Tools
return structured dicts so the pipeline can act deterministically (we prefer
tool-returned data over parsing free-form model text).

```python
from strands import Agent, tool

@tool
def scan_packages(repo_path: str) -> list: ...
@tool
def fetch_changelog(package: str, current: str, target: str) -> str: ...
@tool
def plan_migration(package: str, changelog: str, code: str) -> dict: ...
@tool
def apply_migration(plan: dict) -> dict: ...
@tool
def validate_branch(branch: str) -> dict: ...
```

| Component | Responsibility | Key inputs → outputs |
|-----------|----------------|----------------------|
| `MonitorAgent` | Detect outdated packages | repo → `[{name, current, latest}]` |
| `PlannerAgent` (Nova Pro) | Decide strategy | changelog + code → migration plan (JSON) |
| `ExecutorAgent` | Apply refactor + open PR | plan → `{branch, pr_url, diff, changes}` |
| `ValidatorAgent` | Run + read tests | branch → `{status, details}` |
| `DepGuardOrchestrator` | Sequence the loop, persist state, honor decisions | repo → state writes |

### Tools layer

| Module | Responsibility | Notes |
|--------|----------------|-------|
| `package_tools.py` | Parse manifest, query PyPI for latest | No install; registry API, not `pip list` |
| `changelog_tools.py` | Fetch + summarize notes | Hardcoded package→repo map for the 2 demos |
| `refactor_tools.py` | AST-aware transforms (`libcst`) | Pattern registry only; validate AST after |
| `github_tools.py` | Branch, commit, PR, CI status | CI via **Actions API** `get_workflow_runs(...).conclusion` |
| `notify_tools.py` | Send human alert | Plain Slack webhook or PR comment |

### Models layer

`bedrock_client.py` wraps `bedrock-runtime.converse` for Nova Pro/Lite/Micro.
Planner requests JSON output; low temperature for determinism. Generated code is
AST-parsed before use.

### State layer (DynamoDB)

Single-table design keyed by `pk`/`sk`.

| Entity | pk | sk | Attributes |
|--------|----|----|-----------|
| Migration | `REPO#<repo>` | `MIG#<package>#<target>` | package, from, to, confidence, risk, strategy, reasoning, diff, pr_url, test_summary, status |
| Decision | `REPO#<repo>` | `DEC#<migration_id>` | decision (`approved`/`ignored`), timestamp |
| Run log | `REPO#<repo>` | `RUN#<ts>` | outcome, packages processed |

`status` transitions: `pending_review` → (`approved` | `ignored`) → `merged` |
`closed`. The Streamlit app writes decisions; the orchestrator reads them and
transitions status on its next cycle.

### AgentCore entrypoint

```python
# src/agentcore_app.py
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from src.main import DepGuardOrchestrator

app = BedrockAgentCoreApp()

@app.entrypoint
def invoke(payload):
    repo = payload.get("repo_name")
    if not repo:
        return {"error": "payload must include 'repo_name'"}
    DepGuardOrchestrator(repo_name=repo).run()
    return {"status": "completed", "repo": repo}

if __name__ == "__main__":
    app.run()   # local HTTP server on :8080 for testing before deploy
```

Deploy: `agentcore configure --entrypoint src/agentcore_app.py` →
`agentcore launch` → `agentcore invoke '{"repo_name": "..."}'`.

### Streamlit control panel

Reads pending migrations from DynamoDB, renders package/diff/plan/tests, and
records Approve/Review/Ignore decisions back to DynamoDB. No agent logic runs in
the app. Deployed to Streamlit Community Cloud with a read-mostly IAM user
(reads state, writes only the decision field).

## Data flow (happy path)

1. Scheduler invokes AgentCore entrypoint with `repo_name`.
2. Monitor parses manifest + queries PyPI → outdated packages.
3. Planner (Nova Pro) fetches changelog, returns structured plan.
4. Executor branches, applies scoped `libcst` fix (or version bump only), opens PR.
5. Validator triggers GitHub Actions, polls workflow-run `conclusion`.
6. Orchestrator writes a `pending_review` migration to DynamoDB and notifies.
7. Human opens Streamlit live link, reviews, clicks Approve → decision written.
8. Next agent cycle reads the decision, merges the PR, transitions status.

## Error handling

| Failure | Handling |
|---------|----------|
| Manifest unparseable | Record error, skip repo, don't crash (R1.4) |
| Changelog unavailable | Force `low` confidence → human review (R2.4) |
| Auto-fix unsafe / invalid AST | Fall back to `guided_pr`, never commit broken code (R3.5) |
| CI timeout | Report timeout, escalate to human (R4.5) |
| Actions API unavailable | Local sandbox `pytest` fallback, same result shape (R4.6) |
| Model returns non-JSON | Retry once with stricter prompt; else `low` confidence |

## Testing strategy

- **Unit:** manifest parsing, PyPI version compare, `libcst` transforms on
  fixture files, plan JSON parsing, CI-status mapping (`conclusion` → pass/fail).
- **Integration (controlled demo repo):** full loop for both demo packages;
  assert PR opened, correct `status` written, decision honored.
- **Do not** build tests for out-of-scope paths (Node, general changelog).
- Demo repo carries a tiny fast test suite so GitHub Actions completes in seconds.

## Security

- Secrets from Secrets Manager at runtime only; never committed or in the image.
- Streamlit uses a scoped read-mostly IAM user (state table only).
- AgentCore runtime role: `bedrock:InvokeModel` for Nova, Secrets Manager read,
  DynamoDB CRUD on the state table — least privilege.
- LusiScan never merges without explicit human approval (R5.4).

## Scope cuts (deferred)

Node/Cargo ecosystems, general changelog discovery, webhooks, multi-repo,
Slack interactive buttons. Documented as "future" so the demo stays consistent
with what is built.
