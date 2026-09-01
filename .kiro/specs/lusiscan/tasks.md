# Implementation Plan — LusiScan

Incremental, test-driven coding tasks. Ordered to match the 13-day plan:
core loop first, **AgentCore deploy early** (right after the loop works), then
frontend + validation, then submission. Each task references the requirements
it satisfies. Only coding/config tasks are listed; recording the video and
filling the Devpost form are done outside this list.

- [x] 1. Scaffold the project skeleton
  - Create `pyproject.toml` with pinned deps: `strands-agents`, `bedrock-agentcore`, `bedrock-agentcore-starter-toolkit`, `boto3`, `libcst`, `PyGithub`, `requests`, `streamlit`, `pytest`.
  - Create the `src/` package layout (`agents/`, `tools/`, `models/`, `state/`, `prompts/`), `app/`, `tests/`, `infrastructure/`.
  - Add MIT `LICENSE` and a stub `README.md`.
  - _Requirements: 9.1, 9.2_

- [ ] 2. Prepare the controlled demo repo
  - Create a separate demo repo with a `pyproject.toml` pinning an outdated `requests` and `pydantic` 1.x, a tiny fast `pytest` suite, and a minimal GitHub Actions workflow that runs the tests.
  - _Requirements: 4.1, 9.4_

- [ ] 3. Implement the package monitor
  - [ ] 3.1 Parse declared dependencies from `requirements.txt` / `pyproject.toml` (no install).
    - _Requirements: 1.1_
  - [ ] 3.2 Query PyPI for the latest version and compute outdated packages `[{name, current, latest}]`.
    - _Requirements: 1.2, 1.3_
  - [ ] 3.3 Handle unparseable manifests by recording an error and continuing.
    - _Requirements: 1.4_
  - [ ] 3.4 Unit tests for parsing and version comparison.
    - _Requirements: 1.1, 1.2_

- [ ] 4. Implement the changelog fetcher
  - [ ] 4.1 Hardcode the package→GitHub-repo mapping for `requests` and `pydantic`; fetch release notes for the version range.
    - _Requirements: 2.1_
  - [ ] 4.2 Summarize notes (Nova Lite) focusing on breaking changes / deprecations / migration steps.
    - _Requirements: 2.2_
  - [ ] 4.3 On fetch failure, signal `low` confidence to the planner.
    - _Requirements: 2.4_

- [ ] 5. Implement the Bedrock/Nova client
  - Wrap `bedrock-runtime.converse` for Nova Pro/Lite/Micro; low temperature; request JSON output for planning.
  - _Requirements: 8.1, 8.2, 8.3_

- [ ] 6. Implement the migration planner (Nova Pro)
  - [ ] 6.1 Build the planner prompt and parse a structured plan (`confidence`, `strategy`, `estimated_risk`, `breaking_changes`).
    - _Requirements: 2.3_
  - [ ] 6.2 Retry once on non-JSON output; otherwise default to `low` confidence.
    - _Requirements: 2.4, 8.3_
  - [ ] 6.3 Unit test plan parsing against fixture model responses.
    - _Requirements: 2.3_

- [ ] 7. Implement the scoped refactor engine (`libcst`)
  - [ ] 7.1 Build the pattern registry for the demo packages (pydantic import move + rename; requests = version bump only).
    - _Requirements: 3.1, 3.2_
  - [ ] 7.2 Apply AST-aware transforms and re-parse to verify the file is valid Python before writing.
    - _Requirements: 3.1, 3.4_
  - [ ] 7.3 Leave non-auto-fixable changes untouched and record them as flagged; fall back to `guided_pr` if a fix can't be applied safely.
    - _Requirements: 3.3, 3.5_
  - [ ] 7.4 Validate any model-generated code by AST parsing before use.
    - _Requirements: 8.4_
  - [ ] 7.5 Unit tests: transforms on fixture files, invalid-AST rejection.
    - _Requirements: 3.1, 3.4_

- [ ] 8. Implement GitHub integration (branch / commit / PR)
  - [ ] 8.1 Create branch, commit changes, open PR via PyGithub.
    - _Requirements: 5.1, 5.2_
  - [ ] 8.2 Never merge without a recorded human approval; add merge-on-approval path.
    - _Requirements: 5.4, 5.5_

- [ ] 9. Implement the DynamoDB state store
  - [ ] 9.1 Single-table store: write/read Migration, Decision, Run-log entities with `status` transitions.
    - _Requirements: 6.3, 5.5_
  - [ ] 9.2 Unit test status transitions (`pending_review` → approved/ignored → merged/closed).
    - _Requirements: 5.5_

- [ ] 10. Wire the orchestrator (Strands agent loop)
  - [ ] 10.1 Define stages as `@tool` functions and compose Monitor→Planner→Executor→Validator; invoke agents by calling them directly.
    - _Requirements: 1.x, 2.x, 3.x_
  - [ ] 10.2 Persist migrations and honor recorded decisions on each cycle.
    - _Requirements: 5.5, 6.3_
  - [ ] 10.3 Integration test: full loop locally on both demo packages.
    - _Requirements: 9.4_

- [ ] 11. Deploy to AgentCore Runtime (do this as soon as task 10 works)
  - [ ] 11.1 Add `src/agentcore_app.py` (`BedrockAgentCoreApp` + `@app.entrypoint`); test locally on `:8080`.
    - _Requirements: 6.1, 6.2_
  - [ ] 11.2 `agentcore configure` + `agentcore launch`; grant the runtime role Bedrock, Secrets Manager, and DynamoDB access.
    - _Requirements: 6.1, 6.5_
  - [ ] 11.3 `agentcore invoke` both demo scenarios end-to-end on the live runtime; fix packaging/IAM issues.
    - _Requirements: 6.2, 9.4_
  - [ ] 11.4 Read secrets from Secrets Manager at runtime (no committed secrets).
    - _Requirements: 6.5_

- [ ] 12. Implement test validation (GitHub Actions)
  - [ ] 12.1 Trigger the demo repo's workflow on the temp branch; poll workflow-run status/`conclusion` via the Actions API.
    - _Requirements: 4.1, 4.2, 4.3, 4.4_
  - [ ] 12.2 Report timeout and escalate to human on non-completion.
    - _Requirements: 4.5_
  - [ ] 12.3 Add local-sandbox `pytest` fallback returning the same result shape.
    - _Requirements: 4.6_
  - [ ] 12.4 Unit test the `conclusion` → pass/fail mapping.
    - _Requirements: 4.4_

- [ ] 13. Implement the confidence gate + notifications
  - [ ] 13.1 High confidence + tests pass → auto-PR + "ready to approve" notice; low confidence → guided PR with flagged changes.
    - _Requirements: 5.1, 5.2_
  - [ ] 13.2 `human_required` → notify without touching code.
    - _Requirements: 5.3_
  - [ ] 13.3 Send a plain notification (Slack webhook or PR comment).
    - _Requirements: 5.1, 5.2_

- [ ] 14. Build the Streamlit control panel
  - [ ] 14.1 Read pending migrations from DynamoDB; render package, version change, confidence, risk, reasoning, diff, test summary, PR link.
    - _Requirements: 7.1, 7.2_
  - [ ] 14.2 Approve / Review / Ignore buttons that write a decision to DynamoDB (no agent logic in the app).
    - _Requirements: 7.3, 7.4_

- [ ] 15. Deploy the Streamlit live link
  - Deploy to Streamlit Community Cloud with a read-mostly scoped IAM user; verify the public URL drives the live agent state.
  - _Requirements: 7.5, 9.5_

- [ ] 16. Supporting infrastructure
  - SAM/CloudFormation for the DynamoDB state table + EventBridge Scheduler that invokes the AgentCore runtime on a schedule.
  - _Requirements: 6.3, 6.4_

- [ ] 17. Submission polish
  - Architecture diagram; finalize README (setup + run + demo steps); confirm MIT license visible; dry-run both scenarios through the live stack.
  - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_
