# Requirements — LusiScan

## Introduction

LusiScan is an autonomous AI agent that takes the repetitive, judgment-heavy
work out of dependency upgrades. It runs quietly in the background, detects
outdated Python packages in a target repository, analyzes migration impact,
applies safe automated refactors, validates them with tests, and surfaces to a
human **only** when there is a real decision to make. It is built with the
Strands Agents SDK, reasons with Amazon Nova models on Bedrock, and is deployed
on **Amazon Bedrock AgentCore Runtime** (a committed part of this build). A
Streamlit control panel provides a live demo link where a human reviews pending
migrations and approves, reviews, or ignores them.

This document defines the requirements for the hackathon-scoped build: **Python
only, two demo packages** (`requests` safe patch bump, `pydantic` 1→2
human-in-the-loop), against a controlled demo repository.

### Scope boundaries (explicit non-goals for this build)

- Non-Python ecosystems (Node, Cargo) are out of scope.
- General-purpose changelog discovery is out of scope; changelog/repo mapping is
  hardcoded for the two demo packages.
- General-purpose auto-refactor is out of scope; the refactor engine handles a
  small, pre-defined pattern registry only.
- Webhooks, multi-repo support, and Slack interactive buttons are out of scope
  (a plain PR comment or Slack message is sufficient).

---

## Requirements

### Requirement 1 — Detect outdated dependencies

**User Story:** As a developer, I want the agent to detect which of my Python
dependencies are outdated, so that I don't have to track releases manually.

#### Acceptance Criteria

1. WHEN the agent scans a target repository THEN LusiScan SHALL read the
   declared dependencies from the repository's manifest (`requirements.txt` or
   `pyproject.toml`) without installing them.
2. WHEN determining the latest available version of a package THEN LusiScan
   SHALL query the package registry (PyPI) rather than the local environment.
3. WHEN a declared package has a newer version available THEN LusiScan SHALL
   record it as an outdated package with its name, current version, and latest
   version.
4. IF a manifest cannot be parsed THEN LusiScan SHALL record an error for that
   repository and continue without crashing.

### Requirement 2 — Analyze migration impact

**User Story:** As a developer, I want the agent to understand what changed
between versions, so that it can decide whether an upgrade is safe.

#### Acceptance Criteria

1. WHEN an outdated package is one of the supported demo packages THEN LusiScan
   SHALL fetch its release notes / changelog for the version range being
   upgraded.
2. WHEN a changelog is available THEN LusiScan SHALL summarize it, focusing on
   breaking changes, deprecations, and migration steps.
3. WHEN analyzing an upgrade THEN LusiScan SHALL produce a structured migration
   plan containing at minimum: a confidence level (`high` | `low`), a migration
   strategy (`auto_fix` | `guided_pr` | `human_required`), an estimated risk
   level, and a list of identified breaking changes.
4. IF a changelog cannot be fetched THEN LusiScan SHALL classify the migration
   as `low` confidence and require human review.

### Requirement 3 — Apply scoped, safe refactors

**User Story:** As a developer, I want the agent to apply the mechanical parts
of a migration automatically, so that I only handle the parts that need
judgment.

#### Acceptance Criteria

1. WHEN a migration plan marks a change as auto-fixable THEN LusiScan SHALL
   apply it using an AST-aware transformation (`libcst`), not a blind string
   replacement.
2. WHEN applying a version bump THEN LusiScan SHALL update the pinned version in
   the manifest.
3. WHEN a change is not auto-fixable THEN LusiScan SHALL leave the code
   unchanged for that change and record it as a flagged breaking change for
   human review.
4. WHEN any code transformation is applied THEN the resulting file SHALL remain
   syntactically valid Python (verified by parsing) before it is committed.
5. IF an auto-fix cannot be applied safely THEN LusiScan SHALL fall back to
   `guided_pr` (version bump plus migration notes) rather than committing a
   broken change.

### Requirement 4 — Validate changes with tests

**User Story:** As a developer, I want the agent to run the test suite before
asking me to approve anything, so that I trust its output.

#### Acceptance Criteria

1. WHEN the agent has applied changes on a branch THEN LusiScan SHALL trigger
   the target repository's GitHub Actions workflow on that branch.
2. WHEN reading CI results THEN LusiScan SHALL read the GitHub Actions workflow
   run status/conclusion via the Actions API (NOT the legacy commit-status API).
3. WHILE a workflow run has not completed THEN LusiScan SHALL treat the result
   as pending and continue polling up to a configured timeout.
4. WHEN a workflow run completes THEN LusiScan SHALL classify the result as
   passed (conclusion `success`) or failed (any other conclusion).
5. IF CI does not complete within the timeout THEN LusiScan SHALL report a
   timeout and escalate to human review.
6. IF the GitHub Actions round-trip is unavailable THEN LusiScan SHALL be able
   to run the repository's tests locally in its sandbox and return an equivalent
   pass/fail result.

### Requirement 5 — Human-in-the-loop decisions

**User Story:** As a developer, I want the agent to surface a clear decision
only when needed, so that it saves me time instead of adding noise.

#### Acceptance Criteria

1. WHEN a migration is high confidence and tests pass THEN LusiScan SHALL open a
   pull request and notify the human that it is ready for a quick approval.
2. WHEN a migration is low confidence THEN LusiScan SHALL open a guided pull
   request containing the version bump, the applied auto-fixes, and the flagged
   breaking changes, and request human judgment.
3. WHEN a migration requires architectural decisions THEN LusiScan SHALL notify
   the human without modifying code.
4. LusiScan SHALL NOT merge any pull request without an explicit human approval.
5. WHEN a human records a decision (approve / review / ignore) THEN LusiScan
   SHALL act on it (merge, leave open, or close/skip) on its next cycle.

### Requirement 6 — Autonomous background operation on AgentCore

**User Story:** As a developer, I want the agent to run on its own in the
background, so that it is not another app I have to open and manage.

#### Acceptance Criteria

1. LusiScan SHALL be deployed to Amazon Bedrock AgentCore Runtime as the agent
   host. (Committed requirement for this build.)
2. WHEN deployed THEN LusiScan SHALL expose an entrypoint that accepts an
   invocation payload identifying the target repository and runs the full
   migration pipeline.
3. WHEN the agent runs THEN it SHALL persist its state (pending migrations,
   plans, diffs, test results, decisions) to durable storage (DynamoDB).
4. LusiScan SHALL be invocable on a schedule (e.g., EventBridge Scheduler) so it
   operates without a human initiating each run.
5. Secrets (GitHub token, Slack webhook) SHALL be read at runtime from AWS
   Secrets Manager and SHALL NOT be committed to the repository or baked into
   the container image.

### Requirement 7 — Live control panel (Streamlit)

**User Story:** As a developer (and as a hackathon judge), I want a live link
where I can see pending migrations and make decisions, so that the agent's work
and the human decision moment are visible.

#### Acceptance Criteria

1. LusiScan SHALL provide a Streamlit application that reads agent state from
   DynamoDB and lists pending migrations.
2. WHEN displaying a pending migration THEN the app SHALL show the package,
   version change, confidence, risk, reasoning, the code diff, the test summary,
   and a link to the pull request.
3. WHEN a human selects Approve, Review, or Ignore THEN the app SHALL record
   that decision to durable storage for the agent to act on.
4. The Streamlit app SHALL be the viewer/control panel only; the agent loop
   SHALL run on AgentCore Runtime, not inside the Streamlit process.
5. The Streamlit app SHALL be deployable to a public URL to serve as the live
   demo link, using read-mostly credentials scoped to the state table.

### Requirement 8 — Reasoning with Amazon Nova on Bedrock

**User Story:** As the builder, I want the agent to use Amazon Nova models, so
that the reasoning is fast, cost-effective, and native to the AWS ecosystem.

#### Acceptance Criteria

1. LusiScan SHALL use Amazon Nova Pro (via Bedrock) for migration planning and
   reasoning.
2. LusiScan MAY use Amazon Nova Lite for changelog summarization and Nova Micro
   for confidence classification.
3. WHEN invoking a model for code-related reasoning THEN LusiScan SHALL request
   structured output (e.g., JSON) that the pipeline can parse deterministically.
4. WHEN a model returns generated code THEN LusiScan SHALL validate it by AST
   parsing before using it.

### Requirement 9 — Submission readiness

**User Story:** As the builder, I want the project to meet the hackathon
submission requirements, so that it is eligible and scores well.

#### Acceptance Criteria

1. The repository SHALL include an MIT or Apache 2.0 license visible in the
   repository.
2. The repository SHALL include a README with setup and run instructions.
3. The project SHALL include an architecture diagram.
4. The project SHALL demonstrate both demo scenarios end-to-end (the safe
   auto-fix and the human-in-the-loop major upgrade).
5. The project SHALL provide a live demo link (the deployed Streamlit control
   panel).
