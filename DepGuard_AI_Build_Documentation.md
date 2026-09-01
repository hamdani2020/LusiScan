# AI DevSecOps & Dependency Migration Agent
## Complete Build Documentation -- Agents for Humans Hackathon

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Tech Stack](#3-tech-stack)
4. [Why Amazon Nova Models?](#4-why-amazon-nova-models)
5. [13-Day Sprint Plan](#5-13-day-sprint-plan-solo-demo-first)
6. [Repository Structure](#6-repository-structure)
7. [Core Implementation Guide](#7-core-implementation-guide)
8. [Strands Agents SDK Integration](#8-strands-agents-sdk-integration)
9. [Bedrock AgentCore Deployment](#9-bedrock-agentcore-deployment)
10. [GitHub Integration](#10-github-integration)
11. [The Refactor Engine (Scoped)](#11-the-refactor-engine-scoped)
12. [Testing & Validation Loop](#12-testing--validation-loop)
13. [Notification System](#13-notification-system)
14. [Demo Script](#14-demo-script)
15. [Submission Checklist](#15-submission-checklist)

---

## 1. Project Overview

**Project Name:** `LusiScan` (suggested -- pick your own)

**Track:** Professional Agents

**Problem:** Developers lose dozens of hours upgrading legacy dependencies, fixing breaking API changes, and resolving security vulnerabilities.

**Solution:** An autonomous AI agent that runs quietly in the background —
monitoring repositories, detecting outdated packages, analyzing migration
impact, applying safe refactors, and running tests — and surfaces to a human
*only* when there's a real judgment call to make. It's not another dashboard to
check; it does the mechanical 80% and pings you for the 20% that needs you.

**Why this fits the theme:** The brief asks for agents that take on repetitive,
judgment-heavy work and only interrupt when a decision is needed. Dependency
upgrades are exactly that for developers — mostly mechanical, occasionally
requiring real judgment. Professional Agents track.

**The Human Decision Moment:**
> "Upgraded `pydantic` 1.x to 2.x. Auto-fixed 3 broken imports. Tests passed. Review pull request and approve merge?"

---

## 2. Architecture

```
+---------------------------------------------------------------------+
|                        GitHub Repository                             |
|  +-------------+  +-------------+  +-----------------------------+  |
|  | package.json|  |  src/       |  |  .github/workflows/tests.yml|  |
|  | (or equiv)  |  |  (source)   |  |  (CI/CD tests)              |  |
|  +------+------+  +-------------+  +-----------------------------+  |
+---------+-----------------------------------------------------------+
          |
          v
+---------------------------------------------------------------------+
|                    LusiScan Agent (Strands SDK)                   |
|                                                                      |
|  +-----------------+    +-----------------+    +-----------------+  |
|  |  Monitor Agent  |--->|  Planner Agent  |--->|  Executor Agent |  |
|  |  (Detect drift) |    |  (Decide path)  |    |  (Apply fixes)  |  |
|  +-----------------+    +-----------------+    +-----------------+  |
|          |                      |                      |              |
|          v                      v                      v              |
|  +----------------------------------------------------------------+ |
|  |              Amazon Bedrock -- Nova Pro Model                     | |
|  |  (Reasoning, migration planning, code generation)                | |
|  +----------------------------------------------------------------+ |
|                                                                      |
|  +-----------------+    +-----------------+    +-----------------+  |
|  |  Test Runner    |--->|  Confidence     |--->|  Notifier       |  |
|  |  (Validate PR)  |    |  (High/Low?)    |    |  (Slack/GitHub) |  |
|  +-----------------+    +-----------------+    +-----------------+  |
+---------------------------------------------------------------------+
          |
          v
+---------------------------------------------------------------------+
|                     Human (The Decision Maker)                       |
|                                                                      |
|  +------------------------+    +--------------------------------+   |
|  |  Slack: "Approve?"     | or |  GitHub PR: "Review & Merge"   |   |
|  |  [Approve] [Review]    |    |  (with full diff + context)    |   |
|  +------------------------+    +--------------------------------+   |
+---------------------------------------------------------------------+
```

---

## 3. Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Agent Framework** | Strands Agents SDK | Core agent orchestration, tool calling, state management |
| **LLM** | Amazon Nova Pro (Bedrock) | Reasoning, migration planning, code generation |
| **LLM (Fast tasks)** | Amazon Nova Lite (Bedrock) | Changelog parsing, simple classification |
| **LLM (Classification)** | Amazon Nova Micro (Bedrock) | Confidence scoring, binary decisions |
| **Deployment** | AWS Bedrock AgentCore Runtime | Hosts the Strands agent loop as a managed, serverless runtime (boosts Technical Implementation score) |
| **Scheduling** | Amazon EventBridge Scheduler | Periodically invokes the AgentCore runtime; optional thin Lambda for webhook handling |
| **Storage** | Amazon DynamoDB | Agent state, migration history, package metadata |
| **Secrets** | AWS Secrets Manager | GitHub tokens, Slack webhooks |
| **Git Integration** | GitHub REST API + PyGithub | Repo scanning, PR creation, branch management |
| **Code Analysis** | `libcst` (Python) or `jscodeshift` (JS) | AST-aware structural refactoring |
| **Testing** | GitHub Actions (self-hosted or standard) | Test execution on temp branches |
| **Notifications** | Slack Incoming Webhooks + GitHub Issues/PRs | Human-in-the-loop alerts |
| **Language** | Python 3.11+ | Primary implementation language |

---

## 4. Why Amazon Nova Models?

**Your instinct is correct.** Using Amazon Nova models is a smart strategic choice for this hackathon:

### Advantages

| Advantage | Detail |
|-----------|--------|
| **No Access Request** | Nova Pro, Lite, and Micro are enabled by default in Bedrock. No model access request forms, no waiting for approval. You can start building immediately. |
| **Cost-Effective** | Nova Pro is significantly cheaper than Claude 3.5 Sonnet or GPT-4. For a hackathon with limited credits, this matters. |
| **Fast** | Nova models have low latency, which is critical for an agent that may make multiple LLM calls in a single workflow. |
| **AWS Native** | Using a first-party AWS model with Bedrock AgentCore shows deep ecosystem integration -- judges notice this. |
| **Capable for Code** | Nova Pro scores well on code generation benchmarks (HumanEval, MBPP). It can handle the migration planning and simple refactoring tasks you need. |

### Recommended Model Assignment

| Task | Model | Why |
|------|-------|-----|
| Migration planning & reasoning | **Nova Pro** | Complex reasoning, understanding breaking changes, generating code diffs |
| Changelog summarization | **Nova Lite** | Fast, cheap, perfect for text summarization |
| Confidence classification (High/Low) | **Nova Micro** | Ultra-fast, text-only classification task |

### Bedrock API Setup

```python
import boto3
from botocore.config import Config

# Nova Pro -- primary reasoning model
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1",  # or us-west-2
    config=Config(read_timeout=300)
)

def invoke_nova_pro(system_prompt: str, user_prompt: str):
    response = bedrock_runtime.converse(
        modelId="amazon.nova-pro-v1:0",
        messages=[
            {
                "role": "user",
                "content": [{"text": user_prompt}]
            }
        ],
        system=[{"text": system_prompt}],
        inferenceConfig={
            "maxTokens": 4096,
            "temperature": 0.2,  # Low temp for deterministic code
            "topP": 0.9
        }
    )
    return response["output"]["message"]["content"][0]["text"]
```

---

## 5. 13-Day Sprint Plan (Solo, Demo-First)

**Reality check:** With 13 days and three fragile external integrations
(GitHub, CI, AgentCore), the goal is a *bulletproof demo of a scoped slice*, not
a production tool. Everything below is ordered by judge-visible value. If a day
slips, cut from the bottom, not the top.

### Ruthless priority order (what actually earns points)

> **AgentCore is a committed requirement for this build**, not a fallback. The
> hackathon *rules* list it as optional (scores higher, not mandatory), but we
> are deploying on Bedrock AgentCore Runtime regardless — it's the core AWS
> integration story. It's scheduled early (right after the local loop works) so
> it is never the last risky thing standing before the deadline.

1. Core agent loop working end-to-end on **2 demo packages** — this *is* the project.
2. **AgentCore Runtime deploy** — wrap the loop, deploy, invoke. Do this as soon as the local loop works, not at the end.
3. Streamlit control panel + live demo link (reads AgentCore's state; scores higher on Technical Implementation).
4. GitHub Actions test validation on a **controlled demo repo** (simple workflow-run `conclusion` poll).
5. Demo video + README + architecture diagram + submission.

**Hard scope cuts (label as "future," do not build now):** Node/Cargo ecosystems,
general changelog discovery, webhooks, multi-repo support, Slack interactive
buttons (a plain Slack/GitHub message is enough).

### Phase 1: Core Loop Locally (Days 1-5)

| Day | Task | Deliverable |
|-----|------|-------------|
| **Day 1** | Set up AWS + Bedrock (Nova enabled), install Strands, create repo with MIT license, prepare **one controlled demo repo** with outdated `requests` + `pydantic` and a tiny fast test suite | Dev env + demo repo ready |
| **Day 2** | Package scanner (manifest parse + registry API for latest version) + changelog fetcher, both **hardcoded to the 2 demo packages** | Detects the 2 outdated packages, fetches their notes |
| **Day 3** | Nova Pro migration planner: changelog + code in, structured JSON plan out (`confidence`, `strategy`, `breaking_changes`) | Planner returns a valid structured plan |
| **Day 4** | Scoped refactor engine: `libcst` transforms for the exact pydantic import/rename patterns; `requests` needs no code change | Auto-fix works on the demo repo |
| **Day 5** | Wire Monitor→Planner→Executor as Strands `@tool` functions + orchestrator, state in DynamoDB, branch + PR via PyGithub | **Full loop runs locally end-to-end on both packages** |

### Phase 2: Get onto AgentCore Early (Days 6-8)

Deploy to AgentCore Runtime *as soon as the loop works locally* — while you
still have days to debug IAM, packaging, and invocation. Do not leave it to the
end.

| Day | Task | Deliverable |
|-----|------|-------------|
| **Day 6** | Wrap orchestrator in `BedrockAgentCoreApp` (`@app.entrypoint`), test locally on `:8080`, then `agentcore configure` + `agentcore launch` | Agent deployed to AgentCore Runtime |
| **Day 7** | `agentcore invoke` both demo scenarios end-to-end on the live runtime; fix IAM (Bedrock, Secrets Manager, DynamoDB) and packaging issues | Both scenarios run on the live runtime |
| **Day 8** | GitHub Actions validation: trigger workflow on temp branch, poll the **workflow run `conclusion`** (Actions API) + confidence gate (High = auto-PR, Low = guided PR) + plain notification | Agent reads CI pass/fail; two-tier human-in-the-loop working |

### Phase 3: Frontend + Live Link (Days 9-10)

| Day | Task | Deliverable |
|-----|------|-------------|
| **Day 9** | Streamlit control panel over the runtime's DynamoDB state: list pending migrations, show diff + plan, Approve / Review / Ignore | Working Streamlit app over live agent state |
| **Day 10** | Deploy Streamlit (Community Cloud) → **live demo link**; harden the happy path for the 2 scenarios end-to-end through the live stack | Public live link driving the live AgentCore agent |

### Phase 4: Demo + Submit (Days 11-13)

| Day | Task | Deliverable |
|-----|------|-------------|
| **Day 11** | Full dress-rehearsal run of both scenarios through the live stack; fix whatever breaks | Reliable, repeatable demo run |
| **Day 12** | Record demo video (≤5 min: problem → who → why → live demo), architecture diagram, README, MIT license visible | Video + diagram + docs done |
| **Day 13** | Final Devpost submission, AWS Builder ID, live link; (bonus) builder.aws.com post with `#AgentsforHumans` | Submitted with buffer to spare |

> **Buffer note:** there is intentionally little slack in this plan.
> **AgentCore is not a fallback candidate — it stays.** That's exactly why it's
> scheduled on Days 6-7 (right after the local loop) instead of at the end: if
> the deploy fights you, you have days to fix IAM/packaging rather than hours.
> The parts you *may* trim under time pressure are the softer ones: GitHub
> Actions validation can fall back to running tests locally in the sandbox
> (same `{"status": ...}` result), and Slack notifications can drop to a plain
> PR comment. Protect the AgentCore deploy and the live loop first.

---

## 6. Repository Structure

```
depguard-ai/
├── README.md                          # Setup instructions, architecture overview
├── LICENSE                            # MIT or Apache 2.0
├── architecture-diagram.png           # Visual architecture (required for submission)
├── pyproject.toml                     # Python dependencies
├── src/
│   ├── __init__.py
│   ├── agentcore_app.py               # BedrockAgentCoreApp entrypoint (deploy target)
│   ├── main.py                        # DepGuardOrchestrator (the agent loop)
│   ├── config.py                      # Environment variables, settings
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── monitor_agent.py           # Detects outdated packages
│   │   ├── planner_agent.py           # Decides migration strategy
│   │   ├── executor_agent.py          # Applies code changes
│   │   └── validator_agent.py         # Runs tests, checks results
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── github_tools.py            # GitHub API wrappers
│   │   ├── package_tools.py           # npm/pip/cargo scanners
│   │   ├── changelog_tools.py         # Release notes fetchers
│   │   ├── refactor_tools.py          # AST-aware code transformers
│   │   └── notify_tools.py            # Slack/GitHub notification senders
│   ├── models/
│   │   ├── __init__.py
│   │   └── bedrock_client.py          # Nova Pro/Lite/Micro wrappers
│   ├── state/
│   │   ├── __init__.py
│   │   └── dynamodb_store.py          # Agent state persistence
│   └── prompts/
│       ├── __init__.py
│       ├── migration_planner.txt      # System prompt for Nova Pro
│       ├── changelog_summarizer.txt   # System prompt for Nova Lite
│       └── confidence_classifier.txt  # System prompt for Nova Micro
├── tests/
│   ├── __init__.py
│   ├── test_monitor.py
│   ├── test_planner.py
│   └── test_refactor.py
├── app/
│   └── streamlit_app.py               # Live demo control panel (reads state, posts decisions)
├── infrastructure/
│   └── template.yaml                  # Supporting infra: DynamoDB, secrets, EventBridge Scheduler
├── .bedrock_agentcore.yaml            # Generated by `agentcore configure` (AgentCore Runtime config)
└── demo/
    ├── scenario-a-requests-patch/     # Demo: safe auto-upgrade (requests patch bump)
    └── scenario-b-pydantic-major/     # Demo: human-in-the-loop (pydantic 1 -> 2)
```

---

## 7. Core Implementation Guide

### 7.1 Package Monitor

```python
# src/tools/package_tools.py
import subprocess
import json
from typing import List, Dict

class PackageMonitor:
    """Detects outdated packages in a repository."""

    def scan_python_packages(self, repo_path: str) -> List[Dict]:
        """Scan for outdated Python packages using pip."""
        result = subprocess.run(
            ["pip", "list", "--outdated", "--format=json"],
            capture_output=True,
            text=True,
            cwd=repo_path
        )
        outdated = json.loads(result.stdout)

        # Filter for packages in requirements.txt / pyproject.toml
        # (Implement your filtering logic here)

        return [
            {
                "name": pkg["name"],
                "current_version": pkg["version"],
                "latest_version": pkg["latest_version"],
                "ecosystem": "python"
            }
            for pkg in outdated
        ]

    def scan_node_packages(self, repo_path: str) -> List[Dict]:
        """Scan for outdated Node packages using npm."""
        result = subprocess.run(
            ["npm", "outdated", "--json"],
            capture_output=True,
            text=True,
            cwd=repo_path
        )
        outdated = json.loads(result.stdout)

        return [
            {
                "name": name,
                "current_version": info["current"],
                "latest_version": info["latest"],
                "ecosystem": "node"
            }
            for name, info in outdated.items()
            if "current" in info
        ]
```

### 7.2 Changelog Fetcher

```python
# src/tools/changelog_tools.py
import requests
from typing import Optional, Dict

class ChangelogFetcher:
    """Fetches release notes and migration guides."""

    def __init__(self, github_token: str):
        self.headers = {
            "Authorization": f"token {github_token}",
            "Accept": "application/vnd.github.v3+json"
        }

    def fetch_github_releases(self, owner: str, repo: str) -> list:
        """Fetch recent releases from GitHub."""
        url = f"https://api.github.com/repos/{owner}/{repo}/releases"
        response = requests.get(url, headers=self.headers)
        return response.json()

    def get_migration_notes(self, package_name: str, 
                           current_ver: str, 
                           target_ver: str) -> str:
        """
        Fetch and summarize changelog between two versions.
        Uses Nova Lite for fast summarization.
        """
        # 1. Fetch raw changelog
        releases = self.fetch_github_releases("owner", package_name)

        # 2. Filter releases between current and target
        relevant = self._filter_releases(releases, current_ver, target_ver)

        # 3. Summarize with Nova Lite
        raw_notes = "

".join([r["body"] for r in relevant])
        summary = self._summarize_with_nova_lite(raw_notes)

        return summary

    def _summarize_with_nova_lite(self, raw_changelog: str) -> str:
        """Use Nova Lite for fast, cheap summarization."""
        # Call Bedrock with Nova Lite
        # Prompt: "Summarize these release notes. Focus on breaking changes,
        # deprecated features, and migration steps."
        pass
```

### 7.3 Migration Planner (Nova Pro)

```python
# src/agents/planner_agent.py
from src.models.bedrock_client import BedrockClient

MIGRATION_PLANNER_PROMPT = """You are a senior software engineer specializing in dependency migration.

Your task: Analyze the following package upgrade and determine the safest migration strategy.

## Input
- Package: {package_name}
- Current Version: {current_version}
- Target Version: {target_version}
- Ecosystem: {ecosystem}
- Changelog Summary: {changelog}
- Current Code Snippets: {code_snippets}

## Output Format (JSON)
{
    "confidence": "high" | "low",
    "reasoning": "Explain your confidence assessment...",
    "breaking_changes": [
        {
            "type": "function_rename" | "api_removal" | "behavior_change" | "import_change",
            "description": "What changed",
            "affected_files": ["file1.py", "file2.py"],
            "suggested_fix": "Specific code transformation needed"
        }
    ],
    "migration_strategy": "auto_fix" | "guided_pr" | "human_required",
    "estimated_risk": "low" | "medium" | "high"
}

Rules:
- "auto_fix": Only for minor/patch bumps or simple renames with no behavior changes
- "guided_pr": Major bump with clear but complex changes. Open PR with migration guide
- "human_required": Breaking changes that alter semantics or require architectural decisions
"""

class PlannerAgent:
    def __init__(self):
        self.bedrock = BedrockClient(model_id="amazon.nova-pro-v1:0")

    def plan_migration(self, package_info: dict, changelog: str, 
                      code_snippets: list) -> dict:
        """Generate a structured migration plan using Nova Pro."""

        prompt = MIGRATION_PLANNER_PROMPT.format(
            package_name=package_info["name"],
            current_version=package_info["current_version"],
            target_version=package_info["latest_version"],
            ecosystem=package_info["ecosystem"],
            changelog=changelog,
            code_snippets="

".join(code_snippets)
        )

        response = self.bedrock.generate(prompt)

        # Parse JSON response
        import json
        plan = json.loads(response)
        return plan
```

### 7.4 Scoped Refactor Engine

```python
# src/tools/refactor_tools.py
import libcst as cst
from typing import List, Dict

class PythonRefactorEngine:
    """
    AST-aware code refactoring for Python.
    NOT a general-purpose auto-refactor -- scoped to known patterns.
    """

    def __init__(self, repo_path: str):
        self.repo_path = repo_path

    def apply_rename_migration(self, file_path: str, 
                               old_name: str, 
                               new_name: str) -> bool:
        """
        Rename a function/class across a file.
        Example: pydantic.BaseConfig -> pydantic.ConfigDict
        """
        with open(file_path, "r") as f:
            source = f.read()

        module = cst.parse_module(source)

        # Create a transformer
        class RenameTransformer(cst.CSTTransformer):
            def __init__(self, old_name, new_name):
                self.old_name = old_name
                self.new_name = new_name

            def leave_Name(self, original_node, updated_node):
                if original_node.value == self.old_name:
                    return updated_node.with_changes(value=self.new_name)
                return updated_node

            def leave_Attribute(self, original_node, updated_node):
                if original_node.attr.value == self.old_name:
                    return updated_node.with_changes(
                        attr=cst.Name(value=self.new_name)
                    )
                return updated_node

        transformer = RenameTransformer(old_name, new_name)
        modified_module = module.visit(transformer)

        # Write back if changed
        if module.code != modified_module.code:
            with open(file_path, "w") as f:
                f.write(modified_module.code)
            return True
        return False

    def apply_import_migration(self, file_path: str,
                               old_import: str,
                               new_import: str) -> bool:
        """
        Change an import statement.
        Example: from pydantic import BaseSettings -> from pydantic_settings import BaseSettings
        """
        with open(file_path, "r") as f:
            source = f.read()

        # Use simple string replacement guarded by AST verification
        # Or use libcst ImportTransformer
        # (Implementation depends on specific migration)

        return True
```

### 7.5 GitHub Integration

```python
# src/tools/github_tools.py
from github import Github
from github.Repository import Repository
from typing import Optional

class GitHubIntegration:
    """Handles all GitHub operations."""

    def __init__(self, token: str, repo_name: str):
        self.g = Github(token)
        self.repo = self.g.get_repo(repo_name)

    def create_branch(self, base_branch: str, new_branch: str) -> str:
        """Create a new branch from base."""
        base = self.repo.get_branch(base_branch)
        ref = self.repo.create_git_ref(
            ref=f"refs/heads/{new_branch}",
            sha=base.commit.sha
        )
        return new_branch

    def update_file(self, file_path: str, content: str, 
                   message: str, branch: str):
        """Update a file in a branch."""
        file = self.repo.get_contents(file_path, ref=branch)
        self.repo.update_file(
            path=file_path,
            message=message,
            content=content,
            sha=file.sha,
            branch=branch
        )

    def create_pull_request(self, title: str, body: str,
                           head: str, base: str = "main") -> dict:
        """Open a pull request."""
        pr = self.repo.create_pull(
            title=title,
            body=body,
            head=head,
            base=base
        )
        return {
            "number": pr.number,
            "url": pr.html_url,
            "title": pr.title
        }

    def get_file_content(self, file_path: str, 
                        branch: str = "main") -> str:
        """Fetch file content from repo."""
        file = self.repo.get_contents(file_path, ref=branch)
        return file.decoded_content.decode("utf-8")

    def get_test_status(self, branch: str) -> Optional[str]:
        """Check the latest GitHub Actions workflow-run result for a branch.

        Note: modern GitHub Actions report through the Actions/Checks API, NOT
        the legacy commit-status API (`get_statuses()` / `context ==
        "continuous-integration"` will NOT see Actions results). We read the
        most recent workflow run on the branch and return its status/conclusion.

        Returns one of: "success", "failure", "pending", or None if no run yet.
        """
        runs = self.repo.get_workflow_runs(branch=branch)
        if runs.totalCount == 0:
            return None

        latest = runs[0]  # most recent run first
        if latest.status != "completed":
            return "pending"          # queued / in_progress
        # conclusion: "success", "failure", "cancelled", "timed_out", ...
        return "success" if latest.conclusion == "success" else "failure"
```

### 7.6 Notification System

```python
# src/tools/notify_tools.py
import requests
from typing import Dict

class Notifier:
    """Sends human-in-the-loop notifications."""

    def __init__(self, slack_webhook: str):
        self.slack_webhook = slack_webhook

    def send_high_confidence_alert(self, pr_info: Dict):
        """
        Auto-fix succeeded, tests passed.
        Notify human that PR is ready for quick review.
        """
        message = {
            "text": "DepGuard Auto-Migration Complete",
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": "Dependency Upgrade Ready for Review"
                    }
                },
                {
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": f"*Package:*\n{pr_info['package']}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Version:*\n{pr_info['from']} -> {pr_info['to']}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Changes:*\n{pr_info['changes_count']} files modified"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Tests:*\nAll passing"
                        }
                    ]
                },
                {
                    "type": "actions",
                    "elements": [
                        {
                            "type": "button",
                            "text": {
                                "type": "plain_text",
                                "text": "Review PR"
                            },
                            "url": pr_info["pr_url"],
                            "style": "primary"
                        },
                        {
                            "type": "button",
                            "text": {
                                "type": "plain_text",
                                "text": "Approve & Merge"
                            },
                            "action_id": f"approve_{pr_info['pr_number']}"
                        }
                    ]
                }
            ]
        }
        requests.post(self.slack_webhook, json=message)

    def send_low_confidence_alert(self, plan: Dict, pr_info: Dict):
        """
        Breaking changes detected, human judgment required.
        Show migration plan, ask for approval to proceed.
        """
        message = {
            "text": "DepGuard Needs Your Decision",
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": "Major Upgrade Requires Human Review"
                    }
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": (
                            f"*Package:* {plan['package']} {plan['from']} -> {plan['to']}\n"
                            f"*Risk Level:* {plan['estimated_risk'].upper()}\n"
                            f"*Confidence:* LOW -- Breaking changes detected\n\n"
                            f"*Migration Plan:*\n{plan['reasoning']}"
                        )
                    }
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": (
                            "*Breaking Changes:*\n" +
                            "\n".join([f"- {bc['description']}" for bc in plan['breaking_changes']])
                        )
                    }
                },
                {
                    "type": "actions",
                    "elements": [
                        {
                            "type": "button",
                            "text": {
                                "type": "plain_text",
                                "text": "View Migration Guide"
                            },
                            "url": pr_info["pr_url"]
                        },
                        {
                            "type": "button",
                            "text": {
                                "type": "plain_text",
                                "text": "Approve Migration"
                            },
                            "style": "primary",
                            "action_id": f"approve_migration_{pr_info['pr_number']}"
                        },
                        {
                            "type": "button",
                            "text": {
                                "type": "plain_text",
                                "text": "Ignore"
                            },
                            "style": "danger",
                            "action_id": f"ignore_{pr_info['pr_number']}"
                        }
                    ]
                }
            ]
        }
        requests.post(self.slack_webhook, json=message)
```

---

## 8. Strands Agents SDK Integration

### Installation

```bash
pip install strands-agents
```

### How Strands Actually Defines Tools and Agents

Strands is model-driven and minimal. There are two things to know:

1. **Tools are plain Python functions decorated with `@tool`** (imported from
   `strands`). The function's docstring becomes the description the model sees,
   and its type hints generate the input schema automatically. There is no
   `Tool(...)` wrapper class and no manual `parameters` dict.
2. **An agent is `Agent(model=..., tools=[...])`** and you invoke it by calling
   it directly: `result = agent("do the thing")`. There is no `.run(...)` method
   and no separate `AgentExecutor`.

### Agent Definition

```python
# src/agents/monitor_agent.py
from strands import Agent, tool
from src.tools.package_tools import PackageMonitor
from src.tools.changelog_tools import ChangelogFetcher

# Instantiate the helpers your tools delegate to.
_package_monitor = PackageMonitor()
_changelog_fetcher = ChangelogFetcher(github_token="...")  # inject from Secrets Manager


@tool
def scan_packages(repo_path: str) -> list:
    """Scan a repository for outdated Python packages.

    Args:
        repo_path: Absolute path to the checked-out repository.
    """
    return _package_monitor.scan_python_packages(repo_path)


@tool
def fetch_changelog(package_name: str, current_ver: str, target_ver: str) -> str:
    """Fetch and summarize release notes for a package between two versions.

    Args:
        package_name: The package to look up.
        current_ver: The currently installed version.
        target_ver: The version being upgraded to.
    """
    return _changelog_fetcher.get_migration_notes(package_name, current_ver, target_ver)


# The model string is passed to the Agent; Strands resolves Bedrock model IDs.
monitor_agent = Agent(
    model="amazon.nova-pro-v1:0",
    tools=[scan_packages, fetch_changelog],
    system_prompt=(
        "You monitor repositories for outdated dependencies. Use the provided "
        "tools to scan packages and fetch changelogs, then report findings."
    ),
)
```

### Agent Orchestration

```python
# src/main.py
from src.agents.monitor_agent import monitor_agent
from src.agents.planner_agent import planner_agent
from src.agents.executor_agent import executor_agent
from src.agents.validator_agent import validator_agent

class DepGuardOrchestrator:
    """
    Main orchestrator that runs the full agent pipeline.

    Note: Strands agents are invoked by *calling* them directly
    (`agent("prompt")`), not via a `.run(...)` method. Each call returns an
    AgentResult; use `.message` for the text or have the agent return
    structured data via a tool. The `_data(...)` helper below stands in for
    whatever parsing you do on the agent's response.
    """

    def __init__(self, repo_name: str):
        self.repo_name = repo_name
        self.state_store = StateStore()

    def run(self):
        # Step 1: Monitor -- detect outdated packages
        outdated = _data(monitor_agent(f"Scan repo {self.repo_name} for outdated packages"))

        for package in outdated:
            # Step 2: Plan -- decide migration strategy
            plan = _data(planner_agent(
                f"Plan migration for {package['name']} "
                f"from {package['current_version']} to {package['latest_version']}"
            ))

            # Step 3: Execute or Escalate
            if plan["migration_strategy"] == "auto_fix":
                result = _data(executor_agent(
                    f"Apply auto-fix for {package['name']} per plan: {plan}"
                ))

                # Step 4: Validate
                test_result = _data(validator_agent(
                    f"Run tests on branch {result['branch']}"
                ))

                if test_result["status"] == "passed":
                    # High confidence -- notify for quick approval
                    notifier.send_high_confidence_alert(result)
                else:
                    # Tests failed -- escalate to human
                    notifier.send_test_failure_alert(result, test_result)

            elif plan["migration_strategy"] == "guided_pr":
                # Open PR with migration guide, ask human to execute
                result = _data(executor_agent(
                    f"Create guided PR for {package['name']} with plan: {plan}"
                ))
                notifier.send_low_confidence_alert(plan, result)

            else:  # human_required
                # Just notify, don't touch code
                notifier.send_human_required_alert(plan)
```

---

## 9. Bedrock AgentCore Deployment

### Important: AgentCore Runtime is not the same as Bedrock Agents

There are two different AWS products with similar names, and this matters for
how you deploy:

- **Bedrock Agents** (the `AWS::Bedrock::Agent` CloudFormation resource, with
  `ActionGroups`, `ApiSchema`, and `KnowledgeBases`) is the older managed-agent
  product where AWS orchestrates the loop for you. It does **not** run a Strands
  agent.
- **Bedrock AgentCore Runtime** is the newer, serverless runtime where **you**
  write the agent loop in a framework you already know (Strands, LangGraph,
  etc.) and deploy your code to a managed runtime. This is the path that matches
  our architecture, so this is what we use. There is no
  `agentcore-config.yaml` with action groups — you deploy your Python entrypoint.

### How AgentCore Runtime deployment actually works

You wrap your existing Strands agent with a thin `BedrockAgentCoreApp`
entrypoint, test it locally as an HTTP service, then deploy it with the
AgentCore starter toolkit CLI. The toolkit packages the code, builds a
container image, provisions the runtime, and gives you an invocable endpoint.

#### Step 1: Install the runtime SDK and toolkit

```bash
pip install bedrock-agentcore            # the runtime wrapper (BedrockAgentCoreApp)
pip install bedrock-agentcore-starter-toolkit   # the `agentcore` CLI for deploy
```

#### Step 2: Wrap the orchestrator in an AgentCore entrypoint

```python
# src/agentcore_app.py
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from src.main import DepGuardOrchestrator

app = BedrockAgentCoreApp()


@app.entrypoint
def invoke(payload):
    """AgentCore Runtime entrypoint.

    Receives a JSON payload (the runtime passes whatever you invoke it with).
    For DepGuard we expect a repo name and kick off the full pipeline.
    """
    repo_name = payload.get("repo_name")
    if not repo_name:
        return {"error": "payload must include 'repo_name'"}

    orchestrator = DepGuardOrchestrator(repo_name=repo_name)
    orchestrator.run()
    return {"status": "completed", "repo": repo_name}


if __name__ == "__main__":
    # Runs a local HTTP server on :8080 so you can test before deploying.
    app.run()
```

Test it locally before you deploy anything:

```bash
python -m src.agentcore_app
# In another terminal:
curl -X POST http://localhost:8080/invocations \
  -H "Content-Type: application/json" \
  -d '{"repo_name": "your-org/your-test-repo"}'
```

#### Step 3: Configure and deploy with the AgentCore CLI

```bash
# Point the toolkit at your entrypoint module. This generates the runtime
# config (a .bedrock_agentcore.yaml) and, on launch, builds the container,
# provisions the runtime, and returns an agent runtime ARN + endpoint.
agentcore configure --entrypoint src/agentcore_app.py

# Build, provision, and deploy to AgentCore Runtime.
agentcore launch

# Invoke the deployed agent to confirm it's live.
agentcore invoke '{"repo_name": "your-org/your-test-repo"}'
```

> The IAM execution role AgentCore creates needs `bedrock:InvokeModel` for the
> Nova models plus access to your Secrets Manager entries (GitHub token, Slack
> webhook) and DynamoDB state table. Grant these to the runtime role the
> toolkit provisions, not to a Lambda.

### Supporting infrastructure (DynamoDB, secrets, scheduling)

AgentCore Runtime hosts the agent, but you still want a state table, secrets,
and a scheduled trigger. Keep those in a small SAM/CloudFormation stack — just
drop the fabricated `AWS::Bedrock::Agent` resource, since the agent now lives in
AgentCore Runtime rather than in this template.

```yaml
# infrastructure/template.yaml
AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31
Description: LusiScan -- supporting infrastructure (state, secrets, schedule)

Parameters:
  AgentRuntimeArn:
    Type: String
    Description: ARN of the AgentCore Runtime created by `agentcore launch`

Resources:
  # DynamoDB Table for Agent State
  AgentStateTable:
    Type: AWS::DynamoDB::Table
    Properties:
      TableName: DepGuardState
      BillingMode: PAY_PER_REQUEST
      AttributeDefinitions:
        - AttributeName: pk
          AttributeType: S
        - AttributeName: sk
          AttributeType: S
      KeySchema:
        - AttributeName: pk
          KeyType: HASH
        - AttributeName: sk
          KeyType: RANGE

  # EventBridge Scheduler -- invoke the AgentCore runtime every 6 hours.
  # (Target the AgentCore InvokeAgentRuntime API via an EventBridge
  # connection / API destination, or a thin Lambda that calls it.)
  MonitorSchedule:
    Type: AWS::Scheduler::Schedule
    Properties:
      Name: DepGuardMonitorSchedule
      ScheduleExpression: rate(6 hours)
      FlexibleTimeWindow:
        Mode: 'OFF'
      Target:
        Arn: !Ref AgentRuntimeArn
        RoleArn: !GetAtt SchedulerInvokeRole.Arn
```

> Store the GitHub token and Slack webhook in AWS Secrets Manager and read them
> at runtime from inside the agent — never bake them into the container image
> or pass them as plaintext CLI parameters.

---

## 10. GitHub Integration

### Required GitHub App / Token Permissions

Create a GitHub Personal Access Token (classic) with these scopes:

| Scope | Why |
|-------|-----|
| `repo` | Full repository access (read code, create branches, open PRs) |
| `workflow` | Trigger and read GitHub Actions workflows |

Or create a GitHub App for production-grade integration:
- Repository permissions: `Contents` (read/write), `Pull requests` (read/write), `Actions` (read)
- Subscribe to events: `push`, `pull_request`

### Webhook Setup (Optional but Impressive)

Instead of polling every 6 hours, set up a GitHub webhook that triggers your Lambda when `package.json` or `requirements.txt` is modified:

```python
# Webhook handler in Lambda
def lambda_handler(event, context):
    payload = json.loads(event["body"])

    # Only trigger on changes to dependency files
    modified_files = payload.get("commits", [{}])[0].get("modified", [])
    dep_files = ["package.json", "package-lock.json", "requirements.txt", 
                 "pyproject.toml", "Pipfile", "Cargo.toml"]

    if any(f in modified_files for f in dep_files):
        # Trigger agent analysis
        orchestrator = DepGuardOrchestrator(repo_name=payload["repository"]["full_name"])
        orchestrator.run()

    return {"statusCode": 200, "body": "OK"}
```

---

## 11. The Refactor Engine (Scoped)

### The Golden Rule

**Do NOT try to build a general-purpose auto-refactor.** Build a **pattern-matching engine** that handles 2-3 specific, pre-defined migration scenarios perfectly.

### Pre-Seeded Migration Registry

```python
# src/migrations/registry.py

MIGRATION_REGISTRY = {
    "pydantic": {
        "1.x -> 2.x": {
            "confidence": "low",  # Too complex for auto-fix
            "strategy": "guided_pr",
            "patterns": [
                {
                    "type": "import_change",
                    "old": "from pydantic import BaseSettings",
                    "new": "from pydantic_settings import BaseSettings",
                    "auto_fixable": True
                },
                {
                    "type": "class_rename",
                    "old": "BaseConfig",
                    "new": "ConfigDict",
                    "auto_fixable": True
                },
                {
                    "type": "behavior_change",
                    "old": "validator pre=True",
                    "new": "mode='before'",
                    "auto_fixable": False,
                    "note": "Requires human review for edge cases"
                }
            ]
        }
    },
    "requests": {
        "2.x -> 3.x": {
            "confidence": "high",
            "strategy": "auto_fix",
            "patterns": [
                {
                    "type": "param_rename",
                    "old": "verify=False",
                    "new": "ssl_verify=False",
                    "auto_fixable": True
                }
            ]
        }
    }
}
```

### AST-Aware Transformation

For the patterns marked `auto_fixable: True`, use `libcst` (Python) or `jscodeshift` (JavaScript) to apply the transformation safely.

For patterns marked `auto_fixable: False`, the agent:
1. Opens a PR with only the version bump
2. Attaches a detailed comment explaining the breaking change
3. Links to official migration docs
4. Asks the human to apply the fix

### LLM-Assisted Refactor (Fallback)

For unknown packages, use Nova Pro with few-shot examples:

```python
REFACTOR_PROMPT = """You are a code refactoring assistant.

Given the following breaking change description and code snippet,
produce the refactored code. Only output the modified code block.

## Breaking Change
{breaking_change}

## Current Code
```python
{code}
```

## Refactored Code
```python
"""
```

**Important:** Always validate LLM-generated code with:
1. AST parsing (must be valid syntax)
2. Unit tests (must pass)
3. Diff review (must only touch intended lines)

---

## 12. Testing & Validation Loop

### Test Execution Strategy

We validate migrations by running the **target repo's own GitHub Actions
workflow** on the temp branch and reading the run's result via the Actions API
(see `get_test_status` in 7.5 — it polls the workflow run's
`status`/`conclusion`, not the legacy commit-status API). For the demo, point
this at a repo you control with a tiny, fast test suite so CI finishes in
seconds and never flakes on camera.

> **13-day fallback:** if the GitHub Actions round-trip (push → queue → run →
> poll) eats too much time or flakes, run the repo's tests locally in the
> sandbox instead (`subprocess` → `pytest`) and return the same
> `{"status": ...}` dict. The rest of the pipeline doesn't care where the
> pass/fail came from. Start with Actions; keep this as the escape hatch.

```python
# src/agents/validator_agent.py
import time

class ValidatorAgent:
    """Validates that refactored code passes all tests."""

    def __init__(self, github_integration):
        self.github = github_integration

    def validate_branch(self, branch: str, timeout_minutes: int = 10) -> dict:
        """
        Trigger CI on branch and wait for results.
        Returns: {"status": "passed" | "failed" | "timeout", "details": ...}
        """
        # Push a commit to trigger GitHub Actions
        # (The branch already has the changes from executor)

        # Poll for CI status
        for _ in range(timeout_minutes * 6):  # Poll every 10 seconds
            status = self.github.get_test_status(branch)

            if status == "success":
                return {"status": "passed", "details": "All tests green"}
            elif status == "failure":
                logs = self.github.get_failed_test_logs(branch)
                return {"status": "failed", "details": logs}

            time.sleep(10)

        return {"status": "timeout", "details": "CI did not complete in time"}
```

### Confidence Scoring

```python
def calculate_confidence(plan: dict, test_result: dict) -> str:
    """
    Final confidence score combining planner confidence + test results.
    """
    if plan["confidence"] == "high" and test_result["status"] == "passed":
        return "HIGH"  # Auto-PR, notify for quick approval

    if plan["confidence"] == "high" and test_result["status"] == "failed":
        return "MEDIUM"  # Unexpected failure, human must investigate

    return "LOW"  # Complex migration or test failure, human required
```

---

## 13. Notification System

### Slack App Setup

1. Go to https://api.slack.com/apps
2. Create New App -> From scratch
3. Add features: **Incoming Webhooks** (for sending messages)
4. (Optional) Add **Interactive Components** (for approval buttons)
5. Copy the Webhook URL -> store in AWS Secrets Manager

### GitHub PR as Notification

For every migration, the agent opens a PR. The PR body serves as the primary notification:

```markdown
## LusiScan -- Automated Dependency Migration

### Package: `pydantic` 1.10.0 -> 2.5.0

### Confidence: LOW
Breaking changes detected. Human review required.

### Changes Applied
- Updated `pyproject.toml`: `pydantic = "^2.5.0"`
- Auto-fixed import: `BaseSettings` moved to `pydantic-settings`

### Breaking Changes Requiring Review
1. **Validator mode change**: `pre=True` -> `mode='before'`
   - Affected files: `src/models/user.py`, `src/models/order.py`
   - Impact: May change validation order for nested models

### Test Results
- 47/47 tests passing
- 3 deprecation warnings (non-blocking)

### Migration Guide
[Link to official Pydantic v2 migration docs](https://docs.pydantic.dev/latest/migration/)

---
**Action Required:** Please review the breaking changes above and either:
- Approve this PR if the auto-fixes look correct
- Push additional commits to handle the flagged breaking changes
```

---

## 13.5 Streamlit Control Panel & Live Demo Link

A live demo link scores higher on Technical Implementation (optional per the
rules, but cheap points). A small Streamlit app is the fastest way to get one,
and it makes the demo video far stronger than showing raw logs.

### The one rule that keeps the architecture honest

**Streamlit is a viewer/control panel, not where the agent runs.** The theme is
an agent that works *autonomously in the background* — so the agent loop lives
on AgentCore Runtime and writes its state (pending migrations, plans, diffs,
test results) to DynamoDB. Streamlit just **reads that state and posts approval
actions back**. If the agent loop lived inside Streamlit, you'd have "another
app people open and manage" — the exact thing the brief says not to build.

```
AgentCore Runtime (agent loop)  ──writes──▶  DynamoDB (state)  ◀──reads──  Streamlit UI
        ▲                                                                       │
        └──────────────── approve / ignore action (writes decision) ◀──────────┘
```

### Minimal app

```python
# app/streamlit_app.py
import streamlit as st
from src.state.dynamodb_store import StateStore

store = StateStore()

st.set_page_config(page_title="LusiScan", page_icon="🛡️")
st.title("🛡️ LusiScan — Dependency Migration Agent")

pending = store.list_pending_migrations()  # reads from DynamoDB

if not pending:
    st.success("No pending migrations. The agent is watching quietly. ✅")

for m in pending:
    with st.container(border=True):
        st.subheader(f"{m['package']}  {m['from']} → {m['to']}")
        st.caption(f"Confidence: {m['confidence'].upper()} · Risk: {m['risk']}")
        st.markdown(m["reasoning"])
        with st.expander("View diff"):
            st.code(m["diff"], language="diff")
        st.markdown(f"**Tests:** {m['test_summary']}  ·  [Open PR]({m['pr_url']})")

        c1, c2, c3 = st.columns(3)
        if c1.button("Approve & Merge", key=f"a_{m['id']}", type="primary"):
            store.record_decision(m["id"], "approved")
            st.rerun()
        if c2.button("Review PR", key=f"r_{m['id']}"):
            st.link_button("Go to PR", m["pr_url"])
        if c3.button("Ignore", key=f"i_{m['id']}"):
            store.record_decision(m["id"], "ignored")
            st.rerun()
```

The agent polls DynamoDB for recorded decisions and acts on them (merge the PR,
close it, or skip) — so the human's click in Streamlit drives the autonomous
agent, not a synchronous request in the web app.

### Deploy the live link

Fastest free option for the hackathon:

1. Push the repo to GitHub (already required for submission).
2. Deploy `app/streamlit_app.py` on **Streamlit Community Cloud**
   (share.streamlit.io) — connect the repo, pick the file, deploy.
3. Add AWS credentials + table name as Streamlit **secrets** (read-only IAM user
   scoped to the state table; never commit them).
4. Put the resulting public URL in your Devpost submission as the live demo link.

> Keep the IAM user read-mostly (read state, write only the `decision` field).
> Streamlit Community Cloud is public — don't give it broad AWS permissions.

---

## 14. Demo Script

### Demo Video Structure (5 minutes max)

> Uses the two scoped Python demo packages (`requests` safe patch, `pydantic`
> 1→2 human-in-the-loop) and drives from the Streamlit control panel + a real
> GitHub PR. No Node/JS — that keeps the demo consistent with what you built.

**Scene 1: The Problem (30 sec)**
- Show a `requirements.txt` / `pyproject.toml` with packages a year+ out of date and a `pip audit` flagging a known CVE.
- Voiceover: developers lose hours reading changelogs and hand-patching breaking changes — mechanical work that still eats a day.

**Scene 2: The Agent Works in the Background (90 sec)**
- Open the **Streamlit live link** — it shows the agent has already been running on AgentCore Runtime and surfaced two pending migrations.
- Walk the `requests` (safe patch) card: agent detected it, fetched the changelog, bumped the version, no code change needed, opened a PR, CI is green.
- Show the real GitHub PR with the clean diff and passing GitHub Actions check.

**Scene 3: The Human Decision (90 sec)**
- On the `requests` card, click **Approve & Merge** in Streamlit → the agent (polling DynamoDB) merges the PR. Show the PR flip to merged.
- Emphasize the human-in-the-loop moment: *"Tests passed. One click and it's done."*

**Scene 4: The Judgment Call (60 sec)**
- Walk the `pydantic` 1→2 card: agent classified it **LOW confidence**, auto-fixed the safe import move (`BaseSettings` → `pydantic-settings`), but **flagged** the validator behavior change (`pre=True` → `mode='before'`) for a human.
- Show the guided PR with the migration notes. Click **Review PR**; then choose to defer.
- Point out: the agent did the mechanical part and *stopped* where judgment was needed — it didn't guess.

**Scene 5: Why It Matters (30 sec)**
- "LusiScan handles the 80% of dependency upgrades that are mechanical, so developers focus on the 20% that need their expertise — and it runs in the background, only pinging you for a real decision."
- Show the architecture diagram.
- Mention: Strands Agents SDK + Amazon Nova + deployed on Bedrock AgentCore Runtime, with a live Streamlit demo.

---

## 15. Submission Checklist

### Required Deliverables

- [ ] **Text description** on Devpost explaining problem, solution, and audience
- [ ] **Public GitHub repo** with MIT/Apache license visible in About section
- [ ] **README** with setup instructions
- [ ] **Architecture diagram** (PNG/SVG in repo)
- [ ] **Demo video** (max 5 minutes) covering problem, audience, and why it matters
- [ ] **AWS Builder ID** (create at https://builder.aws.com/)
- [ ] **(Optional but recommended)** Live demo link

### Bonus Points

- [ ] **Builder.aws.com blog post** using hashtag `#AgentsforHumans`
  - Write about your build journey, challenges with Strands SDK, why you chose Nova models
  - Publish before submission deadline
  - You can submit multiple posts

### Code Quality Checklist

- [ ] All secrets in AWS Secrets Manager (never in code)
- [ ] Error handling for all external API calls
- [ ] Logging for agent decisions (why it chose auto_fix vs guided_pr)
- [ ] Unit tests for core logic (monitor, planner, refactor engine)
- [ ] Type hints throughout Python codebase
- [ ] `requirements.txt` or `pyproject.toml` with pinned versions

### Pre-Submission Testing

- [ ] Test on a real repository (create a test repo with outdated packages)
- [ ] Verify Slack notifications render correctly
- [ ] Verify GitHub PRs open with correct diffs
- [ ] Verify AgentCore deployment is live and responding
- [ ] Record demo video in one take (or edit cleanly)
- [ ] Watch video back -- does it clearly show the problem, solution, and human decision moment?

---

## Appendix A: Pre-Defined Demo Scenarios

### Scenario A: Safe Auto-Fix (High Confidence)

**Package:** `requests` 2.28.0 -> 2.31.0 (patch bump, no breaking changes)
**Expected Behavior:**
1. Agent detects update
2. Changelog shows security fix + no breaking changes
3. Agent bumps version in `requirements.txt`
4. No code changes needed
5. Tests pass
6. Agent opens PR, Slack says "Ready to merge -- no action needed"

### Scenario B: Guided Migration (Low Confidence)

**Package:** `pydantic` 1.10.0 -> 2.5.0 (major bump, breaking changes)
**Expected Behavior:**
1. Agent detects update
2. Changelog shows extensive breaking changes
3. Agent classifies as LOW confidence
4. Agent bumps version, auto-fixes simple import changes
5. Agent flags complex behavior changes for human review
6. Agent opens PR with detailed migration guide
7. Slack says "I need your judgment on 3 breaking changes"

### Scenario C: Security Vulnerability (High Priority)

**Package:** `requests` 2.28.0 -> 2.31.0 (security patch, no breaking changes)
**Expected Behavior:**
1. Agent detects via `pip audit` or a security advisory
2. Prioritizes as HIGH urgency
3. Bumps the pinned version; no code change required
4. Tests pass
5. Agent opens PR with security context in description

> Note: this is the same real, safe `requests` bump used in Scenario A — reused
> here to show the security-prioritization path without introducing a second
> ecosystem. Keep the demo Python-only.

---

## Appendix B: Troubleshooting

### Nova Model Not Available in Bedrock
- Check region: Nova models are available in `us-east-1` and `us-west-2`
- Verify in AWS Console: Bedrock -> Model access -> Amazon Nova models should show "Available"

### Strands SDK Installation Issues
- Requires Python 3.10+
- `pip install strands-agents` -- if fails, try `pip install --upgrade pip` first

### GitHub API Rate Limits
- Use authenticated requests (token increases limit from 60/hr to 5,000/hr)
- For demo, don't run monitor too frequently

### Lambda Timeout
- Default Lambda timeout is 3 seconds -- increase to 300 seconds for agent workflows
- Consider Step Functions for long-running migration pipelines

---

## Appendix C: Useful Resources

| Resource | URL |
|----------|-----|
| Strands Agents SDK Docs | https://strandsagents.com/docs |
| AWS Bedrock AgentCore (Runtime) | https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/ |
| Strands → AgentCore deploy guide | https://strandsagents.com/docs/user-guide/deploy/deploy_to_bedrock_agentcore/python/ |
| AgentCore starter toolkit | https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/getting-started-starter-toolkit.html |
| Amazon Nova Models | https://docs.aws.amazon.com/nova/latest/userguide/what-is-nova.html |
| GitHub REST API | https://docs.github.com/en/rest |
| libcst (Python AST) | https://libcst.readthedocs.io/ |
| jscodeshift (JS AST) | https://github.com/facebook/jscodeshift |
| AWS SAM CLI | https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/install-sam-cli.html |
| Slack Block Kit Builder | https://app.slack.com/block-kit-builder |
| Hackathon Rules | https://agentsforhumans.devpost.com/rules |

---

*Good luck. Build the agent loop first, make the refactor engine work for 2 scenarios, and let Nova Pro handle the reasoning. You have got this.*
