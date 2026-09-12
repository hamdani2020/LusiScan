# LusiScan

Autonomous AI agent that takes the repetitive, judgment-heavy work out of
Python dependency upgrades. LusiScan detects outdated packages in a target
repository, analyzes migration impact, applies safe scoped refactors, validates
them with tests, and surfaces a decision to a human only when one is genuinely
needed.

Built with the [Strands Agents SDK](https://github.com/strands-agents), reasoning
on [Amazon Nova](https://aws.amazon.com/ai/generative-ai/nova/) via Bedrock, and
deployed on **Amazon Bedrock AgentCore Runtime**. A Streamlit control panel
provides a live link for reviewing pending migrations.

## Architecture

- **Agent loop** (Strands `@tool` stages: Monitor → Planner → Executor →
  Validator), hosted on Bedrock AgentCore Runtime.
- **Reasoning** with Amazon Nova (Pro / Lite / Micro) via Bedrock.
- **State** in DynamoDB (single-table store of migrations, decisions, run logs).
- **Control panel** in Streamlit, reading state and recording human decisions.

## Project layout

```
src/
  agents/    Strands agent stages + orchestrator
  tools/     package, changelog, refactor, github, notify tools
  models/    Bedrock/Nova client wrappers
  state/     DynamoDB single-table store
  prompts/   Nova prompt templates
app/         Streamlit control panel
tests/       Unit and integration tests
infrastructure/  SAM/CloudFormation (DynamoDB + EventBridge Scheduler)
```

## Setup

> _TODO: setup instructions (Python 3.11+, virtualenv, `pip install -e .[dev]`,
> AWS credentials, Secrets Manager entries for the GitHub token and Slack
> webhook)._

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Run

LusiScan runs its agent loop on **Amazon Bedrock AgentCore Runtime**. The
entrypoint is [`src/agentcore_app.py`](src/agentcore_app.py) — a
`BedrockAgentCoreApp` whose `@app.entrypoint` takes a JSON payload
(`{"repo_name": "owner/repo"}`), runs one full Monitor → Planner → Executor →
Validator cycle for that repo, and returns a structured result.

### Local (`:8080`)

Run the entrypoint locally to exercise the loop before deploying:

```bash
# Starts AgentCore's local HTTP server on :8080.
python -m src.agentcore_app
```

In another shell, invoke it:

```bash
curl -s -X POST http://localhost:8080/invocations \
  -H 'Content-Type: application/json' \
  -d '{"repo_name": "hamdani2020/lusiscan-demo-repo"}' | jq
```

The entrypoint reads its runtime config from the environment (no secrets in
source):

- `GITHUB_TOKEN` — used to read the target repo and open PRs.
- `AWS_REGION` — region for Bedrock (Nova) and any AWS calls.
- `DEPGUARD_STATE_TABLE` (or `STATE_TABLE_NAME`) — DynamoDB state table name.
  Optional; when unset the loop runs **stateless** (no decision persistence),
  which is fine for a smoke run before the table exists (task 16).

AWS credentials come from the standard boto3 chain, and Bedrock Nova model
access must be enabled in the region.

### Deploy to AgentCore Runtime

The reproducible deploy runbook (IAM role, `configure`, `launch`) lives in
[`infrastructure/agentcore/deploy.md`](infrastructure/agentcore/deploy.md). In
short:

```bash
agentcore configure --entrypoint src/agentcore_app.py --name lusiscan \
  --execution-role arn:aws:iam::<ACCOUNT>:role/LusiScanAgentCoreRuntimeRole \
  --requirements-file pyproject.toml --region us-east-1 --non-interactive

# Build (CodeBuild) + deploy. Only non-secret runtime config is passed as --env.
agentcore launch --env AWS_REGION=us-east-1
```

> The GitHub token is **not** passed on the command line. At invoke time the
> runtime reads it from AWS Secrets Manager (secret `lusiscan/github-token`,
> overridable via the `GITHUB_TOKEN_SECRET_ID` env var), so nothing sensitive is
> stored on the runtime config or baked into the image (R6.5). Provision it once:
>
> ```bash
> aws secretsmanager create-secret --name lusiscan/github-token \
>   --secret-string file://<untracked-token-file> --region us-east-1
> ```
>
> For local runs you can instead export `GITHUB_TOKEN` directly — the resolver
> prefers that env var and only falls back to Secrets Manager when it is unset.

### Invoke the live runtime

Once `agentcore status` reports the endpoint is `READY`, invoke it with a repo:

```bash
agentcore invoke '{"repo_name": "hamdani2020/lusiscan-demo-repo"}'
```

A successful run returns a structured summary:

```json
{
  "status": "completed",
  "repo": "hamdani2020/lusiscan-demo-repo",
  "outcome": "completed",
  "errors": [],
  "migrations": [
    { "package": "requests",  "current": "2.31.0",   "target": "2.34.2",
      "strategy": "auto_fix",       "pr_number": 1, "status": "pending_review" },
    { "package": "pydantic",  "current": "1.10.13",  "target": "2.13.5",
      "strategy": "human_required", "pr_number": 2, "status": "pending_review" }
  ]
}
```

## End-to-end verification (task 11.3)

Both demo scenarios were exercised end-to-end against the **live** AgentCore
runtime (`lusiscan-OQlFbt626s`) using the controlled demo repo
[`hamdani2020/lusiscan-demo-repo`](https://github.com/hamdani2020/lusiscan-demo-repo),
whose `pyproject.toml` intentionally pins:

- `requests==2.31.0` — a safe patch/minor bump (**auto-fix** scenario).
- `pydantic==1.10.13` — a 1 → 2 major upgrade (**human-in-the-loop** scenario).

Procedure:

1. Ensure the runtime is deployed and `READY`:

   ```bash
   agentcore status
   ```

2. Invoke the runtime for the demo repo (drives **both** scenarios, since the
   repo contains both outdated pins):

   ```bash
   agentcore invoke '{"repo_name": "hamdani2020/lusiscan-demo-repo"}'
   ```

3. Confirm success in the runtime's CloudWatch logs — a healthy run logs
   `POST /invocations ... 200 OK` and `Invocation completed successfully`:

   ```bash
   aws logs tail /aws/bedrock-agentcore/runtimes/lusiscan-OQlFbt626s-DEFAULT \
     --since 10m --region us-east-1 | grep -viE 'GET /ping' | grep -iE 'invocation|200|error'
   ```

Observed result: `outcome: completed`, `errors: []`, and two migrations —
`requests` 2.31.0 → 2.34.2 (auto-fix, PR opened) and `pydantic` 1.10.13 → 2.13.5
(`human_required`, breaking changes flagged for review, PR opened) — each left
in `pending_review` for a human decision. Re-invoking is idempotent: existing
PRs are reused rather than re-created.

## Tests

```bash
pytest
```

## License

Released under the [MIT License](LICENSE).
