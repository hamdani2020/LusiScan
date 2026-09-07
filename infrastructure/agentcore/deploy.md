# AgentCore Runtime deploy runbook (task 11.2)

This is the reproducible deploy procedure for hosting the LusiScan agent loop on
**Amazon Bedrock AgentCore Runtime** (R6.1) and granting its runtime role the
least-privilege access it needs: **Bedrock** (Nova Pro/Lite/Micro),
**Secrets Manager** (read), and **DynamoDB** (CRUD on the state table). It
satisfies:

- **R6.1** — LusiScan is deployed to Amazon Bedrock AgentCore Runtime as the agent host.
- **R6.5** — Secrets (GitHub token, Slack webhook) are read at runtime from AWS
  Secrets Manager and are never committed or baked into the image.

The entrypoint being deployed is [`src/agentcore_app.py`](../../src/agentcore_app.py)
(`BedrockAgentCoreApp` + `@app.entrypoint`), which wraps
`DepGuardOrchestrator` from `src/main.py` (built and tested locally in task 11.1).

> Status: The artifacts here are checked in so the deploy is reproducible. The
> live `agentcore configure` / `agentcore launch` steps require an AWS account
> with credentials, Bedrock Nova model access enabled, and the AgentCore
> starter toolkit installed — run them from a machine that has those.

---

## Prerequisites

1. **Install the deps** (they are pinned in `pyproject.toml`):

   ```bash
   pip install -e .
   # brings in bedrock-agentcore + bedrock-agentcore-starter-toolkit (the
   # `agentcore` CLI), strands-agents, boto3, etc.
   ```

   Verify the CLI is the Python starter toolkit (not an unrelated `agentcore`
   binary on PATH):

   ```bash
   python -m bedrock_agentcore_starter_toolkit --help  # or:
   agentcore --help
   ```

2. **AWS credentials** with permission to create IAM roles, push the runtime,
   and (for the operator) invoke Bedrock. Confirm:

   ```bash
   aws sts get-caller-identity
   ```

3. **Bedrock model access** enabled for `amazon.nova-pro-v1:0`,
   `amazon.nova-lite-v1:0`, and `amazon.nova-micro-v1:0` in the target region
   (Bedrock console → Model access).

4. **The DynamoDB state table exists** (provisioned by Terraform in task 16) or
   you at least know its name. The runtime role scopes DynamoDB access to this
   one table.

---

## Step 1 — Store the runtime secrets in Secrets Manager (R6.5)

Secrets are **never** committed or passed on the command line into the image.
Create them once in Secrets Manager; the runtime reads them at invoke time. The
secret *names* below are what the IAM policy grants read access to
(`lusiscan/github-token`, `lusiscan/slack-webhook`).

```bash
# GitHub token used by github_tools (branch/commit/PR/merge-on-approval).
aws secretsmanager create-secret \
  --name lusiscan/github-token \
  --secret-string "REPLACE_WITH_GITHUB_TOKEN"

# Slack webhook used by the notifier (optional; task 13).
aws secretsmanager create-secret \
  --name lusiscan/slack-webhook \
  --secret-string "REPLACE_WITH_SLACK_WEBHOOK_URL"
```

> Do not paste real secret values into any file in this repo. Provide them only
> to the `aws secretsmanager` call above (ideally via `--secret-string file://...`
> pointing at an untracked file), or set them through the AWS console.

The application reads the resolved values at runtime from the environment the
runtime injects (`GITHUB_TOKEN`, and the table name from `DEPGUARD_STATE_TABLE`
/ `STATE_TABLE_NAME`) — see the config section in `src/agentcore_app.py`. No
secret material is read, logged, or committed by the code.

---

## Step 2 — Create the runtime IAM role (Bedrock + Secrets Manager + DynamoDB)

The trust policy and least-privilege permissions are checked in next to this
runbook:

- [`runtime-role-trust-policy.json`](./runtime-role-trust-policy.json) — lets the
  AgentCore service (`bedrock-agentcore.amazonaws.com`) assume the role, scoped
  to this account/region via `aws:SourceAccount` / `aws:SourceArn`.
- [`runtime-role-permissions-policy.json`](./runtime-role-permissions-policy.json)
  — grants exactly:
  - `bedrock:InvokeModel` / `InvokeModelWithResponseStream` on the three Nova
    model ARNs (R8.1/8.2);
  - `secretsmanager:GetSecretValue` / `DescribeSecret` on `lusiscan/github-token-*`
    and `lusiscan/slack-webhook-*` only (R6.5);
  - DynamoDB CRUD (`GetItem`/`PutItem`/`UpdateItem`/`DeleteItem`/`Query`/batch/
    conditional) on the single state table and its indexes only.

Create the role and attach the policy with the helper script (it substitutes
`${AWS_ACCOUNT_ID}` / `${AWS_REGION}` / `${DEPGUARD_STATE_TABLE}` and prints the
resulting role ARN):

```bash
AWS_REGION=us-east-1 \
DEPGUARD_STATE_TABLE=lusiscan-state \
./infrastructure/agentcore/create-runtime-role.sh
```

Capture the printed role ARN for the next step, e.g.:

```
arn:aws:iam::123456789012:role/LusiScanAgentCoreRuntimeRole
```

---

## Step 3 — `agentcore configure`

Point the runtime at the entrypoint and the role created above. `--name` keeps
the agent name stable across re-deploys; `--requirements-file` ensures the
runtime image installs the pinned deps.

```bash
agentcore configure \
  --entrypoint src/agentcore_app.py \
  --name lusiscan \
  --execution-role arn:aws:iam::123456789012:role/LusiScanAgentCoreRuntimeRole \
  --requirements-file pyproject.toml \
  --region us-east-1 \
  --non-interactive
```

This writes a `.bedrock_agentcore.yaml` describing the runtime. That file
records config (entrypoint, role ARN, region) — **not** secrets.

---

## Step 4 — `agentcore launch`

Build and deploy the runtime. Pass the **non-secret** runtime config as env
vars (the state table name); secrets are resolved from Secrets Manager, not
passed here.

```bash
agentcore launch \
  --env DEPGUARD_STATE_TABLE=lusiscan-state
```

`launch` packages the code, provisions/uses the ECR image, and creates the
AgentCore runtime with the execution role from Step 3. Wait for it to report the
runtime is ready.

---

## Step 5 — Verify with `agentcore invoke` (handed to task 11.3)

Task 11.3 runs both demo scenarios end-to-end on the live runtime. A smoke
invoke:

```bash
agentcore invoke '{"repo_name": "your-org/demo-repo"}'
```

Expect a structured result from the entrypoint,
`{"status": "completed", "repo": "...", "migrations": [...], ...}`.

---

## Least-privilege rationale (design.md → Security)

| Grant | Scope | Why |
|-------|-------|-----|
| `bedrock:InvokeModel*` | The 3 Nova model ARNs only | Planner/summary/classify reasoning (R8.1/8.2); no wildcard `bedrock:*`. |
| `secretsmanager:GetSecretValue` + `DescribeSecret` | `lusiscan/github-token-*`, `lusiscan/slack-webhook-*` | Read the two runtime secrets by name (R6.5); no other secrets readable. |
| DynamoDB CRUD | The single state table + its indexes | Persist migrations/decisions/run logs (R6.3); scoped to one table, not `dynamodb:*`. |

No `iam:*`, no `s3:*`, no account-wide wildcards. The trust policy is scoped to
the AgentCore service principal in this account/region.

## Teardown

Everything created here is easy to remove after the hackathon:

```bash
# Remove the runtime.
agentcore destroy --name lusiscan   # or delete via the AgentCore console

# Remove the role + inline policy.
aws iam delete-role-policy --role-name LusiScanAgentCoreRuntimeRole \
  --policy-name LusiScanAgentCoreRuntimePolicy
aws iam delete-role --role-name LusiScanAgentCoreRuntimeRole

# Remove the secrets.
aws secretsmanager delete-secret --secret-id lusiscan/github-token --force-delete-without-recovery
aws secretsmanager delete-secret --secret-id lusiscan/slack-webhook --force-delete-without-recovery
```

(The DynamoDB table + EventBridge Scheduler are owned by the Terraform stack in
task 16 and torn down with `terraform destroy`.)
