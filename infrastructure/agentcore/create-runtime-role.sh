#!/usr/bin/env bash
#
# create-runtime-role.sh — create the LusiScan AgentCore Runtime IAM role and
# attach its least-privilege permissions policy (Bedrock + Secrets Manager +
# DynamoDB), satisfying tasks 11.2 and requirements 6.1 / 6.5.
#
# This is the reproducible, checked-in definition of the runtime role that
# `agentcore configure` / `agentcore launch` (see deploy.md) attach to the
# runtime. It contains NO secret material — secrets are read from Secrets
# Manager at runtime (R6.5). It only wires *permission* to read two named
# secrets by ARN pattern; it never reads or embeds their values.
#
# The trust policy (runtime-role-trust-policy.json) and permissions policy
# (runtime-role-permissions-policy.json) use ${AWS_ACCOUNT_ID}, ${AWS_REGION},
# and ${DEPGUARD_STATE_TABLE} placeholders that this script substitutes with
# `envsubst` before calling the AWS API.
#
# Prerequisites:
#   - AWS CLI v2, authenticated with permissions to create IAM roles/policies.
#   - The DynamoDB state table already created (task 16 / Terraform), or at
#     least its name known — the policy scopes DynamoDB access to that table.
#
# Usage:
#   AWS_REGION=us-east-1 \
#   DEPGUARD_STATE_TABLE=lusiscan-state \
#   ./create-runtime-role.sh
#
# Optional overrides:
#   ROLE_NAME              (default: LusiScanAgentCoreRuntimeRole)
#   POLICY_NAME            (default: LusiScanAgentCoreRuntimePolicy)

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${AWS_REGION:?set AWS_REGION (e.g. us-east-1)}"
: "${DEPGUARD_STATE_TABLE:?set DEPGUARD_STATE_TABLE (the DynamoDB state table name)}"

ROLE_NAME="${ROLE_NAME:-LusiScanAgentCoreRuntimeRole}"
POLICY_NAME="${POLICY_NAME:-LusiScanAgentCoreRuntimePolicy}"

# Resolve the account id from the current caller identity so the ARNs are exact.
AWS_ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
export AWS_ACCOUNT_ID AWS_REGION DEPGUARD_STATE_TABLE

echo "Account:      ${AWS_ACCOUNT_ID}"
echo "Region:       ${AWS_REGION}"
echo "State table:  ${DEPGUARD_STATE_TABLE}"
echo "Role:         ${ROLE_NAME}"
echo "Policy:       ${POLICY_NAME}"

# Render the placeholder-bearing policy docs into concrete JSON.
TRUST_DOC="$(envsubst < "${HERE}/runtime-role-trust-policy.json")"
PERM_DOC="$(envsubst < "${HERE}/runtime-role-permissions-policy.json")"

# Create (or update) the role's trust relationship.
if aws iam get-role --role-name "${ROLE_NAME}" >/dev/null 2>&1; then
  echo "Role exists; updating trust policy."
  aws iam update-assume-role-policy \
    --role-name "${ROLE_NAME}" \
    --policy-document "${TRUST_DOC}"
else
  echo "Creating role."
  aws iam create-role \
    --role-name "${ROLE_NAME}" \
    --description "LusiScan AgentCore Runtime execution role (Bedrock Nova + Secrets Manager + DynamoDB state table)" \
    --assume-role-policy-document "${TRUST_DOC}"
fi

# Attach the least-privilege permissions as an inline policy so it lives and
# dies with the role (easy single-command teardown after the hackathon).
echo "Putting inline permissions policy."
aws iam put-role-policy \
  --role-name "${ROLE_NAME}" \
  --policy-name "${POLICY_NAME}" \
  --policy-document "${PERM_DOC}"

ROLE_ARN="$(aws iam get-role --role-name "${ROLE_NAME}" --query 'Role.Arn' --output text)"
echo
echo "Runtime role ready: ${ROLE_ARN}"
echo "Pass this ARN to \`agentcore configure --execution-role\` (see deploy.md)."
