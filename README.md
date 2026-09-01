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

> _TODO: run instructions (local AgentCore entrypoint on `:8080`, `agentcore
> launch`, invoking the demo scenarios, launching the Streamlit panel)._

## Tests

```bash
pytest
```

## License

Released under the [MIT License](LICENSE).
