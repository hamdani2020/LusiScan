"""Orchestrator: the Strands agent loop wiring every stage together (task 10).

This module is the seam where the individually-built pieces of LusiScan become a
single **Monitor → Planner → Executor → Validator** loop. It exposes two things
the rest of the system depends on:

- The five stage functions the design pins to the Strands Agent layer
  (:func:`scan_packages`, :func:`fetch_changelog`, :func:`plan_migration`,
  :func:`apply_migration`, :func:`validate_branch`), each decorated with
  Strands' ``@tool`` and each returning a **structured dict** so the pipeline
  acts deterministically on tool-returned data rather than parsing free-form
  model text (design.md → Agent layer, "Tools return structured dicts").
- :class:`DepGuardOrchestrator`, whose constructor and ``run()`` signature match
  exactly what the AgentCore entrypoint imports and calls
  (``from src.main import DepGuardOrchestrator`` →
  ``DepGuardOrchestrator(repo_name=repo).run()``; design.md → AgentCore
  entrypoint).

## How the loop is composed (task 10.1)

Following the Strands convention (design.md → API correction #1), tools are
plain functions decorated with ``@tool`` and are invoked by **calling them
directly** — there is no ``Tool(...)`` wrapper, no ``.run()``, and no
``AgentExecutor``. The orchestrator sequences them by ordinary function calls:

    Monitor  (scan_packages)  → outdated packages
    Planner  (fetch_changelog → plan_migration) → a structured migration plan
    Executor (apply_migration → open PR)        → a branch + PR + diff
    Validator(validate_branch)                  → a pass/fail result

## Degrading gracefully without the SDK (task 10.1)

The Strands SDK may not be importable in every environment (it isn't in unit
tests). To keep the orchestrator's *pure* logic importable and testable without
the SDK, ``@tool`` is resolved behind a safe import guard: we try
``from strands import tool`` and fall back to an identity decorator otherwise
(mirroring the lazy/injectable dependency pattern already used by
``bedrock_client.py``, ``store.py``, ``github_tools.py``, and
``changelog_tools.py``). Either way the stage functions stay callable.

## Persisting state and honoring decisions each cycle (task 10.2)

Every cycle (design.md → Data flow steps 6-8):

1. **Read recorded decisions first.** For every migration already in the state
   table, the orchestrator reads any human decision the Streamlit control panel
   recorded (``StateStore.get_decision``) and acts on it — ``approved`` merges
   the PR (``github_tools.apply_decision``) and transitions the migration
   ``pending_review → approved → merged``; ``ignored`` closes/skips it and
   transitions ``pending_review → ignored → closed``; ``review`` / none leaves
   the PR open. It **never** merges without an explicit approval (R5.4) and only
   moves state through the legal transitions the store enforces (R5.5).
2. **Discover + plan + execute new migrations**, writing each freshly discovered
   migration to DynamoDB with status ``pending_review`` (``put_migration``) and
   the run outcome via ``put_run_log`` (R6.3).

The orchestrator is the decision **reader/actor**; the Streamlit app is the
decision **writer** (design.md → "State as the contract").

## The Validator seam (task 12 not built yet)

GitHub Actions polling is task 12. So the Validator stage is designed around an
**injectable validator**: :func:`validate_branch` delegates to a callable that
defaults to :func:`_stub_validator` (returns the ``{"status", "details"}`` shape
every downstream consumer expects). Task 12 plugs in the real Actions-API poller
by passing ``validator=`` to the orchestrator — no rework of the loop required.

Secrets (design.md → Security; R6.5): this module never hardcodes a GitHub token
or AWS credentials. GitHub auth is delegated to ``github_tools.get_client``
(injected client, or a token read from the ``GITHUB_TOKEN`` environment variable
that AgentCore populates from Secrets Manager at runtime), and DynamoDB access
is delegated to ``StateStore`` (injected table or lazy ``boto3``). No secret
material is read or stored here.

Design references:
- design.md → Agent layer (Strands): the five ``@tool`` stages and the
  ``DepGuardOrchestrator`` responsibilities.
- design.md → API correction #1: tools are ``@tool`` functions invoked directly;
  no ``Tool(...)`` / ``.run()`` / ``AgentExecutor``.
- design.md → AgentCore entrypoint: ``DepGuardOrchestrator(repo_name=repo).run()``.
- design.md → Data flow steps 2-8; "State as the contract".
- requirements.md → R1.x (detect), R2.x (analyze/plan), R3.x (refactor),
  R5.4/R5.5 (human-in-the-loop, honor decisions), R6.3 (persist state).
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from src.agents import planner_agent
from src.state import store as state_store
from src.state.store import InvalidStatusTransition, StateStore
from src.tools import changelog_tools, github_tools, package_tools, refactor_tools


# --- Strands @tool import guard (task 10.1) -------------------------------
#
# Strands' convention (design.md → API correction #1) is that a tool is just a
# plain function decorated with ``@tool``; the decorated function is still
# callable directly. When the SDK is absent (e.g. in unit tests), we fall back
# to an identity decorator so the stage functions — and the orchestrator's pure
# logic — remain importable and callable without the SDK installed. This mirrors
# the lazy/injectable dependency pattern used across the existing modules.
try:  # pragma: no cover - exercised implicitly by import; both paths are trivial
    from strands import tool as _strands_tool

    _HAS_STRANDS = True
except Exception:  # noqa: BLE001 - any import failure degrades to the identity decorator

    def _strands_tool(func: Callable[..., Any]) -> Callable[..., Any]:
        """Identity decorator standing in for ``strands.tool`` when absent.

        Returns the function unchanged so it stays directly callable — exactly
        how a real Strands ``@tool`` function is invoked (design.md → correction
        #1: tools are called directly, never wrapped or ``.run()``-ed).
        """
        return func

    _HAS_STRANDS = False


# Public alias so stage functions read as ``@tool`` regardless of SDK presence.
tool = _strands_tool


# --- Result-shape constants -----------------------------------------------

# The validator result contract (design.md → ValidatorAgent: branch →
# ``{status, details}``). ``passed`` / ``failed`` mirror the R4.4 mapping the
# real GitHub-Actions validator (task 12) will produce; ``skipped`` is what the
# default stub returns until task 12 is wired in.
VALIDATION_PASSED = "passed"
VALIDATION_FAILED = "failed"
VALIDATION_SKIPPED = "skipped"

# The type of an injectable validator: given a branch name, return a
# ``{"status", "details"}`` dict. Task 12 supplies the GitHub-Actions poller;
# until then :func:`_stub_validator` is used.
Validator = Callable[[str], dict]


# =====================================================================
# Stage functions — the Strands @tool Agent layer (task 10.1)
# =====================================================================


@tool
def scan_packages(repo_path: str) -> dict:
    """Monitor stage: detect outdated packages in ``repo_path`` (R1.x).

    Thin ``@tool`` wrapper over :func:`package_tools.scan_packages`, which parses
    the manifest without installing anything and resolves latest versions from
    PyPI, recording (not raising) per-repo errors so an unparseable manifest
    never crashes the loop (R1.4).

    Args:
        repo_path: Path to the target repository (or directly to its manifest).

    Returns:
        The monitor's :data:`package_tools.ScanResult` dict
        ``{"repo_path", "outdated", "errors"}``.
    """
    return package_tools.scan_packages(repo_path)


@tool
def fetch_changelog(
    package: str,
    current: str,
    target: str,
    *,
    model: Any,  # injected changelog_tools.ChangelogModel (see docstring)
) -> dict:
    """Planner stage (part 1): fetch + summarize a package's changelog (R2.1/2.2).

    ``@tool`` wrapper over :func:`changelog_tools.fetch_and_summarize_changelog`.
    On any failure (unmapped package, network error, empty range, summarization
    failure) the returned :data:`changelog_tools.ChangelogResult` has
    ``confidence`` pinned to ``"low"`` so the planner routes the migration to
    human review (R2.4) — the failure is data, not an exception.

    Args:
        package: Distribution name of a supported demo package.
        current: Currently pinned version (exclusive lower bound).
        target: Version being upgraded to (inclusive upper bound).
        model: Injected Nova Lite client satisfying
            :class:`changelog_tools.ChangelogModel`.

    Returns:
        A :data:`changelog_tools.ChangelogResult` dict.
    """
    return changelog_tools.fetch_and_summarize_changelog(
        package, current, target, model=model
    )


@tool
def plan_migration(
    changelog: dict,
    *,
    model: Any,  # injected planner_agent.PlannerModel (see docstring)
    code: Optional[str] = None,
) -> dict:
    """Planner stage (part 2): turn a changelog into a structured plan (R2.3/8.x).

    ``@tool`` wrapper over :func:`planner_agent.plan_migration`. Returns a
    normalized plan (``confidence`` / ``strategy`` / ``estimated_risk`` /
    ``breaking_changes`` / ``reasoning``) and degrades to a conservative
    ``low`` / ``human_required`` plan on an already-low changelog or a non-JSON
    model failure (R2.4).

    Args:
        changelog: A :data:`changelog_tools.ChangelogResult` from
            :func:`fetch_changelog`.
        model: Injected Nova Pro client satisfying
            :class:`planner_agent.PlannerModel`.
        code: Optional source code to give the planner more context.

    Returns:
        A normalized migration plan dict.
    """
    return planner_agent.plan_migration(changelog, model=model, code=code)


@tool
def apply_migration(
    plan: dict,
    *,
    source_files: Optional[dict[str, str]] = None,
    manifest: Optional[tuple[str, str]] = None,
) -> dict:
    """Executor stage: apply the plan's scoped refactor, never returning broken code.

    ``@tool`` wrapper over :func:`refactor_tools.apply_migration`. Applies the
    auto-fixable ``libcst`` transforms and manifest version bump, re-parses to
    verify validity (R3.4), records flagged breaking changes (R3.3), and falls
    back to ``guided_pr`` if a fix cannot be applied safely (R3.5). PR creation
    is handled separately by :func:`DepGuardOrchestrator._open_pr`.

    Args:
        plan: The migration plan (from :func:`plan_migration`).
        source_files: Optional ``filename -> source`` mapping to transform.
        manifest: Optional ``(filename, content)`` manifest to version-bump.

    Returns:
        The refactor result dict ``{"package", "strategy", "changes", "flagged",
        "diff", "applied"}``.
    """
    return refactor_tools.apply_migration(
        plan, source_files=source_files, manifest=manifest
    )


def _stub_validator(branch: str) -> dict:
    """Default Validator: a stub returning the ``{status, details}`` shape (task 12 seam).

    GitHub Actions polling is task 12. Until it is wired in, this stub returns
    the exact result shape the loop and state writes expect
    (design.md → ValidatorAgent: branch → ``{status, details}``), with a
    ``skipped`` status so no migration is falsely reported as tested. Task 12
    replaces this by injecting a real validator (see :func:`validate_branch`).
    """
    return {
        "status": VALIDATION_SKIPPED,
        "details": (
            f"validation not yet implemented for branch '{branch}'; "
            "GitHub Actions polling lands in task 12"
        ),
    }


@tool
def validate_branch(branch: str, *, validator: Any = None) -> dict:
    """Validator stage: run/read tests for ``branch`` via an injectable validator.

    Designed around a seam so task 12 can plug in the real GitHub-Actions poller
    without reworking the orchestrator (task 10 scope note). By default it
    delegates to :func:`_stub_validator`; pass ``validator=`` to supply the real
    implementation. Either way it returns the ``{"status", "details"}`` shape
    downstream consumers rely on (design.md → ValidatorAgent).

    Args:
        branch: The branch whose tests should be validated.
        validator: Optional callable ``(branch) -> {"status", "details"}``.
            Defaults to :func:`_stub_validator`.

    Returns:
        A ``{"status", "details"}`` validation result dict.
    """
    run = validator or _stub_validator
    return run(branch)


# =====================================================================
# The orchestrator (tasks 10.1 + 10.2)
# =====================================================================


class DepGuardOrchestrator:
    """Sequence the Monitor→Planner→Executor→Validator loop and persist state.

    Constructed with a target ``repo_name`` (matching the AgentCore entrypoint,
    design.md), the orchestrator runs one full cycle per :meth:`run` call:

    1. **Honor recorded decisions** for existing migrations (R5.4, R5.5).
    2. **Discover** outdated packages (Monitor), **plan** each upgrade
       (Planner), **execute** the scoped refactor + open a PR (Executor),
       **validate** the branch (Validator), and **persist** each as a
       ``pending_review`` migration plus a run log (R6.3).

    Every external boundary is **injected** so the whole loop is unit-testable
    without AWS, network, Bedrock, or GitHub (mirroring the DI style of the
    tools it composes):

    Args:
        repo_name: The ``owner/repo`` slug (also used as the state-table
            partition, and as the GitHub repo for PR operations).
        state: A :class:`~src.state.store.StateStore` (inject a fake table in
            tests). When omitted, one is built from ``table``/``table_name``.
        table: Optional DynamoDB ``Table`` (or fake) forwarded to
            :class:`StateStore` when ``state`` is not given.
        table_name: Optional DynamoDB table name used to build a real store when
            neither ``state`` nor ``table`` is provided.
        planner_model: Injected Nova Pro client (satisfies
            :class:`planner_agent.PlannerModel`). Required to plan.
        changelog_model: Injected Nova Lite client (satisfies
            :class:`changelog_tools.ChangelogModel`). Defaults to
            ``planner_model`` since ``BedrockClient`` satisfies both.
        github_client: Injected PyGithub client (a fake in tests); forwarded to
            ``github_tools`` for PR creation and merge-on-approval.
        github_token: Optional token string (falls back to ``GITHUB_TOKEN``) when
            no ``github_client`` is injected.
        validator: Injected Validator callable ``(branch) -> {"status",
            "details"}``. Defaults to the task-12 stub :func:`_stub_validator`.
        repo_path: Filesystem path the Monitor scans. Defaults to ``repo_name``.
        source_provider: Optional callable ``(plan) -> {filename: source}``
            giving the Executor the source files to transform for a migration.
        manifest_provider: Optional callable ``(plan) -> (filename, content)``
            giving the Executor the manifest to version-bump.
        open_prs: Whether the Executor opens PRs (default ``True``). Set
            ``False`` to run the discover/plan/persist path without touching
            GitHub.
    """

    def __init__(
        self,
        repo_name: str,
        *,
        state: Optional[StateStore] = None,
        table: Any | None = None,
        table_name: Optional[str] = None,
        planner_model: Optional[planner_agent.PlannerModel] = None,
        changelog_model: Optional[changelog_tools.ChangelogModel] = None,
        github_client: Any | None = None,
        github_token: Optional[str] = None,
        validator: Optional[Validator] = None,
        repo_path: Optional[str] = None,
        source_provider: Optional[Callable[[dict], dict[str, str]]] = None,
        manifest_provider: Optional[Callable[[dict], tuple[str, str]]] = None,
        open_prs: bool = True,
    ) -> None:
        self.repo_name = repo_name
        self.repo_path = repo_path or repo_name

        # State store: injected, or built from a table / table_name.
        if state is not None:
            self.state = state
        elif table is not None or table_name is not None:
            self.state = StateStore(table=table, table_name=table_name)
        else:
            self.state = None  # type: ignore[assignment]

        self.planner_model = planner_model
        # ``BedrockClient`` satisfies both PlannerModel and ChangelogModel, so
        # the changelog model defaults to the planner model when not given.
        self.changelog_model = changelog_model or planner_model
        self.github_client = github_client
        self.github_token = github_token
        self.validator = validator or _stub_validator
        self.source_provider = source_provider
        self.manifest_provider = manifest_provider
        self.open_prs = open_prs

    # -- public entrypoint (design.md → AgentCore entrypoint) --------------

    def run(self) -> dict:
        """Run one full agent cycle and return a structured summary.

        Order matters (design.md → Data flow): decisions recorded since the last
        cycle are honored **first** (so an approval merges before we discover
        anything new), then new migrations are discovered, planned, executed,
        validated, and persisted. A run log is written at the end (R6.3).

        Returns:
            A summary dict:
            ``{"repo", "decisions", "migrations", "errors", "outcome"}`` where
            ``decisions`` are the decision-actions applied this cycle and
            ``migrations`` are the migrations discovered/persisted this cycle.
        """
        decisions = self._honor_recorded_decisions()
        discovered, errors = self._discover_and_plan()

        outcome = "completed" if not errors else "completed_with_errors"
        if self.state is not None:
            self.state.put_run_log(
                self.repo_name,
                outcome,
                [m["package"] for m in discovered],
            )

        return {
            "repo": self.repo_name,
            "decisions": decisions,
            "migrations": discovered,
            "errors": errors,
            "outcome": outcome,
        }

    # -- decision honoring (task 10.2, R5.4 / R5.5) ------------------------

    def _honor_recorded_decisions(self) -> list[dict]:
        """Read + act on any human decisions for existing migrations (R5.4/5.5).

        For each stored migration still awaiting action (status
        ``pending_review``), read the recorded decision (``get_decision``). If a
        human recorded ``approved``, merge the PR via
        :func:`github_tools.apply_decision` and transition
        ``pending_review → approved → merged``. If ``ignored``, close/skip the PR
        and transition ``pending_review → ignored → closed``. If ``review`` or no
        decision, leave the PR open and the migration untouched. LusiScan never
        merges without an explicit recorded approval (R5.4); illegal state moves
        are guarded by the store's transition machine (R5.5).

        Returns:
            A list of ``{"migration_id", "package", "target", "decision",
            "pr_state", "merged", "status"}`` action records (empty when nothing
            actionable was recorded).
        """
        if self.state is None:
            return []

        actions: list[dict] = []
        pending = self.state.list_migrations(
            self.repo_name, status=state_store.STATUS_PENDING_REVIEW
        )
        for migration in pending:
            package = migration.get("package")
            target = migration.get("target")
            if not package or not target:
                continue
            mig_id = state_store.migration_id(package, target)
            decision = self.state.get_decision(self.repo_name, mig_id)
            if not decision:
                continue  # review / none: leave the PR open, migration untouched
            action = self._act_on_decision(migration, decision)
            if action is not None:
                actions.append(action)
        return actions

    def _act_on_decision(self, migration: dict, decision: dict) -> Optional[dict]:
        """Act on one recorded decision for one migration (R5.4 / R5.5).

        ``approved`` → merge the PR (only ever here, behind an explicit approval)
        and transition to ``merged``; ``ignored`` → close the PR and transition
        to ``closed``; anything else → leave open. The GitHub side is skipped
        when there is no PR number recorded or PR operations are disabled, but
        the state transition still runs so the store reflects the human's call.
        """
        package = migration["package"]
        target = migration["target"]
        decision_value = decision.get("decision")
        pr_number = migration.get("pr_number")

        # The intermediate + terminal statuses for each decision lane.
        if decision_value == state_store.STATUS_APPROVED:
            intermediate, terminal = (
                state_store.STATUS_APPROVED,
                state_store.STATUS_MERGED,
            )
        elif decision_value == state_store.STATUS_IGNORED:
            intermediate, terminal = (
                state_store.STATUS_IGNORED,
                state_store.STATUS_CLOSED,
            )
        else:
            # ``review`` or unrecognized: leave the PR open, merge nothing (R5.4).
            return {
                "migration_id": state_store.migration_id(package, target),
                "package": package,
                "target": target,
                "decision": decision_value,
                "pr_state": github_tools.PR_STATE_LEFT_OPEN,
                "merged": False,
                "status": migration.get("status"),
            }

        # Act on GitHub (merge on approval / close on ignore). The R5.4 gate
        # lives in github_tools.apply_decision — it merges ONLY on an explicit
        # approval — so we hand it the recorded decision as-is.
        pr_result: dict = {
            "pr_state": github_tools.PR_STATE_LEFT_OPEN,
            "merged": False,
        }
        if self.open_prs and pr_number is not None:
            pr_result = github_tools.apply_decision(
                self.repo_name,
                int(pr_number),
                decision,
                client=self.github_client,
                token=self.github_token,
            )

        # Move the migration through the legal transitions (R5.5). The store
        # rejects anything illegal; a terminal migration simply won't re-run.
        status = migration.get("status")
        try:
            self.state.transition_status(
                self.repo_name, package, target, intermediate
            )
            updated = self.state.transition_status(
                self.repo_name, package, target, terminal
            )
            status = updated.get("status", terminal)
        except InvalidStatusTransition:
            # Already advanced (e.g. a re-run): leave the recorded status as-is.
            pass

        return {
            "migration_id": state_store.migration_id(package, target),
            "package": package,
            "target": target,
            "decision": decision_value,
            "pr_state": pr_result.get("pr_state"),
            "merged": pr_result.get("merged", False),
            "status": status,
        }

    # -- discover / plan / execute / validate / persist (task 10.1/10.2) ---

    def _discover_and_plan(self) -> tuple[list[dict], list[dict]]:
        """Run Monitor→Planner→Executor→Validator for each outdated package.

        Composes the stage functions by calling them directly (the Strands
        convention). Each fully-processed migration is persisted with status
        ``pending_review`` (R6.3) and returned. Monitor errors (e.g. an
        unparseable manifest, R1.4) are collected and returned rather than
        raised.

        Returns:
            ``(migrations, errors)`` — the migrations discovered/persisted this
            cycle and any monitor errors recorded along the way.
        """
        scan = scan_packages(self.repo_path)
        errors = list(scan.get("errors", []))
        outdated = scan.get("outdated", [])

        migrations: list[dict] = []
        for pkg in outdated:
            migration = self._process_package(pkg)
            if migration is not None:
                migrations.append(migration)
        return migrations, errors

    def _process_package(self, pkg: dict) -> Optional[dict]:
        """Plan, execute, validate, and persist one outdated package.

        Returns the persisted migration record, or ``None`` when planning cannot
        proceed (e.g. no planner model injected).
        """
        package = pkg["name"]
        current = pkg["current"]
        target = pkg["latest"]

        if self.planner_model is None:
            # Nothing to reason with — skip rather than fabricate a plan.
            return None

        # Don't clobber a migration that has already moved past pending_review
        # (e.g. one merged/closed by a decision honored earlier this cycle, or
        # awaiting a human decision from a prior cycle). Re-writing it as a fresh
        # ``pending_review`` would reset the human-in-the-loop state machine and
        # re-open a settled migration — so we leave the existing record intact
        # and skip re-processing (R5.5). A migration is only (re)processed when
        # it does not yet exist.
        if self.state is not None:
            existing = self.state.get_migration(self.repo_name, package, target)
            if existing is not None:
                return existing

        # Planner: changelog → structured plan.
        changelog = fetch_changelog(
            package, current, target, model=self.changelog_model
        )
        plan = plan_migration(changelog, model=self.planner_model)

        # Executor: apply the scoped refactor (version bump + auto-fixes).
        source_files = (
            self.source_provider(plan) if self.source_provider is not None else None
        )
        manifest = (
            self.manifest_provider(plan)
            if self.manifest_provider is not None
            else None
        )
        refactor = apply_migration(
            plan, source_files=source_files, manifest=manifest
        )

        # Carry the planner's confidence onto the executor result so the PR body
        # renders the right tier (github_tools.build_pr_content reads it).
        migration_view = {
            **refactor,
            "current": current,
            "target": target,
            "confidence": plan.get("confidence"),
        }

        # Executor: open the PR (unless disabled / no changes to commit).
        pr = self._open_pr(migration_view)

        # Validator: run/read tests for the branch (task-12 seam).
        branch = pr.get("branch") if pr else None
        validation = validate_branch(branch or "", validator=self.validator)

        # Persist as a pending_review migration (R6.3, design.md → step 6).
        return self._persist_migration(
            package=package,
            current=current,
            target=target,
            plan=plan,
            refactor=refactor,
            pr=pr,
            validation=validation,
        )

    def _open_pr(self, migration_view: dict) -> dict:
        """Open a PR for a processed migration, or return an empty result.

        Skips GitHub entirely when PR creation is disabled (``open_prs=False``)
        or there are no changes to commit — returning a benign empty result so
        the loop still persists the migration and its plan.
        """
        if not self.open_prs:
            return {}
        if not migration_view.get("changes"):
            return {}
        return github_tools.create_migration_pr(
            self.repo_name,
            migration_view,
            client=self.github_client,
            token=self.github_token,
        )

    def _persist_migration(
        self,
        *,
        package: str,
        current: str,
        target: str,
        plan: dict,
        refactor: dict,
        pr: dict,
        validation: dict,
    ) -> dict:
        """Write a discovered migration to DynamoDB as ``pending_review`` (R6.3).

        Assembles the migration item from the plan (confidence / strategy / risk
        / reasoning), the executor result (diff / resolved strategy / flagged
        changes), the PR (url / number), and the validation summary, then writes
        it via :meth:`StateStore.put_migration` with the initial
        ``pending_review`` status (design.md → Data flow step 6). Returns the
        item that was written (or would be written, when no store is configured).
        """
        migration_item: dict[str, Any] = {
            "package": package,
            "current": current,
            "from": current,
            "target": target,
            "to": target,
            "confidence": plan.get("confidence"),
            "risk": plan.get("estimated_risk"),
            "strategy": refactor.get("strategy", plan.get("strategy")),
            "reasoning": plan.get("reasoning", ""),
            "breaking_changes": plan.get("breaking_changes", []),
            "flagged": refactor.get("flagged", []),
            "applied": refactor.get("applied", []),
            "diff": refactor.get("diff", ""),
            "pr_url": pr.get("pr_url"),
            "pr_number": pr.get("pr_number"),
            "test_summary": validation,
        }

        if self.state is None:
            # No store configured: return the item without persisting so the
            # pure loop is still observable/testable.
            return {
                **migration_item,
                "status": state_store.STATUS_PENDING_REVIEW,
            }

        return self.state.put_migration(
            self.repo_name,
            migration_item,
            status=state_store.STATUS_PENDING_REVIEW,
        )
