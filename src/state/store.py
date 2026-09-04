"""DynamoDB single-table state store (task 9.1).

This module is the single integration point between the LusiScan agent loop and
the Streamlit control panel (design.md → "State as the contract"). Neither side
calls the other directly; they communicate only through this table. The agent
writes migrations and run logs and transitions migration ``status``; the
Streamlit app writes human decisions; on its next cycle the agent reads those
decisions and moves the migration forward (R5.5, R6.3).

Single-table design (design.md → State layer), keyed by ``pk`` / ``sk``:

===========  =====================  ==========================  ===============
Entity       pk                     sk                          notable attrs
===========  =====================  ==========================  ===============
Migration    ``REPO#<repo>``        ``MIG#<package>#<target>``  status, diff, …
Decision     ``REPO#<repo>``        ``DEC#<migration_id>``      decision, ts
Run log      ``REPO#<repo>``        ``RUN#<ts>``                outcome, packages
===========  =====================  ==========================  ===============

Migration ``status`` is a small state machine (design.md → State layer):

    pending_review ──approved──▶ approved ──▶ merged
                 └───ignored───▶ ignored  ──▶ closed

Only those transitions are legal. :func:`transition_status` (and the
convenience wrappers) enforce the machine and reject anything else, so an
out-of-order or nonsensical move (e.g. ``pending_review`` → ``merged``, or any
move out of a terminal state) raises :class:`InvalidStatusTransition` rather
than silently corrupting state (R5.5).

Secrets / clients (design.md → Security; mirrors ``github_tools.get_client``):
this module never hardcodes AWS credentials or a table name in source. A caller
either injects an already-built DynamoDB ``Table`` resource (what tests inject
as a fake, and what the orchestrator can inject after resolving config at
runtime) or passes a table name and lets :class:`StateStore` build the resource
via ``boto3`` — which is imported lazily so the module stays importable, and the
pure state-machine logic stays unit-testable, without ``boto3`` installed.

Design references:
- design.md → State layer (DynamoDB): single-table schema and status machine.
- design.md → Data flow steps 6-8: write ``pending_review`` → human decision →
  next cycle transitions status.
- requirements.md → R6.3 (persist state to DynamoDB), R5.5 (act on the recorded
  decision on the next cycle).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Mapping


# --- Status state machine (design.md → State layer) -----------------------

# The lifecycle of a migration. ``pending_review`` is where every migration
# starts; ``merged`` and ``closed`` are terminal (no further transitions).
STATUS_PENDING_REVIEW = "pending_review"
STATUS_APPROVED = "approved"
STATUS_IGNORED = "ignored"
STATUS_MERGED = "merged"
STATUS_CLOSED = "closed"

# The status a migration is created with (design.md → Data flow step 6).
INITIAL_STATUS = STATUS_PENDING_REVIEW

# Terminal statuses: once reached, no further transition is legal.
TERMINAL_STATUSES = frozenset({STATUS_MERGED, STATUS_CLOSED})

# The legal transition map: current status -> set of statuses reachable from it.
#
#   pending_review ──approved──▶ approved ──▶ merged
#                └────ignored──▶ ignored  ──▶ closed
#
# Any (from, to) pair not represented here is rejected by
# :func:`is_valid_transition` / :func:`transition_status` (R5.5).
_ALLOWED_TRANSITIONS: dict[str, frozenset[str]] = {
    STATUS_PENDING_REVIEW: frozenset({STATUS_APPROVED, STATUS_IGNORED}),
    STATUS_APPROVED: frozenset({STATUS_MERGED}),
    STATUS_IGNORED: frozenset({STATUS_CLOSED}),
    STATUS_MERGED: frozenset(),
    STATUS_CLOSED: frozenset(),
}

# The full set of recognized statuses (keys of the transition map).
VALID_STATUSES = frozenset(_ALLOWED_TRANSITIONS)


class InvalidStatusTransition(ValueError):
    """Raised when a requested migration status transition is not allowed.

    Carries the offending ``from``/``to`` statuses so callers (and the
    orchestrator's logs) can see exactly which move was rejected.
    """

    def __init__(self, from_status: str, to_status: str) -> None:
        self.from_status = from_status
        self.to_status = to_status
        super().__init__(
            f"illegal migration status transition: {from_status!r} -> "
            f"{to_status!r} (allowed from {from_status!r}: "
            f"{sorted(_ALLOWED_TRANSITIONS.get(from_status, frozenset()))})"
        )


def is_valid_transition(from_status: str, to_status: str) -> bool:
    """Return whether ``from_status`` → ``to_status`` is a legal move.

    A move is legal only when it appears in :data:`_ALLOWED_TRANSITIONS`.
    Unknown statuses, no-op self transitions, and any move out of a terminal
    state (``merged`` / ``closed``) all return ``False``.

    Args:
        from_status: The migration's current status.
        to_status: The status being requested.

    Returns:
        ``True`` if the transition is permitted, ``False`` otherwise.
    """
    return to_status in _ALLOWED_TRANSITIONS.get(from_status, frozenset())


# --- Key helpers (single-table design) ------------------------------------


def repo_pk(repo: str) -> str:
    """Build the partition key for a repository (``REPO#<repo>``)."""
    return f"REPO#{repo}"


def migration_sk(package: str, target: str) -> str:
    """Build the sort key for a migration (``MIG#<package>#<target>``)."""
    return f"MIG#{package}#{target}"


def migration_id(package: str, target: str) -> str:
    """Build the migration id used to tie a decision to its migration.

    The id is the migration's sort key (``MIG#<package>#<target>``) so a
    decision (``DEC#<migration_id>``) unambiguously references one migration.
    """
    return migration_sk(package, target)


def decision_sk(mig_id: str) -> str:
    """Build the sort key for a decision (``DEC#<migration_id>``)."""
    return f"DEC#{mig_id}"


def run_log_sk(ts: str) -> str:
    """Build the sort key for a run log (``RUN#<ts>``)."""
    return f"RUN#{ts}"


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string (seconds precision)."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# --- The store ------------------------------------------------------------


class StateStore:
    """Read/write LusiScan state in the DynamoDB single table.

    The store is constructed with either an injected ``table`` resource (a
    boto3 ``Table``, or a fake exposing ``put_item``/``get_item``/``query``/
    ``update_item``) or a ``table_name`` that it resolves to a real boto3
    ``Table`` lazily. Injecting a fake keeps the whole store — including the
    status state machine — unit-testable without ``boto3`` or a live table.

    Args:
        table: A pre-built DynamoDB ``Table`` (or compatible fake). Preferred
            for tests and for runtime injection.
        table_name: The DynamoDB table name to resolve via ``boto3`` when no
            ``table`` is injected.
        region_name: Optional AWS region passed to ``boto3`` when building the
            resource.

    Raises:
        ValueError: If neither ``table`` nor ``table_name`` is provided.
    """

    def __init__(
        self,
        *,
        table: Any | None = None,
        table_name: str | None = None,
        region_name: str | None = None,
    ) -> None:
        if table is None and table_name is None:
            raise ValueError("provide either a table resource or a table_name")
        self._table = table if table is not None else _build_table(
            table_name, region_name=region_name
        )

    # -- Migration entity (R6.3) -------------------------------------------

    def put_migration(
        self,
        repo: str,
        migration: "Mapping[str, Any]",
        *,
        status: str | None = None,
    ) -> dict[str, Any]:
        """Write (create/overwrite) a migration item for ``repo``.

        The item's ``status`` is taken from the ``status`` argument, else from
        ``migration["status"]``, else defaults to :data:`INITIAL_STATUS`
        (``pending_review``) — the state a freshly discovered migration starts
        in (design.md → Data flow step 6). The status is validated to be a
        recognized status; the transition machine is only enforced on *updates*
        (see :meth:`transition_status`).

        Args:
            repo: The ``owner/repo`` slug the migration belongs to.
            migration: The migration attributes. Must include ``package`` and
                ``target`` (used to build the sort key); may include ``current``
                (aka ``from``), ``to``, ``confidence``, ``risk``, ``strategy``,
                ``reasoning``, ``diff``, ``pr_url``, ``test_summary``.
            status: Optional explicit initial status.

        Returns:
            The full item dict that was written (including ``pk``/``sk``).

        Raises:
            KeyError: If ``package`` or ``target`` is missing.
            ValueError: If the resolved status is not a recognized status.
        """
        package = migration["package"]
        target = migration["target"]
        resolved_status = status or migration.get("status") or INITIAL_STATUS
        if resolved_status not in VALID_STATUSES:
            raise ValueError(f"unknown migration status: {resolved_status!r}")

        item: dict[str, Any] = {
            "pk": repo_pk(repo),
            "sk": migration_sk(package, target),
            "entity": "migration",
            **dict(migration),
            "status": resolved_status,
        }
        self._table.put_item(Item=item)
        return item

    def get_migration(
        self, repo: str, package: str, target: str
    ) -> dict[str, Any] | None:
        """Read a single migration item, or ``None`` if it does not exist."""
        response = self._table.get_item(
            Key={"pk": repo_pk(repo), "sk": migration_sk(package, target)}
        )
        return response.get("Item")

    def list_migrations(
        self, repo: str, *, status: str | None = None
    ) -> list[dict[str, Any]]:
        """List a repo's migrations, optionally filtered to one ``status``.

        Migrations are the items whose sort key begins with ``MIG#``. When
        ``status`` is given, only migrations currently in that status are
        returned (used by the Streamlit app to list ``pending_review`` items,
        R7.1).

        Args:
            repo: The ``owner/repo`` slug to list migrations for.
            status: Optional status filter.

        Returns:
            A list of migration item dicts (possibly empty).
        """
        items = self._query_prefix(repo_pk(repo), "MIG#")
        if status is None:
            return items
        return [item for item in items if item.get("status") == status]

    # -- Status transitions (the R5.5 state machine) -----------------------

    def transition_status(
        self, repo: str, package: str, target: str, to_status: str
    ) -> dict[str, Any]:
        """Move a migration to ``to_status``, enforcing the state machine.

        The migration's current status is read first; the requested move is
        checked against :data:`_ALLOWED_TRANSITIONS`. Only a legal move
        (``pending_review`` → ``approved``/``ignored``; ``approved`` →
        ``merged``; ``ignored`` → ``closed``) is written. Anything else — an
        unknown target, a skip (``pending_review`` → ``merged``), a no-op self
        transition, or any move out of a terminal state — raises
        :class:`InvalidStatusTransition` and writes nothing (R5.5).

        Args:
            repo: The ``owner/repo`` slug.
            package: The migration's package.
            target: The migration's target version.
            to_status: The status to move to.

        Returns:
            The updated migration item.

        Raises:
            KeyError: If the migration does not exist.
            InvalidStatusTransition: If the move is not permitted.
        """
        current = self.get_migration(repo, package, target)
        if current is None:
            raise KeyError(
                f"no migration for {repo} {package} {target} to transition"
            )
        from_status = current.get("status", INITIAL_STATUS)
        if not is_valid_transition(from_status, to_status):
            raise InvalidStatusTransition(from_status, to_status)

        response = self._table.update_item(
            Key={"pk": repo_pk(repo), "sk": migration_sk(package, target)},
            UpdateExpression="SET #s = :to",
            ExpressionAttributeNames={"#s": "status"},
            ExpressionAttributeValues={":to": to_status},
            ReturnValues="ALL_NEW",
        )
        updated = response.get("Attributes")
        if updated is not None:
            return updated
        # Fallback for stores that don't echo attributes: reflect locally.
        current["status"] = to_status
        return current

    # -- Decision entity (R5.5) --------------------------------------------

    def put_decision(
        self,
        repo: str,
        mig_id: str,
        decision: str,
        *,
        timestamp: str | None = None,
    ) -> dict[str, Any]:
        """Record a human decision (``approved`` / ``ignored``) for a migration.

        This is what the Streamlit control panel writes (R7.3); the orchestrator
        reads it on the next cycle and calls :meth:`transition_status` to move
        the referenced migration forward (R5.5). The decision value is validated
        to be one of the recorded human decisions.

        Args:
            repo: The ``owner/repo`` slug.
            mig_id: The migration id the decision applies to (see
                :func:`migration_id`).
            decision: ``"approved"`` or ``"ignored"``.
            timestamp: Optional ISO-8601 timestamp; defaults to now (UTC).

        Returns:
            The decision item that was written.

        Raises:
            ValueError: If ``decision`` is not ``approved`` or ``ignored``.
        """
        if decision not in (STATUS_APPROVED, STATUS_IGNORED):
            raise ValueError(
                f"decision must be {STATUS_APPROVED!r} or {STATUS_IGNORED!r}, "
                f"got {decision!r}"
            )
        item = {
            "pk": repo_pk(repo),
            "sk": decision_sk(mig_id),
            "entity": "decision",
            "migration_id": mig_id,
            "decision": decision,
            "timestamp": timestamp or _now_iso(),
        }
        self._table.put_item(Item=item)
        return item

    def get_decision(self, repo: str, mig_id: str) -> dict[str, Any] | None:
        """Read the recorded decision for a migration, or ``None`` if none."""
        response = self._table.get_item(
            Key={"pk": repo_pk(repo), "sk": decision_sk(mig_id)}
        )
        return response.get("Item")

    def apply_decision(
        self, repo: str, package: str, target: str, decision: str
    ) -> dict[str, Any]:
        """Record a decision and transition the migration in one step (R5.5).

        Convenience for the human-in-the-loop contract: ``approved`` records the
        decision and moves ``pending_review`` → ``approved``; ``ignored`` records
        it and moves ``pending_review`` → ``ignored``. The transition machine
        still applies, so applying a decision to an already-terminal migration
        raises :class:`InvalidStatusTransition`.

        Args:
            repo: The ``owner/repo`` slug.
            package: The migration's package.
            target: The migration's target version.
            decision: ``"approved"`` or ``"ignored"``.

        Returns:
            The updated migration item.
        """
        mig_id = migration_id(package, target)
        self.put_decision(repo, mig_id, decision)
        return self.transition_status(repo, package, target, decision)

    # -- Run log entity (R6.3) ---------------------------------------------

    def put_run_log(
        self,
        repo: str,
        outcome: str,
        packages_processed: Iterable[str] | int,
        *,
        timestamp: str | None = None,
    ) -> dict[str, Any]:
        """Append a run-log item recording the outcome of one agent cycle.

        Args:
            repo: The ``owner/repo`` slug the run targeted.
            outcome: A short outcome string (e.g. ``"completed"``).
            packages_processed: Either the list of package names processed or a
                count of them.
            timestamp: Optional ISO-8601 timestamp; defaults to now (UTC).

        Returns:
            The run-log item that was written.
        """
        ts = timestamp or _now_iso()
        item = {
            "pk": repo_pk(repo),
            "sk": run_log_sk(ts),
            "entity": "run_log",
            "outcome": outcome,
            "packages_processed": (
                list(packages_processed)
                if not isinstance(packages_processed, int)
                else packages_processed
            ),
            "timestamp": ts,
        }
        self._table.put_item(Item=item)
        return item

    def list_run_logs(self, repo: str) -> list[dict[str, Any]]:
        """List a repo's run-log items (sort keys beginning with ``RUN#``)."""
        return self._query_prefix(repo_pk(repo), "RUN#")

    # -- Internal query helper ---------------------------------------------

    def _query_prefix(self, pk: str, sk_prefix: str) -> list[dict[str, Any]]:
        """Query all items under ``pk`` whose sort key begins with ``sk_prefix``.

        Uses a ``begins_with`` key-condition query. ``boto3``'s ``Key`` condition
        builder is imported lazily so the module remains importable without
        ``boto3``; injected fakes may implement ``query`` however they like.
        """
        try:  # Real boto3 path.
            from boto3.dynamodb.conditions import Key

            key_condition = Key("pk").eq(pk) & Key("sk").begins_with(sk_prefix)
            response = self._table.query(KeyConditionExpression=key_condition)
        except ImportError:
            # No boto3 available: hand the fake the raw parameters it expects.
            response = self._table.query(pk=pk, sk_prefix=sk_prefix)
        return list(response.get("Items", []))


# --- boto3 resource construction (lazy; no committed secrets) -------------


def _build_table(table_name: str, *, region_name: str | None = None) -> Any:
    """Resolve ``table_name`` to a boto3 DynamoDB ``Table`` resource.

    ``boto3`` is imported here (not at module import time) so the state machine
    and key helpers are usable, and the store is unit-testable via an injected
    fake table, without ``boto3`` installed. Credentials/region come from the
    standard AWS resolution chain — never hardcoded (design.md → Security).
    """
    import boto3  # noqa: PLC0415 - intentional lazy import

    resource = boto3.resource("dynamodb", region_name=region_name)
    return resource.Table(table_name)
