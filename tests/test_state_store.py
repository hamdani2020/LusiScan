"""Unit tests for the DynamoDB state store (tasks 9.1 and 9.2).

Task 9.2 is the focus: the migration ``status`` state machine
(``pending_review`` → ``approved``/``ignored`` → ``merged``/``closed``). The
tests assert every legal transition is honored and every illegal one — skips,
no-ops, unknown targets, and moves out of a terminal state — is rejected with
:class:`~src.state.store.InvalidStatusTransition` while leaving stored state
untouched (R5.5). Task 9.1's write/read paths for the Migration, Decision, and
Run-log entities are covered too (R6.3).

No real DynamoDB (or ``boto3``) is used: a small in-memory fake ``Table``
records ``put_item``/``get_item``/``update_item``/``query`` calls and replays
items, mirroring the dependency-injected-fake convention used by
``tests/test_github_tools.py``. This keeps the state machine fully unit-testable
without any AWS access.

_Requirements: 6.3 (persist Migration/Decision/Run-log state to DynamoDB), 5.5
(record a decision and act on it by transitioning status on the next cycle)._
"""

from __future__ import annotations

import pytest

from src.state import store as st
from src.state.store import InvalidStatusTransition, StateStore


# --- Fake DynamoDB table --------------------------------------------------


class FakeTable:
    """An in-memory stand-in for a boto3 DynamoDB ``Table``.

    Items are keyed by their ``(pk, sk)`` tuple. It implements just the surface
    the store uses: ``put_item``, ``get_item``, ``update_item`` (SET status,
    ``ALL_NEW``), and a ``begins_with`` ``query``. Because ``boto3`` is not
    installed in the test environment, the store's ``_query_prefix`` falls back
    to calling ``query(pk=..., sk_prefix=...)``, which this fake accepts.
    """

    def __init__(self) -> None:
        self.items: dict[tuple[str, str], dict] = {}
        self.put_calls: list[dict] = []
        self.update_calls: list[dict] = []

    def put_item(self, *, Item: dict) -> None:
        self.put_calls.append(Item)
        self.items[(Item["pk"], Item["sk"])] = dict(Item)

    def get_item(self, *, Key: dict) -> dict:
        item = self.items.get((Key["pk"], Key["sk"]))
        return {"Item": dict(item)} if item is not None else {}

    def update_item(
        self,
        *,
        Key: dict,
        UpdateExpression: str,
        ExpressionAttributeNames: dict,
        ExpressionAttributeValues: dict,
        ReturnValues: str = "NONE",
    ) -> dict:
        self.update_calls.append({"Key": Key, "values": ExpressionAttributeValues})
        item = self.items[(Key["pk"], Key["sk"])]
        # Apply the single "SET #s = :to" our store issues.
        attr = ExpressionAttributeNames["#s"]
        item[attr] = ExpressionAttributeValues[":to"]
        if ReturnValues == "ALL_NEW":
            return {"Attributes": dict(item)}
        return {}

    def query(self, *, pk: str, sk_prefix: str) -> dict:
        # Matches the no-boto3 fallback path in StateStore._query_prefix.
        matched = [
            dict(item)
            for (ipk, isk), item in self.items.items()
            if ipk == pk and isk.startswith(sk_prefix)
        ]
        return {"Items": matched}


@pytest.fixture
def table() -> FakeTable:
    return FakeTable()


@pytest.fixture
def state(table: FakeTable) -> StateStore:
    return StateStore(table=table)


REPO = "acme/demo"


def _seed_migration(state: StateStore, **overrides) -> dict:
    """Write a baseline ``pending_review`` migration and return the item."""
    migration = {
        "package": "pydantic",
        "current": "1.10.13",
        "target": "2.9.2",
        "confidence": "low",
        "risk": "medium",
        "strategy": "guided_pr",
        **overrides,
    }
    return state.put_migration(REPO, migration)


# --- Construction ---------------------------------------------------------


class TestConstruction:
    def test_requires_table_or_table_name(self) -> None:
        with pytest.raises(ValueError):
            StateStore()

    def test_injected_table_is_used(self, table: FakeTable) -> None:
        store = StateStore(table=table)
        assert store._table is table


# --- is_valid_transition: the pure state machine (task 9.2, R5.5) ---------


class TestIsValidTransition:
    @pytest.mark.parametrize(
        "from_status,to_status",
        [
            (st.STATUS_PENDING_REVIEW, st.STATUS_APPROVED),
            (st.STATUS_PENDING_REVIEW, st.STATUS_IGNORED),
            (st.STATUS_APPROVED, st.STATUS_MERGED),
            (st.STATUS_IGNORED, st.STATUS_CLOSED),
        ],
    )
    def test_legal_transitions_are_allowed(self, from_status, to_status) -> None:
        assert st.is_valid_transition(from_status, to_status) is True

    @pytest.mark.parametrize(
        "from_status,to_status",
        [
            # Skips that bypass the human decision.
            (st.STATUS_PENDING_REVIEW, st.STATUS_MERGED),
            (st.STATUS_PENDING_REVIEW, st.STATUS_CLOSED),
            # Crossed wires between the two lanes.
            (st.STATUS_APPROVED, st.STATUS_CLOSED),
            (st.STATUS_IGNORED, st.STATUS_MERGED),
            (st.STATUS_APPROVED, st.STATUS_IGNORED),
            # No-op self transitions.
            (st.STATUS_PENDING_REVIEW, st.STATUS_PENDING_REVIEW),
            (st.STATUS_APPROVED, st.STATUS_APPROVED),
            # Out of a terminal state.
            (st.STATUS_MERGED, st.STATUS_APPROVED),
            (st.STATUS_MERGED, st.STATUS_CLOSED),
            (st.STATUS_CLOSED, st.STATUS_MERGED),
            # Unknown statuses on either side.
            ("bogus", st.STATUS_APPROVED),
            (st.STATUS_PENDING_REVIEW, "bogus"),
        ],
    )
    def test_illegal_transitions_are_rejected(self, from_status, to_status) -> None:
        assert st.is_valid_transition(from_status, to_status) is False


# --- transition_status: enforced against stored state (task 9.2) ----------


class TestTransitionStatusHappyPaths:
    def test_pending_to_approved_to_merged(self, state: StateStore) -> None:
        _seed_migration(state)

        approved = state.transition_status(REPO, "pydantic", "2.9.2", st.STATUS_APPROVED)
        assert approved["status"] == st.STATUS_APPROVED

        merged = state.transition_status(REPO, "pydantic", "2.9.2", st.STATUS_MERGED)
        assert merged["status"] == st.STATUS_MERGED

        # Persisted, not just returned.
        assert state.get_migration(REPO, "pydantic", "2.9.2")["status"] == st.STATUS_MERGED

    def test_pending_to_ignored_to_closed(self, state: StateStore) -> None:
        _seed_migration(state)

        ignored = state.transition_status(REPO, "pydantic", "2.9.2", st.STATUS_IGNORED)
        assert ignored["status"] == st.STATUS_IGNORED

        closed = state.transition_status(REPO, "pydantic", "2.9.2", st.STATUS_CLOSED)
        assert closed["status"] == st.STATUS_CLOSED
        assert state.get_migration(REPO, "pydantic", "2.9.2")["status"] == st.STATUS_CLOSED


class TestTransitionStatusRejections:
    def test_skip_pending_to_merged_is_rejected(self, state: StateStore) -> None:
        _seed_migration(state)
        with pytest.raises(InvalidStatusTransition) as exc:
            state.transition_status(REPO, "pydantic", "2.9.2", st.STATUS_MERGED)
        assert exc.value.from_status == st.STATUS_PENDING_REVIEW
        assert exc.value.to_status == st.STATUS_MERGED

    def test_rejected_transition_leaves_status_unchanged(self, state: StateStore) -> None:
        _seed_migration(state)
        with pytest.raises(InvalidStatusTransition):
            state.transition_status(REPO, "pydantic", "2.9.2", st.STATUS_CLOSED)
        # Nothing was written; the migration is still pending.
        stored = state.get_migration(REPO, "pydantic", "2.9.2")
        assert stored["status"] == st.STATUS_PENDING_REVIEW

    def test_cannot_move_out_of_terminal_state(self, state: StateStore) -> None:
        _seed_migration(state)
        state.transition_status(REPO, "pydantic", "2.9.2", st.STATUS_APPROVED)
        state.transition_status(REPO, "pydantic", "2.9.2", st.STATUS_MERGED)
        # merged is terminal.
        with pytest.raises(InvalidStatusTransition):
            state.transition_status(REPO, "pydantic", "2.9.2", st.STATUS_APPROVED)

    def test_crossed_lane_is_rejected(self, state: StateStore) -> None:
        # approved must not jump to closed (the ignored lane's terminal).
        _seed_migration(state)
        state.transition_status(REPO, "pydantic", "2.9.2", st.STATUS_APPROVED)
        with pytest.raises(InvalidStatusTransition):
            state.transition_status(REPO, "pydantic", "2.9.2", st.STATUS_CLOSED)

    def test_transition_missing_migration_raises_keyerror(self, state: StateStore) -> None:
        with pytest.raises(KeyError):
            state.transition_status(REPO, "nope", "9.9.9", st.STATUS_APPROVED)


# --- apply_decision: record + transition in one step (R5.5) ---------------


class TestApplyDecision:
    def test_approved_records_decision_and_transitions(
        self, state: StateStore, table: FakeTable
    ) -> None:
        _seed_migration(state)
        updated = state.apply_decision(REPO, "pydantic", "2.9.2", st.STATUS_APPROVED)

        assert updated["status"] == st.STATUS_APPROVED
        # A decision item was recorded referencing the migration.
        mig_id = st.migration_id("pydantic", "2.9.2")
        decision = state.get_decision(REPO, mig_id)
        assert decision is not None
        assert decision["decision"] == st.STATUS_APPROVED
        assert decision["migration_id"] == mig_id

    def test_ignored_records_decision_and_transitions(self, state: StateStore) -> None:
        _seed_migration(state)
        updated = state.apply_decision(REPO, "pydantic", "2.9.2", st.STATUS_IGNORED)
        assert updated["status"] == st.STATUS_IGNORED

    def test_apply_decision_on_terminal_migration_is_rejected(
        self, state: StateStore
    ) -> None:
        _seed_migration(state)
        state.apply_decision(REPO, "pydantic", "2.9.2", st.STATUS_IGNORED)
        state.transition_status(REPO, "pydantic", "2.9.2", st.STATUS_CLOSED)
        with pytest.raises(InvalidStatusTransition):
            state.apply_decision(REPO, "pydantic", "2.9.2", st.STATUS_APPROVED)


# --- put_decision validation ----------------------------------------------


class TestPutDecision:
    def test_rejects_unknown_decision_value(self, state: StateStore) -> None:
        with pytest.raises(ValueError):
            state.put_decision(REPO, "MIG#x#1", "maybe")

    def test_records_timestamp_when_omitted(self, state: StateStore) -> None:
        item = state.put_decision(REPO, "MIG#pydantic#2.9.2", st.STATUS_APPROVED)
        assert item["timestamp"]  # non-empty ISO string
        assert item["decision"] == st.STATUS_APPROVED


# --- Migration entity write/read (task 9.1, R6.3) -------------------------


class TestMigrationEntity:
    def test_put_migration_defaults_to_pending_review(self, state: StateStore) -> None:
        item = _seed_migration(state)
        assert item["status"] == st.STATUS_PENDING_REVIEW
        assert item["pk"] == "REPO#acme/demo"
        assert item["sk"] == "MIG#pydantic#2.9.2"
        assert item["entity"] == "migration"

    def test_put_migration_preserves_attributes(self, state: StateStore) -> None:
        item = _seed_migration(state, diff="--- a\n+++ b\n", pr_url="http://pr/1")
        assert item["diff"] == "--- a\n+++ b\n"
        assert item["pr_url"] == "http://pr/1"
        assert item["confidence"] == "low"

    def test_put_migration_missing_package_raises(self, state: StateStore) -> None:
        with pytest.raises(KeyError):
            state.put_migration(REPO, {"target": "2.9.2"})

    def test_put_migration_unknown_status_raises(self, state: StateStore) -> None:
        with pytest.raises(ValueError):
            state.put_migration(
                REPO, {"package": "x", "target": "1"}, status="bogus"
            )

    def test_get_missing_migration_returns_none(self, state: StateStore) -> None:
        assert state.get_migration(REPO, "nope", "0") is None

    def test_list_migrations_filters_by_status(self, state: StateStore) -> None:
        _seed_migration(state)  # pending
        state.put_migration(
            REPO, {"package": "requests", "target": "2.32.3"}, status=st.STATUS_APPROVED
        )
        pending = state.list_migrations(REPO, status=st.STATUS_PENDING_REVIEW)
        assert len(pending) == 1
        assert pending[0]["package"] == "pydantic"
        assert len(state.list_migrations(REPO)) == 2


# --- Run-log entity (task 9.1, R6.3) --------------------------------------


class TestRunLogEntity:
    def test_put_run_log_with_package_list(self, state: StateStore) -> None:
        item = state.put_run_log(
            REPO, "completed", ["pydantic", "requests"], timestamp="2024-01-01T00:00:00Z"
        )
        assert item["outcome"] == "completed"
        assert item["packages_processed"] == ["pydantic", "requests"]
        assert item["sk"] == "RUN#2024-01-01T00:00:00Z"

    def test_put_run_log_with_count(self, state: StateStore) -> None:
        item = state.put_run_log(REPO, "completed", 2, timestamp="2024-01-01T00:00:00Z")
        assert item["packages_processed"] == 2

    def test_list_run_logs_returns_only_run_items(self, state: StateStore) -> None:
        _seed_migration(state)  # a MIG# item that must not show up
        state.put_run_log(REPO, "completed", 1, timestamp="2024-01-01T00:00:00Z")
        state.put_run_log(REPO, "completed", 0, timestamp="2024-01-02T00:00:00Z")
        logs = state.list_run_logs(REPO)
        assert len(logs) == 2
        assert all(item["entity"] == "run_log" for item in logs)
