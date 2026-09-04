"""State layer: DynamoDB single-table store."""

from src.state.store import (
    INITIAL_STATUS,
    STATUS_APPROVED,
    STATUS_CLOSED,
    STATUS_IGNORED,
    STATUS_MERGED,
    STATUS_PENDING_REVIEW,
    TERMINAL_STATUSES,
    VALID_STATUSES,
    InvalidStatusTransition,
    StateStore,
    is_valid_transition,
)

__all__ = [
    "INITIAL_STATUS",
    "STATUS_APPROVED",
    "STATUS_CLOSED",
    "STATUS_IGNORED",
    "STATUS_MERGED",
    "STATUS_PENDING_REVIEW",
    "TERMINAL_STATUSES",
    "VALID_STATUSES",
    "InvalidStatusTransition",
    "StateStore",
    "is_valid_transition",
]
