"""Pydantic 1.x-style models.

These use pydantic v1 APIs on purpose. The LusiScan human-in-the-loop demo is a
pydantic 1 -> 2 major upgrade, where these are the constructs that must change:

- `from pydantic import BaseModel, validator`  ->  `field_validator` in v2
- class-based `Config`                         ->  `model_config` in v2
- `.dict()`                                     ->  `.model_dump()` in v2
"""

from pydantic import BaseModel, validator


class User(BaseModel):
    """A minimal user model using pydantic 1.x idioms."""

    id: int
    name: str
    email: str

    class Config:
        # v1-style config block; becomes `model_config` in v2.
        anystr_strip_whitespace = True

    @validator("email")
    def email_must_contain_at(cls, value: str) -> str:
        # v1-style `@validator`; becomes `@field_validator` in v2.
        if "@" not in value:
            raise ValueError("email must contain @")
        return value


def user_summary(user: User) -> dict:
    """Return a plain dict for a user using the v1 `.dict()` API."""
    return user.dict()
