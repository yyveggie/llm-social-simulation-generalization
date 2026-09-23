from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class ScoreResult(BaseModel):

    model_config = ConfigDict(extra="forbid")
    score: float = Field(ge=0, le=100, allow_inf_nan=False)


def valid_score(value: object) -> bool:
    try:
        ScoreResult(score=value)
        return True
    except (ValidationError, TypeError):
        return False
