from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class AmountChoice(BaseModel):

    model_config = ConfigDict(extra="forbid")
    amount: float = Field(ge=0, allow_inf_nan=False)


class RatingChoice(BaseModel):

    model_config = ConfigDict(extra="forbid")
    rating: int = Field(ge=1, le=5, strict=True)


class CardChoice(BaseModel):

    model_config = ConfigDict(extra="forbid")
    card: Literal["Push", "Pull"]


class ABChoice(BaseModel):

    model_config = ConfigDict(extra="forbid")
    pick: Literal["A", "B"]


def valid_amount(value: object) -> bool:
    try:
        AmountChoice(amount=value)
        return True
    except (ValidationError, TypeError):
        return False


def valid_rating(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, float):
        if not value.is_integer():
            return False
        value = int(value)
    try:
        RatingChoice(rating=value)
        return True
    except (ValidationError, TypeError):
        return False


def valid_card(value: object) -> bool:
    try:
        CardChoice(card=value)
        return True
    except (ValidationError, TypeError):
        return False


def valid_ab(value: object) -> bool:
    try:
        ABChoice(pick=value)
        return True
    except (ValidationError, TypeError):
        return False
