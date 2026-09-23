from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator


class ConventionChoice(BaseModel):

    model_config = ConfigDict(extra="forbid")
    name: str
    options: tuple[str, ...]

    @model_validator(mode="after")
    def _in_pool(self) -> "ConventionChoice":
        if self.name not in self.options:
            raise ValueError(f"选择 {self.name!r} 不在候选名池 {list(self.options)} 内")
        return self


def valid_choice(name: object, options: Sequence[str]) -> bool:
    try:
        ConventionChoice(name=name, options=tuple(options))
        return True
    except (ValidationError, TypeError):
        return False
