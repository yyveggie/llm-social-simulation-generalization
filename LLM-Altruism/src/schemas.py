from __future__ import annotations

import re
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

_NUMBER_RE = re.compile(r"\d+")


_REFUSAL_MARKERS = (
    "as an ai language model", "i can't", "i cannot", "i'm unable",
    "i am unable", "i don't have", "i do not have",
)


@dataclass
class ParseResult:
    decision: float | None
    unusable_reason: str | None
    raw_first_number: float | None
    parse_path: str | None = None


_STANDALONE_NUMBER_RE = re.compile(r"[\W\s]*?(\d+)[\W\s]*")
_LETTER_RE = re.compile(r"[A-Za-z]")


INSERTION_TASK_TYPES = ("dictator_insertion", "nonsocial_insertion")
_NAME_WITH_DIGITS_RE = re.compile(r"[A-Za-z][\w.\-]*\d[\w.\-]*")


_OPENAI_MODELNAME_RE = re.compile(
    r"\btext-[a-z]+-?\d{2,3}\b|\b(?:ada|babbage|curie|davinci)-?\d{3}\b", re.I
)


_COMMIT_SHARE_RE = re.compile(
    r"\bi(?:'ll|'d| will| would| am going to| can|'m happy to|'m willing to|'d say)\b"
    r"[^.\n\d]{0,25}?\b(?:share|allocate|give|send|donate|allot|split|contribute|offer)\b"
    r"[^.\n\d]{0,30}?(\d+)",
    re.I,
)
_SHARE_ALL_RE = re.compile(
    r"\b(?:share|allocate|give|send|donate|allot)\s+all\b[^.\n\d]{0,30}?(\d+)", re.I,
)


def _recover_share_from_prose(text: str, upper: float) -> float | None:
    neutral = _NAME_WITH_DIGITS_RE.sub(" ", str(text))
    for rx in (_COMMIT_SHARE_RE, _SHARE_ALL_RE):
        m = rx.search(neutral)
        if m:
            v = float(m.group(1))
            if 0 <= v <= upper:
                return v
    return None


def _recover_zero_option(seq: list[float]) -> float | None:
    if not seq or seq[0] != 0:
        return None
    nonzero = [v for v in seq if v != 0]
    if nonzero and len(set(nonzero)) == 1:
        return nonzero[-1]
    return None


def _standalone_numbers(text: str) -> list[float]:
    out: list[float] = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        m = _STANDALONE_NUMBER_RE.fullmatch(s)
        if m:
            out.append(float(m.group(1)))
    return out


def parse_response(respon: str, upper: float) -> ParseResult:
    if respon is None:
        return ParseResult(None, "empty", None)
    text = str(respon).strip()
    if not text:
        return ParseResult(None, "empty", None)
    if text.startswith("__ERROR__"):
        return ParseResult(None, "api_error", None)


    standalone_in_range = [v for v in _standalone_numbers(text) if 0 <= v <= upper]
    distinct = set(standalone_in_range)


    if 0 in distinct and len(distinct) >= 2:
        rec = _recover_zero_option(standalone_in_range)
        if rec is not None:
            return ParseResult(rec, None, rec, "recovered_zero_option")
        return ParseResult(None, "ambiguous", None)
    if standalone_in_range:
        val = standalone_in_range[-1]
        return ParseResult(val, None, val, "standalone")


    if not _LETTER_RE.search(text):
        seq = [float(x) for x in _NUMBER_RE.findall(text) if 0 <= float(x) <= upper]
        if 0 in set(seq) and len(set(seq)) >= 2:
            rec = _recover_zero_option(seq)
            if rec is not None:
                return ParseResult(rec, None, rec, "recovered_zero_option")
            return ParseResult(None, "ambiguous", None)

    low = text.lower()
    if any(p in low for p in _REFUSAL_MARKERS):


        rec = _recover_share_from_prose(text, upper)
        if rec is not None:
            return ParseResult(rec, None, rec, "recovered_refusal_share")
        return ParseResult(None, "refusal", None)

    if "buy " in low and "tokens from openai" in low:

        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        text = lines[-1] if lines else text


    text = _OPENAI_MODELNAME_RE.sub(" ", text)
    text = text.replace("\n", "")
    m = _NUMBER_RE.search(text)
    if m is None:
        return ParseResult(None, "no_number", None)

    raw_first = float(m.group(0))


    if 0 <= raw_first <= upper:
        return ParseResult(raw_first, None, raw_first, "fallback_first")
    return ParseResult(None, "out_of_range", raw_first)


def _parse_insertion(respon: str, upper: float) -> ParseResult:
    if respon is None:
        return ParseResult(None, "empty", None)
    text = str(respon).strip()
    if not text:
        return ParseResult(None, "empty", None)
    if text.startswith("__ERROR__"):
        return ParseResult(None, "api_error", None)

    neutral = _NAME_WITH_DIGITS_RE.sub(" ", text)

    candidate: float | None = None
    for line in reversed(neutral.splitlines()):
        s = line.strip()
        if not s:
            continue
        m = _STANDALONE_NUMBER_RE.fullmatch(s)
        if m:
            candidate = float(m.group(1))
        break
    if candidate is None:
        tail = neutral[-4:]
        m = _NUMBER_RE.search(tail)
        if m:
            candidate = float(m.group(0))

    if candidate is None:
        low = text.lower()
        if any(p in low for p in _REFUSAL_MARKERS):
            return ParseResult(None, "refusal", None)
        return ParseResult(None, "no_number", None)
    if 0 <= candidate <= upper:
        return ParseResult(candidate, None, candidate, "insertion_tail")
    return ParseResult(None, "out_of_range", candidate)


def parse_for_task(task_type: str, respon: str | None, stakes: float | str) -> ParseResult:
    upper = upper_for(task_type, stakes)
    if task_type in INSERTION_TASK_TYPES:
        return _parse_insertion(respon, upper)
    return parse_response(respon, upper)


def upper_for(task_type: str, stakes: float | str) -> float:
    if task_type == "percent":
        return 100.0
    return float(stakes)


class DecisionResult(BaseModel):

    model_config = ConfigDict(extra="forbid")

    task_type: str
    stakes: float
    upper: float
    decision: float = Field(..., description="落在 [0, upper] 内的决策值")

    @model_validator(mode="after")
    def _check_range(self) -> "DecisionResult":
        if not (0.0 <= self.decision <= self.upper):
            raise ValueError(f"decision {self.decision} 超出 [0, {self.upper}]")
        return self


def validate_response(
    respon: str | None, *, task_type: str, stakes: float | str
) -> DecisionResult | None:
    try:
        upper = upper_for(task_type, stakes)
        pr = parse_for_task(task_type, respon, stakes)
    except (TypeError, ValueError):
        return None
    if pr.decision is None:
        return None
    try:
        return DecisionResult(
            task_type=task_type, stakes=float(stakes), upper=upper, decision=pr.decision
        )
    except (ValidationError, TypeError, ValueError):
        return None


def is_valid_response(respon: str | None, *, task_type: str, stakes: float | str) -> bool:
    return validate_response(respon, task_type=task_type, stakes=stakes) is not None


REQUEST_FAIL_REASONS = ("empty", "api_error")


def classify_response(respon: str | None, *, task_type: str, stakes: float | str) -> str:
    try:
        pr = parse_for_task(task_type, respon, stakes)
    except (TypeError, ValueError):
        return "unusable"
    if pr.decision is not None:
        return "ok"
    if pr.unusable_reason in REQUEST_FAIL_REASONS:
        return "fail"
    return "unusable"
