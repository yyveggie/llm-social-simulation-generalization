from __future__ import annotations

import re


_TQDM = re.compile(r"(\d+)%\|.*?\|\s*(\d+)/(\d+)")
_BRACKET = re.compile(r"\[(\d+)/(\d+)\]")

_INTERACTION = re.compile(r"INTERACTION\s+(\d+)")
_RESUMING = re.compile(r"\bResuming:\s*(\d+)/(\d+)\b", re.IGNORECASE)


def parse_resume_line(line: str) -> tuple[int, int] | None:
    m = _RESUMING.search(line)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def parse_line(line: str) -> tuple[int, int] | None:
    m = _TQDM.search(line)
    if m:
        return int(m.group(2)), int(m.group(3))
    m = _BRACKET.search(line)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = _INTERACTION.search(line)
    if m:
        return int(m.group(1)), 0
    return None
