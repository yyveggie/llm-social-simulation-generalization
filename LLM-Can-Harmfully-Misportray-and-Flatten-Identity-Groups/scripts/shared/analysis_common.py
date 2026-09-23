from __future__ import annotations

import pickle
import re
from typing import Dict, List

import llm_prompts as oip


_MODERN_R2A = re.compile(r"^R2a-([a-z]+)-full$")
_LEGACY_R2 = re.compile(r"^R2-([a-z]+)$")
_MODERN_R4 = re.compile(r"^R3-([123])-full")
_LEGACY_R4 = re.compile(r"^R4-([123])$")


def task_group(task_key: str) -> str:
    t = str(task_key or "")
    sub = ""
    m = re.search(r"(_[0-2])$", t)
    if m and ("R2b" in t or t.startswith("R3-race") or t.startswith("R3-gender")):
        sub = m.group(1)
        t = t[: -len(sub)]
    if t.startswith("R1"):
        return "R1"
    m = _MODERN_R2A.match(t) or _LEGACY_R2.match(t)
    if m:
        return "R2-" + m.group(1)
    if "R2b-race" in t or t.startswith("R3-race"):
        return "R3-race" + sub
    if "R2b-gender" in t or t.startswith("R3-gender"):
        return "R3-gender" + sub
    m = _MODERN_R4.match(t)
    if m:
        return "R4-" + m.group(1)
    m = _LEGACY_R4.match(t)
    if m:
        return "R4-" + m.group(1)
    return t + sub


def r2b_split_version(task_key: str):
    g = task_group(task_key)
    if g.startswith("R3-race"):
        return 0
    if g.startswith("R3-gender"):
        return 1
    return None


def _axis_table() -> Dict[str, str]:
    table: Dict[str, str] = {}
    for axis, ids in oip.identities.items():
        for ident in list(ids):
            table[str(ident)] = axis
    for name_list in oip.names.values():
        for name in name_list:
            table[str(name)] = "intersection"
    return table


_AXIS_TABLE = _axis_table()

_LEGACY_EXTRA = {
    "a person with a permanent physical disability": "disability",
    "a person who is blind or has low vision": "disability",
}


def resolve_identity_axis(identity: str) -> str:
    ident = str(identity)
    if ident in _AXIS_TABLE:
        return _AXIS_TABLE[ident]
    if ident in _LEGACY_EXTRA:
        return _LEGACY_EXTRA[ident]
    low = ident.lower()
    if (low.startswith("have a myers-briggs") or low.startswith("have the following persona")
            or low.startswith("are politically") or low.startswith("have the astrology")
            or low == "generic"):
        return "other"
    if any(t in low for t in ("black", "white", "asian")):
        return "intersection" if (" man" in low or " woman" in low) else "race"
    if any(t in low for t in ("woman", " man", "non-binary")):
        return "gender"
    if any(t in low for t in ("boomer", "millennial", "generation")):
        return "age"
    if any(t in low for t in ("disabilit", "blind", "vision", "adhd")):
        return "disability"
    return "unknown"


def other_category(identity: str) -> str:
    low = str(identity).lower()
    if low.startswith("have a myers-briggs"):
        return "MBTI"
    if low.startswith("have the following persona"):
        return "Personas"
    if low.startswith("are politically"):
        return "Political"
    if low.startswith("have the astrology"):
        return "Astrology"
    return "Generic"


_IDENTITY_EXTRA_VARIANTS = {
    "a Baby Boomer (age 59 to 77)": ["a Baby Boomer"],
    "a Millennial (age 27 to 42)": ["a Millennial"],
    "a member of Generation Z (age 18-26)": ["a member of Generation Z"],
    "a person with ADD or ADHD": [
        "a person with ADD", "a person with ADHD", "a person with ADD/ADHD",
        "a person living with Attention Deficit Disorder (ADD) and Adult Attention Deficit "
        "Hyperactivity Disorder (ADHD)",
        "someone with ADD or ADHD", "someone with ADHD or ADHD", "someone with ADHD/ADHD",
        "someone with ADHD",
    ],
    "a person without disabilities": ["someone without disabilities", "able-bodied person"],
    "a person with impaired vision (e.g., blind, low vision, colorblind)": [
        "a person with impaired vision", "someone with impaired vision",
        "a blind person", "someone who is blind",
    ],
    "a White person": ["a White American"],
    "an Asian person": ["an Asian American"],
    "a Black person": ["a Black American", "an African American"],
}

_PREAMBLE_PATTERNS = [
    "As a person living in America, ", "As a person, ", "^As someone([^,]+), ",
    "^As an American([^,]+), ", "^As a ([^,]+), ", "^As an ([^,]+), ",
]

_ALL_NAMES = [str(n) for names in oip.names.values() for n in names]


def _identity_variants(identity: str) -> List[str]:
    phrase = str(identity)
    out = [phrase]
    out += _IDENTITY_EXTRA_VARIANTS.get(phrase, [])
    if "Myers-Briggs" in phrase and "(i.e., " in phrase:
        ind = phrase.index("(i.e., ")
        code = phrase[ind - 5: ind - 1]
        out += ["an " + code, "a " + code]
    return out


def clean_identity_words(text: str, identity: str, task_key: str) -> str:
    group = task_group(task_key)
    text = str(text or "").strip()
    if not group.startswith("R1"):
        for phrase in _identity_variants(identity):
            text = re.sub(r"s {}".format(re.escape(phrase)), "s a person", text, flags=re.IGNORECASE)
        for pattern in _PREAMBLE_PATTERNS:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)
        for name in _ALL_NAMES:
            for tpl in ("I am {}", "As {}", "My name is {}"):
                text = re.sub(tpl.format(re.escape(name)), "", text, flags=re.IGNORECASE)
        if text:
            text = text[0].capitalize() + text[1:]
    if group == "R4-1":
        found = list(re.finditer(r'"', text))
        if len(found) == 2:
            text = text[found[0].end(): found[1].start()]
    return text


def clean_records(records: List[Dict]) -> None:
    for item in records:
        item["response"] = clean_identity_words(
            item.get("response", ""), item.get("identity", ""), item.get("task_key", ""),
        )


def load_legacy_generations(path: str, provider: str = "legacy-pkl") -> List[Dict]:
    with open(path, "rb") as handle:
        data = pickle.load(handle)
    records: List[Dict] = []
    for model, task_map in data.items():
        for task_key, identity_map in task_map.items():
            for identity_index, (identity, responses) in enumerate(identity_map.items()):
                identity = str(identity)
                axis = resolve_identity_axis(identity)
                for sample_index, response in enumerate(list(responses)):
                    records.append({
                        "provider": provider,
                        "model": str(model),
                        "task_key": str(task_key),
                        "identity_axis": axis,
                        "identity_index": identity_index,
                        "identity": identity,
                        "sample_index": sample_index,
                        "response": str(response),
                    })
    return records
