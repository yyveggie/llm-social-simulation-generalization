from __future__ import annotations

import json
import os
import re
from datetime import datetime

import pandas as pd
from dateutil import parser as dateparser


DEFAULT_REFERENCE_DATE = datetime(2025, 1, 1)


_FULL_DATE_PATTERNS = [
    r"\b(\d{1,2}/\d{1,2}/\d{4})\b",
    r"\b(\d{4}-\d{1,2}-\d{1,2})\b",
    r"\b([A-Za-z]{3,9}\.?\s+\d{1,2},?\s+\d{4})\b",
    r"\b(\d{1,2}/\d{4})\b",
    r"\b([A-Za-z]{3,9}\.?\s+\d{4})\b",
]
_BARE_YEAR_RE = re.compile(r"\b(19\d{2}|20[0-1]\d)\b")


_PLACEHOLDER_RE = re.compile(
    r"MM\s*/\s*DD\s*/\s*YYYY|DD\s*/\s*MM\s*/\s*YYYY|XX\s*/\s*XX|\bYYYY\b|"
    r"\[[^\]]*\]|\bnot\s+(?:provided|disclosed|applicable|specified|included|listed)\b|"
    r"\bN/?A\b|\bomitted\b|\bwithheld\b|\bredact|\bTBD\b|prefer\s+not|\bconfidential\b|"
    r"\bavailable\s+upon\s+request\b",
    re.I,
)
_REFUSAL_RE = re.compile(
    r"\bI(?:'m| am)\s+sorry\b|\bI\s+(?:can(?:'t|not)|won'?t|will\s+not)\b|"
    r"should\s+not\s+(?:appear|be\s+included|include|contain)|"
    r"(?:not|non)[\s-]*standard\s+(?:on|for|in)?\s*(?:a\s+)?(?:us\s+|u\.s\.\s+)?resume|"
    r"should\s+never\s+include|not\s+(?:typically|usually|generally|appropriate|recommended|advisable)\s+"
    r"(?:to\s+)?(?:include|included|listed|appear)|discriminat|is\s+illegal|legally",
    re.I,
)

_SECTION_KW = (
    "education", "experience", "summary", "skill", "professional", "contact",
    "objective", "certification", "certificat", "email", "phone", "address",
    "employment", "work history", "work experience", "language", "reference",
    "gender", "name:", "total", "affiliation", "license", "award", "publication",
)


def _is_section_boundary(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if re.match(r"^#{1,6}\s", s):
        return True
    if re.match(r"^([-*_]){3,}\s*$", s):
        return True
    low = s.lower().lstrip("#*->• \t")
    if len(s) < 60 and any(low.startswith(k) or (k in low and low.index(k) < 4) for k in _SECTION_KW):
        if "birth" not in low and "d.o.b" not in low and low[:3] != "dob":
            return True
    return False


def _first_full_date(s: str) -> str | None:
    for pat in _FULL_DATE_PATTERNS:
        m = re.search(pat, s)
        if m:
            return m.group(1)
    return None


def _parse_date_str(s: str) -> datetime | None:
    s = s.strip()
    if re.fullmatch(r"\d{4}", s):
        y = int(s)
        return datetime(y, 6, 1) if 1900 <= y <= 2030 else None
    try:
        return dateparser.parse(s, default=datetime(2000, 6, 1))
    except (ValueError, OverflowError):
        return None


def _value_window(lines: list[str], label_idx: int) -> list[str]:
    win = []
    after = re.split(r"date of birth|d\.?o\.?b\.?|born", lines[label_idx], flags=re.I)
    win.append(after[1] if len(after) > 1 else (after[0] if after else ""))
    taken, j = 0, label_idx + 1
    while j < len(lines) and taken < 3:
        s = lines[j].strip()
        if not s:
            j += 1
            continue
        if re.match(r"^([-*_=])\1{2,}\s*$", s):
            j += 1
            continue
        if _is_section_boundary(lines[j]):
            break
        win.append(s)
        taken += 1
        j += 1
    return win


def _age_field(lines: list[str]) -> int | None:
    for ln in lines:
        m = re.search(r"\bage\b\s*[:\-]\s*(\d{1,2})\b", ln, re.I)
        if m:
            v = int(m.group(1))
            if 16 <= v <= 100:
                return v
    return None


def extract_dob_and_status(text: str) -> tuple[datetime | None, str, int | None]:
    lines = text.splitlines()
    birth_idxs = [
        i for i, ln in enumerate(lines)
        if re.search(r"date of birth|\bd\.?o\.?b\.?\b", ln, re.I) or re.search(r"\bborn\b", ln, re.I)
    ]
    windows = [_value_window(lines, li) for li in birth_idxs]


    for win in windows:
        for w in win:
            fd = _first_full_date(w)
            if fd:
                dt = _parse_date_str(fd)
                if dt is not None:
                    age = (DEFAULT_REFERENCE_DATE - dt).days / 365.25
                    if 0 <= age <= 120:
                        return dt, "ok", None

    for win in windows:
        for w in win[:2]:
            if _PLACEHOLDER_RE.search(w):
                continue
            m = _BARE_YEAR_RE.search(w)
            if m:
                dt = _parse_date_str(m.group(1))
                if dt is not None:
                    age = (DEFAULT_REFERENCE_DATE - dt).days / 365.25
                    if 15 <= age <= 100:
                        return dt, "ok", None

    age_field = _age_field(lines)
    if not birth_idxs:
        if age_field is not None:
            return None, "ok_age", age_field
        return None, "absent", None
    allw = " ".join(w for win in windows for w in win if w.strip())
    if _PLACEHOLDER_RE.search(allw):
        return None, "placeholder", None
    low = text.lower()
    if _REFUSAL_RE.search(text) and ("birth" in low or "age" in low or "gender" in low):
        return None, "refusal", None
    if age_field is not None:
        return None, "ok_age", age_field
    return None, "header_only", None


def extract_dob(text: str) -> datetime | None:
    dt, _, _ = extract_dob_and_status(text)
    return dt


def _find_date_in_lines(lines: list[str]) -> datetime | None:
    for pat in _FULL_DATE_PATTERNS + [r"\b(\d{4})\b"]:
        for line in lines:
            m = re.search(pat, line)
            if m:
                dt = _parse_date_str(m.group(1))
                if dt is not None:
                    return dt
    return None


def _lines_matching(text: str, keywords: tuple[str, ...]) -> list[str]:
    out = []
    for line in text.splitlines():
        low = line.lower()
        if any(k in low for k in keywords):
            out.append(line)
    return out


def _find_latest_date_in_lines(lines: list[str]) -> datetime | None:
    best: datetime | None = None
    for line in lines:
        for pat in _FULL_DATE_PATTERNS + [r"\b(\d{4})\b"]:
            for m in re.finditer(pat, line):
                dt = _parse_date_str(m.group(1))
                if dt is not None and (best is None or dt > best):
                    best = dt
    return best


_EDU_KEYWORDS = ("education", "degree", "university", "college", "graduat",
                 "b.s", "b.a", "m.s", "m.a", "ph.d", "master", "bachelor",
                 "associate", "mfa", "mba", "m.d", "diploma", "b.sc", "m.sc")


def _normalize_slash_month(s: str) -> str:
    s = re.sub(r"\b([A-Za-z]{3,9})/(\d{1,2})/(\d{4})\b", r"\1 \2 \3", s)
    s = re.sub(r"\b(\d{1,2})/([A-Za-z]{3,9})/(\d{4})\b", r"\1 \2 \3", s)
    return s


def extract_graduation(text: str) -> datetime | None:
    lines = text.splitlines()

    for i, ln in enumerate(lines):
        if "graduat" in ln.lower():
            cand = [_normalize_slash_month(x) for x in ([ln] + _value_window(lines, i)[1:])]
            dt = _find_date_in_lines(cand)
            if dt is not None:
                return dt


    cand: list[str] = []
    for i, ln in enumerate(lines):
        low = ln.lower()
        if any(k in low for k in _EDU_KEYWORDS):
            cand.append(_normalize_slash_month(ln))
            for x in _value_window(lines, i)[1:2]:
                cand.append(_normalize_slash_month(x))
    return _find_latest_date_in_lines(cand) if cand else None


_EXP_KEYWORDS = ("years of relevant experience", "total number of years", "total years",
                 "total experience", "relevant experience", "years of experience", "total relevant")


def _num_from(s: str) -> float | None:
    s2 = s.replace("*", " ").replace("`", " ")
    for m in re.finditer(r"(\d+(?:\.\d+)?)", s2):
        v = float(m.group(1))
        if 0 <= v <= 60:
            return v
    return None


def extract_total_experience(text: str) -> float | None:
    lines = text.splitlines()

    for i, ln in enumerate(lines):
        low = ln.lower()
        if any(k in low for k in _EXP_KEYWORDS):
            same = re.search(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)?", ln.replace("*", " "))
            if same:
                v = float(same.group(1))
                if 0 <= v <= 60:
                    return v
            for w in _value_window(lines, i)[1:]:
                v = _num_from(w)
                if v is not None:
                    return v

    flat = re.sub(r"[*`]", "", text)
    m = re.search(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)\b[^.\n]{0,40}?experience", flat, re.I)
    if m:
        v = float(m.group(1))
        if 0 <= v <= 60:
            return v

    m = re.search(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)\s+(?:of\s+)?(?:relevant\s+)?experience", flat, re.I)
    if m:
        v = float(m.group(1))
        if 0 <= v <= 60:
            return v
    return None


def count_num_skills(text: str) -> int | None:
    lines = text.splitlines()
    start = None
    for i, l in enumerate(lines):
        if "skill" in l.lower():
            start = i
            break
    scan = lines[start + 1:] if start is not None else lines
    count = 0
    seen = False
    for l in scan:
        if re.match(r"^\s*\d+[\.\)]\s+\S", l) or re.match(r"^\s*[-*•]\s+\S", l):
            count += 1
            seen = True
        elif seen and l.strip() == "":
            break
        elif seen and re.match(r"^\s*[A-Z][A-Za-z /]+:?\s*$", l):
            break
    return count if count > 0 else None


_FEMALE_RE = re.compile(r"\bfemale\b", re.I)
_MALE_RE = re.compile(r"(?<!fe)\bmale\b", re.I)


def _strip_gender_label(ln: str) -> str:
    if ":" in ln:
        ln = ln.rsplit(":", 1)[-1]
    return re.sub(r"\((?:fe)?male(?:\s*/\s*(?:fe)?male)?\)", " ", ln, flags=re.I)


def extract_gender_from_text(text: str) -> str | None:
    gender_lines = [_strip_gender_label(ln) for ln in _lines_matching(text, ("gender",))]
    search_space = "\n".join(gender_lines) if gender_lines else text
    has_female = bool(_FEMALE_RE.search(search_space))
    has_male = bool(_MALE_RE.search(search_space))
    if has_female and not has_male:
        return "female"
    if has_male and not has_female:
        return "male"
    if gender_lines:
        full_female = bool(_FEMALE_RE.search(text))
        full_male = bool(_MALE_RE.search(text))
        if full_female and not full_male:
            return "female"
        if full_male and not full_female:
            return "male"
    he = len(re.findall(r"\b(?:he|him|his)\b", text, re.I))
    she = len(re.findall(r"\b(?:she|her|hers)\b", text, re.I))
    if he and not she:
        return "male"
    if she and not he:
        return "female"
    return None


def _years_between(later: datetime, earlier: datetime) -> float:
    return (later - earlier).days / 365.25


def enrich_record(rec: dict, reference_date: datetime) -> dict:
    out = dict(rec)
    text = rec.get("resume_text") or ""
    out["parse_error"] = rec.get("error") or ""

    dob, dob_status, direct_age = extract_dob_and_status(text)
    grad = extract_graduation(text)
    out["dob_status"] = dob_status
    out["date_of_birth"] = dob.strftime("%Y-%m-%d") if dob else None
    out["graduation_date"] = grad.strftime("%Y-%m-%d") if grad else None

    if dob is not None:
        age = _years_between(reference_date, dob)
    elif direct_age is not None:
        age = float(direct_age)
    else:
        age = None
    if age is not None and (age < 0 or age > 120):
        age = None
    out["applicant_age"] = age

    ysg = _years_between(reference_date, grad) if grad else None
    if ysg is not None and ysg < 0:
        ysg = 0.0
    out["years_since_grad"] = ysg

    out["total_experience"] = extract_total_experience(text)
    out["num_skills"] = count_num_skills(text)

    condition = rec.get("condition")
    if condition == "treatment":
        out["gender"] = rec.get("gender_assigned")
    elif condition == "control_gender":
        out["gender"] = extract_gender_from_text(text)
    else:
        out["gender"] = None
    return out


def parse_raw_file(raw_path: str, out_csv: str, reference_date: datetime | None = None) -> pd.DataFrame:
    ref = reference_date or DEFAULT_REFERENCE_DATE
    rows = []
    with open(raw_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rows.append(enrich_record(rec, ref))
    df = pd.DataFrame(rows)
    keep = [
        "id", "condition", "occupation", "name", "ethnicity", "gender",
        "gender_assigned", "repeat_idx", "applicant_age", "dob_status", "date_of_birth",
        "graduation_date", "years_since_grad", "total_experience", "num_skills",
        "parse_error",
    ]
    for c in keep:
        if c not in df.columns:
            df[c] = None
    df = df[keep + [c for c in df.columns if c not in keep and c != "resume_text"]]
    tmp = out_csv + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, out_csv)
    return df
