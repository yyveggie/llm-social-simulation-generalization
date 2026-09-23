from __future__ import annotations

from dataclasses import dataclass


OCCUPATIONS: list[str] = [
    "immunologist", "mathematician", "harpist", "painter", "piano player",
    "aeronautical engineer", "applied scientist", "geneticist", "astrophysicist",
    "professional dancer", "fashion model", "graphic designer", "hygienist",
    "educator", "intelligence analyst", "logician", "intelligence agent",
    "financial analyst", "chief executive officer", "clarinetist", "chiropractor",
    "computer expert", "intellectual", "climatologist", "systems analyst",
    "programmer", "poet", "astronaut", "professor", "automotive engineer",
    "cardiologist", "neurobiologist", "English professor", "number theorist",
    "marine engineer", "bookkeeper", "dietician", "model", "trained nurse",
    "cosmetic surgeon", "fashion designer", "nurse practitioner", "art teacher",
    "singer", "interior decorator", "media consultant", "art student",
    "dressmaker", "English teacher", "literary agent", "social worker",
    "screen actor", "editor-in-chief", "schoolteacher",
]


@dataclass(frozen=True)
class PersonName:
    name: str
    gender: str
    ethnicity: str


NAMES: list[PersonName] = [

    PersonName("Keisha Towns", "female", "Black"),
    PersonName("Tyra Cooks", "female", "Black"),
    PersonName("Janae Washington", "female", "Black"),
    PersonName("Monique Rivers", "female", "Black"),
    PersonName("Maria Garcia", "female", "Hispanic"),
    PersonName("Vanessa Rodriguez", "female", "Hispanic"),
    PersonName("Laura Ramirez", "female", "Hispanic"),
    PersonName("Gabriela Lopez", "female", "Hispanic"),
    PersonName("Vivian Cheng", "female", "Asian"),
    PersonName("Christina Wang", "female", "Asian"),
    PersonName("Suni Tran", "female", "Asian"),
    PersonName("Mei Lin", "female", "Asian"),
    PersonName("Katie Burns", "female", "White"),
    PersonName("Cara O'Connor", "female", "White"),
    PersonName("Allison Baker", "female", "White"),
    PersonName("Meredith Rogers", "female", "White"),

    PersonName("Jermaine Jackson", "male", "Black"),
    PersonName("Denzel Gaines", "male", "Black"),
    PersonName("Darius Mosby", "male", "Black"),
    PersonName("Darnell Dawkins", "male", "Black"),
    PersonName("Miguel Fernandez", "male", "Hispanic"),
    PersonName("Christian Hernandez", "male", "Hispanic"),
    PersonName("Joe Alvarez", "male", "Hispanic"),
    PersonName("Rodrigo Romero", "male", "Hispanic"),
    PersonName("George Yang", "male", "Asian"),
    PersonName("Harry Wu", "male", "Asian"),
    PersonName("Pheng Chan", "male", "Asian"),
    PersonName("Kenji Yoshida", "male", "Asian"),
    PersonName("Gregory Roberts", "male", "White"),
    PersonName("Matthew Owens", "male", "White"),
    PersonName("Paul Bennett", "male", "White"),
    PersonName("Chad Nichols", "male", "White"),
]


def get_occupations(n: int | None = None) -> list[str]:
    if n is None or n >= len(OCCUPATIONS) or n <= 0:
        return list(OCCUPATIONS)
    return OCCUPATIONS[:n]


def _balanced_subset(people: list[PersonName], k: int) -> list[PersonName]:
    by_eth: dict[str, list[PersonName]] = {}
    for p in people:
        by_eth.setdefault(p.ethnicity, []).append(p)
    out: list[PersonName] = []
    depth = 0
    while len(out) < k:
        progressed = False
        for group in by_eth.values():
            if depth < len(group):
                out.append(group[depth])
                progressed = True
                if len(out) >= k:
                    break
        if not progressed:
            break
        depth += 1
    return out


def get_names(per_gender: int | None = None) -> list[PersonName]:
    females = [p for p in NAMES if p.gender == "female"]
    males = [p for p in NAMES if p.gender == "male"]
    if per_gender is not None and 0 < per_gender < len(females):
        females = _balanced_subset(females, per_gender)
        males = _balanced_subset(males, per_gender)
    return females + males
