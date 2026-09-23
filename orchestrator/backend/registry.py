from __future__ import annotations

from .adapters.age_gender_distortion import AgeGenderDistortionAdapter
from .adapters.altruism import AltruismAdapter
from .adapters.base import Adapter
from .adapters.behavioral import BehavioralAdapter
from .adapters.identity import IdentityAdapter
from .adapters.social import SocialAdapter
from .schemas import ProjectInfo

_ADAPTERS: dict[str, Adapter] = {
    a.id: a
    for a in (
        AltruismAdapter(),
        BehavioralAdapter(),
        IdentityAdapter(),
        AgeGenderDistortionAdapter(),
        SocialAdapter(),
    )
}


def get_adapter(project_id: str) -> Adapter:
    if project_id not in _ADAPTERS:
        raise KeyError(f"未知项目：{project_id}（可选：{', '.join(_ADAPTERS)}）")
    return _ADAPTERS[project_id]


def list_projects() -> list[ProjectInfo]:
    return [a.info() for a in _ADAPTERS.values()]
