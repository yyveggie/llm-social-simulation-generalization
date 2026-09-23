from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

ProviderKind = Literal[
    "openai_chat",
    "openai_responses",
    "anthropic",
    "gemini",
    "huggingface",
    "ollama",
]


class UnifiedLLM(BaseModel):
    provider_kind: ProviderKind = "openai_chat"
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    temperature: float = Field(default=0.7, ge=0)
    max_tokens: int = Field(default=1500, ge=1)
    system: str = ""
    concurrency: int = Field(default=8, ge=1)
    timeout: int = Field(default=120, ge=1)
    max_retries: int = Field(default=5, ge=0)


class ProxyConfig(BaseModel):

    max_upstream_concurrency: int = Field(default=4, ge=1)


class UnifiedConfig(BaseModel):
    llm: UnifiedLLM = Field(default_factory=UnifiedLLM)
    proxy: ProxyConfig = Field(default_factory=ProxyConfig)
    python_executable: Optional[str] = None


class ParamField(BaseModel):

    name: str
    label: str
    type: Literal["int", "float", "str", "bool", "select"] = "str"
    default: Any = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    step: Optional[float] = None
    options: Optional[list[Any]] = None
    help: str = ""

    paper_value: Any = None


class ExperimentInfo(BaseModel):
    id: str
    label: str
    description: str = ""
    conclusion: str = ""

    extra_params: list[ParamField] = Field(default_factory=list)


    importance_rank: Optional[int] = None
    importance_tier: str = ""
    importance_basis: str = ""


class ProjectInfo(BaseModel):
    id: str
    name: str
    paper: str
    description: str = ""
    intro: str = ""
    supported_kinds: list[str] = Field(default_factory=list)
    selection_mode: Literal["multi", "single"] = "multi"
    experiments: list[ExperimentInfo] = Field(default_factory=list)
    param_schema: list[ParamField] = Field(default_factory=list)


class LaunchRequest(BaseModel):
    project_id: str = Field(min_length=1)
    experiment_ids: list[str] = Field(min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)

    llm_override: Optional[dict[str, Any]] = None


JobStatus = Literal["running", "paused", "succeeded", "failed", "stopped"]


class JobProgress(BaseModel):
    current: int = Field(default=0, ge=0)
    total: int = Field(default=0, ge=0)
    percent: float = Field(default=0.0, ge=0)


class SampleStats(BaseModel):

    ok: int = Field(default=0, ge=0)
    unusable: int = Field(default=0, ge=0)
    fail: int = Field(default=0, ge=0)
    failed_calls: Optional[int] = Field(default=None, ge=0)


class JobInfo(BaseModel):
    id: str
    project_id: str
    project_name: str
    paper: str = ""
    experiment_id: str
    label: str
    model: str = ""
    params: dict[str, Any] = Field(default_factory=dict)
    status: JobStatus
    pid: Optional[int] = None
    exit_code: Optional[int] = None
    started_at: float
    finished_at: Optional[float] = None
    paused_at: Optional[float] = None
    paused_total: float = Field(default=0.0, ge=0)
    progress: JobProgress = Field(default_factory=JobProgress)
    command: list[str] = Field(default_factory=list)
    cwd: str = ""
    results_dir: str = ""
    log_path: str = ""
    error: Optional[str] = None
    api_calls: int = Field(default=0, ge=0)
    sample_stats: Optional[SampleStats] = None


class JobUnitState(BaseModel):
    experiment_id: str
    label: str = ""
    argv: list[str] = Field(default_factory=list)
    selected: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)


class JobState(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    schema_version: str = Field(default="job_state/v1", alias="schema")
    info: JobInfo
    unit: Optional[JobUnitState] = None
    llm_snapshot: dict[str, Any] = Field(default_factory=dict)


class ChatMessage(BaseModel):

    model_config = ConfigDict(extra="allow")

    role: str = Field(min_length=1)
    content: Any = ""


class ChatCompletionRequest(BaseModel):

    model_config = ConfigDict(extra="allow")

    model: Optional[str] = None
    messages: list[ChatMessage] = Field(min_length=1)
    temperature: Optional[float] = Field(default=None, ge=0)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    max_completion_tokens: Optional[int] = Field(default=None, ge=1)
    top_p: Optional[float] = Field(default=None, ge=0, le=1)


class OpenAIChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    content: Any = ""


class OpenAIChatChoice(BaseModel):
    model_config = ConfigDict(extra="allow")

    message: OpenAIChatMessage


class OpenAIChatResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    choices: list[OpenAIChatChoice] = Field(min_length=1)
