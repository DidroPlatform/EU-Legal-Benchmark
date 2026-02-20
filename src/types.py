from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Literal, Optional, Protocol

JudgeMode = Literal["auto", "reference", "rubric", "mcq"]
FinalResponseSource = Literal["sampled", "prefilled", "part_of_conversation"]
ResponseAPI = Literal["chat.completions", "responses"]
ProgressStage = Literal[
    "start",
    "response_started",
    "response_done",
    "response_queued",
    "response_skipped",
    "response_phase_start",
    "response_phase_done",
    "response_phase_interrupted",
    "judge_started",
    "judge_done",
    "judge_phase_start",
    "judge_phase_done",
    "judge_phase_interrupted",
    "response",
    "judge",
    "generation",
    "judging",
]

GOOGLE_PROVIDER_NAMES: FrozenSet[str] = frozenset({"google_genai", "google-genai", "gemini"})
FINAL_RESPONSE_SOURCES: FrozenSet[FinalResponseSource] = frozenset(
    {"sampled", "prefilled", "part_of_conversation"}
)
RESPONSE_APIS: FrozenSet[ResponseAPI] = frozenset({"chat.completions", "responses"})


class WaitableRateLimiter(Protocol):
    def wait(self) -> None: ...


def clamp_score_01(value: float) -> float:
    return max(0.0, min(1.0, value))


@dataclass(frozen=True)
class LLMMessage:
    role: str
    content: str


@dataclass
class LLMRequest:
    provider: str
    model: str
    messages: List[LLMMessage]
    temperature: float = 0.0
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    max_tokens: Optional[int] = None
    seed: Optional[int] = None
    response_api: ResponseAPI = "chat.completions"
    reasoning_effort: Optional[str] = None
    thinking_budget: Optional[int] = None
    extra_body: Optional[Dict[str, Any]] = None
    request_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    provider: str
    model: str
    text: str
    usage: Dict[str, Optional[int]]
    latency_s: float
    request_id: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None


@dataclass
class NormalizedExample:
    id: str
    dataset_name: str
    provenance: str
    judge_mode: JudgeMode
    instructions: str
    context: str = ""
    reference_answer: Optional[str] = None
    rubric: Optional[List[Dict[str, Any]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    messages: List[LLMMessage] = field(default_factory=list)


@dataclass
class JudgeResult:
    score: float
    passed: bool
    rationale: str
    criteria: Dict[str, float]
    raw: Dict[str, Any] = field(default_factory=dict)
    parse_error: bool = False
