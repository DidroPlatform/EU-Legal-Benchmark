from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


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
    judge_mode: str
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
