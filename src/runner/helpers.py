from __future__ import annotations

import uuid
from typing import Any, Dict, List

from src.config import ModelConfig
from src.types import JudgeResult, LLMMessage, LLMRequest


def _to_jsonable_messages(messages: List[LLMMessage]) -> List[Dict[str, str]]:
    return [{"role": m.role, "content": m.content} for m in messages]


def _request_id(run_id: str, model_name: str, example_id: str, stage: str) -> str:
    value = f"{run_id}:{stage}:{model_name}:{example_id}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, value))


def _cache_key_payload(request: LLMRequest, stage: str) -> Dict[str, Any]:
    return {
        "stage": stage,
        "provider": request.provider,
        "model": request.model,
        "messages": _to_jsonable_messages(request.messages),
        "temperature": request.temperature,
        "top_p": request.top_p,
        "frequency_penalty": request.frequency_penalty,
        "presence_penalty": request.presence_penalty,
        "max_tokens": request.max_tokens,
        "seed": request.seed,
        "response_api": request.response_api,
        "reasoning_effort": request.reasoning_effort,
        "thinking_budget": request.thinking_budget,
        "extra_body": request.extra_body,
    }


def _build_request(model: ModelConfig, messages: List[LLMMessage], request_id: str) -> LLMRequest:
    return LLMRequest(
        provider=model.provider,
        model=model.model,
        messages=messages,
        temperature=model.temperature,
        top_p=model.top_p,
        frequency_penalty=model.frequency_penalty,
        presence_penalty=model.presence_penalty,
        max_tokens=model.max_tokens,
        seed=model.seed,
        reasoning_effort=model.reasoning_effort,
        thinking_budget=model.thinking_budget,
        extra_body=model.extra_body,
        request_id=request_id,
    )


def _model_settings(model: ModelConfig) -> Dict[str, Any]:
    return model.to_settings_dict()


def _judge_descriptor(model: ModelConfig) -> Dict[str, Any]:
    return {
        "name": model.name,
        "provider": model.provider,
        "model": model.model,
        "settings": _model_settings(model),
    }


def build_error_judge_result(error_message: str, context: str = "") -> JudgeResult:
    prefix = f"Judge call failed{' for ' + context if context else ''}: "
    return JudgeResult(
        score=0.0,
        passed=False,
        rationale=prefix + error_message,
        criteria={},
        raw={"error": error_message},
        parse_error=True,
    )


def _progress_enabled(mode: str) -> bool:
    return mode == "log"


def _progress_line(**fields: Any) -> str:
    tokens = [f"{key}={value}" for key, value in fields.items() if value is not None]
    return "[progress]" + ("" if not tokens else f" {' '.join(tokens)}")


def _emit_progress(mode: str, message: str) -> None:
    if _progress_enabled(mode):
        print(message, flush=True)
