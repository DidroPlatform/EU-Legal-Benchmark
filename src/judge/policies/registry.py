from __future__ import annotations

from src.data.policies import get_policy

from .apex_policy import APEX_POLICY_HANDLER
from .base import JudgePolicyHandler
from .default_policy import DEFAULT_POLICY_HANDLER
from .lexam_policy import LEXAM_POLICY_HANDLER
from .prbench_policy import PRBENCH_POLICY_HANDLER

_HANDLERS_BY_ID: dict[str, JudgePolicyHandler] = {
    APEX_POLICY_HANDLER.policy_id: APEX_POLICY_HANDLER,
    LEXAM_POLICY_HANDLER.policy_id: LEXAM_POLICY_HANDLER,
    PRBENCH_POLICY_HANDLER.policy_id: PRBENCH_POLICY_HANDLER,
}


def get_judge_policy_handler(policy_id: str | None) -> JudgePolicyHandler:
    resolved_id = get_policy(policy_id).policy_id
    return _HANDLERS_BY_ID.get(resolved_id, DEFAULT_POLICY_HANDLER)
