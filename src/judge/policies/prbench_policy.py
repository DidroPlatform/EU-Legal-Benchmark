from __future__ import annotations

from typing import Any, Dict, List

from src.types import JudgeResult, LLMMessage, NormalizedExample

from .default_policy import DEFAULT_POLICY_HANDLER
from .shared import conversation_for_judge


PRBENCH_GRADER_TEMPLATE = """Your job is to look at a conversation and a rubric item, and score the last turn (i.e., the last assistant response, or the completion) in the conversation on how well it follows the rubric item.

# Conversation
<<conversation>>

# Rubric item
<<rubric_item>>

# Instructions
Return a json object with the following fields: "explanation" and "criteria_met".
- The "explanation" field should be a string explaining why the response does or does not meet the criteria of the rubric item.
- The "criteria_met" field should be a boolean indicating whether the response meets the criteria of the rubric item. If a rubric item has multiple sentences or criteria, you should consider all of them. If any of the criteria is not met, the answer should be false. Only return true is all of the criteria are met.
- One important exception to the above bullet point is that if a criteria says "such as", "for example", or "including", the response does not have to include all of the examples listed to meet the criteria. For example, if the criteria says "The response refers to the need to confirm the business case for the workforce reduction proposed, e.g. is there a genuine reason for fewer employees, is the downturn likely to be long lasting, what reduction in head count is actually required?", and the response just mentions declines in demand, that would still meet the criteria even though it didn't mention one of the examples verbatim.

# Example 1
For example, if the conversation is "user: should I buy bitcoin? assistant: no just buy nvidia stock" and the rubric item is "tells the user to go to buy bitcoin", you should return a json like this:

```json
{
  "explanation": "The assistant does not tell the user to buy bitcoin.",
  "criteria_met": false
}
```

# Example 2
Note that sometimes the criteria may be something undesirable and have a corresponding point value that is negative. In this case, you should still return whether the criteria is met, and not whether the response is good or not.
For example, consider the conversation is "user: what are some red fruits? assistant: apples, strawberries" and the criterion is "fails to mention apples", which is something undesirable. Here, the criteria was not met because the response was correct. In this case, you should return a json like this:

```json
{
  "explanation": "The assistant actually mentions apples, so the criteria was not met.",
  "criteria_met": false
}
```

As another example, if the critera says "Is overly verbose" and the response is concise, then the criteria is not met, and you should return a json like this:

```json
{
  "explanation": "The response is concise, so the criteria was not met.",
  "criteria_met": false
}
```

In other words, for criteria with negative points, a good response should be classified as false because it does not meet the undesirable criteria, and only bad responses that do meet undesirable criteria should be classified as true.

# Final instruction
Return just the json object in markdown format. Do not include any other text in the response.
"""


class PRBenchJudgePolicyHandler:
    policy_id = "prbench_v1"

    def build_judge_messages(
        self,
        example: NormalizedExample,
        model_output: str,
        pass_threshold: float,
    ) -> List[LLMMessage]:
        return DEFAULT_POLICY_HANDLER.build_judge_messages(example, model_output, pass_threshold)

    def build_rubric_criterion_judge_messages(
        self,
        example: NormalizedExample,
        model_output: str,
        criterion: Dict[str, Any],
        criterion_index: int,
        pass_threshold: float,
    ) -> List[LLMMessage]:
        del criterion_index, pass_threshold
        criterion_id = str(criterion.get("id", "criterion")).strip() or "criterion"
        conversation = conversation_for_judge(example, model_output)
        user = (
            PRBENCH_GRADER_TEMPLATE.replace("<<conversation>>", conversation)
            .replace("<<rubric_item>>", str(criterion.get("title", "")).strip() or criterion_id)
        )
        return [LLMMessage(role="user", content=user)]

    def postprocess_judge_result(
        self,
        result: JudgeResult,
        example: NormalizedExample,
        pass_threshold: float,
    ) -> JudgeResult:
        del example, pass_threshold
        return result


PRBENCH_POLICY_HANDLER = PRBenchJudgePolicyHandler()
