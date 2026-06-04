from typing import Any
import uuid


def create_working_memory() -> dict[str, Any]:
    """
    工作记忆 Working Memory。

    作用：
    - 保存单次请求中的推理状态
    - 不落库
    - 跟随 LangGraph State 在节点之间流转
    """
    return {
        "trace_id": str(uuid.uuid4()),
        "current_agent": "supervisor",
        "sub_results": {},
        "tool_steps": [],
        "tool_observations": [],
        "retry_count": 0,
        "compliance_result": {},
        "need_human_review": False,
    }


def add_sub_result(
    state: dict[str, Any],
    key: str,
    value: Any,
) -> dict[str, Any]:
    sub_results = dict(state.get("sub_results") or {})
    sub_results[key] = value
    return {"sub_results": sub_results}


def add_tool_step(
    state: dict[str, Any],
    step: dict[str, Any],
) -> dict[str, Any]:
    tool_steps = list(state.get("tool_steps") or [])
    tool_steps.append(step)
    return {"tool_steps": tool_steps}


def add_tool_observation(
    state: dict[str, Any],
    observation: dict[str, Any],
) -> dict[str, Any]:
    tool_observations = list(state.get("tool_observations") or [])
    tool_observations.append(observation)
    return {"tool_observations": tool_observations}


def increase_retry_count(state: dict[str, Any]) -> dict[str, int]:
    return {
        "retry_count": int(state.get("retry_count") or 0) + 1
    }