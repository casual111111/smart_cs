import json
import re
from dataclasses import asdict, dataclass
from typing import Any

from app.llm.client import LLMClient
from app.tools.registry import ToolRegistry
import time
import uuid

from app.tools.trace_tool import TraceTool

@dataclass
class AgentToolStep:
    step: int
    action: str
    tool_name: str | None = None
    arguments: dict[str, Any] | None = None
    observation: Any = None
    error: str | None = None


@dataclass
class AgentToolResult:
    final_answer: str
    steps: list[AgentToolStep]


class BaseToolAgent:
    """
    具备工具调用能力的 Agent 基类。

    它不代表一个具体业务 Agent。
    它只是给 TicketAgent / KnowledgeAgent 等专业 Agent 提供：
    1. 工具权限控制
    2. 工具列表生成
    3. ReAct 工具调用循环
    4. JSON 解析
    """

    def __init__(
        self,
        agent_name: str,
        allowed_tools: list[str],
    ):
        self.agent_name = agent_name
        self.allowed_tools = allowed_tools
        self.llm = LLMClient()
        self.tool_registry = ToolRegistry()
        self.trace_tool = TraceTool()

    def get_available_tools(self) -> list[dict[str, Any]]:
        all_tools = self.tool_registry.list_tools()

        return [
            tool
            for tool in all_tools
            if tool["name"] in self.allowed_tools
        ]

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        user_id: str,
    ):
        if tool_name not in self.allowed_tools:
            raise PermissionError(
                f"{self.agent_name} 无权调用工具：{tool_name}"
            )

        return await self.tool_registry.execute(
            tool_name=tool_name,
            arguments=arguments,
            user_id=user_id,
        )

    async def react_with_tools(
        self,
        message: str,
        user_id: str,
        context: str,
        system_instruction: str,
        session_id: str,
        node_name: str,
        max_steps: int = 5,
        trace_id: str | None = None,
    ) -> AgentToolResult:
        """
        通用 ReAct 工具调用循环。

        专业 Agent 传入自己的 system_instruction，
        BaseToolAgent 负责执行通用工具调用逻辑。
        """
        trace_id = trace_id or str(uuid.uuid4())

        if not self.llm.enabled:
            return AgentToolResult(
                final_answer="当前未配置 LLM，无法执行智能工具调用。",
                steps=[],
            )

        steps: list[AgentToolStep] = []
        scratchpad = ""

        for step_index in range(1, max_steps + 1):
            raw = await self.llm.chat(
                system_prompt=self._build_system_prompt(system_instruction),
                user_prompt=self._build_user_prompt(
                    message=message,
                    context=context,
                    scratchpad=scratchpad,
                ),
                temperature=0.0,
            )

            decision = self._parse_json(raw)

            if decision is None:
                return AgentToolResult(
                    final_answer="抱歉，我无法稳定解析下一步操作，建议转人工客服处理。",
                    steps=steps,
                )

            action = decision.get("action")

            if action == "final":
                answer = decision.get("answer") or "已处理完成。"

                steps.append(
                    AgentToolStep(
                        step=step_index,
                        action="final",
                        observation=answer,
                    )
                )

                self.trace_tool.record_trace(
                    trace_id=trace_id,
                    session_id=session_id,
                    user_id=user_id,
                    agent_name=self.agent_name,
                    node_name=node_name,
                    step=step_index,
                    action="final",
                    tool_name=None,
                    tool_args=None,
                    tool_result={"answer": answer},
                    status="success",
                    error=None,
                    latency_ms=None,
                )
                return AgentToolResult(
                    final_answer=answer,
                    steps=steps,
                )

            if action != "tool":
                return AgentToolResult(
                    final_answer="抱歉，我没有理解下一步操作，建议转人工客服处理。",
                    steps=steps,
                )

            tool_name = decision.get("tool_name")
            arguments = decision.get("arguments") or {}

            step = AgentToolStep(
                step=step_index,
                action="tool",
                tool_name=tool_name,
                arguments=arguments,
            )

            start_time = time.perf_counter()

            try:
                tool_result = await self.execute_tool(
                    tool_name=tool_name,
                    arguments=arguments,
                    user_id=user_id,
                )

                latency_ms = int((time.perf_counter() - start_time) * 1000)

                if tool_result.success:
                    step.observation = tool_result.data
                    status = "success"
                    error = None
                else:
                    step.error = tool_result.error
                    step.observation = {
                        "success": False,
                        "error": tool_result.error,
                    }
                    status = "failed"
                    error = tool_result.error

                self.trace_tool.record_trace(
                    trace_id=trace_id,
                    session_id=session_id,
                    user_id=user_id,
                    agent_name=self.agent_name,
                    node_name=node_name,
                    step=step_index,
                    action="tool",
                    tool_name=tool_name,
                    tool_args=arguments,
                    tool_result=step.observation,
                    status=status,
                    error=error,
                    latency_ms=latency_ms,
                )

            except Exception as exc:
                latency_ms = int((time.perf_counter() - start_time) * 1000)

                step.error = str(exc)
                step.observation = {
                    "success": False,
                    "error": str(exc),
                }

                self.trace_tool.record_trace(
                    trace_id=trace_id,
                    session_id=session_id,
                    user_id=user_id,
                    agent_name=self.agent_name,
                    node_name=node_name,
                    step=step_index,
                    action="tool",
                    tool_name=tool_name,
                    tool_args=arguments,
                    tool_result=step.observation,
                    status="error",
                    error=str(exc),
                    latency_ms=latency_ms,
                )


            steps.append(step)

            scratchpad += (
                f"\n第 {step_index} 步：\n"
                f"Action: {tool_name}\n"
                f"Arguments: {json.dumps(arguments, ensure_ascii=False)}\n"
                f"Observation: {json.dumps(step.observation, ensure_ascii=False)}\n"
            )

        return AgentToolResult(
            final_answer="该问题需要进一步处理，已建议转人工客服。",
            steps=steps,
        )

    def _build_system_prompt(self, system_instruction: str) -> str:
        tools = self.get_available_tools()

        return f"""
你是 {self.agent_name}。

你的业务职责：
{system_instruction}

你可以使用以下工具：
{json.dumps(tools, ensure_ascii=False, indent=2)}

你必须严格按照以下 JSON 格式输出。

如果需要调用工具，输出：
{{
  "action": "tool",
  "tool_name": "工具名称",
  "arguments": {{
    "参数名": "参数值"
  }}
}}

如果已经可以回答用户，输出：
{{
  "action": "final",
  "answer": "给用户的最终回复"
}}

要求：
1. 不要输出 JSON 之外的内容。
2. 不要调用未提供的工具。
3. 不要编造工具结果。
4. 工具返回错误时，要根据错误原因给用户清楚说明。
"""

    def _build_user_prompt(
        self,
        message: str,
        context: str,
        scratchpad: str,
    ) -> str:
        return f"""
历史上下文：
{context or "无"}

用户当前消息：
{message}

已完成的工具调用：
{scratchpad or "无"}

请判断下一步应该调用工具，还是直接给出最终回复。
"""

    def _parse_json(self, text: str) -> dict | None:
        if not text:
            return None

        cleaned = text.strip()
        cleaned = re.sub(r"^```json", "", cleaned)
        cleaned = re.sub(r"^```", "", cleaned)
        cleaned = re.sub(r"```$", "", cleaned)
        cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except Exception:
            return None

    def steps_to_dict(self, steps: list[AgentToolStep]) -> list[dict]:
        return [asdict(step) for step in steps]