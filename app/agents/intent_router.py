import json
import re
from typing import Literal

from pydantic import BaseModel

from app.llm.client import LLMClient


IntentType = Literal[
    "knowledge",
    "ticket",
    "complaint",
    "account",
    "refund",
    "unknown",
]


class IntentResult(BaseModel):
    intent: IntentType
    confidence: float
    reason: str


class IntentRouterAgent:
    """
    LLM 意图识别 Agent。

    作用：
    1. 判断用户消息属于哪类业务
    2. 返回 intent、confidence、reason
    3. 如果没有配置 LLM，则自动退回规则路由
    """

    def __init__(self):
        self.llm = LLMClient()
        self.fallback_router = RuleBasedIntentRouter()

    async def run(self, message: str, context: str = "") -> IntentResult:
        message = message.strip()

        if not message:
            return IntentResult(
                intent="unknown",
                confidence=0.0,
                reason="用户输入为空",
            )

        # 工单号查询这种非常确定的情况，不需要浪费 LLM
        if self._looks_like_ticket_query(message):
            return IntentResult(
                intent="ticket",
                confidence=0.98,
                reason="检测到明确工单号，直接路由到工单 Agent",
            )

        # 如果没有配置 LLM，自动使用规则兜底
        if not self.llm.enabled:
            return await self.fallback_router.run(message, context)

        system_prompt = """
你是一个智能客服系统中的意图识别 Agent。

你的任务是根据用户当前消息和历史上下文，判断用户意图。

只能从以下 intent 中选择一个：

1. knowledge
表示用户在询问知识、政策、规则、流程、说明。
例如：
- 退款多久到账？
- 开户流程是什么？
- 理财产品有什么风险？
- 怎么修改手机号？
- 退款规则是什么？

2. refund
表示用户明确想申请退款、退货、退费，或者正在提供退款所需订单号。
例如：
- 我想申请退款
- 我要退钱
- 订单号：123456
- 我的订单号是 ORDER-123456
注意：如果上下文中用户已经说过要退款，那么当前消息只提供订单号，也应判断为 refund。

3. complaint
表示用户投诉、抱怨、不满意、举报。
例如：
- 我要投诉
- 你们服务太差了
- 我要举报
- 客服一直不处理

4. account
表示用户想办理账户相关业务。
例如：
- 我要开户
- 我要注册账户
- 帮我开通账户
- 我的账户被冻结了怎么办

5. ticket
表示用户要查询工单、创建普通工单、转人工客服、查询处理进度。
例如：
- 查询工单 TK-20260529-ABC123
- 我的工单处理到哪了
- 转人工
- 帮我创建工单

6. unknown
表示无法判断意图。

注意区分：
- “退款流程是什么？”属于 knowledge
- “我要申请退款”属于 refund
- “开户流程是什么？”属于 knowledge
- “我要开户”属于 account
- “TK-20260529-ABC123”属于 ticket

你必须只输出 JSON，不要输出其他内容。

JSON 格式：
{
  "intent": "knowledge",
  "confidence": 0.9,
  "reason": "判断原因"
}
"""

        user_prompt = f"""
历史上下文：
{context or "无"}

用户当前消息：
{message}

请判断用户意图。
"""

        raw = await self.llm.chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.0,
        )

        result = self._parse_llm_result(raw)

        if result is None:
            return await self.fallback_router.run(message, context)

        return result

    def _parse_llm_result(self, raw: str) -> IntentResult | None:
        if not raw:
            return None

        cleaned = raw.strip()

        cleaned = re.sub(r"^```json", "", cleaned)
        cleaned = re.sub(r"^```", "", cleaned)
        cleaned = re.sub(r"```$", "", cleaned)
        cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)

            intent = data.get("intent", "unknown")
            confidence = float(data.get("confidence", 0.5))
            reason = data.get("reason", "LLM 意图识别结果")

            allowed_intents = {
                "knowledge",
                "ticket",
                "complaint",
                "account",
                "refund",
                "unknown",
            }

            if intent not in allowed_intents:
                return None

            confidence = max(0.0, min(confidence, 1.0))

            return IntentResult(
                intent=intent,
                confidence=confidence,
                reason=reason,
            )

        except Exception:
            return None

    def _looks_like_ticket_query(self, text: str) -> bool:
        return bool(re.search(r"TK-\d{8}-[A-Za-z0-9]{6}", text))


class RuleBasedIntentRouter:
    """
    规则兜底路由器。

    作用：
    1. 没配置 LLM 时使用
    2. LLM 返回 JSON 解析失败时使用
    3. 保证系统不因为大模型异常而不可用
    """

    async def run(self, message: str, context: str = "") -> IntentResult:
        message = message.strip()

        if not message:
            return IntentResult(
                intent="unknown",
                confidence=0.0,
                reason="用户输入为空",
            )

        if self._looks_like_ticket_query(message):
            return IntentResult(
                intent="ticket",
                confidence=0.92,
                reason="规则兜底：检测到工单号或工单查询表达",
            )

        if self._contains_order_id(message) and self._contains_any(
            context,
            ["退款", "退钱", "退货", "退费", "申请退款"],
        ):
            return IntentResult(
                intent="refund",
                confidence=0.9,
                reason="规则兜底：检测到订单号，且上下文中存在退款需求",
            )

        if self._contains_order_id(message) and self._contains_any(
            message,
            ["退款", "退钱", "退货", "退费"],
        ):
            return IntentResult(
                intent="refund",
                confidence=0.95,
                reason="规则兜底：检测到退款需求和订单号",
            )

        if self._contains_any(message, ["是什么", "怎么", "如何", "多少", "规则", "政策", "流程"]):
            return IntentResult(
                intent="knowledge",
                confidence=0.8,
                reason="规则兜底：检测到知识问答类表达",
            )

        if self._contains_any(message, ["退款", "退钱", "退货", "退费"]):
            return IntentResult(
                intent="refund",
                confidence=0.95,
                reason="规则兜底：检测到退款相关关键词",
            )

        if self._contains_any(message, ["投诉", "差评", "不满意", "举报"]):
            return IntentResult(
                intent="complaint",
                confidence=0.9,
                reason="规则兜底：检测到投诉相关关键词",
            )

        if self._contains_any(message, ["开户", "注册", "开通账户", "账户开通"]):
            return IntentResult(
                intent="account",
                confidence=0.9,
                reason="规则兜底：检测到账户办理相关关键词",
            )

        if self._contains_any(
            message,
            ["申请", "办理", "工单", "人工客服", "转人工", "查询工单", "工单状态", "进度"],
        ):
            return IntentResult(
                intent="ticket",
                confidence=0.8,
                reason="规则兜底：检测到业务办理或工单处理需求",
            )

        return IntentResult(
            intent="knowledge",
            confidence=0.5,
            reason="规则兜底：未匹配到明确意图，默认走知识问答",
        )

    def _contains_any(self, text: str, keywords: list[str]) -> bool:
        return any(keyword in text for keyword in keywords)

    def _contains_order_id(self, text: str) -> bool:
        patterns = [
            r"订单号[:：]?\s*[A-Za-z0-9\-]{6,30}",
            r"订单编号[:：]?\s*[A-Za-z0-9\-]{6,30}",
            r"^[A-Za-z0-9\-]{6,30}$",
        ]

        return any(re.search(pattern, text) for pattern in patterns)

    def _looks_like_ticket_query(self, text: str) -> bool:
        return bool(re.search(r"TK-\d{8}-[A-Za-z0-9]{6}", text))