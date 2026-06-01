import json
import re

from app.llm.client import LLMClient


class LLMComplianceAgent:
    """
    LLM 合规复审 Agent。

    只处理语义级判断：
    - 是否承诺保证收益
    - 是否稳赚不赔
    - 是否零风险
    - 是否只是正常风险提示
    """

    def __init__(self):
        self.llm = LLMClient()

    async def review(
        self,
        content: str,
        rule_triggers: list[str] | None = None,
    ) -> dict:
        if not self.llm.enabled:
            return {
                "passed": True,
                "risk_level": "unknown",
                "reason": "未配置 LLM，跳过 LLM 合规复审",
                "suggestion": "",
            }

        system_prompt = """
你是一个金融客服系统的合规审核员。

你的任务是判断一段客服回复是否合规。

重点检查：
1. 是否承诺保证收益、稳赚不赔、零风险、保本保息。
2. 是否使用内部消息、内幕消息等违规诱导表达。
3. 是否夸大收益、弱化风险。
4. 是否诱导用户购买金融产品。
5. 是否只是正常风险提示，例如“不能承诺保证收益”“投资需谨慎”“历史收益不代表未来”。

判断原则：
- 如果文本是在提醒风险、否定违规承诺，应判定为合规。
- 如果文本是在正向承诺收益、保证赚钱、暗示无风险，应判定为不合规。
- 不要因为出现“保证收益”“稳赚不赔”等词就直接判违规，要结合上下文。
- 只输出 JSON，不要输出额外解释。

JSON 格式必须为：
{
  "passed": true,
  "risk_level": "safe",
  "reason": "判断原因",
  "suggestion": "修改建议"
}
"""

        user_prompt = f"""
待审核客服回复：
{content}

规则初筛触发项：
{rule_triggers or []}

请判断这段回复是否合规。
"""

        raw = await self.llm.chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.0,
        )

        return self._parse_json(raw)

    def _parse_json(self, text: str) -> dict:
        if not text:
            return {
                "passed": True,
                "risk_level": "unknown",
                "reason": "LLM 未返回内容，默认放行",
                "suggestion": "",
            }

        cleaned = text.strip()
        cleaned = re.sub(r"^```json", "", cleaned)
        cleaned = re.sub(r"^```", "", cleaned)
        cleaned = re.sub(r"```$", "", cleaned)
        cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)

            return {
                "passed": bool(data.get("passed", True)),
                "risk_level": data.get("risk_level", "unknown"),
                "reason": data.get("reason", ""),
                "suggestion": data.get("suggestion", ""),
            }

        except Exception:
            return {
                "passed": True,
                "risk_level": "unknown",
                "reason": f"LLM 返回内容无法解析，默认放行。原始返回：{text[:200]}",
                "suggestion": "",
            }