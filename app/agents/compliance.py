import re
from dataclasses import dataclass

from app.agents.compliance_llm import LLMComplianceAgent


@dataclass
class ComplianceViolation:
    term: str
    reason: str
    snippet: str


class ComplianceAgent:
    """
    合规审查 Agent。

    当前版本：
    1. 规则检查明显违规内容
    2. PII 敏感信息脱敏
    3. 对模糊风险内容调用 LLM 复审
    """

    risky_terms = [
        "保证收益",
        "稳赚不赔",
        "零风险",
        "无风险",
        "保本保息",
        "保本高收益",
        "内部消息",
        "内幕消息",
    ]

    safe_context_keywords = [
        "不能",
        "不得",
        "禁止",
        "不应",
        "不会",
        "无法",
        "不可",
        "不保证",
        "不承诺",
        "不能承诺",
        "不能保证",
        "并非",
        "不是",
        "不等于",
        "风险提示",
        "风险提醒",
        "请谨慎",
        "谨慎投资",
        "投资需谨慎",
        "历史收益不代表未来",
        "存在风险",
        "收益存在波动",
        "理财非存款",
    ]

    dangerous_patterns = [
        r"本产品.*保证收益",
        r"该产品.*保证收益",
        r"一定.*收益",
        r"一定赚钱",
        r"必赚",
        r"稳赚",
        r"稳赚不赔",
        r"零风险",
        r"无风险",
        r"保本保息",
        r"保本高收益",
        r"内部消息",
        r"内幕消息",
    ]

    pii_patterns = {
        "phone": r"(?<!\d)1[3-9]\d{9}(?!\d)",
        "id_card": r"(?<!\d)\d{17}[\dXx](?!\d)",
        "bank_card": r"(?<!\d)\d{16,19}(?!\d)",
    }

    def __init__(self):
        self.llm_compliance_agent = LLMComplianceAgent()

    async def run(self, content: str, context: str = "") -> dict:
        return await self.check(content)

    async def check(self, content: str) -> dict:
        sanitized = self._mask_pii(content)

        rule_violations = []
        llm_review_triggers = []

        # 1. 明确危险模式检查
        for pattern in self.dangerous_patterns:
            for match in re.finditer(pattern, sanitized):
                snippet = self._get_snippet(
                    sanitized,
                    match.start(),
                    match.end(),
                )

                if self._is_safe_context(snippet):
                    llm_review_triggers.append(
                        f"风险词出现在疑似安全语境：{match.group()} / {snippet}"
                    )
                    continue

                rule_violations.append(
                    ComplianceViolation(
                        term=match.group(),
                        reason="规则命中：疑似存在违规金融承诺",
                        snippet=snippet,
                    )
                )

        # 2. 普通风险词检查
        for term in self.risky_terms:
            for match in re.finditer(re.escape(term), sanitized):
                snippet = self._get_snippet(
                    sanitized,
                    match.start(),
                    match.end(),
                )

                if self._is_safe_context(snippet):
                    llm_review_triggers.append(
                        f"风险词出现在疑似风险提示语境：{term} / {snippet}"
                    )
                    continue

                rule_violations.append(
                    ComplianceViolation(
                        term=term,
                        reason=f"规则命中：包含高风险金融表述：{term}",
                        snippet=snippet,
                    )
                )

        rule_violations = self._deduplicate_violations(rule_violations)
        llm_review_triggers = list(set(llm_review_triggers))

        # 3. 明显违规：规则直接拦截
        if rule_violations:
            return {
                "passed": False,
                "violations": [
                    {
                        "term": item.term,
                        "reason": item.reason,
                        "snippet": item.snippet,
                    }
                    for item in rule_violations
                ],
                "sanitized_content": sanitized,
                "llm_review": None,
                "decision_source": "rule",
            }

        # 4. 模糊风险：交给 LLM 复审
        if llm_review_triggers:
            llm_review = await self.llm_compliance_agent.review(
                content=sanitized,
                rule_triggers=llm_review_triggers,
            )

            if not llm_review["passed"]:
                return {
                    "passed": False,
                    "violations": [
                        {
                            "term": "LLM_REVIEW",
                            "reason": llm_review.get("reason", "LLM 复审判定存在合规风险"),
                            "snippet": sanitized[:200],
                        }
                    ],
                    "sanitized_content": sanitized,
                    "llm_review": llm_review,
                    "decision_source": "llm",
                }

            return {
                "passed": True,
                "violations": [],
                "sanitized_content": sanitized,
                "llm_review": llm_review,
                "decision_source": "llm",
            }

        # 5. 无风险：规则直接通过
        return {
            "passed": True,
            "violations": [],
            "sanitized_content": sanitized,
            "llm_review": None,
            "decision_source": "rule",
        }

    def _is_safe_context(self, snippet: str) -> bool:
        return any(keyword in snippet for keyword in self.safe_context_keywords)

    def _get_snippet(
        self,
        text: str,
        start: int,
        end: int,
        window: int = 24,
    ) -> str:
        left = max(0, start - window)
        right = min(len(text), end + window)
        return text[left:right]

    def _deduplicate_violations(
        self,
        violations: list[ComplianceViolation],
    ) -> list[ComplianceViolation]:
        seen = set()
        result = []

        for item in violations:
            key = (item.term, item.snippet)

            if key in seen:
                continue

            seen.add(key)
            result.append(item)

        return result

    def _mask_pii(self, content: str) -> str:
        result = content

        for pattern in self.pii_patterns.values():
            result = re.sub(pattern, self._mask_text, result)

        return result

    def _mask_text(self, match: re.Match) -> str:
        text = match.group()

        if len(text) <= 6:
            return "*" * len(text)

        return text[:3] + "*" * (len(text) - 6) + text[-3:]
