from app.llm.client import LLMClient


class LLMAgent:
    """
    大模型 Agent。

    当前主要负责：
    1. 根据知识库检索结果生成自然客服回答
    2. 控制回答风格
    3. 避免乱编
    """

    def __init__(self):
        self.llm = LLMClient()

    async def generate_answer(
        self,
        question: str,
        context: str,
    ) -> str:
        system_prompt = """
你是一个企业智能客服助手。

回答要求：
1. 只能基于提供的知识库内容回答。
2. 不要编造知识库中没有的信息。
3. 回答要简洁、清楚、礼貌。
4. 涉及金融、收益、风险时，不能承诺保证收益、稳赚不赔、零风险。
5. 如果知识库内容不足，就说明“暂时无法从知识库确认，建议转人工客服”。
"""

        user_prompt = f"""
用户问题：
{question}

知识库内容：
{context}

请基于知识库内容生成客服回答。
"""

        answer = await self.llm.chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.2,
        )

        return answer.strip()