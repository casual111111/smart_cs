import os

from dotenv import load_dotenv


load_dotenv()


class LLMClient:
    """
    大模型客户端。

    使用 OpenAI 兼容接口：
    - OpenAI
    - DeepSeek
    - Qwen
    - GLM
    只要支持 chat.completions.create，就可以接入。
    """

    def __init__(self):
        self.api_key = os.getenv("LLM_API_KEY")
        self.base_url = os.getenv("LLM_BASE_URL")
        self.model = os.getenv("LLM_MODEL", "deepseek-chat")

        self.enabled = bool(self.api_key and self.base_url)

        if self.enabled:
            try:
                from openai import AsyncOpenAI
            except ModuleNotFoundError:
                self.enabled = False
                self.client = None
                return

            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        else:
            self.client = None

    async def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
    ) -> str:
        if not self.enabled:
            return ""

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            temperature=temperature,
        )

        return response.choices[0].message.content or ""
