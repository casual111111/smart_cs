import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Literal

from dotenv import load_dotenv
from redis import Redis
from redis.exceptions import RedisError


load_dotenv()


Role = Literal["user", "assistant"]


@dataclass
class MemoryMessage:
    role: Role
    content: str
    created_at: str


class ConversationMemory:
    """
    会话记忆模块。

    当前版本：
    1. 优先使用 Redis 保存会话历史
    2. 如果 Redis 不可用，自动退回内存字典
    3. 保留最近 max_messages 条消息
    4. 支持 TTL 自动过期
    """

    def __init__(self, max_messages: int = 20):
        self.max_messages = max_messages
        self.redis_url = os.getenv("REDIS_URL")
        self.ttl_seconds = int(os.getenv("MEMORY_TTL_SECONDS", "86400"))

        self.redis_client: Redis | None = None
        self.fallback_sessions: dict[str, list[MemoryMessage]] = {}

        if self.redis_url:
            try:
                self.redis_client = Redis.from_url(
                    self.redis_url,
                    decode_responses=True,
                )

                self.redis_client.ping()
                print("[Memory] Redis memory enabled")

            except RedisError as e:
                print(f"[Memory] Redis unavailable, fallback to memory: {e}")
                self.redis_client = None
        else:
            print("[Memory] REDIS_URL not configured, fallback to memory")

    def add_message(self, session_id: str, role: Role, content: str) -> None:
        message = MemoryMessage(
            role=role,
            content=content,
            created_at=datetime.now().isoformat(timespec="seconds"),
        )

        if self.redis_client:
            self._add_message_to_redis(session_id, message)
        else:
            self._add_message_to_memory(session_id, message)

    def get_messages(self, session_id: str) -> list[dict]:
        if self.redis_client:
            return self._get_messages_from_redis(session_id)

        messages = self.fallback_sessions.get(session_id, [])
        return [asdict(message) for message in messages]

    def get_recent_context(self, session_id: str, limit: int = 6) -> str:
        messages = self.get_messages(session_id)[-limit:]

        lines = []

        for message in messages:
            role = message.get("role")
            content = message.get("content")

            if role == "user":
                lines.append(f"用户：{content}")
            else:
                lines.append(f"客服：{content}")

        return "\n".join(lines)

    def count(self, session_id: str) -> int:
        return len(self.get_messages(session_id))

    def _redis_key(self, session_id: str) -> str:
        return f"smart-cs:memory:{session_id}"

    def _add_message_to_redis(
        self,
        session_id: str,
        message: MemoryMessage,
    ) -> None:
        assert self.redis_client is not None

        key = self._redis_key(session_id)

        self.redis_client.rpush(
            key,
            json.dumps(asdict(message), ensure_ascii=False),
        )

        # 只保留最近 max_messages 条
        self.redis_client.ltrim(key, -self.max_messages, -1)

        # 设置过期时间
        self.redis_client.expire(key, self.ttl_seconds)

    def _get_messages_from_redis(self, session_id: str) -> list[dict]:
        assert self.redis_client is not None

        key = self._redis_key(session_id)
        raw_messages = self.redis_client.lrange(key, 0, -1)

        messages = []

        for raw in raw_messages:
            try:
                messages.append(json.loads(raw))
            except json.JSONDecodeError:
                continue

        return messages

    def _add_message_to_memory(
        self,
        session_id: str,
        message: MemoryMessage,
    ) -> None:
        if session_id not in self.fallback_sessions:
            self.fallback_sessions[session_id] = []

        self.fallback_sessions[session_id].append(message)

        if len(self.fallback_sessions[session_id]) > self.max_messages:
            self.fallback_sessions[session_id] = self.fallback_sessions[
                session_id
            ][-self.max_messages:]