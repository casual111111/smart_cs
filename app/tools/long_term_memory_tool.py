import uuid
from datetime import datetime

from app.database import get_db_session
from app.models import (
    ConversationSummary,
    LongTermMemory,
    UserProfile,
)


class LongTermMemoryTool:
    """
    长期记忆工具。

    当前阶段先用 MySQL 实现：
    - 用户画像
    - 会话摘要
    - 通用长期记忆

    后续阶段再把 LongTermMemory 接到向量库。
    """

    def get_or_create_user_profile(self, user_id: str) -> UserProfile:
        db = get_db_session()
        try:
            profile = (
                db.query(UserProfile)
                .filter(UserProfile.user_id == user_id)
                .first()
            )

            if profile is None:
                profile = UserProfile(
                    user_id=user_id,
                    common_questions="",
                    preference_summary="",
                    risk_level="normal",
                    refund_count=0,
                    complaint_count=0,
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                )
                db.add(profile)
                db.commit()
                db.refresh(profile)

            return profile
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def build_user_context(self, user_id: str) -> str:
        """
        给 Supervisor / Router / Agent 使用的长期上下文。
        """
        db = get_db_session()
        try:
            profile = (
                db.query(UserProfile)
                .filter(UserProfile.user_id == user_id)
                .first()
            )

            summaries = (
                db.query(ConversationSummary)
                .filter(ConversationSummary.user_id == user_id)
                .order_by(ConversationSummary.updated_at.desc())
                .limit(3)
                .all()
            )

            memories = (
                db.query(LongTermMemory)
                .filter(LongTermMemory.user_id == user_id)
                .order_by(
                    LongTermMemory.importance.desc(),
                    LongTermMemory.updated_at.desc(),
                )
                .limit(5)
                .all()
            )

            parts: list[str] = []

            if profile:
                parts.append(
                    "用户画像："
                    f"风险等级={profile.risk_level}；"
                    f"退款次数={profile.refund_count}；"
                    f"投诉次数={profile.complaint_count}；"
                    f"常见问题={profile.common_questions or '无'}；"
                    f"偏好={profile.preference_summary or '无'}"
                )

            if summaries:
                parts.append("历史会话摘要：")
                for item in summaries:
                    parts.append(f"- {item.summary}")

            if memories:
                parts.append("长期记忆：")
                for item in memories:
                    parts.append(f"- [{item.memory_type}] {item.content}")

            return "\n".join(parts)
        finally:
            db.close()

    def upsert_conversation_summary(
        self,
        session_id: str,
        user_id: str,
        summary: str,
        message_count: int,
    ) -> ConversationSummary:
        db = get_db_session()
        try:
            item = (
                db.query(ConversationSummary)
                .filter(ConversationSummary.session_id == session_id)
                .first()
            )

            now = datetime.now()

            if item is None:
                item = ConversationSummary(
                    session_id=session_id,
                    user_id=user_id,
                    summary=summary,
                    message_count=message_count,
                    created_at=now,
                    updated_at=now,
                )
                db.add(item)
            else:
                item.summary = summary
                item.message_count = message_count
                item.updated_at = now

            db.commit()
            db.refresh(item)
            return item
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def add_memory(
        self,
        user_id: str,
        memory_type: str,
        content: str,
        source_type: str | None = None,
        source_id: str | None = None,
        importance: int = 1,
    ) -> LongTermMemory:
        db = get_db_session()
        try:
            memory = LongTermMemory(
                memory_id=str(uuid.uuid4()),
                user_id=user_id,
                memory_type=memory_type,
                content=content,
                source_type=source_type,
                source_id=source_id,
                importance=importance,
                embedding=None,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )

            db.add(memory)
            db.commit()
            db.refresh(memory)
            return memory
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def update_profile_by_intent(
        self,
        user_id: str,
        intent: str,
    ) -> None:
        """
        先做简单版本：
        - refund 意图累计退款次数
        - complaint 意图累计投诉次数

        后续可以换成 LLM 总结用户画像。
        """
        db = get_db_session()
        try:
            profile = (
                db.query(UserProfile)
                .filter(UserProfile.user_id == user_id)
                .first()
            )

            now = datetime.now()

            if profile is None:
                profile = UserProfile(
                    user_id=user_id,
                    common_questions="",
                    preference_summary="",
                    risk_level="normal",
                    refund_count=0,
                    complaint_count=0,
                    created_at=now,
                    updated_at=now,
                )
                db.add(profile)

            if intent == "refund":
                profile.refund_count += 1

            if intent == "complaint":
                profile.complaint_count += 1

            if profile.refund_count >= 5 or profile.complaint_count >= 3:
                profile.risk_level = "attention"

            profile.updated_at = now

            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()