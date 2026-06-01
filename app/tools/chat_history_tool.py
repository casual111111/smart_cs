from datetime import datetime

from app.database import get_db_session
from app.models import ChatMessage, ChatSession


class ChatHistoryTool:
    """
    聊天历史工具。

    负责：
    1. 创建或更新会话
    2. 保存聊天消息
    3. 查询会话列表
    4. 查询某个会话的消息
    """
#保证会话存在
    def ensure_session(
        self,
        session_id: str,
        user_id: str,
        title: str | None = None,
    ) -> ChatSession:
        db = get_db_session()

        try:
            session = (
                db.query(ChatSession)
                .filter(ChatSession.session_id == session_id)
                .first()
            )

            now = datetime.now()

            if session is None:
                session = ChatSession(
                    session_id=session_id,
                    user_id=user_id,
                    title=title,
                    created_at=now,
                    updated_at=now,
                )

                db.add(session)
            else:
                session.updated_at = now

                if title and not session.title:
                    session.title = title

            db.commit()
            db.refresh(session)

            return session

        except Exception:
            db.rollback()
            raise

        finally:
            db.close()

    def add_message(
        self,
        session_id: str,
        user_id: str,
        role: str,
        content: str,
    ) -> ChatMessage:
        db = get_db_session()

        try:
            message = ChatMessage(
                session_id=session_id,
                user_id=user_id,
                role=role,
                content=content,
                created_at=datetime.now(),
            )

            db.add(message)

            session = (
                db.query(ChatSession)
                .filter(ChatSession.session_id == session_id)
                .first()
            )

            if session:
                session.updated_at = datetime.now()

            db.commit()
            db.refresh(message)

            return message

        except Exception:
            db.rollback()
            raise

        finally:
            db.close()

    def list_sessions(
        self,
        user_id: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[ChatSession], int]:
        db = get_db_session()

        try:
            query = db.query(ChatSession)

            if user_id:
                query = query.filter(ChatSession.user_id == user_id)

            total = query.count()

            sessions = (
                query
                .order_by(ChatSession.updated_at.desc())
                .offset(offset)
                .limit(limit)
                .all()
            )

            return sessions, total

        finally:
            db.close()

    def get_session(self, session_id: str) -> ChatSession | None:
        db = get_db_session()

        try:
            return (
                db.query(ChatSession)
                .filter(ChatSession.session_id == session_id)
                .first()
            )

        finally:
            db.close()

    def get_messages(
        self,
        session_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[ChatMessage], int]:
        db = get_db_session()

        try:
            query = (
                db.query(ChatMessage)
                .filter(ChatMessage.session_id == session_id)
            )

            total = query.count()

            messages = (
                query
                .order_by(ChatMessage.created_at.asc())
                .offset(offset)
                .limit(limit)
                .all()
            )

            return messages, total

        finally:
            db.close()