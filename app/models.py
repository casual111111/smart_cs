from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text,JSON
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class Ticket(Base):
    """
    工单表 ORM 模型。
    """

    __tablename__ = "tickets"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    ticket_id: Mapped[str] = mapped_column(
        String(64),
        unique=True,
        index=True,
        nullable=False,
    )

    user_id: Mapped[str] = mapped_column(
        String(64),
        index=True,
        nullable=False,
    )

    type: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
    )

    priority: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String(32),
        default="created",
        nullable=False,
    )

    summary: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    order_id: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now,
        nullable=False,
    )

class Order(Base):
    """
    订单表。

    用于模拟真实业务中的订单数据。
    退款前需要先校验订单是否存在、是否属于当前用户、是否可退款。
    """

    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    order_id: Mapped[str] = mapped_column(
        String(64),
        unique=True,
        index=True,
        nullable=False,
    )

    user_id: Mapped[str] = mapped_column(
        String(64),
        index=True,
        nullable=False,
    )

    product_name: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
    )

    amount: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="paid",
    )

    refundable: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now,
        nullable=False,
    )

class User(Base):
    """
    用户表。

    用于登录认证。
    注意：数据库中只保存密码哈希，不保存明文密码。
    """

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    user_id: Mapped[str] = mapped_column(
        String(64),
        unique=True,
        index=True,
        nullable=False,
    )

    username: Mapped[str] = mapped_column(
        String(64),
        unique=True,
        index=True,
        nullable=False,
    )

    hashed_password: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    password: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    role: Mapped[str] = mapped_column(
        String(32),
        default="user",
        nullable=False,
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now,
        nullable=False,
    )

class ChatSession(Base):
    """
    聊天会话表。
    一次连续对话对应一个 session_id。
    """

    __tablename__ = "chat_sessions"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    session_id: Mapped[str] = mapped_column(
        String(64),
        unique=True,
        index=True,
        nullable=False,
    )

    user_id: Mapped[str] = mapped_column(
        String(64),
        index=True,
        nullable=False,
    )

    title: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now,
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now,
        nullable=False,
    )


class ChatMessage(Base):
    """
    聊天消息表。
    保存用户和助手的每一轮消息。
    """

    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    session_id: Mapped[str] = mapped_column(
        String(64),
        index=True,
        nullable=False,
    )

    user_id: Mapped[str] = mapped_column(
        String(64),
        index=True,
        nullable=False,
    )

    role: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
    )

    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now,
        nullable=False,
    )

class AgentTrace(Base):
    """
    Agent 执行轨迹表。

    记录每一次 Agent 节点执行、工具调用、最终回复等信息。
    """

    __tablename__ = "agent_traces"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    trace_id: Mapped[str] = mapped_column(
        String(64),
        index=True,
        nullable=False,
    )

    session_id: Mapped[str] = mapped_column(
        String(64),
        index=True,
        nullable=False,
    )

    user_id: Mapped[str] = mapped_column(
        String(64),
        index=True,
        nullable=False,
    )

    agent_name: Mapped[str] = mapped_column(
        String(64),
        index=True,
        nullable=False,
    )

    node_name: Mapped[str] = mapped_column(
        String(64),
        index=True,
        nullable=False,
    )

    step: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    action: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
    )

    tool_name: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )

    tool_args: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    tool_result: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="success",
    )

    error: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    latency_ms: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now,
        nullable=False,
    )

class UserProfile(Base):
    """
    用户画像长期记忆。

    保存用户长期偏好、常见问题、风险等级等。
    """
    __tablename__ = "user_profiles"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    user_id: Mapped[str] = mapped_column(
        String(64),
        unique=True,
        index=True,
        nullable=False,
    )

    common_questions: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    preference_summary: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    risk_level: Mapped[str] = mapped_column(
        String(32),
        default="normal",
        nullable=False,
    )

    refund_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    complaint_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now,
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now,
        nullable=False,
    )


class ConversationSummary(Base):
    """
    会话摘要长期记忆。

    不保存完整多轮上下文，而是保存压缩后的摘要。
    """
    __tablename__ = "conversation_summaries"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    session_id: Mapped[str] = mapped_column(
        String(64),
        unique=True,
        index=True,
        nullable=False,
    )

    user_id: Mapped[str] = mapped_column(
        String(64),
        index=True,
        nullable=False,
    )

    summary: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    message_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now,
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now,
        nullable=False,
    )


class LongTermMemory(Base):
    """
    通用长期记忆表。

    后续可以接 FAISS / Chroma / Milvus，
    这里先用 MySQL 存结构化长期记忆。
    """
    __tablename__ = "long_term_memories"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    memory_id: Mapped[str] = mapped_column(
        String(64),
        unique=True,
        index=True,
        nullable=False,
    )

    user_id: Mapped[str] = mapped_column(
        String(64),
        index=True,
        nullable=False,
    )

    memory_type: Mapped[str] = mapped_column(
        String(32),
        index=True,
        nullable=False,
    )

    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    source_type: Mapped[str | None] = mapped_column(
        String(32),
        nullable=True,
    )

    source_id: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )

    importance: Mapped[int] = mapped_column(
        Integer,
        default=1,
        nullable=False,
    )

    embedding: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now,
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now,
        nullable=False,
    )


class HumanReviewTask(Base):
    """
    人工审核任务表。

    用于承接投诉、合规拦截、低置信度路由等需要人工介入的客服会话。
    """

    __tablename__ = "human_review_tasks"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    review_id: Mapped[str] = mapped_column(
        String(64),
        unique=True,
        index=True,
        nullable=False,
    )

    session_id: Mapped[str] = mapped_column(
        String(64),
        index=True,
        nullable=False,
    )

    trace_id: Mapped[str] = mapped_column(
        String(64),
        index=True,
        nullable=False,
    )

    user_id: Mapped[str] = mapped_column(
        String(64),
        index=True,
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String(32),
        index=True,
        nullable=False,
        default="pending",
    )

    reason: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    request_content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    agent_response: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    reviewer_id: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )

    reviewer_comment: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now,
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now,
        nullable=False,
    )

    reviewed_at: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True,
    )

class KnowledgeChunk(Base):
    """
    RAG 知识库 chunk 表。

    MySQL 只保存 chunk 原文和元信息；
    embedding 交给 Qdrant 保存。
    """

    __tablename__ = "knowledge_chunks"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    chunk_id: Mapped[str] = mapped_column(
        String(128),
        unique=True,
        index=True,
        nullable=False,
    )

    source: Mapped[str] = mapped_column(
        String(255),
        index=True,
        nullable=False,
    )

    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    qdrant_point_id: Mapped[str | None] = mapped_column(
        String(64),
        unique=True,
        index=True,
        nullable=True,
    )

    metadata_json: Mapped[dict | None] = mapped_column(
        "metadata",
        JSON,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now,
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now,
        nullable=False,
    )