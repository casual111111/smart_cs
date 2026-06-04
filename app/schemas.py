from pydantic import BaseModel
from datetime import datetime
from typing import Literal

TicketStatus = Literal[
    "created",
    "processing",
    "resolved",
    "closed",
]

from pydantic import BaseModel, Field, ConfigDict


class ChatRequest(BaseModel):
    message: str = Field(..., description="用户输入的消息")
    session_id: str | None = Field(
        default=None,
        description="会话ID。首次对话不用传；继续上一轮对话时传上一次返回的 session_id",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "message": "我想申请退款，订单号：ORDER-TEST-001",
                "session_id": None,
            }
        }
    )


class ChatResponse(BaseModel):
    response: str
    session_id: str
    intent: str
    intent_reason: str
    intent_confidence: float
    compliance_passed: bool
    need_human_review: bool = False
    review_task_id: str | None = None
    memory_count: int
    trace_id: str

class TicketResponse(BaseModel):
    ticket_id: str
    user_id: str
    type: str
    priority: str
    status: str
    summary: str
    order_id: str | None = None
    created_at: datetime

    class Config:
        from_attributes = True

class TicketListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[TicketResponse]

class TicketStatusUpdateRequest(BaseModel):
    status: TicketStatus


class TicketStatusUpdateResponse(BaseModel):
    ticket_id: str
    old_status: str
    new_status: str
    message: str

class OrderResponse(BaseModel):
    order_id: str
    user_id: str
    product_name: str
    amount: float
    status: str
    refundable: bool
    created_at: datetime

    class Config:
        from_attributes = True


class OrderListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[OrderResponse]

class OrderCreateRequest(BaseModel):
    username: str
    order_id: str
    product_name: str
    amount: float
    status: str = "paid"
    refundable: bool = True

class UserRegisterRequest(BaseModel):
    username: str
    password: str


class UserResponse(BaseModel):
    user_id: str
    username: str
    role: str
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

class ChatSessionResponse(BaseModel):
    session_id: str
    user_id: str
    title: str | None = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ChatSessionListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[ChatSessionResponse]


class ChatMessageResponse(BaseModel):
    session_id: str
    user_id: str
    role: str
    content: str
    created_at: datetime

    class Config:
        from_attributes = True


class ChatMessageListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[ChatMessageResponse]

class ToolExecuteRequest(BaseModel):
    tool_name: str
    arguments: dict = {}


class HumanReviewTaskResponse(BaseModel):
    review_id: str
    session_id: str
    trace_id: str
    user_id: str
    status: str
    reason: str
    request_content: str
    agent_response: str
    reviewer_id: str | None = None
    reviewer_comment: str | None = None
    created_at: datetime
    updated_at: datetime
    reviewed_at: datetime | None = None

    class Config:
        from_attributes = True


class HumanReviewTaskListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[HumanReviewTaskResponse]


class HumanReviewDecisionRequest(BaseModel):
    comment: str | None = None
