import asyncio
import json
from app.tools.registry import ToolRegistry

from fastapi import FastAPI, HTTPException, Query,status,Depends
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.responses import StreamingResponse
from app.schemas import (
    ChatMessageListResponse,
    ChatSessionListResponse,
    ChatRequest,
    ChatResponse,
    HumanReviewDecisionRequest,
    HumanReviewTaskListResponse,
    HumanReviewTaskResponse,
    OrderListResponse,
    OrderResponse,
    OrderCreateRequest,
    TicketListResponse,
    TicketResponse,
    TicketStatus,
    TicketStatusUpdateRequest,
    TicketStatusUpdateResponse,
    TokenResponse,
    UserRegisterRequest,
    UserResponse,
    ToolExecuteRequest,
)
from app.supervisor import Supervisor
from app.tools.order_tool import OrderTool
from app.tools.user_tool import UserTool
from app.tools.chat_history_tool import ChatHistoryTool
from app.tools.human_review_tool import HumanReviewTool
from app.models import User

from app.auth import (
    authenticate_user,
    create_access_token,
    get_current_user,
    hash_password,
    require_admin
)
from typing import Annotated
import uuid
from app.tools.trace_tool import TraceTool
from app.mcp.schemas import (
    MCPToolsCallRequest,
    MCPToolsCallResponse,
    MCPToolsListRequest,
    MCPToolsListResponse,
)
from app.mcp.server import MCPToolServer


app = FastAPI(
    title="Smart Customer Service Multi-Agent",
    description="Python 实现的智能客服多 Agent 系统",
    version="0.1.0"
)

supervisor = Supervisor()
order_tool = OrderTool()
user_tool = UserTool()
chat_history_tool = ChatHistoryTool()
tool_registry = ToolRegistry()
trace_tool = TraceTool()
human_review_tool = HumanReviewTool()
mcp_tool_server = MCPToolServer()
@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "version": "0.1.0",
    }



@app.post("/api/auth/register", response_model=UserResponse)
async def register(request: UserRegisterRequest):
    exists = user_tool.get_by_username(request.username)

    if exists:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="用户名已存在",
        )

    user = user_tool.create_user(
        user_id=str(uuid.uuid4()),
        username=request.username,
        hashed_password=hash_password(request.password),
        role="user",
        password=request.password
    )

    return user


@app.post("/api/auth/login", response_model=TokenResponse)
async def login(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
):
    user = authenticate_user(
        username=form_data.username,
        password=form_data.password,
    )

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = create_access_token(
        data={
            "sub": user.user_id,
            "username": user.username,
            "role": user.role,
        }
    )

    return TokenResponse(
        access_token=token,
        token_type="bearer",
    )


@app.get("/api/users/me", response_model=UserResponse)
async def get_me(
    current_user: Annotated[User, Depends(get_current_user)],
):
    return current_user


@app.post("/api/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    current_user: Annotated[User, Depends(get_current_user)],
):
    result = await supervisor.handle(
        message=request.message,
        user_id=current_user.user_id,
        session_id=request.session_id,
    )

    return ChatResponse(**result)

@app.post("/api/chat/stream")
async def chat_stream(
    request: ChatRequest,
    current_user: Annotated[User, Depends(get_current_user)],
):
    async def event_generator():
        result = await supervisor.handle(
            message=request.message,
            user_id=current_user.user_id,
            session_id=request.session_id,
        )

        # 先发送元信息
        meta = {
            "session_id": result["session_id"],
            "trace_id": result["trace_id"],
            "intent": result["intent"],
            "intent_confidence": result["intent_confidence"],
            "intent_reason": result["intent_reason"],
            "compliance_passed": result["compliance_passed"],
            "need_human_review": result["need_human_review"],
            "review_task_id": result["review_task_id"],
            "memory_count": result["memory_count"],
        }

        yield f"event: meta\ndata: {json.dumps(meta, ensure_ascii=False)}\n\n"

        response_text = result["response"]

        # 简单模拟流式输出
        for char in response_text:
            data = {
                "content": char
            }

            yield f"event: message\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.015)

        # 结束事件
        yield "event: done\ndata: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )


@app.get("/api/tickets/{ticket_id}", response_model=TicketResponse)
async def get_ticket(
    ticket_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
):
    ticket = supervisor.ticket_agent.get_ticket(ticket_id)

    if ticket is None:
        raise HTTPException(
            status_code=404,
            detail=f"工单不存在：{ticket_id}",
        )

    if current_user.role != "admin" and ticket.user_id != current_user.user_id:
        raise HTTPException(
            status_code=403,
            detail="无权查看该工单",
        )

    return ticket


@app.get("/api/tickets", response_model=TicketListResponse)
async def list_tickets(
    current_user: Annotated[User, Depends(get_current_user)],
    user_id: str | None = None,
    status: TicketStatus | None = None,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    if current_user.role == "admin":
        query_user_id = user_id
    else:
        query_user_id = current_user.user_id

    tickets, total = supervisor.ticket_agent.list_tickets(
        user_id=query_user_id,
        status=status,
        limit=limit,
        offset=offset,
    )

    return TicketListResponse(
        total=total,
        limit=limit,
        offset=offset,
        items=tickets,
    )

@app.patch(
    "/api/tickets/{ticket_id}/status",
    response_model=TicketStatusUpdateResponse,
)
async def update_ticket_status(
    ticket_id: str,
    request: TicketStatusUpdateRequest,
    admin_user: Annotated[User, Depends(require_admin)],
):
    ticket, old_status = supervisor.ticket_agent.update_ticket_status(
        ticket_id=ticket_id,
        new_status=request.status,
    )

    if ticket is None:
        raise HTTPException(
            status_code=404,
            detail=f"工单不存在：{ticket_id}",
        )

    return TicketStatusUpdateResponse(
        ticket_id=ticket.ticket_id,
        old_status=old_status,
        new_status=ticket.status,
        message="工单状态更新成功",
    )

@app.get("/api/orders", response_model=OrderListResponse)
async def list_orders(
    current_user: Annotated[User, Depends(get_current_user)],
    user_id: str | None = None,
    status: str | None = None,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0)
    
):  
    if current_user.role == "admin":
        query_user_id = user_id
    else:
        query_user_id = current_user.user_id

    orders, total = order_tool.list_orders(
        user_id=query_user_id,
        status=status,
        limit=limit,
        offset=offset,
    )

    return OrderListResponse(
        total=total,
        limit=limit,
        offset=offset,
        items=orders,
    )


@app.get("/api/orders/{order_id}", response_model=OrderResponse)
async def get_order(
    order_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
):
    order = order_tool.get_order(order_id)

    if order is None:
        raise HTTPException(
            status_code=404,
            detail=f"订单不存在：{order_id}",
        )

    if current_user.role != "admin" and order.user_id != current_user.user_id:
        raise HTTPException(
            status_code=403,
            detail="无权查看该订单",
        )

    return order

@app.post("/api/orders", response_model=OrderResponse)
async def create_order(
    request: OrderCreateRequest,
    admin_user: Annotated[User, Depends(require_admin)],
):
    user = user_tool.get_by_username(request.username)

    if user is None:
        raise HTTPException(
            status_code=404,
            detail=f"用户不存在：{request.username}",
        )

    exists = order_tool.get_order(request.order_id)

    if exists:
        raise HTTPException(
            status_code=400,
            detail=f"订单已存在：{request.order_id}",
        )

    order = order_tool.create_order(
        order_id=request.order_id,
        user_id=user.user_id,
        product_name=request.product_name,
        amount=request.amount,
        status=request.status,
        refundable=request.refundable,
    )

    return order

@app.get("/api/chat/sessions", response_model=ChatSessionListResponse)
async def list_chat_sessions(
    current_user: Annotated[User, Depends(get_current_user)],
    user_id: str | None = None,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    if current_user.role == "admin":
        query_user_id = user_id
    else:
        query_user_id = current_user.user_id

    sessions, total = chat_history_tool.list_sessions(
        user_id=query_user_id,
        limit=limit,
        offset=offset,
    )

    return ChatSessionListResponse(
        total=total,
        limit=limit,
        offset=offset,
        items=sessions,
    )


@app.get(
    "/api/chat/sessions/{session_id}/messages",
    response_model=ChatMessageListResponse,
)
async def list_chat_messages(
    session_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    limit: int = Query(default=100, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    session = chat_history_tool.get_session(session_id)

    if session is None:
        raise HTTPException(
            status_code=404,
            detail=f"会话不存在：{session_id}",
        )

    if current_user.role != "admin" and session.user_id != current_user.user_id:
        raise HTTPException(
            status_code=403,
            detail="无权查看该会话",
        )

    messages, total = chat_history_tool.get_messages(
        session_id=session_id,
        limit=limit,
        offset=offset,
    )

    return ChatMessageListResponse(
        total=total,
        limit=limit,
        offset=offset,
        items=messages,
    )

@app.get("/api/tools")
async def list_tools(
    current_user: Annotated[User, Depends(get_current_user)],
):
    tools = tool_registry.list_tools()

    return {
        "total": len(tools),
        "items": tools,
    }

@app.post("/api/tools/execute")
async def execute_tool(
    request: ToolExecuteRequest,
    current_user: Annotated[User, Depends(get_current_user)],
):
    result = await tool_registry.execute(
        tool_name=request.tool_name,
        arguments=request.arguments,
        user_id=current_user.user_id,
    )

    return {
        "success": result.success,
        "tool_name": result.tool_name,
        "data": result.data,
        "error": result.error,
    }

@app.get("/api/traces")
async def list_agent_traces(
    current_user: Annotated[User, Depends(get_current_user)],
    session_id: str | None = None,
    trace_id: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    if current_user.role == "admin":
        query_user_id = None
    else:
        query_user_id = current_user.user_id

    traces, total = trace_tool.list_traces(
        user_id=query_user_id,
        session_id=session_id,
        trace_id=trace_id,
        limit=limit,
        offset=offset,
    )

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "items": [
            {
                "id": item.id,
                "trace_id": item.trace_id,
                "session_id": item.session_id,
                "user_id": item.user_id,
                "agent_name": item.agent_name,
                "node_name": item.node_name,
                "step": item.step,
                "action": item.action,
                "tool_name": item.tool_name,
                "tool_args": trace_tool.parse_json(item.tool_args),
                "tool_result": trace_tool.parse_json(item.tool_result),
                "status": item.status,
                "error": item.error,
                "latency_ms": item.latency_ms,
                "created_at": item.created_at,
            }
            for item in traces
        ],
    }


@app.get("/api/admin/reviews", response_model=HumanReviewTaskListResponse)
async def list_human_review_tasks(
    admin_user: Annotated[User, Depends(require_admin)],
    status: str | None = Query(default="pending"),
    user_id: str | None = None,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    tasks, total = human_review_tool.list_review_tasks(
        status=status,
        user_id=user_id,
        limit=limit,
        offset=offset,
    )

    return HumanReviewTaskListResponse(
        total=total,
        limit=limit,
        offset=offset,
        items=tasks,
    )


@app.post(
    "/api/admin/reviews/{review_id}/approve",
    response_model=HumanReviewTaskResponse,
)
async def approve_human_review_task(
    review_id: str,
    request: HumanReviewDecisionRequest,
    admin_user: Annotated[User, Depends(require_admin)],
):
    task = human_review_tool.approve_review_task(
        review_id=review_id,
        reviewer_id=admin_user.user_id,
        comment=request.comment,
    )

    if task is None:
        raise HTTPException(
            status_code=404,
            detail=f"人工审核任务不存在：{review_id}",
        )

    return task


@app.post(
    "/api/admin/reviews/{review_id}/reject",
    response_model=HumanReviewTaskResponse,
)
async def reject_human_review_task(
    review_id: str,
    request: HumanReviewDecisionRequest,
    admin_user: Annotated[User, Depends(require_admin)],
):
    task = human_review_tool.reject_review_task(
        review_id=review_id,
        reviewer_id=admin_user.user_id,
        comment=request.comment,
    )

    if task is None:
        raise HTTPException(
            status_code=404,
            detail=f"人工审核任务不存在：{review_id}",
        )

    return task


@app.get("/api/metrics")
async def get_metrics(
    admin_user: Annotated[User, Depends(require_admin)],
):
    trace_metrics = trace_tool.get_metrics()
    review_metrics = human_review_tool.get_review_metrics()

    total_requests = trace_metrics.get("total_requests", 0)

    return {
        **trace_metrics,
        "human_review_rate": (
            round(review_metrics["total"] / total_requests, 4)
            if total_requests
            else 0.0
        ),
        "human_review": review_metrics,
    }

@app.post("/mcp/tools/list", response_model=MCPToolsListResponse)
async def mcp_tools_list(
    request: MCPToolsListRequest,
    current_user: User = Depends(get_current_user),
):
    return mcp_tool_server.list_tools(
        request_id=request.id,
    )


@app.post("/mcp/tools/call", response_model=MCPToolsCallResponse)
async def mcp_tools_call(
    request: MCPToolsCallRequest,
    current_user: User = Depends(get_current_user),
):
    return await mcp_tool_server.call_tool(
        name=request.params.name,
        arguments=request.params.arguments,
        user_id=current_user.user_id,
        request_id=request.id,
    )