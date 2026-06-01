import uuid
from datetime import datetime

from app.database import get_db_session
from app.models import Ticket


class TicketTool:
    """
    工单工具类。

    负责：
    1. 创建工单
    2. 查询单个工单
    3. 查询工单列表
    4. 更新工单状态

    Agent 不再直接操作数据库，而是调用 Tool。
    """

    def create_ticket(
        self,
        user_id: str,
        ticket_type: str,
        priority: str,
        summary: str,
        order_id: str | None = None,
    ) -> Ticket:
        db = get_db_session()

        try:
            ticket_id = f"TK-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"

            ticket = Ticket(
                ticket_id=ticket_id,
                user_id=user_id,
                type=ticket_type,
                priority=priority,
                status="created",
                summary=summary,
                order_id=order_id,
                created_at=datetime.now(),
            )

            db.add(ticket)
            db.commit()
            db.refresh(ticket)

            return ticket

        except Exception:
            db.rollback()
            raise

        finally:
            db.close()

    def get_ticket(self, ticket_id: str) -> Ticket | None:
        db = get_db_session()

        try:
            return (
                db.query(Ticket)
                .filter(Ticket.ticket_id == ticket_id)
                .first()
            )

        finally:
            db.close()

    def list_tickets(
        self,
        user_id: str | None = None,
        status: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[Ticket], int]:
        db = get_db_session()

        try:
            query = db.query(Ticket)

            if user_id:
                query = query.filter(Ticket.user_id == user_id)

            if status:
                query = query.filter(Ticket.status == status)

            total = query.count()

            tickets = (
                query
                .order_by(Ticket.created_at.desc())
                .offset(offset)
                .limit(limit)
                .all()
            )

            return tickets, total

        finally:
            db.close()

    def update_ticket_status(
        self,
        ticket_id: str,
        new_status: str,
    ) -> tuple[Ticket | None, str | None]:
        db = get_db_session()

        try:
            ticket = (
                db.query(Ticket)
                .filter(Ticket.ticket_id == ticket_id)
                .first()
            )

            if ticket is None:
                return None, None

            old_status = ticket.status
            ticket.status = new_status

            db.commit()
            db.refresh(ticket)

            return ticket, old_status

        except Exception:
            db.rollback()
            raise

        finally:
            db.close()
            