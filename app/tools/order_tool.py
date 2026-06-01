from datetime import datetime

from app.database import get_db_session
from app.models import Order


class OrderTool:
    """
    订单工具类。

    负责：
    1. 查询订单
    2. 创建测试订单
    3. 判断订单是否可退款
    4. 查询订单列表
    """

    def get_order(self, order_id: str) -> Order | None:
        db = get_db_session()

        try:
            return (
                db.query(Order)
                .filter(Order.order_id == order_id)
                .first()
            )

        finally:
            db.close()

    def list_orders(
        self,
        user_id: str | None = None,
        status: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[Order], int]:
        db = get_db_session()

        try:
            query = db.query(Order)

            if user_id:
                query = query.filter(Order.user_id == user_id)

            if status:
                query = query.filter(Order.status == status)

            total = query.count()

            orders = (
                query
                .order_by(Order.created_at.desc())
                .offset(offset)
                .limit(limit)
                .all()
            )

            return orders, total

        finally:
            db.close()

    def create_order(
        self,
        order_id: str,
        user_id: str,
        product_name: str,
        amount: float,
        status: str = "paid",
        refundable: bool = True,
    ) -> Order:
        db = get_db_session()

        try:
            order = Order(
                order_id=order_id,
                user_id=user_id,
                product_name=product_name,
                amount=amount,
                status=status,
                refundable=refundable,
                created_at=datetime.now(),
            )

            db.add(order)
            db.commit()
            db.refresh(order)

            return order

        except Exception:
            db.rollback()
            raise

        finally:
            db.close()

    def is_refundable(self, order: Order) -> tuple[bool, str]:
        if not order.refundable:
            return False, "该订单被标记为不可退款"

        if order.status == "refunded":
            return False, "该订单已经退款，不能重复申请"

        if order.status == "cancelled":
            return False, "该订单已取消，不能申请退款"

        if order.status not in ["paid", "completed"]:
            return False, f"该订单当前状态为 {order.status}，暂不支持退款"

        return True, "订单符合退款申请条件"