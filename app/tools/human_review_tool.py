import uuid
from datetime import datetime

from app.database import get_db_session
from app.models import HumanReviewTask


class HumanReviewTool:
    """
    人工审核工具。

    负责创建、查询和处理人工审核任务。
    """

    def create_review_task(
        self,
        session_id: str,
        trace_id: str,
        user_id: str,
        reason: str,
        request_content: str,
        agent_response: str,
    ) -> HumanReviewTask:
        db = get_db_session()

        try:
            task = HumanReviewTask(
                review_id=f"RV-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}",
                session_id=session_id,
                trace_id=trace_id,
                user_id=user_id,
                status="pending",
                reason=reason,
                request_content=request_content,
                agent_response=agent_response,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )

            db.add(task)
            db.commit()
            db.refresh(task)

            return task

        except Exception:
            db.rollback()
            raise

        finally:
            db.close()

    def list_review_tasks(
        self,
        status: str | None = "pending",
        user_id: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[HumanReviewTask], int]:
        db = get_db_session()

        try:
            query = db.query(HumanReviewTask)

            if status:
                query = query.filter(HumanReviewTask.status == status)

            if user_id:
                query = query.filter(HumanReviewTask.user_id == user_id)

            total = query.count()

            tasks = (
                query
                .order_by(HumanReviewTask.created_at.desc())
                .offset(offset)
                .limit(limit)
                .all()
            )

            return tasks, total

        finally:
            db.close()

    def get_review_task(self, review_id: str) -> HumanReviewTask | None:
        db = get_db_session()

        try:
            return (
                db.query(HumanReviewTask)
                .filter(HumanReviewTask.review_id == review_id)
                .first()
            )

        finally:
            db.close()

    def approve_review_task(
        self,
        review_id: str,
        reviewer_id: str,
        comment: str | None = None,
    ) -> HumanReviewTask | None:
        return self._update_review_task(
            review_id=review_id,
            status="approved",
            reviewer_id=reviewer_id,
            comment=comment,
        )

    def reject_review_task(
        self,
        review_id: str,
        reviewer_id: str,
        comment: str | None = None,
    ) -> HumanReviewTask | None:
        return self._update_review_task(
            review_id=review_id,
            status="rejected",
            reviewer_id=reviewer_id,
            comment=comment,
        )

    def get_review_metrics(self) -> dict:
        db = get_db_session()

        try:
            total = db.query(HumanReviewTask).count()
            pending = (
                db.query(HumanReviewTask)
                .filter(HumanReviewTask.status == "pending")
                .count()
            )
            approved = (
                db.query(HumanReviewTask)
                .filter(HumanReviewTask.status == "approved")
                .count()
            )
            rejected = (
                db.query(HumanReviewTask)
                .filter(HumanReviewTask.status == "rejected")
                .count()
            )

            return {
                "total": total,
                "pending": pending,
                "approved": approved,
                "rejected": rejected,
            }

        finally:
            db.close()

    def _update_review_task(
        self,
        review_id: str,
        status: str,
        reviewer_id: str,
        comment: str | None,
    ) -> HumanReviewTask | None:
        db = get_db_session()

        try:
            task = (
                db.query(HumanReviewTask)
                .filter(HumanReviewTask.review_id == review_id)
                .first()
            )

            if task is None:
                return None

            task.status = status
            task.reviewer_id = reviewer_id
            task.reviewer_comment = comment
            task.reviewed_at = datetime.now()
            task.updated_at = datetime.now()

            db.commit()
            db.refresh(task)

            return task

        except Exception:
            db.rollback()
            raise

        finally:
            db.close()
