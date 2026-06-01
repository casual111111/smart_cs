from datetime import datetime

from app.database import get_db_session
from app.models import User


class UserTool:
    """
    用户工具类。

    负责：
    1. 创建用户
    2. 查询用户
    """

    def get_by_username(self, username: str) -> User | None:
        db = get_db_session()

        try:
            return (
                db.query(User)
                .filter(User.username == username)
                .first()
            )

        finally:
            db.close()

    def get_by_user_id(self, user_id: str) -> User | None:
        db = get_db_session()

        try:
            return (
                db.query(User)
                .filter(User.user_id == user_id)
                .first()
            )

        finally:
            db.close()

    def create_user(
        self,
        user_id: str,
        username: str,
        password:str,
        hashed_password: str,
        role: str = "user",
    ) -> User:
        db = get_db_session()

        try:
            user = User(
                user_id=user_id,
                username=username,
                hashed_password=hashed_password,
                role=role,
                is_active=True,
                created_at=datetime.now(),
                password=password
            
            )

            db.add(user)
            db.commit()
            db.refresh(user)

            return user

        except Exception:
            db.rollback()
            raise

        finally:
            db.close()