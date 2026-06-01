import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker


load_dotenv()


DATABASE_URL = os.getenv(
    "DATABASE_URL"
)


class Base(DeclarativeBase):
    pass


# 如果还是 SQLite，就自动创建 data 目录
if DATABASE_URL.startswith("sqlite"):
    Path("data").mkdir(exist_ok=True)


connect_args = {}

if DATABASE_URL.startswith("sqlite"):
    connect_args = {
        "check_same_thread": False,
    }


engine = create_engine(
    DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
    pool_recycle=3600,
    connect_args=connect_args,
)


SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)


def init_db() -> None:
    """
    初始化数据库表。
    """
    import app.models

    Base.metadata.create_all(bind=engine)


def get_db_session():
    return SessionLocal()