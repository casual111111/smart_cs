"""create human review tasks

Revision ID: a82f8f1d7c31
Revises: cea90686c96c
Create Date: 2026-06-04 16:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "a82f8f1d7c31"
down_revision: Union[str, Sequence[str], None] = "cea90686c96c"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "human_review_tasks",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("review_id", sa.String(length=64), nullable=False),
        sa.Column("session_id", sa.String(length=64), nullable=False),
        sa.Column("trace_id", sa.String(length=64), nullable=False),
        sa.Column("user_id", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("request_content", sa.Text(), nullable=False),
        sa.Column("agent_response", sa.Text(), nullable=False),
        sa.Column("reviewer_id", sa.String(length=64), nullable=True),
        sa.Column("reviewer_comment", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("reviewed_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_human_review_tasks_review_id"),
        "human_review_tasks",
        ["review_id"],
        unique=True,
    )
    op.create_index(
        op.f("ix_human_review_tasks_session_id"),
        "human_review_tasks",
        ["session_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_human_review_tasks_status"),
        "human_review_tasks",
        ["status"],
        unique=False,
    )
    op.create_index(
        op.f("ix_human_review_tasks_trace_id"),
        "human_review_tasks",
        ["trace_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_human_review_tasks_user_id"),
        "human_review_tasks",
        ["user_id"],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(
        op.f("ix_human_review_tasks_user_id"),
        table_name="human_review_tasks",
    )
    op.drop_index(
        op.f("ix_human_review_tasks_trace_id"),
        table_name="human_review_tasks",
    )
    op.drop_index(
        op.f("ix_human_review_tasks_status"),
        table_name="human_review_tasks",
    )
    op.drop_index(
        op.f("ix_human_review_tasks_session_id"),
        table_name="human_review_tasks",
    )
    op.drop_index(
        op.f("ix_human_review_tasks_review_id"),
        table_name="human_review_tasks",
    )
    op.drop_table("human_review_tasks")
