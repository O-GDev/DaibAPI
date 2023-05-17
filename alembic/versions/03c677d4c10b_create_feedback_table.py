"""create feedback table

Revision ID: 03c677d4c10b
Revises: c03547bbb0ea
Create Date: 2023-05-17 08:56:54.897525

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '03c677d4c10b'
down_revision = 'c03547bbb0ea'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table('feedback',
                    sa.Column('id', sa.Integer(), nullable=False, ),
                    sa.Column('message1', sa.String(), nullable=False),
                    sa.Column('message2', sa.String(), nullable=False),
                    sa.Column('message3', sa.String(), nullable=False),
                    sa.Column('created_at', sa.TIMESTAMP(timezone=True), nullable=True, server_default=sa.text('NOW()')),
                    )
    pass


def downgrade() -> None:
    op.drop_table('feedback')
    pass
