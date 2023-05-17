"""create feedback table

Revision ID: 2a941e18a960
Revises: b56d27d62ca0
Create Date: 2023-05-17 23:43:04.279252

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '2a941e18a960'
down_revision = 'b56d27d62ca0'
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
