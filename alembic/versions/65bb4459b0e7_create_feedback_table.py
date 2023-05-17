"""create feedback table

Revision ID: 65bb4459b0e7
Revises: 5f2efbf7f9c7
Create Date: 2023-05-17 21:34:20.186679

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '65bb4459b0e7'
down_revision = '5f2efbf7f9c7'
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

