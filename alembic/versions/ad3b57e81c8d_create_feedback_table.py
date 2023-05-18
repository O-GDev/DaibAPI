"""create feedback table

Revision ID: ad3b57e81c8d
Revises: cac3e74fa532
Create Date: 2023-05-18 22:35:27.504194

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'ad3b57e81c8d'
down_revision = 'cac3e74fa532'
branch_labels = None
depends_on = None



def upgrade() -> None:
    op.create_table('feedback',
                    sa.Column('id', sa.Integer(), nullable=False, ),
                    sa.Column('message1', sa.String(), nullable=False),
                    sa.Column('message2', sa.String(), nullable=False),
                    sa.Column('message3', sa.String(), nullable=False),
                    sa.Column('created_at', sa.TIMESTAMP(timezone=True), nullable=True, server_default=sa.text('now()')),
                    )
    pass


def downgrade() -> None:
    op.drop_table('feedback')
    pass
