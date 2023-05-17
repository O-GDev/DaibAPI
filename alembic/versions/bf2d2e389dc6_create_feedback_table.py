"""create feedback table

Revision ID: bf2d2e389dc6
Revises: b58355cefc31
Create Date: 2023-05-17 05:44:56.279712

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'bf2d2e389dc6'
down_revision = 'b58355cefc31'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table('feedback', sa.Column('id', sa.Integer(), nullable=False, primary_key=True), sa.Column('title', sa.String(),nullable=False))    
    pass


def downgrade() -> None:
    op.drop_table('users')
    pass
