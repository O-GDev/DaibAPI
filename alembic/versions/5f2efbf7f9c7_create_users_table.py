"""create users table

Revision ID: 5f2efbf7f9c7
Revises: 
Create Date: 2023-05-17 21:32:22.017370

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '5f2efbf7f9c7'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table('users',
                    sa.Column('id', sa.Integer(), nullable=False, ),
                    sa.Column('first_name', sa.String(), nullable=False),
                    sa.Column('last_name', sa.String(), nullable=False),
                    sa.Column('email', sa.String(), nullable=False),
                    sa.Column('password', sa.String(), nullable=False),
                    sa.Column('occupation', sa.String(), nullable=False),
                    sa.Column('house_address', sa.String(), nullable=False),
                    sa.Column('phone_number', sa.String(), nullable=False),
                    sa.Column('profile_pics', sa.String(), nullable=False),
                    sa.Column('created_at', sa.TIMESTAMP(timezone=True), nullable=True, server_default=sa.text('NOW()')),
                    sa.PrimaryKeyConstraint('id'),
                    sa.UniqueConstraint('email')
                    )
    pass


def downgrade() -> None:
    op.drop_table("users")
    pass
