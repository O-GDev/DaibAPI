"""create users table

Revision ID: b56d27d62ca0
Revises: 
Create Date: 2023-05-17 23:40:24.935148

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'b56d27d62ca0'
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
                    sa.Column('occupation', sa.String(),),
                    sa.Column('house_address', sa.String(),),
                    sa.Column('phone_number', sa.String(),),
                    sa.Column('profile_pics', sa.String(),),
                    sa.Column('created_at', sa.TIMESTAMP(timezone=True),nullable=True, server_default=sa.text('NOW()')),
                    sa.PrimaryKeyConstraint('id'),
                    sa.UniqueConstraint('email')
                    )
    pass


def downgrade() -> None:
    op.drop_table("users")
    pass