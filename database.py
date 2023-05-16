from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker


SQLALCHEMY_DATABASE_URL = 'postgresql://zaumzxccdkpzco:5e4e9f5d29cb4205416935fa97a7b52a1bcfda3fe8169c43bba619fb90926bc0@ec2-52-54-200-216.compute-1.amazonaws.com:5432/d14ktrd77faf78'

engine = create_engine(SQLALCHEMY_DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()