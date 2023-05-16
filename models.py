from database import Base
from sqlalchemy import Column, Integer, String
from sqlalchemy.sql.expression import null


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True, nullable = False )
    first_name = Column(String, unique=True, index=True, nullable = False)
    last_name = Column(String, unique=True, index=True, nullable = False)
    email = Column(String, unique=True, index=True, nullable = False)
    password = Column(String, nullable = False)
    occupation = Column(String, unique=True, index=True, nullable = False)
    house_address = Column(String, unique=True, index=True, nullable = False)
    phone_number = Column(String, unique=True, index=True, nullable = False)
    profile_pics = Column(String, unique=True, index=True, nullable = False)


class Feedback(Base):
    __tablename__ = "feedback"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    message1 = Column(String, index=True )
    message2 = Column(String, index=True )
    message3 = Column(String, index=True )