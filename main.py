from fastapi import FastAPI ,HTTPException ,File, UploadFile,status
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Boolean
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
# from passlib.hash import bcrypt
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import pickle
import json
import os
import psycopg2
import sqlalchemy
import databases





diabetes_dataset = pd.read_csv('diabetes.csv') 

# print(diabetes_dataset.head())
diabetes_dataset.head()

X = diabetes_dataset.drop(columns = 'Outcome', axis=1)

Y = diabetes_dataset['Outcome']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)

# print(X.shape, X_train.shape, X_test.shape)

classifier = svm.SVC(kernel='linear')

classifier.fit(X_train, Y_train)

X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

# print('Accuracy score of the training data : ', training_data_accuracy)

X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

# print('Accuracy score of the test data : ', test_data_accuracy)


filename = 'diabetes_model.sav'
pickle.dump(classifier, open(filename, 'wb'))

DATABASE_URL = "postgresql://jzvfuzitepousi:6002de9a9c937cd7e19d12f03bda50f051f931a9f3c76f5f7ae548c37b872472@ec2-3-234-204-26.compute-1.amazonaws.com:5432/d9g26788lhb1o6"

database = databases.Database(DATABASE_URL)

metadata = sqlalchemy.MetaData()


user = sqlalchemy.Table(
    "user",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),
    sqlalchemy.Column("username", sqlalchemy.String),
    sqlalchemy.Column("email", sqlalchemy.String),
    sqlalchemy.Column("password", sqlalchemy.String),
)

feedback = sqlalchemy.Table(
    "feedback",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),
    sqlalchemy.Column("email", sqlalchemy.String),
    sqlalchemy.Column("message1", sqlalchemy.String),
    sqlalchemy.Column("message2", sqlalchemy.String),
    sqlalchemy.Column("message3", sqlalchemy.String),
)

engine = sqlalchemy.create_engine(
    DATABASE_URL
)

metadata.create_all(engine)
# Set up the database connection
# SQLALCHEMY_DATABASE_URL = 'postgresql://postgres:Gbogo321@localhost/Diabetes_db'
# engine = create_engine(SQLALCHEMY_DATABASE_URL)
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
# Base = declarative_base()

origins = [
"http://192.168.43.177:8000"
]


class model_input(BaseModel):
    
    pregnancies : int
    Glucose : int
    BloodPressure : int
    SkinThickness : int
    Insulin : int
    BMI : float
    DiabetesPedigreeFunction : float
    Age : int      

# Define the User model
# class User(Base):
#     __tablename__ = "userss"
#     id = Column(Integer, primary_key=True, index=True)
#     username = Column(String, unique=True, index=True)
#     email = Column(String, unique=True, index=True)
#     password = Column(String)

# class Feed(Base):
#     __tablename__ = "feed"
#     id = Column(Integer, primary_key=True, index=True)
#     email = Column(String, unique=True, index=True)
#     message1 = Column(String, index=True )
#     message2 = Column(String, index=True )
#     message3 = Column(String, index=True )

    
# Create the tables in the database
# Base.metadata.create_all(bind=engine)

# Define the request body schema
class UserCreate(BaseModel):
    username: str
    email: str
    password: str

class UserCreates(BaseModel):
    id: int
    username: str
    email: str
    password: str
 

class UserLogin(BaseModel):
    email: str
    password: str

class Feedbacks(BaseModel):
    email: str
    message1: str
    message2: str
    message3: str



diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))

# class PredictDiabetes(BaseModel):

# Create the FastAPI app
app = FastAPI()
#for signup page
# Define the signup endpoint
# @app.on_event("startup")
# async def startup():
#     await database.connect()

# @app.on_event("shutdown")
# async def shutdown():
#     await database.disconnect()
@app.post("/signup")
async def signup(userC: UserCreate):
    query = user.insert().values(username=userC.username, email=userC.email, password=userC.password)
    last_record_id = await database.execute(query)
    return {**userC.dict(), "id": last_record_id}
    # await
    # Create a new user object from the request body
    # db_user = User(username=user.username, email=user.email, password=user.password)
    
    # Add the user to the database
    # db = SessionLocal()
    # db.add(db_user)
    # db.commit()
    # db.refresh(db_user)
    
    # Return the newly created user
    # return {"message": db_user}


# @app.get("/")
# async def root():
#     raise HTTPException(status_code=404, detail="page not found")
#     return {}

# #for login page
@app.get("/login/") 
async def login(userL: UserLogin):
    # Get the user from the database by email
    db_user = user.select().where(user.c.email == userL.email)
    db_user_ = await database.fetch_one(db_user)
    #     db = SessionLocal()
#     db_user = db.query(User).filter(User.email == user.email).first()
    
#     # Check if the user exists and the password is correct
    # if not db_user or user.column.password != userL.password:
    #     raise HTTPException(status_code=401, detail="Invalid email or password")
    if db_user_.email != userL.email or db_user_.password != userL.password:
        raise HTTPException(status_code=401, detail="Invalid email or password")
   
#     # Return the user
    return {"message":"Signin Successful"}

class Profile(BaseModel):
    name: str
    email: str
    bio: str


# #for profile page
@app.put("/profile/")
async def create_profile(profile: Profile, profile_pic: UploadFile = File(...)):

    return {"profile": profile, "profile_pic_filename": profile_pic.filename}
@app.get("/profile/")
async def get_profiles():
    # db = SessionLocal()
    # profiles = db.query(User.email,User.username,)
    # db.close()
    profiles = user.select()
    db_profiles_ = await database.fetch_one(profiles)
    return{"message": db_profiles_}
class PasswordResetRequest(BaseModel):
    email: str

class UserOut(BaseModel):
    username: str
    email: str

# password_reset_requests = []

@app.get("/forgot-password/")
async def request_password_reset(request: PasswordResetRequest):
    # Retrieve user with matching email from database
    user_ = user.select().where(user.c.email == request.email)
    # db = SessionLocal()
    # user = db.query(User).filter(User.email == request.email).first()
    # db.close()

    if user_:
        # Send password reset email to the user's email address
        return {"message": "Password reset email sent"}
    else:
        return {"message": "User not found"}

# @app.get("/users/")
# async def list_users():
#     # Retrieve all users from database
#     db = SessionLocal()
#     users = db.query(User).all()
#     db.close()

#     # Return list of UserOut objects
#     return [UserOut(username=user.username, email=user.email, id=user.id) for user in users]

@app.delete("/users/{user_email}")
async def delete_user(user_email: str):
    user = user.delete().where(user_email.email == user.email)
    # Delete user with given ID from database
#     db = SessionLocal()
#     user = db.query(User).filter(User.email == user_email).first()
#     db.delete(user)
#     db.commit()
#     db.close()

    return {"message": "User deleted"}

@app.post("/feedback/")
def Feedback(feed_back: Feedbacks):
      db_feedback = feedback.insert().values(message1=feed_back.message1,message2=feed_back.message2,message3=feed_back.message3)
    #   db = SessionLocal()
    #   db_feedback = Feed(feed_back.email,feed_back.message1,feed_back.message2,feed_back.message3) 
    #   db.add(db_feedback)
    #   db.commit()
    #   db.refresh(db_feedback)
      return {"message": "Thank you for your feedback!"}

@app.post('/diabetes_prediction')
def diabetes_predd(input_parameters : model_input):
    
    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)
    
    preg = input_dictionary['pregnancies']
    glu = input_dictionary['Glucose']
    bp = input_dictionary['BloodPressure']
    skin = input_dictionary['SkinThickness']
    insulin = input_dictionary['Insulin']
    bmi = input_dictionary['BMI']
    dpf = input_dictionary['DiabetesPedigreeFunction']
    age = input_dictionary['Age']
    
    
    input_list = [preg, glu, bp, skin, insulin, bmi, dpf, age]
    
    prediction = diabetes_model.predict([input_list])
    
    if (prediction == 0):
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'
    return{"message" : "successful"} 
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
) 
