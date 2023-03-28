from fastapi import FastAPI ,HTTPException ,File, UploadFile,status
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Boolean, false
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
# from passlib.hash import bcrypt
import passlib.hash as _hash
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
import starlette.responses as _responses
import fastapi.security as _security
from auth import AuthHandler
import secrets
from fastapi.staticfiles import StaticFiles
from PIL import Image


app = FastAPI()

# _JWT_SECRET = ""

app.mount("/static", StaticFiles(directory="static"), name="static") 
auth_handler = AuthHandler()



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

DATABASE_URL = "postgresql://u5dvujpuqjert0:pf9f89717aad0c4f7610283a554c92312a53abc3472c6258ff0108fc38462d927@ec2-3-82-135-155.compute-1.amazonaws.com:5432/ddq46g29n1a8p4"
database = databases.Database(DATABASE_URL)

metadata = sqlalchemy.MetaData()


user = sqlalchemy.Table(
    "user",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),
    sqlalchemy.Column("first_name", sqlalchemy.String,unique=True),
    sqlalchemy.Column("last_name", sqlalchemy.String,unique=True),
    sqlalchemy.Column("email", sqlalchemy.String,unique=True),
    sqlalchemy.Column("password", sqlalchemy.String),
    sqlalchemy.Column("occupation", sqlalchemy.String),
    sqlalchemy.Column("house_address", sqlalchemy.String),
    sqlalchemy.Column("phone_number", sqlalchemy.String),
    sqlalchemy.Column("profile_pics", sqlalchemy.String),
    # sqlalchemy.Column("date_created", default = _dt.datetime.utcnow),    
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

def verify_password(self, password: str):
    return _hash.bycrypt.verify(password,self.hashed_password)
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
    first_name: str
    last_name:str
    email: str
    password: str

class UserCreates(BaseModel):
    id: int
    first_name:str
    last_name:str
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


@app.get("/")
async def root():
    # raise HTTPException(status_code=404, detail="page not found")
    return _responses.RedirectResponse("/docs")
# @app.get("/")
# async def root():
  #  raise HTTPException(status_code=404, detail="page not found")
   # return {}
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
    await database.connect()
    db_user_create = user.select().where(user.c.email == userC.email or user.c.first_name == userC.first_name or user.c.last_name == userC.last_name)
    db_user_create_ = await database.fetch_one(db_user_create)
    if db_user_create_ is None:
        hashed_password = auth_handler.get_password_hash(userC.password)
        query = user.insert().values(first_name=userC.first_name, last_name=userC.last_name, email=userC.email, password=hashed_password)
        last_record_id = await database.execute(query)
        token = auth_handler.encode_token(userC.email)
        return {**userC.dict(), "id": last_record_id, "status":'ok', "token": token}
    else:
        raise HTTPException(status_code=400, detail="user already exist") 
   
# #for login page
@app.post("/login")
async def login(userL: UserLogin):
    await database.connect()
    # Get the user from the database by email
    db_user = user.select().where(user.c.email == userL.email)
    db_user_ = await database.fetch_one(db_user)
    
    # if db_user_ is None or db_user_.password != userL.password :
    if db_user_ is None or not auth_handler.verify_password(userL.password, db_user_.password):
        raise HTTPException(status_code=401, detail="invalid email or password")
    else:
        token = auth_handler.encode_token(db_user_.email)
        return{"message":"Signin Successful","status":"ok", "token": token }
    
    # Return the user
    # return{db_user_.password,userL.password}
    # return {"message":"Signin Successful"}

class Profile(BaseModel):
    email: str
    occupation: str
    house_address: str
    phone_number: str

class Profiles(BaseModel):
    email:str
# #for profile page

# @app.post("/uploadfile/")
# async def create_upload_file(file: UploadFile):
#     await database.connect()
#     return {"filename": file_upload.filename}
@app.post("/uploadfile/profile")
async def create_upload_file(userC: Profiles, file: UploadFile = File(...)):
    # FILEPATH = "./static/images/"
    # filename = file.filename
    # extension = filename.split(".")[1]

    # if extension not in ["PNG", "JPG"]:
    #     return {"status":"error", "detail": "File extension not supported"}
    
    # token_name = secrets.token_hex(10)+"."+extension
    # generated_name = FILEPATH + token_name
    # file_content = await file.read()

    # with open(generated_name, "wb") as file:
    #     file.write(file_content)


    # img = Image.open(generated_name)    
    # img = img.resize(size =(200, 200))
    # img.save(generated_name)

    # file.close()

    # db_user = user.select().where(user.c.email == userC.email)
    # db_user_ = await database.fetch_one(db_user)

    # if db_user:
    #     profile_pics = token_name
    #     await profile_pics.save()

    # else:
    #     raise HTTPException(
    #         status_code=status.HTTP_401_UNAUTHORIZED,
    #         detail="Not authorised to perform this action",
    #         headers={"WWW-Athenticate": "Bearer"}
    #     )    
    # file_url = "https://diabetes-apis.herokuapp.com/" + generated_name[1:]
    return{"status": "ok", }



@app.put("/profile")
async def create_profile(profile: Profile, profile_pic: UploadFile):
    await database.connect()
    # file_upload = user.insert().values(profile_pics=profile_pic)
    # profiles = user.select().where(Profile.email == user.c.email)
    # db_profiles_ = await database.fetch_one(profiles)
    return {"profile": profile, "profile_pic_filename": profile_pic.filename}
@app.post("/profile")
async def get_profiles(userP: Profiles):
    await database.connect()
    # db = SessionLocal()
    # profiles = db.query(User.email,User.username,)
    # db.close()
    # query = user.insert().values(email=userP.email)
    profiles = user.select().where(userP.email == user.c.email)
    db_profiles_ = await database.fetch_one(profiles)
    return{"last_name": db_profiles_.last_name,"first_name": db_profiles_.first_name,"email": db_profiles_.email,"occupation":db_profiles_.occupation,"house_address":db_profiles_.house_address,"phone_number":db_profiles_.phone_number,"diabetes-type":db_profiles_.diabetes_type}
class PasswordResetRequest(BaseModel):
    email: str

class UserOut(BaseModel):
    username: str
    email: str

# password_reset_requests = []

@app.get("/forgot-password")
async def request_password_reset(request: PasswordResetRequest):
    await database.connect()
    # Retrieve user with matching email from database
    user_ = user.select().where(user.c.email == request.email)    
    db_user_ = await database.fetch_one(user_)
    # db = SessionLocal()
    # user = db.query(User).filter(User.email == request.email).first()
    # db.close()

    if db_user_.c.email == request.email:
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
    await database.connect()
    user = user.delete().where(user_email.email == user.email)
    # Delete user with given ID from database
#     db = SessionLocal()
#     user = db.query(User).filter(User.email == user_email).first()
#     db.delete(user)
#     db.commit()
#     db.close()

    return {"message": "User deleted"}

@app.post("/feedback")
async def Feedback(feed_back: Feedbacks):
      await database.connect()
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
    # return{"message" : "successful"} 
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
) 
