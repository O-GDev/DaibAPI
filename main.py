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
import aiofiles
import uuid
from starlette.requests import Request
from fastapi.templating import Jinja2Templates
from sklearn.linear_model import LogisticRegression
import datetime
import time
import asyncio


app = FastAPI()
# templates = Jinja2Templates(directory="template")
# the filename of the saved model
# filename = 'diabetes_model.sav'
# load the saved model
# loaded_model = pickle.load(open(filename, 'rb'))
# _JWT_SECRET = ""

# app.mount("/static", StaticFiles(directory="static"), name="static") 
auth_handler = AuthHandler()
BASEDIR = os.path.dirname(__file__)



diabetes_dataset = pd.read_csv('diabetes.csv') 

# print(diabetes_dataset.head())
diabetes_dataset.head()

X = diabetes_dataset.drop(columns = 'Outcome', axis=1)

Y = diabetes_dataset['Outcome']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2,)

# print(X.shape, X_train.shape, X_test.shape)

classifier = svm.SVC(kernel='linear', probability=True)

# classifier = LogisticRegression(probability=True)
classifier.fit(X_train, Y_train)

X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

# print('Accuracy score of the training data : ', training_data_accuracy)

X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

# print('Accuracy score of the test data : ', test_data_accuracy)


filename = 'diabetes_model.sav'
pickle.dump(classifier, open(filename, 'wb'))

DATABASE_URL = "postgresql://biqxjsxynyhqin:2c11b52c42eb4e08c70ea9b178b9e1ae4996a0744e79bd6eed6df72dfadf50f8@ec2-107-21-67-46.compute-1.amazonaws.com:5432/d68l9na6c3l27c"
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
    sqlalchemy.Column("profile_pics", sqlalchemy.String,),
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

Remind = sqlalchemy.Table(
    "Remind",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),
    sqlalchemy.Column("setdate", sqlalchemy.String),
    sqlalchemy.Column("message", sqlalchemy.String),
    # sqlalchemy.Column("message3", sqlalchemy.String),
)

engine = sqlalchemy.create_engine(
    DATABASE_URL
)

metadata.create_all(engine)

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

class Reminder(BaseModel):
    date: str
    message: str
    email: str


diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))

# class PredictDiabetes(BaseModel):


@app.get("/")
async def root():
    # raise HTTPException(status_code=404, detail="page not found")
    return _responses.RedirectResponse("/docs")
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
        last_record_id = await database.execute(db_user)
        token = auth_handler.encode_token(db_user_.email)
        return{"status":"ok", "token": token, "id": last_record_id, **userL.dict()}
    
    # Return the user
    # return{db_user_.password,userL.password}
    # return {"message":"Signin Successful"}

class Profile(BaseModel):
    email: str
    occupation: str
    house_address: str
    phone_number: str
    # image: str

class Profiles(BaseModel):
    email:str
    # image:str
# @app.put("/profile")
# async def update_profiles(profU: Profile):
#     return{} occupation=profU.occupation,house_address=profU.house_address,phone_number=profU.phone_number

async def handle_file_upload(file: UploadFile) -> str: 
    _, ext = os.path.splitext(file.filename)
    img_dir = os.path.join(BASEDIR, 'statics/media')
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    content = await file.read()
    if file.content_type not in ['image/jpeg', 'image/png']:
        raise HTTPException(status_code=406, detail="Only .jpeg or .png  files allowed")
    file_name = f'{uuid.uuid4().hex}{ext}'
    async with aiofiles.open(os.path.join(img_dir, file_name), mode='wb') as f:
        await f.write(content)

    return file_name
# https://youtu.be/UNFDILca9M8
# https://www.youtube.com/watch?v=aFtYsghw-1k
# https://youtu.be/1GpOS5mrGHI
@app.patch("/profile_picture/{id}")
async def update_profile(UserP:Profile,image: UploadFile = File(...)):
    await database.connect()
    Images = await handle_file_upload(image)
    # auth = await login(token)
    prof = user.insert().values(profile_pics = Images,occupation = UserP.occupation,house_address = UserP.house_address,
                                phone_number = UserP.phone_number).where(UserP.email == user.c.email)
    # db_prof_ = await database.fetch_one(prof)    
    db_user = user.select().where(UserP.email == user.c.email)
    db_user_ = await database.fetch_one(db_user)
    return {db_user_.dict()}


@app.post("/profile")
async def get_profiles(userP: Profiles):
    await database.connect()
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
    if db_user_ is None:
        # Send password reset email to the user's email address
        return {"message": "User not found"}
    else:
        return {"message": "Password reset email sent"}

@app.delete("/users/{user_email}")
async def delete_user(user_email: str):
    await database.connect()
    user = user.delete().where(user_email.email == user.email)
    return {"message": "User deleted"}

@app.post("/feedback")
async def Feedback(feed_back: Feedbacks):
      await database.connect()
      db_feedback = feedback.insert().values(message1=feed_back.message1,message2=feed_back.message2,message3=feed_back.message3)
      return {"message": "Thank you for your feedback!"}

# @app.post('/diabetes_prediction')
# def diabetes_predd(input_parameters : model_input):
    
#     input_data = input_parameters.json()
#     input_dictionary = json.loads(input_data)
    
#     preg = input_dictionary['pregnancies']
#     glu = input_dictionary['Glucose']
#     bp = input_dictionary['BloodPressure']
#     skin = input_dictionary['SkinThickness']
#     insulin = input_dictionary['Insulin']
#     bmi = input_dictionary['BMI']
#     dpf = input_dictionary['DiabetesPedigreeFunction']
#     age = input_dictionary['Age']
    
    
#     input_list = [preg, glu, bp, skin, insulin, bmi, dpf, age]
    
#     prediction = diabetes_model.predict([input_list])
    
#     if (prediction == 0):
#         return 'The person is not diabetic'
#     else:
#         return 'The person is diabetic'
    # return{"message" : "successful"} 
    

# @app.post('/predict')
# async def predict(input_parameters : model_input):
#     result = {}
#     # if request.method == "POST":
#         # get the features to predict
#         # form = await request.form()
#         # form data
#     input_data = input_parameters.json()
#     input_dictionary = json.loads(input_data)
    
#     preg = input_dictionary['pregnancies']
#     glu = input_dictionary['Glucose']
#     bp = input_dictionary['BloodPressure']
#     skin = input_dictionary['SkinThickness']
#     insulin = input_dictionary['Insulin']
#     bmi = input_dictionary['BMI']
#     dpf = input_dictionary['DiabetesPedigreeFunction']
#     age = input_dictionary['Age']

#     input_list = [preg, glu, bp, skin, insulin, bmi, dpf, age]
    
#     prediction = diabetes_model.predict([input_list])

#     confidence = diabetes_model.predict_proba([input_list])
#     result['confidence'] = str(round(np.amax(confidence[0]) * 100 ,2))
        
    
#     if (prediction[0] == 0):
#         return 'The person is not diabetic'
#     else:
#         return {"message": result}
@app.post('/predict')
async def predict(input_parameters : model_input):
    # result = {}
    # if request.method == "POST":
        # get the features to predict
        # form = await request.form()
        # form data
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

    confidence = diabetes_model.predict_proba([input_list])
    
    result = np.amax(confidence[0])

    res = (result * 100)
     
    resi = round(res, 2 ) 
        
    
    if (prediction[0] == 0):
        return {"message":"The person is not diabetic","status":"notit"}
    else:
        return {"message": resi,"status":"it"}



        



      

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
) 
