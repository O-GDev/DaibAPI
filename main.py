from fastapi import FastAPI ,HTTPException ,File, UploadFile,status, Depends,Response
from sqlalchemy import create_engine, Column, Integer, String, Boolean, false
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
import json as js
from starlette.staticfiles import StaticFiles
import schemas, models, oauth2
from database import engine, get_db, SessionLocal 
from sqlalchemy.orm import Session
from fastapi.security.oauth2 import OAuth2PasswordRequestForm
from typing import Annotated




models.Base.metadata.create_all(bind=engine)
app = FastAPI(title="Diabetes Prediction API")
# templates = Jinja2Templates(directory="template")
# the filename of the saved model
# filename = 'diabetes_model.sav'
# load the saved model
# loaded_model = pickle.load(open(filename, 'rb'))
# _JWT_SECRET = ""


# app.mount("/static", StaticFiles(directory="static"), name="static") 
auth_handler = AuthHandler()
BASEDIR = os.path.dirname(__file__)
app.mount("/static", StaticFiles(directory=BASEDIR + "/statics"), name="static")
# app.mount("/statics", StaticFiles(directory=BASEDIR + "/statics"), name="statics")


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

# DATABASE_URL = "postgresql://biqxjsxynyhqin:2c11b52c42eb4e08c70ea9b178b9e1ae4996a0744e79bd6eed6df72dfadf50f8@ec2-107-21-67-46.compute-1.amazonaws.com:5432/d68l9na6c3l27c"
# database = databases.Database(DATABASE_URL)

# metadata = sqlalchemy.MetaData()


# user = sqlalchemy.Table(
#     "user",
#     metadata,
#     sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),
#     sqlalchemy.Column("first_name", sqlalchemy.String,unique=True),
#     sqlalchemy.Column("last_name", sqlalchemy.String,unique=True),
#     sqlalchemy.Column("email", sqlalchemy.String,unique=True),
#     sqlalchemy.Column("password", sqlalchemy.String),
#     sqlalchemy.Column("occupation", sqlalchemy.String),
#     sqlalchemy.Column("house_address", sqlalchemy.String),
#     sqlalchemy.Column("phone_number", sqlalchemy.String),
#     sqlalchemy.Column("profile_pics", sqlalchemy.String,),
#     # sqlalchemy.Column("date_created", default = _dt.datetime.utcnow),    
# )



# feedback = sqlalchemy.Table(
#     "feedback",
#     metadata,
#     sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),
#     sqlalchemy.Column("email", sqlalchemy.String),
#     sqlalchemy.Column("message1", sqlalchemy.String),
#     sqlalchemy.Column("message2", sqlalchemy.String),
#     sqlalchemy.Column("message3", sqlalchemy.String),
# )



# engine = sqlalchemy.create_engine(
#     DATABASE_URL,
# )
# Base = declarative_base()
# metadata = sqlalchemy.MetaData()

# metadata.create_all(engine)

origins = [
"http://192.168.43.177:8000"
]


    

def verify_password(self, password: str):
    return _hash.bycrypt.verify(password,self.hashed_password)

# Define the request body schema



diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))

# class PredictDiabetes(BaseModel):


@app.get("/")
async def root():
    # raise HTTPException(status_code=404, detail="page not found")
    return _responses.RedirectResponse("/docs")

@app.post("/signup",response_model=schemas.UserResponse, status_code=status.HTTP_200_OK)
async def signup(userC:schemas.UserCreate,db: Session = Depends(get_db)):
    db_user_create = db.query(models.User).filter(userC.first_name == models.User.first_name,userC.last_name == models.User.last_name,userC.email == models.User.email).first()
    # db_user_create_ = await database.fetch_one(db_user_create)
    if db_user_create is None:
        hashed_password = auth_handler.get_password_hash(userC.password)
        query = models.User(first_name = userC.first_name,last_name = userC.last_name,email = userC.email,password = hashed_password)
        db.add(query)
        db.commit()
        db.refresh(query)
        return {"":query,"status":status.HTTP_200_OK}
        # return Response(status_code=status.HTTP_200_OK)               
    else:
        raise HTTPException(status_code=400, detail="user already exist")
        
    # token = auth_handler.encode_token(userC.email)
    # return {**userC.dict(),}

# #for login page
@app.post("/login",response_model=schemas.Token, status_code=status.HTTP_200_OK)
async def login(userL: Annotated[OAuth2PasswordRequestForm, Depends()],db: Session = Depends(get_db)):
    # Get the user from the database by email
    db_user_ =  db.query(models.User).filter(userL.username == models.User.email).first()
     
    # if db_user_ is None or db_user_.password != userL.password :
    if db_user_ is None or not auth_handler.verify_password(userL.password, db_user_.password):
        raise HTTPException(status_code=401, detail="invalid email or password")
    else:
        # token = auth_handler.encode_token(db_user_.email)
        access_token = oauth2.create_access_token(data={"user_id": db_user_.id})
        # status = Response(status_code=status.HTTP_200_OK)
        return {"access_token": access_token, "token_type":"bearer", "status":status.HTTP_200_OK}
    
    # Return the user
    # return{db_user_.password,userL.password}
#     # return {"message":"Signin Successful"}

async def handle_file_upload(file: UploadFile) -> str:
    _, ext = os.path.splitext(file.filename)
    img_dir = os.path.join(BASEDIR, 'statics/')
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    content = await file.read()
    if file.content_type not in ['image/jpeg', 'image/png']:
        raise HTTPException(status_code=406, detail="Only .jpeg or .png  files allowed")
    file_name = f'{uuid.uuid4().hex}{ext}'
    async with aiofiles.open(os.path.join(img_dir, file_name), mode='wb') as f:
        await f.write(content)

    return file_name

@app.put("/profile_picture", status_code=status.HTTP_200_OK)
async def update_profile(image: UploadFile = File(...),db: Session = Depends(get_db),get_current_user: int = Depends(oauth2.get_current_user)):
    Images = await handle_file_upload(image)

    user = db.query(models.User).filter(get_current_user.id == models.User.id)

    if user.first() == None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, 
                            detail=f"user does not exist") 
    else:
        user.profile_pics = Images
        # user.occupation = UserP.occupation
        # user.house_address = UserP.house_address
        # user.phone_number = UserP.phone_number
        # user.update(user.dict(exclude_unset=True),synchronize_session=False)
        db.commit()
        # print(Images)
        # db.refresh(user)  
        return {"message":"successful","": user}
        
 
@app.post("/feedback", status_code=status.HTTP_200_OK)
async def Feedback(feed_back: schemas.Feedbacks,db: Session = Depends(get_db),get_current_user: int = Depends(oauth2.get_current_user)):
    #   await database.connect() user = db.query(models.User).filter(get_current_user.id == models.User.id)
    user = db.query(models.User).filter(get_current_user.id == models.User.id)
    if user.first() == None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, 
                            detail=f"user does not exist") 
    else:
        query = models.Feedback(message1 = feed_back.message1,message2 = feed_back. message2,message3 = feed_back.message3)
        db.add(query)
        db.commit()
        db.refresh(query)
        return Response(status_code=status.HTTP_200_OK)


@app.post('/predict', status_code=status.HTTP_200_OK)
async def predict(input_parameters : schemas.model_input,get_current_user: int = Depends(oauth2.get_current_user)):
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


@app.get("/profile", status_code=status.HTTP_200_OK)
async def get_profiles(db: Session = Depends(get_db),get_current_user: int = Depends(oauth2.get_current_user)):
    db_profiles_ =  db.query(models.User).filter(get_current_user.id == models.User.id).first()
    return db_profiles_
# {"last_name": db_profiles_.last_name,"first_name": db_profiles_.first_name,"email": db_profiles_.email,"occupation":db_profiles_.occupation,"house_address":db_profiles_.house_address,"phone_number":db_profiles_.phone_number,"diabetes-type":db_profiles_.diabetes_type}

@app.get("/forgot-password", status_code=status.HTTP_200_OK)
async def request_password_reset(request: schemas.PasswordResetRequest,db: Session = Depends(get_db),get_current_user: int = Depends(oauth2.get_current_user)):
    # Retrieve user with matching email from database
    user_ = db.query(models.User).filter(request.email == models.User.email)
    if user_ is None:
        # Send password reset email to the user's email address
        return {"message": "User not found"}
    else:
        return {"message": "Password reset email sent"}

@app.delete("/users/{user_email}", status_code=status.HTTP_200_OK) 
async def delete_user(user_email: str,db: Session = Depends(get_db),get_current_user: int = Depends(oauth2.get_current_user)):
    
    user = db.query(models.User).filter(get_current_user.id == models.User.id)

    if user.first() == None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, 
                            detail=f"user {user_email} does not exist") 
    else:
        user.delete(synchronize_session=False)
        db.commit()
        db.refresh(user)
        return {"message": "User deleted",}


             
# @app.post("/signup")
# async def signup(userC: schemas.UserCreate):
#     await database.connect()
#     db_user_create = user.select().where(user.c.email == userC.email or user.c.first_name == userC.first_name or user.c.last_name == userC.last_name)
#     db_user_create_ = await database.fetch_one(db_user_create)
#     if db_user_create_ is None:
#         hashed_password = auth_handler.get_password_hash(userC.password)
#         query = user.insert().values(first_name=userC.first_name, last_name=userC.last_name, email=userC.email, password=hashed_password)
#         last_record_id = await database.execute(query)
#         token = auth_handler.encode_token(userC.email)
#         return {**userC.dict(), "id": last_record_id, "status":'ok', "token": token}
#     else:
#         raise HTTPException(status_code=400, detail="user already exist") 
   
# # #for login page
# @app.post("/login",response_model=schemas.UserLoginResponse)
# async def login(userL: schemas.UserLogin):
#     await database.connect()
#     # Get the user from the database by email
#     db_user = user.select().where(user.c.email == userL.email)
#     db_user_ = await database.fetch_one(db_user)
    
#     # if db_user_ is None or db_user_.password != userL.password :
#     if db_user_ is None or not auth_handler.verify_password(userL.password, db_user_.password):
#         raise HTTPException(status_code=401, detail="invalid email or password")
#     else:
#         last_record_id = await database.execute(db_user)
#         token = auth_handler.encode_token(db_user_.email)
#         return{"status":"ok", "token": token, "id": last_record_id, **userL.dict()}
    
#     # Return the user
#     # return{db_user_.password,userL.password}
#     # return {"message":"Signin Successful"}


#     # image:str
# # @app.put("/profile")
# # async def update_profiles(profU: Profile):
# #     return{} occupation=profU.occupation,house_address=profU.house_address,phone_number=profU.phone_number

# async def handle_file_upload(file: UploadFile) -> str: 
#     _, ext = os.path.splitext(file.filename)
#     img_dir = os.path.join(BASEDIR, 'statics/media')
#     if not os.path.exists(img_dir):
#         os.makedirs(img_dir)
#     content = await file.read()
#     if file.content_type not in ['image/jpeg', 'image/png']:
#         raise HTTPException(status_code=406, detail="Only .jpeg or .png  files allowed")
#     file_name = f'{uuid.uuid4().hex}{ext}'
#     async with aiofiles.open(os.path.join(img_dir, file_name), mode='wb') as f:
#         await f.write(content)

#     return file_name
# # https://youtu.be/UNFDILca9M8
# # https://www.youtube.com/watch?v=aFtYsghw-1k
# # https://youtu.be/1GpOS5mrGHI


# @app.patch("/profile_picture/{id}")
# async def update_profile(UserP:schemas.Profile,image: UploadFile = File(...)):
#     await database.connect()
#     Images = await handle_file_upload(image)
#     # auth = await login(token)
#     user.insert().values(profile_pics = Images,occupation = UserP.occupation,house_address = UserP.house_address,
#                                 phone_number = UserP.phone_number).where(UserP.email == user.c.email)
#     # db_prof_ = await database.fetch_one(prof)    
#     db_user = user.select().where(UserP.email == user.c.email)
#     db_user_ = await database.fetch_one(db_user)
#     return {db_user_.dict()}


# @app.post("/profile")
# async def get_profiles(userP: schemas.Profiles):
#     await database.connect()
#     profiles = user.select().where(userP.email == user.c.email)
#     db_profiles_ = await database.fetch_one(profiles)
#     return{"last_name": db_profiles_.last_name,"first_name": db_profiles_.first_name,"email": db_profiles_.email,"occupation":db_profiles_.occupation,"house_address":db_profiles_.house_address,"phone_number":db_profiles_.phone_number,"diabetes-type":db_profiles_.diabetes_type}



# # password_reset_requests = []

# @app.get("/forgot-password")
# async def request_password_reset(request: schemas.PasswordResetRequest):
#     await database.connect()
#     # Retrieve user with matching email from database
#     user_ = user.select().where(user.c.email == request.email)    
#     db_user_ = await database.fetch_one(user_)
#     if db_user_ is None:
#         # Send password reset email to the user's email address
#         return {"message": "User not found"}
#     else:
#         return {"message": "Password reset email sent"}

# @app.delete("/users/{user_email}")
# async def delete_user(user_email: str):
#     # await database.connect()
#     user = user.delete().where(user_email.email == user.email)
#     return {"message": "User deleted"}

# @app.post("/feedback")
# async def Feedback(feed_back: schemas.Feedbacks):
#     #   await database.connect()
#       db_feedback = feedback.insert().values(message1=feed_back.message1,message2=feed_back.message2,message3=feed_back.message3)
#       return {"message": "Thank you for your feedback!"}

# # @app.post('/diabetes_prediction')
# # def diabetes_predd(input_parameters : model_input):
    
# #     input_data = input_parameters.json()
# #     input_dictionary = json.loads(input_data)
    
# #     preg = input_dictionary['pregnancies']
# #     glu = input_dictionary['Glucose']
# #     bp = input_dictionary['BloodPressure']
# #     skin = input_dictionary['SkinThickness']
# #     insulin = input_dictionary['Insulin']
# #     bmi = input_dictionary['BMI']
# #     dpf = input_dictionary['DiabetesPedigreeFunction']
# #     age = input_dictionary['Age']
    
    
# #     input_list = [preg, glu, bp, skin, insulin, bmi, dpf, age]
    
# #     prediction = diabetes_model.predict([input_list])
    
# #     if (prediction == 0):
# #         return 'The person is not diabetic'
# #     else:
# #         return 'The person is diabetic'
#     # return{"message" : "successful"} 
    

# # @app.post('/predict')
# # async def predict(input_parameters : model_input):
# #     result = {}
# #     # if request.method == "POST":
# #         # get the features to predict
# #         # form = await request.form()
# #         # form data
# #     input_data = input_parameters.json()
# #     input_dictionary = json.loads(input_data)
    
# #     preg = input_dictionary['pregnancies']
# #     glu = input_dictionary['Glucose']
# #     bp = input_dictionary['BloodPressure']
# #     skin = input_dictionary['SkinThickness']
# #     insulin = input_dictionary['Insulin']
# #     bmi = input_dictionary['BMI']
# #     dpf = input_dictionary['DiabetesPedigreeFunction']
# #     age = input_dictionary['Age']

# #     input_list = [preg, glu, bp, skin, insulin, bmi, dpf, age]
    
# #     prediction = diabetes_model.predict([input_list])

# #     confidence = diabetes_model.predict_proba([input_list])
# #     result['confidence'] = str(round(np.amax(confidence[0]) * 100 ,2))
        
    
# #     if (prediction[0] == 0):
# #         return 'The person is not diabetic'
# #     else:
# #         return {"message": result}


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
) 
# web: gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app







