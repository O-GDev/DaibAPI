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
from predictDiabetes import PredictDiabetesHandle
import emailUtil



models.Base.metadata.create_all(bind=engine)
app = FastAPI(title="Diabetes Prediction API")
auth_handler = AuthHandler()
BASEDIR = os.path.dirname(__file__)
app.mount("/statics", StaticFiles(directory=BASEDIR + "/statics/"), name="statics")

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
async def update_profile(image: UploadFile = File(...),db: Session = Depends(get_db),
                         get_current_user: int = Depends(oauth2.get_current_user),):
    Images = await handle_file_upload(image)

    user = db.query(models.User).filter(get_current_user.id == models.User.id).first()

    if user == None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, 
                            detail=f"user does not exist") 
    elif(Images == None):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, 
                            detail=f"Empty Field") 
    else:
        user.profile_pics = Images
        db.commit()
        return {"status":status.HTTP_200_OK,"user_details": user.profile_pics}
        
@app.put("/profiles")
async def update_profiles_details(UserP:schemas.Profile,db: Session = Depends(get_db),get_current_user: int = Depends(oauth2.get_current_user)):
      user = db.query(models.User).filter(get_current_user.id == models.User.id).first()


      if user == None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, 
                            detail=f"user does not exist") 
      elif(UserP == None):
          return HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                               detail=f"Empty Field")
      else:
        user.occupation = UserP.occupation
        user.house_address = UserP.house_address
        user.phone_number = UserP.phone_number
        db.commit()
        return {"status":status.HTTP_200_OK}


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
async def predict(input: schemas.model_input,get_current_user: int = Depends(oauth2.get_current_user)):
    user_data = [[input.preg, input.glu, input.bp, input.skin, input.insulin, input.bmi, input.dpf, input.age]]  
    prediction = await PredictDiabetesHandle(user_data)
    return  {"message": prediction}



@app.get("/profile", status_code=status.HTTP_200_OK)
async def get_profiles(db: Session = Depends(get_db),get_current_user: int = Depends(oauth2.get_current_user)):
    db_profiles_ =  db.query(models.User).filter(get_current_user.id == models.User.id).first()
    return db_profiles_
# {"last_name": db_profiles_.last_name,"first_name": db_profiles_.first_name,"email": db_profiles_.email,"occupation":db_profiles_.occupation,"house_address":db_profiles_.house_address,"phone_number":db_profiles_.phone_number,"diabetes-type":db_profiles_.diabetes_type}

# async def HandleRequestPassword():

@app.post("/forgot-password", status_code=status.HTTP_200_OK)
async def request_password_reset(request: schemas.PasswordResetRequest,db: Session = Depends(get_db),):
    # Retrieve user with matching email from database
    user_ = db.query(models.User).filter(request.email == models.User.email)
    if user_ is None:
        return {"message": "User not found"}
    else:
        reset_code = str(uuid.uuid1())
        user_.token = reset_code
        db.commit() 
        # Send password reset email to the user's email address
        subject = f"Hello {user_.first_name}"
        recipient = [request.email]
        message = """
        <!Doctype html>
        <html>
        <title>Reset Password</title>
        <body>
        <div style="width:100%;font-family: monospace;">
        <h1>Hello, {0:}</h1>
        <p>Someone requested for a password reset link. If you are aware of this, Click the button right below this, Otherwise Ignore!</p>
        <a href="https://diabetes-prediction-api.herokuapp.com/forgot-password?reset_password_token={1:}" style="box-sizing:border-box;border-color:light-blue"></a>
        </div>
        </body>
        </html>
        """.format(user_.first_name, reset_code)

        await emailUtil(subject, recipient, message)
        return {
            "status":status.HTTP_200_OK,
            "message": "we've sent a reset link to your email"
            }

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


        
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
) 



