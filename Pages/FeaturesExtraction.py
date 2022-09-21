#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 10:06:42 2022

@author: zineberrabi
"""
import pandas as pd
import numpy as np


def preprocess_data(dataframe):
    

    dataframe.columns=['time', 'value']
    dataframe['time']= pd.to_datetime(dataframe['time'])
    dataframe['hours']=dataframe['time'].dt.hour
    dataframe['DayOfWeek']=dataframe['time'].dt.dayofweek
    dataframe['sin_time'] = np.sin(2*np.pi*dataframe.hours/24)
    dataframe['cos_time'] = np.cos(2*np.pi*dataframe.hours/24)
    dataframe['DayOfWeek'][dataframe['DayOfWeek']<5]=1
    dataframe['DayOfWeek'][dataframe['DayOfWeek']>4]=0


    return dataframe

def create_data_lag(data,n):
  
  df=data.copy()

  for lag in range(1,n+1):
    df.loc[:,("val_"+str(lag))]=[1.0  for i in range(len(df))]
  for i in range(n,len(df)):
    for lag in range(1,n+1):
      df.loc[i,("val_"+str(lag))]=df.loc[i-lag,('y')].astype(float)
  
  #df['y']= df['y']
  df['hours']=df['ds'].dt.hour
  df['DayOfWeek']=df['ds'].dt.dayofweek
  df['DayOfWeek'][df['DayOfWeek']<5]=1
  df['DayOfWeek'][df['DayOfWeek']>4]=0
  df['sin_time'] = np.sin(2*np.pi*df.hours/24)
  df['cos_time'] = np.cos(2*np.pi*df.hours/24)
  
  return df[n:].drop(['hours'], axis=1)


#Fonction de prediction du jour à partir d'une valeur
def make_prediction_from_num(data,model,num):
    
    df=data.copy()
    lastdate=df['ds'].iloc[-1]
    h=6
    future=pd.date_range(lastdate , periods=1000 , freq= '2h')
      
    Y=list(df['y'])
    if(df.shape[1]<14):
      data_pred=pd.DataFrame(columns=['val-1','val-2','val-3','val-4','val-5','val-6','val-7','val-8','DayOfWeek','sin_time','cos_time','y','ds'])
    else :
      data_pred=pd.DataFrame(columns=['val-1','val-2','val-3','val-4','val-5','val-6','val-7','val-8','val-9','val-10','val-11','val-12','DayOfWeek','sin_time','cos_time','y','ds'])
    

    for i in range(0,len(future)):
      H=future[i].hour
      sin = np.sin(2*np.pi*H/24)
      cos = np.cos(2*np.pi*H/24)
      D=future[i].dayofweek
      D=[1 if(D<5) else 0]
      if(df.shape[1]<14):
        single_data= np.array([Y[-1],Y[-2],Y[-3],Y[-4],Y[-5],Y[-6],Y[-7],Y[-8],D[0],sin, cos]).reshape(1,11)
      else:
        single_data= np.array([Y[-1],Y[-2],Y[-3],Y[-4],Y[-5],Y[-6],Y[-7],Y[-8],Y[-9],Y[-10],Y[-11],Y[-12],D[0],sin, cos]).reshape(1,15)
      
      Y_prediction = model.predict(single_data)
      Y.append(round(Y_prediction[0],5))
      if(df.shape[1]<14):
        data_pred.loc[i]= [Y[-1],Y[-2],Y[-3],Y[-4],Y[-5],Y[-6],Y[-7],Y[-8],D[0],sin, cos, round(Y_prediction[0],5) ,future[i]]
      else:
        data_pred.loc[i]= [Y[-1],Y[-2],Y[-3],Y[-4],Y[-5],Y[-6],Y[-7],Y[-8],D[0],Y[-9],Y[-10],Y[-11],Y[-12],sin, cos, round(Y_prediction[0],5) ,future[i]]
     
      if(round(Y_prediction[0],5)>num):
          date=future[i]
          break
    
    return data_pred,date


#Fonction des predictions
def make_prediction(data,model,day):
  df=data.copy()
  lastdate=df['ds'].iloc[-1]
  h=6
  future=pd.date_range(lastdate , periods=day*12 , freq= '2h')

  Y=list(df['y'])
  if(df.shape[1]<14):
    data_pred=pd.DataFrame(columns=['val-1','val-2','val-3','val-4','val-5','val-6','val-7','val-8','DayOfWeek','sin_time','cos_time','y','ds'])
  else :
    data_pred=pd.DataFrame(columns=['val-1','val-2','val-3','val-4','val-5','val-6','val-7','val-8','val-9','val-10','val-11','val-12','DayOfWeek','sin_time','cos_time','y','ds'])
  for i in range(0,len(future)):
    H=future[i].hour
    sin = np.sin(2*np.pi*H/24)
    cos = np.cos(2*np.pi*H/24)
    D=future[i].dayofweek
    D=[1 if(D<5) else 0]
    if(df.shape[1]<14):
      single_data= np.array([Y[-1],Y[-2],Y[-3],Y[-4],Y[-5],Y[-6],Y[-7],Y[-8],D[0],sin, cos]).reshape(1,11)
    else:
      single_data= np.array([Y[-1],Y[-2],Y[-3],Y[-4],Y[-5],Y[-6],Y[-7],Y[-8],Y[-9],Y[-10],Y[-11],Y[-12],D[0],sin, cos]).reshape(1,15)
    
    Y_prediction = model.predict(single_data)
    Y.append(round(Y_prediction[0],5))
    if(df.shape[1]<14):
      data_pred.loc[i]= [Y[-1],Y[-2],Y[-3],Y[-4],Y[-5],Y[-6],Y[-7],Y[-8],D[0],sin, cos, round(Y_prediction[0],5) ,future[i]]
    else:
      data_pred.loc[i]= [Y[-1],Y[-2],Y[-3],Y[-4],Y[-5],Y[-6],Y[-7],Y[-8],D[0],Y[-9],Y[-10],Y[-11],Y[-12],sin, cos, round(Y_prediction[0],5) ,future[i]]
   

  return data_pred
