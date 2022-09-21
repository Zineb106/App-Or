#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 11:44:10 2022

@author: zineberrabi
"""
#Importer les biblio 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import io
import streamlit as st
from pandas.api.types import is_string_dtype

import numpy
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import ExtraTreesRegressor
import pandas as pd
from pandas import to_datetime
from pandas import DataFrame
import plotly.express as px
from FeaturesExtraction import create_data_lag, make_prediction_from_num,make_prediction


st.set_page_config(
    page_title="Prediction Dashboard",
    page_icon="✅",
    layout="wide",
)

col1, mid, col2 = st.columns([1,1,10])
with col1:
    st.image('images/Orange.svg.png', width=100)
with col2:
    st.title("Dashboard - Prévision")


#Fonction d'extraction des champs ( heure , jour de la semaine , décalages)


uploaded_file = st.file_uploader("Upload your CSV Data ")
df=None

# Début du traitemet
if uploaded_file is None:
   uploaded_file=pd.read_csv('Data/Connect Pro - Entreprises_Groupes_Users-data-2022-09-16 09 54 33.csv')
   df=uploaded_file
   st.subheader("Démonstration sur le nombre d'utilisateurs Connect Pro ")
   

if uploaded_file is not None: 
    
    #Lire le fichier
    if df is None:
        try:
            df = pd.read_csv(uploaded_file)
        except : 
            df = pd.read_excel(uploaded_file ,engine='openpyxl')
    n=df.shape[1]
    
    if (n == 2):
   
        choix = [df.columns[0], df.columns[1]]
       
    if (n == 3): 
        
        choix = [df.columns[0], df.columns[1] , df.columns[2]]
        
    if(df.shape[1] == 4):
        
        choix = [df.columns[0], df.columns[1] , df.columns[2], df.columns[3]]
 
         
    if(df.shape[1] ==5 ):
        
        choix = [df.columns[0], df.columns[1] , df.columns[2], df.columns[3], df.columns[4]]
            
    if(df.shape[1] == 6):
        
        choix = [ df.columns[0], df.columns[1] , df.columns[2], df.columns[3], df.columns[4], df.columns[5] ]
    
    print(df.shape[1])        

    #proposer le choix d'attribut
    #st.subheader('Choix de la variable (numérique) a prédir :')
    choixF= st.selectbox('Choisissez une variable à prédir :' , choix)
    choixD= st.selectbox('indiquez la colonne Date :' , choix)
    
    
        
    if(choixD, choixF):
        
        df=df[[choixD, choixF]]
        
        
        if(is_string_dtype(df[choixF])):
            for i in range(df.shape[0]):
                x=df[choixF].iloc[i]
                if 'K' in x:
                    if len(x)>1:
                        df[choixF].iloc[i]=float(x.replace('K',''))*1000
                
                if ',' in x:
                    if len(x)>1:
                        df[choixF].iloc[i]=round(float(x.replace(',','.')),2)
                
        #renommer les colonnes
        df.columns = ['ds', 'y']
        df['ds']= to_datetime(df['ds'])
        df=df.interpolate()
        
        placeholder1 = st.empty()
        with placeholder1.container():
            
            fig55=px.line(df, x='ds', y='y', height=500)
            st.write(fig55, use_container_width = True )
            
        
        

        #st.write(df)
        dfC=create_data_lag(df,8)
        
        #st.write(dfC)
        
        st.write('Les données suivent une tendance de type :')
        option_1 = st.checkbox('générale (Trend) ')
        option_2 = st.checkbox('saisonnière')
        
        #day = st.slider('Nombre de jour à predir : ', min_value=1, max_value=60, step=2)
            
        X,y=dfC.drop(['y', 'ds'], axis=1), dfC['y']
     
        if(option_1) :
           
            
           
            
    
            model_linReg = LinearRegression()
           
        else:
            
            if(option_2):
                
                model_linReg = ExtraTreesRegressor(n_estimators=100)
                
        if(option_2 or option_1):
        
            model_linReg.fit(X,y)
            
            genre = st.radio(
                 "Comment voullez vous avoir l'information :",
                 ('Voir la prediction sur un nombre de jour', 'Prevoir la date où la valeur seras atteinte'))
        
            if genre =='Voir la prediction sur un nombre de jour':
                day = st.slider('Nombre de jour à predir : ', min_value=1, max_value=60, step=2)
                forecast = make_prediction(df,model_linReg,day)
                    
            if genre =='Prevoir la date où la valeur seras atteinte': 
                value_=st.text_input('Entrer la valeur')
                if(value_):
                #forecast = create_prediction_number(df, day,model_linReg)
                    forecast , val= make_prediction_from_num(df, model_linReg,int(value_))
                
                
                    st.text("la valeur "+value_ +" sera atteinte le : ",)
                    st.write(val)
                
                    

            try:    
            
                y_real = dfC[['y','ds']]
                
                
                df_pred = forecast[['ds', 'y']]
                
                print(df_pred)
                
                df_concate=pd.concat([y_real, df_pred])
                
                
                fig2 = px.line(y_real, x='ds', y='y')
                fig2.add_trace(go.Scatter(mode='lines',x=df_pred["ds"], y=round(df_pred["y"], 2), name="prediction"))
                ts_chart = st.plotly_chart(fig2, use_container_width = True, use_container_height = True)
            
            except Exception as e :
                print(e)
                


