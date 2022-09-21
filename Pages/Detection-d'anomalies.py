#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 10:53:51 2022

@author: zineberrabi
"""

import time  # to simulate a real time data, time loop

import numpy as np  # np mean, np random
import pandas as pd  # read csv, df manipulation
import plotly.express as px  # interactive charts
import streamlit as st  # 🎈 data web app development



from datetime import timedelta
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn import metrics


#from feature_engineering import preprocess_data
import plotly.graph_objects as go
import datetime

from FeaturesExtraction import preprocess_data

st.set_page_config(
    page_title="Detection d'anomalie Dashboard",
    page_icon="✅",
    layout="wide",
)

data = pd.read_csv('Data/Nombre total de sessions-actives_OR.csv')


data=preprocess_data(data)
df=data.copy()

df['value']=StandardScaler().fit_transform(np.array(df['value']).reshape(-1,1))





col1, mid, col2 = st.columns([1,1,15])
with col1:
    st.image('images/Orange.svg.png', width=100)
with col2:
    st.title("Dashboard - Détection d'anomalie ")



# creating a single-element container
placeholder = st.empty()

placeholder1 = st.empty()

placeholder2 = st.empty()


# prepare  data and algorithm prediction

X=df.drop(['time'], axis=1)
clf =  DBSCAN(eps=0.21,min_samples=6).fit(X)
data['anomaly']=clf.fit_predict(X)

nw=data[data['anomaly']==-1]


with placeholder.container():
    st.markdown("### Nombre de session actives")
        
    fig = px.line(data, x='time', y='value', height=500)
    fig.add_trace(go.Scatter(mode="markers", x=nw["time"], y=nw["value"], name="anomaly"))
    ts_chart = st.plotly_chart(fig, use_container_width = True)
    
    

with placeholder1.container():

    # create three columns
 

    # create two columns for charts
    fig_col1, fig_col2 = st.columns(2)
    with fig_col1:
       
        fig55 = px.box(data, y="value" , points="all")
        st.write(fig55 , height=500)
   
    with fig_col2:
        
       st.markdown("### Detailed Data View")
       st.dataframe(df , height=500)
       time.sleep(1)


