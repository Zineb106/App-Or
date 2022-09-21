#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 10:34:55 2022

@author: zineberrabi
"""

import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Main page",
    page_icon="✅",
    layout="wide",
)


with open("style.css") as f:
    st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

image = Image.open('appmultipages/pages/images/Orange.svg.png')
st.image(image, width = 100)


#new_title="<p style='font-family:serif; color:gis ; font-size: 42px;'>Solution ML pour la supervision</p>"
st.markdown("# Solution Machine Learning")
#st.sidebar.markdown("# Cette application est concu pour améliorer la supervision avec des algorithmes de machine learning")
#st.markdown(new_title, unsafe_allow_html=True)

new_title1 = "<p style='font-family:serif; color:black ; font-size: 20px;'>Cette application a été conçue pour la détection d'anomalie et la prevision sur les series temporelles en utilisant une approche de machine learning</p>"
st.sidebar.markdown(new_title1, unsafe_allow_html=True)




image1 = Image.open('appmultipages/pages/images/ML.jpeg')


st.image(image1, caption='Intelligence artificielle')

