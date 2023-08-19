#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
from PIL import Image
import requests
import json
import pickle
from transformers import pipeline
import numpy as np
import base64


api_url_image = "https://movie-genre-prediction-osp24vwspq-an.a.run.app/image_predict/"

# def load_image_model():
#     model = "some model"
#     return model

# model = load_image_model()

st.markdown("""# Movie Genre Predictor
""")

col1, col2 = st.columns(2)

with col1:
    st.header("Your Poster")
    uploaded_file = st.file_uploader("Upload image", type=['png', 'jpg'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image)
        files = {'file': ('image.jpg', open(image, 'rb'))}
        response = requests.post(api_url_image, files=files)
        genre_result = response.json()["prediction"]

    st.header("Your Sypnosis")
    txt = st.text_area("Enter your sypnosis")
    sypnosis_button = st.button("Run plot analysis")
    if sypnosis_button:
        # params = txt
        st.write(txt)


with col2:
    if uploaded_file is not None:
        st.header("The Movie Genre is...")
        st.write(f"{genre_result}")
        st.balloons()
    # else:
    #     st.write("Please upload an image file")
    if sypnosis_button:
        st.header("The Movie Genre is...")
        st.write(f"genre_result from plot")
        st.balloons()

    if uploaded_file is not None and sypnosis_button:
        st.header("The Movie Genre is...")
        st.write(f"both")
        st.snow()