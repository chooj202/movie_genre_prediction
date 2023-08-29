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
import io


api_url_image = "https://movie-genre-prediction-osp24vwspq-an.a.run.app/image_predict/"
api_url = "https://movie-genre-prediction-2-osp24vwspq-an.a.run.app/predict/"


st.markdown("""# Movie Genre Predictor :movie_camera:
""")

col1, col2 = st.columns(2)

with col1:
    st.header("Your Poster :camera:")
    uploaded_file = st.file_uploader("Upload image", type=['png', 'jpg'])
    # poster_button = st.button("Run poster analysis")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image)
        image_data = uploaded_file.getvalue()
        files = {'file': ("image.jpg", io.BytesIO(image_data), "image/jpeg")}


    st.header("Your Sypnosis :scroll:")
    txt = st.text_area("Enter your sypnosis")
    # sypnosis_button = st.button("Run plot analysis")
    # if sypnosis_button:
    #     # params = txt
    #     st.write(txt)

    both_button = st.button("Run Analysis")


with col2:
    # if uploaded_file is not None and poster_button:
    #     st.header("The Movie Genre is...")
    #     st.write(f"{genre_result}")
    #     st.balloons()
    # # else:
    # #     st.write("Please upload an image file")
    # if txt is not None and sypnosis_button:
    #     st.header("The Movie Genre is...")
    #     st.write(f"genre_result from plot")
    #     st.balloons()

    if uploaded_file is not None and txt is not None and both_button :
        st.header("The Movie Genre is...")
        st.write(f"not horror")
        response = requests.post(api_url, params={"sypnosis":txt}, files=files)
        genre_result = response.json()["prediction"]
        st.write(genre_result)
        st.write(genre_result[0:])
        st.balloons()
