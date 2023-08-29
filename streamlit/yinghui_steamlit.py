#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
from PIL import Image
import requests
import io


api_url_image = "https://movie-genre-prediction-osp24vwspq-an.a.run.app/image_predict/"
api_url = "https://movie-genre-prediction-2-osp24vwspq-an.a.run.app/predict/"


st.markdown("""# Movie Genre Predictor :movie_camera:
""")

col1, col2 = st.columns(2)

with col1:
    st.header("Your Poster :camera:")
    uploaded_file = st.file_uploader("Upload image", type=['png', 'jpg'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image)
        image_data = uploaded_file.getvalue()
        files = {'file': ("image.jpg", io.BytesIO(image_data), "image/jpeg")}


    st.header("Your Synopsis :scroll:")
    txt = st.text_area("Enter your synopsis")
    both_button = st.button("Run Analysis")


with col2:
    if uploaded_file is not None and txt is not None and both_button :
        st.header("The Movie Genre is...")
        response = requests.post(api_url, params={"sypnosis":txt}, files=files)
        genre_result = response.json()["prediction"]
        output = ', '.join(genre_result)
        if 'Romance' in genre_result:
            output += ' :heart:'
        st.write(output)
        st.balloons()
