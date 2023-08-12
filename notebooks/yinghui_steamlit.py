#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
from PIL import Image
import requests
import json

api_url = ""

def load_image_model():
    model = "some model"
    return model

model = load_image_model()

st.markdown("""# Movie Genre Predictor
""")

col1, col2 = st.columns(2)

with col1:
    st.header("Your Poster")
    uploaded_file = st.file_uploader("Upload image", type=['png', 'jpg'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image)
        # response = requests.get(api_url, params=image)
        # genre_result = response.json()

    st.header("Your Sypnosis")
    txt = st.text_area("Enter your sypnosis")
    sypnosis_button = st.button("Run plot analysis")
    if sypnosis_button:
        params = txt
        display_txt = st.write(txt)
        display_txt
        # response = requests.get(api_url, params=image)
        # genre_result = response.json()


with col2:
    if uploaded_file is not None:
        st.header("The Movie Genre is...")
        st.write(f"genre_result from image")
        st.balloons()
    # else:
    #     st.write("Please upload an image file")
    if sypnosis_button:
        st.header("The Movie Genre is...")
        st.write(f"genre_result from plot")
        st.balloons()


# st.markdown("""# This is a header
# ## This is a sub header
# This is text""")

# df = pd.DataFrame({
#     'first column': list(range(1, 11)),
#     'second column': np.arange(10, 101, 10)
# })

# # this slider allows the user to select a number of lines
# # to display in the dataframe
# # the selected value is returned by st.slider
# line_count = st.slider('Select a line count', 1, 10, 3)

# # and used to select the displayed lines
# head_df = df.head(line_count)

# head_df
