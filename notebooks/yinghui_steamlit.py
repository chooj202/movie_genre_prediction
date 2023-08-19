#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
from PIL import Image
import requests
import json
import pickle
# from transformers import pipeline
import numpy as np
import torch








# Load Models for NLP
with open('/home/yinghui/code/chooj202/movie_genre_prediction/movie_genre_prediction/raw_data/trained_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)
with open('/home/yinghui/code/chooj202/movie_genre_prediction/movie_genre_prediction/raw_data/tokenizer.pkl', 'rb') as f:
    loaded_tokenizer = pickle.load(f)

# text_classification = pipeline(
#     "text-classification",
#     model=loaded_model,
#     tokenizer=loaded_tokenizer
# )



#Load Models for Image Detection
model_image = pickle.load(open('image_reg.pkl','rb'))

#Preprocess Image
def preprocess_image(image):
    image = image.resize((256, 256))
    image_np = np.array(image)
    return image_np

#Load Models for both
model_both = pickle.load(open('both.pkl','rb'))


api_url = ""

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
        processed_uploaded_file = preprocess_image(image)
        # response = requests.get(api_url, params=image)
        # genre_result = response.json()

    st.header("Your Sypnosis")
    txt = st.text_area("Enter your sypnosis")
    sypnosis_button = st.button("Run plot analysis")
    if sypnosis_button:
        # params = txt
        st.write(txt)
        # result = text_classification(txt)

        # #Preprocess NLP
        # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        # labels = ['Action',
        #             'Adventure',
        #             'Comedy',
        #             'Crime',
        #             'Fantasy',
        #             'Horror',
        #             'Mystery',
        #             'Romance',
        #             'Sci-Fi',
        #             'Thriller']
        # id2label = {idx:label for idx, label in enumerate(labels)}

        # encoding = tokenizer(txt, return_tensors="pt")
        # encoding = {k: v.to(model_nlp.model.device) for k,v in encoding.items()}
        # outputs = model_nlp.model(**encoding)
        # logits = outputs.logits
        # sigmoid = torch.nn.Sigmoid()
        # probs = sigmoid(logits.squeeze().cpu())
        # predictions = np.zeros(probs.shape)
        # predictions[np.where(probs >= 0.5)] = 1
        # # turn predicted id's into actual label names
        # predicted_labels = [id2label[idx] for idx, label in enumerate(predictions) if label == 1.0]

        # response = requests.get(api_url, params=image)
        # genre_result = response.json()


with col2:
    if uploaded_file is not None:
        st.header("The Movie Genre is...")
        st.code(model_image.predict(processed_uploaded_file))
        st.write(f"genre_result from image")
        st.balloons()
    # else:
    #     st.write("Please upload an image file")
    if sypnosis_button:
        st.header("The Movie Genre is...")
        # st.write(f"{result}")
        # st.code(model_nlp.predict(clean_txt))
        st.write(f"genre_result from plot")
        st.balloons()

    if uploaded_file is not None and sypnosis_button:
        st.header("The Movie Genre is...")
        st.write(f"both")
        st.snow()
