#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import MultiLabelBinarizer


# In[2]:


def get_df(your_folder_path):
    """get df and drop duplicates"""

    folder_path = your_folder_path
    dataframes = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(file_path)
            dataframes.append(df)
    merged_df = pd.concat(dataframes, ignore_index=True)
    merged_df = merged_df.drop(columns=['Unnamed: 0'])
    merged_df = merged_df.drop_duplicates()

    return merged_df


# In[4]:

def cleaning(sentence):

    # Basic cleaning
    sentence = sentence.strip() ## remove whitespaces
    sentence = sentence.lower() ## lowercase
    sentence = ''.join(char for char in sentence if not char.isdigit()) ## remove numbers

    # Advanced cleaning
    for punctuation in string.punctuation:
        sentence = sentence.replace(punctuation, '') ## remove punctuation

    tokenized_sentence = word_tokenize(sentence) ## tokenize
    stop_words = set(stopwords.words('english')) ## define stopwords

    tokenized_sentence_cleaned = [ ## remove stopwords
        w for w in tokenized_sentence if not w in stop_words
    ]

    lemmatized = [
        WordNetLemmatizer().lemmatize(word, pos = "v")
        for word in tokenized_sentence_cleaned
    ]

    cleaned_sentence = ' '.join(word for word in lemmatized)

    return cleaned_sentence


def process_df(merged_df):
    """Classifies the genre labels"""

    #Separating genre label
    merged_df["genre"] = (
        merged_df["genre"]
        .apply(eval)
        .apply(lambda x: [genre.strip() for genre in x])
        )

    #Label genre

    multilabel_binarizer = MultiLabelBinarizer()
    multilabel_binarizer.fit(merged_df["genre"])

    # transform target variable
    y = multilabel_binarizer.transform(merged_df['genre'])
    genre_names = multilabel_binarizer.classes_

    # Adding into df
    for i in range(len(genre_names)):
        merged_df[f"{genre_names[i]}"] = y[:,i]

    return merged_df
