#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st

import numpy as np
import pandas as pd

st.markdown("""# Movie Genre Predictor
## Somthing""")

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Upload image", type=['png', 'jpg'])
    if uploaded_file is not None:
        # To read file as bytes:
        bytes_data = uploaded_file.getvalue()
        st.write(bytes_data)

        # To convert to a string based IO:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        st.write(stringio)

        # To read file as string:
        string_data = stringio.read()
        st.write(string_data)

    st.header("Your Poster")
    st.image("https://static.streamlit.io/examples/cat.jpg")

with col2:
   st.header("The Movie Genre is...")
   st.markdown('XXX and YYY')


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
