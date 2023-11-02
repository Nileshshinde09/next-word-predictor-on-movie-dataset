import streamlit as st
import pandas as pd
from main import run
num=False
text=False
sentence_length=False
st.header("This app shows the demo of the next word predictor model, which I built on the Three Idiots movie dataset using LSTM (stacked LSTM).")
mode=['Prediction On Raw Input','Prediction On Preprocessed Input']
selectedMode=st.selectbox("Select Mode",mode)
text=st.text_area(label="Enter Text For Prediction")
sentence_length=st.number_input(label='Enter Sentence Word Length', min_value=3, max_value=20,step=1)
if mode=='Prediction On Preprocessed Input':
    num=1
if mode=='Prediction On Raw Input':
    num=0

if st.button(label='Enter'):
    if num==False and text==False and sentence_length==False:
        st.warning("All Fields Are Mandatory")
    else:
        run(num,text,int(sentence_length))