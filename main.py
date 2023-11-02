import pickle
import streamlit as st
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model 
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import numpy as np

def run(num=0,text = "school",sentence_length=10):
    if num==0:
        max_len=322
    else:
        max_len=176
    with open('dataset.txt','r') as file:
        dataset=file.read()
        file.close()
    with open('tokenizer.pkl','rb') as tokenfil:
        tokenizer=pickle.load(tokenfil)
    model_without_preprocessing=load_model("next_word_pred_model.h5")
    model_with_preprocessing=load_model("next_word_pred_model_with_preprocessing.h5")
    if num==1:
        model=model_with_preprocessing
    else:
        model=model_without_preprocessing
    text=text.lower()
    for _ in range(sentence_length):
        # tokenize
        token_text = tokenizer.texts_to_sequences([text])[0]
        # padding
        padded_token_text = pad_sequences([token_text], maxlen=max_len-1, padding='pre')
        # predict
        pos = np.argmax(model.predict(padded_token_text))
        for word,index in tokenizer.word_index.items():
            if index == pos:
                text = text + " " + word
    else:

        if num==1:
            #Lemmatize each word in the sentence
            lemmatized_sentence = [lemmatizer.lemmatize(word, pos="a") for word in text]

            # Join the lemmatized words back into a sentence
            lemmatized_sentence = " ".join(lemmatized_sentence)
            lemmatized_sentence=lemmatizer.lemmatize(text, pos="v")
            st.header("Prediction On Preprocessed Input")
            st.success(lemmatized_sentence)
        else:
            st.header("Prediction On Raw Input")
            st.success(text)