import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
import os

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)


# Load the vectorizer and model with error handling
tfidf = None
model = None
try:
    if os.path.getsize('vectorizer.pkl') > 0:
        with open('vectorizer.pkl', 'rb') as f:
            tfidf = pickle.load(f)
    if os.path.getsize('model.pkl') > 0:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
except EOFError as e:
    st.error(f"Error loading pickled files: {e}")

st.title('SMS spam classifier')

input_sms = st.text_input('Enter the message')

if st.button('Predict'):
    # 1. preprocess
    transform_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transform_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. display
    if result == 1:
        st.header('Spam')
    else:
        st.header('Not spam')
