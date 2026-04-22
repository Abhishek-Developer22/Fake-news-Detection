import streamlit as st
import pickle
import pandas as pd
import re

# Load model & vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text

# Prediction function
def predict_news(news):
    news = clean_text(news)
    vector = vectorizer.transform([news])
    prediction = model.predict(vector)[0]
    proba = model.predict_proba(vector)[0]

    if prediction == 0:
        return "Fake News", proba[0]
    else:
        return "Real News", proba[1]

# UI
st.set_page_config(page_title="Fake News Detector")

st.title("📰 Fake News Detection App")
st.write("Enter news text below to check if it is Real or Fake.")

user_input = st.text_area("Enter News Here")

if st.button("Check News"):
    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        result, confidence = predict_news(user_input)

        if result == "Real News":
            st.success(f"✅ {result}")
        else:
            st.error(f"❌ {result}")

        st.info(f"Confidence: {round(confidence*100,2)}%")