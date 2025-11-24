import streamlit as st
import joblib
import json
import numpy as np

# Load model + vectorizer + labels
model = joblib.load("tfidf_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

with open("label_map.json", "r") as f:
    label_map = json.load(f)

# Reverse label map for output
inv_label_map = {v: k for k, v in label_map.items()}

# Streamlit UI
st.title("📈 News Headline Sentiment Analysis 📰")
st.write("Analyze financial news sentiment instantly!")

headline = st.text_input("Enter a news headline:")

if st.button("Analyze"):
    if headline.strip() == "":
        st.warning("⚠ Please enter a headline.")
    else:
        vec = vectorizer.transform([headline])
        prediction = model.predict(vec)[0]
        sentiment = inv_label_map[int(prediction)]

        # Display result
        colors = {"negative": "🔴", "neutral": "🟡", "positive": "🟢"}
        st.subheader(f"Sentiment: {colors[sentiment]} **{sentiment.upper()}**")
