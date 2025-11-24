import streamlit as st
import joblib
import json

# Load saved model + vectorizer
model = joblib.load("tfidf_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Correct label mapping
inv_label_map = {
    -1: "Negative",
     0: "Neutral",
     1: "Positive"
}

st.set_page_config(page_title="News Sentiment App")
st.title("📉📈 News Headline Sentiment Analysis 📰")

st.write("Analyze financial news sentiment instantly!")

headline = st.text_input("Enter a news headline:")

if st.button("Analyze"):
    if headline.strip():
        transformed = vectorizer.transform([headline])
        prediction = model.predict(transformed)[0]

        sentiment = inv_label_map.get(prediction, "Unknown sentiment label")

        st.success(f"Result: **{sentiment}**")
        st.caption(f"Model raw output: {prediction}")
    else:
        st.warning("Please enter a headline!")
