import streamlit as st
import joblib
import json

# Load Vectorizer
@st.cache_resource
def load_vectorizer():
    return joblib.load("tfidf_vectorizer.pkl")

# Load Model
@st.cache_resource
def load_model():
    return joblib.load("tfidf_model.pkl")

# Load Label Map
@st.cache_resource
def load_label_map():
    with open("label_map.json", "r") as f:
        label_map = json.load(f)
    inv_label_map = {v: k for k, v in label_map.items()}  # reverse mapping
    return inv_label_map

# Load components
vectorizer = load_vectorizer()
model = load_model()
inv_label_map = load_label_map()

# Streamlit UI
st.markdown("## 📈 News Headline Sentiment Analysis 📰")
st.write("Analyze financial news sentiment instantly!")

headline = st.text_input("Enter a news headline:")

if st.button("Analyze"):
    if headline.strip() == "":
        st.warning("⚠ Please enter a headline...")
    else:
        try:
            input_vec = vectorizer.transform([headline])
            prediction = model.predict(input_vec)[0]
            pred_label = int(prediction)

            # Map prediction to label
            sentiment = inv_label_map.get(pred_label, "Unknown")

            # Display result with styling
            if sentiment == "positive":
                st.success(f"🎉 Sentiment: **Positive**")
            elif sentiment == "neutral":
                st.info(f"😐 Sentiment: **Neutral**")
            elif sentiment == "negative":
                st.error(f"📉 Sentiment: **Negative**")
            else:
                st.warning("Result: Unknown sentiment label")

            # Debug info (optional)
            st.write("Model raw output:", prediction)

        except Exception as e:
            st.error("❌ Error analyzing sentiment")
            st.write(e)
