import streamlit as st
import pickle
import re
import nltk
import numpy as np  # ← add this
from nltk.corpus import stopwords


# Ensure stopwords are downloaded
nltk.download('stopwords')

# Load saved LinearSVC model and vectorizer
model = pickle.load(open("fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

stop_words = set(stopwords.words('english'))

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# Streamlit UI
st.title("📰 Fake News Detection App")
st.write("Enter a news article below to check whether it is REAL or FAKE.")

user_input = st.text_area("Enter News Article Here:")

if st.button("Predict"):
    if user_input:
        cleaned_text = clean_text(user_input)
        vector = vectorizer.transform([cleaned_text])
        prediction = model.predict(vector)[0]

        # Get decision function for confidence-like score
        decision_score = model.decision_function(vector)[0]
        # Convert decision score to a 0-100 scale for display
        confidence = 100 * (1 / (1 + np.exp(-decision_score)))  # sigmoid approximation

        result = "REAL" if prediction == 1 else "FAKE"

        # Show results
        st.subheader(f"Prediction: {result}")
        st.write(f"Confidence: {confidence:.2f}%")
    else:
        st.warning("Please enter a news article to predict.")
