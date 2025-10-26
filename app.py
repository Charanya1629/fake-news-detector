import streamlit as st
import joblib
import pandas as pd

st.title("📰 Fake News Detector")

# Load
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.write("Type a headline and press Predict or upload CSV with column 'text' for batch predictions.")

# Single prediction
text = st.text_area("Enter news headline:")
if st.button("Predict"):
    if text.strip():
        X = vectorizer.transform([text])
        pred = model.predict(X)[0]
        prob = model.predict_proba(X).max()
        st.success(f"Prediction: **{pred}** (confidence: {prob:.2f})")
    else:
        st.warning("Please enter a headline.")

# Batch upload
uploaded = st.file_uploader("Upload CSV (must have 'text' column)", type="csv")
if uploaded:
    df = pd.read_csv(uploaded)
    if 'text' not in df.columns:
        st.error("CSV must contain a 'text' column.")
    else:
        X = vectorizer.transform(df['text'].fillna(''))
        df['prediction'] = model.predict(X)
        df['confidence'] = model.predict_proba(X).max(axis=1)
        st.write(df)
        st.download_button("Download results", df.to_csv(index=False), "results.csv", "text/csv")
