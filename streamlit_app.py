import streamlit as st
import requests
from bs4 import BeautifulSoup
import pickle
import os
from textblob import TextBlob
from newsapi import NewsApiClient
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ===============================
# CONFIGURATION
# ===============================

NEWS_API_KEY = "266cf82f948a4071982dc2811f170936"   # Replace if needed
MAX_LEN = 300

# ===============================
# LOAD MODEL & TOKENIZER
# ===============================

@st.cache_resource
def load_resources():
    model = load_model("test4.keras")
    tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
    return model, tokenizer

model, tokenizer = load_resources()

# ===============================
# PREDICTION FUNCTION
# ===============================

def predict_news(text_input):
    seq = tokenizer.texts_to_sequences([text_input])
    padded = pad_sequences(seq, maxlen=MAX_LEN)
    prob = model.predict(padded)[0][0]
    label = "Real" if prob >= 0.5 else "Fake"
    return label, prob

# ===============================
# SENTIMENT ANALYSIS
# ===============================

def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

# ===============================
# FETCH NEWS FROM NEWSAPI
# ===============================

def get_news_url(keyword):
    try:
        newsapi = NewsApiClient(api_key=NEWS_API_KEY)
        result = newsapi.get_everything(q=keyword, language="en")

        if result["totalResults"] == 0:
            return None, None

        return result["articles"][0]["url"], result["articles"][0]["title"]

    except Exception as e:
        st.error("NewsAPI Error: Check your API key.")
        return None, None

# ===============================
# SCRAPE ARTICLE
# ===============================

def get_news_text(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        article = ""
        for p in soup.find_all("p"):
            article += p.get_text()

        return article

    except:
        st.error("Failed to scrape article.")
        return ""

# ===============================
# STREAMLIT UI
# ===============================

st.set_page_config(page_title="AI News Credibility Analyzer", layout="centered")

st.title("📰 AI News Credibility Analyzer")
st.write("Analyze latest news or paste article manually.")

option = st.radio("Choose Input Mode:", ["Fetch from NewsAPI", "Paste Article Manually"])

text_input = ""
url = ""

# ===============================
# FETCH MODE
# ===============================

if option == "Fetch from NewsAPI":

    keyword = st.text_input("Enter News Topic")

    if st.button("Analyze News"):

        if keyword:

            url, title = get_news_url(keyword)

            if url:
                st.subheader("📰 News Title")
                st.write(title)

                text_input = get_news_text(url)

            else:
                st.warning("No news found or API issue.")

        else:
            st.warning("Please enter a topic.")

# ===============================
# MANUAL MODE
# ===============================

else:
    text_input = st.text_area("Paste Full News Article Here")

# ===============================
# ANALYSIS SECTION
# ===============================

if text_input:

    label, prob = predict_news(text_input)
    sentiment_score = analyze_sentiment(text_input)

    st.subheader("🔍 Prediction Result")

    if label == "Real":
        st.success("Real News ✅")
    else:
        st.error("Fake News ❌")

    st.subheader("📊 Confidence Score")
    st.progress(float(prob))
    st.write(f"{prob * 100:.2f}% confidence")

    st.subheader("📈 Sentiment Analysis")

    if sentiment_score > 0:
        st.write("Positive Sentiment 🙂")
    elif sentiment_score < 0:
        st.write("Negative Sentiment 😠")
    else:
        st.write("Neutral Sentiment 😐")