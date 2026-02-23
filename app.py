# ===============================
# 1️⃣ IMPORT LIBRARIES
# ===============================

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import requests
from bs4 import BeautifulSoup
from newsapi import NewsApiClient

# ===============================
# 2️⃣ LOAD MODEL + TOKENIZER
# ===============================

print("Loading model...")

model = load_model("test4.keras")
tokenizer = pickle.load(open("tokenizer.pkl", "rb"))

maxlen = 300

# ===============================
# 3️⃣ FETCH NEWS
# ===============================

def get_news_url(keyword):
    newsapi = NewsApiClient(api_key="266cf82f948a4071982dc2811f170936")
    result = newsapi.get_everything(q=keyword, language="en")

    if result["totalResults"] == 0:
        return None, None

    return result["articles"][0]["url"], result["articles"][0]["title"]

# ===============================
# 4️⃣ SCRAPE ARTICLE TEXT
# ===============================

def get_news_text(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    article = ""
    for p in soup.find_all("p"):
        article += p.get_text()

    return article

# ===============================
# 5️⃣ PREDICT FAKE OR REAL
# ===============================

def predict_news(text_input):
    seq = tokenizer.texts_to_sequences([text_input])
    padded = pad_sequences(seq, maxlen=maxlen)

    prob = model.predict(padded)[0][0]

    label = "Real" if prob >= 0.5 else "Fake"
    score = prob * 100

    return label, score

# ===============================
# 6️⃣ MAIN PROGRAM
# ===============================

if __name__ == "__main__":

    keyword = input("Enter topic: ")

    url, title = get_news_url(keyword)

    if url:
        print("\nFetching article...")

        article_text = get_news_text(url)

        label, score = predict_news(article_text)

        print("\nTitle:", title)
        print("Prediction:", label)
        print("Credibility Score:", round(score, 2), "%")
    else:
        print("No news found.")