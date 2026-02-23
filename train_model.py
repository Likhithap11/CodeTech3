# ===============================
# 1️⃣ IMPORT LIBRARIES
# ===============================

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing import text
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle

# ===============================
# 2️⃣ LOAD DATASET
# ===============================

print("Loading dataset...")

df = pd.read_csv("WELFake_Dataset.csv")
df.fillna("", inplace=True)

df["tot_news"] = df["title"] + " " + df["text"]

X = df["tot_news"]
Y = df["label"]

# ===============================
# 3️⃣ TRAIN TEST SPLIT
# ===============================

x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=42)

# ===============================
# 4️⃣ TOKENIZATION
# ===============================

max_features = 10000
maxlen = 300

tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(x_train)

x_train_seq = tokenizer.texts_to_sequences(x_train)
x_test_seq = tokenizer.texts_to_sequences(x_test)

x_train_pad = pad_sequences(x_train_seq, maxlen=maxlen)
x_test_pad = pad_sequences(x_test_seq, maxlen=maxlen)

# ===============================
# 5️⃣ BUILD LSTM MODEL
# ===============================

print("Building model...")

model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=128, input_length=300))
model.add(LSTM(64))
model.add(Dense(1, activation='sigmoid'))

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# ===============================
# 6️⃣ TRAIN MODEL
# ===============================

print("Training model...")

model.fit(
    x_train_pad,
    y_train,
    epochs=5,
    batch_size=64,
    validation_data=(x_test_pad, y_test)
)

# ===============================
# 7️⃣ SAVE MODEL + TOKENIZER
# ===============================

model.save("test4.keras")
pickle.dump(tokenizer, open("tokenizer.pkl", "wb"))

print("\n✅ Model and tokenizer saved successfully!")
# Evaluate model
loss, accuracy = model.evaluate(x_test_pad, y_test)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")