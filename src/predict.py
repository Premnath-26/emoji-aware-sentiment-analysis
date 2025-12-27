# src/predict.py
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.preprocessing import clean_text, compute_emoji_score

def predict_sentiment(text, model, tokenizer):
    seq = tokenizer.texts_to_sequences([clean_text(text)])
    padded = pad_sequences(seq, maxlen=100)
    emoji_score = np.array([[compute_emoji_score(text)]])
    pred = model.predict({"input_1": padded, "input_2": emoji_score})
    return "Positive" if pred.argmax() == 1 else "Negative"
