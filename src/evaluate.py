import numpy as np
from sklearn.metrics import classification_report

def evaluate_model(model, X_text, X_emoji, y):
    preds = model.predict({"input_1": X_text, "input_2": X_emoji})
    print(classification_report(
        np.argmax(y, 1),
        np.argmax(preds, 1)
    ))
