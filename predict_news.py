import argparse
import os
import pickle
import re
from pathlib import Path

import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from tensorflow import keras

tf.get_logger().setLevel("ERROR")


def clean_news_text(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z0-9\s.,!?'-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def predict_category(text, model, preprocessing):
    tokenizer = preprocessing["news_tokenizer"]
    max_len = preprocessing["max_news_len"]
    class_names = preprocessing["class_names"]

    sequence = tokenizer.texts_to_sequences([clean_news_text(text)])
    padded = keras.preprocessing.sequence.pad_sequences(
        sequence,
        maxlen=max_len,
        padding="post",
        truncating="post",
    )

    probs = model.predict(padded, verbose=0)[0]
    label_id = int(np.argmax(probs))
    return class_names[label_id], float(np.max(probs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="best_ag_news_lstm.keras")
    parser.add_argument("--preprocessing", default="ag_news_preprocessing.pkl")
    parser.add_argument(
        "--text",
        default="Technology stocks rise as chip demand increases",
    )
    args = parser.parse_args()

    if not Path(args.model).exists():
        raise FileNotFoundError(args.model)
    if not Path(args.preprocessing).exists():
        raise FileNotFoundError(args.preprocessing)

    with Path(args.preprocessing).open("rb") as f:
        preprocessing = pickle.load(f)

    model = keras.models.load_model(args.model)
    label, confidence = predict_category(args.text, model, preprocessing)
    print(f"{label} ({confidence:.3f})")


if __name__ == "__main__":
    main()
