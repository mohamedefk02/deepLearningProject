import os
import pickle
import re
from pathlib import Path

import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from tensorflow import keras
import tensorflow as tf

tf.get_logger().setLevel("ERROR")


def _clean_news_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z0-9\s.,!?'-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_classification_model(model_path: Path, preprocessing_path: Path):
    with preprocessing_path.open("rb") as f:
        preprocessing = pickle.load(f)

    model = keras.models.load_model(model_path)
    return {"model": model, "preprocessing": preprocessing}


def run_classification(store, text: str):
    preprocessing = store["preprocessing"]
    tokenizer = preprocessing["news_tokenizer"]
    max_len = preprocessing["max_news_len"]
    class_names = preprocessing["class_names"]

    sequence = tokenizer.texts_to_sequences([_clean_news_text(text)])
    padded = keras.preprocessing.sequence.pad_sequences(
        sequence,
        maxlen=max_len,
        padding="post",
        truncating="post",
    )

    probs = store["model"].predict(padded, verbose=0)[0]
    label_id = int(np.argmax(probs))
    scores = {class_names[i]: float(probs[i]) for i in range(len(class_names))}

    return {
        "label": class_names[label_id],
        "confidence": float(probs[label_id]),
        "scores": scores,
    }

