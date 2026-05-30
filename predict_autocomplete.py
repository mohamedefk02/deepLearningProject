import argparse
import contextlib
import json
import os
import re
from pathlib import Path

import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull):
    import tensorflow as tf
    from tensorflow import keras

tf.get_logger().setLevel("ERROR")


FRANKENSTEIN_URL = "https://www.gutenberg.org/files/84/84-0.txt"
SEQ_LENGTH = 60


def normalize_spaces(text):
    return re.sub(r"\s+", " ", text.lower()).strip()


def load_text():
    path = keras.utils.get_file("frankenstein_gutenberg_84.txt", FRANKENSTEIN_URL)
    raw_text = Path(path).read_text(encoding="utf-8")

    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK FRANKENSTEIN"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK FRANKENSTEIN"
    start_idx = raw_text.find(start_marker)
    end_idx = raw_text.find(end_marker)
    if start_idx != -1 and end_idx != -1:
        raw_text = raw_text[start_idx:end_idx]

    return normalize_spaces(raw_text)


def build_vocab(text):
    chars = sorted(set(text))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for ch, i in char_to_idx.items()}
    return char_to_idx, idx_to_char


def load_saved_vocab(path):
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    chars = artifact["chars"]
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for ch, i in char_to_idx.items()}
    return char_to_idx, idx_to_char


def prepare_prompt(prompt, char_to_idx):
    prompt = normalize_spaces(prompt)
    prompt = "".join(ch for ch in prompt if ch in char_to_idx)
    if len(prompt) < SEQ_LENGTH:
        prompt = (" " * (SEQ_LENGTH - len(prompt))) + prompt
    return prompt[-SEQ_LENGTH:]


def predict_text(model, prompt, char_to_idx, idx_to_char, length, temperature=None):
    vocab_size = len(char_to_idx)
    context = prepare_prompt(prompt, char_to_idx)
    generated = prompt

    for _ in range(length):
        encoded = np.array([[char_to_idx.get(ch, char_to_idx[" "]) for ch in context]])
        one_hot = tf.one_hot(encoded, depth=vocab_size)
        probs = model.predict(one_hot, verbose=0)[0]

        if temperature is None:
            next_idx = int(np.argmax(probs))
        else:
            probs = np.asarray(probs).astype("float64")
            logits = np.log(probs + 1e-8) / temperature
            exp_logits = np.exp(logits)
            sampled_probs = exp_logits / np.sum(exp_logits)
            next_idx = int(np.random.choice(len(sampled_probs), p=sampled_probs))

        next_char = idx_to_char[next_idx]
        generated += next_char
        context = (context + next_char)[-SEQ_LENGTH:]

    return generated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="best_frankenstein_char_lstm.keras")
    parser.add_argument("--preprocessing", default="frankenstein_char_preprocessing.json")
    parser.add_argument("--prompt", default="it was on a dreary night of november")
    parser.add_argument("--length", type=int, default=200)
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Use sampling instead of greedy decoding. Try 0.7, 0.8, or 1.0.",
    )
    args = parser.parse_args()

    if Path(args.preprocessing).exists():
        char_to_idx, idx_to_char = load_saved_vocab(args.preprocessing)
    else:
        text = load_text()
        char_to_idx, idx_to_char = build_vocab(text)

    model = keras.models.load_model(args.model)

    prediction = predict_text(
        model,
        args.prompt,
        char_to_idx,
        idx_to_char,
        args.length,
        temperature=args.temperature,
    )
    print(prediction)


if __name__ == "__main__":
    main()
