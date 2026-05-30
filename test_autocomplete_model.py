import argparse
import re
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras


FRANKENSTEIN_URL = "https://www.gutenberg.org/files/84/84-0.txt"
SEQ_LENGTH = 60


def normalize_spaces(text):
    return re.sub(r"\s+", " ", text.lower()).strip()


def load_frankenstein_text():
    path = keras.utils.get_file("frankenstein_gutenberg_84.txt", FRANKENSTEIN_URL)
    raw_text = Path(path).read_text(encoding="utf-8")

    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK FRANKENSTEIN"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK FRANKENSTEIN"

    start_idx = raw_text.find(start_marker)
    end_idx = raw_text.find(end_marker)
    if start_idx != -1 and end_idx != -1:
        raw_text = raw_text[start_idx:end_idx]

    return normalize_spaces(raw_text)


def build_character_maps(text):
    chars = sorted(set(text))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for ch, i in char_to_idx.items()}
    return chars, char_to_idx, idx_to_char


def prepare_prompt(prompt, char_to_idx):
    prompt = normalize_spaces(prompt)
    prompt = "".join(ch for ch in prompt if ch in char_to_idx)
    if len(prompt) < SEQ_LENGTH:
        prompt = (" " * (SEQ_LENGTH - len(prompt))) + prompt
    return prompt[-SEQ_LENGTH:]


def predict_next_char_distribution(model, context, char_to_idx, vocab_size):
    encoded = np.array(
        [[char_to_idx.get(ch, char_to_idx[" "]) for ch in context]],
        dtype=np.int32,
    )
    one_hot = tf.one_hot(encoded, depth=vocab_size)
    return model.predict(one_hot, verbose=0)[0]


def generate_greedy(model, prompt, char_to_idx, idx_to_char, vocab_size, length):
    context = prepare_prompt(prompt, char_to_idx)
    generated = prompt

    for _ in range(length):
        probs = predict_next_char_distribution(model, context, char_to_idx, vocab_size)
        next_idx = int(np.argmax(probs))
        next_char = idx_to_char[next_idx]
        generated += next_char
        context = (context + next_char)[-SEQ_LENGTH:]

    return generated


def generate_with_temperature(model, prompt, char_to_idx, idx_to_char, vocab_size, length, temperature):
    context = prepare_prompt(prompt, char_to_idx)
    generated = prompt

    for _ in range(length):
        probs = predict_next_char_distribution(model, context, char_to_idx, vocab_size)
        logits = np.log(probs + 1e-8) / temperature
        probs = np.exp(logits) / np.sum(np.exp(logits))
        next_idx = int(np.random.choice(len(probs), p=probs))
        next_char = idx_to_char[next_idx]
        generated += next_char
        context = (context + next_char)[-SEQ_LENGTH:]

    return generated


def main():
    parser = argparse.ArgumentParser(description="Test the trained Frankenstein autocomplete LSTM model.")
    parser.add_argument(
        "--model",
        default="best_frankenstein_char_lstm.keras",
        help="Path to the trained .keras model file.",
    )
    parser.add_argument(
        "--prompt",
        default="it was on a dreary night of november",
        help="Prompt used to start text generation.",
    )
    parser.add_argument(
        "--length",
        type=int,
        default=250,
        help="Number of characters to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Optional sampling temperature. If omitted, greedy decoding is used.",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    text = load_frankenstein_text()
    chars, char_to_idx, idx_to_char = build_character_maps(text)
    vocab_size = len(chars)

    model = keras.models.load_model(model_path)

    print("TensorFlow version:", tf.__version__)
    print("Loaded model:", model_path)
    print("Model input shape:", model.input_shape)
    print("Model output shape:", model.output_shape)
    print("Vocabulary size:", vocab_size)
    print("Prompt:", args.prompt)
    print()

    if args.temperature is None:
        generated = generate_greedy(
            model,
            args.prompt,
            char_to_idx,
            idx_to_char,
            vocab_size,
            args.length,
        )
        print("Generated text with greedy decoding:")
    else:
        generated = generate_with_temperature(
            model,
            args.prompt,
            char_to_idx,
            idx_to_char,
            vocab_size,
            args.length,
            args.temperature,
        )
        print(f"Generated text with temperature={args.temperature}:")

    print(generated)


if __name__ == "__main__":
    main()
