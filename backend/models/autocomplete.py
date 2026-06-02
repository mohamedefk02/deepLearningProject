import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import json
import random
import re
from pathlib import Path

import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import keras
import tensorflow as tf

tf.get_logger().setLevel("ERROR")

SEED = 42
SEQ_LENGTH = 60


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def _load_vocab(path: Path):
    artifact = json.loads(path.read_text(encoding="utf-8"))
    chars = artifact["chars"]
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for ch, i in char_to_idx.items()}
    seq_length = int(artifact.get("seq_length", SEQ_LENGTH))
    return chars, char_to_idx, idx_to_char, seq_length


def load_autocomplete_model(model_path: Path, preprocessing_path: Path):
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    chars, char_to_idx, idx_to_char, seq_length = _load_vocab(preprocessing_path)
    model = keras.models.load_model(model_path)
    return {
        "model": model,
        "chars": chars,
        "char_to_idx": char_to_idx,
        "idx_to_char": idx_to_char,
        "seq_length": seq_length,
        "vocab_size": len(chars),
    }


def _prepare_prompt(prompt: str, char_to_idx: dict[str, int]) -> str:
    cleaned = _normalize(prompt)
    return "".join(ch for ch in cleaned if ch in char_to_idx)


def _context(text: str, seq_length: int) -> str:
    if len(text) < seq_length:
        return (" " * (seq_length - len(text))) + text
    return text[-seq_length:]


def _next_distribution(store, text: str):
    char_to_idx = store["char_to_idx"]
    context = _context(text, store["seq_length"])
    encoded = np.array([[char_to_idx.get(ch, char_to_idx[" "]) for ch in context]], dtype=np.int32)
    one_hot = tf.one_hot(encoded, depth=store["vocab_size"])
    return store["model"].predict(one_hot, verbose=0)[0]


def _sample_with_temperature(probs, temperature: float) -> int:
    if temperature <= 0:
        return int(np.argmax(probs))

    probs = np.asarray(probs).astype("float64")
    logits = np.log(probs + 1e-12) / temperature
    exp_logits = np.exp(logits)
    sampled_probs = exp_logits / np.sum(exp_logits)
    sampled_probs = sampled_probs / np.sum(sampled_probs)
    return int(np.random.choice(len(sampled_probs), p=sampled_probs))


def _generate_single(store, prompt: str, length: int, method: str, temperature: float):
    current_text = prompt
    for _ in range(length):
        probs = _next_distribution(store, current_text)
        if method == "greedy":
            next_idx = int(np.argmax(probs))
        else:
            next_idx = _sample_with_temperature(probs, temperature)
        current_text += store["idx_to_char"][next_idx]
    return current_text


def _generate_beams(store, prompt: str, length: int, beam_width: int):
    beams = [(prompt, 0.0)]
    for _ in range(length):
        candidates = []
        for current_text, score in beams:
            probs = _next_distribution(store, current_text)
            log_probs = np.log(probs + 1e-12)
            top_indices = np.argsort(log_probs)[-beam_width:]
            for idx in top_indices:
                idx = int(idx)
                candidates.append((current_text + store["idx_to_char"][idx], score + float(log_probs[idx])))
        beams = sorted(candidates, key=lambda item: item[1], reverse=True)[:beam_width]
    return beams


def run_autocomplete(store, text: str, method: str, length: int, temperature: float, beam_width: int):
    prompt = _prepare_prompt(text, store["char_to_idx"])
    if not prompt:
        prompt = " "

    if method == "beam":
        beams = _generate_beams(store, prompt, length, beam_width)
        completions = [
            {
                "text": generated,
                "generated": generated[len(prompt):],
                "score": score,
            }
            for generated, score in beams
        ]
    else:
        generated = _generate_single(store, prompt, length, method, temperature)
        completions = [
            {
                "text": generated,
                "generated": generated[len(prompt):],
                "score": None,
            }
        ]

    return {"completions": completions, "method_used": method}

