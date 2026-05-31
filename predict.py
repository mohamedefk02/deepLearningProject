import argparse
import json
import os
import pickle
import re
import random
from pathlib import Path

import numpy as np

# Suppress TensorFlow logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

try:
    import tensorflow as tf
    from tensorflow import keras
    tf.get_logger().setLevel("ERROR")
except ImportError:
    print("Error: TensorFlow/Keras not found. Please install them using 'pip install tensorflow'")
    exit(1)

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# --- Shared Utilities ---

def clean_text(text, task):
    text = str(text).lower().strip()
    if task == "news":
        text = re.sub(r"<.*?>", " ", text)
        text = re.sub(r"[^a-z0-9\s.,!?'-]", " ", text)
    elif task == "translation":
        text = re.sub(r"[^a-zA-Z\u00C0-\u00FF?.!,;:'\- ]+", " ", text)
    
    text = re.sub(r"\s+", " ", text).strip()
    return text

# --- Autocomplete Task ---

def sample_with_temperature(probs, temperature):
    """
    Perform temperature sampling on the probability distribution.
    Higher temperature -> more diversity/randomness.
    Lower temperature -> more conservative/greedy.
    """
    if temperature is None or temperature <= 0:
        return int(np.argmax(probs))
    
    probs = np.asarray(probs).astype("float64")
    # Add a small epsilon to avoid log(0)
    logits = np.log(probs + 1e-12) / temperature
    exp_logits = np.exp(logits)
    probs = exp_logits / np.sum(exp_logits)
    
    # Ensure probabilities sum to 1.0 for np.random.choice
    probs = probs / np.sum(probs)
    return int(np.random.choice(len(probs), p=probs))

def predict_autocomplete(model, prompt, preprocessing_path, length=100, method="greedy", temperature=0.7, beam_width=5):
    with open(preprocessing_path, "r", encoding="utf-8") as f:
        artifact = json.load(f)
    
    chars = artifact["chars"]
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for ch, i in char_to_idx.items()}
    vocab_size = len(chars)
    seq_length = 60

    # Pre-clean prompt to match training data
    clean_prompt = clean_text(prompt, "autocomplete")
    clean_prompt = "".join(ch for ch in clean_prompt if ch in char_to_idx)
    
    # Pad or truncate to seq_length
    def get_context(text):
        if len(text) < seq_length:
            return (" " * (seq_length - len(text))) + text
        return text[-seq_length:]

    if method == "beam":
        return generate_beam_search(model, clean_prompt, char_to_idx, idx_to_char, vocab_size, seq_length, length, beam_width)
    
    # Greedy or Temperature sampling
    current_text = clean_prompt
    for _ in range(length):
        context = get_context(current_text)
        encoded = np.array([[char_to_idx.get(ch, char_to_idx[" "]) for ch in context]])
        one_hot = tf.one_hot(encoded, depth=vocab_size)
        probs = model.predict(one_hot, verbose=0)[0]

        if method == "greedy":
            next_idx = int(np.argmax(probs))
        else: # temperature
            next_idx = sample_with_temperature(probs, temperature)

        next_char = idx_to_char[next_idx]
        current_text += next_char

    return current_text

def generate_beam_search(model, prompt, char_to_idx, idx_to_char, vocab_size, seq_length, length, beam_width):
    """
    Implement Beam Search to find sequences with higher cumulative log probability.
    """
    def get_context(text):
        if len(text) < seq_length:
            return (" " * (seq_length - len(text))) + text
        return text[-seq_length:]

    # beams = list of (current_text, cumulative_log_prob)
    beams = [(prompt, 0.0)]

    for _ in range(length):
        candidates = []
        for current_text, score in beams:
            context = get_context(current_text)
            encoded = np.array([[char_to_idx.get(ch, char_to_idx[" "]) for ch in context]])
            one_hot = tf.one_hot(encoded, depth=vocab_size)
            probs = model.predict(one_hot, verbose=0)[0]
            
            # Use log probabilities to avoid underflow
            log_probs = np.log(probs + 1e-12)
            
            # Get top-k candidates for this beam
            top_indices = np.argsort(log_probs)[-beam_width:]
            for idx in top_indices:
                candidates.append((current_text + idx_to_char[idx], score + log_probs[idx]))
        
        # Sort all candidates and keep the top beam_width
        beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]

    return beams[0][0]

# --- News Classification Task ---

def predict_news(model, text, preprocessing_path):
    with open(preprocessing_path, "rb") as f:
        preprocessing = pickle.load(f)
    
    tokenizer = preprocessing["news_tokenizer"]
    max_len = preprocessing["max_news_len"]
    class_names = preprocessing["class_names"]

    sequence = tokenizer.texts_to_sequences([clean_text(text, "news")])
    padded = keras.preprocessing.sequence.pad_sequences(
        sequence, maxlen=max_len, padding="post", truncating="post"
    )

    probs = model.predict(padded, verbose=0)[0]
    label_id = int(np.argmax(probs))
    return class_names[label_id], float(np.max(probs))

# --- Translation Task ---

def build_inference_models(training_model):
    encoder_inputs = training_model.inputs[0]
    _, h_fwd, c_fwd, h_bwd, c_bwd = training_model.get_layer("bidirectional_encoder").output
    
    state_h = keras.layers.Concatenate()([h_fwd, h_bwd])
    state_c = keras.layers.Concatenate()([c_fwd, c_bwd])
    encoder_model = keras.Model(inputs=encoder_inputs, outputs=[state_h, state_c])

    lstm_units = 256
    decoder_inputs = keras.Input(shape=(None,), name="decoder_token_input")
    decoder_state_input_h = keras.Input(shape=(lstm_units * 2,), name="decoder_state_input_h")
    decoder_state_input_c = keras.Input(shape=(lstm_units * 2,), name="decoder_state_input_c")

    decoder_embedding = training_model.get_layer("decoder_embedding")(decoder_inputs)
    decoder_lstm_layer = training_model.get_layer("decoder_lstm")
    decoder_outputs, state_h_out, state_c_out = decoder_lstm_layer(
        decoder_embedding, initial_state=[decoder_state_input_h, decoder_state_input_c]
    )

    decoder_softmax = training_model.get_layer("decoder_softmax")
    decoder_outputs = decoder_softmax(decoder_outputs)

    decoder_model = keras.Model(
        [decoder_inputs, decoder_state_input_h, decoder_state_input_c],
        [decoder_outputs, state_h_out, state_c_out]
    )
    return encoder_model, decoder_model

def predict_translation(model, sentence, preprocessing_path):
    with open(preprocessing_path, "rb") as f:
        pp = pickle.load(f)
    
    eng_tokenizer = pp["eng_tokenizer"]
    fra_tokenizer = pp["fra_tokenizer"]
    max_encoder_len = pp["max_encoder_len"]
    max_decoder_len = pp["max_decoder_len"]
    start_token_id = pp["start_token_id"]
    end_token_id = pp["end_token_id"]
    fra_index_to_word = {idx: word for word, idx in fra_tokenizer.word_index.items()}

    encoder_model, decoder_model = build_inference_models(model)

    seq = eng_tokenizer.texts_to_sequences([clean_text(sentence, "translation")])
    seq = keras.preprocessing.sequence.pad_sequences(seq, maxlen=max_encoder_len, padding="post")

    states = encoder_model.predict(seq, verbose=0)
    target_seq = np.array([[start_token_id]], dtype=np.int32)

    decoded_tokens = []
    for _ in range(max_decoder_len):
        output_tokens, h, c = decoder_model.predict([target_seq] + states, verbose=0)
        sampled_token_id = int(np.argmax(output_tokens[0, -1, :]))

        if sampled_token_id == end_token_id or sampled_token_id == 0:
            break

        decoded_tokens.append(fra_index_to_word.get(sampled_token_id, "<unk>"))
        target_seq = np.array([[sampled_token_id]], dtype=np.int32)
        states = [h, c]

    return " ".join(decoded_tokens)

# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="Test model predictions.")
    parser.add_argument("task", choices=["autocomplete", "news", "translation"], help="Task to run")
    parser.add_argument("--input", required=True, help="Input text for prediction")
    parser.add_argument("--model", help="Path to .keras model file")
    parser.add_argument("--preprocessing", help="Path to preprocessing file")
    parser.add_argument("--length", type=int, default=100, help="Generation length (for autocomplete)")
    parser.add_argument("--method", choices=["greedy", "temperature", "beam"], default="temperature", help="Decoding method (for autocomplete)")
    parser.add_argument("--temp", type=float, default=0.7, help="Sampling temperature (for autocomplete)")
    parser.add_argument("--beam-width", type=int, default=5, help="Beam width (for autocomplete beam search)")

    args = parser.parse_args()

    # Defaults
    defaults = {
        "autocomplete": {
            "model": "best_frankenstein_char_lstm.keras",
            "preprocessing": "frankenstein_char_preprocessing.json"
        },
        "news": {
            "model": "best_ag_news_lstm.keras",
            "preprocessing": "ag_news_preprocessing.pkl"
        },
        "translation": {
            "model": "best_eng_fra_seq2seq_lstm.keras",
            "preprocessing": "translation_preprocessing.pkl"
        }
    }

    model_path = args.model or defaults[args.task]["model"]
    prep_path = args.preprocessing or defaults[args.task]["preprocessing"]

    if not Path(model_path).exists():
        print(f"Error: Model file {model_path} not found.")
        return
    if not Path(prep_path).exists():
        print(f"Error: Preprocessing file {prep_path} not found.")
        return

    print(f"Loading {args.task} model...")
    model = keras.models.load_model(model_path)

    print("-" * 30)
    if args.task == "autocomplete":
        result = predict_autocomplete(model, args.input, prep_path, args.length, args.method, args.temp, args.beam_width)
        print(f"Method: {args.method} (temp: {args.temp if args.method=='temperature' else 'N/A'}, beam: {args.beam_width if args.method=='beam' else 'N/A'})")
        print(f"Generated text:\n{result}")
    elif args.task == "news":
        label, conf = predict_news(model, args.input, prep_path)
        print(f"Predicted Category: {label} (Confidence: {conf:.3f})")
    elif args.task == "translation":
        result = predict_translation(model, args.input, prep_path)
        print(f"Translation: {result}")
    print("-" * 30)

if __name__ == "__main__":
    main()
