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

LSTM_UNITS = 256


def _clean_sentence(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"[^a-zA-Z\u00C0-\u00FF?.!,;:'\- ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _build_inference_models(training_model):
    encoder_inputs = training_model.inputs[0]
    _, h_fwd, c_fwd, h_bwd, c_bwd = training_model.get_layer("bidirectional_encoder").output

    state_h = keras.layers.Concatenate()([h_fwd, h_bwd])
    state_c = keras.layers.Concatenate()([c_fwd, c_bwd])
    encoder_model = keras.Model(encoder_inputs, [state_h, state_c])

    decoder_inputs = keras.Input(shape=(None,), name="decoder_token_input")
    decoder_state_input_h = keras.Input(shape=(LSTM_UNITS * 2,), name="decoder_state_input_h")
    decoder_state_input_c = keras.Input(shape=(LSTM_UNITS * 2,), name="decoder_state_input_c")

    decoder_embedding = training_model.get_layer("decoder_embedding")(decoder_inputs)
    decoder_lstm_layer = training_model.get_layer("decoder_lstm")
    decoder_outputs, state_h_out, state_c_out = decoder_lstm_layer(
        decoder_embedding,
        initial_state=[decoder_state_input_h, decoder_state_input_c],
    )

    decoder_softmax = training_model.get_layer("decoder_softmax")
    decoder_outputs = decoder_softmax(decoder_outputs)

    decoder_model = keras.Model(
        [decoder_inputs, decoder_state_input_h, decoder_state_input_c],
        [decoder_outputs, state_h_out, state_c_out],
    )
    return encoder_model, decoder_model


def load_translation_model(model_path: Path, preprocessing_path: Path):
    with preprocessing_path.open("rb") as f:
        preprocessing = pickle.load(f)

    model = keras.models.load_model(model_path)
    encoder_model, decoder_model = _build_inference_models(model)
    return {
        "model": model,
        "encoder_model": encoder_model,
        "decoder_model": decoder_model,
        "preprocessing": preprocessing,
        "fra_index_to_word": {
            idx: word for word, idx in preprocessing["fra_tokenizer"].word_index.items()
        },
    }


def run_translation(store, sentence: str):
    pp = store["preprocessing"]
    eng_tokenizer = pp["eng_tokenizer"]
    max_encoder_len = pp["max_encoder_len"]
    max_decoder_len = pp["max_decoder_len"]
    start_token_id = pp["start_token_id"]
    end_token_id = pp["end_token_id"]

    seq = eng_tokenizer.texts_to_sequences([_clean_sentence(sentence)])
    seq = keras.preprocessing.sequence.pad_sequences(seq, maxlen=max_encoder_len, padding="post")

    states = store["encoder_model"].predict(seq, verbose=0)
    target_seq = np.array([[start_token_id]], dtype=np.int32)

    words = []
    for _ in range(max_decoder_len):
        output_tokens, h, c = store["decoder_model"].predict([target_seq] + states, verbose=0)
        sampled_token_id = int(np.argmax(output_tokens[0, -1, :]))

        if sampled_token_id == end_token_id or sampled_token_id == 0:
            break

        words.append(store["fra_index_to_word"].get(sampled_token_id, "<unk>"))
        target_seq = np.array([[sampled_token_id]], dtype=np.int32)
        states = [h, c]

    translation = " ".join(words)
    return {"translation": translation, "words": words}
