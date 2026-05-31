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


LSTM_UNITS = 256


def clean_sentence(text):
    text = str(text).lower().strip()
    text = re.sub(r"[^a-zA-Z\u00C0-\u00FF?.!,;:'\- ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_inference_models(training_model):
    # Encoder: Bidirectional
    encoder_inputs = training_model.get_layer("encoder_inputs").input
    
    # Extract states from bidirectional layer
    # Note: Keras Bidirectional LSTM with return_state returns [output, h_fwd, c_fwd, h_bwd, c_bwd]
    _, h_fwd, c_fwd, h_bwd, c_bwd = training_model.get_layer("bidirectional_encoder").output
    
    state_h = keras.layers.Concatenate()([h_fwd, h_bwd])
    state_c = keras.layers.Concatenate()([c_fwd, c_bwd])
    
    encoder_model = keras.Model(encoder_inputs, [state_h, state_c])

    # Decoder
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


def translate(sentence, training_model, preprocessing):
    eng_tokenizer = preprocessing["eng_tokenizer"]
    fra_tokenizer = preprocessing["fra_tokenizer"]
    max_encoder_len = preprocessing["max_encoder_len"]
    max_decoder_len = preprocessing["max_decoder_len"]
    start_token_id = preprocessing["start_token_id"]
    end_token_id = preprocessing["end_token_id"]
    fra_index_to_word = {idx: word for word, idx in fra_tokenizer.word_index.items()}

    encoder_model, decoder_model = build_inference_models(training_model)

    seq = eng_tokenizer.texts_to_sequences([clean_sentence(sentence)])
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="best_eng_fra_seq2seq_lstm.keras")
    parser.add_argument("--preprocessing", default="translation_preprocessing.pkl")
    parser.add_argument("--sentence", default="i am happy")
    args = parser.parse_args()

    if not Path(args.model).exists():
        raise FileNotFoundError(args.model)
    if not Path(args.preprocessing).exists():
        raise FileNotFoundError(args.preprocessing)

    with Path(args.preprocessing).open("rb") as f:
        preprocessing = pickle.load(f)

    model = keras.models.load_model(args.model)
    print(translate(args.sentence, model, preprocessing))


if __name__ == "__main__":
    main()
