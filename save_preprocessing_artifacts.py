import json
import pickle
import re
from pathlib import Path

import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras


FRANKENSTEIN_URL = "https://www.gutenberg.org/files/84/84-0.txt"
SEQ_LENGTH = 60

TRANSLATION_MAX_WORDS = 12_000
NEWS_MAX_WORDS = 30_000
NEWS_MAX_LEN = 120


def normalize_spaces(text):
    return re.sub(r"\s+", " ", text.lower()).strip()


def clean_sentence(text):
    text = str(text).lower().strip()
    text = re.sub(r"[^a-zA-Z\u00C0-\u00FF?.!,;:'\- ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_news_text(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z0-9\s.,!?'-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def save_frankenstein_artifact():
    path = keras.utils.get_file("frankenstein_gutenberg_84.txt", FRANKENSTEIN_URL)
    raw_text = Path(path).read_text(encoding="utf-8")

    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK FRANKENSTEIN"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK FRANKENSTEIN"
    start_idx = raw_text.find(start_marker)
    end_idx = raw_text.find(end_marker)
    if start_idx != -1 and end_idx != -1:
        raw_text = raw_text[start_idx:end_idx]

    text = normalize_spaces(raw_text)
    chars = sorted(set(text))

    artifact = {
        "seq_length": SEQ_LENGTH,
        "chars": chars,
        "vocab_size": len(chars),
        "source": FRANKENSTEIN_URL,
    }

    Path("frankenstein_char_preprocessing.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("Saved frankenstein_char_preprocessing.json")


def save_translation_artifact():
    fra_path = Path("fra-eng") / "fra.txt"
    if not fra_path.exists():
        raise FileNotFoundError(
            "Missing fra-eng/fra.txt. Extract fra-eng.zip or run the translation dataset cell first."
        )

    raw_pairs = pd.read_csv(
        fra_path,
        sep="\t",
        header=None,
        names=["english", "french", "metadata"],
        usecols=[0, 1, 2],
    )

    pairs = raw_pairs[["english", "french"]].dropna().head(10_000).copy()
    pairs["english"] = pairs["english"].map(clean_sentence)
    pairs["french"] = pairs["french"].map(clean_sentence)
    pairs = pairs[
        (pairs["english"].str.len() > 0) & (pairs["french"].str.len() > 0)
    ].reset_index(drop=True)

    pairs["french_in"] = "<start> " + pairs["french"]
    pairs["french_out"] = pairs["french"] + " <end>"

    eng_tokenizer = keras.preprocessing.text.Tokenizer(
        num_words=TRANSLATION_MAX_WORDS,
        filters="",
        lower=False,
        oov_token="<unk>",
    )
    fra_tokenizer = keras.preprocessing.text.Tokenizer(
        num_words=TRANSLATION_MAX_WORDS,
        filters="",
        lower=False,
        oov_token="<unk>",
    )

    eng_tokenizer.fit_on_texts(pairs["english"])
    fra_tokenizer.fit_on_texts(pd.concat([pairs["french_in"], pairs["french_out"]]))

    encoder_sequences = eng_tokenizer.texts_to_sequences(pairs["english"])
    decoder_input_sequences = fra_tokenizer.texts_to_sequences(pairs["french_in"])

    artifact = {
        "eng_tokenizer": eng_tokenizer,
        "fra_tokenizer": fra_tokenizer,
        "max_encoder_len": max(len(seq) for seq in encoder_sequences),
        "max_decoder_len": max(len(seq) for seq in decoder_input_sequences),
        "eng_vocab_size": min(TRANSLATION_MAX_WORDS, len(eng_tokenizer.word_index) + 1),
        "fra_vocab_size": min(TRANSLATION_MAX_WORDS, len(fra_tokenizer.word_index) + 1),
        "start_token_id": fra_tokenizer.word_index["<start>"],
        "end_token_id": fra_tokenizer.word_index["<end>"],
    }

    with Path("translation_preprocessing.pkl").open("wb") as f:
        pickle.dump(artifact, f)
    print("Saved translation_preprocessing.pkl")


def save_ag_news_artifact():
    ag_train_raw = tfds.load("ag_news_subset", split="train", as_supervised=True)

    train_texts = []
    for text_tensor, _ in tfds.as_numpy(ag_train_raw):
        train_texts.append(text_tensor.decode("utf-8"))

    tokenizer = keras.preprocessing.text.Tokenizer(
        num_words=NEWS_MAX_WORDS,
        oov_token="<unk>",
    )
    tokenizer.fit_on_texts([clean_news_text(text) for text in train_texts])

    artifact = {
        "news_tokenizer": tokenizer,
        "max_news_len": NEWS_MAX_LEN,
        "news_vocab_size": min(NEWS_MAX_WORDS, len(tokenizer.word_index) + 1),
        "class_names": ["World", "Sports", "Business", "Sci/Tech"],
    }

    with Path("ag_news_preprocessing.pkl").open("wb") as f:
        pickle.dump(artifact, f)
    print("Saved ag_news_preprocessing.pkl")


def main():
    print("TensorFlow version:", tf.__version__)
    save_frankenstein_artifact()
    save_translation_artifact()
    save_ag_news_artifact()


if __name__ == "__main__":
    main()
