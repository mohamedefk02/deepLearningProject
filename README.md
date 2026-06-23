# Deep Learning Project

This repository contains a deep learning project for three text-based models built with TensorFlow and Keras.

## Models

1. **Frankenstein Character-Level Autocomplete LSTM**
   - Generates text continuations from Project Gutenberg's *Frankenstein*.
   - Uses a character-level vocabulary and an LSTM-based sequence model.
   - Inference is provided by `predict_autocomplete.py`.

2. **English-French Translation Seq2Seq LSTM**
   - Translates short English sentences into French.
   - Uses a bidirectional encoder and a decoder inference pipeline.
   - Inference is provided by `predict_translation.py`.

3. **AG News Topic Classification LSTM**
   - Predicts news category labels from text.
   - Uses tokenization, padding, and an LSTM classifier.
   - Inference is provided by `predict_news.py`.

## Repository Structure

- `best_frankenstein_char_lstm.keras` - Trained Frankenstein autocomplete model.
- `frankenstein_char_preprocessing.json` - Preprocessing artifact for char-level model.
- `best_eng_fra_seq2seq_lstm.keras` - Trained English-French translation model.
- `translation_preprocessing.pkl` - Preprocessing artifact for translation model.
- `best_ag_news_lstm.keras` - Trained AG News classification model.
- `ag_news_preprocessing.pkl` - Preprocessing artifact for news classification.
- `predict_autocomplete.py` - Run the character-level autocomplete model.
- `predict_translation.py` - Run the English-French translation model.
- `predict_news.py` - Run the AG News classification model.
- `save_preprocessing_artifacts.py` - Recreate preprocessing artifacts from source data.
- `requirements.txt` - Python package requirements.
- `part1_text_autocomplete_lstm.ipynb` - Notebook for building the autocomplete model.
- `part2_eng_fra_translation_seq2seq_lstm.ipynb` - Notebook for translation.
- `part3_ag_news_classification_lstm.ipynb` - Notebook for news classification.

## Installation

Create a Python virtual environment and install dependencies:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Autocomplete

```bash
python predict_autocomplete.py --prompt "the man said that" --length 120 --temperature 0.45
```

### Translation

```bash
python predict_translation.py --sentence "i am happy"
```

### News Classification

```bash
python predict_news.py --text "Technology stocks rise as chip demand increases"
```

## Recreate Preprocessing Artifacts

If preprocessing artifacts are missing or need to be regenerated, run:

```bash
python save_preprocessing_artifacts.py
```

Note: `save_preprocessing_artifacts.py` downloads the Frankenstein text from Project Gutenberg, expects `fra-eng/fra.txt` for translation data, and uses TensorFlow Datasets to build AG News tokenizer artifacts.

## Notes

- The project uses TensorFlow 2.x and Keras.
- The saved models and preprocessing artifacts are required for inference.
- The notebooks show the training and data preparation workflows.
