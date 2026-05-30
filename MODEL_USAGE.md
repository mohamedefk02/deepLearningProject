# Model Usage

This project now saves preprocessing artifacts separately from the trained `.keras` models.

## Saved Artifacts

```text
best_frankenstein_char_lstm.keras
frankenstein_char_preprocessing.json

best_eng_fra_seq2seq_lstm.keras
translation_preprocessing.pkl

best_ag_news_lstm.keras
ag_news_preprocessing.pkl
```

The preprocessing files are required because text models need the same vocabulary, tokenization, sequence lengths, and label mapping used during training.

## Regenerate Preprocessing Artifacts

Run this if you need to recreate the preprocessing files:

```bash
python save_preprocessing_artifacts.py
```

## Autocomplete Prediction

```bash
python predict_autocomplete.py --prompt "the man said that" --length 120 --temperature 0.45
```

Without temperature, the script uses greedy decoding:

```bash
python predict_autocomplete.py --prompt "the man said that" --length 120
```

## Translation Prediction

```bash
python predict_translation.py --sentence "i am happy"
```

## AG News Prediction

```bash
python predict_news.py --text "Technology stocks rise as chip demand increases"
```
