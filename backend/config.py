from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

AUTOCOMPLETE_MODEL = ROOT / "best_frankenstein_char_lstm.keras"
AUTOCOMPLETE_PREPROCESSING = ROOT / "frankenstein_char_preprocessing.json"

TRANSLATION_MODEL = ROOT / "best_eng_fra_seq2seq_lstm.keras"
TRANSLATION_PREPROCESSING = ROOT / "translation_preprocessing.pkl"

CLASSIFICATION_MODEL = ROOT / "best_ag_news_lstm.keras"
CLASSIFICATION_PREPROCESSING = ROOT / "ag_news_preprocessing.pkl"

