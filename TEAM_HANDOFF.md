# Team Handoff: LSTM Project Models

To use the models I've trained, you need two types of files: the **.keras** files (the model's "brain") and the **.pkl/.json** files (the "filters" that clean and tokenize the text).

## 1. Required Files
Ensure you have these in your folder:

| Task | Model File | Preprocessing File |
| :--- | :--- | :--- |
| **Autocomplete** | `best_frankenstein_char_lstm.keras` | `frankenstein_char_preprocessing.json` |
| **Translation** | `best_eng_fra_seq2seq_lstm.keras` | `translation_preprocessing.pkl` |
| **Classification** | `best_ag_news_lstm.keras` | `ag_news_preprocessing.pkl` |

Also include the unified testing script: `predict.py`.

## 2. Setup
Install the necessary libraries:
```bash
pip install tensorflow numpy pandas scikit-learn
```

## 3. How to Run Predictions
I've created a unified script called `predict.py` to make testing easy.

### Autocomplete (Part 1)
```bash
python predict.py autocomplete --input "it was on a dreary " --method temperature --temp 0.7
```

### Translation (Part 2)
```bash
python predict.py translation --input "i am happy"
```

### News Classification (Part 3)
```bash
python predict.py news --input "The latest stock market trends show a significant rise."
```

## 4. Notes
- The models were trained on **TensorFlow 2.15+**.
- If you get an error about `WINDOW_LENGTH`, ensure the `predict.py` script matches the version I sent in the zip.
