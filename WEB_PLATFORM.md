# Web Platform Usage

This adds a FastAPI backend and Vite React frontend for trying the three trained LSTM models from a browser.

The existing CLI commands are unchanged:

```bash
python predict.py autocomplete --input "seed text"
python predict.py translation --input "i am happy"
python predict.py news --input "Technology stocks rise..."
```

## Local Run on Windows

Run the backend from the project root in the Windows Conda environment that already has TensorFlow:

```bash
conda activate dl_env
pip install -r backend/requirements.txt
python -m uvicorn backend.main:app --reload
```

Backend URLs:

```text
http://localhost:8000/health
http://localhost:8000/docs
```

Run the frontend in another terminal:

```bash
cd frontend
npm install
npm run dev
```

Frontend URL:

```text
http://localhost:5173
```

## Web API Responses

- Autocomplete returns a list of completions. Greedy and temperature return one item; beam returns multiple candidates.
- Translation returns the full translation and a `words` array for the animation.
- News classification returns the winning label, confidence, and all four class scores for the chart.

## Oracle Cloud VM Notes

Deployment is possible if the VM can run TensorFlow and has enough memory for all three models. Use a Python 3.11 environment, install backend requirements, upload the model/preprocessing files, build the React frontend, then serve the app behind Nginx or another reverse proxy.

