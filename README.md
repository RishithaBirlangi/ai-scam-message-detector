# AI Scam Message Detector

This project classifies text messages as `spam/scam` or `safe` using machine learning and a simple Streamlit web app.

## Features

- Loads and trains on the `spam.csv` SMS spam dataset
- Cleans message text before vectorization
- Uses `TF-IDF` with a `Multinomial Naive Bayes` classifier
- Predicts scam probability for user-entered text
- Flags suspicious keywords such as `bank`, `otp`, `verify`, and `claim`
- Shows basic safety recommendations in the web app

## Files

- `app.py`: Streamlit web interface
- `main.py`: Command-line version
- `detector.py`: Shared dataset, training, and prediction logic
- `spam.csv`: Dataset used by the app

## Installation

```bash
pip install -r requirements.txt
```

## Run The App

```bash
streamlit run app.py
```

## Run The CLI Version

```bash
python main.py
```

## Workflow

1. Load the spam dataset.
2. Clean and normalize message text.
3. Convert text into numeric features using TF-IDF.
4. Train a Naive Bayes classifier.
5. Predict whether a new message is suspicious.

## Expected Outcome

Users can paste a message into the app and quickly see:

- Whether the message looks suspicious
- The estimated scam probability
- Matched suspicious keywords
- Safety recommendations for next steps
