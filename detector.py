from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


DATASET_PATH = Path(__file__).with_name("spam.csv")
SCAM_KEYWORDS = [
    "bank",
    "account",
    "password",
    "otp",
    "verify",
    "urgent",
    "click",
    "link",
    "prize",
    "win",
    "payment",
    "blocked",
    "suspended",
    "update",
    "gift",
    "offer",
    "claim",
]
SAFETY_RECOMMENDATIONS = [
    "Do not click unknown links or download attachments.",
    "Do not share OTPs, passwords, or bank details by message.",
    "Verify the sender through an official website or phone number.",
    "If money or urgency is involved, pause and confirm with a trusted source.",
]


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_dataset(dataset_path: Path = DATASET_PATH) -> pd.DataFrame:
    df = pd.read_csv(dataset_path, usecols=["v1", "v2"], encoding="latin-1")
    df = df.rename(columns={"v1": "label", "v2": "message"})
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    df["message"] = df["message"].fillna("").apply(clean_text)
    return df.dropna(subset=["label"])


def train_model(test_size: float = 0.2, random_state: int = 42) -> dict:
    df = load_dataset()
    x_train, x_test, y_train, y_test = train_test_split(
        df["message"],
        df["label"],
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"],
    )

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)

    model = MultinomialNB()
    model.fit(x_train_vec, y_train)

    y_pred = model.predict(x_test_vec)
    accuracy = accuracy_score(y_test, y_pred)

    return {
        "dataset": df,
        "vectorizer": vectorizer,
        "model": model,
        "accuracy": accuracy,
    }


def find_scam_keywords(message: str) -> list[str]:
    lowered_message = message.lower()
    return [keyword for keyword in SCAM_KEYWORDS if keyword in lowered_message]


def predict_message(message: str, model, vectorizer) -> dict:
    cleaned_message = clean_text(message)
    vector_input = vectorizer.transform([cleaned_message])
    prediction = int(model.predict(vector_input)[0])
    probabilities = model.predict_proba(vector_input)[0]
    keyword_hits = find_scam_keywords(message)
    scam_probability = float(probabilities[1])

    adjusted_probability = max(scam_probability, min(0.99, 0.15 * len(keyword_hits)))
    is_scam = prediction == 1 or bool(keyword_hits)

    return {
        "is_scam": is_scam,
        "model_prediction": prediction,
        "scam_probability": adjusted_probability,
        "keyword_hits": keyword_hits,
        "cleaned_message": cleaned_message,
    }
