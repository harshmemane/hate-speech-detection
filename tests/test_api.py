from fastapi.testclient import TestClient
import sys
import os
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Create a tiny dummy model for testing
# so tests work without the real model file
tfidf = TfidfVectorizer()
tfidf.fit(["hello world", "hate speech example"])
clf = LogisticRegression()
clf.fit(tfidf.transform(["hello world", "hate speech example"]), [0, 1])

os.makedirs("app", exist_ok=True)
with open("app/hate_speech_model.pkl", "wb") as f:
    pickle.dump({"tfidf": tfidf, "clf": clf}, f)

# Now import the app
sys.path.insert(0, "app")
from main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict():
    response = client.post(
        "/predict",
        json={"text": "Have a great day!"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "label" in data
    assert "confidence" in data
    assert data["label"] in ["Not Hate", "Hate Speech"]

def test_predict_returns_confidence():
    response = client.post(
        "/predict",
        json={"text": "some test text"}
    )
    data = response.json()
    assert 0.0 <= data["confidence"] <= 1.0
