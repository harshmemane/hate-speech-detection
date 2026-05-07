from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import os

# ── Load model ──────────────────────────────────────────
MODEL_PATH = "/app/hate_speech_model.pkl"

with open(MODEL_PATH, "rb") as f:
    bundle = pickle.load(f)

tfidf = bundle["tfidf"]
clf   = bundle["clf"]

LABEL_MAP = {0: "Not Hate", 1: "Hate Speech"}

# ── App ─────────────────────────────────────────────────
app = FastAPI(
    title="Hate Speech Detection API",
    description="MLOps project by Harsh Gokul Memane — MIT ADT University",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ── Schemas ─────────────────────────────────────────────
class TextInput(BaseModel):
    text: str

class PredictionOutput(BaseModel):
    text:       str
    label:      str
    label_id:   int
    confidence: float

# ── Routes ──────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status": "ok",
        "project": "Hate Speech Detection System",
        "author":  "Harsh Gokul Memane",
        "college": "MIT ADT University",
        "docs":    "/docs"
    }

@app.get("/health")
def health():
    return {"status": "healthy", "model": "TF-IDF + LogisticRegression"}

@app.post("/predict", response_model=PredictionOutput)
def predict(body: TextInput):
    vec        = tfidf.transform([body.text])
    label_id   = int(clf.predict(vec)[0])
    confidence = float(clf.predict_proba(vec)[0][label_id])
    return {
        "text":       body.text,
        "label":      LABEL_MAP[label_id],
        "label_id":   label_id,
        "confidence": round(confidence, 4)
    }
