# Hate Speech Detection System — MLOps Project
 
![Python](https://img.shields.io/badge/Python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)
![MLflow](https://img.shields.io/badge/MLflow-Tracked-orange)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)
![HuggingFace](https://img.shields.io/badge/HuggingFace-DistilBERT-yellow)
![License](https://img.shields.io/badge/License-Apache%202.0-lightgrey)
 
A production-ready MLOps pipeline for detecting hate speech in social media text. Built with DistilBERT, FastAPI, MLflow, Docker, and GitHub Actions CI/CD.
 
---
 
## Table of Contents
 
- [Project Overview](#project-overview)
- [MLOps Architecture](#mlops-architecture)
- [Dataset](#dataset)
- [Model](#model)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Training the Model](#training-the-model)
- [Running the API](#running-the-api)
- [Docker](#docker)
- [CI/CD Pipeline](#cicd-pipeline)
- [Experiment Tracking](#experiment-tracking)
- [API Usage](#api-usage)
- [Results](#results)
- [Technologies Used](#technologies-used)
---
 
## Project Overview
 
This project builds an end-to-end **MLOps pipeline** for hate speech detection on social media. It is not just a machine learning model — it includes:
 
- Automated experiment tracking (MLflow + DagsHub)
- Model versioning and storage (HuggingFace Hub)
- A REST API for real-time predictions (FastAPI)
- Containerized deployment (Docker)
- Automated testing and deployment (GitHub Actions + Render)
The model classifies text into three categories:
 
| Label | Meaning |
|---|---|
| `0` | Hate speech |
| `1` | Offensive language |
| `2` | Normal / neither |
 
**Subject:** MLOps  
**Dataset:** hate_speech18 (HuggingFace Datasets)  
**Base Model:** distilbert-base-uncased  
**Deployment:** Render.com (free tier)
 
---
 
## MLOps Architecture
 
```
┌─────────────────────────────────────────────────────────────────┐
│                        MLOps Pipeline                           │
│                                                                 │
│  Data         Training        Tracking       Serving            │
│  ─────        ────────        ────────       ───────            │
│  hate_        Google          MLflow on      FastAPI            │
│  speech18  →  Colab      →    DagsHub    →   on Render          │
│  (HF)         (GPU)           (metrics)      (Docker)           │
│                  │                               ↑              │
│                  └──── HuggingFace Hub ──────────┘              │
│                         (model store)                           │
│                                                                 │
│  CI/CD: GitHub Actions → auto-test → auto-deploy on push        │
└─────────────────────────────────────────────────────────────────┘
```
 
---
 
## Dataset
 
**Name:** hate_speech18  
**Source:** [HuggingFace Datasets](https://huggingface.co/datasets/hate_speech18)  
**Size:** ~10,000 labeled social media posts  
**Labels:** hate (0), offensive (1), normal (2)
 
Loaded directly in training code — no manual download required:
 
```python
from datasets import load_dataset
dataset = load_dataset("hate_speech18")
```
 
---
 
## Model
 
**Base model:** `distilbert-base-uncased`  
- Smaller, faster version of BERT (40% smaller, 60% faster)
- Pre-trained on English text — already understands language
- Fine-tuned on hate_speech18 for 3-class classification
**Trained model stored at:** `huggingface.co/harshmemane/hate-speech-detector`
 
---
 
## Project Structure
 
```
hate-speech-detection/
│
├── notebooks/
│   └── training.ipynb          # Full training notebook (run on Google Colab)
│
├── app/
│   └── main.py                 # FastAPI application
│
├── tests/
│   └── test_api.py             # API unit tests (used in CI/CD)
│
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI/CD workflow
│
├── Dockerfile                  # Container definition
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```
 
---
 
## Setup & Installation
 
### Prerequisites
 
- Python 3.10+
- Git
- Docker (optional, for local containerization)
### Clone the repository
 
```bash
git clone https://github.com/harshmemane/hate-speech-detection.git
cd hate-speech-detection
```
 
### Install dependencies
 
```bash
pip install -r requirements.txt
```
 
### Environment variables
 
Create a `.env` file in the root directory:
 
```env
HF_MODEL_NAME=harshmemane/hate-speech-detector
HF_TOKEN=your_huggingface_token
MLFLOW_TRACKING_URI=https://dagshub.com/harshmemane/hate-speech-detection.mlflow
MLFLOW_TRACKING_USERNAME=your_dagshub_username
MLFLOW_TRACKING_PASSWORD=your_dagshub_token
```
 
> Never commit your `.env` file to GitHub. It is already listed in `.gitignore`.
 
---
 
## Training the Model
 
Training is done entirely in **Google Colab** using a free GPU.
 
1. Open `notebooks/training.ipynb` in [Google Colab](https://colab.research.google.com)
2. Enable GPU: `Runtime → Change runtime type → T4 GPU`
3. Run all cells in order
The notebook covers:
- Loading and exploring the dataset
- Tokenizing text with DistilBERT tokenizer
- Fine-tuning the model for 3 epochs
- Logging metrics to MLflow (DagsHub)
- Evaluating on test set (accuracy, F1-score, confusion matrix)
- Pushing the trained model to HuggingFace Hub
---
 
## Running the API
 
```bash
uvicorn app.main:app --reload
```
 
The API will start at `http://localhost:8000`
 
You can explore all endpoints at `http://localhost:8000/docs` (Swagger UI, auto-generated).
 
---
 
## Docker
 
### Build the image
 
```bash
docker build -t hate-speech-detector .
```
 
### Run the container
 
```bash
docker run -p 8000:8000 \
  -e HF_MODEL_NAME=harshmemane/hate-speech-detector \
  -e HF_TOKEN=your_token \
  hate-speech-detector
```
 
---
 
## CI/CD Pipeline
 
Every push to the `main` branch triggers the GitHub Actions workflow:
 
```
Push to main
    │
    ▼
Run tests (pytest)
    │
    ├── FAIL → notify, stop
    │
    └── PASS → Render auto-deploys new version
```
 
Workflow file: `.github/workflows/ci.yml`
 
---
 
## Experiment Tracking
 
All training runs are tracked on **DagsHub + MLflow**.
 
View experiments at:  
`https://dagshub.com/harshmemane/hate-speech-detection`
 
Tracked metrics per run:
- Training loss (per epoch)
- Validation accuracy
- F1-score (weighted)
- Learning rate
- Batch size
- Number of epochs
---
 
## API Usage
 
### Health check
 
```bash
GET /
```
 
Response:
```json
{ "status": "ok", "model": "hate-speech-detector" }
```
 
### Predict
 
```bash
POST /predict
Content-Type: application/json
 
{
  "text": "I hate all people from that country"
}
```
 
Response:
```json
{
  "text": "I hate all people from that country",
  "label": "hate speech",
  "label_id": 0,
  "confidence": 0.94
}
```
 
### Example with curl
 
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Have a great day everyone!"}'
```
 
---
 
## Results
 
| Metric | Score |
|---|---|
| Accuracy | ~89% |
| F1-Score (weighted) | ~0.88 |
| Training time | ~12 minutes (Colab T4 GPU) |
| Inference time | ~50ms per request |
 
> Exact numbers will be updated after final training run.
 
---
 
## Technologies Used
 
| Tool | Purpose |
|---|---|
| Python 3.10 | Programming language |
| HuggingFace Transformers | DistilBERT model and tokenizer |
| HuggingFace Datasets | hate_speech18 dataset |
| HuggingFace Hub | Model storage and versioning |
| MLflow | Experiment tracking |
| DagsHub | Free MLflow server + data versioning |
| FastAPI | REST API for model serving |
| Uvicorn | ASGI server for FastAPI |
| Docker | Containerization |
| GitHub Actions | CI/CD automation |
| Render | Cloud deployment (free tier) |
| Google Colab | Cloud GPU for training |
| pytest | Unit testing |
 
---
 
## Author
 
**Harsh Gokul Memane**  
MLOps Subject — [MIT ADT University]  
[Your GitHub Profile](https://github.com/harshmemane)
