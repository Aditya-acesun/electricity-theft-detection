---
title: Electricity Theft Detection
emoji: ⚡
colorFrom: yellow
colorTo: red
sdk: docker
pinned: false
---
# ⚡ Electricity Theft Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?style=flat-square&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-FF4B4B?style=flat-square&logo=streamlit)
![Docker](https://img.shields.io/badge/Docker-Deployed-2496ED?style=flat-square&logo=docker)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Spaces-FFD21E?style=flat-square&logo=huggingface)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

**An end-to-end AI system for detecting non-technical losses (NTL) in power distribution networks using ensemble machine learning and real-time explainability.**

[🚀 Live Demo](https://huggingface.co/spaces/Aditya-acesun/electricity-theft-detection) · [📊 API Docs](https://aditya-acesun-electricity-theft-detection.hf.space/docs) · [🐛 Report Bug](https://github.com/Aditya-acesun/electricity-theft-detection/issues)

</div>

---

## 📌 Overview

Electricity theft accounts for **$96 billion in annual global losses** to utility providers. Traditional rule-based detection methods fail to catch sophisticated tampering patterns. This project builds a production-grade ML pipeline that:

- Ingests 365-day smart meter consumption time-series
- Engineers 24 domain-specific behavioral features
- Scores customers using a weighted ensemble model
- Explains predictions via SHAP feature importance
- Streams real-time simulation of detection scenarios

The system is containerized with Docker and deployed on Hugging Face Spaces with a custom Streamlit frontend and FastAPI backend.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Frontend                        │
│        Single Customer │ Batch Analysis │ Live Simulation    │
└──────────────────────────────┬──────────────────────────────┘
                               │ HTTP
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI Backend                         │
│   /predict    /predict-batch    /explain    /health          │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                    ML Pipeline                               │
│  Feature Engineering → SMOTE → Ensemble Model → SHAP        │
└─────────────────────────────────────────────────────────────┘
```

---

## 🤖 Model

### Ensemble Architecture

| Model | Weight | Rationale |
|---|---|---|
| XGBoost | 50% | Primary learner — best on tabular fraud data |
| Random Forest | 25% | Reduces variance, handles feature noise |
| Gradient Boosting | 25% | Captures complex non-linear theft patterns |

### Training Details

| Metric | Value |
|---|---|
| Dataset | SGCC (State Grid Corporation of China) |
| Training samples | ~42,000 customers |
| Features | 24 engineered from 365-day readings |
| Class imbalance handling | **SMOTE** (Synthetic Minority Oversampling) |
| ROC-AUC | 0.7662 |
| Evaluation | Stratified k-fold cross-validation |

### Feature Engineering (24 Features)

Features are extracted from raw 365-day daily consumption readings:

| Category | Features |
|---|---|
| **Statistical** | mean, std, min, max, median, skewness, kurtosis |
| **Temporal** | weekday vs weekend ratio, monthly averages |
| **Anomaly** | zero-reading count, negative reading count, spike ratio |
| **Trend** | linear slope over time, variance in rolling windows |
| **Behavioral** | peak hour ratio, off-peak consumption ratio |

---

## 🧪 How It Works

### 1. Single Customer Analysis
Input 365 daily readings → get a theft probability score, risk tier, and SHAP explanation showing which features drove the prediction.

### 2. Batch Analysis
Upload a CSV of multiple customers → system scores all of them and returns a ranked risk table, downloadable as a report.

### 3. Live Simulation
Select customer type (Normal / Theft) → the system streams day-by-day meter readings and scores the customer every 10 days using the real API, showing the model catching theft patterns as they develop in real-time.

---

## 🔍 SHAP Explainability

The `/explain` endpoint uses `shap.TreeExplainer` on the XGBoost base model to return feature-level contribution scores for any prediction. This allows:

- Understanding *why* a customer was flagged
- Auditing the model for bias
- Building trust with domain experts (utility engineers)

```json
// GET /explain
{
  "shap_values": {
    "zero_reading_count": 0.412,
    "std_consumption": 0.298,
    "spike_ratio": 0.187,
    ...
  },
  "prediction": "theft",
  "confidence": 0.834
}
```

---

## 🚀 Running Locally

### Prerequisites
- Python 3.10+
- Docker (optional)

### Setup

```bash
git clone https://github.com/Aditya-acesun/electricity-theft-detection
cd electricity-theft-detection
pip install -r requirements.txt
```

### Start Backend (FastAPI)

```bash
uvicorn api.main:app --reload --port 8000
```

API docs available at `http://localhost:8000/docs`

### Start Frontend (Streamlit)

```bash
streamlit run frontend/app.py --server.port 8502
```

### Docker

```bash
docker build -t theft-detection .
docker run -p 8000:8000 -p 8502:8502 theft-detection
```

---

## 📁 Project Structure

```
electricity-theft-detection/
├── api/
│   ├── main.py              # FastAPI app, routes
│   ├── predict.py           # Prediction logic
│   └── explain.py           # SHAP endpoint
├── frontend/
│   └── app.py               # Streamlit UI
├── models/
│   ├── saved/               # Serialized model files (.pkl)
│   └── train.py             # Training script
├── data/
│   └── preprocessing.py     # Feature engineering pipeline
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🌐 Deployment

The app is deployed on **Hugging Face Spaces** using a custom Docker container:

- FastAPI backend runs on port `8000` (internal)
- Streamlit frontend is exposed on port `7860` (HF default)
- Model files are stored at `/app/models/` inside the container
- Health check endpoint at `/healthz` keeps the Space alive

**Key deployment considerations:**
- scikit-learn and XGBoost versions are pinned in `requirements.txt` to prevent serialization mismatches between local training and HF inference
- SHAP is initialized lazily to reduce cold start time

---

## 📈 Results

The model was evaluated on a held-out 20% test split:

```
              precision    recall  f1-score
    Normal       0.94      0.89      0.91
     Theft       0.61      0.74      0.67

  ROC-AUC: 0.7662
```

> **Note on threshold tuning:** Default decision threshold was tuned from `0.867 → 0.15` to prioritize recall on the theft class — in real utility fraud detection, false negatives (missed theft) are far more costly than false positives (manual investigation).

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| ML | XGBoost, scikit-learn, SHAP |
| Imbalance | imbalanced-learn (SMOTE) |
| Backend | FastAPI, Uvicorn |
| Frontend | Streamlit |
| Visualization | Plotly, Altair |
| Containerization | Docker |
| Deployment | Hugging Face Spaces |

---

## 🔮 Future Work

- [ ] Add LSTM-based temporal model for sequence-aware detection
- [ ] Integrate real-time SCADA data stream via WebSocket
- [ ] Add customer clustering to segment theft typologies
- [ ] Build model monitoring dashboard (data drift, prediction distribution)
- [ ] Experiment with Graph Neural Networks on meter network topology

---

## 👤 Author

**Aditya** — AI & Data Science, TCET Mumbai  
[LinkedIn](https://www.linkedin.com/in/aditya-yadav-3a3493356) · [Hugging Face](https://huggingface.co/Aditya-acesun) · [GitHub](https://github.com/Aditya-acesun)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
<i>Built with domain knowledge, not just code.</i>
</div>
