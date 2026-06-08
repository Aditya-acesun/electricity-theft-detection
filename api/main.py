from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib
import json
import os

app = FastAPI(
    title="Electricity Theft Detection API",
    description="Ensemble ML fraud detection system",
    version="2.0.0"
)

# ── Auto-detect path ──────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = "/app" if os.path.exists("/app/xgb_model.pkl") else os.path.join(BASE_DIR, "models", "saved")
print(f"📁 Loading models from: {MODEL_DIR}")

def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Not found: {path}")
    return joblib.load(path)

try:
    xgb_model = load_model(f"{MODEL_DIR}/xgb_model.pkl")
    print("✅ XGBoost loaded")
    rf_model  = load_model(f"{MODEL_DIR}/rf_model.pkl")
    print("✅ Random Forest loaded")
    gb_model  = load_model(f"{MODEL_DIR}/gb_model.pkl")
    print("✅ Gradient Boosting loaded")
    imputer   = load_model(f"{MODEL_DIR}/imputer.pkl")
    print("✅ Imputer loaded")

    with open(f"{MODEL_DIR}/model_info.json") as f:
        model_info = json.load(f)
    print("✅ Model info loaded")

    THRESHOLD = model_info['best_threshold']
    FEATURES  = model_info['features']
    WEIGHTS   = model_info.get('weights', [0.5, 0.25, 0.25])
    print(f"✅ Ensemble ready — AUC: {model_info['roc_auc']} | Threshold: {THRESHOLD}")

    # Load SHAP explainer for XGBoost (primary model)
    try:
        import shap
        explainer = shap.TreeExplainer(xgb_model)
        SHAP_AVAILABLE = True
        print("✅ SHAP explainer ready")
    except Exception as e:
        SHAP_AVAILABLE = False
        explainer = None
        print(f"⚠️ SHAP not available: {e}")

except Exception as e:
    print(f"❌ STARTUP ERROR: {e}")
    xgb_model = rf_model = gb_model = imputer = None
    model_info = {}
    THRESHOLD = 0.15
    FEATURES  = []
    WEIGHTS   = [0.5, 0.25, 0.25]
    SHAP_AVAILABLE = False
    explainer = None


def ensemble_predict_proba(X: pd.DataFrame) -> float:
    xgb_proba = xgb_model.predict_proba(X)[:, 1]
    rf_proba  = rf_model.predict_proba(X)[:, 1]
    gb_proba  = gb_model.predict_proba(X)[:, 1]
    ensemble  = (
        WEIGHTS[0] * xgb_proba +
        WEIGHTS[1] * rf_proba  +
        WEIGHTS[2] * gb_proba
    )
    return float(ensemble[0])


def engineer_features(readings):
    r        = pd.Series(readings)
    mean_val = r.mean()
    std_val  = r.std()
    mid = len(readings) // 2
    q   = len(readings) // 4

    max_consec = count = 0
    for v in readings:
        if v == 0 or pd.isna(v):
            count += 1
            max_consec = max(max_consec, count)
        else:
            count = 0

    first_half  = pd.Series(readings[:mid])
    second_half = pd.Series(readings[mid:])
    q1 = pd.Series(readings[:q])
    q4 = pd.Series(readings[3*q:])

    return pd.DataFrame([{
        'mean_consumption':   mean_val,
        'std_consumption':    std_val,
        'max_consumption':    r.max(),
        'min_consumption':    r.min(),
        'median_consumption': r.median(),
        'skewness':           r.skew(),
        'kurtosis':           r.kurtosis(),
        'zero_reading_rate':  (r == 0).mean(),
        'negative_count':     (r < 0).sum(),
        'missing_rate':       r.isna().mean(),
        'coeff_variation':    std_val / (mean_val + 1e-5),
        'low_reading_rate':   (r < 1).mean(),
        'first_half_mean':    first_half.mean(),
        'second_half_mean':   second_half.mean(),
        'consumption_trend':  second_half.mean() - first_half.mean(),
        'p25':                r.quantile(0.25),
        'p75':                r.quantile(0.75),
        'iqr':                r.quantile(0.75) - r.quantile(0.25),
        'max_consec_zeros':   max_consec,
        'max_to_mean_ratio':  r.max() / (mean_val + 1e-5),
        'q1_mean':            q1.mean(),
        'q4_mean':            q4.mean(),
        'q1_q4_ratio':        q1.mean() / (q4.mean() + 1e-5),
        'sudden_change':      abs(second_half.mean() - first_half.mean()) / (mean_val + 1e-5),
    }])


class CustomerData(BaseModel):
    customer_id: str
    readings: list[float]


@app.get("/")
def root():
    return {"message": "ETD Ensemble API v2.0", "status": "running"}

@app.get("/health")
def health():
    return {
        "status":    "healthy" if xgb_model is not None else "degraded",
        "model":     "Ensemble v2.0",
        "threshold": THRESHOLD,
        "shap":      SHAP_AVAILABLE,
    }

@app.get("/info")
def info():
    return {
        "model":      "Ensemble (XGB 50% + RF 25% + GB 25%)",
        "n_features": len(FEATURES),
        "threshold":  THRESHOLD,
        "roc_auc":    model_info.get("roc_auc", 0.7721),
        "theft_f1":   model_info.get("theft_f1", 0.0),
        "weights":    WEIGHTS,
    }

@app.post("/predict")
async def predict(data: CustomerData):
    if xgb_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded.")
    try:
        if len(data.readings) < 100:
            raise HTTPException(status_code=400, detail=f"Need 100+ readings, got {len(data.readings)}")

        X     = engineer_features(data.readings)
        X_imp = pd.DataFrame(imputer.transform(X), columns=X.columns)

        proba      = ensemble_predict_proba(X_imp)
        prediction = "THEFT SUSPECTED" if proba >= THRESHOLD else "NORMAL"
        risk_score = round(proba * 100, 2)
        risk_level = (
            "CRITICAL" if risk_score >= 80 else
            "HIGH"     if risk_score >= 60 else
            "MEDIUM"   if risk_score >= 40 else
            "LOW"
        )

        return {
            "customer_id": data.customer_id,
            "prediction":  prediction,
            "risk_score":  risk_score,
            "risk_level":  risk_level,
            "confidence":  round(proba, 4),
            "threshold":   THRESHOLD,
            "n_readings":  len(data.readings),
            "model":       "Ensemble v2.0"
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=f"{str(e)} | {traceback.format_exc()}")


@app.post("/explain")
async def explain(data: CustomerData):
    """SHAP explanation endpoint — returns feature contributions."""
    if xgb_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded.")
    if not SHAP_AVAILABLE:
        raise HTTPException(status_code=503, detail="SHAP not available.")
    try:
        if len(data.readings) < 100:
            raise HTTPException(status_code=400, detail=f"Need 100+ readings, got {len(data.readings)}")

        X     = engineer_features(data.readings)
        X_imp = pd.DataFrame(imputer.transform(X), columns=X.columns)

        # SHAP values from XGBoost (primary model)
        shap_values = explainer.shap_values(X_imp)

        # For binary classification shap_values is array of shape (1, n_features)
        if isinstance(shap_values, list):
            sv = shap_values[1][0]   # theft class
        else:
            sv = shap_values[0]      # single output

        feature_names  = list(X_imp.columns)
        feature_values = list(X_imp.iloc[0].values)
        base_value     = float(explainer.expected_value[1] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value)

        # Build sorted contributions
        contributions = []
        for name, val, feat_val in zip(feature_names, sv, feature_values):
            contributions.append({
                "feature":       name,
                "shap_value":    round(float(val), 4),
                "feature_value": round(float(feat_val), 4),
                "direction":     "increases_risk" if val > 0 else "decreases_risk"
            })

        # Sort by absolute impact
        contributions.sort(key=lambda x: abs(x['shap_value']), reverse=True)

        # Get ensemble proba too
        proba = ensemble_predict_proba(X_imp)

        return {
            "customer_id":   data.customer_id,
            "base_value":    round(base_value, 4),
            "final_score":   round(proba * 100, 2),
            "prediction":    "THEFT SUSPECTED" if proba >= THRESHOLD else "NORMAL",
            "contributions": contributions[:12],  # top 12 features
            "model":         "XGBoost (primary explainer)"
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=f"{str(e)} | {traceback.format_exc()}")
