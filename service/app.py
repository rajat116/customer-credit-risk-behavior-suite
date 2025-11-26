# service/app.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .schemas import CustomerFeatures, RiskResponse
from .model_loader import (
    prepare_inputs,
    get_pd_model,
    get_lgd_model,
    get_kmeans,
    get_iforest,
)

app = FastAPI(
    title="Customer Credit Risk & Behaviour API",
    version="1.0.0",
    description="PD, LGD approximation, segmentation & behaviour anomaly scoring.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=RiskResponse)
def predict_risk(payload: CustomerFeatures):
    X_s = prepare_inputs(payload.features)

    pd_model = get_pd_model()
    lgd_model = get_lgd_model()
    kmeans = get_kmeans()
    iforest = get_iforest()

    # -------- Correct calculations --------
    pd_score = float(pd_model.predict_proba(X_s)[0, 1])      # PD
    lgd = float(lgd_model.predict(X_s)[0])                   # LGD
    expected_loss = pd_score * lgd                           # Correct Expected Loss

    cluster_id = int(kmeans.predict(X_s)[0])
    anomaly_raw = float(iforest.decision_function(X_s)[0])
    anomaly_score = float(-anomaly_raw)

    return RiskResponse(
        pd_score=pd_score,
        expected_loss=expected_loss,   # Correct EL
        cluster_id=cluster_id,
        anomaly_score=anomaly_score,
    )