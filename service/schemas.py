# service/schemas.py
from typing import List
from pydantic import BaseModel


class CustomerFeatures(BaseModel):
    features: List[float]  # aligned with feature_cols.json


class RiskResponse(BaseModel):
    pd_score: float
    expected_loss: float
    cluster_id: int
    anomaly_score: float