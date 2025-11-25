# monitoring/dashboard_streamlit.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import json
import requests
import streamlit as st
from pathlib import Path

from src.config import MODELS_DIR

API_URL = "http://localhost:8000/predict"

st.title("Customer Credit Risk Behaviour Dashboard")

with open(MODELS_DIR / "feature_cols.json") as f:
    feature_cols = json.load(f)

st.write("Enter feature values in correct order:")

values = []
for col in feature_cols:
    val = st.number_input(col, value=0.0)
    values.append(val)

if st.button("Score customer"):
    payload = {"features": values}
    resp = requests.post(API_URL, json=payload)
    if resp.status_code == 200:
        data = resp.json()
        st.metric("PD Score", f"{data['pd_score']:.3f}")
        st.metric("Expected Loss", f"{data['expected_loss']:.2f}")
        st.metric("Cluster ID", str(data['cluster_id']))
        st.metric("Anomaly Score", f"{data['anomaly_score']:.3f}")
    else:
        st.error(f"API error: {resp.status_code} {resp.text}")