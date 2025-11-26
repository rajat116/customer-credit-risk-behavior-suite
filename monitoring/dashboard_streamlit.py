# monitoring/dashboard_streamlit.py

# -------------------------------------------------------------
# Load environment variables (GEMINI_API_KEY from .env)
# -------------------------------------------------------------
from dotenv import load_dotenv
load_dotenv()

import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import json
import requests
import streamlit as st
import google.generativeai as genai

from src.config import MODELS_DIR


API_URL = "http://localhost:8000/predict"

# -------------------------------------------------------------
# Load model feature names (model order)
# -------------------------------------------------------------
with open(MODELS_DIR / "feature_cols.json") as f:
    feature_cols = json.load(f)

# -------------------------------------------------------------
# Human-friendly labels + descriptions
# -------------------------------------------------------------
FRIENDLY_INFO = {
    "util": (
        "Revolving Credit Utilization (%)",
        "Percentage of credit currently being used compared to total available credit."
    ),
    "age": (
        "Age (years)",
        "Customer's age in years. Older customers often default less."
    ),
    "num_30_59": (
        "# of 30–59 Day Late Payments",
        "Number of times the customer was 30–59 days late on a payment."
    ),
    "num_60_89": (
        "# of 60–89 Day Late Payments",
        "Number of times the customer was 60–89 days late on a payment."
    ),
    "num_90": (
        "# of 90+ Day Late Payments",
        "Number of times the customer was over 90 days late. This is the most severe delinquency."
    ),
    "DebtRatio": (
        "Debt Ratio",
        "Total monthly debt obligations divided by monthly income."
    ),
    "MonthlyIncome": (
        "Monthly Income ($)",
        "Customer's monthly income in USD."
    ),
    "num_credit_lines": (
        "Open Credit Lines",
        "Number of loans, credit cards, and other open credit lines."
    ),
    "num_real_estate": (
        "Real Estate Loans",
        "How many real estate loans or mortgage accounts the customer has."
    ),
    "NumberOfDependents": (
        "Dependents",
        "Number of family members financially dependent on the customer."
    ),
}

# -------------------------------------------------------------
# Human-friendly DISPLAY ORDER (NOT model order)
# -------------------------------------------------------------
DISPLAY_ORDER = [
    "util",
    "age",
    "num_30_59",      # mild delinquency
    "num_60_89",      # moderate
    "num_90",         # severe
    "DebtRatio",
    "MonthlyIncome",
    "num_credit_lines",
    "num_real_estate",
    "NumberOfDependents",
]

# -------------------------------------------------------------
# Gemini LLM Explanation Function
# -------------------------------------------------------------
def llm_explanation(pd_score, el, anomaly, cluster):
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return "Gemini API key missing. Add GEMINI_API_KEY to your .env file."

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")

    prompt = f"""
You are a senior banking underwriter. Provide a clear and concise credit risk explanation.

Inputs:
- Probability of Default: {pd_score:.2f}
- Expected Loss: {el:.0f}
- Behavioural Anomaly Score: {anomaly:.2f}
- Behavioural Cluster: {cluster}

Return:
1. The customer's overall risk category (very low / low / moderate / high / very high).
2. 3–4 bullet points interpreting these values in simple business language.
3. A final underwriting recommendation (approve / conditional approve / decline).
4. Tone: concise, structured, professional.
"""

    response = model.generate_content(prompt)
    return response.text


# -------------------------------------------------------------
# Dashboard Title & Intro
# -------------------------------------------------------------
st.title("📊 Customer Credit Risk Scoring Dashboard")

st.markdown("""
This dashboard provides **AI-powered credit risk scoring** using a model trained on a  
FICO-style consumer credit dataset.

### 🔍 What the system predicts:
- **PD (Probability of Default)** – likelihood of loan default  
- **LGD (Loss Given Default)** – expected monetary loss *if* default happens  
- **Expected Loss** = PD × LGD  
- **Cluster ID** – behaviour-based customer segment  
- **Anomaly Score** – how unusual their financial behaviour is  

### 📝 What you need to enter:
Provide customer financial & behavioural information below.  
Human-friendly names are shown, but data is internally mapped to the correct model order.
""")

st.markdown("---")

# -------------------------------------------------------------
# Input Section
# -------------------------------------------------------------
st.header("📝 Enter Customer Information")

inputs = {}

for col in DISPLAY_ORDER:
    label, desc = FRIENDLY_INFO.get(col, (col, ""))
    val = st.number_input(label, help=desc, value=0.0, min_value=0.0)
    inputs[col] = val

# Map into the exact order model expects
values = [inputs[col] for col in feature_cols]

st.markdown("---")

# -------------------------------------------------------------
# Prediction Section
# -------------------------------------------------------------
if st.button("⚡ Score Customer"):
    payload = {"features": values}
    resp = requests.post(API_URL, json=payload)

    if resp.status_code == 200:
        data = resp.json()
        st.success("Prediction complete!")

        # Main metrics
        col1, col2 = st.columns(2)
        col1.metric("PD Score", f"{data['pd_score']:.3f}")
        col1.metric("Expected Loss ($)", f"{data['expected_loss']:.2f}")

        col2.metric("Cluster ID", str(data["cluster_id"]))
        col2.metric("Anomaly Score", f"{data['anomaly_score']:.3f}")

        st.markdown("---")

        # Gemini LLM explanation
        st.markdown("### 🤖 AI-Generated Credit Explanation (Gemini)")
        with st.spinner("Generating explanation..."):
            explanation = llm_explanation(
                data["pd_score"],
                data["expected_loss"],
                data["anomaly_score"],
                data["cluster_id"]
            )
        st.info(explanation)

    else:
        st.error(f"API error {resp.status_code}: {resp.text}")