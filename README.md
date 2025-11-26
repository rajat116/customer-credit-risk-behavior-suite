Here is a **clean, professional, concise README.md** for your project — not too long, but complete enough for GitHub and recruiters.

---

# 📘 **Customer Credit Risk & Behaviour Scoring**

A full end-to-end machine learning system for **credit risk assessment**, including:

* **PD (Probability of Default)**
* **LGD (Loss Given Default)**
* **Expected Loss = PD × LGD**
* **Customer Segmentation (KMeans)**
* **Behavioural Anomaly Detection (Isolation Forest)**
* **Time-Series Default Rate Forecasting (ARIMA)**
* **Autoencoder for deep behavioural embeddings**
* **FastAPI model-serving microservice**
* **Streamlit dashboard for interactive scoring**

Built using Python, DuckDB, XGBoost, LightGBM, PyTorch, and FastAPI.

---

## 🚀 Features

### 🧠 Machine Learning Models

* **PD Model:** XGBoost classifier using 10 engineered financial features
* **LGD Model:** LightGBM regressor with realistic synthetic LGD target
* **Expected Loss:** Automatically computed inside API
* **Segmentation:** KMeans clustering of scaled financial profiles
* **Anomaly Detection:** Isolation Forest on customer behaviour
* **Autoencoder:** 8-dim latent embeddings (PyTorch)
* **Time-Series:** ARIMA model for monthly default rate

### 🔧 Engineering & Serving

* **FastAPI** microservice for scoring (`/predict`)
* **Dockerfile** for containerized deployment
* **Model loader with caching** for fast inference
* **Consistent feature ordering** via `feature_cols.json`

### 📊 Dashboard

Interactive **Streamlit dashboard** with:

* Human-friendly input fields
* Tooltips describing each feature
* Clean, fintech-style UX
* Displays PD, Expected Loss, Cluster, and Anomaly Score

---

## 📂 Project Structure

```
project/
│
├── src/
│   ├── features_sql.py        # DuckDB-based feature engineering
│   ├── train_pd_model.py      # PD model training
│   ├── train_lgd_model.py     # LGD model training
│   ├── train_unsupervised.py  # KMeans + IsolationForest
│   ├── train_timeseries.py    # ARIMA default rate forecast
│   ├── train_autoencoder.py   # PyTorch autoencoder
│   ├── config.py              # Paths
│   └── utils.py               # Shared utilities
│
├── service/
│   ├── app.py                 # FastAPI app
│   ├── model_loader.py        # Cached model loading + scaling
│   └── schemas.py             # Input/output schemas
│
├── monitoring/
│   └── dashboard_streamlit.py # Streamlit frontend
│
├── models/                    # Saved models & scalers
├── data/                      # Raw + processed data
└── Dockerfile                 # FastAPI deployment
```

---

## ▶️ Running the System

### **1. Start FastAPI**

```bash
uvicorn service.app:app --reload --host 0.0.0.0 --port 8000
```

### **2. Start Streamlit**

```bash
streamlit run monitoring/dashboard_streamlit.py
```

### API Docs

[http://localhost:8000/docs](http://localhost:8000/docs)

### Streamlit Dashboard

[http://localhost:8501](http://localhost:8501)

---

## 📈 Example Predictions

### Good customer:

* PD ≈ **0.05**
* Expected Loss ≈ **400**
* Cluster = **3**
* Anomaly Score = small/negative

### Risky customer:

* PD ≈ **0.8**
* Expected Loss ≈ **13k**
* Cluster = **3**
* Anomaly Score = positive

---

## 🛠 Technologies Used

* Python 3.11
* DuckDB
* XGBoost, LightGBM
* PyTorch
* FastAPI
* Streamlit
* scikit-learn
* statsmodels (ARIMA)
* Docker

---

## 📄 License

MIT License — free to use, modify, and build upon.

---