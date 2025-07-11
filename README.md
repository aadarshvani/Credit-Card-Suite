
## Credit Card Intelligence Suite 🧠💳

````markdown
# 💳 Credit Card Intelligence Suite

A modular, production-ready machine learning pipeline to tackle three critical problems in the credit card ecosystem:
1. 🔁 **Customer Churn Prediction**
2. 🚨 **Fraud Detection**
3. 💰 **Customer Lifetime Value (CLV) Estimation**

> ⚙️ Built using **Python**, **DVC**, **MLflow**, **Streamlit**, **FastAPI**, and containerized with **Docker**.  
> 🎯 Designed for real-time deployment with reusable components and tracked pipelines.

---

## 🚀 Project Highlights

| Feature                      | Details |
|-----------------------------|---------|
| 🔄 **Modular Pipelines**     | Each model (Churn, Fraud, CLV) has its own pipeline |
| 📊 **Trackable via DVC**     | Data & model versioning |
| 🧪 **MLflow Integration**    | Experiment tracking |
| 🖥️ **Streamlit Frontend**    | Interactive 3-page web UI |
| ⚡ **FastAPI Backend**       | Separate REST API for each model |
| 🐳 **Dockerized**            | Reproducible across environments |
| ☁️ **Cloud-Deployable**      | Compatible with Railway / AWS / GCP |

---

## 🧠 Business Use Cases

### 1. 🔁 Churn Prediction
Predict which customers are likely to stop using their credit cards, enabling proactive retention strategies.

### 2. 🚨 Fraud Detection
Identify suspicious transactions in real-time, helping prevent financial losses and protect user trust.

### 3. 💰 CLV Estimation
Forecast future value a customer will bring, guiding targeted marketing and credit offers.

---

## 🧱 Project Structure

```bash
credit-card-suite/
│
├── data/                     # Raw, processed, and feature-level data (DVC-managed)
├── models/                   # Trained model artifacts (not in Git, tracked by DVC)
├── src/                      # Core Python code per model
│   ├── churn/
│   ├── fraud/
│   ├── clv/
│   ├── utils/                # Logging, MLflow, helpers
│   └── config/               # Constants and paths
│
├── params/                   # Separate YAML files for model hyperparameters
│
├── backend/                  # FastAPI backend with model endpoints
├── frontend/                 # Streamlit frontend with multi-page interface
│
├── Dockerfile                # Docker container config
├── docker-compose.yml        # For full app deployment
├── dvc.yaml                  # DVC pipeline stages
├── setup.py                  # Editable install for src/
├── requirements.txt          # Global requirements
├── .gitignore
└── README.md
````

---

## ⚙️ How It Works

### 🔄 DVC Pipelines

Each model has its own stages:

* `data_ingestion`
* `data_preprocessing`
* `feature_engineering`
* `model_training`
* `model_evaluation`

```bash
dvc repro       # Reproduces pipeline
dvc dag         # Shows dependency graph
```

### 🧪 MLflow

All training runs and parameters are logged with MLflow:

```bash
mlflow ui       # Open UI at localhost:5000
```

### 🖥️ Streamlit Frontend

Three separate interactive pages:

* `1_Churn_Prediction.py`
* `2_Fraud_Detection.py`
* `3_CLV_Prediction.py`

Launch via:

```bash
streamlit run frontend/Home.py
```

### ⚡ FastAPI Backend

RESTful endpoints for each model under `/predict/churn`, `/predict/fraud`, etc.

```bash
uvicorn backend.main:app --reload
```

---

## 🐳 Run With Docker

```bash
# Build and run app (backend + streamlit)
docker-compose up --build
```

---

## 🧪 Tech Stack

| Layer        | Tools Used                           |
| ------------ | ------------------------------------ |
| Language     | Python 3.10                          |
| MLOps        | DVC, MLflow                          |
| Data Science | scikit-learn, pandas, numpy          |
| API          | FastAPI                              |
| UI           | Streamlit                            |
| Versioning   | Git, DVC                             |
| Deployment   | Docker, Railway (or Streamlit Cloud) |

---

## 📈 Sample Results (optional if MLflow not yet included)

You can include:

* Accuracy / ROC AUC for Fraud
* F1-score for Churn
* R² and MAPE for CLV

MLflow will show performance across model versions and parameters.

---

## ✅ Setup Instructions

1. **Clone this repo**

```bash
git clone https://github.com/yourusername/credit-card-suite.git
cd credit-card-suite
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
pip install -e .
```

3. **Run DVC pipeline**

```bash
dvc repro
```

4. **Run UI or API**

```bash
# Frontend
streamlit run frontend/Home.py

# Backend
uvicorn backend.main:app --reload
```

---

## 🏆 Why This Project for Amex?

* ✅ Focus on real-world financial ML problems
* ✅ Emphasis on interpretability and production deployment
* ✅ End-to-end engineering from pipeline to API/UX
* ✅ Designed to scale across multiple models and teams
* ✅ Demonstrates maturity in MLOps and system design


## 📄 License

MIT License

---

## 🤝 Contact

**Aadarsh Vani**
[LinkedIn](https://www.linkedin.com/in/aadarsh-vani-a60a641a0/) • [GitHub](https://github.com/aadarshvani)

