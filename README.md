# 💳 Credit Card Intelligence Suite

A modular, production-ready machine learning pipeline to tackle three critical problems in the credit card ecosystem:
1. 🔁 **Customer Churn Prediction**
2. 🚨 **Fraud Detection**
3. 💰 **Customer Lifetime Value (CLV) Estimation**

> ⚙️ Built using **Python**, **DVC**, **MLflow**, **Streamlit**, **FastAPI**, and containerized with **Docker**.

## 🚀 Project Highlights

| Feature                      | Details |
|-----------------------------|---------|
| 🔄 **Modular Pipelines**     | Separate end-to-end pipeline for each model |
| 📊 **DVC Integration**       | Data & model versioning |
| 🧪 **MLflow Tracking**       | Experiment tracking and model registry |
| 🖥️ **Streamlit Frontend**    | Interactive multi-page dashboard |
| ⚡ **FastAPI Backend**       | RESTful APIs for each model |
| 🐳 **Dockerized**            | Container-based deployment |
| ☁️ **Cloud-ready**           | Deploy on AWS, GCP, or Railway |

## 🧠 Business Use Cases

- **Churn Prediction**: Identify at-risk customers for proactive retention campaigns
- **Fraud Detection**: Real-time transaction monitoring and anomaly detection
- **CLV Estimation**: Optimize customer acquisition and retention strategies

## 🧱 Project Structure

```
credit-card-suite/
├── data/                     # Raw, processed, and feature datasets
├── models/                   # Trained model artifacts
├── src/                      # Core ML modules
│   ├── churn/                # Churn prediction pipeline
│   ├── fraud/                # Fraud detection pipeline
│   ├── clv/                  # CLV estimation pipeline
│   └── utils/                # Shared utilities
├── params/                   # Model configuration files
├── backend/                  # FastAPI REST API
├── frontend/                 # Streamlit dashboard
├── docker-compose.yml        # Multi-service deployment
└── dvc.yaml                  # DVC pipeline definitions
```

## 🛠️ Technology Stack

**ML & Data**: Python, Scikit-learn, Pandas, NumPy  
**MLOps**: DVC, MLflow, Docker  
**Backend**: FastAPI, Uvicorn  
**Frontend**: Streamlit

## 🏃‍♂️ Quick Start

### Local Development
```bash
# Clone and install
git clone https://github.com/aadarshvani/Credit-Card-Suite.git
cd Credit-Card-Suite
pip install -r requirements.txt
pip install -e .

# Run DVC pipeline
dvc repro

# Start applications
streamlit run frontend/Home.py          # Frontend on :8501
uvicorn backend.main:app --reload      # Backend on :8000
mlflow ui                               # MLflow on :5000
```

### Docker Deployment
```bash
docker-compose up --build
```

## 🔗 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/churn/predict` | POST | Customer churn probability |
| `/fraud/detect` | POST | Transaction fraud score |
| `/clv/predict` | POST | Customer lifetime value |

## 🏆 Why This Matters for Financial Services

### **Technical Excellence**
- **End-to-End MLOps**: Complete pipeline from data to deployment
- **Reproducible Workflows**: DVC ensures consistent results
- **Scalable Architecture**: Modular design supports independent updates
- **Production Ready**: Docker containers with monitoring

### **Business Impact**
- **Risk Mitigation**: Automated fraud detection reduces losses
- **Customer Retention**: Proactive churn prediction improves retention
- **Revenue Optimization**: CLV modeling enhances acquisition strategies

## 📞 Contact

**Aadarsh Vani**  
🐙 GitHub: [@aadarshvani](https://github.com/aadarshvani)  
💼 LinkedIn: [linkedin.com/in/aadarshvani](https://linkedin.com/in/aadarshvani)

