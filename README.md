# 💳 Credit Card Suite

**AI-powered platform for Credit Card Approval & Churn Prediction**  
[Live Demo](https://creditcardsuite-frontend-latest.onrender.com) | [GitHub Repo](https://github.com/aadarshvani/Credit-Card-Suite.git)  
Author: [Aadarsh Vani](https://github.com/aadarshvani) | [LinkedIn](https://www.linkedin.com/in/aadarsh-vani-a60a641a0/)

---

## 🚀 Overview

Credit Card Suite is a full-stack, production-ready solution for:
- **Credit Card Approval Prediction**: Instantly assess applicant risk and approval likelihood.
- **Customer Churn Prediction**: Predict the probability of existing customers leaving.

Built with:
- **FastAPI** (backend, ML inference APIs)
- **Streamlit** (frontend, interactive dashboards)
- **Docker & Docker Compose** (easy deployment)
- **DVC** (data & model versioning)

---

## 🌐 Live Demo

- **Frontend**: [https://creditcardsuite-frontend-latest.onrender.com](https://creditcardsuite-frontend-latest.onrender.com)
- **Backend**: [https://creditcardsuite-backend-latest.onrender.com](https://creditcardsuite-backend-latest.onrender.com)
- **API Docs**: `/docs` on backend

---

## 🏗️ Project Structure

```
Credit-Card-Suite/
│
├── backend/         # FastAPI backend (ML APIs)
│   └── routers/     # API endpoints (approval, churn)
├── frontend/        # Streamlit frontend (UI)
│   └── pages/       # Churn & Approval prediction pages
├── src/             # ML pipeline (feature engineering, training, evaluation)
│   ├── churn/
│   └── approval/
├── artifacts/       # Model transformers, feature names
├── models/          # Trained ML models
├── data/            # Raw and processed data
├── docker-compose.yml
├── Dockerfile       # Backend Dockerfile
├── frontend/Dockerfile
└── README.md
```

---

## ✨ Features

### 1. Credit Card Approval Prediction
- Predicts approval/denial for new applicants.
- Considers demographics, income, employment, family, housing, and more.
- Returns risk score, probability, and actionable recommendation.

### 2. Customer Churn Prediction
- Predicts likelihood of existing customer churn.
- Uses transaction patterns, credit utilization, demographics, and more.
- Returns churn risk, probability, and retention recommendation.

### 3. Modern, Interactive UI
- Built with Streamlit for rapid, user-friendly interaction.
- Real-time results, clear explanations, and risk visualization.

### 4. Robust, Scalable Backend
- FastAPI for high-performance ML inference.
- Modular, production-ready API design.
- CORS enabled for frontend-backend integration.

### 5. Easy Deployment
- **Dockerized**: One-command setup with Docker Compose.
- **Cloud Ready**: Deployed on Render (see live demo above).

---

## 🖥️ Quick Start

### 1. Clone the Repo

```bash
git clone https://github.com/aadarshvani/Credit-Card-Suite.git
cd Credit-Card-Suite
```

### 2. Build & Run with Docker Compose

```bash
docker-compose up --build
```

- Backend: [http://localhost:8000](http://localhost:8000)
- Frontend: [http://localhost:8501](http://localhost:8501)

### 3. Local Development

- **Backend**:  
  ```bash
  cd backend
  pip install -r requirements.txt
  uvicorn main:app --reload
  ```
- **Frontend**:  
  ```bash
  cd frontend
  pip install streamlit requests
  streamlit run Home.py
  ```

---

## 🧩 API Endpoints

### Approval Prediction

- **POST** `/api/v1/approval`
  - Input: Applicant details (demographics, income, etc.)
  - Output: Approval prediction, risk score, probability, recommendation

### Churn Prediction

- **POST** `/api/v1/churn`
  - Input: Customer details (usage, demographics, etc.)
  - Output: Churn prediction, risk score, probability, recommendation

- **Health Checks**: `/api/v1/approval/health`, `/api/v1/churn/health`

See [API Docs](https://creditcardsuite-backend-latest.onrender.com/docs) for full schema.

---

## 🐳 Docker Compose

```yaml
version: "3.8"
services:
  backend:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    volumes:
      - ./artifacts:/app/artifacts
      - ./models:/app/models
    working_dir: /app/backend

  frontend:
    build:
      context: .
      dockerfile: frontend/Dockerfile
    ports:
      - "8501:8501"
    depends_on:
      - backend
    environment:
      - BACKEND_URL=http://backend:8000
    working_dir: /app/frontend
```

---

## 📊 Example Use Cases

- **Banks**: Automate credit card approval, reduce risk, improve customer retention.
- **Fintechs**: Integrate ML-powered credit risk and churn analytics.
- **Data Science**: End-to-end ML pipeline, from data to deployment.

---

## 🛠️ Tech Stack

- **Python 3.10**
- **FastAPI**
- **Streamlit**
- **scikit-learn, imbalanced-learn**
- **Docker, Docker Compose**
- **DVC** (Data Version Control)
- **Render** (Cloud deployment)

---

## 👤 Author

- **Aadarsh Vani**  
  [GitHub](https://github.com/aadarshvani) | [LinkedIn](https://www.linkedin.com/in/aadarsh-vani-a60a641a0/)

---

## ⭐️ Contributing

Contributions, issues, and feature requests are welcome!  
Feel free to check [issues page](https://github.com/aadarshvani/Credit-Card-Suite/issues).

---

## 📄 License

This project is licensed under the MIT License. 