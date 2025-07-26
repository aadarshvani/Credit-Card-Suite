# Credit Card Suite

This project provides a full-stack solution for credit card approval and churn prediction using machine learning models, with a FastAPI backend and a Streamlit frontend.

## Project Structure
- **backend/**: FastAPI backend serving ML models
- **frontend/**: Streamlit frontend for user interaction
- **artifacts/**, **models/**: Model and transformer files

## Dockerized Setup

### 1. Build Docker Images

```
docker build -t credit-card-backend -f Dockerfile .
docker build -t credit-card-frontend -f frontend/Dockerfile .
```

### 2. Run with Docker Compose

```
docker-compose up --build
```

- Backend: http://localhost:8000
- Frontend: http://localhost:8501

### 3. Stopping the Services

```
docker-compose down
```

## Notes
- Ensure all model and transformer files are present in `models/` and `artifacts/` before building images.
- The frontend expects the backend to be available at `http://backend:8000` (Docker Compose handles networking).
- For local development, you can run backend and frontend separately using `uvicorn` and `streamlit`.

## Troubleshooting
- If you change model files, rebuild the images or restart the containers.
- Check logs with `docker-compose logs` for debugging. 