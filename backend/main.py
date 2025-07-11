from fastapi import FastAPI
from routers import churn
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Credit Card Churn Prediction API",
    description="Serve predictions from ML model via FastAPI",
    version="1.0"
)

# Allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include churn router
app.include_router(churn.router, prefix="/churn")
