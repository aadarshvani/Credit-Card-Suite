from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import approval, churn
import uvicorn

app = FastAPI(
    title="Credit Card Suite API",
    description="API for Credit Card Fraud Detection and Churn Prediction",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(approval.router, prefix="/api/v1", tags=["Approval/Fraud Detection"])
app.include_router(churn.router, prefix="/api/v1", tags=["Churn Prediction"])

@app.get("/")
async def root():
    return {
        "message": "Credit Card Suite API",
        "version": "1.0.0",
        "endpoints": {
            "approval": "/api/v1/approval",
            "churn": "/api/v1/churn"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Credit Card Suite API"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
