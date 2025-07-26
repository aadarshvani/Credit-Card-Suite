from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import os
from typing import List, Optional

router = APIRouter()

# Load the trained model, transformer, and feature names
model_path = os.path.join("..", "models", "churn.pkl")
transformer_path = os.path.join("..", "artifacts", "churn_transformer.pkl")
feature_names_path = os.path.join("..", "artifacts", "churn_feature_names.pkl")

try:
    # Load the model, transformer, and feature names
    model = joblib.load(model_path)
    scaler = joblib.load(transformer_path)
    feature_names = joblib.load(feature_names_path)
    print("Churn model, transformer, and feature names loaded successfully")
except Exception as e:
    print(f"Error loading churn model: {e}")
    model = None
    scaler = None
    feature_names = None

class ChurnRequest(BaseModel):
    # Based on the actual churn data columns
    customer_age: int
    credit_limit: float
    total_revolving_bal: float
    avg_open_to_buy: float
    total_amt_chng_q4_q1: float
    total_trans_amt: float
    total_trans_ct: int
    total_ct_chng_q4_q1: float
    avg_utilization_ratio: float
    gender: str
    education_level: str
    marital_status: str
    income_category: str
    card_category: str

class ChurnResponse(BaseModel):
    prediction: int
    probability: float
    churn_risk: float
    recommendation: str

@router.post("/churn", response_model=ChurnResponse)
async def predict_churn(request: ChurnRequest):
    """
    Predict customer churn probability
    """
    if model is None or scaler is None or feature_names is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Convert request to DataFrame
        input_data = pd.DataFrame([request.dict()])
        
        # One-hot encode categorical variables (same as in feature engineering)
        cat_cols = input_data.select_dtypes(include=['object']).columns
        input_data_encoded = pd.get_dummies(input_data, columns=cat_cols, drop_first=True)
        
        # Create a DataFrame with all expected features, filling missing ones with 0
        aligned_data = pd.DataFrame(0, index=[0], columns=feature_names)
        
        # Fill in the features we have
        for col in input_data_encoded.columns:
            if col in feature_names:
                aligned_data[col] = input_data_encoded[col].values
        
        # Scale features
        input_scaled = scaler.transform(aligned_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]
        
        # Get probability of churn (positive class)
        probability = probabilities[1] if len(probabilities) > 1 else probabilities[0]
        
        # Calculate churn risk (0-100)
        churn_risk = probability * 100
        
        # Determine recommendation
        if prediction == 0:
            recommendation = "LOW RISK - Customer likely to stay"
        else:
            recommendation = "HIGH RISK - Customer likely to churn"
        
        return ChurnResponse(
            prediction=int(prediction),
            probability=float(probability),
            churn_risk=float(churn_risk),
            recommendation=recommendation
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@router.get("/churn/health")
async def churn_health():
    """
    Check if the churn model is loaded and ready
    """
    return {
        "model_loaded": model is not None,
        "transformer_loaded": scaler is not None,
        "feature_names_loaded": feature_names is not None,
        "status": "ready" if (model is not None and scaler is not None and feature_names is not None) else "not_ready"
    }
