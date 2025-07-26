from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import os
from typing import List, Optional

router = APIRouter()

# Load the trained model, transformer, and feature names
model_path = os.path.join("..", "models", "approval.pkl")
transformer_path = os.path.join("..", "artifacts", "approval_transformer.pkl")
feature_names_path = os.path.join("..", "artifacts", "approval_feature_names.pkl")

try:
    # Load the model, transformer, and feature names
    model = joblib.load(model_path)
    scaler = joblib.load(transformer_path)
    feature_names = joblib.load(feature_names_path)
    print("Approval model, transformer, and feature names loaded successfully")
except Exception as e:
    print(f"Error loading approval model: {e}")
    model = None
    scaler = None
    feature_names = None

class ApprovalRequest(BaseModel):
    # Based on the actual approval data features
    cnt_children: int
    amt_income_total: float
    days_birth: int
    days_employed: int
    flag_mobil: int
    flag_work_phone: int
    flag_phone: int
    flag_email: int
    cnt_fam_members: int
    code_gender: str
    flag_own_car: int
    flag_own_realty: int
    name_income_type: str
    name_education_type: str
    name_family_status: str
    name_housing_type: str
    occupation_type: str

class ApprovalResponse(BaseModel):
    prediction: int
    probability: float
    risk_score: float
    recommendation: str

@router.post("/approval", response_model=ApprovalResponse)
async def predict_approval(request: ApprovalRequest):
    """
    Predict credit card approval/fraud detection
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
        
        # Get probability of positive class (fraud/denial)
        probability = probabilities[1] if len(probabilities) > 1 else probabilities[0]
        
        # Calculate risk score (0-100)
        risk_score = probability * 100
        
        # Determine recommendation
        if prediction == 0:
            recommendation = "APPROVE - Low risk application"
        else:
            recommendation = "DENY - High risk application"
        
        return ApprovalResponse(
            prediction=int(prediction),
            probability=float(probability),
            risk_score=float(risk_score),
            recommendation=recommendation
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@router.get("/approval/health")
async def approval_health():
    """
    Check if the approval model is loaded and ready
    """
    return {
        "model_loaded": model is not None,
        "transformer_loaded": scaler is not None,
        "feature_names_loaded": feature_names is not None,
        "status": "ready" if (model is not None and scaler is not None and feature_names is not None) else "not_ready"
    }