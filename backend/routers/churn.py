from fastapi import APIRouter
from pydantic import BaseModel
import numpy as np
import pickle
import os

router = APIRouter()

# Load model and transformer
MODEL_PATH = os.path.join("..", "models", "churn_model.pkl")
TRANSFORMER_PATH = os.path.join("..", "artifacts", "column_transformer.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(TRANSFORMER_PATH, "rb") as f:
    transformer = pickle.load(f)

# Request schema
class ChurnInput(BaseModel):
    CreditScore: int
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float

@router.post("/predict")
def predict_churn(input: ChurnInput):
    try:
        input_dict = input.dict()
        input_df = np.array([[input_dict[col] for col in input_dict]])

        import pandas as pd
        df_input = pd.DataFrame(input_df, columns=input_dict.keys())

        # Transform input
        X_transformed = transformer.transform(df_input)

        # Predict
        prediction = model.predict(X_transformed)[0]
        prob = model.predict_proba(X_transformed)[0][1]

        return {
            "prediction": int(prediction),
            "probability": round(float(prob), 4)
        }

    except Exception as e:
        return {"error": str(e)}
