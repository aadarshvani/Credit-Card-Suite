import streamlit as st
import requests

st.title("💳 Credit Card Churn Prediction")
st.write("Fill the details to predict if the customer will churn.")

# User input fields
user_input = {
    "CreditScore": st.number_input("Credit Score", 300, 900, 600),
    "Geography": st.selectbox("Geography", ["France", "Germany", "Spain"]),
    "Gender": st.selectbox("Gender", ["Male", "Female"]),
    "Age": st.slider("Age", 18, 100, 35),
    "Tenure": st.slider("Tenure", 0, 10, 3),
    "Balance": st.number_input("Balance", 0.0, 250000.0, 60000.0),
    "NumOfProducts": st.slider("Number of Products", 1, 4, 2),
    "HasCrCard": st.selectbox("Has Credit Card", [0, 1]),
    "IsActiveMember": st.selectbox("Is Active Member", [0, 1]),
    "EstimatedSalary": st.number_input("Estimated Salary", 1000.0, 200000.0, 50000.0)
}

if st.button("Predict Churn"):
    with st.spinner("Predicting..."):
        response = requests.post("http://127.0.0.1:8000/churn/predict", json=user_input)

        if response.status_code == 200:
            result = response.json()
            churn = result["prediction"]
            prob = result["probability"]

            if churn == 1:
                st.error(f"⚠️ High chance of churn! (Probability: {prob})")
            else:
                st.success(f"✅ Customer likely to stay. (Probability: {prob})")
        else:
            st.error("❌ Failed to get prediction from backend.")
