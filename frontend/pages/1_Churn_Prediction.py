import streamlit as st
import requests

st.set_page_config(page_title="Churn Prediction", page_icon="🔮", layout="wide")

st.title("🔮 Customer Churn Prediction")
st.markdown("Use the form below to predict if a customer will churn based on their details.")

# Group inputs using columns for better layout
with st.form("churn_form"):
    st.subheader("Customer Information")

    col1, col2 = st.columns(2)

    with col1:
        credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650, step=1)
        geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.slider("Age", min_value=18, max_value=100, value=35)

    with col2:
        tenure = st.selectbox("Tenure (years with bank)", [1 ,2 ,3 ,4 ,5 , 6, 7, 8, 9, 10])
        balance = st.number_input("Account Balance", min_value=0.0, value=50000.0, step=100.0)
        num_of_products = st.selectbox("Number of Products", [1, 2, 3, 4])
        estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=60000.0, step=100.0)

    has_cr_card = st.radio("Has Credit Card?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    is_active_member = st.radio("Is Active Member?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")

    submitted = st.form_submit_button("🔍 Predict")

# Prediction logic
if submitted:
    input_data = {
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_of_products,
        "HasCrCard": has_cr_card,
        "IsActiveMember": is_active_member,
        "EstimatedSalary": estimated_salary
    }

    with st.spinner("Making prediction..."):
        try:
            response = requests.post("http://127.0.0.1:8000/churn/predict", json=input_data)
            if response.status_code == 200:
                prediction = response.json()["prediction"]
                if prediction == 1:
                    st.error("❌ Customer is likely to churn.")
                else:
                    st.success("✅ Customer is likely to stay.")
            else:
                st.warning(f"⚠️ Request failed with status code {response.status_code}")
                st.text(response.text)
        except Exception as e:
            st.error(f"❌ Something went wrong: {e}")
