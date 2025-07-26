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
        customer_age = st.number_input("Customer Age", min_value=18, max_value=100, value=35, step=1)
        credit_limit = st.number_input("Credit Limit", min_value=0.0, value=10000.0, step=100.0)
        total_revolving_bal = st.number_input("Total Revolving Balance", min_value=0.0, value=1000.0, step=100.0)
        avg_open_to_buy = st.number_input("Average Open to Buy", min_value=0.0, value=9000.0, step=100.0)
        total_amt_chng_q4_q1 = st.number_input("Total Amount Change Q4/Q1", min_value=0.0, value=1.5, step=0.1)
        total_trans_amt = st.number_input("Total Transaction Amount", min_value=0.0, value=2000.0, step=100.0)
        total_trans_ct = st.number_input("Total Transaction Count", min_value=0, value=50, step=1)

    with col2:
        total_ct_chng_q4_q1 = st.number_input("Total Count Change Q4/Q1", min_value=0.0, value=1.2, step=0.1)
        avg_utilization_ratio = st.number_input("Average Utilization Ratio", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
        gender = st.selectbox("Gender", ["M", "F"])
        education_level = st.selectbox("Education Level", ["Uneducated", "High School", "College", "Graduate", "Post-Graduate", "Doctorate", "Unknown"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unknown"])
        income_category = st.selectbox("Income Category", ["Less than $40K", "$40K - $60K", "$60K - $80K", "$80K - $120K", "$120K +", "Unknown"])
        card_category = st.selectbox("Card Category", ["Blue", "Silver", "Gold", "Platinum"])

    submitted = st.form_submit_button("🔍 Predict")

# Prediction logic
if submitted:
    input_data = {
        "customer_age": customer_age,
        "credit_limit": credit_limit,
        "total_revolving_bal": total_revolving_bal,
        "avg_open_to_buy": avg_open_to_buy,
        "total_amt_chng_q4_q1": total_amt_chng_q4_q1,
        "total_trans_amt": total_trans_amt,
        "total_trans_ct": total_trans_ct,
        "total_ct_chng_q4_q1": total_ct_chng_q4_q1,
        "avg_utilization_ratio": avg_utilization_ratio,
        "gender": gender,
        "education_level": education_level,
        "marital_status": marital_status,
        "income_category": income_category,
        "card_category": card_category
    }

    with st.spinner("Making prediction..."):
        try:
            response = requests.post("https://creditcardsuite-backend.onrender.com/api/v1/churn", json=input_data)
            if response.status_code == 200:
                result = response.json()
                prediction = result["prediction"]
                probability = result["probability"]
                churn_risk = result["churn_risk"]
                recommendation = result["recommendation"]
                
                if prediction == 1:
                    st.error(f"❌ Customer is likely to churn. (Risk Score: {churn_risk:.1f}%)")
                else:
                    st.success(f"✅ Customer is likely to stay. (Risk Score: {churn_risk:.1f}%)")
                
                st.info(f"Probability: {probability:.2f}")
                st.info(f"Recommendation: {recommendation}")
            else:
                st.warning(f"⚠️ Request failed with status code {response.status_code}")
                st.text(response.text)
        except Exception as e:
            st.error(f"❌ Something went wrong: {e}")
