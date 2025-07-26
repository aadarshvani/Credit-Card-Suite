import streamlit as st

st.set_page_config(page_title="Credit Card Suite", page_icon="💳", layout="wide")

st.title("💳 Credit Card Suite")
st.markdown("Welcome to the Credit Card Suite - Your AI-powered credit card analysis platform.")

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔮 Churn Prediction")
    st.write("""
    Predict customer churn probability using our trained machine learning model.
    
    **Features:**
    - Customer age and demographics
    - Credit limit and utilization
    - Transaction patterns
    - Risk assessment with probability scores
    """)
    st.info("Use the sidebar to navigate to Churn Prediction")

with col2:
    st.subheader("✅ Approval Prediction")
    st.write("""
    Predict credit card application approval using our trained machine learning model.
    
    **Features:**
    - Applicant demographics
    - Income and employment details
    - Housing and family information
    - Risk assessment with approval recommendations
    """)
    st.info("Use the sidebar to navigate to Approval Prediction")

# System info
st.markdown("---")
st.subheader("📊 System Information")
col3, col4, col5 = st.columns(3)

with col3:
    st.metric("Models Available", "2")
    
with col4:
    st.metric("API Status", "✅ Active")
    
with col5:
    st.metric("Frontend Status", "✅ Active")

# Quick start guide
st.markdown("---")
st.subheader("🚀 Quick Start")
st.write("""
1. **For Churn Prediction**: Navigate to "Churn Prediction" in the sidebar
2. **For Approval Prediction**: Navigate to "Approval Prediction" in the sidebar
3. **API Access**: Backend is running on http://127.0.0.1:8000
4. **Documentation**: Visit http://127.0.0.1:8000/docs for API docs
""")

st.success("🎉 System is ready! Use the sidebar to access the prediction tools.")
