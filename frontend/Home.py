import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Credit Card Intelligence Suite",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
# st.markdown("""
# <style>
#     .main-header {
#         font-size: 3rem;
#         font-weight: bold;
#         text-align: center;
#         margin-bottom: 2rem;
#     }
#     .metric-container {
#         background-color: #f0f2f6;
#         padding: 1rem;
#         border-radius: 10px;
#         margin: 0.5rem 0;
#     }
# </style>
# """, unsafe_allow_html=True)

# Main header
st.markdown("# 💳 Credit Card Intelligence Suite")
st.markdown("### Real-time Analytics Dashboard for Financial Services")

# Sidebar for navigation and filters
with st.sidebar:
    st.header("🔧 Dashboard Controls")
    
    # Date range selector
    date_range = st.date_input(
        "Select Date Range",
        value=(datetime.now() - timedelta(days=30), datetime.now()),
        key="date_range"
    )
    
    # Refresh button
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.rerun()
    
    # Model status
    st.subheader("📊 Model Status")
    st.success("✅ Churn Model: Active")
    st.success("✅ Fraud Model: Active") 
    st.success("✅ CLV Model: Active")
    
    # Quick stats
    st.subheader("📈 Quick Stats")
    st.metric("Total Customers", "1,234,567", "↗️ +2.3%")
    st.metric("Active Cards", "987,654", "↗️ +1.8%")
    st.metric("Monthly Revenue", "$12.5M", "↗️ +5.2%")

# Main dashboard content
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="🔁 Churn Risk",
        value="8.2%",
        delta="-0.5%",
        delta_color="inverse"
    )

with col2:
    st.metric(
        label="🚨 Fraud Alerts",
        value="127",
        delta="+12",
        delta_color="inverse"
    )

with col3:
    st.metric(
        label="💰 Avg CLV",
        value="$2,456",
        delta="+$123",
        delta_color="normal"
    )

with col4:
    st.metric(
        label="⭐ Satisfaction",
        value="94.2%",
        delta="+1.2%",
        delta_color="normal"
    )

# Interactive charts section
st.markdown("---")
st.subheader("📊 Real-time Analytics")

# Create sample data for demonstrations
@st.cache_data
def generate_sample_data():
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    
    # Churn data
    churn_data = pd.DataFrame({
        'date': dates,
        'churn_rate': np.random.normal(8.2, 1.5, len(dates)),
        'predictions': np.random.randint(50, 200, len(dates))
    })
    
    # Fraud data
    fraud_data = pd.DataFrame({
        'date': dates,
        'fraud_cases': np.random.poisson(15, len(dates)),
        'amount_blocked': np.random.exponential(5000, len(dates))
    })
    
    # CLV data
    clv_data = pd.DataFrame({
        'segment': ['Premium', 'Gold', 'Silver', 'Basic'],
        'avg_clv': [4500, 2800, 1200, 650],
        'customer_count': [15000, 45000, 78000, 125000]
    })
    
    return churn_data, fraud_data, clv_data

churn_data, fraud_data, clv_data = generate_sample_data()

# Charts in tabs
tab1, tab2, tab3 = st.tabs(["📈 Trends", "🎯 Predictions", "💡 Insights"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Churn Rate Trend")
        fig_churn = px.line(
            churn_data.tail(30), 
            x='date', 
            y='churn_rate',
            title='Daily Churn Rate (%)'
        )
        fig_churn.update_layout(height=400)
        st.plotly_chart(fig_churn, use_container_width=True)
    
    with col2:
        st.subheader("Fraud Detection")
        fig_fraud = px.bar(
            fraud_data.tail(30), 
            x='date', 
            y='fraud_cases',
            title='Daily Fraud Cases Detected'
        )
        fig_fraud.update_layout(height=400)
        st.plotly_chart(fig_fraud, use_container_width=True)

with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("CLV by Segment")
        fig_clv = px.bar(
            clv_data, 
            x='segment', 
            y='avg_clv',
            title='Average Customer Lifetime Value by Segment'
        )
        fig_clv.update_layout(height=400)
        st.plotly_chart(fig_clv, use_container_width=True)
    
    with col2:
        st.subheader("Customer Distribution")
        fig_dist = px.pie(
            clv_data, 
            values='customer_count', 
            names='segment',
            title='Customer Distribution by Segment'
        )
        fig_dist.update_layout(height=400)
        st.plotly_chart(fig_dist, use_container_width=True)

with tab3:
    st.subheader("🎯 Key Insights & Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **🔁 Churn Insights:**
        - Churn rate decreased by 0.5% this month
        - Premium customers show lowest churn (2.1%)
        - Peak churn occurs on Mondays
        """)
        
        st.warning("""
        **🚨 Fraud Alerts:**
        - 127 suspicious transactions detected
        - $2.1M in potential losses prevented
        - International transactions need attention
        """)
    
    with col2:
        st.success("""
        **💰 CLV Opportunities:**
        - Premium segment shows highest growth
        - Cross-selling potential in Gold segment
        - Basic customers ready for upgrades
        """)
        
        st.error("""
        **⚠️ Action Required:**
        - Review high-risk churn predictions
        - Investigate fraud pattern in region X
        - Update CLV model with new features
        """)

# Interactive data exploration
st.markdown("---")
st.subheader("🔍 Interactive Data Explorer")

# Data selector
data_type = st.selectbox(
    "Select Data Type",
    ["Churn Predictions", "Fraud Cases", "CLV Estimates"],
    key="data_explorer"
)

# Filter controls
col1, col2, col3 = st.columns(3)

with col1:
    if data_type == "Churn Predictions":
        risk_level = st.select_slider(
            "Risk Level",
            options=["Low", "Medium", "High", "Critical"],
            value="Medium"
        )
    elif data_type == "Fraud Cases":
        amount_range = st.slider(
            "Transaction Amount Range",
            min_value=0,
            max_value=10000,
            value=(0, 5000),
            step=100
        )
    else:
        clv_range = st.slider(
            "CLV Range",
            min_value=0,
            max_value=10000,
            value=(1000, 5000),
            step=100
        )

with col2:
    region = st.multiselect(
        "Select Regions",
        ["North America", "Europe", "Asia Pacific", "Latin America"],
        default=["North America", "Europe"]
    )

with col3:
    customer_segment = st.radio(
        "Customer Segment",
        ["All", "Premium", "Gold", "Silver", "Basic"],
        horizontal=True
    )

# Display filtered data
if st.button("🔍 Apply Filters", use_container_width=True):
    # Generate sample filtered data
    sample_data = pd.DataFrame({
        'Customer ID': [f'CUST_{i:06d}' for i in range(1, 101)],
        'Segment': np.random.choice(['Premium', 'Gold', 'Silver', 'Basic'], 100),
        'Region': np.random.choice(region if region else ['North America'], 100),
        'Risk Score': np.random.uniform(0, 100, 100),
        'CLV': np.random.uniform(500, 8000, 100),
        'Last Transaction': pd.date_range(start='2024-01-01', periods=100, freq='D')
    })
    
    st.dataframe(
        sample_data,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Risk Score": st.column_config.ProgressColumn(
                "Risk Score",
                help="Risk score from 0-100",
                min_value=0,
                max_value=100,
            ),
            "CLV": st.column_config.NumberColumn(
                "CLV",
                help="Customer Lifetime Value in USD",
                min_value=0,
                max_value=10000,
                step=1,
                format="$%.0f",
            ),
        }
    )

# Footer
st.markdown("---")
st.markdown("### 🚀 Model Performance Dashboard")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🔁 Churn Model")
    st.progress(89, text="Accuracy: 89%")
    st.progress(92, text="Precision: 92%")
    st.progress(87, text="Recall: 87%")

with col2:
    st.subheader("🚨 Fraud Model")
    st.progress(96, text="Accuracy: 96%")
    st.progress(94, text="Precision: 94%")
    st.progress(93, text="Recall: 93%")

with col3:
    st.subheader("💰 CLV Model")
    st.progress(84, text="R² Score: 84%")
    st.progress(78, text="MAE: 78%")
    st.progress(82, text="MAPE: 82%")

# Real-time updates simulation
with st.container():
    st.subheader("🔄 Real-time Updates")
    
    # Auto-refresh toggle
    auto_refresh = st.checkbox("Enable Auto-refresh (every 30 seconds)")
    
    if auto_refresh:
        st.info("🔄 Auto-refresh enabled. Dashboard will update every 30 seconds.")
        # In a real app, you would use st.rerun() with a timer
        
    # Last updated timestamp
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")