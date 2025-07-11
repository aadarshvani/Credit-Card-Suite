import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Churn Prediction",
    page_icon="🔁",
    layout="wide"
)

# Header
st.markdown("# 🔁 Customer Churn Prediction")
st.markdown("### Identify at-risk customers and prevent churn")

# Sidebar for controls
with st.sidebar:
    st.header("🎯 Prediction Controls")
    
    # Model selection
    model_type = st.selectbox(
        "Select Model",
        ["Random Forest", "XGBoost", "Logistic Regression", "Neural Network"],
        index=1
    )
    
    # Prediction type
    prediction_mode = st.radio(
        "Prediction Mode",
        ["Single Customer", "Batch Prediction", "Real-time Monitoring"],
        index=0
    )
    
    # Risk threshold
    risk_threshold = st.slider(
        "Risk Threshold (%)",
        min_value=0,
        max_value=100,
        value=70,
        help="Customers above this threshold are considered high-risk"
    )
    
    st.markdown("---")
    st.subheader("📊 Model Info")
    st.info(f"""
    **Current Model**: {model_type}
    **Accuracy**: 89.2%
    **Precision**: 87.5%
    **Recall**: 84.3%
    **F1-Score**: 85.8%
    """)

# Main content area
if prediction_mode == "Single Customer":
    st.subheader("🎯 Single Customer Prediction")
    
    # Input form
    with st.form("customer_prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            customer_id = st.text_input("Customer ID", value="CUST_001234")
            age = st.number_input("Age", min_value=18, max_value=100, value=35)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
        
        with col2:
            account_length = st.number_input("Account Length (months)", min_value=0, max_value=500, value=24)
            credit_limit = st.number_input("Credit Limit ($)", min_value=0, max_value=100000, value=5000)
            current_balance = st.number_input("Current Balance ($)", min_value=0, max_value=100000, value=2500)
            utilization_rate = st.slider("Credit Utilization Rate (%)", min_value=0, max_value=100, value=50)
        
        with col3:
            monthly_spend = st.number_input("Avg Monthly Spend ($)", min_value=0, max_value=10000, value=800)
            num_transactions = st.number_input("Monthly Transactions", min_value=0, max_value=1000, value=25)
            customer_service_calls = st.number_input("Service Calls (last 6 months)", min_value=0, max_value=50, value=2)
            satisfaction_score = st.slider("Satisfaction Score", min_value=1, max_value=10, value=7)
        
        submitted = st.form_submit_button("🔮 Predict Churn Risk", use_container_width=True)
    
    if submitted:
        # Simulate prediction
        risk_score = np.random.uniform(0, 100)
        risk_level = "High" if risk_score > risk_threshold else "Medium" if risk_score > 40 else "Low"
        
        # Display results
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📊 Prediction Results")
            
            # Risk gauge
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = risk_score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Churn Risk Score"},
                delta = {'reference': risk_threshold},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkred" if risk_score > risk_threshold else "orange" if risk_score > 40 else "green"},
                    'steps': [
                        {'range': [0, 40], 'color': "lightgreen"},
                        {'range': [40, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': risk_threshold
                    }
                }
            ))
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Risk factors
            st.subheader("🔍 Key Risk Factors")
            factors = {
                "High Credit Utilization": 85,
                "Recent Service Calls": 72,
                "Low Satisfaction Score": 68,
                "Decreased Monthly Spend": 45,
                "Long Account History": 30
            }
            
            for factor, score in factors.items():
                st.progress(score, text=f"{factor}: {score}%")
        
        with col2:
            st.subheader("📋 Summary")
            
            if risk_level == "High":
                st.error(f"🚨 **HIGH RISK** ({risk_score:.1f}%)")
                st.warning("**Immediate Action Required:**")
                st.write("• Contact customer within 24 hours")
                st.write("• Offer retention incentives")
                st.write("• Schedule account review")
            elif risk_level == "Medium":
                st.warning(f"⚠️ **MEDIUM RISK** ({risk_score:.1f}%)")
                st.info("**Recommended Actions:**")
                st.write("• Monitor account activity")
                st.write("• Send satisfaction survey")
                st.write("• Consider targeted offers")
            else:
                st.success(f"✅ **LOW RISK** ({risk_score:.1f}%)")
                st.info("**Maintenance Actions:**")
                st.write("• Continue regular monitoring")
                st.write("• Maintain service quality")
                st.write("• Explore upsell opportunities")
            
            # Customer insights
            st.subheader("💡 Customer Insights")
            st.metric("Lifetime Value", f"${np.random.randint(1000, 8000):,}")
            st.metric("Months to Churn", f"{np.random.randint(1, 12)} months")
            st.metric("Retention Cost", f"${np.random.randint(50, 300)}")

elif prediction_mode == "Batch Prediction":
    st.subheader("📊 Batch Prediction Analysis")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Customer Data (CSV)",
        type=['csv'],
        help="Upload a CSV file with customer data for batch prediction"
    )
    
    if uploaded_file is None:
        # Show sample data
        st.info("📁 Upload a CSV file or use the sample data below")
        
        # Generate sample batch data
        sample_data = pd.DataFrame({
            'Customer_ID': [f'CUST_{i:06d}' for i in range(1, 101)],
            'Age': np.random.randint(18, 80, 100),
            'Account_Length': np.random.randint(1, 120, 100),
            'Credit_Limit': np.random.randint(1000, 25000, 100),
            'Monthly_Spend': np.random.randint(100, 3000, 100),
            'Utilization_Rate': np.random.uniform(0, 100, 100),
            'Churn_Risk': np.random.uniform(0, 100, 100)
        })
        
        if st.button("🔮 Run Batch Prediction", use_container_width=True):
            st.subheader("📈 Batch Prediction Results")
            
            # Add risk categories
            sample_data['Risk_Level'] = sample_data['Churn_Risk'].apply(
                lambda x: 'High' if x > 70 else 'Medium' if x > 40 else 'Low'
            )
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Customers", len(sample_data))
            with col2:
                high_risk = len(sample_data[sample_data['Risk_Level'] == 'High'])
                st.metric("High Risk", high_risk, delta=f"{high_risk/len(sample_data)*100:.1f}%")
            with col3:
                medium_risk = len(sample_data[sample_data['Risk_Level'] == 'Medium'])
                st.metric("Medium Risk", medium_risk, delta=f"{medium_risk/len(sample_data)*100:.1f}%")
            with col4:
                low_risk = len(sample_data[sample_data['Risk_Level'] == 'Low'])
                st.metric("Low Risk", low_risk, delta=f"{low_risk/len(sample_data)*100:.1f}%")
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Risk distribution
                fig_dist = px.histogram(
                    sample_data, 
                    x='Risk_Level', 
                    color='Risk_Level',
                    title='Customer Risk Distribution',
                    color_discrete_map={'High': 'red', 'Medium': 'orange', 'Low': 'green'}
                )
                st.plotly_chart(fig_dist, use_container_width=True)
            
            with col2:
                # Risk vs Age
                fig_age = px.scatter(
                    sample_data, 
                    x='Age', 
                    y='Churn_Risk',
                    color='Risk_Level',
                    title='Churn Risk by Age',
                    color_discrete_map={'High': 'red', 'Medium': 'orange', 'Low': 'green'}
                )
                st.plotly_chart(fig_age, use_container_width=True)
            
            # Data table with filtering
            st.subheader("🔍 Detailed Results")
            
            # Filter controls
            col1, col2, col3 = st.columns(3)
            with col1:
                risk_filter = st.multiselect(
                    "Filter by Risk Level",
                    ['High', 'Medium', 'Low'],
                    default=['High', 'Medium', 'Low']
                )
            with col2:
                age_range = st.slider(
                    "Age Range",
                    min_value=18,
                    max_value=80,
                    value=(18, 80)
                )
            with col3:
                top_n = st.number_input(
                    "Show Top N Customers",
                    min_value=10,
                    max_value=100,
                    value=50
                )
            
            # Apply filters
            filtered_data = sample_data[
                (sample_data['Risk_Level'].isin(risk_filter)) &
                (sample_data['Age'].between(age_range[0], age_range[1]))
            ].head(top_n)
            
            st.dataframe(
                filtered_data,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Churn_Risk": st.column_config.ProgressColumn(
                        "Churn Risk %",
                        help="Churn risk percentage",
                        min_value=0,
                        max_value=100,
                    ),
                    "Credit_Limit": st.column_config.NumberColumn(
                        "Credit Limit",
                        help="Credit limit in USD",
                        min_value=0,
                        max_value=100000,
                        step=1,
                        format="$%d",
                    ),
                }
            )
            
            # Download results
            csv = filtered_data.to_csv(index=False)
            st.download_button(
                label="📥 Download Results",
                data=csv,
                file_name='churn_predictions.csv',
                mime='text/csv',
                use_container_width=True
            )

else:  # Real-time Monitoring
    st.subheader("🔄 Real-time Churn Monitoring")
    
    # Monitoring controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        auto_refresh = st.checkbox("Auto-refresh", value=True)
        refresh_interval = st.selectbox(
            "Refresh Interval",
            ["5 seconds", "10 seconds", "30 seconds", "1 minute"],
            index=2
        )
    
    with col2:
        alert_threshold = st.slider(
            "Alert Threshold (%)",
            min_value=50,
            max_value=95,
            value=80
        )
    
    with col3:
        monitoring_region = st.selectbox(
            "Monitor Region",
            ["All Regions", "North America", "Europe", "Asia Pacific"],
            index=0
        )
    
    # Real-time alerts
    st.subheader("🚨 Real-time Alerts")
    
    # Generate sample alerts
    alerts = [
        {"Customer": "CUST_001234", "Risk": 92, "Reason": "Sudden spending decrease", "Time": "2 minutes ago"},
        {"Customer": "CUST_005678", "Risk": 87, "Reason": "Multiple service calls", "Time": "5 minutes ago"},
        {"Customer": "CUST_009012", "Risk": 83, "Reason": "Low satisfaction score", "Time": "8 minutes ago"}
    ]
    
    for alert in alerts:
        if alert["Risk"] > alert_threshold:
            st.error(f"🚨 **HIGH RISK ALERT** - {alert['Customer']} ({alert['Risk']}%) - {alert['Reason']} - {alert['Time']}")
    
    # Real-time metrics
    st.subheader("📊 Live Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Active Monitoring", "1,234", "↗️ +23")
    with col2:
        st.metric("High Risk Alerts", "12", "↗️ +3")
    with col3:
        st.metric("Avg Risk Score", "23.4%", "↘️ -1.2%")
    with col4:
        st.metric("Prevented Churn", "145", "↗️ +8")
    
    # Live chart
    st.subheader("📈 Live Risk Trend")
    
    # Generate sample time series data
    times = pd.date_range(start=datetime.now() - timedelta(hours=2), end=datetime.now(), freq='5min')
    risk_data = pd.DataFrame({
        'time': times,
        'avg_risk': np.random.normal(25, 5, len(times)),
        'high_risk_count': np.random.poisson(3, len(times))
    })
    
    fig_live = px.line(
        risk_data, 
        x='time', 
        y='avg_risk',
        title='Average Risk Score - Last 2 Hours',
        labels={'avg_risk': 'Average Risk Score (%)', 'time': 'Time'}
    )
    fig_live.add_hline(y=alert_threshold, line_dash="dash", line_color="red", annotation_text="Alert Threshold")
    fig_live.update_layout(height=400)
    st.plotly_chart(fig_live, use_container_width=True)
    
    # Action center
    st.subheader("🎯 Action Center")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Quick Actions:**")
        if st.button("📧 Send Retention Campaign", use_container_width=True):
            st.success("✅ Retention campaign sent to 12 high-risk customers")
        if st.button("📞 Schedule Callback", use_container_width=True):
            st.success("✅ Callbacks scheduled for top 5 risk customers")
        if st.button("🎁 Generate Offers", use_container_width=True):
            st.success("✅ Personalized offers generated for medium-risk customers")
    
    with col2:
        st.markdown("**System Status:**")
        st.success("✅ Model API: Online")
        st.success("✅ Data Pipeline: Running")
        st.success("✅ Alert System: Active")
        st.info("ℹ️ Last model update: 2 hours ago")

# Footer
st.markdown("---")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Model: {model_type} | Threshold: {risk_threshold}%")