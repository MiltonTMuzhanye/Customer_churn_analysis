import streamlit as st
import pandas as pd
import requests
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.customer_churn.utils.logger import default_logger as logger

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("📊 Customer Churn Prediction System")
st.markdown("---")

# API endpoint
API_URL = "http://localhost:8000/api/v1"

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Single Prediction", "Batch Prediction", "Dashboard", "Model Info"]
)

# Initialize session state
if 'predictions' not in st.session_state:
    st.session_state.predictions = []
if 'customers' not in st.session_state:
    st.session_state.customers = []

# Single Prediction Page
if page == "Single Prediction":
    st.header("🔮 Single Customer Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Customer Information")
        customer_id = st.text_input("Customer ID", value="CUST-001")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    
    with col2:
        st.subheader("Services & Billing")
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=50.0)
        total_charges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=500.0)
    
    # Predict button
    if st.button("Predict Churn", type="primary"):
        # Prepare data
        customer_data = {
            "customerID": customer_id,
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges
        }
        
        try:
            # Make API call
            response = requests.post(f"{API_URL}/predict", json=customer_data)
            
            if response.status_code == 200:
                result = response.json()
                
                # Display results
                st.markdown("### Prediction Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Prediction",
                        "Churn" if result['churn_prediction'] else "No Churn"
                    )
                
                with col2:
                    st.metric(
                        "Probability",
                        f"{result['churn_probability']:.2%}"
                    )
                
                with col3:
                    risk_color = {
                        "Low": "🟢",
                        "Medium": "🟡",
                        "High": "🔴"
                    }
                    st.metric(
                        "Risk Level",
                        f"{risk_color.get(result['risk_level'], '')} {result['risk_level']}"
                    )
                
                # Store in session state
                st.session_state.predictions.append({
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'customer_id': customer_id,
                    'prediction': result['churn_prediction'],
                    'probability': result['churn_probability'],
                    'risk_level': result['risk_level']
                })
                
                # Show probability gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=result['churn_probability'] * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Churn Probability"},
                    gauge={'axis': {'range': [None, 100]},
                           'bar': {'color': "darkblue"},
                           'steps': [
                               {'range': [0, 30], 'color': "lightgreen"},
                               {'range': [30, 60], 'color': "yellow"},
                               {'range': [60, 100], 'color': "salmon"}
                           ]}
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig)
                
            else:
                st.error(f"Error: {response.text}")
                
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

# Batch Prediction Page
elif page == "Batch Prediction":
    st.header("📊 Batch Prediction")
    
    st.markdown("""
    Upload a CSV file with customer data for batch prediction.
    The file should have the same columns as the single prediction form.
    """)
    
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.subheader("Data Preview")
            st.dataframe(df.head())
            st.write(f"Total rows: {len(df)}")
            
            if st.button("Predict All", type="primary"):
                with st.spinner("Making predictions..."):
                    # Prepare data
                    data = df.to_dict('records')
                    
                    # Make API call
                    response = requests.post(
                        f"{API_URL}/batch-predict",
                        json={"customers": data}
                    )
                    
                    if response.status_code == 200:
                        results = response.json()
                        
                        # Display results
                        st.subheader("Prediction Results")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Total Customers", results['total_customers'])
                        with col2:
                            st.metric("Churn Prediction", results['churn_count'])
                        
                        # Convert results to DataFrame
                        predictions_df = pd.DataFrame(results['predictions'])
                        st.dataframe(predictions_df)
                        
                        # Download results
                        csv = predictions_df.to_csv(index=False)
                        st.download_button(
                            label="Download Predictions",
                            data=csv,
                            file_name="predictions.csv",
                            mime="text/csv"
                        )
                        
                        # Show distribution
                        fig = px.pie(
                            names=predictions_df['churn_prediction'].map({True: 'Churn', False: 'No Churn'}),
                            title="Prediction Distribution"
                        )
                        st.plotly_chart(fig)
                        
                    else:
                        st.error(f"Error: {response.text}")
                        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# Dashboard Page
elif page == "Dashboard":
    st.header("📈 Prediction Dashboard")
    
    if st.session_state.predictions:
        # Recent predictions table
        st.subheader("Recent Predictions")
        recent_df = pd.DataFrame(st.session_state.predictions[-10:])
        st.dataframe(recent_df)
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        predictions_df = pd.DataFrame(st.session_state.predictions)
        
        with col1:
            st.metric("Total Predictions", len(predictions_df))
        with col2:
            churn_count = predictions_df['prediction'].sum()
            st.metric("Churn Predictions", churn_count)
        with col3:
            avg_prob = predictions_df['probability'].mean()
            st.metric("Average Probability", f"{avg_prob:.2%}")
        with col4:
            risk_dist = predictions_df['risk_level'].value_counts()
            most_common_risk = risk_dist.index[0] if not risk_dist.empty else "N/A"
            st.metric("Most Common Risk Level", most_common_risk)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Prediction distribution
            fig = px.pie(
                names=predictions_df['prediction'].map({True: 'Churn', False: 'No Churn'}),
                title="Prediction Distribution"
            )
            st.plotly_chart(fig)
        
        with col2:
            # Risk distribution
            fig = px.bar(
                predictions_df['risk_level'].value_counts().reset_index(),
                x='risk_level',
                y='count',
                title="Risk Level Distribution",
                labels={'risk_level': 'Risk Level', 'count': 'Count'}
            )
            st.plotly_chart(fig)
        
        # Probability distribution
        fig = px.histogram(
            predictions_df,
            x='probability',
            nbins=20,
            title="Probability Distribution"
        )
        st.plotly_chart(fig)
        
    else:
        st.info("No predictions made yet. Make some predictions to see the dashboard!")

# Model Info Page
else:
    st.header("ℹ️ Model Information")
    
    try:
        response = requests.get(f"{API_URL}/model-info")
        
        if response.status_code == 200:
            info = response.json()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Model Details")
                st.write(f"**Model Name:** {info.get('model_name', 'N/A')}")
                st.write(f"**Version:** {info.get('model_version', 'N/A')}")
                st.write(f"**Threshold:** {info.get('threshold', 0.5)}")
                st.write(f"**Training Date:** {info.get('training_date', 'N/A')}")
                st.write(f"**Last Updated:** {info.get('last_updated', 'N/A')}")
            
            with col2:
                st.subheader("Performance Metrics")
                metrics = info.get('metrics', {})
                if metrics:
                    for key, value in metrics.items():
                        st.write(f"**{key.replace('_', ' ').title()}:** {value:.4f}")
                else:
                    st.write("No metrics available")
            
            st.subheader("Features Used")
            features = info.get('features', [])
            if features:
                st.write(", ".join(features))
            else:
                st.write("No feature information available")
            
        else:
            st.error(f"Error: {response.text}")
            
    except Exception as e:
        st.error(f"Error fetching model info: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        Customer Churn Prediction System v1.0.0
    </div>
    """,
    unsafe_allow_html=True
)