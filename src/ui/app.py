import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Import custom modules
from api.inference import ChurnPredictor
from features.build_features import FeatureEngineer

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction Dashboard",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .risk-high {
        color: #dc3545;
        font-weight: bold;
    }
    .risk-medium {
        color: #fd7e14;
        font-weight: bold;
    }
    .risk-low {
        color: #28a745;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class ChurnDashboard:
    """Streamlit dashboard for churn prediction"""
    
    def __init__(self):
        """Initialize dashboard"""
        self.predictor = None
        self.initialize_session_state()
        
        try:
            self.predictor = ChurnPredictor()
            st.session_state['predictor_loaded'] = True
        except Exception as e:
            st.error(f"Failed to load predictor: {e}")
            st.session_state['predictor_loaded'] = False
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'predictor_loaded' not in st.session_state:
            st.session_state['predictor_loaded'] = False
        if 'predictions' not in st.session_state:
            st.session_state['predictions'] = []
        if 'customer_data' not in st.session_state:
            st.session_state['customer_data'] = None
    
    def render_header(self):
        """Render dashboard header"""
        st.markdown('<h1 class="main-header">📉 Customer Churn Prediction Dashboard</h1>', 
                   unsafe_allow_html=True)
        
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <p style="font-size: 1.2rem; color: #6c757d;">
                Predict customer churn risk and take proactive retention actions
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Status indicator
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.session_state['predictor_loaded']:
                st.success("✅ Prediction model loaded successfully")
            else:
                st.error("❌ Prediction model not loaded")
    
    def render_sidebar(self):
        """Render sidebar navigation"""
        with st.sidebar:
            st.image("https://img.icons8.com/color/96/000000/customer-insight.png", 
                    width=100)
            
            st.markdown("## Navigation")
            page = st.radio(
                "Go to",
                ["Single Prediction", "Batch Prediction", "Model Insights", 
                 "Business Analytics", "Customer Segments"]
            )
            
            st.markdown("---")
            
            st.markdown("## Model Info")
            st.info("""
            **Model**: Gradient Boosting  
            **Accuracy**: 80%  
            **ROC AUC**: 0.85  
            **Recall**: 75%
            """)
            
            st.markdown("---")
            
            st.markdown("## Quick Actions")
            if st.button("Clear All Predictions"):
                st.session_state['predictions'] = []
                st.session_state['customer_data'] = None
                st.rerun()
            
            if st.button("Export Results"):
                self.export_results()
    
    def single_prediction_page(self):
        """Single customer prediction page"""
        st.markdown('<h2 class="sub-header">Single Customer Prediction</h2>', 
                   unsafe_allow_html=True)
        
        if not st.session_state['predictor_loaded']:
            st.warning("Please load the prediction model first")
            return
        
        # Create form for customer data
        with st.form("customer_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Customer Demographics")
                gender = st.selectbox("Gender", ["Female", "Male"])
                senior_citizen = st.selectbox("Senior Citizen", [0, 1], 
                                             format_func=lambda x: "Yes" if x == 1 else "No")
                partner = st.selectbox("Partner", ["Yes", "No"])
                dependents = st.selectbox("Dependents", ["Yes", "No"])
                tenure = st.slider("Tenure (months)", 0, 72, 12)
            
            with col2:
                st.markdown("### Services")
                phone_service = st.selectbox("Phone Service", ["Yes", "No"])
                multiple_lines = st.selectbox("Multiple Lines", 
                                            ["Yes", "No", "No phone service"])
                internet_service = st.selectbox("Internet Service", 
                                               ["DSL", "Fiber optic", "No"])
                
                # Additional services in expander
                with st.expander("Additional Services"):
                    col2a, col2b = st.columns(2)
                    with col2a:
                        online_security = st.selectbox("Online Security", 
                                                      ["Yes", "No", "No internet service"])
                        online_backup = st.selectbox("Online Backup", 
                                                    ["Yes", "No", "No internet service"])
                        device_protection = st.selectbox("Device Protection", 
                                                        ["Yes", "No", "No internet service"])
                    
                    with col2b:
                        tech_support = st.selectbox("Tech Support", 
                                                   ["Yes", "No", "No internet service"])
                        streaming_tv = st.selectbox("Streaming TV", 
                                                   ["Yes", "No", "No internet service"])
                        streaming_movies = st.selectbox("Streaming Movies", 
                                                       ["Yes", "No", "No internet service"])
            
            col3, col4 = st.columns(2)
            
            with col3:
                st.markdown("### Contract & Billing")
                contract = st.selectbox("Contract", 
                                       ["Month-to-month", "One year", "Two year"])
                paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
                payment_method = st.selectbox("Payment Method", [
                    "Electronic check", "Mailed check", 
                    "Bank transfer (automatic)", "Credit card (automatic)"
                ])
            
            with col4:
                st.markdown("### Charges")
                monthly_charges = st.number_input("Monthly Charges ($)", 
                                                 0.0, 200.0, 70.0, 5.0)
                total_charges = st.number_input("Total Charges ($)", 
                                               0.0, 10000.0, 1000.0, 100.0)
            
            # Submit button
            submitted = st.form_submit_button("Predict Churn Risk", type="primary")
        
        if submitted:
            # Create customer data dictionary
            customer_data = {
                'gender': gender,
                'SeniorCitizen': senior_citizen,
                'Partner': partner,
                'Dependents': dependents,
                'tenure': tenure,
                'PhoneService': phone_service,
                'MultipleLines': multiple_lines,
                'InternetService': internet_service,
                'OnlineSecurity': online_security,
                'OnlineBackup': online_backup,
                'DeviceProtection': device_protection,
                'TechSupport': tech_support,
                'StreamingTV': streaming_tv,
                'StreamingMovies': streaming_movies,
                'Contract': contract,
                'PaperlessBilling': paperless_billing,
                'PaymentMethod': payment_method,
                'MonthlyCharges': monthly_charges,
                'TotalCharges': total_charges
            }
            
            # Store in session state
            st.session_state['customer_data'] = customer_data
            
            # Make prediction
            with st.spinner("Making prediction..."):
                try:
                    prediction, probability = self.predictor.predict_single(customer_data)
                    
                    # Get additional insights
                    risk_factors = self.predictor.get_risk_factors(customer_data)
                    suggested_actions = self.predictor.get_suggested_actions(
                        customer_data, probability
                    )
                    
                    # Store prediction
                    prediction_result = {
                        'prediction': prediction,
                        'probability': probability,
                        'risk_factors': risk_factors,
                        'suggested_actions': suggested_actions,
                        'customer_data': customer_data
                    }
                    
                    st.session_state['predictions'].append(prediction_result)
                    
                    # Display results
                    self.display_prediction_results(prediction_result)
                    
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
    
    def display_prediction_results(self, prediction_result):
        """Display prediction results"""
        st.markdown("---")
        st.markdown('<h2 class="sub-header">Prediction Results</h2>', 
                   unsafe_allow_html=True)
        
        # Create metrics columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Prediction card
            prediction = prediction_result['prediction']
            probability = prediction_result['probability']
            
            if prediction == 1:
                st.markdown("""
                <div class="metric-card">
                    <h3 style="color: #dc3545;">🚨 High Churn Risk</h3>
                    <p style="font-size: 2rem; font-weight: bold;">{:.1%}</p>
                    <p>Probability of churn</p>
                </div>
                """.format(probability), unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="metric-card">
                    <h3 style="color: #28a745;">✅ Low Churn Risk</h3>
                    <p style="font-size: 2rem; font-weight: bold;">{:.1%}</p>
                    <p>Probability of staying</p>
                </div>
                """.format(1 - probability), unsafe_allow_html=True)
        
        with col2:
            # Risk score
            if probability > 0.7:
                risk_class = "risk-high"
                risk_text = "High Risk"
            elif probability > 0.4:
                risk_class = "risk-medium"
                risk_text = "Medium Risk"
            else:
                risk_class = "risk-low"
                risk_text = "Low Risk"
            
            st.markdown(f"""
            <div class="metric-card">
                <h3>Risk Score</h3>
                <p style="font-size: 2rem; font-weight: bold;" class="{risk_class}">
                    {risk_text}
                </p>
                <p>Based on probability threshold</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            # Confidence
            confidence = probability if prediction == 1 else 1 - probability
            st.markdown("""
            <div class="metric-card">
                <h3>Model Confidence</h3>
                <p style="font-size: 2rem; font-weight: bold;">{:.1%}</p>
                <p>Prediction confidence level</p>
            </div>
            """.format(confidence), unsafe_allow_html=True)
        
        # Risk factors and recommendations
        st.markdown("---")
        
        col4, col5 = st.columns(2)
        
        with col4:
            st.markdown("### 🚨 Risk Factors")
            for factor in prediction_result['risk_factors']:
                st.markdown(f"- {factor}")
        
        with col5:
            st.markdown("### 🎯 Recommended Actions")
            for action in prediction_result['suggested_actions']:
                st.markdown(f"- {action}")
        
        # Probability gauge chart
        st.markdown("---")
        st.markdown("### Probability Visualization")
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = probability * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Churn Probability (%)"},
            delta = {'reference': 50},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "green"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    def batch_prediction_page(self):
        """Batch prediction page"""
        st.markdown('<h2 class="sub-header">Batch Prediction</h2>', 
                   unsafe_allow_html=True)
        
        if not st.session_state['predictor_loaded']:
            st.warning("Please load the prediction model first")
            return
        
        # File upload
        st.markdown("### Upload Customer Data")
        uploaded_file = st.file_uploader(
            "Choose a CSV file with customer data",
            type="csv",
            help="File should contain customer data with all required features"
        )
        
        if uploaded_file is not None:
            try:
                # Load data
                data = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(data)} customers")
                
                # Display preview
                with st.expander("Preview Data"):
                    st.dataframe(data.head())
                
                # Make predictions
                if st.button("Run Batch Prediction", type="primary"):
                    with st.spinner(f"Predicting churn for {len(data)} customers..."):
                        # Convert to list of dictionaries
                        customers_data = data.to_dict('records')
                        
                        # Make predictions
                        predictions = self.predictor.predict_batch(customers_data)
                        
                        # Get summaries
                        summary = self.predictor.get_prediction_summary(predictions)
                        
                        # Display summary
                        self.display_batch_summary(summary, data, predictions)
                        
            except Exception as e:
                st.error(f"Error processing file: {e}")
    
    def display_batch_summary(self, summary, data, predictions):
        """Display batch prediction summary"""
        st.markdown("---")
        st.markdown("### 📊 Prediction Summary")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Customers", summary['total_customers'])
        
        with col2:
            st.metric("Predicted to Churn", 
                     summary['predicted_churn_count'],
                     f"{summary['predicted_churn_rate']:.1%}")
        
        with col3:
            st.metric("Average Probability", 
                     f"{summary['average_churn_probability']:.1%}")
        
        with col4:
            high_risk = summary['risk_distribution']['high_risk']
            st.metric("High Risk Customers", high_risk)
        
        # Risk distribution chart
        st.markdown("### Risk Distribution")
        
        risk_data = summary['risk_distribution']
        fig = px.pie(
            values=[risk_data['high_risk'], risk_data['medium_risk'], risk_data['low_risk']],
            names=['High Risk', 'Medium Risk', 'Low Risk'],
            color=['High Risk', 'Medium Risk', 'Low Risk'],
            color_discrete_map={
                'High Risk': '#dc3545',
                'Medium Risk': '#fd7e14',
                'Low Risk': '#28a745'
            }
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
        
        # Add predictions to data
        data['Churn_Prediction'] = [p[0] for p in predictions]
        data['Churn_Probability'] = [p[1] for p in predictions]
        data['Risk_Level'] = ['High' if p > 0.7 else 'Medium' if p > 0.4 else 'Low' 
                             for p in data['Churn_Probability']]
        
        # Display results table
        with st.expander("View Detailed Results"):
            st.dataframe(data[['customerID', 'Churn_Prediction', 
                             'Churn_Probability', 'Risk_Level']].head(20))
        
        # Download button
        csv = data.to_csv(index=False)
        st.download_button(
            label="Download Results CSV",
            data=csv,
            file_name="churn_predictions.csv",
            mime="text/csv"
        )
    
    def model_insights_page(self):
        """Model insights page"""
        st.markdown('<h2 class="sub-header">Model Insights</h2>', 
                   unsafe_allow_html=True)
        
        # Feature importance visualization
        st.markdown("### Top Features Driving Churn")
        
        # Example feature importance (in real app, load from model)
        feature_importance = {
            'Contract_Month-to-month': 0.25,
            'tenure': 0.18,
            'InternetService_Fiber optic': 0.15,
            'PaymentMethod_Electronic check': 0.12,
            'MonthlyCharges': 0.10,
            'PaperlessBilling_Yes': 0.08,
            'NumServices': 0.07,
            'OnlineSecurity_No': 0.05
        }
        
        # Create dataframe
        importance_df = pd.DataFrame({
            'Feature': list(feature_importance.keys()),
            'Importance': list(feature_importance.values())
        }).sort_values('Importance', ascending=True)
        
        # Plot
        fig = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            color='Importance',
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            height=400,
            title="Feature Importance",
            xaxis_title="Importance",
            yaxis_title="Feature"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Business interpretation
        st.markdown("### Business Interpretation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("""
            **Top Risk Factors:**
            1. Month-to-month contracts
            2. Short tenure (<1 year)
            3. Fiber optic service
            4. Electronic check payment
            5. High monthly charges
            """)
        
        with col2:
            st.success("""
            **Retention Opportunities:**
            1. Convert to annual contracts
            2. Early engagement for new customers
            3. Service optimization for fiber users
            4. Promote automatic payments
            5. Review pricing for high-charge customers
            """)
    
    def business_analytics_page(self):
        """Business analytics page"""
        st.markdown('<h2 class="sub-header">Business Analytics</h2>', 
                   unsafe_allow_html=True)
        
        # Revenue impact analysis
        st.markdown("### Revenue Impact Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_ltv = st.number_input("Average Customer LTV ($)", 
                                     1000, 10000, 2500, 500)
        
        with col2:
            retention_cost = st.number_input("Retention Cost per Customer ($)", 
                                           10, 500, 50, 10)
        
        with col3:
            intervention_rate = st.slider("Intervention Success Rate (%)", 
                                         0, 100, 60, 5) / 100
        
        if st.session_state['predictions']:
            # Calculate business impact
            total_predictions = len(st.session_state['predictions'])
            high_risk = sum(1 for p in st.session_state['predictions'] 
                          if p['probability'] > 0.7)
            
            revenue_at_risk = high_risk * avg_ltv
            retention_cost_total = high_risk * retention_cost
            revenue_saved = revenue_at_risk * intervention_rate
            net_value = revenue_saved - retention_cost_total
            
            # Display metrics
            st.markdown("---")
            
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            
            with metric_col1:
                st.metric("High Risk Customers", high_risk)
            
            with metric_col2:
                st.metric("Revenue at Risk", f"${revenue_at_risk:,.0f}")
            
            with metric_col3:
                st.metric("Retention Cost", f"${retention_cost_total:,.0f}")
            
            with metric_col4:
                st.metric("Net Value", f"${net_value:,.0f}",
                         delta_color="inverse" if net_value < 0 else "normal")
        
        # Churn trends visualization
        st.markdown("### Churn Trends Analysis")
        
        # Example trend data
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
        churn_rates = [26.5, 25.8, 27.2, 26.0, 24.5, 23.8]
        prediction_accuracy = [78, 79, 81, 82, 83, 85]
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Scatter(x=months, y=churn_rates, name="Churn Rate (%)",
                      line=dict(color='red', width=3)),
            secondary_y=False,
        )
        
        fig.add_trace(
            go.Scatter(x=months, y=prediction_accuracy, name="Model Accuracy (%)",
                      line=dict(color='blue', width=3)),
            secondary_y=True,
        )
        
        fig.update_layout(
            title="Churn Rate vs Model Accuracy Over Time",
            xaxis_title="Month",
            height=400
        )
        
        fig.update_yaxes(title_text="Churn Rate (%)", secondary_y=False)
        fig.update_yaxes(title_text="Model Accuracy (%)", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def customer_segments_page(self):
        """Customer segmentation page"""
        st.markdown('<h2 class="sub-header">Customer Segments</h2>', 
                   unsafe_allow_html=True)
        
        # Segment analysis
        st.markdown("### Customer Segmentation by Risk Level")
        
        # Example segment data
        segments = {
            'High Risk': {
                'count': 150,
                'avg_tenure': 8,
                'avg_monthly_charge': 85,
                'churn_rate': 65,
                'recommendation': 'Immediate intervention required'
            },
            'Medium Risk': {
                'count': 300,
                'avg_tenure': 24,
                'avg_monthly_charge': 65,
                'churn_rate': 35,
                'recommendation': 'Proactive engagement needed'
            },
            'Low Risk': {
                'count': 550,
                'avg_tenure': 48,
                'avg_monthly_charge': 45,
                'churn_rate': 15,
                'recommendation': 'Focus on retention and upsell'
            }
        }
        
        # Create segment cards
        for segment, data in segments.items():
            with st.expander(f"{segment} Customers ({data['count']} customers)"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Avg Tenure", f"{data['avg_tenure']} months")
                
                with col2:
                    st.metric("Avg Monthly Charge", f"${data['avg_monthly_charge']}")
                
                with col3:
                    st.metric("Churn Rate", f"{data['churn_rate']}%")
                
                st.info(f"**Recommendation**: {data['recommendation']}")
        
        # Segment comparison chart
        st.markdown("### Segment Comparison")
        
        segment_df = pd.DataFrame({
            'Segment': list(segments.keys()),
            'Count': [segments[s]['count'] for s in segments],
            'Avg Tenure': [segments[s]['avg_tenure'] for s in segments],
            'Avg Monthly Charge': [segments[s]['avg_monthly_charge'] for s in segments],
            'Churn Rate': [segments[s]['churn_rate'] for s in segments]
        })
        
        fig = px.bar(
            segment_df,
            x='Segment',
            y=['Avg Tenure', 'Avg Monthly Charge', 'Churn Rate'],
            barmode='group',
            title="Segment Characteristics"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def export_results(self):
        """Export prediction results"""
        if st.session_state['predictions']:
            # Convert to DataFrame
            results = []
            for i, pred in enumerate(st.session_state['predictions']):
                results.append({
                    'Customer_ID': f"CUST_{i:04d}",
                    'Churn_Prediction': 'Yes' if pred['prediction'] == 1 else 'No',
                    'Churn_Probability': pred['probability'],
                    'Risk_Level': 'High' if pred['probability'] > 0.7 else 
                                 'Medium' if pred['probability'] > 0.4 else 'Low',
                    'Risk_Factors': '; '.join(pred['risk_factors']),
                    'Suggested_Actions': '; '.join(pred['suggested_actions'])
                })
            
            results_df = pd.DataFrame(results)
            
            # Create download link
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download All Predictions",
                data=csv,
                file_name="churn_predictions_export.csv",
                mime="text/csv"
            )
        else:
            st.warning("No predictions to export")
    
    def run(self):
        """Run the dashboard"""
        self.render_header()
        self.render_sidebar()
        
        # Get current page from sidebar
        with st.sidebar:
            page = st.radio(
                "Go to",
                ["Single Prediction", "Batch Prediction", "Model Insights", 
                 "Business Analytics", "Customer Segments"],
                label_visibility="collapsed"
            )
        
        # Render selected page
        if page == "Single Prediction":
            self.single_prediction_page()
        elif page == "Batch Prediction":
            self.batch_prediction_page()
        elif page == "Model Insights":
            self.model_insights_page()
        elif page == "Business Analytics":
            self.business_analytics_page()
        elif page == "Customer Segments":
            self.customer_segments_page()

# Run the dashboard
if __name__ == "__main__":
    dashboard = ChurnDashboard()
    dashboard.run()
