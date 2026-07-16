import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelExplainer:
    """Model interpretability and explainability"""
    
    def __init__(self, model, preprocessor, feature_names: List[str]):
        """Initialize explainer with model and feature names"""
        self.model = model
        self.preprocessor = preprocessor
        self.feature_names = feature_names
        
        # Extract actual model if it's in a pipeline
        if hasattr(model, 'named_steps'):
            self.actual_model = model.named_steps['model']
        else:
            self.actual_model = model
    
    def get_feature_importance(self, top_n: int = 20) -> Optional[pd.DataFrame]:
        """Get feature importance for tree-based models"""
        try:
            if hasattr(self.actual_model, 'feature_importances_'):
                importances = self.actual_model.feature_importances_
                indices = np.argsort(importances)[::-1][:top_n]
                
                # Create DataFrame
                importance_df = pd.DataFrame({
                    'feature': [self.feature_names[i] for i in indices],
                    'importance': importances[indices]
                })
                
                return importance_df
            else:
                logger.warning("Model doesn't have feature_importances_ attribute")
                return None
                
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return None
    
    def explain_prediction(self, customer_data: pd.DataFrame, 
                          customer_id: str = None) -> Dict[str, Any]:
        """Explain prediction for a single customer"""
        try:
            if customer_id:
                customer = customer_data[customer_data['customerID'] == customer_id]
            else:
                customer = customer_data.iloc[[0]]
            
            # Make prediction
            prediction = self.model.predict(customer)
            probability = self.model.predict_proba(customer)[:, 1][0]
            
            explanation = {
                'customer_id': customer_id,
                'prediction': 'Churn' if prediction[0] == 1 else 'No Churn',
                'probability': float(probability),
                'risk_level': self._get_risk_level(probability),
                'risk_factors': [],
                'retention_actions': [],
                'feature_contributions': {}
            }
            
            # Extract customer features
            customer_row = customer.iloc[0]
            
            # Business logic for risk factors
            if 'Contract' in customer_row and customer_row['Contract'] == 'Month-to-month':
                explanation['risk_factors'].append('Month-to-month contract (higher risk)')
                explanation['retention_actions'].append('Offer annual contract discount')
                explanation['feature_contributions']['contract_type'] = 'High'
            
            if 'tenure' in customer_row and customer_row['tenure'] < 12:
                explanation['risk_factors'].append(f'New customer (tenure: {customer_row["tenure"]} months)')
                explanation['retention_actions'].append('Engage with onboarding program')
                explanation['feature_contributions']['tenure'] = 'High'
            
            if 'MonthlyCharges' in customer_row and customer_row['MonthlyCharges'] > 70:
                explanation['risk_factors'].append(f'High monthly charges (${customer_row["MonthlyCharges"]})')
                explanation['retention_actions'].append('Review service package for cost optimization')
                explanation['feature_contributions']['monthly_charges'] = 'High'
            
            if 'NumServices' in customer_row and customer_row['NumServices'] < 3:
                explanation['risk_factors'].append(f'Few subscribed services ({customer_row["NumServices"]})')
                explanation['retention_actions'].append('Offer bundled service promotion')
                explanation['feature_contributions']['num_services'] = 'Medium'
            
            if 'InternetService' in customer_row and customer_row['InternetService'] == 'Fiber optic':
                explanation['risk_factors'].append('Fiber optic service')
                explanation['retention_actions'].append('Check service satisfaction')
                explanation['feature_contributions']['internet_service'] = 'Medium'
            
            if 'PaymentMethod' in customer_row and customer_row['PaymentMethod'] == 'Electronic check':
                explanation['risk_factors'].append('Electronic check payment')
                explanation['retention_actions'].append('Promote automatic payment')
                explanation['feature_contributions']['payment_method'] = 'Medium'
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error explaining prediction: {e}")
            return {'error': str(e)}
    
    def analyze_customer_segments(self, data: pd.DataFrame, 
                                 predictions: np.ndarray,
                                 probabilities: np.ndarray) -> Dict[str, Any]:
        """Analyze customer segments based on predictions"""
        try:
            # Add predictions to data
            data = data.copy()
            data['churn_prediction'] = predictions
            data['churn_probability'] = probabilities
            
            # Create risk segments
            data['risk_segment'] = pd.cut(
                data['churn_probability'],
                bins=[0, 0.3, 0.7, 1.0],
                labels=['Low Risk', 'Medium Risk', 'High Risk']
            )
            
            segment_analysis = {}
            
            for segment in ['High Risk', 'Medium Risk', 'Low Risk']:
                segment_data = data[data['risk_segment'] == segment]
                
                if len(segment_data) > 0:
                    analysis = {
                        'count': len(segment_data),
                        'percentage': len(segment_data) / len(data),
                        'avg_probability': segment_data['churn_probability'].mean(),
                        'characteristics': {}
                    }
                    
                    # Analyze characteristics for key features
                    key_features = ['Contract', 'tenure', 'MonthlyCharges', 
                                   'InternetService', 'PaymentMethod']
                    
                    for feature in key_features:
                        if feature in segment_data.columns:
                            if segment_data[feature].dtype == 'object':
                                # For categorical features
                                top_value = segment_data[feature].value_counts().index[0]
                                top_percentage = segment_data[feature].value_counts().iloc[0] / len(segment_data)
                                analysis['characteristics'][feature] = {
                                    'most_common': top_value,
                                    'percentage': top_percentage
                                }
                            else:
                                # For numerical features
                                analysis['characteristics'][feature] = {
                                    'mean': segment_data[feature].mean(),
                                    'median': segment_data[feature].median()
                                }
                    
                    segment_analysis[segment] = analysis
            
            return segment_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing customer segments: {e}")
            return {'error': str(e)}
    
    def generate_business_recommendations(self, segment_analysis: Dict[str, Any]) -> Dict[str, List[str]]:
        """Generate business recommendations based on segment analysis"""
        recommendations = {}
        
        # Recommendations for high-risk segment
        if 'High Risk' in segment_analysis:
            recommendations['High Risk'] = [
                "Immediate retention specialist outreach",
                "Offer loyalty discount (15-20%)",
                "Personalized service review call",
                "Contract upgrade incentive with 25% discount",
                "Executive escalation for key accounts"
            ]
        
        # Recommendations for medium-risk segment
        if 'Medium Risk' in segment_analysis:
            recommendations['Medium Risk'] = [
                "Proactive check-in within 30 days",
                "Service optimization recommendations",
                "Cross-sell relevant services",
                "Customer satisfaction survey",
                "Payment method optimization"
            ]
        
        # Recommendations for low-risk segment
        if 'Low Risk' in segment_analysis:
            recommendations['Low Risk'] = [
                "Regular engagement through newsletters",
                "Upsell premium features",
                "Referral program invitation",
                "Annual contract renewal reminder",
                "Loyalty program enrollment"
            ]
        
        return recommendations
    
    def plot_segment_analysis(self, segment_analysis: Dict[str, Any]):
        """Plot segment analysis visualization"""
        try:
            # Prepare data for plotting
            segments = []
            counts = []
            percentages = []
            avg_probabilities = []
            
            for segment, analysis in segment_analysis.items():
                if isinstance(analysis, dict) and 'count' in analysis:
                    segments.append(segment)
                    counts.append(analysis['count'])
                    percentages.append(analysis['percentage'] * 100)
                    avg_probabilities.append(analysis['avg_probability'] * 100)
            
            # Create subplots
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Plot 1: Segment counts
            colors = ['#28a745', '#fd7e14', '#dc3545']  # Green, Orange, Red
            axes[0].bar(segments, counts, color=colors[:len(segments)])
            axes[0].set_title('Customer Count by Risk Segment')
            axes[0].set_ylabel('Number of Customers')
            axes[0].tick_params(axis='x', rotation=45)
            
            # Add value labels
            for i, v in enumerate(counts):
                axes[0].text(i, v, str(v), ha='center', va='bottom')
            
            # Plot 2: Segment percentages
            wedges, texts, autotexts = axes[1].pie(
                percentages, 
                labels=segments, 
                colors=colors[:len(segments)],
                autopct='%1.1f%%',
                startangle=90
            )
            axes[1].set_title('Segment Distribution')
            
            # Plot 3: Average probabilities
            bars = axes[2].bar(segments, avg_probabilities, color=colors[:len(segments)])
            axes[2].set_title('Average Churn Probability by Segment')
            axes[2].set_ylabel('Probability (%)')
            axes[2].tick_params(axis='x', rotation=45)
            
            # Add value labels
            for i, v in enumerate(avg_probabilities):
                axes[2].text(i, v, f'{v:.1f}%', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting segment analysis: {e}")
    
    def _get_risk_level(self, probability: float) -> str:
        """Determine risk level based on probability"""
        if probability > 0.7:
            return 'High'
        elif probability > 0.4:
            return 'Medium'
        else:
            return 'Low'
    
    def save_explanation_report(self, explanation: Dict[str, Any], 
                               filepath: str = 'reports/customer_explanation.json'):
        """Save explanation report to JSON"""
        import json
        import os
        
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(explanation, f, indent=2)
            
            logger.info(f"Explanation report saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving explanation report: {e}")