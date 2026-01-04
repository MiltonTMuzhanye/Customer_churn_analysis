import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from features.build_features import FeatureEngineer
from features.feature_pipeline import FeaturePipeline

class TestFeatureEngineer:
    """Test FeatureEngineer class"""
    
    def test_create_features(self):
        """Test feature creation"""
        engineer = FeatureEngineer()
        
        # Create test data
        test_data = pd.DataFrame({
            'gender': ['Male', 'Female', 'Male'],
            'tenure': [1, 34, 2],
            'MonthlyCharges': [29.85, 56.95, 53.85],
            'TotalCharges': [29.85, 1889.5, 108.15],
            'OnlineSecurity': ['No', 'Yes', 'No'],
            'OnlineBackup': ['No', 'Yes', 'Yes'],
            'Contract': ['Month-to-month', 'One year', 'Month-to-month'],
            'Churn': ['No', 'Yes', 'No']
        })
        
        # Create features
        featured_data = engineer.create_features(test_data)
        
        # Assertions
        assert 'TenureGroup' in featured_data.columns
        assert 'NumServices' in featured_data.columns
        assert 'ChargeRatio' in featured_data.columns
        assert 'HighValue' in featured_data.columns
        assert 'ContractMonths' in featured_data.columns
        assert 'MonthlyValue' in featured_data.columns
        
        # Check specific feature calculations
        assert featured_data['NumServices'].iloc[0] == 0  # No services
        assert featured_data['NumServices'].iloc[1] == 2  # Two services
        assert featured_data['ContractMonths'].iloc[0] == 1  # Month-to-month
        assert featured_data['ContractMonths'].iloc[1] == 12  # One year
    
    def test_create_features_missing_columns(self):
        """Test feature creation with missing columns"""
        engineer = FeatureEngineer()
        
        # Create minimal test data
        test_data = pd.DataFrame({
            'tenure': [1, 34, 2],
            'MonthlyCharges': [29.85, 56.95, 53.85],
            'TotalCharges': [29.85, 1889.5, 108.15],
            'Contract': ['Month-to-month', 'One year', 'Month-to-month']
        })
        
        # Should still work without service columns
        featured_data = engineer.create_features(test_data)
        
        # Assertions
        assert 'TenureGroup' in featured_data.columns
        assert 'NumServices' in featured_data.columns
        assert 'ChargeRatio' in featured_data.columns
        assert featured_data['NumServices'].iloc[0] == 0  # Default to 0
    
    def test_get_feature_description(self):
        """Test feature description generation"""
        engineer = FeatureEngineer()
        
        # Get feature descriptions
        descriptions = engineer.get_feature_description()
        
        # Assertions
        assert isinstance(descriptions, dict)
        assert 'TenureGroup' in descriptions
        assert 'NumServices' in descriptions
        assert 'ChargeRatio' in descriptions
        assert 'HighValue' in descriptions
        assert len(descriptions) >= 5

class TestFeaturePipeline:
    """Test FeaturePipeline class"""
    
    def test_fit_transform(self):
        """Test fitting and transforming data"""
        pipeline = FeaturePipeline()
        
        # Create test data with mixed types
        test_data = pd.DataFrame({
            'tenure': [1, 34, 2, 45, 2],
            'MonthlyCharges': [29.85, 56.95, 53.85, 42.30, 70.70],
            'gender': ['Female', 'Male', 'Male', 'Male', 'Female'],
            'Contract': ['Month-to-month', 'One year', 'Month-to-month', 
                        'One year', 'Month-to-month'],
            'NumServices': [0, 2, 1, 3, 0],  # Engineered feature
            'Churn': ['No', 'No', 'Yes', 'No', 'Yes']
        })
        
        # Fit and transform
        X = test_data.drop('Churn', axis=1)
        X_processed = pipeline.fit_transform(X)
        
        # Assertions
        assert isinstance(X_processed, np.ndarray)
        assert X_processed.shape[0] == len(test_data)
        assert X_processed.shape[1] > len(test_data.columns) - 1  # More features after encoding
    
    def test_get_feature_names(self):
        """Test getting feature names"""
        pipeline = FeaturePipeline()
        
        # Create test data
        test_data = pd.DataFrame({
            'tenure': [1, 34, 2],
            'MonthlyCharges': [29.85, 56.95, 53.85],
            'gender': ['Female', 'Male', 'Male'],
            'Contract': ['Month-to-month', 'One year', 'Month-to-month']
        })
        
        # Fit pipeline
        pipeline.fit(test_data)
        feature_names = pipeline.get_feature_names()
        
        # Assertions
        assert isinstance(feature_names, list)
        assert len(feature_names) > 0
        assert 'tenure' in feature_names
        assert 'MonthlyCharges' in feature_names
        # Categorical features should be one-hot encoded
        assert any('gender' in name for name in feature_names)
        assert any('Contract' in name for name in feature_names)
    
    def test_transform_without_fit(self):
        """Test transforming without fitting first"""
        pipeline = FeaturePipeline()
        
        # Create test data
        test_data = pd.DataFrame({
            'tenure': [1, 34, 2],
            'gender': ['Female', 'Male', 'Male']
        })
        
        # Should raise ValueError
        with pytest.raises(ValueError):
            pipeline.transform(test_data)
    
    def test_handle_missing_categorical(self):
        """Test handling missing categorical values"""
        pipeline = FeaturePipeline()
        
        # Create test data with missing categorical values
        test_data = pd.DataFrame({
            'tenure': [1, 34, 2, 45],
            'gender': ['Female', 'Male', None, 'Female'],
            'Contract': ['Month-to-month', None, 'Month-to-month', 'One year']
        })
        
        # Fit and transform
        X_processed = pipeline.fit_transform(test_data)
        
        # Assertions
        assert isinstance(X_processed, np.ndarray)
        assert not np.any(np.isnan(X_processed))  # No NaN values
    
    def test_handle_missing_numerical(self):
        """Test handling missing numerical values"""
        pipeline = FeaturePipeline()
        
        # Create test data with missing numerical values
        test_data = pd.DataFrame({
            'tenure': [1, None, 2, 45],
            'MonthlyCharges': [29.85, 56.95, None, 42.30],
            'gender': ['Female', 'Male', 'Male', 'Female']
        })
        
        # Fit and transform
        X_processed = pipeline.fit_transform(test_data)
        
        # Assertions
        assert isinstance(X_processed, np.ndarray)
        assert not np.any(np.isnan(X_processed))  # No NaN values

if __name__ == '__main__':
    pytest.main([__file__, '-v'])