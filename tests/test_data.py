import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.load_data import DataLoader
from data.clean_data import DataCleaner
from data.validate_data import DataValidator

class TestDataLoader:
    """Test DataLoader class"""
    
    def test_load_raw_data(self):
        """Test loading raw data"""
        loader = DataLoader()
        
        # Create test data
        test_data = pd.DataFrame({
            'customerID': ['1', '2'],
            'gender': ['Male', 'Female'],
            'SeniorCitizen': [0, 1],
            'Churn': ['No', 'Yes']
        })
        
        # Save test data
        test_file = 'test_data.csv'
        test_data.to_csv(test_file, index=False)
        
        # Load data
        loaded_data = loader.load_raw_data(test_file)
        
        # Assertions
        assert isinstance(loaded_data, pd.DataFrame)
        assert len(loaded_data) == 2
        assert 'customerID' in loaded_data.columns
        assert 'Churn' in loaded_data.columns
        
        # Clean up
        os.remove(test_file)
    
    def test_validate_data_schema(self):
        """Test data schema validation"""
        loader = DataLoader()
        
        # Create test data
        test_data = pd.DataFrame({
            'customerID': ['1', '2', '3'],
            'gender': ['Male', 'Female', 'Male'],
            'SeniorCitizen': [0, 1, 0],
            'MonthlyCharges': [29.85, 56.95, 53.85],
            'TotalCharges': ['29.85', '1889.5', '108.15'],
            'Churn': ['No', 'Yes', 'No']
        })
        
        # Validate schema
        validation = loader.validate_data_schema(test_data)
        
        # Assertions
        assert validation['row_count'] == 3
        assert validation['column_count'] == 6
        assert 'MonthlyCharges' in validation['basic_stats']
        assert isinstance(validation['missing_values'], dict)

class TestDataCleaner:
    """Test DataCleaner class"""
    
    def test_clean_data(self):
        """Test data cleaning"""
        cleaner = DataCleaner()
        
        # Create test data with issues
        test_data = pd.DataFrame({
            'customerID': ['1', '2', '3'],
            'gender': ['Male', 'Female', 'Male'],
            'SeniorCitizen': [0, 1, 0],
            'TotalCharges': ['29.85', '', '108.15'],  # Missing value
            'MultipleLines': ['Yes', 'No', 'No internet service'],
            'Churn': ['No', 'Yes', 'No']
        })
        
        # Clean data
        cleaned_data = cleaner.clean_data(test_data)
        
        # Assertions
        assert 'customerID' not in cleaned_data.columns
        assert pd.api.types.is_numeric_dtype(cleaned_data['TotalCharges'])
        assert cleaned_data['MultipleLines'].iloc[2] == 'No'  # Standardized
        assert cleaned_data['TotalCharges'].isnull().sum() == 0
    
    def test_clean_data_no_customerid(self):
        """Test cleaning data without customerID"""
        cleaner = DataCleaner()
        
        # Create test data without customerID
        test_data = pd.DataFrame({
            'gender': ['Male', 'Female'],
            'TotalCharges': ['29.85', '56.95'],
            'Churn': ['No', 'Yes']
        })
        
        # Should not raise error
        cleaned_data = cleaner.clean_data(test_data)
        assert isinstance(cleaned_data, pd.DataFrame)

class TestDataValidator:
    """Test DataValidator class"""
    
    def test_check_data_quality(self):
        """Test data quality checks"""
        validator = DataValidator()
        
        # Create test data with various issues
        test_data = pd.DataFrame({
            'gender': ['Male', 'Female', 'Male', np.nan],
            'SeniorCitizen': [0, 1, 0, 0],
            'MonthlyCharges': [29.85, 56.95, 53.85, 200.0],  # Potential outlier
            'TotalCharges': [29.85, 1889.5, 108.15, 10000.0],
            'Churn': ['No', 'Yes', 'No', 'No']
        })
        
        # Check data quality
        quality_checks = validator.check_data_quality(test_data)
        
        # Assertions
        assert isinstance(quality_checks, dict)
        assert 'missing_values' in quality_checks
        assert 'duplicates' in quality_checks
        assert 'outliers' in quality_checks
        assert quality_checks['duplicates'] == 0
    
    def test_validate_target_variable(self):
        """Test target variable validation"""
        validator = DataValidator()
        
        # Create test data
        test_data = pd.DataFrame({
            'gender': ['Male', 'Female', 'Male', 'Female'],
            'Churn': ['No', 'Yes', 'No', 'Yes']
        })
        
        # Validate target variable
        target_stats = validator.validate_target_variable(test_data, 'Churn')
        
        # Assertions
        assert isinstance(target_stats, dict)
        assert 'distribution' in target_stats
        assert 'percentage' in target_stats
        assert target_stats['imbalance_ratio'] == 1.0  # Balanced data
    
    def test_validate_target_variable_missing(self):
        """Test target variable validation with missing column"""
        validator = DataValidator()
        
        # Create test data without target column
        test_data = pd.DataFrame({
            'gender': ['Male', 'Female']
        })
        
        # Should raise ValueError
        with pytest.raises(ValueError):
            validator.validate_target_variable(test_data, 'Churn')

if __name__ == '__main__':
    pytest.main([__file__, '-v'])