# Customer_churn_analysis

This project predict customer churn by building a machine learning model using the Telco Customer Churn dataset. The goal is to identify customers who are likely to discontinue service, allowing the company to take proactive measures to retain them as customers

## Dataset
  Source: Kaggle (Telco Customer Churn)
  
## Key Steps

  ### 1. Data Preprocessing
    -removed customeID
    -TotalCharges converted to numeric type and removed missing values with median
    
  ### 2. Feature Engineering
    - Created TenureGroup by binning tenure into meaningful categories
    - Created NumServices: number of optional services subscribed to
    - Created ChargeRatio: MonthlyCharges / TotalCharges
    
  ### 3. Exploratory Data Analysis (EDA)
    - Plotted churn distribution, tenure, charges, service usage
    - Examined correlation among numeric features
    
  ### 4. Preprocessing Pipeline
    - Numeric features: assigned using median, then scaled
    - Categorical features: assigned using mode, then one hot encoded
    
  ### 5. Model Training & Evaluation
    Trained the following models:
      - Logistic Regression
      - Random Forest
      - XGBoost
      - Gradient Boosting


    Metrics evaluated:
      - Accuracy
      - ROC AUC
      - Average Precision
      - Confusion Matrix
      - Feature Importance
      
  ### 6. Hyperparameter Tuning
    Used GridSearchCV to tune the XGBoost model with a focus on:

      - learning_rate
      - n_estimators
      - max_depth
      - subsample
      - colsample_bytree

    Best model ROC AUC after tuning: 0.846

## Results

| Model              | Accuracy | ROC AUC | Avg Precision |
|-------------------|----------|---------|----------------|
| Gradient Boosting | 0.80     | 0.84    | 0.66           |
| LogisticRegression| 0.74     | 0.84    | 0.63           |
| XGBoost           | 0.76     | 0.82    | 0.63           |
| Random Forest     | 0.78     | 0.82    | 0.61           |

## Conclusion

The Gradient Boosting model achieved the best overall performance in identifying churned customers with the highest ROC AUC and balanced precision recall. This project demonstrates the power of tree based models and feature engineering in solving classification problems
