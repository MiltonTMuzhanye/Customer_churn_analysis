import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    """Model training and evaluation"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model = None
        
    def prepare_data(self, data: pd.DataFrame, test_size: float = 0.2) -> tuple:
        """Prepare train/test split"""
        print("📊 Preparing data for modeling...")
        
        # Separate features and target
        X = data.drop('Churn', axis=1)
        y = data['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=self.random_state, 
            stratify=y
        )
        
        print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
        print(f"Churn rate - Train: {y_train.mean():.2%}, Test: {y_test.mean():.2%}")
        
        return X_train, X_test, y_train, y_test
    
    def define_models(self, X_train: pd.DataFrame, y_train: pd.Series, preprocessor) -> dict:
        """Define model configurations with pipelines"""
        
        # Calculate class weights for imbalance
        class_weight = (len(y_train) - sum(y_train)) / sum(y_train)
        
        from sklearn.pipeline import Pipeline
        
        models = {
            'LogisticRegression': Pipeline([
                ('preprocess', preprocessor),
                ('model', LogisticRegression(
                    max_iter=1000, 
                    class_weight='balanced',
                    random_state=self.random_state
                ))
            ]),
            'RandomForest': Pipeline([
                ('preprocess', preprocessor),
                ('model', RandomForestClassifier(
                    n_estimators=200,
                    class_weight='balanced_subsample',
                    random_state=self.random_state
                ))
            ]),
            'XGBoost': Pipeline([
                ('preprocess', preprocessor),
                ('model', XGBClassifier(
                    use_label_encoder=False,
                    eval_metric='logloss',
                    scale_pos_weight=class_weight,
                    random_state=self.random_state
                ))
            ]),
            'GradientBoosting': Pipeline([
                ('preprocess', preprocessor),
                ('model', GradientBoostingClassifier(
                    random_state=self.random_state
                ))
            ])
        }
        
        return models
    
    def train_and_evaluate(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                          y_train: pd.Series, y_test: pd.Series, preprocessor) -> dict:
        """Train and evaluate all models"""
        
        print("🚀 Training and evaluating models...")
        self.models = self.define_models(X_train, y_train, preprocessor)
        self.results = {}
        
        for name, pipeline in self.models.items():
            print(f"\n{'='*40}")
            print(f"Training {name}")
            print(f"{'='*40}")
            
            # Cross-validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            
            # Extract the model from pipeline for CV
            X_train_processed = preprocessor.fit_transform(X_train)
            
            model = pipeline.named_steps['model']
            cv_scores = cross_val_score(
                model, X_train_processed, y_train, 
                cv=cv, scoring='roc_auc'
            )
            
            # Train final model
            pipeline.fit(X_train, y_train)
            
            # Predictions
            y_pred = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test)[:, 1]
            
            # Metrics
            from sklearn.metrics import average_precision_score
            
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_proba),
                'avg_precision': average_precision_score(y_test, y_proba),
                'cv_mean': np.mean(cv_scores),
                'cv_std': np.std(cv_scores),
                'classification_report': classification_report(y_test, y_pred, output_dict=True),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
            }
            
            self.results[name] = metrics
            
            print(f"ROC AUC: {metrics['roc_auc']:.3f}")
            print(f"Accuracy: {metrics['accuracy']:.3f}")
            print(f"CV ROC AUC: {metrics['cv_mean']:.3f} ± {metrics['cv_std']:.3f}")
        
        # Select best model
        self.select_best_model()
        
        return self.results
    
    def select_best_model(self) -> str:
        """Select the best model based on ROC AUC"""
        best_model_name = max(self.results.items(), key=lambda x: x[1]['roc_auc'])[0]
        self.best_model = self.models[best_model_name]
        
        print(f"\n🏆 Best Model: {best_model_name}")
        print(f"ROC AUC: {self.results[best_model_name]['roc_auc']:.3f}")
        
        return best_model_name
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """Get results as DataFrame"""
        results_list = []
        for model_name, metrics in self.results.items():
            results_list.append({
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'ROC_AUC': metrics['roc_auc'],
                'Avg_Precision': metrics['avg_precision'],
                'CV_ROC_AUC': metrics['cv_mean'],
                'CV_Std': metrics['cv_std']
            })
        
        return pd.DataFrame(results_list).sort_values('ROC_AUC', ascending=False)