from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline

class HyperparameterTuner:
    """Hyperparameter tuning for models"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.best_params = None
        self.best_score = None
        self.best_estimator = None
        
    def tune_xgboost(self, X_train, y_train, preprocessor, 
                     n_jobs: int = -1, verbose: int = 1) -> Pipeline:
        """Hyperparameter tuning for XGBoost"""
        print("\n🎯 Tuning XGBoost hyperparameters...")
        
        # Create base pipeline
        xgb_pipe = Pipeline([
            ('preprocess', preprocessor),
            ('model', XGBClassifier(
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=self.random_state
            ))
        ])
        
        # Define parameter grid
        param_grid = {
            'model__learning_rate': [0.01, 0.1],
            'model__n_estimators': [100, 200],
            'model__max_depth': [3, 5],
            'model__subsample': [0.8, 1.0],
            'model__colsample_bytree': [0.8, 1.0]
        }
        
        # Grid search
        grid_search = GridSearchCV(
            xgb_pipe,
            param_grid,
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state),
            scoring='roc_auc',
            n_jobs=n_jobs,
            verbose=verbose
        )
        
        grid_search.fit(X_train, y_train)
        
        self.best_params = grid_search.best_params_
        self.best_score = grid_search.best_score_
        self.best_estimator = grid_search.best_estimator_
        
        print(f"Best parameters: {self.best_params}")
        print(f"Best ROC AUC: {self.best_score:.3f}")
        
        return self.best_estimator
    
    def tune_random_forest(self, X_train, y_train, preprocessor,
                          n_jobs: int = -1) -> Pipeline:
        """Hyperparameter tuning for Random Forest"""
        print("\n🎯 Tuning Random Forest hyperparameters...")
        
        from sklearn.ensemble import RandomForestClassifier
        
        # Create base pipeline
        rf_pipe = Pipeline([
            ('preprocess', preprocessor),
            ('model', RandomForestClassifier(
                random_state=self.random_state,
                class_weight='balanced_subsample'
            ))
        ])
        
        # Define parameter grid
        param_grid = {
            'model__n_estimators': [100, 200, 300],
            'model__max_depth': [10, 20, None],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4]
        }
        
        # Grid search
        grid_search = GridSearchCV(
            rf_pipe,
            param_grid,
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state),
            scoring='roc_auc',
            n_jobs=n_jobs,
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        self.best_params = grid_search.best_params_
        self.best_score = grid_search.best_score_
        self.best_estimator = grid_search.best_estimator_
        
        print(f"Best parameters: {self.best_params}")
        print(f"Best ROC AUC: {self.best_score:.3f}")
        
        return self.best_estimator