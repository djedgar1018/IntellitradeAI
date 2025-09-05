"""
XGBoost Model for Trading Signal Generation
Implements XGBoost classifier/regressor for trading decisions
"""
# models/xgboost_model.py
import numpy as np

def make_model(params=None):
    try:
        from xgboost import XGBClassifier
    except Exception as e:
        raise RuntimeError("XGBoost not available. Install `xgboost` or skip.") from e
    params = params or dict(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=-1
    )
    return XGBClassifier(**params)

class XGBoostPredictor:
    """XGBoost Predictor wrapper class"""
    
    def __init__(self, params=None):
        self.params = params or dict(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=-1
        )
        self.model = None
        self.is_trained = False
    
    def train(self, X, y):
        """Train the XGBoost model"""
        try:
            from xgboost import XGBClassifier
        except Exception as e:
            raise RuntimeError("XGBoost not available. Install `xgboost` or skip.") from e
        
        self.model = XGBClassifier(**self.params)
        self.model.fit(X, y)
        self.is_trained = True
        return self.model
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict_proba(X)
    
    def get_feature_importance(self):
        """Get feature importances"""
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before getting feature importance")
        return self.model.feature_importances_
