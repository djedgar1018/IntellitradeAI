"""
Random Forest Model for Trading Signal Generation
Implements Random Forest classifier for trading decisions
"""
# models/random_forest_model.py
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def make_model(params=None):
    params = params or dict(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1)
    return RandomForestClassifier(**params)

class RandomForestPredictor:
    """Random Forest Predictor wrapper class"""
    
    def __init__(self, params=None):
        self.params = params or dict(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1)
        self.model = None
        self.is_trained = False
    
    def train(self, X, y):
        """Train the Random Forest model"""
        self.model = RandomForestClassifier(**self.params)
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

