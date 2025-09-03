"""
Random Forest Model for Trading Signal Generation
Implements Random Forest classifier for trading decisions
"""
# models/random_forest_model.py
from sklearn.ensemble import RandomForestClassifier

def make_model(params=None):
    params = params or dict(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1)
    return RandomForestClassifier(**params)

