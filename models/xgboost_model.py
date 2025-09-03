"""
XGBoost Model for Trading Signal Generation
Implements XGBoost classifier/regressor for trading decisions
"""
# models/xgboost_model.py
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
