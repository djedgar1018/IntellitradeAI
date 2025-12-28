"""
LightGBM Model for Trading Signal Generation
Implements LightGBM classifier for fast and accurate trading decisions
"""
import numpy as np

class SkipLightGBM(Exception):
    pass


def make_model(params=None):
    """Create a LightGBM classifier with optimized parameters"""
    try:
        from lightgbm import LGBMClassifier
    except ImportError as e:
        raise SkipLightGBM("LightGBM not installed; skipping.") from e
    
    params = params or {
        'n_estimators': 500,
        'max_depth': 8,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    
    return LGBMClassifier(**params)


class LightGBMPredictor:
    """LightGBM Predictor wrapper class with advanced features"""
    
    def __init__(self, params=None):
        self.params = params or {
            'n_estimators': 500,
            'max_depth': 8,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        self.model = None
        self.is_trained = False
        self.feature_importance_ = None
    
    def train(self, X, y, eval_set=None, early_stopping_rounds=50):
        """Train the LightGBM model with optional early stopping"""
        try:
            from lightgbm import LGBMClassifier, early_stopping, log_evaluation
        except ImportError as e:
            raise SkipLightGBM("LightGBM not installed; skipping.") from e
        
        self.model = LGBMClassifier(**self.params)
        
        if eval_set is not None:
            callbacks = [
                early_stopping(stopping_rounds=early_stopping_rounds),
                log_evaluation(period=100)
            ]
            self.model.fit(
                X, y,
                eval_set=[eval_set],
                callbacks=callbacks
            )
        else:
            self.model.fit(X, y)
        
        self.is_trained = True
        self.feature_importance_ = self.model.feature_importances_
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
        return self.feature_importance_
