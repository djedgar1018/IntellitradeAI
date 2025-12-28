"""
Stacking Ensemble Model for Improved Trading Predictions
Implements stacking with multiple base learners and a meta-learner
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import warnings
warnings.filterwarnings('ignore')


class StackingEnsemble:
    """
    Stacking Ensemble that combines multiple models with a meta-learner
    Specifically designed for financial time series prediction
    """
    
    def __init__(self, use_time_series_cv=True, n_splits=5):
        self.use_time_series_cv = use_time_series_cv
        self.n_splits = n_splits
        self.base_models = {}
        self.meta_model = None
        self.is_trained = False
        self.feature_names = None
        self.base_predictions = None
        
    def _get_base_models(self):
        """Initialize base models with optimized parameters"""
        models = {}
        
        models['rf'] = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        try:
            from xgboost import XGBClassifier
            models['xgb'] = XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                scale_pos_weight=1,
                random_state=42,
                n_jobs=-1,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        except ImportError:
            print("XGBoost not available, skipping")
        
        try:
            from lightgbm import LGBMClassifier
            models['lgbm'] = LGBMClassifier(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        except ImportError:
            print("LightGBM not available, skipping")
        
        models['gb'] = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )
        
        return models
    
    def _get_meta_model(self):
        """Get the meta-learner model"""
        return LogisticRegression(
            C=1.0,
            penalty='l2',
            solver='lbfgs',
            max_iter=1000,
            random_state=42
        )
    
    def _get_cv_splitter(self, n_samples):
        """Get appropriate cross-validation splitter"""
        if self.use_time_series_cv:
            return TimeSeriesSplit(n_splits=self.n_splits)
        else:
            from sklearn.model_selection import StratifiedKFold
            return StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
    
    def train(self, X, y, verbose=True):
        """
        Train the stacking ensemble
        
        Args:
            X: Training features
            y: Training labels
            verbose: Print training progress
        """
        if verbose:
            print("Training Stacking Ensemble...")
            print(f"  → Training samples: {len(X)}")
            print(f"  → Features: {X.shape[1] if hasattr(X, 'shape') else len(X.columns)}")
        
        X_arr = X.values if hasattr(X, 'values') else np.array(X)
        y_arr = y.values if hasattr(y, 'values') else np.array(y)
        
        self.feature_names = X.columns.tolist() if hasattr(X, 'columns') else None
        
        self.base_models = self._get_base_models()
        cv_splitter = self._get_cv_splitter(len(X_arr))
        
        meta_features = np.zeros((len(X_arr), len(self.base_models) * 2))
        
        if verbose:
            print(f"  → Training {len(self.base_models)} base models...")
        
        for i, (name, model) in enumerate(self.base_models.items()):
            if verbose:
                print(f"    - Training {name}...")
            
            try:
                class_preds = cross_val_predict(
                    model, X_arr, y_arr, 
                    cv=cv_splitter, 
                    method='predict'
                )
                proba_preds = cross_val_predict(
                    model, X_arr, y_arr, 
                    cv=cv_splitter, 
                    method='predict_proba'
                )[:, 1]
                
                meta_features[:, i*2] = class_preds
                meta_features[:, i*2 + 1] = proba_preds
                
                model.fit(X_arr, y_arr)
                
                if verbose:
                    cv_acc = accuracy_score(y_arr, class_preds)
                    print(f"      CV Accuracy: {cv_acc:.4f}")
                    
            except Exception as e:
                print(f"    Error training {name}: {str(e)}")
                meta_features[:, i*2] = 0
                meta_features[:, i*2 + 1] = 0.5
        
        if verbose:
            print("  → Training meta-learner...")
        
        self.meta_model = self._get_meta_model()
        self.meta_model.fit(meta_features, y_arr)
        
        meta_preds = self.meta_model.predict(meta_features)
        final_acc = accuracy_score(y_arr, meta_preds)
        
        if verbose:
            print(f"  ✓ Stacking Ensemble trained - Final Accuracy: {final_acc:.4f}")
        
        self.is_trained = True
        self.base_predictions = meta_features
        
        return self
    
    def predict(self, X):
        """Make predictions using the stacking ensemble"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_arr = X.values if hasattr(X, 'values') else np.array(X)
        
        meta_features = self._get_meta_features(X_arr)
        
        return self.meta_model.predict(meta_features)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_arr = X.values if hasattr(X, 'values') else np.array(X)
        
        meta_features = self._get_meta_features(X_arr)
        
        return self.meta_model.predict_proba(meta_features)
    
    def _get_meta_features(self, X):
        """Generate meta-features from base model predictions"""
        meta_features = np.zeros((len(X), len(self.base_models) * 2))
        
        for i, (name, model) in enumerate(self.base_models.items()):
            try:
                class_preds = model.predict(X)
                proba_preds = model.predict_proba(X)[:, 1]
                
                meta_features[:, i*2] = class_preds
                meta_features[:, i*2 + 1] = proba_preds
            except Exception as e:
                print(f"Error predicting with {name}: {str(e)}")
                meta_features[:, i*2] = 0
                meta_features[:, i*2 + 1] = 0.5
        
        return meta_features
    
    def get_feature_importance(self):
        """Get aggregated feature importance from base models"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        importance_dict = {}
        
        for name, model in self.base_models.items():
            if hasattr(model, 'feature_importances_'):
                for i, imp in enumerate(model.feature_importances_):
                    feat_name = self.feature_names[i] if self.feature_names else f'feature_{i}'
                    if feat_name not in importance_dict:
                        importance_dict[feat_name] = []
                    importance_dict[feat_name].append(imp)
        
        avg_importance = {k: np.mean(v) for k, v in importance_dict.items()}
        
        return dict(sorted(avg_importance.items(), key=lambda x: x[1], reverse=True))
    
    def evaluate(self, X, y):
        """Evaluate the model on test data"""
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'auc_roc': roc_auc_score(y, y_proba),
            'classification_report': classification_report(y, y_pred, output_dict=True)
        }
        
        return metrics


class ImprovedTriSignalFusion:
    """
    Enhanced Tri-Signal Fusion with adaptive weighting
    Combines ML ensemble, pattern recognition, and news sentiment
    """
    
    def __init__(self, initial_weights=None):
        self.weights = initial_weights or {
            'ml': 0.55,
            'pattern': 0.28,
            'news': 0.17
        }
        self.confidence_threshold = 0.7
        self.adaptive_weights = None
        
    def fuse_signals(self, ml_signal, ml_confidence, pattern_signal, pattern_confidence, 
                    news_signal, news_confidence):
        """
        Fuse signals from three sources with adaptive weighting
        
        Returns:
            final_signal: -1 (SELL), 0 (HOLD), 1 (BUY)
            final_confidence: 0-1 confidence score
        """
        w = self.adaptive_weights or self.weights
        
        if ml_confidence > 0.85:
            return ml_signal, ml_confidence * 0.95
        
        if pattern_confidence > 0.80 and ml_confidence < 0.6:
            return pattern_signal, pattern_confidence * 0.9
        
        weighted_signal = (
            w['ml'] * ml_signal * ml_confidence +
            w['pattern'] * pattern_signal * pattern_confidence +
            w['news'] * news_signal * news_confidence
        )
        
        total_confidence = (
            w['ml'] * ml_confidence +
            w['pattern'] * pattern_confidence +
            w['news'] * news_confidence
        )
        
        if abs(weighted_signal) < 0.3:
            return 0, 1 - abs(weighted_signal)
        
        final_signal = 1 if weighted_signal > 0.3 else (-1 if weighted_signal < -0.3 else 0)
        final_confidence = min(abs(weighted_signal) / total_confidence, 1.0) if total_confidence > 0 else 0.5
        
        return final_signal, final_confidence
    
    def update_weights(self, performance_history):
        """Update weights based on historical performance"""
        if len(performance_history) < 30:
            return
        
        ml_acc = np.mean([p['ml_correct'] for p in performance_history[-30:]])
        pattern_acc = np.mean([p['pattern_correct'] for p in performance_history[-30:]])
        news_acc = np.mean([p['news_correct'] for p in performance_history[-30:]])
        
        total_acc = ml_acc + pattern_acc + news_acc
        if total_acc > 0:
            self.adaptive_weights = {
                'ml': ml_acc / total_acc,
                'pattern': pattern_acc / total_acc,
                'news': news_acc / total_acc
            }
