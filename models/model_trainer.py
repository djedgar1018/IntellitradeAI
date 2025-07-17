"""
Model Trainer Module
Handles ML model training logic for trading signals
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
from datetime import datetime
from config import config

# Import individual model classes
from .random_forest_model import RandomForestPredictor
from .xgboost_model import XGBoostPredictor

class ModelTrainer:
    """Main class for training and managing ML models"""
    
    def __init__(self):
        self.models = {}
        self.model_cache_dir = config.PATHS["model_cache"]
        self.ensure_cache_directory()
        
    def ensure_cache_directory(self):
        """Create model cache directory if it doesn't exist"""
        if not os.path.exists(self.model_cache_dir):
            os.makedirs(self.model_cache_dir)
    
    def prepare_training_data(self, data, target_column='signal', test_size=0.2):
        """
        Prepare data for model training
        
        Args:
            data: DataFrame with features and target
            target_column: Column name for target variable
            test_size: Fraction of data for testing
            
        Returns:
            X_train, X_test, y_train, y_test: Training and testing data
        """
        try:
            # Separate features and target
            feature_columns = [col for col in data.columns if col != target_column]
            X = data[feature_columns]
            y = data[target_column]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            raise Exception(f"Error preparing training data: {str(e)}")
    
    def train_random_forest(self, data, symbol, model_type='classifier'):
        """
        Train Random Forest model
        
        Args:
            data: DataFrame with features and target
            symbol: Trading symbol
            model_type: Type of model ('classifier' or 'regressor')
            
        Returns:
            model: Trained Random Forest model
            metrics: Training metrics
        """
        try:
            # Initialize model
            rf_model = RandomForestPredictor(model_type=model_type)
            
            # Train model
            metrics = rf_model.train(data)
            
            # Save model
            model_key = f"{symbol}_rf_{model_type}"
            self.models[model_key] = rf_model
            
            # Cache model to disk
            self.save_model(rf_model, model_key)
            
            return rf_model, metrics
            
        except Exception as e:
            raise Exception(f"Error training Random Forest: {str(e)}")
    
    def train_xgboost(self, data, symbol, model_type='classifier'):
        """
        Train XGBoost model
        
        Args:
            data: DataFrame with features and target
            symbol: Trading symbol
            model_type: Type of model ('classifier' or 'regressor')
            
        Returns:
            model: Trained XGBoost model
            metrics: Training metrics
        """
        try:
            # Initialize model
            xgb_model = XGBoostPredictor(model_type=model_type)
            
            # Train model
            metrics = xgb_model.train(data)
            
            # Save model
            model_key = f"{symbol}_xgb_{model_type}"
            self.models[model_key] = xgb_model
            
            # Cache model to disk
            self.save_model(xgb_model, model_key)
            
            return xgb_model, metrics
            
        except Exception as e:
            raise Exception(f"Error training XGBoost: {str(e)}")
    
    def train_all_models(self, data, symbol, model_type='classifier'):
        """
        Train all available models
        
        Args:
            data: DataFrame with features and target
            symbol: Trading symbol
            model_type: Type of model ('classifier' or 'regressor')
            
        Returns:
            results: Dictionary with training results for each model
        """
        try:
            results = {}
            
            # Train Random Forest
            try:
                rf_model, rf_metrics = self.train_random_forest(data, symbol, model_type)
                results['random_forest'] = {
                    'model': rf_model,
                    'metrics': rf_metrics,
                    'status': 'success'
                }
            except Exception as e:
                results['random_forest'] = {
                    'model': None,
                    'metrics': None,
                    'status': 'failed',
                    'error': str(e)
                }
            
            # Train XGBoost
            try:
                xgb_model, xgb_metrics = self.train_xgboost(data, symbol, model_type)
                results['xgboost'] = {
                    'model': xgb_model,
                    'metrics': xgb_metrics,
                    'status': 'success'
                }
            except Exception as e:
                results['xgboost'] = {
                    'model': None,
                    'metrics': None,
                    'status': 'failed',
                    'error': str(e)
                }
            
            return results
            
        except Exception as e:
            raise Exception(f"Error training all models: {str(e)}")
    
    def save_model(self, model, model_key):
        """
        Save model to cache directory
        
        Args:
            model: Trained model object
            model_key: Unique key for the model
        """
        try:
            cache_path = os.path.join(self.model_cache_dir, f"{model_key}.joblib")
            joblib.dump(model, cache_path)
            
        except Exception as e:
            print(f"Error saving model {model_key}: {str(e)}")
    
    def load_model(self, model_key):
        """
        Load model from cache directory
        
        Args:
            model_key: Unique key for the model
            
        Returns:
            model: Loaded model object or None if not found
        """
        try:
            cache_path = os.path.join(self.model_cache_dir, f"{model_key}.joblib")
            
            if os.path.exists(cache_path):
                return joblib.load(cache_path)
            
            return None
            
        except Exception as e:
            print(f"Error loading model {model_key}: {str(e)}")
            return None
    
    def get_model_performance(self, model_key):
        """
        Get performance metrics for a trained model
        
        Args:
            model_key: Unique key for the model
            
        Returns:
            performance: Dictionary with performance metrics
        """
        try:
            if model_key in self.models:
                model = self.models[model_key]
                
                # Get performance metrics from model
                if hasattr(model, 'get_performance_metrics'):
                    return model.get_performance_metrics()
                else:
                    return {'status': 'No performance metrics available'}
            
            return {'status': 'Model not found'}
            
        except Exception as e:
            return {'status': 'Error retrieving performance', 'error': str(e)}
    
    def compare_models(self, data, symbol, model_type='classifier'):
        """
        Compare performance of different models
        
        Args:
            data: DataFrame with features and target
            symbol: Trading symbol
            model_type: Type of model ('classifier' or 'regressor')
            
        Returns:
            comparison: Dictionary with comparison results
        """
        try:
            # Train all models
            results = self.train_all_models(data, symbol, model_type)
            
            # Create comparison
            comparison = {
                'symbol': symbol,
                'model_type': model_type,
                'timestamp': datetime.now().isoformat(),
                'models': {}
            }
            
            for model_name, result in results.items():
                if result['status'] == 'success' and result['metrics']:
                    comparison['models'][model_name] = {
                        'accuracy': result['metrics'].get('accuracy', 0),
                        'precision': result['metrics'].get('precision', 0),
                        'recall': result['metrics'].get('recall', 0),
                        'f1_score': result['metrics'].get('f1_score', 0),
                        'status': 'success'
                    }
                else:
                    comparison['models'][model_name] = {
                        'status': 'failed',
                        'error': result.get('error', 'Unknown error')
                    }
            
            return comparison
            
        except Exception as e:
            raise Exception(f"Error comparing models: {str(e)}")
    
    def get_best_model(self, comparison_results, metric='accuracy'):
        """
        Get the best performing model from comparison results
        
        Args:
            comparison_results: Results from compare_models
            metric: Metric to use for comparison ('accuracy', 'f1_score', etc.)
            
        Returns:
            best_model: Name of the best performing model
        """
        try:
            best_model = None
            best_score = -1
            
            for model_name, results in comparison_results['models'].items():
                if results['status'] == 'success' and metric in results:
                    score = results[metric]
                    if score > best_score:
                        best_score = score
                        best_model = model_name
            
            return best_model, best_score
            
        except Exception as e:
            return None, 0
    
    def cleanup_old_models(self, days_old=30):
        """
        Clean up old cached models
        
        Args:
            days_old: Number of days after which to remove models
        """
        try:
            current_time = datetime.now()
            
            for filename in os.listdir(self.model_cache_dir):
                if filename.endswith('.joblib'):
                    file_path = os.path.join(self.model_cache_dir, filename)
                    file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                    
                    if (current_time - file_time).days > days_old:
                        os.remove(file_path)
                        print(f"Removed old model: {filename}")
                        
        except Exception as e:
            print(f"Error cleaning up old models: {str(e)}")