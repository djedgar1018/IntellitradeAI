"""
Robust End-to-End Model Trainer Module
Comprehensive ML model training with feature engineering and multiple algorithms
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from config import config
from utils.indicators import TechnicalIndicators

# Import individual model classes
from .random_forest_model import RandomForestPredictor
from .xgboost_model import XGBoostPredictor

class RobustModelTrainer:
    """
    Robust End-to-End Model Trainer
    Handles comprehensive feature engineering, multiple algorithms, and model evaluation
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.label_encoders = {}
        self.model_cache_dir = config.PATHS["model_cache"]
        self.feature_cache_dir = os.path.join(self.model_cache_dir, "features")
        self.ensure_cache_directories()
        self.trained_models = {}
        self.model_performance = {}
        
        # Initialize technical indicators
        self.technical_indicators = TechnicalIndicators()
        
        # Algorithm configurations
        self.algorithm_configs = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'xgboost': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 10],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }
        }
        
    def ensure_cache_directories(self):
        """Create cache directories if they don't exist"""
        for directory in [self.model_cache_dir, self.feature_cache_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
    
    def _detect_asset_class(self, symbol: str) -> str:
        """
        Detect asset class from symbol name.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Asset class: 'crypto', 'stocks', 'forex', or 'options'
        """
        symbol_upper = symbol.upper()
        
        crypto_symbols = ['BTC', 'ETH', 'SOL', 'XRP', 'ADA', 'DOGE', 'DOT', 'AVAX', 
                         'MATIC', 'LINK', 'UNI', 'AAVE', 'ATOM', 'FIL', 'LTC', 'BCH']
        if any(cs in symbol_upper for cs in crypto_symbols):
            return 'crypto'
        
        forex_pairs = ['USD', 'EUR', 'GBP', 'JPY', 'CHF', 'AUD', 'CAD', 'NZD']
        if len(symbol) == 6 and all(symbol[i:i+3].upper() in forex_pairs for i in [0, 3]):
            return 'forex'
        
        if 'CALL' in symbol_upper or 'PUT' in symbol_upper or len(symbol) > 10:
            return 'options'
        
        return 'stocks'
    
    def engineer_features(self, data, symbol):
        """
        Comprehensive feature engineering pipeline
        
        Args:
            data: DataFrame with OHLCV data
            symbol: Trading symbol
            
        Returns:
            engineered_data: DataFrame with engineered features
        """
        try:
            print(f"Engineering features for {symbol}...")
            
            # Ensure we have required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Make a copy to avoid modifying original data
            engineered_data = data.copy()
            
            # 1. Technical Indicators
            print("  → Calculating technical indicators...")
            engineered_data = self.technical_indicators.calculate_all_indicators(engineered_data)
            
            # 2. Price-based Features
            print("  → Creating price-based features...")
            engineered_data = self._create_price_features(engineered_data)
            
            # 3. Volume-based Features
            print("  → Creating volume-based features...")
            engineered_data = self._create_volume_features(engineered_data)
            
            # 4. Volatility Features
            print("  → Creating volatility features...")
            engineered_data = self._create_volatility_features(engineered_data)
            
            # 5. Momentum Features
            print("  → Creating momentum features...")
            engineered_data = self._create_momentum_features(engineered_data)
            
            # 6. Pattern Features
            print("  → Creating pattern features...")
            engineered_data = self._create_pattern_features(engineered_data)
            
            # 7. Lagged Features
            print("  → Creating lagged features...")
            engineered_data = self._create_lagged_features(engineered_data)
            
            # 8. Target Variable
            print("  → Creating target variable...")
            engineered_data = self._create_target_variable(engineered_data)
            
            # 9. Clean and validate features
            print("  → Cleaning and validating features...")
            engineered_data = self._clean_features(engineered_data)
            
            print(f"  ✓ Feature engineering complete: {len(engineered_data.columns)} features")
            
            return engineered_data
            
        except Exception as e:
            raise Exception(f"Error in feature engineering: {str(e)}")
    
    def _create_price_features(self, data):
        """Create price-based features"""
        # High-Low ratio
        data['hl_ratio'] = data['high'] / data['low']
        
        # Open-Close ratio
        data['oc_ratio'] = data['open'] / data['close']
        
        # Price position within the day's range
        data['price_position'] = (data['close'] - data['low']) / (data['high'] - data['low'])
        
        # Gap features
        data['gap_up'] = (data['open'] > data['close'].shift(1)).astype(int)
        data['gap_down'] = (data['open'] < data['close'].shift(1)).astype(int)
        
        # Intraday returns
        data['intraday_return'] = (data['close'] - data['open']) / data['open']
        
        # Price relative to moving averages
        for period in [5, 10, 20, 50]:
            sma = data['close'].rolling(window=period).mean()
            data[f'price_vs_sma_{period}'] = data['close'] / sma
        
        return data
    
    def _create_volume_features(self, data):
        """Create volume-based features"""
        # Volume moving averages
        data['volume_ma_5'] = data['volume'].rolling(window=5).mean()
        data['volume_ma_20'] = data['volume'].rolling(window=20).mean()
        
        # Volume ratios
        data['volume_ratio_5'] = data['volume'] / data['volume_ma_5']
        data['volume_ratio_20'] = data['volume'] / data['volume_ma_20']
        
        # Volume price trend
        data['volume_price_trend'] = data['volume'] * data['price_change']
        
        # On-Balance Volume (OBV)
        data['obv'] = (data['volume'] * np.sign(data['price_change'])).cumsum()
        
        return data
    
    def _create_volatility_features(self, data):
        """Create volatility-based features"""
        # Different volatility measures
        for window in [5, 10, 20]:
            data[f'volatility_{window}'] = data['close'].rolling(window=window).std()
            data[f'volatility_ratio_{window}'] = data[f'volatility_{window}'] / data['close']
        
        # Garman-Klass volatility estimator
        data['gk_volatility'] = np.sqrt(
            0.5 * (np.log(data['high'] / data['low']))**2 - 
            (2 * np.log(2) - 1) * (np.log(data['close'] / data['open']))**2
        )
        
        return data
    
    def _create_momentum_features(self, data):
        """Create momentum-based features"""
        # Rate of change for different periods
        for period in [1, 3, 5, 10, 20]:
            data[f'roc_{period}'] = data['close'].pct_change(periods=period)
        
        # Momentum oscillator
        data['momentum_10'] = data['close'] / data['close'].shift(10)
        data['momentum_20'] = data['close'] / data['close'].shift(20)
        
        # Price acceleration
        data['price_acceleration'] = data['close'].pct_change() - data['close'].pct_change().shift(1)
        
        return data
    
    def _create_pattern_features(self, data):
        """Create pattern-based features"""
        # Candlestick patterns
        data['doji'] = (abs(data['close'] - data['open']) / (data['high'] - data['low']) < 0.1).astype(int)
        data['hammer'] = ((data['close'] > data['open']) & 
                         ((data['close'] - data['open']) / (data['high'] - data['low']) > 0.6)).astype(int)
        data['shooting_star'] = ((data['open'] > data['close']) & 
                               ((data['open'] - data['close']) / (data['high'] - data['low']) > 0.6)).astype(int)
        
        # Support and resistance levels
        data['high_20'] = data['high'].rolling(window=20).max()
        data['low_20'] = data['low'].rolling(window=20).min()
        data['near_resistance'] = (data['close'] / data['high_20'] > 0.98).astype(int)
        data['near_support'] = (data['close'] / data['low_20'] < 1.02).astype(int)
        
        return data
    
    def _create_lagged_features(self, data):
        """Create lagged features"""
        # Lagged returns
        for lag in [1, 2, 3, 5]:
            data[f'return_lag_{lag}'] = data['close'].pct_change().shift(lag)
        
        # Lagged volume
        for lag in [1, 2, 3]:
            data[f'volume_lag_{lag}'] = data['volume'].shift(lag)
        
        # Lagged indicators
        if 'rsi' in data.columns:
            data['rsi_lag_1'] = data['rsi'].shift(1)
            data['rsi_lag_2'] = data['rsi'].shift(2)
        
        return data
    
    def _create_target_variable(self, data):
        """Create target variable for prediction"""
        # Future return (next day)
        data['future_return'] = data['close'].pct_change().shift(-1)
        
        # Binary classification: up (1) or down/flat (0)
        data['target'] = (data['future_return'] > 0).astype(int)
        
        # Multi-class classification: strong up (2), up (1), flat (0), down (-1), strong down (-2)
        conditions = [
            data['future_return'] > 0.02,  # Strong up
            data['future_return'] > 0,     # Up
            data['future_return'] > -0.02, # Flat
            data['future_return'] > -0.05, # Down
        ]
        choices = [2, 1, 0, -1]
        data['target_multiclass'] = np.select(conditions, choices, default=-2)
        
        return data
    
    def _clean_features(self, data):
        """Clean and validate features"""
        # Remove infinite values
        data = data.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values with forward fill, then backward fill
        data = data.fillna(method='ffill').fillna(method='bfill')
        
        # Fill any remaining NaN values with 0
        data = data.fillna(0)
        
        # Remove rows where target is NaN
        data = data.dropna(subset=['target'])
        
        # Remove first few rows that might have NaN due to indicators
        data = data.iloc[50:]  # Skip first 50 rows
        
        # Final NaN check - fill any remaining NaN with 0
        data = data.fillna(0)
        
        return data
    
    def prepare_training_data(self, data, target_column='target', test_size=0.2, 
                             scale_features=True, select_features=True, feature_selection_k=50):
        """
        Comprehensive data preparation for model training
        
        Args:
            data: DataFrame with features and target
            target_column: Column name for target variable
            test_size: Fraction of data for testing
            scale_features: Whether to scale features
            select_features: Whether to perform feature selection
            feature_selection_k: Number of top features to select
            
        Returns:
            X_train, X_test, y_train, y_test: Training and testing data
        """
        try:
            print(f"Preparing training data...")
            
            # Separate features and target
            feature_columns = [col for col in data.columns 
                             if col not in [target_column, 'target_multiclass', 'future_return']]
            X = data[feature_columns]
            y = data[target_column]
            
            print(f"  → Features: {len(feature_columns)}")
            print(f"  → Samples: {len(X)}")
            print(f"  → Target distribution: {y.value_counts().to_dict()}")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
            # Scale features if requested
            if scale_features:
                print("  → Scaling features...")
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Convert back to DataFrame to preserve column names
                X_train = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
                X_test = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)
                
                # Store scaler for later use
                self.scalers[target_column] = scaler
            
            # Feature selection if requested
            if select_features and len(feature_columns) > feature_selection_k:
                print(f"  → Selecting top {feature_selection_k} features...")
                selector = SelectKBest(score_func=f_classif, k=feature_selection_k)
                X_train_selected = selector.fit_transform(X_train, y_train)
                X_test_selected = selector.transform(X_test)
                
                # Get selected feature names
                selected_features = X.columns[selector.get_support()].tolist()
                print(f"  → Selected features: {len(selected_features)}")
                
                # Convert back to DataFrame
                X_train = pd.DataFrame(X_train_selected, columns=selected_features, index=X_train.index)
                X_test = pd.DataFrame(X_test_selected, columns=selected_features, index=X_test.index)
                
                # Store selector for later use
                self.feature_selectors[target_column] = selector
            
            print("  ✓ Data preparation complete")
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            raise Exception(f"Error preparing training data: {str(e)}")
    
    def train_model_with_optimization(self, X_train, X_test, y_train, y_test, 
                                    algorithm='random_forest', optimize_hyperparams=True):
        """
        Train model with hyperparameter optimization
        
        Args:
            X_train, X_test, y_train, y_test: Training and testing data
            algorithm: Algorithm to use ('random_forest', 'xgboost')
            optimize_hyperparams: Whether to optimize hyperparameters
            
        Returns:
            model: Trained model
            metrics: Performance metrics
        """
        try:
            print(f"Training {algorithm} model...")
            
            if algorithm == 'random_forest':
                from sklearn.ensemble import RandomForestClassifier
                
                if optimize_hyperparams:
                    print("  → Optimizing hyperparameters...")
                    model = RandomForestClassifier(random_state=42, n_jobs=-1)
                    param_grid = self.algorithm_configs['random_forest']
                    
                    # Reduce grid for faster training
                    param_grid = {
                        'n_estimators': [100, 200],
                        'max_depth': [10, 20],
                        'min_samples_split': [2, 5],
                        'min_samples_leaf': [1, 2]
                    }
                    
                    grid_search = GridSearchCV(
                        model, param_grid, cv=3, scoring='accuracy',
                        n_jobs=-1, verbose=1
                    )
                    grid_search.fit(X_train, y_train)
                    model = grid_search.best_estimator_
                    
                    print(f"  → Best parameters: {grid_search.best_params_}")
                else:
                    model = RandomForestClassifier(
                        n_estimators=100, max_depth=20, random_state=42, n_jobs=-1
                    )
                    model.fit(X_train, y_train)
                    
            elif algorithm == 'xgboost':
                import xgboost as xgb
                
                if optimize_hyperparams:
                    print("  → Optimizing hyperparameters...")
                    model = xgb.XGBClassifier(random_state=42, n_jobs=-1)
                    param_grid = {
                        'n_estimators': [100, 200],
                        'max_depth': [3, 6],
                        'learning_rate': [0.01, 0.1],
                        'subsample': [0.8, 1.0]
                    }
                    
                    grid_search = GridSearchCV(
                        model, param_grid, cv=3, scoring='accuracy',
                        n_jobs=-1, verbose=1
                    )
                    grid_search.fit(X_train, y_train)
                    model = grid_search.best_estimator_
                    
                    print(f"  → Best parameters: {grid_search.best_params_}")
                else:
                    model = xgb.XGBClassifier(
                        n_estimators=100, max_depth=6, random_state=42, n_jobs=-1
                    )
                    model.fit(X_train, y_train)
            
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")
            
            # Evaluate model
            print("  → Evaluating model...")
            metrics = self._evaluate_model(model, X_train, X_test, y_train, y_test)
            
            print(f"  ✓ Model training complete - Accuracy: {metrics['accuracy']:.4f}")
            return model, metrics
            
        except Exception as e:
            raise Exception(f"Error training model: {str(e)}")
    
    def _evaluate_model(self, model, X_train, X_test, y_train, y_test):
        """
        Comprehensive model evaluation
        
        Args:
            model: Trained model
            X_train, X_test, y_train, y_test: Training and testing data
            
        Returns:
            metrics: Dictionary with evaluation metrics
        """
        try:
            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Probabilities (if available)
            try:
                y_train_proba = model.predict_proba(X_train)[:, 1]
                y_test_proba = model.predict_proba(X_test)[:, 1]
            except:
                y_train_proba = y_train_pred
                y_test_proba = y_test_pred
            
            # Calculate metrics
            metrics = {
                'train_accuracy': accuracy_score(y_train, y_train_pred),
                'test_accuracy': accuracy_score(y_test, y_test_pred),
                'accuracy': accuracy_score(y_test, y_test_pred),  # Main accuracy
                'train_auc': roc_auc_score(y_train, y_train_proba) if len(np.unique(y_train)) > 1 else 0,
                'test_auc': roc_auc_score(y_test, y_test_proba) if len(np.unique(y_test)) > 1 else 0,
                'confusion_matrix': confusion_matrix(y_test, y_test_pred).tolist(),
                'classification_report': classification_report(y_test, y_test_pred, output_dict=True),
                'feature_importance': None,
                'cross_val_score': None
            }
            
            # Feature importance (if available)
            if hasattr(model, 'feature_importances_'):
                feature_names = X_train.columns if hasattr(X_train, 'columns') else [f'feature_{i}' for i in range(X_train.shape[1])]
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                metrics['feature_importance'] = importance_df.head(20).to_dict('records')
            
            # Cross-validation score
            try:
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
                metrics['cross_val_score'] = {
                    'mean': cv_scores.mean(),
                    'std': cv_scores.std(),
                    'scores': cv_scores.tolist()
                }
            except:
                pass
            
            # Additional metrics from classification report
            if 'classification_report' in metrics:
                report = metrics['classification_report']
                metrics['precision'] = report['weighted avg']['precision']
                metrics['recall'] = report['weighted avg']['recall']
                metrics['f1_score'] = report['weighted avg']['f1-score']
            
            return metrics
            
        except Exception as e:
            print(f"Error evaluating model: {str(e)}")
            return {
                'accuracy': 0,
                'train_accuracy': 0,
                'test_accuracy': 0,
                'error': str(e)
            }
    
    def run_comprehensive_training(self, data, symbol, algorithms=['random_forest', 'xgboost'], 
                                 optimize_hyperparams=True):
        """
        Run comprehensive end-to-end training pipeline
        
        Args:
            data: Raw OHLCV data
            symbol: Trading symbol
            algorithms: List of algorithms to train
            optimize_hyperparams: Whether to optimize hyperparameters
            
        Returns:
            results: Dictionary with training results
        """
        try:
            print(f"Starting comprehensive training for {symbol}")
            print("=" * 50)
            
            # Step 1: Feature Engineering
            engineered_data = self.engineer_features(data, symbol)
            
            # Step 2: Data Preparation
            X_train, X_test, y_train, y_test = self.prepare_training_data(
                engineered_data, target_column='target'
            )
            
            # Step 3: Train Models
            results = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'data_shape': {
                    'train': X_train.shape,
                    'test': X_test.shape,
                    'features': list(X_train.columns)
                },
                'models': {}
            }
            
            for algorithm in algorithms:
                try:
                    print(f"\nTraining {algorithm.upper()} model...")
                    model, metrics = self.train_model_with_optimization(
                        X_train, X_test, y_train, y_test,
                        algorithm=algorithm,
                        optimize_hyperparams=optimize_hyperparams
                    )
                    
                    # Store model and results
                    model_key = f"{symbol}_{algorithm}"
                    self.trained_models[model_key] = {
                        'model': model,
                        'scaler': self.scalers.get('target'),
                        'feature_selector': self.feature_selectors.get('target'),
                        'feature_names': list(X_train.columns)
                    }
                    
                    asset_class = self._detect_asset_class(symbol)
                    
                    results['models'][algorithm] = {
                        'status': 'success',
                        'metrics': metrics,
                        'model_key': model_key,
                        'asset_class': asset_class
                    }
                    
                    self.model_performance[model_key] = {
                        'metrics': metrics,
                        'asset_class': asset_class,
                        'symbol': symbol,
                        'algorithm': algorithm,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    # Save model to disk
                    self.save_model(self.trained_models[model_key], model_key)
                    
                except Exception as e:
                    print(f"Error training {algorithm}: {str(e)}")
                    results['models'][algorithm] = {
                        'status': 'failed',
                        'error': str(e)
                    }
            
            # Step 4: Model Comparison
            results['comparison'] = self._compare_model_results(results['models'])
            
            # Step 5: Select Best Model
            best_model, best_score = self._select_best_model(results['models'])
            results['best_model'] = {
                'algorithm': best_model,
                'score': best_score
            }
            
            print(f"\n" + "=" * 50)
            print(f"TRAINING COMPLETE FOR {symbol}")
            print(f"Best model: {best_model} (Accuracy: {best_score:.4f})")
            print("=" * 50)
            
            return results
            
        except Exception as e:
            raise Exception(f"Error in comprehensive training: {str(e)}")
    
    def _compare_model_results(self, models):
        """Compare results from multiple models"""
        comparison = {}
        
        for algorithm, result in models.items():
            if result['status'] == 'success':
                metrics = result['metrics']
                comparison[algorithm] = {
                    'accuracy': metrics.get('accuracy', 0),
                    'precision': metrics.get('precision', 0),
                    'recall': metrics.get('recall', 0),
                    'f1_score': metrics.get('f1_score', 0),
                    'auc': metrics.get('test_auc', 0)
                }
            else:
                comparison[algorithm] = {
                    'accuracy': 0,
                    'error': result.get('error', 'Unknown error')
                }
        
        return comparison
    
    def _select_best_model(self, models):
        """Select the best performing model"""
        best_model = None
        best_score = -1
        
        for algorithm, result in models.items():
            if result['status'] == 'success':
                score = result['metrics'].get('accuracy', 0)
                if score > best_score:
                    best_score = score
                    best_model = algorithm
        
        return best_model, best_score
    
    def predict_with_model(self, model_key, data):
        """
        Make predictions using a trained model
        
        Args:
            model_key: Key for the trained model
            data: Input data for prediction
            
        Returns:
            predictions: Model predictions
        """
        try:
            if model_key not in self.trained_models:
                raise ValueError(f"Model {model_key} not found")
            
            model_info = self.trained_models[model_key]
            model = model_info['model']
            scaler = model_info.get('scaler')
            selector = model_info.get('feature_selector')
            
            # Ensure data has the correct features
            feature_names = model_info['feature_names']
            if hasattr(data, 'columns'):
                # Select and reorder columns to match training data
                data = data.reindex(columns=feature_names, fill_value=0)
            
            # Apply scaling if used during training
            if scaler is not None:
                data = scaler.transform(data)
            
            # Apply feature selection if used during training
            if selector is not None:
                data = selector.transform(data)
            
            # Make predictions
            predictions = model.predict(data)
            
            # Get probabilities if available
            try:
                probabilities = model.predict_proba(data)
                return predictions, probabilities
            except:
                return predictions, None
                
        except Exception as e:
            raise Exception(f"Error making predictions: {str(e)}")
    
    def save_model(self, model_data, model_key):
        """
        Save model and associated data to cache
        
        Args:
            model_data: Dictionary containing model and preprocessing objects
            model_key: Unique key for the model
        """
        try:
            cache_path = os.path.join(self.model_cache_dir, f"{model_key}.joblib")
            joblib.dump(model_data, cache_path)
            print(f"  → Model saved: {cache_path}")
            
        except Exception as e:
            print(f"Error saving model {model_key}: {str(e)}")
    
    def load_model(self, model_key):
        """
        Load model from cache
        
        Args:
            model_key: Unique key for the model
            
        Returns:
            model_data: Dictionary containing model and preprocessing objects
        """
        try:
            cache_path = os.path.join(self.model_cache_dir, f"{model_key}.joblib")
            
            if os.path.exists(cache_path):
                model_data = joblib.load(cache_path)
                self.trained_models[model_key] = model_data
                return model_data
            
            return None
            
        except Exception as e:
            print(f"Error loading model {model_key}: {str(e)}")
            return None
    
    def get_model_summary(self):
        """
        Get summary of all trained models
        
        Returns:
            summary: Dictionary with model summaries
        """
        summary = {
            'total_models': 0,
            'models': {}
        }
        
        # Check both in-memory models and cached models
        all_models = {}
        
        # Add in-memory models
        for model_key, model_info in self.trained_models.items():
            all_models[model_key] = {
                'features': len(model_info['feature_names']),
                'has_scaler': model_info.get('scaler') is not None,
                'has_selector': model_info.get('feature_selector') is not None,
                'algorithm': type(model_info['model']).__name__
            }
        
        # Check cache directory for saved models
        if os.path.exists(self.model_cache_dir):
            for filename in os.listdir(self.model_cache_dir):
                if filename.endswith('.joblib'):
                    model_key = filename.replace('.joblib', '')
                    
                    # Only add if not already in memory
                    if model_key not in all_models:
                        try:
                            # Load model to get info
                            model_data = joblib.load(os.path.join(self.model_cache_dir, filename))
                            all_models[model_key] = {
                                'features': len(model_data.get('feature_names', [])),
                                'has_scaler': 'scaler' in model_data,
                                'has_selector': 'feature_selector' in model_data,
                                'algorithm': type(model_data['model']).__name__
                            }
                        except Exception as e:
                            print(f"Error loading cached model {filename}: {e}")
        
        summary['total_models'] = len(all_models)
        summary['models'] = all_models
        
        return summary
    
    def make_predictions(self, features, symbol, algorithm='random_forest'):
        """Make predictions using a trained model"""
        try:
            # Load the trained model
            model_filename = f"{symbol}_{algorithm}.joblib"
            model_path = os.path.join(os.path.dirname(__file__), 'cache', model_filename)
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"No trained model found: {model_filename}")
            
            model_data = joblib.load(model_path)
            model = model_data['model']
            scaler = model_data.get('scaler')
            feature_selector = model_data.get('feature_selector')
            feature_names = model_data.get('feature_names', [])
            
            # Prepare features
            if len(features) == 0:
                raise ValueError("No features available for prediction")
            
            # Drop target column if present
            prediction_features = features.drop(['target'], axis=1, errors='ignore')
            
            # Ensure we have the right feature names
            if feature_names:
                # Reorder/filter features to match training
                missing_features = set(feature_names) - set(prediction_features.columns)
                if missing_features:
                    # Add missing features with zeros
                    for feature in missing_features:
                        prediction_features[feature] = 0
                
                # Select only the features used in training
                prediction_features = prediction_features[feature_names]
            
            # Apply scaling
            if scaler:
                prediction_features = scaler.transform(prediction_features)
            
            # Apply feature selection
            if feature_selector:
                prediction_features = feature_selector.transform(prediction_features)
            
            # Make predictions
            predictions = model.predict(prediction_features)
            
            return predictions
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return None


# Legacy ModelTrainer class for backward compatibility
class ModelTrainer(RobustModelTrainer):
    """Legacy ModelTrainer class that provides backward compatibility"""
    
    def __init__(self):
        super().__init__()
        print("Note: Using enhanced RobustModelTrainer with comprehensive features")
    
    def train_random_forest(self, data, symbol, model_type='classifier'):
        """Legacy method - use run_comprehensive_training instead"""
        results = self.run_comprehensive_training(data, symbol, algorithms=['random_forest'])
        
        if 'random_forest' in results['models'] and results['models']['random_forest']['status'] == 'success':
            model_key = results['models']['random_forest']['model_key']
            model = self.trained_models[model_key]['model']
            metrics = results['models']['random_forest']['metrics']
            return model, metrics
        else:
            raise Exception("Random Forest training failed")
    
    def train_xgboost(self, data, symbol, model_type='classifier'):
        """Legacy method - use run_comprehensive_training instead"""
        results = self.run_comprehensive_training(data, symbol, algorithms=['xgboost'])
        
        if 'xgboost' in results['models'] and results['models']['xgboost']['status'] == 'success':
            model_key = results['models']['xgboost']['model_key']
            model = self.trained_models[model_key]['model']
            metrics = results['models']['xgboost']['metrics']
            return model, metrics
        else:
            raise Exception("XGBoost training failed")
    
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
    
    def get_performance_summary(self, symbol: str = None) -> dict:
        """
        Get a summary of all model performance metrics including precision, recall, F1.
        
        Args:
            symbol: Optional symbol to filter results
            
        Returns:
            Dictionary with performance summary
        """
        summary = {
            'models': {},
            'overall': {},
            'by_asset_class': {}
        }
        
        asset_class_metrics = {
            'stocks': {'accuracies': [], 'precisions': [], 'recalls': [], 'f1_scores': []},
            'crypto': {'accuracies': [], 'precisions': [], 'recalls': [], 'f1_scores': []},
            'forex': {'accuracies': [], 'precisions': [], 'recalls': [], 'f1_scores': []},
            'options': {'accuracies': [], 'precisions': [], 'recalls': [], 'f1_scores': []}
        }
        
        for key, perf in self.model_performance.items():
            if symbol and symbol not in key:
                continue
            
            metrics = perf.get('metrics', {})
            model_metrics = {
                'accuracy': round(metrics.get('accuracy', 0) * 100, 2),
                'precision': round(metrics.get('precision', 0) * 100, 2),
                'recall': round(metrics.get('recall', 0) * 100, 2),
                'f1_score': round(metrics.get('f1_score', 0) * 100, 2),
                'train_accuracy': round(metrics.get('train_accuracy', 0) * 100, 2),
                'test_accuracy': round(metrics.get('test_accuracy', 0) * 100, 2),
                'auc': round(metrics.get('test_auc', 0), 4)
            }
            summary['models'][key] = model_metrics
            
            asset_class = perf.get('asset_class', 'stocks')
            if asset_class in asset_class_metrics:
                asset_class_metrics[asset_class]['accuracies'].append(model_metrics['accuracy'])
                asset_class_metrics[asset_class]['precisions'].append(model_metrics['precision'])
                asset_class_metrics[asset_class]['recalls'].append(model_metrics['recall'])
                asset_class_metrics[asset_class]['f1_scores'].append(model_metrics['f1_score'])
        
        for ac, data in asset_class_metrics.items():
            if data['accuracies']:
                summary['by_asset_class'][ac] = {
                    'avg_accuracy': round(np.mean(data['accuracies']), 2),
                    'avg_precision': round(np.mean(data['precisions']), 2),
                    'avg_recall': round(np.mean(data['recalls']), 2),
                    'avg_f1_score': round(np.mean(data['f1_scores']), 2),
                    'model_count': len(data['accuracies'])
                }
        
        if summary['models']:
            accuracies = [m['accuracy'] for m in summary['models'].values()]
            precisions = [m['precision'] for m in summary['models'].values()]
            recalls = [m['recall'] for m in summary['models'].values()]
            f1_scores = [m['f1_score'] for m in summary['models'].values()]
            
            summary['overall'] = {
                'avg_accuracy': round(np.mean(accuracies), 2),
                'avg_precision': round(np.mean(precisions), 2),
                'avg_recall': round(np.mean(recalls), 2),
                'avg_f1_score': round(np.mean(f1_scores), 2),
                'total_models': len(summary['models']),
                'best_model': max(summary['models'].items(), key=lambda x: x[1]['f1_score'])[0] if summary['models'] else None
            }
        
        if hasattr(self, '_perf_tracker'):
            summary['trading_metrics'] = self._perf_tracker.get_all_metrics()
        
        return summary
    
    def record_trading_result(self, asset_class: str, predicted: str, actual: str, 
                              profit_loss: float = 0) -> dict:
        """
        Record a trading prediction result for metrics calculation.
        Integrates with goal-based optimizer's ModelPerformanceTracker.
        
        Args:
            asset_class: 'stocks', 'crypto', 'forex', 'options'
            predicted: Predicted signal ('BUY', 'SELL', 'HOLD')
            actual: What the correct signal should have been
            profit_loss: P&L if trade was taken
            
        Returns:
            Updated metrics for this asset class
        """
        try:
            from trading.goal_based_optimizer import ModelPerformanceTracker
            
            if not hasattr(self, '_perf_tracker'):
                self._perf_tracker = ModelPerformanceTracker()
            
            self._perf_tracker.record_prediction(asset_class, predicted, actual, profit_loss)
            return self._perf_tracker.get_metrics(asset_class)
        except ImportError:
            return {'error': 'ModelPerformanceTracker not available'}
    
    def get_asset_class_metrics(self) -> dict:
        """Get performance metrics by asset class"""
        if hasattr(self, '_perf_tracker'):
            return self._perf_tracker.get_all_metrics()
        return {}