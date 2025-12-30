"""
LSTM Model for Sequence-Based Price Prediction
Deep learning approach for capturing temporal patterns in volatile assets
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
    from tensorflow.keras.layers import Bidirectional
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.regularizers import l2
    HAS_TF = True
except ImportError:
    HAS_TF = False

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score


class SkipLSTM(Exception):
    """Exception raised when TensorFlow is not available"""
    pass


def make_model(input_dim, params=None):
    """Legacy function for backward compatibility"""
    try:
        import tensorflow as tf
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import LSTM, Dense
    except Exception as e:
        raise SkipLSTM("TensorFlow not installed; skipping LSTM.") from e

    params = params or dict(units=32, lr=1e-3)
    model = Sequential([
        LSTM(params["units"], input_shape=(None, input_dim)),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(params["lr"]),
                  loss="binary_crossentropy", metrics=["accuracy"])
    return model


def to_sequences(X, y, window=20):
    """Convert DataFrame to sequences for LSTM"""
    Xs, ys = [], []
    arr = X.values.astype("float32")
    for i in range(len(arr)-window-1):
        Xs.append(arr[i:i+window])
        ys.append(y.values[i+window])
    return np.array(Xs), np.array(ys)


class LSTMPricePredictor:
    """LSTM-based model for predicting significant price movements"""
    
    def __init__(self, sequence_length: int = 30, n_features: int = 20):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        if not HAS_TF:
            print("Warning: TensorFlow not available. LSTM models disabled.")
    
    def build_model(self, n_features: int = None) -> Optional[Sequential]:
        """Build LSTM architecture for price prediction"""
        if not HAS_TF:
            return None
        
        if n_features:
            self.n_features = n_features
        
        model = Sequential([
            LSTM(128, return_sequences=True, 
                 input_shape=(self.sequence_length, self.n_features),
                 kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.3),
            
            LSTM(64, return_sequences=True,
                 kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.3),
            
            LSTM(32, return_sequences=False,
                 kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
            Dropout(0.2),
            
            Dense(16, activation='relu'),
            
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def prepare_sequences(self, X: np.ndarray, y: np.ndarray = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Prepare sequential data for LSTM input"""
        n_samples = len(X)
        n_sequences = n_samples - self.sequence_length
        
        if n_sequences <= 0:
            raise ValueError(f"Not enough samples ({n_samples}) for sequence length ({self.sequence_length})")
        
        X_seq = np.zeros((n_sequences, self.sequence_length, X.shape[1]))
        
        for i in range(n_sequences):
            X_seq[i] = X[i:i + self.sequence_length]
        
        if y is not None:
            y_seq = y[self.sequence_length:]
        else:
            y_seq = None
        
        return X_seq, y_seq
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            validation_split: float = 0.2,
            epochs: int = 100,
            batch_size: int = 32,
            verbose: int = 0) -> Dict:
        """Train the LSTM model"""
        if not HAS_TF:
            return {'error': 'TensorFlow not available'}
        
        X_scaled = self.scaler.fit_transform(X)
        X_seq, y_seq = self.prepare_sequences(X_scaled, y)
        
        if self.model is None:
            self.build_model(n_features=X.shape[1])
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=verbose
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-6,
                verbose=verbose
            )
        ]
        
        history = self.model.fit(
            X_seq, y_seq,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.is_fitted = True
        
        return {
            'epochs_trained': len(history.history['loss']),
            'final_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1],
            'final_accuracy': history.history['accuracy'][-1],
            'final_val_accuracy': history.history['val_accuracy'][-1]
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not HAS_TF or not self.is_fitted:
            return np.array([])
        
        X_scaled = self.scaler.transform(X)
        X_seq, _ = self.prepare_sequences(X_scaled)
        
        predictions = self.model.predict(X_seq, verbose=0)
        
        return predictions.flatten()
    
    def predict_binary(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Make binary predictions"""
        proba = self.predict(X)
        return (proba >= threshold).astype(int)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Evaluate model performance"""
        if not HAS_TF or not self.is_fitted:
            return {'error': 'Model not fitted or TensorFlow unavailable'}
        
        X_scaled = self.scaler.transform(X)
        X_seq, y_seq = self.prepare_sequences(X_scaled, y)
        
        loss, accuracy = self.model.evaluate(X_seq, y_seq, verbose=0)
        
        y_pred = self.predict_binary(X)
        y_true = y[self.sequence_length:]
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'sklearn_accuracy': accuracy_score(y_true, y_pred),
            'n_samples': len(y_seq)
        }


class BidirectionalLSTM(LSTMPricePredictor):
    """Bidirectional LSTM for capturing patterns in both directions"""
    
    def build_model(self, n_features: int = None) -> Optional[Model]:
        """Build bidirectional LSTM architecture"""
        if not HAS_TF:
            return None
        
        if n_features:
            self.n_features = n_features
        
        model = Sequential([
            Bidirectional(
                LSTM(64, return_sequences=True),
                input_shape=(self.sequence_length, self.n_features)
            ),
            BatchNormalization(),
            Dropout(0.3),
            
            Bidirectional(LSTM(32, return_sequences=False)),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model


class LSTMEnsemble:
    """Ensemble of LSTM models for robust predictions"""
    
    def __init__(self, n_models: int = 3, sequence_length: int = 30):
        self.n_models = n_models
        self.sequence_length = sequence_length
        self.models = []
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict:
        """Train all models in ensemble"""
        if not HAS_TF:
            return {'error': 'TensorFlow not available'}
        
        results = []
        
        for i in range(self.n_models):
            print(f"Training model {i+1}/{self.n_models}...")
            
            if i == 0:
                model = LSTMPricePredictor(sequence_length=self.sequence_length)
            elif i == 1:
                model = BidirectionalLSTM(sequence_length=self.sequence_length)
            else:
                model = LSTMPricePredictor(sequence_length=self.sequence_length + 10)
            
            result = model.fit(X, y, **kwargs)
            results.append(result)
            self.models.append(model)
        
        self.is_fitted = True
        
        return {
            'n_models': len(self.models),
            'individual_results': results
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions (average of all models)"""
        if not self.is_fitted:
            return np.array([])
        
        predictions = []
        min_len = float('inf')
        
        for model in self.models:
            pred = model.predict(X)
            if len(pred) > 0:
                predictions.append(pred)
                min_len = min(min_len, len(pred))
        
        if not predictions:
            return np.array([])
        
        aligned_preds = [p[-int(min_len):] for p in predictions]
        
        ensemble_pred = np.mean(aligned_preds, axis=0)
        
        return ensemble_pred
    
    def predict_binary(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Make binary ensemble predictions"""
        proba = self.predict(X)
        return (proba >= threshold).astype(int)


def train_lstm_for_symbol(symbol: str, 
                          threshold: float = 5.0,
                          horizon: int = 5,
                          sequence_length: int = 30) -> Optional[Dict]:
    """Train LSTM model for a specific symbol"""
    if not HAS_TF:
        return {'error': 'TensorFlow not available'}
    
    try:
        import yfinance as yf
        from datetime import datetime, timedelta
        from models.volatility_features import VolatilityFeatures
        
        data = yf.download(symbol, 
                          start=datetime.now() - timedelta(days=1825),
                          end=datetime.now(),
                          progress=False)
        
        if data.empty or len(data) < 300:
            return {'error': f'Insufficient data for {symbol}'}
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        data.columns = [c.lower() for c in data.columns]
        
        vol_features = VolatilityFeatures()
        data = vol_features.calculate_all_volatility_features(data)
        
        future_return = (data['close'].shift(-horizon) - data['close']) / data['close'] * 100
        data['target'] = (future_return > threshold).astype(int)
        
        data = data.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(data) < 200:
            return {'error': f'Insufficient data after processing for {symbol}'}
        
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'target']
        feature_cols = [c for c in data.columns if c not in exclude_cols]
        
        X = data[feature_cols].values
        y = data['target'].values
        
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        model = LSTMPricePredictor(sequence_length=sequence_length)
        
        train_result = model.fit(X_train, y_train, 
                                epochs=50, 
                                batch_size=32,
                                verbose=0)
        
        eval_result = model.evaluate(X_test, y_test)
        
        return {
            'symbol': symbol,
            'threshold': threshold,
            'horizon': horizon,
            'sequence_length': sequence_length,
            'n_features': len(feature_cols),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'training': train_result,
            'evaluation': eval_result,
            'accuracy': eval_result.get('accuracy', 0)
        }
        
    except Exception as e:
        return {'error': str(e)}
