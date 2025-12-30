"""
Transformer Architecture for Market Regime Detection
Attention-based model for capturing complex market patterns
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (
        Input, Dense, Dropout, LayerNormalization,
        MultiHeadAttention, GlobalAveragePooling1D, Add
    )
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    HAS_TF = True
except ImportError:
    HAS_TF = False

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


class PositionalEncoding(tf.keras.layers.Layer):
    """Positional encoding layer for Transformer"""
    
    def __init__(self, sequence_length: int, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.d_model = d_model
        
    def build(self, input_shape):
        position = np.arange(self.sequence_length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.d_model, 2) * (-np.log(10000.0) / self.d_model))
        
        pe = np.zeros((self.sequence_length, self.d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term[:self.d_model//2])
        
        self.pos_encoding = tf.constant(pe, dtype=tf.float32)
        
    def call(self, x):
        return x + self.pos_encoding[:tf.shape(x)[1], :]


class TransformerBlock(tf.keras.layers.Layer):
    """Single Transformer encoder block"""
    
    def __init__(self, d_model: int, num_heads: int, ff_dim: int, dropout_rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        
        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation='relu'),
            Dense(d_model)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
        
    def call(self, x, training=False):
        attn_output = self.mha(x, x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(Add()([x, attn_output]))
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(Add()([out1, ffn_output]))


class TransformerPricePredictor:
    """Transformer-based model for market prediction"""
    
    def __init__(self, 
                 sequence_length: int = 30,
                 d_model: int = 64,
                 num_heads: int = 4,
                 ff_dim: int = 128,
                 num_blocks: int = 2,
                 dropout_rate: float = 0.1):
        """
        Initialize Transformer predictor
        
        Args:
            sequence_length: Number of time steps
            d_model: Model dimension
            num_heads: Number of attention heads
            ff_dim: Feed-forward dimension
            num_blocks: Number of Transformer blocks
            dropout_rate: Dropout rate
        """
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_blocks = num_blocks
        self.dropout_rate = dropout_rate
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        if not HAS_TF:
            print("Warning: TensorFlow not available. Transformer models disabled.")
    
    def build_model(self, n_features: int) -> Optional[Model]:
        """Build Transformer architecture"""
        if not HAS_TF:
            return None
        
        inputs = Input(shape=(self.sequence_length, n_features))
        
        x = Dense(self.d_model)(inputs)
        x = PositionalEncoding(self.sequence_length, self.d_model)(x)
        
        for _ in range(self.num_blocks):
            x = TransformerBlock(
                self.d_model, 
                self.num_heads, 
                self.ff_dim, 
                self.dropout_rate
            )(x)
        
        x = GlobalAveragePooling1D()(x)
        
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.1)(x)
        
        outputs = Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def prepare_sequences(self, X: np.ndarray, y: np.ndarray = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Prepare sequential data for Transformer input"""
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
        """Train the Transformer model"""
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


class MarketRegimeDetector:
    """
    Detects market regime (bull/bear/sideways) using Transformer
    Useful for adjusting trading strategy based on market conditions
    """
    
    REGIMES = {
        0: 'bear',
        1: 'sideways',
        2: 'bull'
    }
    
    def __init__(self, sequence_length: int = 20):
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def _calculate_regime_labels(self, data: pd.DataFrame, 
                                  bull_threshold: float = 5.0,
                                  bear_threshold: float = -5.0,
                                  horizon: int = 10) -> np.ndarray:
        """Calculate regime labels based on future returns"""
        future_return = (data['close'].shift(-horizon) - data['close']) / data['close'] * 100
        
        labels = np.ones(len(data))
        labels[future_return > bull_threshold] = 2
        labels[future_return < bear_threshold] = 0
        
        return labels
    
    def build_model(self, n_features: int) -> Optional[Model]:
        """Build regime detection model"""
        if not HAS_TF:
            return None
        
        inputs = Input(shape=(self.sequence_length, n_features))
        
        x = Dense(32)(inputs)
        x = PositionalEncoding(self.sequence_length, 32)(x)
        
        x = TransformerBlock(32, 2, 64, 0.1)(x)
        x = GlobalAveragePooling1D()(x)
        
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.2)(x)
        
        outputs = Dense(3, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def fit(self, data: pd.DataFrame, 
            bull_threshold: float = 5.0,
            bear_threshold: float = -5.0,
            horizon: int = 10,
            epochs: int = 50,
            verbose: int = 0) -> Dict:
        """Train the regime detector"""
        if not HAS_TF:
            return {'error': 'TensorFlow not available'}
        
        labels = self._calculate_regime_labels(data, bull_threshold, bear_threshold, horizon)
        
        exclude_cols = ['open', 'high', 'low', 'close', 'volume']
        feature_cols = [c for c in data.columns if c.lower() not in exclude_cols]
        
        if not feature_cols:
            return {'error': 'No features available'}
        
        X = data[feature_cols].values
        y = labels
        
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
        
        X_scaled = self.scaler.fit_transform(X)
        
        n_sequences = len(X) - self.sequence_length
        X_seq = np.zeros((n_sequences, self.sequence_length, X.shape[1]))
        for i in range(n_sequences):
            X_seq[i] = X_scaled[i:i + self.sequence_length]
        y_seq = y[self.sequence_length:]
        
        if self.model is None:
            self.build_model(n_features=X.shape[1])
        
        split = int(len(X_seq) * 0.8)
        X_train, X_test = X_seq[:split], X_seq[split:]
        y_train, y_test = y_seq[:split], y_seq[split:]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=32,
            callbacks=[
                EarlyStopping(patience=10, restore_best_weights=True)
            ],
            verbose=verbose
        )
        
        self.is_fitted = True
        
        return {
            'epochs': len(history.history['loss']),
            'final_accuracy': history.history['accuracy'][-1],
            'val_accuracy': history.history['val_accuracy'][-1]
        }
    
    def predict_regime(self, data: pd.DataFrame) -> Dict:
        """Predict current market regime"""
        if not HAS_TF or not self.is_fitted:
            return {'regime': 'unknown', 'probabilities': {}}
        
        exclude_cols = ['open', 'high', 'low', 'close', 'volume']
        feature_cols = [c for c in data.columns if c.lower() not in exclude_cols]
        
        X = data[feature_cols].tail(self.sequence_length).values
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
        X_scaled = self.scaler.transform(X)
        X_seq = X_scaled.reshape(1, self.sequence_length, -1)
        
        probs = self.model.predict(X_seq, verbose=0)[0]
        regime_idx = np.argmax(probs)
        
        return {
            'regime': self.REGIMES[regime_idx],
            'regime_index': int(regime_idx),
            'probabilities': {
                'bear': float(probs[0]),
                'sideways': float(probs[1]),
                'bull': float(probs[2])
            },
            'confidence': float(max(probs))
        }


def train_transformer_for_symbol(symbol: str,
                                  threshold: float = 5.0,
                                  horizon: int = 5,
                                  sequence_length: int = 30) -> Optional[Dict]:
    """Train Transformer model for a specific symbol"""
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
        
        model = TransformerPricePredictor(sequence_length=sequence_length)
        
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
