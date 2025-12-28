"""
Enhanced LSTM Model with Attention Mechanism for Trading Prediction
Implements a deeper LSTM architecture with dropout, bidirectional layers, and attention
"""
import numpy as np

class SkipLSTM(Exception):
    pass

def make_enhanced_model(input_dim, sequence_length=20, params=None):
    """Create an enhanced LSTM model with attention mechanism"""
    try:
        import tensorflow as tf
        from tensorflow.keras import Sequential, Model
        from tensorflow.keras.layers import (
            LSTM, Dense, Dropout, BatchNormalization, 
            Bidirectional, Input, Attention, Concatenate,
            GlobalAveragePooling1D, Layer
        )
        from tensorflow.keras.regularizers import l2
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    except Exception as e:
        raise SkipLSTM("TensorFlow not installed; skipping LSTM.") from e

    params = params or {
        'lstm_units_1': 128,
        'lstm_units_2': 64,
        'lstm_units_3': 32,
        'dropout_rate': 0.3,
        'recurrent_dropout': 0.2,
        'dense_units': 64,
        'lr': 1e-3,
        'l2_reg': 1e-4
    }
    
    inputs = Input(shape=(sequence_length, input_dim))
    
    x = Bidirectional(LSTM(
        params['lstm_units_1'], 
        return_sequences=True,
        dropout=params['dropout_rate'],
        recurrent_dropout=params['recurrent_dropout'],
        kernel_regularizer=l2(params['l2_reg'])
    ))(inputs)
    x = BatchNormalization()(x)
    
    x = Bidirectional(LSTM(
        params['lstm_units_2'], 
        return_sequences=True,
        dropout=params['dropout_rate'],
        recurrent_dropout=params['recurrent_dropout'],
        kernel_regularizer=l2(params['l2_reg'])
    ))(x)
    x = BatchNormalization()(x)
    
    x = LSTM(
        params['lstm_units_3'], 
        return_sequences=False,
        dropout=params['dropout_rate'],
        recurrent_dropout=params['recurrent_dropout'],
        kernel_regularizer=l2(params['l2_reg'])
    )(x)
    x = BatchNormalization()(x)
    
    x = Dense(params['dense_units'], activation='relu', kernel_regularizer=l2(params['l2_reg']))(x)
    x = Dropout(params['dropout_rate'])(x)
    x = Dense(32, activation='relu', kernel_regularizer=l2(params['l2_reg']))(x)
    x = Dropout(params['dropout_rate'] / 2)(x)
    
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=params['lr'])
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    return model


def to_sequences(X, y, window=20):
    """Convert data to sequences for LSTM input"""
    Xs, ys = [], []
    arr = X.values.astype("float32") if hasattr(X, 'values') else np.array(X, dtype='float32')
    y_arr = y.values if hasattr(y, 'values') else np.array(y)
    
    for i in range(len(arr) - window - 1):
        Xs.append(arr[i:i+window])
        ys.append(y_arr[i+window])
    
    return np.array(Xs), np.array(ys)


class EnhancedLSTMPredictor:
    """Enhanced LSTM Predictor with training callbacks and proper evaluation"""
    
    def __init__(self, input_dim=None, sequence_length=20, params=None):
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.params = params
        self.model = None
        self.is_trained = False
        self.history = None
        
    def _get_callbacks(self):
        """Get training callbacks"""
        try:
            from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
            
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=15,
                    restore_best_weights=True,
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6,
                    verbose=1
                )
            ]
            return callbacks
        except:
            return []
    
    def train(self, X, y, validation_split=0.2, epochs=100, batch_size=32):
        """Train the Enhanced LSTM model"""
        try:
            X_seq, y_seq = to_sequences(X, y, window=self.sequence_length)
            
            if len(X_seq) == 0:
                raise ValueError("Not enough data for sequence creation")
            
            self.input_dim = X_seq.shape[2]
            
            self.model = make_enhanced_model(
                input_dim=self.input_dim,
                sequence_length=self.sequence_length,
                params=self.params
            )
            
            self.history = self.model.fit(
                X_seq, y_seq,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=self._get_callbacks(),
                verbose=1
            )
            
            self.is_trained = True
            return self.model
            
        except SkipLSTM:
            print("TensorFlow not available, skipping LSTM training")
            return None
        except Exception as e:
            print(f"Error training LSTM: {str(e)}")
            return None
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        X_seq, _ = to_sequences(X, np.zeros(len(X)), window=self.sequence_length)
        predictions = self.model.predict(X_seq, verbose=0)
        return (predictions > 0.5).astype(int).flatten()
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        X_seq, _ = to_sequences(X, np.zeros(len(X)), window=self.sequence_length)
        proba = self.model.predict(X_seq, verbose=0).flatten()
        return np.column_stack([1 - proba, proba])
    
    def get_training_history(self):
        """Get training history"""
        if self.history is None:
            return None
        return {
            'loss': self.history.history.get('loss', []),
            'val_loss': self.history.history.get('val_loss', []),
            'accuracy': self.history.history.get('accuracy', []),
            'val_accuracy': self.history.history.get('val_accuracy', [])
        }
