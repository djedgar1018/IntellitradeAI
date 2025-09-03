"""
LSTM Model for Time Series Prediction
Implements LSTM neural network for trading signal generation
"""
# models/lstm_model.py
import numpy as np

class SkipLSTM(Exception):
    pass

def make_model(input_dim, params=None):
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
    Xs, ys = [], []
    arr = X.values.astype("float32")
    for i in range(len(arr)-window-1):
        Xs.append(arr[i:i+window])
        ys.append(y.values[i+window])
    return np.array(Xs), np.array(ys)
