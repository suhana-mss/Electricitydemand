"""LSTM forecaster."""
import numpy as np
from .data import inverse_target


def build_lstm(lookback, n_features, units=64):
    from tensorflow import keras
    from tensorflow.keras import layers
    m = keras.Sequential([
        layers.Input(shape=(lookback, n_features)),
        layers.LSTM(units, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(units // 2),
        layers.Dense(16, activation="relu"),
        layers.Dense(1),
    ])
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return m


def train(model, split, epochs=20, batch_size=64):
    from tensorflow import keras
    Xtr, ytr = split["train"]
    Xv, yv = split["val"]
    es = keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)
    return model.fit(Xtr, ytr, validation_data=(Xv, yv), epochs=epochs,
                     batch_size=batch_size, callbacks=[es], verbose=2)


def evaluate(model, split, scaler, target_idx):
    X, y = split["test"]
    p = inverse_target(scaler, target_idx, model.predict(X, verbose=0).ravel())
    t = inverse_target(scaler, target_idx, y)
    mae = float(np.mean(np.abs(p - t)))
    rmse = float(np.sqrt(np.mean((p - t) ** 2)))
    mask = np.abs(t) > 1e-3  # MAPE unstable near zero
    mape = float(np.mean(np.abs((p[mask] - t[mask]) / t[mask])) * 100)
    ss_res, ss_tot = np.sum((t - p) ** 2), np.sum((t - t.mean()) ** 2)
    return dict(MAE=mae, RMSE=rmse, MAPE=mape, R2=float(1 - ss_res / ss_tot)), p, t
