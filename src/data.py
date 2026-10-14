"""Data loading, preprocessing, feature engineering, windowing."""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

TARGET = "demand_kw"


def make_synthetic(days=400, seed=0):
    """Synthetic hourly household demand + weather (fallback when no dataset is present)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=days * 24, freq="h")
    doy = idx.dayofyear.values
    hour = idx.hour.values
    temp = 22 + 10 * np.sin(2 * np.pi * (doy - 100) / 365) + 4 * np.sin(2 * np.pi * (hour - 14) / 24) + rng.normal(0, 1.2, len(idx))
    humidity = np.clip(60 - 0.8 * (temp - 22) + rng.normal(0, 5, len(idx)), 20, 100)
    daily = 0.6 + 0.5 * np.exp(-((hour - 8) ** 2) / 6) + 0.9 * np.exp(-((hour - 20) ** 2) / 8)
    weekend = (idx.dayofweek.values >= 5) * 0.15
    cooling = 0.05 * np.maximum(temp - 26, 0) ** 1.2 + 0.03 * np.maximum(12 - temp, 0)
    demand = daily + weekend + cooling + rng.normal(0, 0.08, len(idx))
    return pd.DataFrame({TARGET: np.maximum(demand, 0.05), "temperature": temp, "humidity": humidity}, index=idx)


def load_household(path):
    """UCI Individual Household Electric Power Consumption (semicolon txt) -> hourly kW."""
    df = pd.read_csv(path, sep=";", na_values=["?"], low_memory=False)
    df["ts"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S")
    df = df.set_index("ts").sort_index()
    return df[["Global_active_power"]].astype(float).resample("h").mean().rename(columns={"Global_active_power": TARGET})


def load_weather(path):
    """Optional supplementary weather CSV with a datetime column + temperature/humidity columns."""
    w = pd.read_csv(path)
    tcol = next(c for c in w.columns if "time" in c.lower() or "date" in c.lower())
    w[tcol] = pd.to_datetime(w[tcol])
    w = w.set_index(tcol).sort_index()
    keep = [c for c in w.columns if any(k in c.lower() for k in ("temp", "humid"))]
    return w[keep].select_dtypes("number").resample("h").mean()


def load_data(household_path=None, weather_path=None):
    if household_path and os.path.exists(household_path):
        df = load_household(household_path)
        if weather_path and os.path.exists(weather_path):
            df = df.join(load_weather(weather_path), how="left")
        source = "household"
    else:
        df, source = make_synthetic(), "synthetic"
    return df, source


def preprocess(df):
    df = df[~df.index.duplicated()].sort_index()
    df = df.asfreq("h")
    df = df.interpolate(limit_direction="both")
    # clip extreme outliers (1st/99.9th pct) instead of dropping legit peaks
    lo, hi = df[TARGET].quantile([0.001, 0.999])
    df[TARGET] = df[TARGET].clip(lo, hi)
    return df


def add_features(df):
    d = df.copy()
    h = d.index.hour
    d["hour_sin"] = np.sin(2 * np.pi * h / 24)
    d["hour_cos"] = np.cos(2 * np.pi * h / 24)
    d["dow"] = d.index.dayofweek
    d["is_weekend"] = (d.index.dayofweek >= 5).astype(int)
    d["month"] = d.index.month
    d["lag_24h"] = d[TARGET].shift(24)
    d["roll_24h"] = d[TARGET].rolling(24).mean()
    return d.dropna()


def make_windows(arr, target_idx, lookback, horizon=1):
    X, y = [], []
    for i in range(lookback, len(arr) - horizon + 1):
        X.append(arr[i - lookback:i])
        y.append(arr[i + horizon - 1, target_idx])
    return np.array(X, dtype="float32"), np.array(y, dtype="float32")


def prepare(df, lookback=24, horizon=1):
    """Chronological 70/15/15 split; scaler fit on train only (no leakage)."""
    feats = add_features(preprocess(df))
    cols = list(feats.columns)
    n = len(feats)
    i1, i2 = int(n * 0.7), int(n * 0.85)
    scaler = MinMaxScaler().fit(feats.iloc[:i1].values)
    arr = scaler.transform(feats.values)
    t = cols.index(TARGET)
    X, y = make_windows(arr, t, lookback, horizon)
    ts = feats.index[lookback + horizon - 1:]
    s1, s2 = i1 - lookback - horizon + 1, i2 - lookback - horizon + 1
    split = dict(train=(X[:s1], y[:s1]), val=(X[s1:s2], y[s1:s2]), test=(X[s2:], y[s2:]))
    return dict(split=split, scaler=scaler, cols=cols, target_idx=t, index=ts, s2=s2, feats=feats)


def inverse_target(scaler, target_idx, y):
    return y * (scaler.data_max_[target_idx] - scaler.data_min_[target_idx]) + scaler.data_min_[target_idx]
