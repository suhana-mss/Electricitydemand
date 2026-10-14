"""Train the LSTM and save artifacts.  Usage:
   python train.py [--household data/household_power_consumption.txt] [--weather data/weather.csv] [--epochs 20]
Without dataset files it falls back to a synthetic dataset so the pipeline runs end-to-end."""
import argparse
import json
import os
import pickle

from src.data import load_data, prepare
from src.model import build_lstm, train, evaluate

ap = argparse.ArgumentParser()
ap.add_argument("--household", default="data/household_power_consumption.txt")
ap.add_argument("--weather", default="data/weather.csv")
ap.add_argument("--epochs", type=int, default=20)
ap.add_argument("--lookback", type=int, default=24)
a = ap.parse_args()

os.makedirs("artifacts", exist_ok=True)
df, source = load_data(a.household, a.weather)
print(f"Data source: {source}, rows: {len(df)}")
d = prepare(df, lookback=a.lookback)
Xtr = d["split"]["train"][0]
model = build_lstm(a.lookback, Xtr.shape[2])
train(model, d["split"], epochs=a.epochs)
metrics, _, _ = evaluate(model, d["split"], d["scaler"], d["target_idx"])
print("Test metrics:", metrics)

model.save("artifacts/lstm.keras")
with open("artifacts/prep.pkl", "wb") as f:
    pickle.dump({k: d[k] for k in ("scaler", "cols", "target_idx", "index", "s2")} | {"source": source, "lookback": a.lookback}, f)
d["feats"].to_pickle("artifacts/feats.pkl")
with open("artifacts/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("Saved to artifacts/")
