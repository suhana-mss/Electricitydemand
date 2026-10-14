# GenAI-Enhanced Electricity Demand Forecasting

LSTM forecasting + Explainable AI (SHAP, Feature Ablation, DAG) + GenAI energy-management recommendations + Streamlit dashboard.

## Pipeline
1. `src/data.py`: load, clean, feature engineering (hour sin/cos, weekday/weekend, month, lag/rolling), chronological train/val/test split (scaler fit on train only).
2. `src/model.py`: 2-layer LSTM (Keras), MAE / RMSE / MAPE / R² evaluation (MAPE ignores near-zero targets).
3. `src/explain.py`: SHAP (Shapley over feature channels), Feature Ablation, DAG of variable relationships.
4. `src/recommend.py`: LLM recommendations constrained to forecast/XAI facts (set `ANTHROPIC_API_KEY`); rule-based fallback otherwise.
5. `app.py`: dashboard with forecast, explanations, recommendations, feedback.

## Run
```bash
pip install -r requirements.txt
# Optional real data (Kaggle):
#   data/household_power_consumption.txt  (Individual Household Electric Power Consumption)
#   data/weather.csv                      (Electricity Consumption Based on Weather Data)
python train.py --epochs 20      # falls back to synthetic data if files are missing
streamlit run app.py
```
Optional: `export ANTHROPIC_API_KEY=...` (and `LLM_MODEL=...`) for LLM-written recommendations.

## Notes
- The DAG is an interpretability aid, not proof of causality.
- With the weather CSV, make sure its timestamps overlap the household data; otherwise weather columns are interpolated/empty.
