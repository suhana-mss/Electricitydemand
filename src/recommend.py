"""GenAI recommendation module. Uses an LLM when ANTHROPIC_API_KEY is set, else a rule-based fallback.
The prompt is constrained to the supplied forecast/XAI facts so the model cannot invent numbers."""
import json
import os

FRIENDLY = {
    "lag_24h": "usage at the same time yesterday",
    "roll_24h": "average usage over the last 24 hours",
    "hour_sin": "time of day", "hour_cos": "time of day",
    "temperature": "outdoor temperature", "humidity": "humidity",
    "is_weekend": "weekend vs weekday", "dow": "day of week",
    "month": "season", "demand_kw": "recent consumption",
}


def top_factors(shap_vals, k=3):
    merged = {}
    for name, v in shap_vals.items():
        key = FRIENDLY.get(name, name)
        merged[key] = merged.get(key, 0.0) + v
    return sorted(merged.items(), key=lambda kv: abs(kv[1]), reverse=True)[:k]


def build_context(forecast_kw, recent_avg_kw, peak_hour, shap_vals, ablation):
    return dict(
        forecast_kw=round(float(forecast_kw), 3),
        recent_avg_kw=round(float(recent_avg_kw), 3),
        change_vs_recent_pct=round(100 * (forecast_kw - recent_avg_kw) / max(recent_avg_kw, 1e-6), 1),
        expected_peak_hour=int(peak_hour),
        top_factors=[{"factor": n, "effect_kw": round(v, 3), "direction": "raises" if v > 0 else "lowers"}
                     for n, v in top_factors(shap_vals)],
    )


def rule_based(ctx):
    recs = []
    ch = ctx["change_vs_recent_pct"]
    if ch > 10:
        recs.append(f"Demand is forecast {ch}% above your recent average ({ctx['forecast_kw']} kW). Postpone non-urgent appliance use.")
    elif ch < -10:
        recs.append(f"Demand is forecast {abs(ch)}% below your recent average, a good window for flexible tasks such as laundry.")
    else:
        recs.append(f"Demand ({ctx['forecast_kw']} kW) is close to your recent average; keep current habits.")
    recs.append(f"Expected peak around {ctx['expected_peak_hour']}:00. Shift dishwashers, washing machines and EV charging outside this hour.")
    for f in ctx["top_factors"][:2]:
        if f["factor"] == "outdoor temperature" and f["direction"] == "raises":
            recs.append("Temperature is pushing demand up: raise the AC setpoint by 1-2 °C or pre-cool earlier.")
        elif f["direction"] == "raises":
            recs.append(f"'{f['factor']}' is raising demand; review related usage.")
    return recs


def generate(ctx, use_llm=True):
    key = os.environ.get("ANTHROPIC_API_KEY")
    if use_llm and key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=key)
            prompt = ("You are an energy advisor. Using ONLY the JSON facts below (do not invent numbers), "
                      "write 3-4 short, plain-language, actionable energy-saving recommendations for a household. "
                      "One per line, no preamble.\n\n" + json.dumps(ctx))
            msg = client.messages.create(model=os.environ.get("LLM_MODEL", "claude-sonnet-5-5"),
                                         max_tokens=400, messages=[{"role": "user", "content": prompt}])
            lines = [l.strip("-• ").strip() for l in msg.content[0].text.splitlines() if l.strip()]
            return lines, "llm"
        except Exception as e:  # fall back gracefully
            return rule_based(ctx), f"rule-based (LLM failed: {type(e).__name__})"
    return rule_based(ctx), "rule-based"
