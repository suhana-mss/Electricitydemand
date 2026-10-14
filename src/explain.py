"""Explainability: SHAP (feature-level), Feature Ablation, and a DAG of variable relationships."""
import numpy as np


def _masked_predict_factory(model, x, background):
    """Return f(Z) where Z is a binary matrix over features; feature j off => replaced with background mean."""
    bg = background.mean(axis=0)  # (lookback, F)

    def f(Z):
        Z = np.atleast_2d(Z)
        batch = np.where(Z[:, None, :] > 0.5, x[None, :, :], bg[None, :, :]).astype("float32")
        return model.predict(batch, verbose=0).ravel()
    return f


def shap_explain(model, x, background, cols):
    """Exact-ish Shapley values over feature channels for one window x of shape (lookback, F)."""
    import shap
    F = len(cols)
    f = _masked_predict_factory(model, x, background)
    expl = shap.KernelExplainer(f, np.zeros((1, F)))
    sv = expl.shap_values(np.ones((1, F)), nsamples=min(2 ** F, 512), silent=True)
    sv = np.array(sv).reshape(-1)
    return dict(zip(cols, sv.tolist()))


def feature_ablation(model, x, background, cols):
    """Change in prediction when each feature channel is replaced by its background mean."""
    bg = background.mean(axis=0)
    base = float(model.predict(x[None], verbose=0).ravel()[0])
    batch = []
    for j in range(len(cols)):
        xa = x.copy()
        xa[:, j] = bg[:, j]
        batch.append(xa)
    preds = model.predict(np.array(batch, dtype="float32"), verbose=0).ravel()
    return {c: base - float(p) for c, p in zip(cols, preds)}


def build_dag(cols):
    """Assumed (not proven causal) relationships among variables, restricted to available columns."""
    import networkx as nx
    edges = [("temperature", "demand_kw"), ("humidity", "demand_kw"), ("temperature", "humidity"),
             ("hour_sin", "demand_kw"), ("hour_cos", "demand_kw"), ("dow", "is_weekend"),
             ("is_weekend", "demand_kw"), ("month", "temperature"), ("month", "demand_kw"),
             ("lag_24h", "demand_kw"), ("roll_24h", "demand_kw")]
    g = nx.DiGraph()
    g.add_edges_from([(a, b) for a, b in edges if a in cols and b in cols])
    assert nx.is_directed_acyclic_graph(g)
    return g


def draw_dag(g, path=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx
    fig, ax = plt.subplots(figsize=(7, 4))
    pos = nx.spring_layout(g, seed=3)
    nx.draw_networkx(g, pos, ax=ax, node_color="#cfe3ff", node_size=1600, font_size=8, arrows=True)
    ax.axis("off")
    if path:
        fig.savefig(path, dpi=120, bbox_inches="tight")
    return fig
