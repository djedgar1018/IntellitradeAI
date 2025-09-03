# ranking/scorecard.py
from .criteria import CRITERIA, DEFAULT_WEIGHTS

def score(tool_metrics: dict, weights: dict | None = None):
    w = weights or DEFAULT_WEIGHTS
    return sum(w[k]*tool_metrics.get(k,0) for k in CRITERIA)
