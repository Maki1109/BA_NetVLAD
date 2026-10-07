
"""Metrics: Recall@N (VPR) and loop-closure precision/recall."""
import numpy as np

def recall_at_n(scores, gt_pos, ns=(1, 5, 10)):
    """scores [Q, R] cosine sims; gt_pos [Q, R] bool positives."""
    out = {}
    valid = gt_pos.any(axis=1)
    s, g = scores[valid], gt_pos[valid]
    for n in ns:
        n = min(n, s.shape[1])
        top = np.argpartition(-s, n - 1, axis=1)[:, :n]
        hit = g[np.arange(len(g))[:, None], top].any(axis=1)
        out[f'R@{n}'] = float(hit.mean())
    return out

def precision_recall_curve(scores, gt_pos, steps=100):
    """Sweep top-1 similarity threshold -> PR curve for LCD.

    A query is a true positive only if it is accepted AND its top-1 match is a
    ground-truth positive. The previous version scored "does this query have any
    positive in the reference set" instead, so every accepted query with a
    positive existing counted as correct regardless of what was retrieved and
    the curve was flat at precision 1.0.
    """
    top1 = scores.argmax(axis=1)
    s1 = scores[np.arange(len(scores)), top1]
    has_gt = gt_pos.any(axis=1)                       # query is loop-capable
    correct = gt_pos[np.arange(len(gt_pos)), top1]    # retrieved the right place
    n_pos = max(int(has_gt.sum()), 1)
    ths = np.linspace(s1.min(), s1.max(), steps)
    P, R, T = [], [], []
    for t in ths:
        pred = s1 >= t
        tp = int((pred & correct).sum())
        n_pred = int(pred.sum())
        if n_pred == 0: continue
        P.append(tp / n_pred); R.append(tp / n_pos); T.append(t)
    return np.array(P), np.array(R), np.array(T)

def average_precision(scores, gt_pos, steps=200):
    """AP = area under the PR curve (the metric used by most LCD papers)."""
    P, R, _ = precision_recall_curve(scores, gt_pos, steps)
    if len(R) == 0: return 0.0
    o = np.argsort(R)
    return float(np.trapz(P[o], R[o]))

def max_recall_at_full_precision(scores, gt_pos, steps=200):
    P, R, _ = precision_recall_curve(scores, gt_pos, steps)
    ok = P >= 1.0 - 1e-9
    return float(R[ok].max()) if ok.any() else 0.0
