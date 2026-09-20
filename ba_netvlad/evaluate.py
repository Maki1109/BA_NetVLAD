
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
    """Sweep top-1 similarity threshold -> PR curve for LCD."""
    s1 = scores.max(axis=1)
    y = gt_pos.any(axis=1)
    ths = np.linspace(s1.min(), s1.max(), steps)
    P, R, T = [], [], []
    for t in ths:
        pred = s1 >= t
        tp = (pred & y).sum(); fp = (pred & ~y).sum(); fn = ((~pred) & y).sum()
        if tp + fp == 0: continue
        P.append(tp / (tp + fp)); R.append(tp / (tp + fn + 1e-9)); T.append(t)
    return np.array(P), np.array(R), np.array(T)
