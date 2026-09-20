
"""Lazy triplet loss with in-batch hard positive / hard negative mining."""
import torch
import torch.nn.functional as F

def pairwise_cosine_dist(a, b=None):
    if b is None: b = a
    a = F.normalize(a, p=2, dim=1); b = F.normalize(b, p=2, dim=1)
    return 1.0 - a @ b.t()

def lazy_triplet_loss(emb, labels, margin=0.1, xy=None, neg_radius=None):
    """emb [N,D], labels [N]. For each anchor:
    L = max(0, margin + max_pos D(anchor,pos) - min_neg D(anchor,neg)).
    Anchors without an in-batch positive are skipped.

    `neg = labels != labels[i]` alone makes the frame one metre across a cell
    boundary a negative while the frame one metre inside it is a positive. The
    hardest negative is then a near-duplicate of the anchor, dn ~= dp, and the
    loss cannot fall below `margin` however long you train -- the flat 0.1000
    plateau. Pass `xy` (poses) and `neg_radius` to use NetVLAD's definitely-
    negative rule: only places at least `neg_radius` metres away may be negatives.
    """
    D = pairwise_cosine_dist(emb)
    N = emb.size(0)
    if xy is not None and neg_radius is not None:
        far = torch.cdist(xy, xy) >= neg_radius
    else:
        far = None
    total, cnt = 0.0, 0
    for i in range(N):
        pos = (labels == labels[i]); pos[i] = False
        if not pos.any():
            continue
        neg = (labels != labels[i])
        if far is not None:
            neg = neg & far[i]
        if not neg.any():
            continue
        dp = D[i][pos].max()
        dn = D[i][neg].min()
        total = total + torch.clamp(margin + dp - dn, min=0.0)
        cnt += 1
    return total / max(cnt, 1)
