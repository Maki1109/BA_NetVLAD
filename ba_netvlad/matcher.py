
"""Step 4: cosine matching against the Super Dictionary + temporal confirmation."""
import numpy as np
from collections import deque

class CosineDict:
    def __init__(self, dict_data):
        self.descs = dict_data['descs']            # pre-L2-normalized [M, D]
        self.node_ids = dict_data['node_ids']; self.poses = dict_data['poses']

    def scores(self, q, exclude=None):
        """Full similarity vector over the dictionary. `exclude` is a boolean mask
        of entries that are ineligible (the temporal window around the live frame);
        they are driven to -inf so they can never win retrieval."""
        q = q / max(np.linalg.norm(q), 1e-8)
        s = self.descs @ q
        if exclude is not None:
            s = np.where(exclude, -np.inf, s)
        return s

    def topk(self, q, k=5, exclude=None):
        s = self.scores(q, exclude)
        kk = min(k, len(s))
        idx = np.argpartition(-s, kk - 1)[:kk]
        idx = idx[np.argsort(-s[idx])]
        return idx, s[idx]

class TemporalVoter:
    """Vote ring buffer: require >= m of last r frames agreeing on a node.

    The buffer must be advanced on EVERY frame, including frames with no
    candidate (push NO_VOTE), otherwise 'last r frames' silently means
    'last r candidate frames' and the window stretches over arbitrary time.
    """
    NO_VOTE = -1

    def __init__(self, r=5, m=3):
        self.buf = deque(maxlen=r); self.m = m
    def vote(self, node_id):
        node_id = int(node_id)
        self.buf.append(node_id)
        if node_id == self.NO_VOTE:
            return False
        return sum(1 for v in self.buf if v == node_id) >= self.m
    def reset(self):
        self.buf.clear()

def second_best_other_place(scores, j1, margin_window=25, ref_index=None):
    """Best score from a DIFFERENT place than the top-1 match.

    scores[1] from a sequence dictionary is almost always ref j1+-1 -- the very
    next image of the same place -- so `s1 - scores[1]` measures frame-to-frame
    jitter (~1e-2), never place distinctiveness, and no sane delta can pass it.
    Excluding a +-margin_window band around j1 turns the test back into the
    intended 'best place beats every other place by delta'.
    """
    pos = np.arange(len(scores)) if ref_index is None else np.asarray(ref_index)
    other = np.abs(pos - pos[j1]) > margin_window
    if not other.any():
        return -1.0
    s2 = float(np.max(np.where(other, scores, -np.inf)))
    return s2 if np.isfinite(s2) else -1.0

def confirm_loop(idx, scores, voter, tau=0.75, delta=0.05, topk_node_map=None,
                 all_scores=None, margin_window=25, ref_index=None):
    """Returns (accepted, info). scores are cosine similarities for idx (desc order).

    Pass `all_scores` (the full dictionary vector, already temporally masked) to
    get the correct best-other-place margin; without it the legacy adjacent-frame
    margin is used and `delta` is effectively unsatisfiable on sequence data.
    """
    if len(scores) == 0 or not np.isfinite(scores[0]):
        voter.vote(TemporalVoter.NO_VOTE)
        return False, {}
    s1, j1 = float(scores[0]), int(idx[0])
    if all_scores is not None:
        s2 = second_best_other_place(all_scores, j1, margin_window, ref_index)
    else:
        s2 = float(scores[1]) if len(scores) > 1 else 0.0
    node = topk_node_map[j1] if topk_node_map is not None else j1
    cand = (s1 >= tau) and ((s1 - s2) >= delta)
    # vote unconditionally so the ring buffer tracks real time, not candidates
    voted = voter.vote(node if cand else TemporalVoter.NO_VOTE)
    return (cand and voted), {'node': int(node), 'match_idx': j1, 's1': s1, 's2': s2,
                              'margin': s1 - s2}
