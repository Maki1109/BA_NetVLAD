"""Runtime (growable) Super Dictionary for live loop-closure detection.

Extends the offline ba_netvlad.dictionary format (descs, node_ids, poses, files)
with a per-entry timestamp, and supports appending new keyframes at runtime
instead of only loading a static .npz built by scripts/03_build_dictionary.py.

The on-disk format stays a superset of the offline one: a dictionary saved here
can be loaded by ba_netvlad.dictionary.load_dictionary (the 'times' field is
simply ignored by that loader), and an offline-built .npz can be loaded here as
a "warm start" -- entries without a stored time are treated as having time
-inf, i.e. always eligible for matching (see `eligible_mask`) until enough new,
timestamped keyframes accumulate.
"""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class DynamicCosineDict:
    dim: int = 128
    pca: object = None          # optional callable: raw_vlad_vec -> whitened 128-D vec
    capacity: int = 4096        # initial preallocation; grows by doubling

    def __post_init__(self):
        self._n = 0
        self._descs = np.zeros((self.capacity, self.dim), np.float32)   # L2-normalized
        self._poses = np.zeros((self.capacity, 2), np.float32)          # (x, y), meters
        self._times = np.full(self.capacity, -np.inf, np.float64)       # seconds
        self._ids = np.zeros(self.capacity, np.int64)
        self._files = []                                                # optional image paths, len == n

    def __len__(self):
        return self._n

    def _grow(self, min_extra=1):
        if self._n + min_extra <= self.capacity:
            return
        new_cap = max(self.capacity * 2, self._n + min_extra)
        for name, fill in (('_descs', 0.0), ('_poses', 0.0), ('_times', -np.inf), ('_ids', 0)):
            arr = getattr(self, name)
            pad_shape = (new_cap - self.capacity,) + arr.shape[1:]
            setattr(self, name, np.concatenate([arr, np.full(pad_shape, fill, arr.dtype)], axis=0))
        self.capacity = new_cap

    def add(self, desc, x, y, t, node_id=None, image_path=''):
        """desc: raw (pre-normalization) descriptor, already through the PCA
        projection if self.pca is set by the caller -- this class only L2-normalizes."""
        self._grow(1)
        d = np.asarray(desc, np.float32)
        d = d / max(np.linalg.norm(d), 1e-8)
        i = self._n
        self._descs[i] = d
        self._poses[i] = (x, y)
        self._times[i] = t
        self._ids[i] = i if node_id is None else node_id
        self._files.append(image_path)
        self._n += 1
        return i

    def last_pose(self):
        if self._n == 0:
            return None
        return self._poses[self._n - 1]

    def eligible_mask(self, now_t, min_time_gap_s):
        """True where a dictionary entry is old enough to be a genuine revisit
        candidate rather than a frame from the path just walked. This is the
        live-robot equivalent of `loop_index_gap` in the offline KITTI replay
        (ba_netvlad/replay.py): both exclude *recent* history along the same
        trajectory, not spatially nearby places -- a real loop closure is
        spatially near the query by definition, so distance must never be an
        exclusion criterion here."""
        if self._n == 0:
            return np.zeros(0, bool)
        return (now_t - self._times[:self._n]) > min_time_gap_s

    def scores(self, query_desc, exclude=None):
        if self._n == 0:
            return np.zeros(0, np.float32)
        q = np.asarray(query_desc, np.float32)
        q = q / max(np.linalg.norm(q), 1e-8)
        s = self._descs[:self._n] @ q
        if exclude is not None:
            s = np.where(exclude, -np.inf, s)
        return s

    def topk(self, query_desc, k=5, exclude=None):
        s = self.scores(query_desc, exclude)
        if len(s) == 0:
            return np.zeros(0, np.int64), np.zeros(0, np.float32)
        kk = min(k, len(s))
        idx = np.argpartition(-s, kk - 1)[:kk]
        idx = idx[np.argsort(-s[idx])]
        return idx, s[idx]

    def second_best_other_place(self, scores, top1_idx, margin_window_m):
        """Best score among entries at least `margin_window_m` away (Euclidean,
        in the same odom/map frame as the stored poses) from the top-1 match's
        stored pose. Mirrors ba_netvlad.matcher.second_best_other_place, but
        uses real distance instead of a frame-index window since a live
        dictionary has no meaningful frame ordering guarantee."""
        if self._n == 0:
            return -1.0
        p1 = self._poses[top1_idx]
        d = np.linalg.norm(self._poses[:self._n] - p1[None, :], axis=1)
        other = d > margin_window_m
        if not other.any():
            return -1.0
        s2 = float(np.max(np.where(other, scores, -np.inf)))
        return s2 if np.isfinite(s2) else -1.0

    def pose(self, idx):
        return self._poses[idx]

    def image_path(self, idx):
        return self._files[idx]

    def save_npz(self, path):
        np.savez(
            path,
            descs=self._descs[:self._n],
            node_ids=self._ids[:self._n],
            poses=self._poses[:self._n],
            times=self._times[:self._n],
            files=np.array(self._files, dtype=object),
        )

    @classmethod
    def load_npz(cls, path, dim=128, pca=None):
        z = np.load(path, allow_pickle=True)
        n = len(z['descs'])
        obj = cls(dim=dim, pca=pca, capacity=max(n, 1))
        obj._n = n
        obj._descs[:n] = z['descs']
        obj._poses[:n] = z['poses']
        obj._ids[:n] = z['node_ids']
        # Offline .npz files (built by scripts/03_build_dictionary.py) have no
        # per-entry time -> treat as "always eligible" (-inf) rather than
        # guessing; only newly-added live keyframes carry a real timestamp.
        obj._times[:n] = z['times'] if 'times' in z.files else -np.inf
        obj._files = list(z['files']) if 'files' in z.files else [''] * n
        return obj
