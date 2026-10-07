"""ORB local-feature branch: geometric verification of NetVLAD loop candidates.

Sharp frames keep the cheap local-feature path: a NetVLAD candidate is accepted
only if ORB matches between the query and the candidate keyframe survive a
RANSAC epipolar check. Blurry frames skip this (blur removes the corners ORB
needs), so they rely on the global descriptor and temporal voting alone.
"""
import cv2
import numpy as np

class ORBVerifier:
    def __init__(self, n_features=1000, ratio=0.8, ransac_thresh=3.0,
                 min_inliers=15, max_side=640):
        self.orb = cv2.ORB_create(nfeatures=n_features)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.ratio, self.ransac_thresh = ratio, ransac_thresh
        self.min_inliers, self.max_side = min_inliers, max_side
        self._cache = {}            # reference path -> (keypoint xy, descriptors)

    def extract(self, img_bgr):
        g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
        s = self.max_side / max(g.shape)
        if s < 1.0:                              # bound cost; same scale for q and ref
            g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        kp, des = self.orb.detectAndCompute(g, None)
        if des is None or len(kp) == 0:
            return np.zeros((0, 2), np.float32), None
        return np.array([k.pt for k in kp], np.float32), des

    def reference(self, path):
        if path not in self._cache:
            img = cv2.imread(str(path))
            self._cache[path] = (self.extract(img) if img is not None
                                 else (np.zeros((0, 2), np.float32), None))
        return self._cache[path]

    def inliers(self, q, r):
        """q, r = (xy, desc). Returns the number of RANSAC inlier matches."""
        (xq, dq), (xr, dr) = q, r
        if dq is None or dr is None or len(dq) < 8 or len(dr) < 8:
            return 0
        pairs = self.bf.knnMatch(dq, dr, k=2)
        good = [m[0] for m in pairs if len(m) == 2 and m[0].distance < self.ratio * m[1].distance]
        if len(good) < 8:
            return 0                             # too few for a fundamental matrix
        pq = xq[[m.queryIdx for m in good]]
        pr = xr[[m.trainIdx for m in good]]
        _, mask = cv2.findFundamentalMat(pq, pr, cv2.FM_RANSAC, self.ransac_thresh, 0.99)
        return int(mask.sum()) if mask is not None else 0

    def verify_candidates(self, q_img, ref_paths):
        """Inlier count for each candidate path. The query is extracted once."""
        q = self.extract(q_img)
        return [self.inliers(q, self.reference(p)) for p in ref_paths], len(q[0])
