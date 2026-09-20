
"""Datasets: sequence-with-poses (loop replay) and folder-based VPR split (Recall@N)."""
import csv
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, Sampler
from .augmentation import motion_blur

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], np.float32)

def to_tensor_224(img_bgr, size=224, augment=None):
    if augment is not None:
        img_bgr = augment(img_bgr)
    img = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(np.ascontiguousarray(img[:, :, ::-1]).transpose(2, 0, 1)).float() / 255.0
    for c in range(3):
        t[c] = (t[c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
    return t

class TrainAugment:
    """Random crop + hflip + motion blur (p_blur) -- the blur branch must be trained."""
    def __init__(self, blur_lengths=(8, 16, 24, 32), p_blur=0.4, size=256, crop=224):
        self.blur_lengths = blur_lengths; self.p_blur = p_blur
        self.size, self.crop = size, crop

    def __call__(self, img):
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
        x0 = np.random.randint(0, self.size - self.crop + 1)
        y0 = np.random.randint(0, self.size - self.crop + 1)
        img = img[y0:y0+self.crop, x0:x0+self.crop]
        if np.random.rand() < 0.5:
            img = img[:, ::-1]
        if np.random.rand() < self.p_blur:
            L = self.blur_lengths[np.random.randint(len(self.blur_lengths))]
            img = motion_blur(img, L, angle_deg=np.random.uniform(0, 180))
        return img

class SequenceDataset(Dataset):
    """Frames + poses.csv (columns: timestamp,filename,x,y[,z,qx,qy,qz,qw]).
    node_id = spatial grid cell key so weakly-supervised triplets can be formed."""
    def __init__(self, root, cell_size=4.0, transform=None):
        self.root, self.transform = root, (transform or (lambda im: im))
        rows = list(csv.DictReader(open(f"{root}/poses.csv")))
        rows.sort(key=lambda r: float(r['timestamp']))
        self.ts   = np.array([float(r['timestamp']) for r in rows])
        self.files= [f"{root}/{r['filename']}" for r in rows]
        self.xy   = np.array([[float(r['x']), float(r['y'])] for r in rows])
        self.cell = cell_size
        keys = {}
        self.node_ids = np.empty(len(rows), np.int64)
        for i, (x, y) in enumerate(self.xy):
            key = (int(np.floor(x / cell_size)), int(np.floor(y / cell_size)))
            self.node_ids[i] = keys.setdefault(key, len(keys))

    def __len__(self): return len(self.files)

    def __getitem__(self, i):
        img = cv2.imread(self.files[i])
        return to_tensor_224(self.transform(img)), self.node_ids[i], i

class GroupSampler(Sampler):
    """Yields batches of B groups x G images from the same node (lazy-triplet mining)."""
    def __init__(self, ds, groups_per_batch=4, imgs_per_group=4, seed=0):
        self.ds, self.B, self.G, self.rng = ds, groups_per_batch, imgs_per_group, np.random.RandomState(seed)
        self.by_node = {}
        for i, n in enumerate(ds.node_ids):
            self.by_node.setdefault(int(n), []).append(i)

    def __iter__(self):
        nodes = np.array([n for n, idx in self.by_node.items() if len(idx) >= 2])
        order = self.rng.permutation(len(nodes))
        for s in range(0, len(order), self.B):
            batch = []
            for n in nodes[order[s:s+self.B]]:
                idx = self.by_node[n]
                batch += list(self.rng.choice(idx, size=self.G, replace=len(idx) < self.G))
            if batch:
                yield batch

    def __len__(self):
        nodes = [n for n, idx in self.by_node.items() if len(idx) >= 2]
        return len(nodes) // self.B

class FolderVPRDataset(Dataset):
    """Standard VPR split: root/ref/*.jpg, root/query/*.jpg.
    GT via poses_{ref,query}.csv (x,y) radius, or root/gt.npy [Q,R] bool."""
    def __init__(self, root, radius=10.0, transform=None):
        import glob, os
        self.root, self.transform = root, (transform or (lambda im: im))
        self.ref = sorted(glob.glob(f"{root}/ref/*.jpg"))
        self.qry = sorted(glob.glob(f"{root}/query/*.jpg"))
        if os.path.exists(f"{root}/gt.npy"):
            self.gt = np.load(f"{root}/gt.npy").astype(bool)
        else:
            pr = np.array([[float(r['x']), float(r['y'])] for r in
                           csv.DictReader(open(f"{root}/poses_ref.csv"))])
            pq = np.array([[float(r['x']), float(r['y'])] for r in
                           csv.DictReader(open(f"{root}/poses_query.csv"))])
            d = np.linalg.norm(pq[:, None, :] - pr[None, :, :], axis=2)
            self.gt = d <= radius

    def __len__(self): return len(self.qry)

    def _load(self, f):
        return to_tensor_224(self.transform(cv2.imread(f)))

    def get_ref(self, i): return self._load(self.ref[i])
    def __getitem__(self, i): return self._load(self.qry[i]), i
