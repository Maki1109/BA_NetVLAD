
"""Super Dictionary: encode reference frames -> memory-mapped descriptor store."""
import numpy as np
import torch
from torch.utils.data import DataLoader

@torch.no_grad()
def encode_dataset(model, ds, device='cpu', batch_size=16, num_workers=2):
    model.eval()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, collate_fn=lambda b: torch.stack(b))
    descs = []
    for x in loader:
        descs.append(model(x.to(device)).cpu())
    return torch.cat(descs).numpy().astype(np.float32)

@torch.no_grad()
def encode_vlad(model, ds, device='cpu', batch_size=16):
    """Pre-head NetVLAD vectors (K*dim), for fitting PCA+whitening."""
    model.eval()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=0, collate_fn=lambda b: torch.stack(b))
    out = []
    for x in loader:
        out.append(model(x.to(device), return_vlad=True)[1].cpu())
    return torch.cat(out).numpy().astype(np.float32)

def fit_pca_whiten(X, out_dim=128, eps=1e-6):
    """PCA + whitening, as specified in BA-NetVLAD step 3.

    Whitening is not cosmetic here: VLAD vectors of one sequence share a huge
    common mode (pairwise cosine ~0.88 on KITTI00). Equalising the eigenvalue
    spectrum removes it, which is what restores usable dynamic range to the
    cosine scores that `tau` and `delta` are thresholds on.
    """
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)
    mu = X.mean(0)
    _, s, Vt = np.linalg.svd(X - mu, full_matrices=False)
    k = min(out_dim, len(s))
    W = (Vt[:k].T / np.maximum(s[:k] / np.sqrt(max(len(X) - 1, 1)), eps))
    return mu.astype(np.float32), W.astype(np.float32)

def apply_pca_whiten(X, mu, W):
    single = (X.ndim == 1)
    X = np.atleast_2d(X).astype(np.float32)
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)
    Y = (X - mu) @ W
    Y = Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-8)
    return Y[0] if single else Y

def build_dictionary(model, ds, out_path, node_ids=None, poses=None, files=None,
                     device='cpu', batch_size=16, pca=False, out_dim=128,
                     src_index=None):
    if pca:
        vlad = encode_vlad(model, ds, device, batch_size)
        mu, W = fit_pca_whiten(vlad, out_dim)
        descs = apply_pca_whiten(vlad, mu, W)
        extra = {'pca_mu': mu, 'pca_W': W}
    else:
        descs = encode_dataset(model, ds, device, batch_size)
        extra = {}
    np.savez(out_path, descs=descs,
             node_ids=node_ids if node_ids is not None else np.arange(len(descs)),
             poses=poses if poses is not None else np.zeros((len(descs), 2), np.float32),
             files=np.array(files if files is not None else ['']*len(descs)),
             # original sequence index of each entry -- the temporal exclusion
             # window needs it once refs are keyframe/blur subsampled
             src_index=(np.asarray(src_index) if src_index is not None
                        else np.arange(len(descs))),
             **extra)
    return descs

def load_dictionary(path):
    z = np.load(path, allow_pickle=True)
    d = z['descs'].astype(np.float32)
    n = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-8)
    out = {'descs': n, 'raw': d, 'node_ids': z['node_ids'],
           'poses': z['poses'], 'files': z['files'], 'pca': None,
           'src_index': (z['src_index'] if 'src_index' in z.files
                         else np.arange(len(d)))}
    if 'pca_mu' in z.files:
        mu, W = z['pca_mu'], z['pca_W']
        out['pca'] = lambda v, mu=mu, W=W: apply_pca_whiten(v, mu, W)
    return out
