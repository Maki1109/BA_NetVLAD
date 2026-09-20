
"""Standard VPR benchmark (folder split with GT): overall Recall@N + PR curve plot."""
import argparse, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ba_netvlad.model import load_ckpt
from ba_netvlad.dataset import FolderVPRDataset
from ba_netvlad.dictionary import encode_dataset
from ba_netvlad.evaluate import recall_at_n, precision_recall_curve
# Force single-process DataLoader on Windows: spawn-based multiprocessing needs
# every arg picklable, and FolderVPRDataset's lambda transform + module-level
# Ref class are not, so num_workers>0 crashes with a PicklingError.
import torch.utils.data.dataloader as dataloader
_orig_init = dataloader.DataLoader.__init__
def _single_proc_init(self, dataset, *a, **kw):
    kw['num_workers'] = 0
    _orig_init(self, dataset, *a, **kw)
dataloader.DataLoader.__init__ = _single_proc_init

def identity_transform(im):
    return im

class Ref(torch.utils.data.Dataset):
    def __init__(self, ds): self.ds = ds
    def __len__(self): return len(self.ds.ref)
    def __getitem__(self, i): return self.ds.get_ref(i)

class Query(torch.utils.data.Dataset):
    # FolderVPRDataset.__getitem__ returns (tensor, idx); encode_dataset's
    # collate_fn does torch.stack(batch) expecting bare tensors, same reason
    # script 03 wraps SequenceDataset before encoding.
    def __init__(self, ds): self.ds = ds
    def __len__(self): return len(self.ds)
    def __getitem__(self, i): return self.ds[i][0]

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='folder with ref/, query/, gt.npy or poses csvs')
    ap.add_argument('--model', default='ckpt/ba_netvlad.pt')
    ap.add_argument('--radius', type=float, default=10.0)
    args = ap.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    ds = FolderVPRDataset(args.root, radius=args.radius, transform=identity_transform)
    model = load_ckpt(args.model, stride=16).to(dev)
    q = encode_dataset(model, Query(ds), dev); r = encode_dataset(model, Ref(ds), dev)
    q /= np.linalg.norm(q, axis=1, keepdims=True); r /= np.linalg.norm(r, axis=1, keepdims=True)
    print(recall_at_n(q @ r.T, ds.gt))
    P, R, T = precision_recall_curve(q @ r.T, ds.gt)
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.plot(R, P); plt.xlabel('recall'); plt.ylabel('precision'); plt.grid()
        os.makedirs('results', exist_ok=True)
        out_png = f"results/pr_curve_{os.path.basename(os.path.normpath(args.root))}.png"
        plt.savefig(out_png, dpi=150)
        print(f'wrote {out_png}')
    except Exception as e:
        print('plot skipped:', e)
