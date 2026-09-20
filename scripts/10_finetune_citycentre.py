
"""Fine-tune BA-NetVLAD on CityCentre for the Bước 7 benchmark.

Same rationale as scripts/09_finetune_gpw.py: CityCentre has no poses.csv, but
the ref/query split built from the odd/even synchronized-camera convention
(see the CityCentre section of Bước 7) gives an exact 1:1 place correspondence
-- ref/000i.jpg and query/000i.jpg are the same physical location. We use that
index as the triplet node_id.

Caveat (report alongside any resulting numbers): only 2 views per place (vs 3
for GPW), and fine-tune/eval share the same 1237 places -- this measures domain
adaptation capacity, not held-out generalization, same as GPW.
"""
import argparse, glob, os, sys, time
import numpy as np, cv2, torch
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ba_netvlad.model import load_ckpt
from ba_netvlad.dataset import to_tensor_224, TrainAugment, GroupSampler
from ba_netvlad.loss import lazy_triplet_loss

import torch.utils.data.dataloader as dataloader
_orig_init = dataloader.DataLoader.__init__
def _single_proc_init(self, dataset, *a, **kw):
    kw['num_workers'] = 0
    _orig_init(self, dataset, *a, **kw)
dataloader.DataLoader.__init__ = _single_proc_init


class CityCentreDataset(Dataset):
    """node_id = place index (0..N-1), shared across ref/ and query/."""
    def __init__(self, root, augment=None):
        ref = sorted(glob.glob(f"{root}/ref/*.jpg"))
        qry = sorted(glob.glob(f"{root}/query/*.jpg"))
        assert len(ref) == len(qry), f"ref/query count mismatch: {len(ref)} vs {len(qry)}"
        self.files = ref + qry
        self.node_ids = np.array(list(range(len(ref))) + list(range(len(qry))), np.int64)
        self.augment = augment

    def __len__(self): return len(self.files)

    def __getitem__(self, i):
        img = cv2.imread(self.files[i])
        return to_tensor_224(img, augment=self.augment), self.node_ids[i], i


def collate(b):
    return torch.stack([x[0] for x in b]), torch.tensor([x[1] for x in b])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/citycentre')
    ap.add_argument('--init_ckpt', default='ckpt/ba_kitti00.pt')
    ap.add_argument('--epochs', type=int, default=30,
                    help='1237 places / groups=16 -> ~77 batches/epoch, so 30 epochs '
                         'already matches the ~2300 steps that worked for GPW')
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--margin', type=float, default=0.1)
    ap.add_argument('--groups', type=int, default=16)
    ap.add_argument('--gimgs', type=int, default=2, help='only 2 views/place here')
    ap.add_argument('--out', default='ckpt/ba_citycentre_finetuned.pt')
    args = ap.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    ds = CityCentreDataset(args.data, augment=TrainAugment(p_blur=0.0, size=256, crop=224))
    model = load_ckpt(args.init_ckpt, stride=16, pretrained=False).to(dev)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    for ep in range(args.epochs):
        loader = DataLoader(ds, batch_sampler=GroupSampler(ds, args.groups, args.gimgs, seed=ep),
                            collate_fn=collate)
        tot, nb, t0 = 0.0, 0, time.time()
        for x, lab in loader:
            x, lab = x.to(dev), lab.to(dev)
            loss = lazy_triplet_loss(model(x), lab, args.margin)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); nb += 1
        print(f"epoch {ep:02d}  loss={tot/max(nb,1):.4f}  batches={nb}  ({time.time()-t0:.0f}s)")

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    torch.save({'model': model.state_dict(), 'args': vars(args)}, args.out)
    print('saved', args.out)
