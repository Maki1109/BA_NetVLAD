
"""Fine-tune BA-NetVLAD on Gardens Point Walking (GPW) for the Bước 7 benchmark.

GPW has no poses.csv (no metric x,y), so SequenceDataset/GroupSampler (pose-cell
grouping) do not apply directly. What GPW DOES give us is an exact 1:1 place
correspondence by filename index across day_left/day_right/night_right (frame i
in each folder is the same physical place). We use that correspondence directly
as the triplet "node_id" instead of a pose cell.

Caveat (report this alongside any resulting numbers): fine-tuning and evaluating
on the same 200 places is not a held-out generalization test -- GPW's day/night
200-frame subset is too small to hold out a separate split and still learn
anything. This measures whether the architecture CAN close a domain gap once
adapted, not out-of-domain generalization. Treat results accordingly.
"""
import argparse, glob, os, sys, time
import numpy as np, cv2, torch
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ba_netvlad.model import BANetVLAD, load_ckpt
from ba_netvlad.dataset import to_tensor_224, TrainAugment, GroupSampler
from ba_netvlad.loss import lazy_triplet_loss

# Windows-safe DataLoader (spawn needs picklable args; keep workers in-process).
import torch.utils.data.dataloader as dataloader
_orig_init = dataloader.DataLoader.__init__
def _single_proc_init(self, dataset, *a, **kw):
    kw['num_workers'] = 0
    _orig_init(self, dataset, *a, **kw)
dataloader.DataLoader.__init__ = _single_proc_init


class GPWDataset(Dataset):
    """node_id = place index (0..N-1), shared across day_left/day_right/night_right."""
    def __init__(self, root, subfolders=('day_left', 'day_right', 'night_right'),
                augment=None):
        self.files, self.node_ids = [], []
        for sf in subfolders:
            fs = sorted(glob.glob(f"{root}/{sf}/*.jpg"))
            self.files += fs
            self.node_ids += list(range(len(fs)))
        self.node_ids = np.array(self.node_ids, np.int64)
        self.augment = augment

    def __len__(self): return len(self.files)

    def __getitem__(self, i):
        img = cv2.imread(self.files[i])
        return to_tensor_224(img, augment=self.augment), self.node_ids[i], i


def collate(b):
    return torch.stack([x[0] for x in b]), torch.tensor([x[1] for x in b])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/GardensPoint')
    ap.add_argument('--init_ckpt', default='ckpt/ba_kitti00.pt',
                    help='start from the KITTI00-trained model, not from scratch')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--lr', type=float, default=3e-5,
                    help='low LR: adapting an already-converged model on ~600 images')
    ap.add_argument('--margin', type=float, default=0.1)
    ap.add_argument('--groups', type=int, default=8)
    ap.add_argument('--gimgs', type=int, default=3)
    ap.add_argument('--out', default='ckpt/ba_gpw_finetuned.pt')
    args = ap.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    ds = GPWDataset(args.data, augment=TrainAugment(p_blur=0.0, size=256, crop=224))
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
