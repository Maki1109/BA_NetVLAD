"""Train BA-NetVLAD with lazy triplet loss on a sequence dataset (poses.csv)."""
import argparse, os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ba_netvlad.model import BANetVLAD
from ba_netvlad.dataset import SequenceDataset, GroupSampler, TrainAugment
from ba_netvlad.loss import lazy_triplet_loss
from torch.utils.data import DataLoader

# Định nghĩa collate_fn thành một hàm chuẩn thay vì dùng lambda
# Hàm này ở ngoài if __name__ == '__main__' để Windows có thể pickle nó
def custom_collate_fn(b):
    return (torch.stack([x[0] for x in b]), torch.tensor([x[1] for x in b]),
            torch.tensor([x[2] for x in b]))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/synth_loop')
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--no_kmeans_init', action='store_true',
                    help='skip the k-means NetVLAD seeding (not recommended)')
    ap.add_argument('--margin', type=float, default=0.1)
    ap.add_argument('--groups', type=int, default=4)
    ap.add_argument('--gimgs', type=int, default=4)
    ap.add_argument('--cell', type=float, default=4.0)
    ap.add_argument('--neg_radius', type=float, default=25.0,
                    help='definitely-negative distance in metres; None disables')
    ap.add_argument('--stride', type=int, default=16)
    ap.add_argument('--freeze_epochs', type=int, default=2)
    ap.add_argument('--no_pretrained', action='store_true')
    ap.add_argument('--out', default='ckpt/ba_netvlad.pt')
    args = ap.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    ds = SequenceDataset(args.data, cell_size=args.cell,
                         transform=TrainAugment(p_blur=0.4))
    model = BANetVLAD(stride=args.stride, pretrained=not args.no_pretrained).to(dev)

    if not args.no_kmeans_init:
        # Seed NetVLAD from real local descriptors BEFORE the first step. Without
        # this the soft-assignment is uniform, every image maps to nearly the same
        # vector, and the triplet loss starts and stays pinned at exactly `margin`.
        import numpy as np, cv2
        from ba_netvlad.dataset import to_tensor_224
        probe = np.linspace(0, len(ds) - 1, min(64, len(ds))).astype(int)
        X = torch.stack([to_tensor_224(cv2.imread(ds.files[i])) for i in probe]).to(dev)
        alpha = model.init_netvlad(X)
        print(f'NetVLAD k-means init done (alpha={alpha:.2f})')

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    xy_all = torch.as_tensor(ds.xy, dtype=torch.float32)

    for ep in range(args.epochs):
        frozen = ep < args.freeze_epochs
        for p_ in model.features.parameters():
            p_.requires_grad = not frozen
        model.train()
        
        # Đã thay lambda bằng hàm custom_collate_fn
        # Nếu Windows vẫn gặp vấn đề bộ nhớ khi load dữ liệu, 
        # bạn có thể đổi num_workers=2 thành num_workers=0
        loader = DataLoader(ds, batch_sampler=GroupSampler(ds, args.groups, args.gimgs, seed=ep),
                            num_workers=2, collate_fn=custom_collate_fn)
        tot, nb, t0 = 0.0, 0, time.time()
        for x, lab, sidx in loader:
            x, lab = x.to(dev), lab.to(dev)
            loss = lazy_triplet_loss(model(x), lab, args.margin,
                                     xy=xy_all[sidx].to(dev),
                                     neg_radius=args.neg_radius)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); nb += 1
            
        print(f"epoch {ep:02d}  loss={tot/max(nb,1):.4f}  batches={nb}  "
              f"({time.time()-t0:.0f}s)  [{'frozen' if frozen else 'full'}|{dev}]")
              
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({'model': model.state_dict(), 'args': vars(args)}, args.out)
    print('saved', args.out)