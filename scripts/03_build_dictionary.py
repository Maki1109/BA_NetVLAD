"""Encode reference split -> Super Dictionary (npz)."""
import argparse, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ba_netvlad.model import load_ckpt
from ba_netvlad.dataset import SequenceDataset
from ba_netvlad.dictionary import build_dictionary
# Bổ sung monkey-patch để ép DataLoader chạy đơn luồng, tránh lỗi pickle lambda bên trong thư viện
import torch.utils.data.dataloader as dataloader
original_init = dataloader.DataLoader.__init__

def custom_init(self, dataset, *args, **kwargs):
    kwargs['num_workers'] = 0  # Ép vô hiệu hóa multiprocessing trên Windows
    original_init(self, dataset, *args, **kwargs)

dataloader.DataLoader.__init__ = custom_init

# Định nghĩa hàm transform bình thường thay vì dùng lambda
def identity_transform(im):
    return im

class Wrap(torch.utils.data.Dataset):          # plain tensors for encode_dataset
    def __init__(self, sub):
        self.sub = sub
    def __len__(self): return len(self.sub)
    def __getitem__(self, i): return self.sub[i][0]

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='ckpt/ba_netvlad.pt')
    ap.add_argument('--data', default='data/synth_loop')
    ap.add_argument('--out', default='dict/synth_dict.npz')
    ap.add_argument('--ref_fraction', type=float, default=0.5,
                    help='first N frames become refs, rest act as the query sequence')
    ap.add_argument('--cell', type=float, default=4.0,
                    help='pose cell for node_ids; keep consistent with training')
    ap.add_argument('--epsilon', type=float, default=None,
                    help='blur gate: store only SHARP keyframes, per method step 4')
    ap.add_argument('--keyframe_interval', type=int, default=1)
    ap.add_argument('--pca', action='store_true',
                    help='compress the pre-head VLAD with PCA+whitening (method step 3)')
    ap.add_argument('--out_dim', type=int, default=128)
    ap.add_argument('--no_pretrained', action='store_true')
    args = ap.parse_args()

    # Sử dụng hàm chuẩn thay vì lambda
    ds = SequenceDataset(args.data, cell_size=args.cell, transform=identity_transform)
    n_ref = int(len(ds) * args.ref_fraction)
    keep = list(range(0, n_ref, args.keyframe_interval))
    if args.epsilon is not None:
        # The Super Dictionary is defined as holding sharp keyframes only; the
        # original script stored every reference frame, blurry ones included.
        import cv2
        from ba_netvlad.blur import BlurGate
        gate = BlurGate(epsilon=args.epsilon)
        keep = [i for i in keep if gate(cv2.imread(ds.files[i]))[1]]
        print(f'blur gate eps={args.epsilon}: kept {len(keep)}/{n_ref} sharp refs')
    sub = torch.utils.data.Subset(ds, keep)

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_ckpt(args.model, stride=16, pretrained=not args.no_pretrained).to(dev)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

    # Truyền biến sub vào lớp Wrap
    build_dictionary(model, Wrap(sub), args.out,
                     node_ids=ds.node_ids[keep], poses=ds.xy[keep],
                     files=[ds.files[i] for i in keep], device=dev,
                     pca=args.pca, out_dim=args.out_dim, src_index=keep)
    print('saved', args.out, 'entries:', len(keep))