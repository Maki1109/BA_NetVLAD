
"""End-to-end sanity check (no downloads, no GPU required). Run: python tests/smoke_test.py"""
import os, sys, tempfile
import numpy as np, cv2, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ba_netvlad.blur import laplacian_beta
from ba_netvlad.augmentation import motion_blur
from ba_netvlad.model import BANetVLAD
from ba_netvlad.loss import lazy_triplet_loss
from ba_netvlad.matcher import CosineDict, TemporalVoter, confirm_loop
from ba_netvlad.evaluate import recall_at_n

def check(name, cond):
    print(('PASS' if cond else 'FAIL'), '-', name); assert cond, name

# 1) blur gate separates sharp from motion-blurred
rng = np.random.RandomState(0)
sharp = (rng.rand(240, 320, 3) * 255).astype(np.uint8)
for _ in range(40):   # add edges so Laplacian has signal
    cv2.line(sharp, tuple(rng.randint(0, 300, 2)), tuple(rng.randint(0, 200, 2)),
             (255, 255, 255), 2)
b_sharp = laplacian_beta(sharp)
b_blur  = laplacian_beta(motion_blur(sharp, 25))
check(f'blur gate (sharp={b_sharp:.0f} >> blur={b_blur:.0f})', b_sharp > 3 * b_blur)

# 2) model forward: [1,3,224,224] -> normalized 128-D
m = BANetVLAD(dim=32, k=8, out_dim=128, stride=16, pretrained=False)
g = m(torch.randn(2, 3, 224, 224))
check(f'model output shape {tuple(g.shape)} + unit norm', g.shape == (2, 128)
      and torch.allclose(g.norm(dim=1), torch.ones(2), atol=1e-4))

# 3) lazy triplet loss: separates easy batch, backprops
emb = torch.randn(8, 128, requires_grad=True)
lab = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
loss = lazy_triplet_loss(emb, lab, margin=0.2)
loss.backward()
check(f'lazy triplet loss finite ({float(loss):.4f})', torch.isfinite(loss).all()
      and emb.grad is not None)

# 4) matcher + temporal voting requires m-of-r agreement
descs = np.eye(128, dtype=np.float32)[:50]
cd = CosineDict({'descs': descs, 'node_ids': np.arange(50),
                 'poses': np.zeros((50, 2), np.float32)})
q = descs[7].copy()
idx, sc = cd.topk(q, k=3)
v = TemporalVoter(r=5, m=3)
oks = [confirm_loop(idx, sc, v, tau=0.9, delta=0.05)[0] for _ in range(4)]
check(f'temporal voting (accepts at vote 3+: {oks})', oks == [False, False, True, True])

# 5) recall@n perfect on toy scores
S = np.eye(10, dtype=np.float32); S[S == 0] = 0.1
check('recall@1 == 1.0 on toy data', recall_at_n(S, np.eye(10).astype(bool))['R@1'] == 1.0)
print('\nALL SMOKE TESTS PASSED')
