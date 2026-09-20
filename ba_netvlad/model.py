
"""MobileNetV3-Small backbone + NetVLAD layer + 128-D compact head."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class NetVLAD(nn.Module):
    """Soft-assignment VLAD aggregation (Arandjelovic et al., CVPR 2016)."""
    def __init__(self, dim, k=64):
        super().__init__()
        self.k, self.dim = k, dim
        self.assign = nn.Conv2d(dim, k, kernel_size=1)
        nn.init.normal_(self.assign.weight, std=1e-3)
        nn.init.zeros_(self.assign.bias)
        self.centers = nn.Parameter(torch.randn(k, dim) * 0.1)

    @torch.no_grad()
    def init_from_descriptors(self, x, alpha=None):
        """Seed centers by k-means on real local descriptors and set the
        assignment conv to 2*alpha*c_k / -alpha*||c_k||^2 (Arandjelovic Sec. 3.2).

        Random std=1e-3 logits make the softmax uniform (a_k == 1/K), which turns
        VLAD into `sum_i x_i - N*c_k` -- global sum pooling minus a constant. Every
        image then maps to nearly the same vector, the triplet loss parks at exactly
        `margin`, and no learning rate escapes it. This init is what breaks that.
        """
        X = x.permute(0, 2, 3, 1).reshape(-1, self.dim).cpu().numpy()
        X = X[np.random.RandomState(0).choice(len(X), min(len(X), 20000), replace=False)]
        Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)
        c = Xn[np.random.RandomState(0).choice(len(Xn), self.k, replace=False)]
        for _ in range(25):                     # Lloyd iterations on the unit sphere
            lab = np.argmax(Xn @ c.T, axis=1)
            for j in range(self.k):
                sel = Xn[lab == j]
                if len(sel):
                    c[j] = sel.mean(0) / max(np.linalg.norm(sel.mean(0)), 1e-8)
        c = torch.from_numpy(c.astype(np.float32))
        if alpha is None:                       # peak the softmax to ~top-2 mass
            d = torch.cdist(torch.from_numpy(Xn[:2000].astype(np.float32)), c)
            s = d.sort(dim=1).values
            alpha = float(np.log(100.0) / max((s[:, 1] - s[:, 0]).mean(), 1e-6))
        self.centers.copy_(c)
        self.assign.weight.copy_((2.0 * alpha * c)[:, :, None, None])
        self.assign.bias.copy_(-alpha * c.pow(2).sum(1))
        return alpha

    def forward(self, x):                       # [B, C, H, W]
        B, C, H, W = x.shape
        a = self.assign(x).view(B, self.k, -1)  # logits [B, K, N]
        a = torch.softmax(a, dim=1)             # a_k(x_i)
        x = x.view(B, C, -1)                    # [B, C, N]
        resid = x.unsqueeze(1) - self.centers.t().reshape(1, self.k, C, 1)
        V = (a.unsqueeze(2) * resid).sum(-1)    # [B, K, C] sum_i a_k (x_i - c_k)
        V = F.normalize(V, p=2, dim=2)          # intra-normalization
        V = V.view(B, -1)
        return F.normalize(V, p=2, dim=1)

class BANetVLAD(nn.Module):
    """Full BA-NetVLAD global descriptor network.

    stride=16 keeps N=196 local descriptors (224px input); stride=32 keeps N=49
    (faster, slightly lower recall). Output: L2-normalized 128-D vector.
    """
    def __init__(self, dim=96, k=64, out_dim=128, stride=16, pretrained=True):
        super().__init__()
        bb = models.mobilenet_v3_small(
            weights=(models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
                     if pretrained else None))
        last_idx = 8 if stride == 16 else 11    # idx8: stride16/48ch, idx11: stride32/96ch
        self.features = bb.features[:last_idx + 1]
        c_in = 48 if stride == 16 else 96
        self.proj1 = nn.Conv2d(c_in, dim, kernel_size=1)
        self.netvlad = NetVLAD(dim, k)
        # Single biasless linear projection, matching the PCA+whitening the method
        # specifies. The old Linear(6144,256)+ReLU+Linear(256,128) squeezed VLAD
        # through a 256-D bottleneck and then ADDED an output bias: a constant
        # vector shared by every descriptor. That DC term dominated the image
        # dependent part, so all pairwise cosines landed in [0.998, 1.000] and no
        # `delta` margin test could ever fire.
        self.head = nn.Linear(k * dim, out_dim, bias=False)

    @torch.no_grad()
    def init_netvlad(self, x_img):
        """Run a batch of real images through the backbone and seed NetVLAD from it."""
        return self.netvlad.init_from_descriptors(self.proj1(self.features(x_img)))

    def forward(self, x, return_vlad=False):
        x = self.features(x)
        x = self.proj1(x)
        vlad = self.netvlad(x)
        g = F.normalize(self.head(vlad), p=2, dim=1)
        return (g, vlad) if return_vlad else g

def load_ckpt(path, **kw):
    m = BANetVLAD(**kw)
    sd = torch.load(path, map_location='cpu')
    sd = sd.get('model', sd)
    if 'head.0.weight' in sd:
        raise RuntimeError(
            f"{path} was trained with the old MLP head (Linear-ReLU-Linear+bias), "
            "which collapses every descriptor into a cone of pairwise cosine "
            ">0.998 and makes the `delta` margin test unsatisfiable. Retrain with "
            "scripts/02_train_netvlad.py to produce a checkpoint for the current head.")
    m.load_state_dict(sd)
    return m.eval()
