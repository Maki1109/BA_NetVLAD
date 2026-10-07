"""Evaluate off-the-shelf VPR models (MixVPR, SALAD, NetVLAD, CosPlace, EigenPlaces ...,
loaded through gmberton/VPR-methods-evaluation) under the SAME protocol as BA-NetVLAD's
Recall@N, with optional synthetic motion blur applied to the queries only.

Protocols (mirrors ba_netvlad/replay.py and scripts/05_folder_vpr_recall.py):
  kitti05 / kitti06 : refs = the frames stored in dict/<ds>_dict.npz (same files, same
      src_index); queries = every frame of data/<ds>. Positive = ref within --cell metres
      AND |query_idx - ref_idx| > --loop_index_gap; the same index window is masked out of
      retrieval. Recall@N is taken over loop-capable queries only.
  gpw_day / gpw_night : ref = --gpw_ref folder, query = day_right / night_right, identity GT.

Blur: motion_blur(img, L) then a JPEG round trip, i.e. what 08_make_blurred_copy.py writes.
Only queries are blurred (refs stay sharp), as in the KITTI00 blur experiment.

Usage:
  python scripts/11_baseline_eval.py --method salad --dataset gpw_day --blur 0 20 30
"""
import argparse, csv, json, os, sys, time
import cv2, numpy as np, torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ba_netvlad.augmentation import motion_blur
from ba_netvlad.evaluate import recall_at_n, average_precision, max_recall_at_full_precision

DEFAULT_SIZE = {'mixvpr': None, 'salad': (322, 322), 'ba_netvlad': (224, 224)}   # others: 320x320
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def load_model(method, backbone, dim, vpr_repo, device, ckpt=None):
    if method == 'ba_netvlad':                  # our model, same checkpoints as scripts/05
        from ba_netvlad.model import load_ckpt
        return load_ckpt(ckpt, stride=16).eval().to(device)
    # torch>=2.9 hub: the forked-repo check raises KeyError('Authorization') without a
    # GitHub token; it is only a safety warning, so skip it for the trusted baseline repos.
    torch.hub._validate_not_a_forked_repo = lambda *a, **k: True
    if device == 'cpu':         # MixVPR's loader calls torch.load without map_location
        _load = torch.load
        torch.load = lambda *a, **k: _load(*a, **{'map_location': 'cpu', **k})
    sys.path.insert(0, vpr_repo)
    cwd = os.getcwd()
    os.chdir(vpr_repo)          # vpr_models downloads weights into ./trained_models
    try:
        import vpr_models
        model = vpr_models.get_model(method, backbone, dim)
    finally:
        os.chdir(cwd)
    return model.eval().to(device)


def prep(img_bgr, size):
    if size is not None:
        img_bgr = cv2.resize(img_bgr, (size[1], size[0]), interpolation=cv2.INTER_AREA)
    t = (img_bgr[:, :, ::-1].astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(np.ascontiguousarray(t.transpose(2, 0, 1)))


def blur_like_script08(img, L):
    if L <= 1:
        return img
    ok, buf = cv2.imencode('.jpg', motion_blur(img, L, 0))     # same JPEG round trip as 08
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


@torch.inference_mode()
def encode(model, files, size, blur, device, bs):
    descs, t_model = [], 0.0
    for s in range(0, len(files), bs):
        batch = torch.stack([prep(blur_like_script08(cv2.imread(f), blur), size)
                             for f in files[s:s + bs]]).to(device)
        t0 = time.time()
        d = model(batch).float().cpu().numpy()
        t_model += time.time() - t0
        descs.append(d)
    d = np.concatenate(descs)
    return d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-8), t_model / len(files)


def build_protocol(args):
    """-> ref_files, query_files, gt_fn(S_shape)->(G, mask). Returns (ref, qry, G, excl)."""
    if args.dataset in ('kitti05', 'kitti06'):
        d = np.load(os.path.join(ROOT, f'dict/{args.dataset}_dict.npz'), allow_pickle=True)
        root = os.path.join(args.data_root, args.dataset)
        rows = list(csv.DictReader(open(f'{root}/poses.csv')))
        rows.sort(key=lambda r: float(r['timestamp']))
        qry = [f"{root}/{r['filename']}" for r in rows]
        qxy = np.array([[float(r['x']), float(r['y'])] for r in rows])
        ref_index = np.asarray(d['src_index'])
        ref = [qry[j] for j in ref_index]          # refs are frames of the same sequence
        ref_xy = d['poses']
        qi = np.arange(len(qry))[:, None]
        excl = np.abs(ref_index[None, :] - qi) <= args.loop_index_gap
        dist = np.linalg.norm(qxy[:, None, :] - ref_xy[None, :, :], axis=2)
        G = (dist <= args.cell) & ~excl
        return ref, qry, G, excl
    gp = os.path.join(args.data_root, 'GardensPoint')
    qname = {'gpw_day': 'day_right', 'gpw_night': 'night_right'}[args.dataset]
    ls = lambda n: [os.path.join(gp, n, f) for f in sorted(os.listdir(os.path.join(gp, n)))]
    ref, qry = ls(args.gpw_ref), ls(qname)
    return ref, qry, np.eye(len(qry), len(ref), dtype=bool), np.zeros((len(qry), len(ref)), bool)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', required=True)
    ap.add_argument('--backbone', default=None)
    ap.add_argument('--dim', type=int, default=None)
    ap.add_argument('--ckpt', default=None, help='BA-NetVLAD checkpoint (method ba_netvlad)')
    ap.add_argument('--dataset', required=True, choices=['kitti05', 'kitti06', 'gpw_day', 'gpw_night'])
    ap.add_argument('--blur', type=float, nargs='+', default=[0, 20, 30])
    ap.add_argument('--data_root', default=os.path.join(ROOT, 'data'))
    ap.add_argument('--vpr_repo', default=os.path.join(os.path.dirname(ROOT), 'baselines_auto_VPR'))
    ap.add_argument('--gpw_ref', default='day_left', choices=['day_left', 'day_right'])
    ap.add_argument('--cell', type=float, default=4.0)
    ap.add_argument('--loop_index_gap', type=int, default=400)
    ap.add_argument('--image_size', type=int, nargs=2, default=None, help='H W; default per method')
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--out_dir', default=os.path.join(ROOT, 'results/baselines'))
    ap.add_argument('--cache_dir', default=os.path.join(ROOT, 'results/baselines/desc_cache'))
    args = ap.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    size = tuple(args.image_size) if args.image_size else DEFAULT_SIZE.get(args.method, (320, 320))
    tag = f"{args.method}{'_' + str(args.dim) if args.dim else ''}"
    if args.method == 'ba_netvlad':
        tag += '_' + os.path.splitext(os.path.basename(args.ckpt))[0]
    os.makedirs(args.out_dir, exist_ok=True); os.makedirs(args.cache_dir, exist_ok=True)

    ref_files, qry_files, G, excl = build_protocol(args)
    print(f'{tag} on {args.dataset}: {len(ref_files)} refs, {len(qry_files)} queries, '
          f'{int(G.any(1).sum())} loop-capable, device={dev}, size={size}')
    model = load_model(args.method, args.backbone, args.dim, args.vpr_repo, dev, args.ckpt)

    rc = f'{args.cache_dir}/{tag}_{args.dataset}_ref_{args.gpw_ref}.npy'
    use_dict = args.method == 'ba_netvlad' and args.dataset.startswith('kitti')
    if use_dict:
        # BA-NetVLAD on KITTI = the replay protocol: refs are the Super Dictionary entries
        # and queries go through the same PCA+whitening that built it.
        from ba_netvlad.dictionary import load_dictionary
        dd = load_dictionary(os.path.join(ROOT, f'dict/{args.dataset}_dict.npz'))
        R, t_ref = dd['descs'], float('nan')
        if dd['pca'] is not None:
            net, proj = model, dd['pca']
            class Projected(torch.nn.Module):
                def forward(self, x):
                    return torch.from_numpy(proj(net(x, return_vlad=True)[1].cpu().numpy()))
            model = Projected()
    elif os.path.exists(rc):
        R, t_ref = np.load(rc), float('nan')
    else:
        R, t_ref = encode(model, ref_files, size, 0, dev, args.batch_size)
        np.save(rc, R)

    for L in args.blur:
        out = f'{args.out_dir}/{tag}_{args.dataset}_blur{int(L)}.json'
        qc = f'{args.cache_dir}/{tag}_{args.dataset}_q_blur{int(L)}.npy'
        if os.path.exists(qc):
            Q, t_q = np.load(qc), float('nan')
        else:
            Q, t_q = encode(model, qry_files, size, L, dev, args.batch_size)
            np.save(qc, Q)
        S = Q @ R.T
        S = np.where(excl, -1.0, S)       # masked entries can never win (and keep PR finite)
        res = {'method': tag, 'dataset': args.dataset, 'blur_px': L, 'dim': int(R.shape[1]),
               'image_size': size, 'n_ref': len(ref_files), 'n_query': len(qry_files),
               'n_loopable_queries': int(G.any(1).sum()), 'recall': recall_at_n(S, G),
               'AP': average_precision(S, G),
               'max_recall_at_100_precision': max_recall_at_full_precision(S, G),
               'sec_per_image_model_only': t_q, 'device': dev}
        json.dump(res, open(out, 'w'), indent=2)
        print(L, res['recall'], f"AP={res['AP']:.3f}")
