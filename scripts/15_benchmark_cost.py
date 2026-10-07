"""Clean CPU cost benchmark (batch 1, forward pass only, no other job running).
Writes results/baselines/cost.json. Usage: python scripts/15_benchmark_cost.py"""
import importlib.util, json, os, sys, time
import numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
spec = importlib.util.spec_from_file_location('ev', f'{ROOT}/scripts/11_baseline_eval.py')
ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
import cv2
img = cv2.imread(f'{ROOT}/data/kitti05/frames/00100.jpg')
repo = os.path.join(os.path.dirname(ROOT), 'baselines_auto_VPR')
cfg = [('BA-NetVLAD', 'ba_netvlad', None, (224, 224)), ('NetVLAD', 'netvlad', 4096, (320, 320)),
       ('MixVPR', 'mixvpr', 4096, None), ('SALAD', 'salad', None, (322, 322))]
out = {}
for name, m, dim, size in cfg:
    model = ev.load_model(m, None, dim, repo, 'cpu', f'{ROOT}/ckpt/ba_kitti00.pt')
    x = ev.prep(img, size).unsqueeze(0)
    n_par = sum(p.numel() for p in model.parameters())
    ts = []
    with torch.inference_mode():
        for i in range(3 + 15):
            t = time.perf_counter(); d = model(x); ts.append(time.perf_counter() - t)
    ts = ts[3:]
    out[name] = {'ms_median': 1000 * float(np.median(ts)), 'ms_std': 1000 * float(np.std(ts)),
                 'params_M': n_par / 1e6, 'dim': int(d.shape[-1]), 'input': list(x.shape[-2:]),
                 'desc_KB_fp32': int(d.shape[-1]) * 4 / 1024, 'threads': torch.get_num_threads()}
    print(name, out[name], flush=True)
json.dump(out, open(f'{ROOT}/results/baselines/cost.json', 'w'), indent=2)
