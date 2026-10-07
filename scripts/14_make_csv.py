"""Write result CSVs from results/baselines/*.json -> results/csv/
  <dataset>.csv  long format: one row per method x blur (kitti05, kitti06, gpw_day, gpw_night)
  summary_wide.csv  one row per method x dataset, columns sharp/20px/30px R@1,R@5,R@10,AP
Missing runs are written as empty cells, so it can be re-run while evaluations are still going.
Usage: python scripts/14_make_csv.py"""
import csv, json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = f'{ROOT}/results/baselines'; OUT = f'{ROOT}/results/csv'; os.makedirs(OUT, exist_ok=True)
DATASETS = [('kitti05', 'KITTI05'), ('kitti06', 'KITTI06'),
            ('gpw_day', 'GardensPoint day-day'), ('gpw_night', 'GardensPoint day-night')]
METHODS = [('ba_netvlad_ba_kitti00', 'BA-NetVLAD (KITTI00 model)', None),
           ('ba_netvlad_ba_gpw_finetuned2', 'BA-NetVLAD (GPW fine-tuned, test places)', ('gpw_day', 'gpw_night')),
           ('netvlad_4096', 'NetVLAD', None), ('mixvpr_4096', 'MixVPR 4096-D', None),
           ('mixvpr_128', 'MixVPR 128-D', None), ('salad', 'SALAD', None)]
BLURS = (0, 20, 30); KEYS = ('R@1', 'R@5', 'R@10')
load = lambda t, d, b: (json.load(open(p)) if os.path.exists(p := f'{RES}/{t}_{d}_blur{b}.json') else None)
f = lambda v: '' if v is None else f'{v:.4f}'
wide = []
for ds, dname in DATASETS:
    rows = []
    for tag, name, only in METHODS:
        if only and ds not in only: continue
        w = {'dataset': dname, 'method': name}
        for b in BLURS:
            r = load(tag, ds, b)
            rows.append({'dataset': dname, 'method': name, 'dim': r['dim'] if r else '', 'blur_px': b,
                         **{k: f(r and r['recall'][k]) for k in KEYS}, 'AP': f(r and r['AP']),
                         'n_queries': r['n_query'] if r else '', 'n_refs': r['n_ref'] if r else '',
                         'n_loop_queries': r['n_loopable_queries'] if r else ''})
            lab = 'sharp' if b == 0 else f'blur{b}px'
            for k in KEYS + ('AP',):
                w[f'{lab}_{k}'] = f(r and (r['AP'] if k == 'AP' else r['recall'][k]))
        wide.append(w)
    with open(f'{OUT}/{ds}.csv', 'w', newline='') as fh:
        wr = csv.DictWriter(fh, rows[0].keys()); wr.writeheader(); wr.writerows(rows)
with open(f'{OUT}/summary_wide.csv', 'w', newline='') as fh:
    wr = csv.DictWriter(fh, wide[0].keys()); wr.writeheader(); wr.writerows(wide)
missing = [(d, m) for d, m in ((w['dataset'], w['method']) for w in wide
           for _ in [0] if any(w[k] == '' for k in w)) ]
print('wrote', sorted(os.listdir(OUT)))
for d, m in missing: print('incomplete:', d, '|', m)
