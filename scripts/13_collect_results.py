"""Gather every result of the method comparison into one folder: results/final/

  results/final/
    README.md                 what each file is and how it was produced
    csv/                      per-dataset CSVs + summary_wide.csv (scripts/14_make_csv.py)
    all_results.csv           one row per (method, dataset, blur): R@1/5/10, AP, dim, time
    comparison.md             the comparison tables (Markdown)
    tables_comparison.tex     the same tables for the paper
    runs/                     the raw per-run JSON files of scripts/11_baseline_eval.py
    ba_netvlad_pipeline/      BA-NetVLAD full-pipeline results (KITTI replay, blur, sweeps)
    figures/                  PR curves and paper figures
    logs/                     run logs

Usage: python scripts/12_make_tables.py && python scripts/13_collect_results.py
"""
import csv, glob, json, os, shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, 'results'); OUT = os.path.join(R, 'final')


def cp(src, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    for f in glob.glob(src):
        if os.path.isfile(f):
            shutil.copy2(f, dst_dir)


if os.path.isdir(OUT):
    shutil.rmtree(OUT)
os.makedirs(OUT)

cp(f'{R}/baselines/*.json', f'{OUT}/runs')
cp(f'{R}/baselines/comparison.md', OUT)
cp(f'{R}/csv/*.csv', f'{OUT}/csv')
cp(os.path.join(ROOT, 'paper/tables_comparison.tex'), OUT)
cp(f'{R}/baselines/gpw_summary.md', OUT)
for pat in ('kitti*_replay*.json', 'kitti00_blur*.json', 'kitti05_sweep_*.json', 'sweep*.json'):
    cp(f'{R}/{pat}', f'{OUT}/ba_netvlad_pipeline')
cp(f'{R}/*.png', f'{OUT}/figures'); cp(os.path.join(ROOT, 'paper/figures/*.pdf'), f'{OUT}/figures')
for pat in ('baselines*.log', 'par_*.log'):
    cp(f'{R}/{pat}', f'{OUT}/logs')

rows = []
for f in sorted(glob.glob(f'{OUT}/runs/*.json')):
    d = json.load(open(f))
    if 'recall' not in d:      # cost.json etc.
        continue
    rows.append({'method': d['method'], 'dataset': d['dataset'], 'blur_px': int(d['blur_px']),
                 'dim': d['dim'], 'R@1': d['recall']['R@1'], 'R@5': d['recall']['R@5'],
                 'R@10': d['recall']['R@10'], 'AP': d['AP'],
                 'max_recall_at_100_precision': d['max_recall_at_100_precision'],
                 'n_ref': d['n_ref'], 'n_query': d['n_query'], 'n_loopable': d['n_loopable_queries'],
                 'image_size': d['image_size'], 'sec_per_image': d['sec_per_image_model_only']})
with open(f'{OUT}/all_results.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, rows[0].keys()); w.writeheader(); w.writerows(rows)

open(f'{OUT}/README.md', 'w').write(f"""# Final results: BA-NetVLAD vs NetVLAD, MixVPR, SALAD

{len(rows)} runs in `all_results.csv` (also one JSON each in `runs/`).

Produced by `scripts/11_baseline_eval.py` (one protocol for all methods), tables by
`scripts/12_make_tables.py`, this folder by `scripts/13_collect_results.py`.

* Datasets: KITTI05, KITTI06 (positive <= 4 m, +-400-frame exclusion, references = the Super
  Dictionary frames), Gardens Point day->day / day->night (ref = day_left, identity ground truth).
* Blur: linear motion kernel (0, 20, 30 px) on the queries only, JPEG round trip, references sharp.
* `ba_netvlad_pipeline/`: full-pipeline replay results of BA-NetVLAD itself (KITTI00/05/06, blur,
  tau/delta sweeps) from `scripts/04_replay_evaluate.py`; not produced by the comparison script.
* `GPW fine-tuned` BA-NetVLAD was trained on the test places and is not comparable with zero-shot rows.
* CPU timings are indicative only (several jobs shared the CPU).
""")
print(f'collected {len(rows)} runs into {OUT}')
