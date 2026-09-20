
"""Convert a text ground-truth pair list ('query_idx ref_idx' per line) to gt.npy
for FolderVPRDataset. Indices are 0-based, sorted-filename order of query/ and ref/.
Usage: python scripts/07_pairs_to_gt_npy.py --pairs gt.txt --n_query 200 --n_ref 200 --out root/gt.npy"""
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--pairs', required=True)
ap.add_argument('--n_query', type=int, required=True)
ap.add_argument('--n_ref', type=int, required=True)
ap.add_argument('--out', required=True)
args = ap.parse_args()

gt = np.zeros((args.n_query, args.n_ref), bool)
for line in open(args.pairs):
    line = line.strip()
    if not line: continue
    a, b = line.replace(',', ' ').split()[:2]
    gt[int(a), int(b)] = True
np.save(args.out, gt)
print(f"gt.npy {gt.shape}, positives={gt.sum()}")
