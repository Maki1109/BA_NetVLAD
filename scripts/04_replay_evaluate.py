
"""Run the full BA-NetVLAD control flow over the query split -> branch-split recall,
accepted-loop precision/recall. Writes JSON summary."""
import argparse, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ba_netvlad.model import load_ckpt
from ba_netvlad.dictionary import load_dictionary
from ba_netvlad.replay import run_replay

ap = argparse.ArgumentParser()
ap.add_argument('--model', default='ckpt/ba_netvlad.pt')
ap.add_argument('--dict', default='dict/synth_dict.npz')
ap.add_argument('--query_root', default='data/synth_loop')
ap.add_argument('--epsilon', type=float, default=120.0)
ap.add_argument('--tau', type=float, default=0.75)
ap.add_argument('--delta', type=float, default=0.05)
ap.add_argument('--r', type=int, default=5)
ap.add_argument('--m', type=int, default=3)
ap.add_argument('--keyframe_interval', type=int, default=5)
ap.add_argument('--loop_index_gap', type=int, default=100,
                help='min |query_idx - ref_idx| for a match to count as a loop; '
                     'also the temporal exclusion window applied to retrieval')
ap.add_argument('--cell', type=float, default=4.0,
                help='GT radius in metres (was hardcoded to 4.0)')
ap.add_argument('--margin_window', type=int, default=25,
                help='frames around the top-1 match excluded when picking s2, so '
                     'delta measures place distinctiveness, not frame-to-frame jitter')
ap.add_argument('--vote_cell', type=float, default=None,
                help='metres per voting node; must exceed keyframe_interval * '
                     'inter-frame distance or m-of-r agreement is unreachable')
ap.add_argument('--mode', choices=['ba', 'netvlad', 'orb'], default='ba',
                help="ba: ORB verification on sharp frames only, NetVLAD-only on blurry; "
                     "netvlad: no ORB (ablation); orb: ORB verification on every frame (baseline)")
ap.add_argument('--orb_min_inliers', type=int, default=15,
                help='RANSAC inliers needed to verify a candidate')
ap.add_argument('--orb_topk', type=int, default=5, help='candidates verified per query')
ap.add_argument('--no_pretrained', action='store_true')
ap.add_argument('--out', default='results/replay.json')
args = ap.parse_args()

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
model = load_ckpt(args.model, stride=16, pretrained=not args.no_pretrained).to(dev)
d = load_dictionary(args.dict)
res = run_replay(model, d, args.query_root, epsilon=args.epsilon,
                 tau=args.tau, delta=args.delta, r=args.r, m=args.m,
                 keyframe_interval=args.keyframe_interval,
                 loop_index_gap=args.loop_index_gap, cell_size=args.cell,
                 margin_window=args.margin_window, vote_cell=args.vote_cell,
                 project=d.get('pca'), device=dev,
                 mode=args.mode, orb_min_inliers=args.orb_min_inliers,
                 orb_topk=args.orb_topk)
res['accepted_loops'] = res['accepted_loops'][:50]
os.makedirs(os.path.dirname(args.out), exist_ok=True)
json.dump(res, open(args.out, 'w'), indent=2)
print(json.dumps({k: v for k, v in res.items() if k not in ('accepted_loops', 'orb_stats')}, indent=2))
print(f"(first accepted loops: {res['accepted_loops'][:3]})")
