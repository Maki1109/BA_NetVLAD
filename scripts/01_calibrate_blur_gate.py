
"""Step 7.1 on laptop: sweep synthetic motion blur -> beta response -> epsilon knee."""
import argparse, glob, os, sys
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ba_netvlad.blur import laplacian_beta
from ba_netvlad.augmentation import motion_blur

ap = argparse.ArgumentParser()
ap.add_argument('--frames', default='data/synth_loop/frames')
ap.add_argument('--lengths', default='0,5,10,15,20,25,30,40')
ap.add_argument('--ratio', type=float, default=0.5, help='knee = ratio * median sharp beta')
args = ap.parse_args()

files = sorted(glob.glob(f"{args.frames}/*.jpg"))[:200]
base = [laplacian_beta(cv2.imread(f)) for f in files]
sharp_med = float(np.median(base))
print(f"sharp beta: median={sharp_med:.1f}  p10={np.percentile(base,10):.1f}")
print(f"{'smear px':>8} {'beta median':>12} {'ratio':>7}")
for L in [int(x) for x in args.lengths.split(',')]:
    vals = [laplacian_beta(motion_blur(cv2.imread(f), L)) for f in files[:50]]
    med = float(np.median(vals))
    print(f"{L:>8} {med:>12.1f} {med/sharp_med:>7.2f}")
print(f"\nsuggested epsilon = {args.ratio * sharp_med:.1f}  (ratio={args.ratio}; "
      f"validate with ORB-inlier knee on real robot data)")
