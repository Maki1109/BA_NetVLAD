

import argparse, csv, os, glob
import numpy as np, cv2

ap = argparse.ArgumentParser()
ap.add_argument('--kitti', required=True, help='KITTI odometry root (with poses/ and sequences/)')
ap.add_argument('--seq', default='00')
ap.add_argument('--out', required=True)
ap.add_argument('--skip', type=int, default=1, help='subsample every Nth frame')
ap.add_argument('--img_dir', default='image_0')
args = ap.parse_args()

pose_file = f"{args.kitti}/poses/{args.seq}.txt"
img_root  = f"{args.kitti}/sequences/{args.seq}/{args.img_dir}"
imgs = sorted(glob.glob(f"{img_root}/*.png"))
poses = np.loadtxt(pose_file).reshape(-1, 3, 4)
assert len(imgs) == len(poses), f"{len(imgs)} imgs vs {len(poses)} poses"

os.makedirs(f"{args.out}/frames", exist_ok=True)
rows = []
kept = 0
for i in range(0, len(imgs), args.skip):
    P = poses[i]
    x, y = float(P[0, 3]), float(P[2, 3])
    f = f"frames/{kept:05d}.jpg"
    cv2.imwrite(f"{args.out}/{f}", cv2.imread(imgs[i]))
    rows.append({'timestamp': kept, 'filename': f, 'x': x, 'y': y})
    kept += 1
with open(f"{args.out}/poses.csv", 'w', newline='') as fh:
    wcsv = csv.DictWriter(fh, ['timestamp', 'filename', 'x', 'y'])
    wcsv.writeheader(); wcsv.writerows(rows)
print(f"wrote {kept}/{len(imgs)} frames (skip={args.skip}) to {args.out}")
