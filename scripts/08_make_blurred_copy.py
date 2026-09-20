
"""Pre-blur a sequence's frames -> blur-branch benchmark copy.
Usage: python scripts/08_make_blurred_copy.py --src data/kitti00 --dst data/kitti00_blur --length 25"""
import argparse, csv, os
import cv2
import sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ba_netvlad.augmentation import motion_blur

ap = argparse.ArgumentParser()
ap.add_argument('--src', required=True)
ap.add_argument('--dst', required=True)
ap.add_argument('--length', type=float, default=25)
ap.add_argument('--angle', type=float, default=0)
args = ap.parse_args()

os.makedirs(f"{args.dst}/frames", exist_ok=True)
rows = list(csv.DictReader(open(f"{args.src}/poses.csv")))
for r in rows:
    img = cv2.imread(f"{args.src}/{r['filename']}")
    cv2.imwrite(f"{args.dst}/{r['filename']}",
                motion_blur(img, args.length, args.angle))
with open(f"{args.dst}/poses.csv", 'w', newline='') as fh:
    wcsv = csv.DictWriter(fh, ['timestamp', 'filename', 'x', 'y'])
    wcsv.writeheader(); wcsv.writerows(rows)
print(f"wrote blurred copy (L={args.length}px) to {args.dst}")
