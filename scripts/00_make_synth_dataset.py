
"""Generate a synthetic square-loop dataset for an instant end-to-end test.
Output: <out>/poses.csv (header: timestamp,filename,x,y) + <out>/frames/*.jpg."""
import argparse, csv, os
import numpy as np, cv2

def make(out, n_side=8, period=6.0, imsize=480, seed=0):
    rng = np.random.RandomState(seed)
    per = 12; pts = []
    for s in range(4 * per):                 # one square lap
        t = s / per
        seg, u = int(t), t % 1.0
        L = n_side * period
        p = [(u * L, 0), (L, u * L), (L - u * L, L), (0, L - u * L)][seg]
        pts.append(p)
    os.makedirs(f"{out}/frames", exist_ok=True)
    cells = {}
    for gx in range(n_side):
        for gy in range(n_side):             # one random texture per map cell
            base = rng.randint(40, 215, (imsize, imsize, 3), np.uint8)
            for _ in range(rng.randint(8, 20)):
                c = tuple(int(x) for x in rng.randint(0, 255, 3))
                cv2.rectangle(base, tuple(rng.randint(0, imsize - 80, 2)),
                              tuple(rng.randint(0, imsize - 80, 2) + 80), c, -1)
            cells[(gx, gy)] = base
    rows = []
    for i, (x, y) in enumerate(pts):
        gx = min(int(x // period), n_side - 1)
        gy = min(int(y // period), n_side - 1)
        img = cells[(gx, gy)]
        if rng.rand() < 0.3:                 # intra-place appearance variation
            img = np.roll(img, shift=int(rng.randint(-60, 60)), axis=1)
        f = f"frames/{i:05d}.jpg"
        cv2.imwrite(f"{out}/{f}", img)
        rows.append({'timestamp': i, 'filename': f, 'x': x, 'y': y})
    with open(f"{out}/poses.csv", 'w', newline='') as fh:
        wcsv = csv.DictWriter(fh, ['timestamp', 'filename', 'x', 'y'])
        wcsv.writeheader()
        wcsv.writerows(rows)
    print(f"wrote {len(rows)} frames to {out}")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='data/synth_loop')
    args = ap.parse_args(); make(args.out)
