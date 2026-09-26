# BA-NetVLAD

Blur-Aware NetVLAD for loop closure detection under high-speed rotation / motion blur.

## Motivation

When a mobile robot rotates quickly, motion blur acts as a low-pass filter on the
image, wiping out the high-frequency corners that local features (ORB, SIFT, ...)
depend on. This breaks the local-feature matching that most loop-closure detection
(LCD) pipelines rely on, and odometry drift goes unchecked.

**BA-NetVLAD** routes around this instead of trying to fix it:

1. **Blur gate** — every frame is scored by variance-of-Laplacian sharpness
   $\beta(I) = \mathrm{Var}(\nabla^2 I)$ on a fixed-resolution grayscale crop, with
   hysteresis between `epsilon` and `epsilon_high` to avoid branch-flapping.
2. **Sharp branch** — ORB local features for real-time tracking, NetVLAD global
   descriptor at keyframe intervals.
3. **Blurry branch** — skip ORB entirely; run *every* frame through a
   MobileNetV3-Small + NetVLAD backbone to get a compact global descriptor. NetVLAD
   aggregates large-scale spatial structure (walls, buildings) rather than corners,
   so it stays discriminative even when local texture is destroyed.
4. **Super Dictionary** — a cosine-similarity index of NetVLAD descriptors from
   *sharp* keyframes only. A blurry query frame is matched directly against it, with
   temporal voting ($m$-of-$r$ agreement) before a loop closure is confirmed.

## Repository layout

```
ba_netvlad/            core library
  blur.py                Step 1: BlurGate (variance-of-Laplacian + hysteresis)
  augmentation.py         synthetic motion-blur kernel (training + benchmark copies)
  model.py                MobileNetV3-Small + NetVLAD + linear head, k-means init
  loss.py                 lazy triplet loss with a definitely-negative radius
  dataset.py              SequenceDataset / FolderVPRDataset / GroupSampler
  dictionary.py           Super Dictionary build/load, PCA+whitening
  matcher.py              Step 4: cosine top-k, temporal voting, confirm_loop
  evaluate.py             Recall@N, precision-recall curve
  replay.py               full BA-NetVLAD control-flow replay over a sequence

scripts/
  00_make_synth_dataset.py     synthetic loop dataset (no download required)
  01_calibrate_blur_gate.py    sweep synthetic blur -> suggest epsilon
  02_train_netvlad.py          train with lazy triplet loss (k-means NetVLAD init)
  03_build_dictionary.py       encode a reference split into the Super Dictionary
  04_replay_evaluate.py        full-pipeline replay: Recall@N + loop-closure P/R
  05_folder_vpr_recall.py      standard VPR benchmark (ref/query + ground truth)
  06_kitti_to_sequence.py      KITTI odometry -> SequenceDataset layout
  07_pairs_to_gt_npy.py        pair list -> gt.npy for FolderVPRDataset
  08_make_blurred_copy.py      pre-blur a sequence for the blur-branch benchmark
  09_finetune_gpw.py           fine-tune on Gardens Point Walking (index-based places)
  10_finetune_citycentre.py    fine-tune on City Centre (odd/even synced-camera places)

smoke_test.py           end-to-end sanity check, no GPU/download required
main.tex                full report: method, bugs found, training, all experiments
```

`ckpt/`, `dict/`, and `results/` hold the trained checkpoints, built Super
Dictionaries, and evaluation outputs referenced by the report.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python smoke_test.py             # should print 5x PASS
```

## Data

Not included in this repo (see `.gitignore`). Expected layout under `data/`:

```
data/kitti00/  {frames/*.jpg, poses.csv}          # KITTI odometry, via scripts/06
data/kitti05/  data/kitti06/                      # same layout, other sequences
data/GardensPoint/  {day_left, day_right, night_right}/*.jpg   # GPW raw download
data/gpw_day/  data/gpw_night/  {ref/, query/, gt.npy}         # built from GPW, see below
data/CityCentre/*.jpg                              # City Centre raw (odd/even synced pair convention)
data/citycentre/  {ref/, query/, gt.npy}           # built from CityCentre, see below
```

`poses.csv` columns: `timestamp,filename,x,y[,z,qx,qy,qz,qw]`.

## Pipeline

```
RGB frame ──> [1] Blur gate (β vs ε) ──┬─ sharp: ORB/local + NetVLAD at keyframes
                                       └─ blur : skip ORB, NetVLAD every frame
                                                      │
                                [2] NetVLAD 128-D ──> [3] Cosine vs Super Dictionary
                                                      │
                                [4] Confirm: s1>τ ∧ margin>δ ∧ m-of-r voting
                                                      │
                                        Loop closure / relocalization
```

### End to end on KITTI

```bash
# 0. sanity check
python smoke_test.py

# 1. calibrate the blur gate on your own sequence
python scripts/01_calibrate_blur_gate.py --frames data/kitti00/frames \
    --lengths 0,5,10,15,20,25,30,40 --ratio 0.5

# 2. train (k-means-initializes NetVLAD, then lazy triplet loss)
python scripts/02_train_netvlad.py --data data/kitti00 \
    --epochs 60 --lr 1e-4 --cell 4.0 --neg_radius 25.0 --out ckpt/ba_kitti00.pt

# 3. build the Super Dictionary (blur-gated keyframes only, PCA+whitening)
python scripts/03_build_dictionary.py --model ckpt/ba_kitti00.pt --data data/kitti00 \
    --ref_fraction 0.65 --cell 4.0 --epsilon <ε from step 1> --pca \
    --out dict/kitti00_dict.npz

# 4. replay the full pipeline and evaluate
python scripts/04_replay_evaluate.py --model ckpt/ba_kitti00.pt \
    --dict dict/kitti00_dict.npz --query_root data/kitti00 \
    --epsilon <ε> --tau 0.5 --delta 0.05 --r 5 --m 3 \
    --margin_window 25 --vote_cell 15 --loop_index_gap 400 --cell 4.0 \
    --out results/kitti00_replay.json

# 5. motion-blur benchmark (the core BA-NetVLAD claim)
python scripts/08_make_blurred_copy.py --src data/kitti00 --dst data/kitti00_blur20 --length 20
python scripts/04_replay_evaluate.py --model ckpt/ba_kitti00.pt \
    --dict dict/kitti00_dict.npz --query_root data/kitti00_blur20 \
    --epsilon <ε> --loop_index_gap 400 --cell 4.0 --out results/kitti00_blur20.json
```

### Standard VPR benchmarks (Gardens Point Walking, City Centre)

GPW and City Centre have no per-frame pose, so they use an index-based ground truth
instead of a spatial radius:

- **GPW**: `day_left[i]`, `day_right[i]`, `night_right[i]` are the same physical
  place for every `i`. Split into `ref/`+`query/` and generate `gt.npy` with
  `scripts/07_pairs_to_gt_npy.py` (identity pairing).
- **City Centre**: the 2474-image release alternates two synchronized cameras —
  `odd[i]` and `even[i]` are the same place at the same instant. `ref/` = odd
  frames, `query/` = even frames, identity `gt.npy`.

```bash
python scripts/05_folder_vpr_recall.py --root data/gpw_day --model ckpt/ba_kitti00.pt
python scripts/05_folder_vpr_recall.py --root data/gpw_night --model ckpt/ba_kitti00.pt
python scripts/05_folder_vpr_recall.py --root data/citycentre --model ckpt/ba_kitti00.pt --radius 25
```

Since these are visually far from KITTI's driving footage, zero-shot recall is
low; `scripts/09_finetune_gpw.py` / `scripts/10_finetune_citycentre.py` adapt the
model using the index-based place correspondence above (report the fine-tuned
numbers as domain-adaptation capacity, not held-out generalization — both
datasets are too small to hold out a separate split).

## Results summary

Full derivation and discussion in `main.tex`. Headline numbers (KITTI00-trained
model, see report for the exact operating point and every caveat):

| Experiment | Metric | Result |
|---|---|---|
| KITTI00 sharp replay | R@5 | 93.6% |
| KITTI00 motion-blur (20px / 30px) | R@5 retained vs sharp | 99.8% / 99.6% |
| KITTI05 (zero-shot) | R@5 vs KITTI00 | −1.26 pts |
| KITTI06 (negative control) | accepted loops / precision | 2 / 100% |
| GPW day→night (fine-tuned) | R@5 | 98% |
| City Centre (fine-tuned) | R@1 | 8% (harder domain gap, see report) |

## Notes on the reference implementation

The instructions this project started from produced a checkpoint that never
actually learned anything (loss pinned at exactly the triplet margin — a textbook
representation-collapse signature) and an evaluation script that didn't exclude
the temporal neighborhood of a query frame before retrieval. Nine independent
bugs across the blur-gate hysteresis, the NetVLAD head, the loss's negative
sampling, the temporal voter, and Windows `DataLoader` multiprocessing were found
and fixed; the full diagnostic trail (with the numbers that exposed each one) is
in `main.tex`.
