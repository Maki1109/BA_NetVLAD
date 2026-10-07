# BA-NetVLAD vs. NetVLAD, MixVPR, SALAD
Same protocol for every method (scripts/11_baseline_eval.py). Blur = linear motion kernel on the queries only, references sharp. KITTI: positive within 4 m, ±400-frame exclusion; GPW: ref = day_left, identity ground truth. Bold = best in column among the comparable (not fine-tuned on test) methods. † fine-tuned on the 200 GPW test places (not comparable with the zero-shot rows). "Kept" = R@5 at 30 px / R@5 sharp ("–" when sharp R@5 < 30%). "..." = not run yet. CPU timings are indicative only: some runs shared the CPU with another job.

## Table: Recall@N (%), KITTI05

| Method | Dim | Sharp R@1 / R@5 | Blur 20 px R@1 / R@5 | Blur 30 px R@1 / R@5 | Kept |
|---|---|---|---|---|---|
| BA-NetVLAD (ours, KITTI00 model) | 128 | 90.7 / 92.5 | 89.8 / 92.6 | **90.0** / **93.3** | 101% |
| NetVLAD (VGG16, Pitts30K) | 4096 | **91.7** / **94.3** | **91.2** / 93.5 | 89.7 / 93.2 | 99% |
| MixVPR | 4096 | 91.4 / 93.2 | 89.7 / 92.2 | 87.8 / 91.0 | 98% |
| MixVPR (128-D) | 128 | 88.7 / 90.5 | 85.7 / 89.3 | 83.6 / 87.6 | 97% |
| SALAD | 8448 | 90.0 / 93.0 | 88.9 / **93.6** | 86.9 / **93.3** | 100% |

## Table: Recall@N (%), KITTI06

| Method | Dim | Sharp R@1 / R@5 | Blur 20 px R@1 / R@5 | Blur 30 px R@1 / R@5 | Kept |
|---|---|---|---|---|---|
| BA-NetVLAD (ours, KITTI00 model) | 128 | **98.0** / **99.2** | **98.0** / **99.2** | **98.0** / **98.8** | 100% |
| NetVLAD (VGG16, Pitts30K) | 4096 | 96.8 / 98.0 | 96.8 / 98.0 | 96.8 / 98.0 | 100% |
| MixVPR | 4096 | **98.0** / 98.4 | 97.2 / 98.0 | 97.2 / 98.0 | 100% |
| MixVPR (128-D) | 128 | 96.8 / 98.0 | 96.4 / 98.0 | 93.7 / 97.6 | 100% |
| SALAD | 8448 | 97.2 / 98.0 | 96.0 / 98.0 | 94.9 / 98.0 | 100% |

## Table: Recall@N (%), GPW day→day

| Method | Dim | Sharp R@1 / R@5 | Blur 20 px R@1 / R@5 | Blur 30 px R@1 / R@5 | Kept |
|---|---|---|---|---|---|
| BA-NetVLAD (ours, KITTI00 model) | 128 | 12.0 / 46.0 | 12.0 / 35.0 | 10.0 / 34.5 | 75% |
| BA-NetVLAD (ours, GPW fine-tuned)† | 128 | 49.0 / 98.5 | 43.0 / 97.5 | 42.0 / 96.5 | 98% |
| NetVLAD (VGG16, Pitts30K) | 4096 | 38.0 / 96.5 | **33.5** / 89.0 | 25.0 / 81.5 | 84% |
| MixVPR | 4096 | **39.5** / **97.0** | 28.5 / 92.5 | 23.5 / 88.5 | 91% |
| MixVPR (128-D) | 128 | 23.5 / 86.5 | 25.0 / 75.0 | 19.0 / 69.0 | 80% |
| SALAD | 8448 | 34.0 / 96.5 | 29.5 / **93.5** | **27.5** / **92.5** | 96% |

## Table: Recall@N (%), GPW day→night

| Method | Dim | Sharp R@1 / R@5 | Blur 20 px R@1 / R@5 | Blur 30 px R@1 / R@5 | Kept |
|---|---|---|---|---|---|
| BA-NetVLAD (ours, KITTI00 model) | 128 | 2.5 / 10.0 | 3.5 / 11.0 | 3.0 / 11.0 | – |
| BA-NetVLAD (ours, GPW fine-tuned)† | 128 | 42.5 / 98.0 | 28.5 / 78.5 | 24.0 / 67.5 | 69% |
| NetVLAD (VGG16, Pitts30K) | 4096 | 22.5 / 66.0 | 4.5 / 19.0 | 2.5 / 11.0 | 17% |
| MixVPR | 4096 | 26.5 / 70.5 | 11.5 / 36.0 | 9.0 / 25.0 | 35% |
| MixVPR (128-D) | 128 | 19.5 / 58.5 | 6.5 / 25.5 | 4.5 / 18.5 | 32% |
| SALAD | 8448 | **30.5** / **86.5** | **24.5** / **84.0** | **24.0** / **80.5** | 93% |

## Table: Average precision (top-1 score threshold sweep, all queries)

**KITTI05**

| Method | Sharp | Blur 20 px | Blur 30 px |
|---|---|---|---|
| BA-NetVLAD (ours, KITTI00 model) | 0.876 | 0.870 | 0.870 |
| NetVLAD (VGG16, Pitts30K) | 0.902 | **0.892** | **0.871** |
| MixVPR | **0.907** | 0.886 | 0.863 |
| MixVPR (128-D) | 0.877 | 0.841 | 0.782 |
| SALAD | 0.895 | 0.882 | 0.853 |

**KITTI06**

| Method | Sharp | Blur 20 px | Blur 30 px |
|---|---|---|---|
| BA-NetVLAD (ours, KITTI00 model) | 0.973 | **0.972** | **0.974** |
| NetVLAD (VGG16, Pitts30K) | 0.963 | 0.963 | 0.957 |
| MixVPR | **0.974** | 0.960 | 0.952 |
| MixVPR (128-D) | 0.962 | 0.931 | 0.851 |
| SALAD | 0.966 | 0.943 | 0.909 |

**GPW day→day**

| Method | Sharp | Blur 20 px | Blur 30 px |
|---|---|---|---|
| BA-NetVLAD (ours, KITTI00 model) | 0.021 | 0.017 | 0.015 |
| BA-NetVLAD (ours, GPW fine-tuned)† | 0.246 | 0.177 | 0.173 |
| NetVLAD (VGG16, Pitts30K) | 0.160 | **0.127** | 0.062 |
| MixVPR | **0.182** | 0.101 | 0.071 |
| MixVPR (128-D) | 0.075 | 0.073 | 0.037 |
| SALAD | 0.135 | 0.080 | **0.076** |

**GPW day→night**

| Method | Sharp | Blur 20 px | Blur 30 px |
|---|---|---|---|
| BA-NetVLAD (ours, KITTI00 model) | 0.001 | 0.003 | 0.002 |
| BA-NetVLAD (ours, GPW fine-tuned)† | 0.155 | 0.089 | 0.065 |
| NetVLAD (VGG16, Pitts30K) | 0.070 | 0.005 | 0.001 |
| MixVPR | 0.103 | 0.029 | 0.020 |
| MixVPR (128-D) | 0.051 | 0.013 | 0.008 |
| SALAD | **0.113** | **0.074** | **0.071** |

## Table: Cost

| Method | Dim | KB / image (float32) | MB / 1000 images | ms / image (CPU, model only) |
|---|---|---|---|---|
| BA-NetVLAD (ours, KITTI00 model) | 128 | 0.5 | 0.5 | 5 |
| BA-NetVLAD (ours, GPW fine-tuned)† | 128 | 0.5 | 0.5 | 4 |
| NetVLAD (VGG16, Pitts30K) | 4096 | 16.0 | 15.6 | 222 |
| MixVPR | 4096 | 16.0 | 15.6 | 83 |
| MixVPR (128-D) | 128 | 0.5 | 0.5 | 370 |
| SALAD | 8448 | 33.0 | 32.2 | 983 |
