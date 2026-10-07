#!/usr/bin/env bash
# Parallel resume: 4 independent jobs sharing the CPU (descriptor caches make finished passes free).
cd "$(dirname "$0")/.."
D=$PWD/data
E="python3 scripts/11_baseline_eval.py --data_root $D"
OMP_NUM_THREADS=4 $E --method salad --dataset kitti05 --blur 0 20 30 > results/par_salad05.log 2>&1 &
OMP_NUM_THREADS=4 $E --method salad --dataset kitti06 --blur 0 20 30 > results/par_salad06.log 2>&1 &
( OMP_NUM_THREADS=2 $E --method mixvpr --dim 128 --dataset kitti05 --blur 0 20 30
  OMP_NUM_THREADS=2 $E --method mixvpr --dim 128 --dataset kitti06 --blur 0 20 30 ) > results/par_mix128.log 2>&1 &
( OMP_NUM_THREADS=2 $E --method netvlad --dim 4096 --dataset gpw_day --blur 0 20 30
  OMP_NUM_THREADS=2 $E --method netvlad --dim 4096 --dataset gpw_night --blur 0 20 30 ) > results/par_netvlad_gpw.log 2>&1 &
wait
python3 scripts/12_make_tables.py > /dev/null
echo ALL_DONE > results/par_done.flag
