#!/usr/bin/env bash
# Resume the baseline comparison. Runs that already have all three blur results are skipped;
# encoded descriptors are cached in results/baselines/desc_cache, so a stopped run only
# loses the pass it was in the middle of.
# Usage: bash scripts/run_baselines.sh   (then: python scripts/12_make_tables.py)
cd "$(dirname "$0")/.."
D=$PWD/data
run() {   # run <tag> <dataset> <args...>
    local tag=$1 ds=$2; shift 2
    if [ -f "results/baselines/${tag}_${ds}_blur30.json" ]; then echo "skip $tag $ds"; return; fi
    python3 scripts/11_baseline_eval.py "$@" --dataset "$ds" --data_root "$D" --blur 0 20 30
}
for ds in kitti05 kitti06; do run ba_netvlad_ba_kitti00 $ds --method ba_netvlad --ckpt ckpt/ba_kitti00.pt; done
for ds in kitti05 kitti06; do run mixvpr_4096 $ds --method mixvpr --dim 4096; done
for ds in kitti05 kitti06; do run netvlad_4096 $ds --method netvlad --dim 4096; done
for ds in kitti05 kitti06; do run salad $ds --method salad; done
for ds in kitti05 kitti06; do run mixvpr_128 $ds --method mixvpr --dim 128; done
for ds in gpw_day gpw_night; do run netvlad_4096 $ds --method netvlad --dim 4096; done
python3 scripts/12_make_tables.py > /dev/null
echo ALL_DONE
