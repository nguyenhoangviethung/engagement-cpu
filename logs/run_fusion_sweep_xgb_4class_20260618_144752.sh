#!/usr/bin/env bash
set -euo pipefail
cd /home/bear/engagement-cpu/scripts/..
ln -sfn /home/bear/engagement-cpu/scripts/../logs/fusion_sweep_xgb_4class_20260618_144752.log /home/bear/engagement-cpu/scripts/../logs/latest_fusion_sweep_xgb_4class.log
trap 'status=$?; if [[ $status -ne 0 ]]; then echo "=== fusion sweep xgb failed with exit code $status at $(date) ===" | tee -a /home/bear/engagement-cpu/scripts/../logs/fusion_sweep_xgb_4class_20260618_144752.log; fi' EXIT
echo "=== fusion sweep xgb started at $(date) ===" | tee -a /home/bear/engagement-cpu/scripts/../logs/fusion_sweep_xgb_4class_20260618_144752.log
bash /home/bear/engagement-cpu/scripts/../scripts/lib/run_python.sh --env thesis --workdir /home/bear/engagement-cpu/scripts/..   env PYTHONPATH=/home/bear/engagement-cpu/scripts/../src python -u -m engagement_daisee.multiclass.fusion_sweep_xgb   --output-json /home/bear/engagement-cpu/scripts/../checkpoints/runs/fusion_sweep_xgb_4class_20260618_144752/summary.json   --weight-step 0.05   --min-accuracy 0.75   --min-balanced-accuracy 0.75   --max-accuracy 1.0   --max-balanced-accuracy 1.0   --latency-warmup 30   --latency-iters 200 2>&1 | tee -a /home/bear/engagement-cpu/scripts/../logs/fusion_sweep_xgb_4class_20260618_144752.log
echo "=== fusion sweep xgb finished at $(date) ===" | tee -a /home/bear/engagement-cpu/scripts/../logs/fusion_sweep_xgb_4class_20260618_144752.log
