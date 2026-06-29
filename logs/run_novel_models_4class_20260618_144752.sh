#!/usr/bin/env bash
set -euo pipefail
cd /home/bear/engagement-cpu/scripts/..
mkdir -p /home/bear/engagement-cpu/scripts/../checkpoints/runs/novel_models_4class_20260618_144752
ln -sfn /home/bear/engagement-cpu/scripts/../logs/novel_models_4class_20260618_144752.log /home/bear/engagement-cpu/scripts/../logs/latest_novel_models_4class.log
trap 'status=$?; if [[ $status -ne 0 ]]; then echo "=== novel models failed with exit code $status at $(date) ===" | tee -a /home/bear/engagement-cpu/scripts/../logs/novel_models_4class_20260618_144752.log; fi' EXIT
echo "=== novel models pipeline started at $(date) ===" | tee -a /home/bear/engagement-cpu/scripts/../logs/novel_models_4class_20260618_144752.log
while tmux has-session -t engagement_accuracy_boost_xgb_4class 2>/dev/null; do
  echo "Waiting for resource session engagement_accuracy_boost_xgb_4class at $(date)" | tee -a /home/bear/engagement-cpu/scripts/../logs/novel_models_4class_20260618_144752.log
  sleep 30
done

run_method() {
  local method="$1"
  shift
  echo "=== method $method started at $(date) ===" | tee -a /home/bear/engagement-cpu/scripts/../logs/novel_models_4class_20260618_144752.log
  set +e
  bash /home/bear/engagement-cpu/scripts/../scripts/lib/run_python.sh --env thesis --workdir /home/bear/engagement-cpu/scripts/..     env PYTHONPATH=/home/bear/engagement-cpu/scripts/../src python -u -m engagement_daisee.multiclass.novel_models_4class     --method "$method"     --manifest /home/bear/engagement-cpu/scripts/../data/processed/runs/daisee_4class_final_dataset/feature_manifest.csv     --output-dir /home/bear/engagement-cpu/scripts/../checkpoints/runs/novel_models_4class_20260618_144752/"$method"     --report-json /home/bear/engagement-cpu/scripts/../checkpoints/runs/novel_models_4class_20260618_144752/"${method}"/summary.json     --cpu-threads 4 --latency-warmup 20 --latency-iters 100 "$@" 2>&1 | tee -a /home/bear/engagement-cpu/scripts/../logs/novel_models_4class_20260618_144752.log
  local status="${PIPESTATUS[0]}"
  set -e
  if [[ "$status" -eq 0 ]]; then
    echo "=== method $method finished at $(date) ===" | tee -a /home/bear/engagement-cpu/scripts/../logs/novel_models_4class_20260618_144752.log
  else
    echo "=== method $method failed with exit code $status at $(date); continuing ===" | tee -a /home/bear/engagement-cpu/scripts/../logs/novel_models_4class_20260618_144752.log
    overall_status=1
  fi
}

overall_status=0
run_method ordinal --n-estimators 500 --round-step 25
run_method minirocket --num-kernels 128
run_method deep_forest --n-estimators 120 --folds 3
echo "=== novel models pipeline finished at $(date) ===" | tee -a /home/bear/engagement-cpu/scripts/../logs/novel_models_4class_20260618_144752.log
exit "$overall_status"
