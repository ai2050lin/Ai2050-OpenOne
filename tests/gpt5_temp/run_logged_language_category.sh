#!/usr/bin/env bash
set -euo pipefail

cd /home/rankrank/Documents/OpenOne/Ai2050-OpenOne

MODEL="${1:?usage: $0 MODEL CATEGORY [cases_per_category]}"
CATEGORY="${2:?usage: $0 MODEL CATEGORY [cases_per_category]}"
CASES_PER_CATEGORY="${3:-10}"
OUTPUT_DIR="${OUTPUT_DIR:-results/gpt5_systematic_language_v2_driver595_stage10}"
BATCH_SIZE="${BATCH_SIZE:-1}"
CASE_CHUNK_SIZE="${CASE_CHUNK_SIZE:-1}"
PROGRESS_EVERY="${PROGRESS_EVERY:-2}"
MAX_SECONDS="${MAX_SECONDS:-1800}"
ALLOW_EXISTING_GPU_PROCS="${ALLOW_EXISTING_GPU_PROCS:-0}"

RUN_ID="$(date +%Y%m%d_%H%M%S)_${MODEL}_${CATEGORY}"
LOG_ROOT="results/gpt5_gpu_lock_logs"
LOG_DIR="${LOG_ROOT}/${RUN_ID}"
mkdir -p "$LOG_DIR"

exec > >(tee -a "$LOG_DIR/run.log") 2>&1

echo "=== Logged language category run ==="
date '+%Y-%m-%d %H:%M:%S %Z'
echo "run_id=${RUN_ID}"
echo "model=${MODEL}"
echo "category=${CATEGORY}"
echo "cases_per_category=${CASES_PER_CATEGORY}"
echo "output_dir=${OUTPUT_DIR}"
echo

cleanup() {
  set +e
  if [[ -n "${GPU_MONITOR_PID:-}" ]]; then kill "$GPU_MONITOR_PID" 2>/dev/null || true; fi
  if [[ -n "${KERNEL_FOLLOW_PID:-}" ]]; then kill "$KERNEL_FOLLOW_PID" 2>/dev/null || true; fi
  wait "${GPU_MONITOR_PID:-}" 2>/dev/null || true
  wait "${KERNEL_FOLLOW_PID:-}" 2>/dev/null || true
}
trap cleanup EXIT

snapshot() {
  local label="$1"
  {
    echo "=== ${label} $(date '+%Y-%m-%d %H:%M:%S %Z') ==="
    echo "--- uname ---"
    uname -a
    echo "--- nvidia version ---"
    cat /proc/driver/nvidia/version 2>/dev/null || true
    echo "--- nvidia params ---"
    cat /proc/driver/nvidia/params 2>/dev/null | grep -E 'EnableGpuFirmware|EnableGpuFirmwareLogs|RmMsg|RegistryDwords' || true
    echo "--- nvidia-smi ---"
    timeout 8s nvidia-smi || echo "nvidia-smi timeout_or_failed=$?"
    echo "--- compute apps ---"
    timeout 8s nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits || true
    echo "--- python stack ---"
    python - <<'PY' || true
import sys
print("python", sys.version)
try:
    import torch
    print("torch", torch.__version__, "runtime", torch.version.cuda, "available", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu", torch.cuda.get_device_name(0))
except Exception as exc:
    print("torch_error", repr(exc))
try:
    import transformers
    print("transformers", transformers.__version__)
except Exception as exc:
    print("transformers_error", repr(exc))
try:
    import accelerate
    print("accelerate", accelerate.__version__)
except Exception as exc:
    print("accelerate_error", repr(exc))
PY
    echo "--- relevant processes ---"
    ps -eo pid,ppid,etime,stat,pcpu,pmem,wchan:32,cmd | grep -E 'python tests/gpt5|systematic_language_benchmark|nvidia-smi|cuda|ComfyUI|python main.py' | grep -v grep || true
  } >>"$LOG_DIR/snapshots.log" 2>&1
}

compute_apps="$(timeout 8s nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true)"
if [[ -n "$compute_apps" && "$ALLOW_EXISTING_GPU_PROCS" != "1" ]]; then
  echo "Refusing to run because existing compute GPU process(es) are active:"
  echo "$compute_apps"
  echo
  echo "Close them first, or rerun with ALLOW_EXISTING_GPU_PROCS=1 if this is intentional."
  snapshot "refused_existing_gpu_process"
  exit 20
fi

START_ISO="$(date --iso-8601=seconds)"
snapshot "before"

echo "=== Start kernel log follower ==="
journalctl -b -k -f -o short-iso >"$LOG_DIR/kernel.follow.log" 2>&1 &
KERNEL_FOLLOW_PID="$!"

echo "=== Start lightweight GPU/process monitor ==="
(
  while true; do
    echo "=== monitor $(date '+%Y-%m-%d %H:%M:%S %Z') ==="
    timeout 5s nvidia-smi --query-gpu=timestamp,temperature.gpu,power.draw,memory.used,utilization.gpu,pstate --format=csv,noheader || echo "gpu_query timeout_or_failed=$?"
    timeout 5s nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits || true
    ps -eo pid,etime,stat,pcpu,pmem,wchan:28,cmd | grep -E 'python tests/gpt5|systematic_language_benchmark|nvidia-smi|cuda|python main.py' | grep -v grep || true
    sleep 10
  done
) >"$LOG_DIR/gpu_process_monitor.log" 2>&1 &
GPU_MONITOR_PID="$!"

echo "=== Run benchmark ==="
set +e
PYTHONUNBUFFERED=1 timeout --foreground --kill-after=30s "${MAX_SECONDS}s" \
  python tests/gpt5/systematic_language_benchmark.py "$MODEL" \
    --cases-per-category "$CASES_PER_CATEGORY" \
    --batch-size "$BATCH_SIZE" \
    --case-chunk-size "$CASE_CHUNK_SIZE" \
    --progress-every "$PROGRESS_EVERY" \
    --categories "$CATEGORY" \
    --output-dir "$OUTPUT_DIR" \
    --hard-exit-after-model
RC="$?"
set -e
echo "benchmark_exit_code=${RC}"

snapshot "after"

echo "=== Kernel log since start ==="
journalctl -b -k --since "$START_ISO" --no-pager >"$LOG_DIR/kernel.since-start.log" 2>&1 || true
grep -Ei 'NVRM|Xid|GSP|GPU is probably locked|nvidia|uvm|drm|soft lockup|hung|blocked|timeout|reset|os_acquire' \
  "$LOG_DIR/kernel.since-start.log" >"$LOG_DIR/kernel.since-start.filtered.log" || true

echo "=== Checkpoint status ==="
python - "$OUTPUT_DIR" "$MODEL" "$CATEGORY" <<'PY' | tee "$LOG_DIR/checkpoint_status.log"
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
model = sys.argv[2]
category = sys.argv[3]
p = out / "checkpoints" / model / f"{category}.json"
print("checkpoint", p)
if not p.exists():
    print("exists false")
    raise SystemExit(0)
d = json.loads(p.read_text())
print("exists true")
print("num_cases", d.get("num_cases"))
print("complete", d.get("complete"))
if d.get("aggregate"):
    full = d["aggregate"]["overall"]["full"]
    print("accuracy", full.get("accuracy"))
    print("mean_margin", full.get("mean_margin"))
PY

echo "log_dir=${LOG_DIR}"
exit "$RC"
