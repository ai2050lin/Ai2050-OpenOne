#!/usr/bin/env bash
set -euo pipefail

cd /home/rankrank/Documents/OpenOne/Ai2050-OpenOne

MODEL="${1:?usage: $0 MODEL [phase289 args...]}"
shift || true

export OPENONE_CONSERVATIVE_ENV="${OPENONE_CONSERVATIVE_ENV:-openone-cu130-py312}"
export OUTPUT_DIR="${OUTPUT_DIR:-results/gpt5_phase289_contract_pilot}"
export MAX_SECONDS="${MAX_SECONDS:-3600}"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"
export PYTORCH_NO_CUDA_MEMORY_CACHING="${PYTORCH_NO_CUDA_MEMORY_CACHING:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export ALLOW_EXISTING_GPU_PROCS="${ALLOW_EXISTING_GPU_PROCS:-0}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "$OPENONE_CONSERVATIVE_ENV" ]]; then
  if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
  elif [[ -f /home/rankrank/miniconda3/etc/profile.d/conda.sh ]]; then
    source /home/rankrank/miniconda3/etc/profile.d/conda.sh
  else
    echo "conda was not found; cannot activate ${OPENONE_CONSERVATIVE_ENV}" >&2
    exit 2
  fi
  conda activate "$OPENONE_CONSERVATIVE_ENV"
fi

if [[ -z "${PROBE_TORCH_DTYPE:-}" ]]; then
  case "$MODEL" in
    deepseek7b) export PROBE_TORCH_DTYPE="bfloat16" ;;
    qwen3|glm4) export PROBE_TORCH_DTYPE="float16" ;;
    *) export PROBE_TORCH_DTYPE="bfloat16" ;;
  esac
fi

RUN_ID="$(date +%Y%m%d_%H%M%S)_phase289_${MODEL}"
LOG_ROOT="results/gpt5_gpu_lock_logs"
LOG_DIR="${LOG_ROOT}/${RUN_ID}"
mkdir -p "$LOG_DIR"

exec > >(tee -a "$LOG_DIR/run.log") 2>&1

echo "=== Phase289 conservative logged run ==="
date '+%Y-%m-%d %H:%M:%S %Z'
echo "run_id=${RUN_ID}"
echo "model=${MODEL}"
echo "output_dir=${OUTPUT_DIR}"
echo "conda_env=${CONDA_DEFAULT_ENV:-none}"
echo "probe_torch_dtype=${PROBE_TORCH_DTYPE}"
echo "args=$*"
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
    cat /proc/driver/nvidia/params 2>/dev/null | grep -E 'EnableGpuFirmware|EnableGpuFirmwareLogs|RegistryDwords' || true
    echo "--- nvidia-smi ---"
    timeout 8s nvidia-smi || echo "nvidia-smi timeout_or_failed=$?"
    echo "--- compute apps ---"
    timeout 8s nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits || true
    echo "--- python stack ---"
    python - <<'PY' || true
import os, sys
print("python", sys.version)
print("conda_env", os.environ.get("CONDA_DEFAULT_ENV"))
print("PROBE_TORCH_DTYPE", os.environ.get("PROBE_TORCH_DTYPE"))
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
PY
    echo "--- relevant processes ---"
    ps -eo pid,ppid,etime,stat,pcpu,pmem,wchan:32,cmd | grep -E 'phase289_contract_scan|python tests/gpt5|nvidia-smi|cuda|ComfyUI|python main.py' | grep -v grep || true
  } >>"$LOG_DIR/snapshots.log" 2>&1
}

compute_apps="$(timeout 8s nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true)"
if [[ -n "$compute_apps" && "$ALLOW_EXISTING_GPU_PROCS" != "1" ]]; then
  echo "Refusing to run because existing compute GPU process(es) are active:"
  echo "$compute_apps"
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
    ps -eo pid,etime,stat,pcpu,pmem,wchan:28,cmd | grep -E 'phase289_contract_scan|python tests/gpt5|nvidia-smi|cuda|python main.py' | grep -v grep || true
    sleep 10
  done
) >"$LOG_DIR/gpu_process_monitor.log" 2>&1 &
GPU_MONITOR_PID="$!"

echo "=== Run Phase289 ==="
set +e
PYTHONUNBUFFERED=1 timeout --foreground --kill-after=30s "${MAX_SECONDS}s" \
  python tests/gpt5/phase289_contract_scan.py "$MODEL" \
    --output-dir "$OUTPUT_DIR" \
    --hard-exit-after-model \
    "$@"
RC="$?"
set -e
echo "phase289_exit_code=${RC}"

snapshot "after"

echo "=== Kernel log since start ==="
journalctl -b -k --since "$START_ISO" --no-pager >"$LOG_DIR/kernel.since-start.log" 2>&1 || true
grep -Ei 'NVRM|Xid|GSP|GPU is probably locked|nvidia|uvm|drm|soft lockup|hung|blocked|timeout|reset|os_acquire' \
  "$LOG_DIR/kernel.since-start.log" >"$LOG_DIR/kernel.since-start.filtered.log" || true

echo "=== Result files ==="
find "$OUTPUT_DIR" -maxdepth 4 -type f | sort | tee "$LOG_DIR/result_files.log" || true
echo "log_dir=${LOG_DIR}"
exit "$RC"
