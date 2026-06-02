#!/usr/bin/env bash
set -euo pipefail

cd /home/rankrank/Documents/OpenOne/Ai2050-OpenOne

export OPENONE_NORMAL_ENV="${OPENONE_NORMAL_ENV:-openone-cuda121}"
export OUTPUT_DIR="${OUTPUT_DIR:-results/gpt5_phase307_cuda_stability_probe}"
export MAX_SECONDS="${MAX_SECONDS:-900}"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"
export PYTORCH_NO_CUDA_MEMORY_CACHING="${PYTORCH_NO_CUDA_MEMORY_CACHING:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PROBE_ATTN_IMPLEMENTATION="${PROBE_ATTN_IMPLEMENTATION:-eager}"
export PROBE_TORCH_DTYPE="${PROBE_TORCH_DTYPE:-bfloat16}"
export PROBE_DEVICE_MAP_AUTO_MODELS="${PROBE_DEVICE_MAP_AUTO_MODELS:-glm4,deepseek7b}"
export PROBE_MAX_GPU_MEMORY="${PROBE_MAX_GPU_MEMORY:-18GiB}"
export PROBE_MAX_CPU_MEMORY="${PROBE_MAX_CPU_MEMORY:-96GiB}"
export ENABLE_SNAPSHOT_NVIDIA_SMI="${ENABLE_SNAPSHOT_NVIDIA_SMI:-1}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "$OPENONE_NORMAL_ENV" ]]; then
  if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
  elif [[ -f /home/rankrank/miniconda3/etc/profile.d/conda.sh ]]; then
    source /home/rankrank/miniconda3/etc/profile.d/conda.sh
  else
    echo "conda was not found; cannot activate ${OPENONE_NORMAL_ENV}" >&2
    exit 2
  fi
  conda activate "$OPENONE_NORMAL_ENV"
fi

MODE="${1:?usage: $0 MODE [phase307 args...]}"
shift || true
MODEL="${1:-qwen3}"
if [[ $# -gt 0 ]]; then shift || true; fi

RUN_ID="$(date +%Y%m%d_%H%M%S)_phase307_${MODE}_${MODEL}"
LOG_ROOT="results/gpt5_gpu_lock_logs"
LOG_DIR="${LOG_ROOT}/${RUN_ID}"
mkdir -p "$LOG_DIR"

exec > >(tee -a "$LOG_DIR/run.log") 2>&1

echo "=== Phase307 conservative logged run ==="
date '+%Y-%m-%d %H:%M:%S %Z'
echo "run_id=${RUN_ID}"
echo "mode=${MODE}"
echo "model=${MODEL}"
echo "output_dir=${OUTPUT_DIR}"
echo "conda_env=${CONDA_DEFAULT_ENV:-none}"
echo "PROBE_TORCH_DTYPE=${PROBE_TORCH_DTYPE}"
echo "PROBE_ATTN_IMPLEMENTATION=${PROBE_ATTN_IMPLEMENTATION}"
echo "CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING}"
echo "PYTORCH_NO_CUDA_MEMORY_CACHING=${PYTORCH_NO_CUDA_MEMORY_CACHING}"
echo "args=$*"

cleanup() {
  set +e
  if [[ -n "${KERNEL_FOLLOW_PID:-}" ]]; then kill "$KERNEL_FOLLOW_PID" 2>/dev/null || true; fi
  wait "${KERNEL_FOLLOW_PID:-}" 2>/dev/null || true
}
trap cleanup EXIT

START_ISO="$(date --iso-8601=seconds)"
{
  echo "=== before $(date '+%Y-%m-%d %H:%M:%S %Z') ==="
  uname -a
  cat /proc/driver/nvidia/version 2>/dev/null || true
  if [[ "$ENABLE_SNAPSHOT_NVIDIA_SMI" == "1" ]]; then
    timeout 8s nvidia-smi || echo "nvidia-smi timeout_or_failed=$?"
  fi
  python - <<'PY' || true
import os, sys
print("python", sys.version)
print("conda_env", os.environ.get("CONDA_DEFAULT_ENV"))
try:
    import torch
    print("torch", torch.__version__, "runtime", torch.version.cuda, "available", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu", torch.cuda.get_device_name(0))
except Exception as exc:
    print("torch_error", repr(exc))
PY
} >>"$LOG_DIR/snapshots.log" 2>&1

journalctl -b -k -f -o short-iso >"$LOG_DIR/kernel.follow.log" 2>&1 &
KERNEL_FOLLOW_PID="$!"

set +e
PYTHONUNBUFFERED=1 timeout --foreground --kill-after=30s "${MAX_SECONDS}s" \
  python tests/gpt5/phase307_cuda_stability_probe.py \
    --mode "$MODE" \
    --model "$MODEL" \
    --output-dir "$OUTPUT_DIR" \
    --hard-exit-after-model \
    "$@"
RC="$?"
set -e
echo "phase307_exit_code=${RC}"

journalctl -b -k --since "$START_ISO" --no-pager >"$LOG_DIR/kernel.since-start.log" 2>&1 || true
grep -Ei 'NVRM|Xid|GSP|GPU is probably locked|nvidia|uvm|drm|soft lockup|hung|blocked|timeout|reset|os_acquire' \
  "$LOG_DIR/kernel.since-start.log" >"$LOG_DIR/kernel.since-start.filtered.log" || true

echo "=== filtered kernel lines ==="
cat "$LOG_DIR/kernel.since-start.filtered.log" || true
echo "log_dir=${LOG_DIR}"
exit "$RC"
