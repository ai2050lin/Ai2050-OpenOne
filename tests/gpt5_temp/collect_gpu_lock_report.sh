#!/usr/bin/env bash
set -euo pipefail

cd /home/rankrank/Documents/OpenOne/Ai2050-OpenOne

BOOT_SELECTOR="${1:--1}"
REPORT_ROOT="results/gpt5_gpu_lock_logs/post_reboot_reports"
REPORT_DIR="${REPORT_ROOT}/$(date +%Y%m%d_%H%M%S)_boot_${BOOT_SELECTOR//-/minus}"
mkdir -p "$REPORT_DIR"

exec > >(tee -a "$REPORT_DIR/collect.log") 2>&1

echo "=== GPU lock post-reboot report ==="
date '+%Y-%m-%d %H:%M:%S %Z'
echo "boot_selector=${BOOT_SELECTOR}"
echo

echo "=== Current system ==="
uname -a
cat /proc/driver/nvidia/version 2>/dev/null || true
cat /proc/driver/nvidia/params 2>/dev/null | grep -E 'EnableGpuFirmware|EnableGpuFirmwareLogs|RmMsg|RegistryDwords' || true
timeout 8s nvidia-smi || echo "nvidia-smi timeout_or_failed=$?"
echo

echo "=== Boot list ==="
journalctl --list-boots --no-pager | tail -12 | tee "$REPORT_DIR/boot-list.txt"
echo

echo "=== Previous/current boot kernel log ==="
journalctl -b "$BOOT_SELECTOR" -k --no-pager >"$REPORT_DIR/kernel.full.log" 2>&1 || true
grep -Ei 'NVRM|Xid|GSP|GPU is probably locked|nvidia|uvm|drm|soft lockup|hung|blocked|timeout|reset|os_acquire|BUG|Oops|Call Trace|rcu|watchdog' \
  "$REPORT_DIR/kernel.full.log" >"$REPORT_DIR/kernel.filtered.log" || true
tail -240 "$REPORT_DIR/kernel.filtered.log" | tee "$REPORT_DIR/kernel.filtered.tail.txt"
echo

echo "=== Previous/current boot user journal relevant lines ==="
journalctl -b "$BOOT_SELECTOR" --no-pager >"$REPORT_DIR/journal.full.log" 2>&1 || true
grep -Ei 'systematic_language_benchmark|run_logged_language_category|python tests/gpt5|nvidia-smi|NVRM|Xid|GPU|cuda|torch|timeout|killed|segfault' \
  "$REPORT_DIR/journal.full.log" >"$REPORT_DIR/journal.filtered.log" || true
tail -240 "$REPORT_DIR/journal.filtered.log" | tee "$REPORT_DIR/journal.filtered.tail.txt"
echo

echo "=== Current D-state / GPU processes ==="
ps -eo pid,ppid,etime,stat,pcpu,pmem,wchan:32,cmd | awk '$4 ~ /D/ || $0 ~ /python tests\\/gpt5|systematic_language_benchmark|nvidia-smi|cuda|python main.py/ {print}' \
  | tee "$REPORT_DIR/current-processes.txt"
echo

echo "=== Checkpoint status ==="
python - <<'PY' | tee "results/gpt5_gpu_lock_logs/post_reboot_reports/checkpoint_status_latest.txt" | tee "$REPORT_DIR/checkpoint-status.txt"
import json
from pathlib import Path
for out in [
    Path("results/gpt5_systematic_language_v2_driver595_stage10"),
    Path("results/gpt5_systematic_language_v2_driver570_stage10"),
]:
    if not out.exists():
        continue
    print("==", out)
    for model_dir in sorted((out / "checkpoints").glob("*")) if (out / "checkpoints").exists() else []:
        print("--", model_dir.name)
        for f in sorted(model_dir.glob("*.json")):
            try:
                d = json.loads(f.read_text())
                print(f.stem, d.get("num_cases"), d.get("complete"))
            except Exception as exc:
                print(f.stem, "error", exc)
PY
echo

echo "report_dir=${REPORT_DIR}"
