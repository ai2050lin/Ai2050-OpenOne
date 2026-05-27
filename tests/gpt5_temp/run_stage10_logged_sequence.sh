#!/usr/bin/env bash
set -euo pipefail

cd /home/rankrank/Documents/OpenOne/Ai2050-OpenOne

OUTPUT_DIR="${OUTPUT_DIR:-results/gpt5_systematic_language_v2_driver595_stage10}"
CASES_PER_CATEGORY="${CASES_PER_CATEGORY:-10}"
CATEGORIES=(
  svo_agent
  passive_agent
  negation_yesno
  conditional
  comparison
  temporal
  recursive_binding
  quantifier
  translation
)

MODELS=("$@")
if [[ ${#MODELS[@]} -eq 0 ]]; then
  MODELS=(qwen3 glm4 deepseek7b)
fi

echo "=== Logged stage10 sequence ==="
date '+%Y-%m-%d %H:%M:%S %Z'
echo "output_dir=${OUTPUT_DIR}"
echo "models=${MODELS[*]}"
echo

for model in "${MODELS[@]}"; do
  for category in "${CATEGORIES[@]}"; do
    checkpoint="${OUTPUT_DIR}/checkpoints/${model}/${category}.json"
    if [[ -f "$checkpoint" ]]; then
      complete="$(python - "$checkpoint" "$CASES_PER_CATEGORY" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
target = int(sys.argv[2])
try:
    d = json.loads(p.read_text())
    print("yes" if d.get("complete") and int(d.get("num_cases", -1)) == target else "no")
except Exception:
    print("no")
PY
)"
      if [[ "$complete" == "yes" ]]; then
        echo "[skip] ${model} ${category}"
        continue
      fi
    fi

    echo "[run] ${model} ${category}"
    OUTPUT_DIR="$OUTPUT_DIR" tests/gpt5_temp/run_logged_language_category.sh \
      "$model" "$category" "$CASES_PER_CATEGORY"
    echo "[done] ${model} ${category}"
    sleep 5
  done
done

python - "$OUTPUT_DIR" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
for f in sorted(out.glob("*_systematic_language.json")):
    d = json.loads(f.read_text())
    o = d["aggregate"]["overall"]["full"]
    print(d["model"], "n", o["n"], "acc", round(o["accuracy"], 3), "mean_margin", round(o["mean_margin"], 3))
PY
