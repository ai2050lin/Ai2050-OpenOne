#!/usr/bin/env python3
"""Build the C360 browser heatmap payload from the audited full field."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result/phase1874_c340_qwen_full_coordinate_capture"
C360 = ROOT / "tests/glm5/result/phase1894_c360_campaign_synthesis_and_heatmap"
TARGET = ROOT / "frontend/public/vis_data/research_kernel/c360_single_sample_operator_field.json"
ROLES = ("primary", "secondary", "relation", "context", "query", "boundary")
CHECKPOINTS = (0, 12, 24, 36, 37)


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    index = [json.loads(line) for line in (RESULT / "raw/hidden_index.jsonl").read_text(encoding="utf-8").splitlines() if line]
    selected = next(row for row in index if row["panel"] == "apple_factorial" and row["partition"] == "confirmation" and row["factor_a"] == 1 and row["factor_b"] == 1 and row["surface"] == "report" and row["order"] == 1)
    states = np.load(RESULT / "raw/role_states.float16.npy", mmap_mode="r")
    field = np.asarray(states[selected["hidden_index"]], np.float32)
    rows = []
    for checkpoint in CHECKPOINTS:
        source = "c360_embedding" if checkpoint == 0 else "c360_hidden_state"
        checkpoint_label = "embedding" if checkpoint == 0 else "final_norm" if checkpoint == 37 else f"hidden_q{checkpoint}"
        for role_i, role in enumerate(ROLES):
            rows.append({
                "source": source,
                "checkpoint": checkpoint,
                "role": role,
                "label": f"{checkpoint_label} / {role}",
                "values": field[checkpoint, role_i].tolist(),
            })
    final = load(C360 / "analysis/final.json")
    payload = {
        "schema": "c360_single_sample_operator_field.v1",
        "result_type": "single_sample_operator_field_heatmap",
        "phase": 1894,
        "campaign": "C360",
        "model": "Qwen3-4B",
        "case_id": selected["case_id"],
        "dimensions": list(range(2560)),
        "default_coordinates": list(range(12)),
        "total_rows": len(rows),
        "rows": rows,
        "summary": final["headline"]["gates"],
        "coordinate_semantics": "每列是Qwen3-4B的物理激活坐标；q0为词嵌入，q12/q24/q36为HiddenState，q37为final norm。",
        "claim_boundary": "显示一个冻结确认样本的完整2560坐标切片。A/B有局部单样本增益，二阶I未通过联合门；图递归与因果中介未获资格，粗跨模型响应也不等于功能双模拟。",
        "raw_archive": str((RESULT / "raw/role_states.float16.npy").relative_to(ROOT)),
    }
    TARGET.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(json.dumps({"target": str(TARGET), "rows": len(rows), "coordinates": 2560, "bytes": TARGET.stat().st_size}, ensure_ascii=False))


if __name__ == "__main__":
    main()
