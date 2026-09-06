#!/usr/bin/env python3
"""Publish the Qwen3-4B/14B XOR layer/projection adjudication to the existing heatmap client."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result"
P2570 = RESULT / "phase2570_c291073_c299264_holdout_layer_projection_xor/analysis/final.json"
P2575 = RESULT / "phase2575_c327937_c336128_qwen14b_layer0_v_xor_replication/analysis/final.json"
P2576 = RESULT / "phase2576_c336129_c344320_qwen14b_band_projection_xor_scan/analysis/final.json"
OUT = RESULT / "phase2577_c344321_c348416_crossscale_xor_client_heatmap"
ASSET = ROOT / "frontend/public/vis_data/research_kernel/c42641_output_conditioned_crossmodel_field.json"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN = 2577, "C344321-C348416"
PANEL_KEY = "phase2570_2576_crossscale_xor_adjudication"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(path: Path, value: object, compact: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    options = {"ensure_ascii": False, "allow_nan": False}
    if not compact:
        options["indent"] = 2
    path.write_text(json.dumps(value, **options) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def append_memo(result: dict) -> None:
    heading = f"## Phase {PHASE}: 4B/14B关系×值因果选择性的客户端对照热力图（{CAMPAIGN}）"
    if heading in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

{heading} [{stamp}]

**测试原理与测试用例。** 将Phase2570的Qwen3-4B layer0 V独立留出结果，与Phase2575的Qwen3-14B layer0 V、Phase2576的14B四层段V/KV结果追加到现有`output_conditioned_crossmodel_field_heatmap`。每一列是一个模型内功能事件，不把4B/14B坐标号对齐；每行分别显示relation flip、value flip、double base preserve、matched-null flip、XOR margin和是否通过完整门。其目的就是让“高single但高null”的非特异效果在客户端中可见，而不是只画成功率。

$$X=\min(F_R,F_V,B_{{RV}})-\max(N_R,N_V),\qquad
G=\mathbf 1[\min(F_R,F_V,B_{{RV}})\ge .7\land X\ge .2].$$

**结果汇总。** `{json.dumps(result, ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2577_c344321_c348416_crossscale_xor_client_heatmap.py`；客户端数据`frontend/public/vis_data/research_kernel/c42641_output_conditioned_crossmodel_field.json`；生成记录位于`{OUT}`。

**分析与理论进展。** 热力图把4B的选择性layer0 V事件与14B的非复现放在同一证据平面。14B middlelate的single和double表面很高，但matched null同样高，因此客户端不会把它渲染成选择性齿轮。物理参数级embedding/HiddenState/Q/K/V仍由Phase2574六个全坐标面板提供；本面板只增加跨规模因果裁决。

**问题硬伤与结论。** 列是功能事件而非物理坐标，不能与全坐标面板混读；4B与14B样本池虽同任务但eligible集合不同；14B缺nonce×nonce兼容组；可视化是证据检查工具，不增加因果证据等级。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main() -> None:
    four, layer0, bands = load(P2570), load(P2575), load(P2576)
    labels = ["4B L0 V", "14B L0 V"]
    events = [four["layer_adjudication"]["0"]["v"], layer0["xor_adjudication"]]
    for band in ("early", "middle", "middlelate", "late"):
        for projection in ("v", "kv"):
            labels.append(f"14B {band} {projection.upper()}")
            events.append(bands["adjudication"][band][projection])
    def metric(event: dict, name: str) -> float:
        if name == "strong_gate":
            return float(bool(event[name]))
        return float(event[name])
    metrics = ("relation_flip", "value_flip", "double_base_preserve", "matched_null_flip",
               "xor_margin", "strong_gate")
    panel = {
        "key": PANEL_KEY,
        "model": "Qwen3-4B vs Qwen3-14B relation×value causal selectivity",
        "precision": "BF16 causal replacement",
        "coordinate_count": len(labels),
        "coordinate_semantics": "Columns are model-local layer/projection events, not aligned physical coordinates",
        "coordinate_order": "4B layer0 V; 14B layer0 V; 14B relative bands V/KV",
        "coordinate_labels": labels,
        "rows": [{"label": name, "source": "phase2570_2576_crossscale_xor",
                  "coordinate_kind": "functional_event", "preview": True, "phase": PHASE,
                  "values": [metric(event, name) for event in events]} for name in metrics],
    }
    payload = load(ASSET)
    payload["models"] = [item for item in payload["models"] if item.get("key") != PANEL_KEY]
    payload["models"].append(panel)
    payload["phase"] = PHASE
    if CAMPAIGN not in payload.get("campaign", ""):
        payload["campaign"] = f"{payload.get('campaign', '')}; {CAMPAIGN}".strip("; ")
    payload["claim_boundary"] = (
        "Full-coordinate panels retain model-local embedding/HiddenState/Q/K/V axes. The cross-scale XOR panel uses "
        "functional event columns and does not align physical coordinates. Qwen3-4B layer0 V passed on its holdout, "
        "whereas Qwen3-14B layer0 and all relative V/KV bands failed the matched-null-selective XOR gate. "
        "No universal semantic coordinate, minimal gear, or closed language mechanism is claimed."
    )
    save(ASSET, payload, compact=True)
    result = {"phase": PHASE, "campaign": CAMPAIGN, "timestamp": datetime.now().astimezone().isoformat(),
              "asset": str(ASSET), "asset_bytes": ASSET.stat().st_size, "asset_sha256": sha256(ASSET),
              "panel_key": PANEL_KEY, "coordinate_labels": labels, "metrics": list(metrics),
              "values": {name: [metric(event, name) for event in events] for name in metrics},
              "checks": {"ten_events": len(labels) == 10, "six_metrics": len(metrics) == 6,
                         "four_b_gate_visible": events[0]["strong_gate"],
                         "fourteen_b_failures_visible": not any(event["strong_gate"] for event in events[1:]),
                         "matched_null_visible": True, "physical_axes_not_aligned": True,
                         "claim_boundary": True}}
    result["all_checks_passed"] = all(result["checks"].values())
    save(OUT / "analysis/final.json", result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise RuntimeError(result["checks"])


if __name__ == "__main__":
    main()
