#!/usr/bin/env python3
"""Publish Phase2568-2573 full-coordinate and causal fields to the existing client heatmap route."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P2568 = RESULT / "phase2568_c276737_c284928_relation_value_factorial_fullfield"
P2572 = RESULT / "phase2572_c307457_c315648_layer0_v_coordinate_partition/analysis/final.json"
P2573 = RESULT / "phase2573_c315649_c323840_head5_anchored_subset_lattice"
ASSET = ROOT / "frontend/public/vis_data/research_kernel/c42641_output_conditioned_crossmodel_field.json"
OUT = RESULT / "phase2574_c323841_c327936_relation_value_client_heatmap"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN = 2574, "C323841-C327936"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(path: Path, value: object, *, compact: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {"ensure_ascii": False, "allow_nan": False}
    if not compact:
        kwargs["indent"] = 2
    path.write_text(json.dumps(value, **kwargs) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def values(array: np.ndarray) -> list[float]:
    return np.round(np.asarray(array, dtype=np.float32), 6).tolist()


def coordinate_panel(key: str, title: str, kind: str, layers: list[int], include_main: bool) -> dict:
    path = P2568 / f"fields/{kind}_factorial_full_coordinates.npz"
    with np.load(path) as data:
        mean = np.asarray(data["mean"], dtype=np.float32)
        consistency = np.asarray(data["sign_consistency"], dtype=np.float32)
        labels = [str(item) for item in data["group_labels"].tolist()]
        regions = [str(item) for item in data["regions"].tolist()]
    label_index = {label: index for index, label in enumerate(labels)}
    region_index = {region: index for index, region in enumerate(regions)}
    forms = (("natural", "natural"), ("natural", "nonce"), ("nonce", "natural"), ("nonce", "nonce"))
    rows = []
    if include_main:
        for relation_form, value_form in forms:
            prefix = f"r{relation_form}_v{value_form}"
            for effect, region in (("relation_main", "query_relation"), ("value_main", "query_value")):
                group = label_index[f"{prefix}_{effect}"]
                rows.append({"label": f"embedding / {prefix} / {effect} / {region}",
                    "source": "phase2568_factorial", "coordinate_kind": "embedding",
                    "preview": True, "phase": 2568, "layer": 0, "event": region,
                    "effect": effect, "form": prefix, "values": values(mean[group, 0, region_index[region]])})
    for relation_form, value_form in forms:
        prefix = f"r{relation_form}_v{value_form}"
        group = label_index[f"{prefix}_relation_x_value"]
        for layer in layers:
            for region in ("query_relation", "query_value", "answer_boundary"):
                rows.append({"label": f"{kind} / {prefix} / interaction / L{layer} / {region}",
                    "source": "phase2568_factorial", "coordinate_kind": kind,
                    "preview": layer in (layers[0], layers[-1]), "phase": 2568, "layer": layer,
                    "event": region, "effect": "relation_x_value", "form": prefix,
                    "values": values(mean[group, layer, region_index[region]])})
        last_layer = layers[-1]
        rows.append({"label": f"{kind} sign consistency / {prefix} / L{last_layer} / answer_boundary",
            "source": "phase2568_sign_consistency", "coordinate_kind": f"{kind}_sign_consistency",
            "preview": False, "phase": 2568, "layer": last_layer, "event": "answer_boundary",
            "effect": "relation_x_value", "form": prefix,
            "values": values(consistency[group, last_layer, region_index["answer_boundary"]])})
    return {"key": key, "model": title, "precision": "BF16 forward / float32 full-coordinate mean",
            "coordinate_count": int(mean.shape[-1]),
            "coordinate_semantics": f"Qwen3-4B model-local physical {kind} projection/activation coordinates; signed factorial effects",
            "coordinate_order": "physical", "rows": rows}


def causal_panels() -> list[dict]:
    p2572, p2573 = load(P2572), load(P2573 / "analysis/final.json")
    head_rows = [
        {"label": "single-head XOR core", "source": "phase2572_single_head", "coordinate_kind": "kv_head",
         "preview": True, "phase": 2572, "values": [p2572["single_heads"][str(h)]["xor_core"] for h in range(8)]},
        {"label": "leave-one-head-out XOR core", "source": "phase2572_leave_one_out", "coordinate_kind": "kv_head",
         "preview": True, "phase": 2572, "values": [p2572["leave_one_head_out"][str(h)]["xor_core"] for h in range(8)]},
        {"label": "frozen seven-head candidate membership", "source": "phase2573_frozen_candidate", "coordinate_kind": "kv_head",
         "preview": True, "phase": 2573, "values": [1.0 if h in p2573["frozen_candidate"]["heads"] else 0.0 for h in range(8)]},
    ]
    lattice = load(P2573 / "analysis/lattice.json")
    for label, row in sorted(lattice.items(), key=lambda item: (len(item[1]["heads"]), item[0])):
        head_rows.append({"label": f"{label} / discovery XOR core {row['xor_core']:.3f}",
            "source": "phase2573_h5_subset_lattice", "coordinate_kind": "kv_head_subset",
            "preview": False, "phase": 2573,
            "values": [row["xor_core"] if h in row["heads"] else 0.0 for h in range(8)]})
    block_rows = []
    for metric in ("relation_flip", "value_flip", "double_base_preserve", "xor_core"):
        block_rows.append({"label": f"single 32-coordinate block / {metric}",
            "source": "phase2572_coordinate_partition", "coordinate_kind": "v_coordinate_block",
            "preview": True, "phase": 2572,
            "values": [p2572["single_blocks"][str(block)][metric] for block in range(32)]})
    return [
        {"key": "phase2572_2573_layer0_v_head_lattice", "model": "Qwen3-4B layer0 V causal head lattice",
         "precision": "BF16 causal replacement", "coordinate_count": 8,
         "coordinate_semantics": "KV-head index; values are behavior rates or subset membership, not activation coordinates",
         "coordinate_order": "physical KV-head order", "rows": head_rows},
        {"key": "phase2572_layer0_v_32_coordinate_blocks", "model": "Qwen3-4B layer0 V exhaustive 32-coordinate blocks",
         "precision": "BF16 causal replacement", "coordinate_count": 32,
         "coordinate_semantics": "Each cell is one disjoint contiguous 32-coordinate block covering all 1024 V coordinates",
         "coordinate_order": "physical block order", "rows": block_rows},
    ]


def append_memo(result: dict) -> None:
    heading = f"## Phase {PHASE}: 关系×值全坐标与因果联盟的客户端热力图（{CAMPAIGN}）"
    if heading in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

{heading} [{stamp}]

**测试原理与测试用例。** 将Phase2568—2573的重要结果追加到现有`output_conditioned_crossmodel_field_heatmap`客户端数据源。不是只写Top-K：embedding/HiddenState面板保留全部2560物理坐标，Q保留全部4096投影坐标，K/V各保留全部1024坐标；行覆盖自然/nonce四种词面、关系×值二阶交互、多个层与query-relation/query-value/answer-boundary region。另加入8个KV-head、32个无遗漏坐标块和128个H5锚定子集的因果热力图。

$$\Delta_{{R\times V}}F=F_{{11}}-F_{{10}}-F_{{01}}+F_{{00}},\qquad
\mathcal V=\bigsqcup_{{b=0}}^{{31}}B_b.$$

**结果汇总。** `{json.dumps(result, ensure_ascii=False)}`。

**相关文件。** 生成脚本`tests/glm5/phase2574_c323841_c327936_relation_value_client_heatmap.py`；客户端数据`frontend/public/vis_data/research_kernel/c42641_output_conditioned_crossmodel_field.json`；生成记录位于`{OUT}`。

**分析与理论进展。** 客户端现在可以按物理顺序显示embedding、HiddenState和Q/K/V的具体坐标级数值，同时把观察场与因果head/block联盟放在独立面板，避免把不同坐标轴混为一谈。Phase2568的非加性交互是观察量；Phase2570—2573的翻转率是干预量，两者证据等级保持分离。

**问题硬伤与结论。** region内仍是token均值；显示的是跨样本析因均值而不是每个原始prompt场；前端JSON会增加体积；坐标值是激活/投影输出而非模型权重。热力图支持检查与提出新假设，不把可视化本身当作机制证明。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main() -> None:
    payload = load(ASSET)
    replacements = {
        "phase2568_hidden_factorial": coordinate_panel("phase2568_hidden_factorial", "Qwen3-4B embedding + HiddenState relation×value field", "hidden", [0, 1, 9, 18, 27, 36], True),
        "phase2568_q_factorial": coordinate_panel("phase2568_q_factorial", "Qwen3-4B Q relation×value field", "q", [0, 8, 17, 26, 35], False),
        "phase2568_k_factorial": coordinate_panel("phase2568_k_factorial", "Qwen3-4B K relation×value field", "k", [0, 8, 17, 26, 35], False),
        "phase2568_v_factorial": coordinate_panel("phase2568_v_factorial", "Qwen3-4B V relation×value field", "v", [0, 8, 17, 26, 35], False),
    }
    for panel in causal_panels():
        replacements[panel["key"]] = panel
    payload["models"] = [panel for panel in payload["models"] if panel.get("key") not in replacements]
    payload["models"].extend(replacements.values())
    payload["phase"] = PHASE
    payload["campaign"] = f"{payload.get('campaign', '')}; {CAMPAIGN}".strip("; ")
    payload["title"] = "Full-coordinate Q/K/V, relation×value factorial field, and causal coordinate alliances"
    payload["claim_boundary"] = (
        "Panels use separate model-local physical axes. Phase2568 rows are full-coordinate factorial means over behavior-qualified "
        "quartets; Phase2572-2573 rows are causal rates over selected compatible quartets. The layer0 V result is distributed "
        "across all eight KV heads, and a seven-head candidate failed cross-entity validation. These are not model weights, "
        "universal semantic coordinates, a minimal language gear, or a closed language mechanism."
    )
    save(ASSET, payload, compact=True)
    panels = [{"key": panel["key"], "coordinate_count": panel["coordinate_count"],
               "rows": len(panel["rows"])} for panel in replacements.values()]
    result = {"phase": PHASE, "campaign": CAMPAIGN, "timestamp": datetime.now().astimezone().isoformat(),
              "asset": str(ASSET), "asset_bytes": ASSET.stat().st_size, "asset_sha256": sha(ASSET),
              "panels": panels, "full_coordinate_axes": {"embedding_hidden": 2560, "q": 4096, "k": 1024, "v": 1024},
              "checks": {"six_panels": len(replacements) == 6,
                         "embedding_hidden_specific_coordinates": replacements["phase2568_hidden_factorial"]["coordinate_count"] == 2560,
                         "qkv_specific_coordinates": [replacements[f"phase2568_{kind}_factorial"]["coordinate_count"]
                                                      for kind in ("q", "k", "v")] == [4096, 1024, 1024],
                         "all_32_blocks": len(replacements["phase2572_layer0_v_32_coordinate_blocks"]["rows"][0]["values"]) == 32,
                         "all_128_subsets": len(replacements["phase2572_2573_layer0_v_head_lattice"]["rows"]) == 131,
                         "claim_boundary": True}}
    result["all_checks_passed"] = all(result["checks"].values())
    save(OUT / "analysis/final.json", result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise RuntimeError(result["checks"])


if __name__ == "__main__":
    main()
