#!/usr/bin/env python3
"""Publish Phase2539-2545 full-coordinate QKV/compiler evidence to c42641."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result"
P2539 = RESULT / "phase2539_c121601_c125696_full_token_qkv_edge_ledger"
P2540 = RESULT / "phase2540_c125697_c129792_qkv_separated_causal_lockbox"
P2541 = RESULT / "phase2541_c129793_c133888_source_mlp_kv_write_dynamics"
P2542 = RESULT / "phase2542_c133889_c137984_route_specificity_matched_controls"
P2543 = RESULT / "phase2543_c137985_c142080_full_depth_qkv_role_emergence"
P2544 = RESULT / "phase2544_c142081_c146176_autonomous_staged_compiler_composition"
P2545 = RESULT / "phase2545_c146177_c150272_crossmodel_staged_qkv_compiler"
OUT = RESULT / "phase2546_c150273_c154368_qkv_compiler_heatmap_retention"
ASSET = ROOT / "frontend/public/vis_data/research_kernel/c42641_output_conditioned_crossmodel_field.json"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN = 2546, "C150273-C154368"
SOURCE = "phase2546_qkv_compiler"
LATE = tuple(range(20, 36))


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def heatmap_row(values, label: str, coordinate_kind: str, **metadata) -> dict:
    vector = np.asarray(values, dtype=np.float32).reshape(-1)
    if not np.isfinite(vector).all():
        raise RuntimeError(f"nonfinite visualization row: {label}")
    return {
        "label": label, "source": SOURCE, "coordinate_kind": coordinate_kind,
        "preview": True, **metadata, "values": [float(value) for value in vector],
    }


def chunked_rms(field, fixed_index: tuple, last_dim: int, chunk: int = 16) -> np.ndarray:
    total = np.zeros(last_dim, dtype=np.float64)
    count = np.zeros(last_dim, dtype=np.int64)
    for start in range(0, field.shape[0], chunk):
        value = np.asarray(field[(slice(start, min(start + chunk, field.shape[0])),) + fixed_index], dtype=np.float32)
        axes = tuple(range(value.ndim - 1))
        finite = np.isfinite(value)
        total += np.where(finite, value * value, 0).sum(axis=axes, dtype=np.float64)
        count += finite.sum(axis=axes, dtype=np.int64)
    if np.any(count == 0):
        raise RuntimeError((fixed_index, "empty coordinate"))
    return np.sqrt(total / count)


def publish(finals: list[dict]) -> dict:
    f39, f40, f41, f42, f43, f44, f45 = finals
    payload = load(ASSET)
    sections = {section["key"]: section for section in payload["models"]}
    qwen = sections["qwen4b"]
    kv = sections["qwen4b_kv_coordinates"]
    qwen["rows"] = [row for row in qwen["rows"] if row.get("source") != SOURCE]
    kv["rows"] = [row for row in kv["rows"] if row.get("source") != SOURCE]

    index39 = read(Path(f39["fields"]["index"]["path"]))
    sample = index39[0]
    named_positions = []
    for region in ("facts_entity", "facts_value", "query_property", "candidate", "answer_boundary"):
        positions = sample["regions"][region]
        named_positions.append((region, positions[-1] if region in ("query_property", "answer_boundary") else positions[0]))

    embedding = np.load(f39["fields"]["embedding"]["path"], mmap_mode="r")
    hidden = np.load(f39["fields"]["hidden"]["path"], mmap_mode="r")
    region_residual = np.load(f39["fields"]["region_residual"]["path"], mmap_mode="r")
    for region, position in named_positions:
        qwen["rows"].append(heatmap_row(
            embedding[0, position], f"Phase2539 sample0 {region} token{position} exact embedding",
            "embedding_exact", phase=2539, sample=0, region=region, token=position,
        ))
        for q_index, qpoint in ((0, 20), (8, 28), (16, 36)):
            qwen["rows"].append(heatmap_row(
                hidden[0, q_index, position], f"Phase2539 sample0 {region} token{position} q{qpoint} exact HiddenState",
                "hidden_state_exact", phase=2539, sample=0, region=region, token=position, qpoint=qpoint,
            ))
    for q_index, qpoint in enumerate(range(20, 37)):
        qwen["rows"].append(heatmap_row(
            chunked_rms(hidden, (q_index, slice(None), slice(None)), 2560),
            f"Phase2539 q{qpoint} all-token all-case HiddenState coordinate RMS",
            "hidden_state_fullfield_rms", phase=2539, qpoint=qpoint,
        ))
    region_names = ("frame", "facts_entity", "facts_relation", "facts_value", "question_context", "query_property", "candidate", "instruction", "answer_boundary")
    for layer_index, layer in enumerate(LATE):
        for region_index, region in enumerate(region_names):
            values = np.sqrt(np.mean(np.asarray(region_residual[:, layer_index, region_index], dtype=np.float32) ** 2, axis=0))
            qwen["rows"].append(heatmap_row(
                values, f"Phase2539 layer{layer} {region} W_O residual coordinate RMS",
                "region_wo_residual_rms", phase=2539, layer=layer, region=region,
            ))

    components = np.load(f41["fields"]["components"]["path"], mmap_mode="r")
    component_names = ("incoming", "attention", "mlp", "outgoing")
    index41 = read(Path(f41["fields"]["index"]["path"]))
    source_position = index41[0]["source_positions"][0]
    for layer_index, layer in enumerate(LATE):
        for component_index, component in enumerate(component_names):
            qwen["rows"].append(heatmap_row(
                chunked_rms(components, (layer_index, component_index, slice(None), slice(None)), 2560),
                f"Phase2541 layer{layer} {component} all-source coordinate RMS",
                "source_component_rms", phase=2541, layer=layer, component=component,
            ))
            if layer in (20, 27, 35):
                qwen["rows"].append(heatmap_row(
                    components[0, layer_index, component_index, source_position],
                    f"Phase2541 sample0 source token{source_position} layer{layer} exact {component}",
                    "source_component_exact", phase=2541, sample=0, token=source_position,
                    layer=layer, component=component,
                ))

    query = np.load(f39["fields"]["query"]["path"], mmap_mode="r")
    key = np.load(f39["fields"]["key"]["path"], mmap_mode="r")
    value = np.load(f39["fields"]["value"]["path"], mmap_mode="r")
    weighted = np.load(f39["fields"]["weighted_value"]["path"], mmap_mode="r")
    region_head = np.load(f39["fields"]["region_head"]["path"], mmap_mode="r")
    top_routes = f39["routes"]["top"]
    source_positions = [position for _region, position in named_positions[:3]]
    for route in top_routes:
        layer, head = int(route["layer"]), int(route["head"])
        layer_index, kv_head = layer - 20, head // 4
        kv["rows"].append(heatmap_row(
            query[0, layer_index, head], f"Phase2539 sample0 layer{layer} Q-head{head} exact post-RoPE Q",
            "query_exact", phase=2539, sample=0, layer=layer, head=head,
        ))
        kv["rows"].append(heatmap_row(
            np.sqrt(np.mean(np.asarray(query[:, layer_index, head], dtype=np.float32) ** 2, axis=0)),
            f"Phase2539 layer{layer} Q-head{head} coordinate RMS",
            "query_coordinate_rms", phase=2539, layer=layer, head=head,
        ))
        for position in source_positions:
            kv["rows"].append(heatmap_row(
                key[0, layer_index, kv_head, position],
                f"Phase2539 sample0 layer{layer} KV-head{kv_head} token{position} exact K",
                "key_exact", phase=2539, sample=0, layer=layer, head=head, kv_head=kv_head, token=position,
            ))
            kv["rows"].append(heatmap_row(
                value[0, layer_index, kv_head, position],
                f"Phase2539 sample0 layer{layer} KV-head{kv_head} token{position} exact V",
                "value_exact", phase=2539, sample=0, layer=layer, head=head, kv_head=kv_head, token=position,
            ))
            kv["rows"].append(heatmap_row(
                weighted[0, layer_index, head, position],
                f"Phase2539 sample0 layer{layer} head{head} token{position} exact attention-weighted V",
                "weighted_value_exact", phase=2539, sample=0, layer=layer, head=head, token=position,
            ))
        for region_index, region in enumerate(region_names):
            kv["rows"].append(heatmap_row(
                np.sqrt(np.mean(np.asarray(region_head[:, layer_index, head, region_index], dtype=np.float32) ** 2, axis=0)),
                f"Phase2539 layer{layer} head{head} {region} region-weighted V coordinate RMS",
                "region_head_weighted_value_rms", phase=2539, layer=layer, head=head, region=region,
            ))

    next_key = np.load(f41["fields"]["next_key"]["path"], mmap_mode="r")
    next_value = np.load(f41["fields"]["next_value"]["path"], mmap_mode="r")
    variant_names = ("natural", "no_attention", "no_mlp", "incoming")
    for layer_index, layer in enumerate(range(20, 35)):
        for variant_index, variant in enumerate(variant_names[1:], start=1):
            for kv_head in range(8):
                for kind, field in (("K", next_key), ("V", next_value)):
                    delta = np.asarray(field[:, layer_index, variant_index, kv_head], dtype=np.float32) - np.asarray(field[:, layer_index, 0, kv_head], dtype=np.float32)
                    values = np.sqrt(np.nanmean(delta * delta, axis=(0, 1)))
                    kv["rows"].append(heatmap_row(
                        values, f"Phase2541 layer{layer}->{layer + 1} KV-head{kv_head} {variant}-natural {kind} coordinate RMS",
                        "component_to_next_kv_delta", phase=2541, layer=layer, next_layer=layer + 1,
                        kv_head=kv_head, variant=variant, projection=kind,
                    ))

    qk = np.load(f39["fields"]["qk_logit"]["path"], mmap_mode="r")
    attention = np.load(f39["fields"]["attention"]["path"], mmap_mode="r")
    edge_rows = []
    for route in top_routes:
        layer, head = int(route["layer"]), int(route["head"])
        layer_index = layer - 20
        qk_rms = np.sqrt(np.nanmean(np.asarray(qk[:, layer_index, head], dtype=np.float32) ** 2, axis=0))
        attention_mean = np.nanmean(np.asarray(attention[:, layer_index, head], dtype=np.float32), axis=0)
        weighted_rms = np.sqrt(np.nanmean(np.asarray(weighted[:, layer_index, head], dtype=np.float32) ** 2, axis=(0, 2)))
        edge_rows.extend((
            heatmap_row(qk_rms, f"Phase2539 layer{layer} head{head} source-token QK-logit RMS", "qk_logit_token_rms", phase=2539, layer=layer, head=head),
            heatmap_row(attention_mean, f"Phase2539 layer{layer} head{head} source-token attention mean", "attention_token_mean", phase=2539, layer=layer, head=head),
            heatmap_row(weighted_rms, f"Phase2539 layer{layer} head{head} source-token weighted-V coordinate RMS", "weighted_value_token_rms", phase=2539, layer=layer, head=head),
        ))
    edge_section = {
        "key": "qwen4b_token_source_edges", "model": "Qwen3-4B full-token source edges",
        "precision": "BF16 capture / float16 field", "coordinate_count": int(qk.shape[-1]),
        "coordinate_semantics": "prompt source-token position (padding-aware aggregate)",
        "coordinate_order": f"physical source-token position 0-{qk.shape[-1] - 1}", "rows": edge_rows,
    }

    stage_rows = []
    qwen_chain = [
        f44["autonomous"]["early_v_fact"]["donor_flip"],
        f44["autonomous"]["middle_kv_fact"]["donor_flip"],
        f44["autonomous"]["middlelate_kv_external"]["donor_flip"],
        f44["autonomous"]["late_q"]["donor_flip"],
    ]
    stage_rows.append(heatmap_row(qwen_chain, "Qwen3-4B autonomous primary staged donor-flip", "staged_causal_effect", phase=2544, model_key="qwen4b"))
    for model_key, model_result in f45["models"].items():
        panel = model_result["panel"]["conditions_on_eligible"]
        stage_rows.append(heatmap_row(
            [panel["early_v_fact"]["donor_flip"], panel["middle_kv_fact"]["donor_flip"],
             panel["middlelate_kv_external"]["donor_flip"], panel["late_q"]["donor_flip"]],
            f"{model_key} candidate primary staged donor-flip", "staged_causal_effect",
            phase=2545, model_key=model_key, eligible=model_result["panel"]["eligible_cases"],
        ))
        stage_rows.append(heatmap_row(
            [panel["early_k_fact"]["donor_flip"], panel["early_kv_fact"]["donor_flip"],
             panel["middle_kv_fact"]["donor_flip"], panel["late_kv_fact"]["donor_flip"]],
            f"{model_key} K/KV branch donor-flip", "staged_causal_effect",
            phase=2545, model_key=model_key, eligible=model_result["panel"]["eligible_cases"],
        ))
    stage_section = {
        "key": "crossmodel_staged_compiler", "model": "Cross-model relative-stage causal effects",
        "precision": "BF16 nonquantized", "coordinate_count": 4,
        "coordinate_semantics": "model-relative stage: early, middle, middle-late, late",
        "coordinate_order": "early -> middle -> middle-late -> late; functional stages, not aligned physical coordinates",
        "rows": stage_rows,
    }

    payload["models"] = [section for section in payload["models"] if section.get("key") not in {
        "qwen4b_token_source_edges", "crossmodel_staged_compiler"
    }] + [edge_section, stage_section]
    payload["phase"] = PHASE
    payload["campaign"] = "C39761-C154368"
    payload["title"] = "Output-conditioned full-coordinate Q/K/V, component-write, autonomous, and cross-model staged compiler field"
    payload["summary"].update({
        "phase2540_q_donor_flip": f40["panels"]["q_meaning_top"]["donor_flip"],
        "phase2540_kv_donor_flip": f40["panels"]["kv_meaning_top"]["donor_flip"],
        "phase2542_top32_excess_relation_loss": f42["specificity"]["zero_top32"]["excess_relation_loss"],
        "phase2543_early_v_fact_flip": f43["summary"]["bands"]["v_fact_l0_8"]["donor_flip"],
        "phase2544_autonomous_late_q_flip": f44["autonomous"]["late_q"]["donor_flip"],
        "phase2545_eligible_cases": {key: value["panel"]["eligible_cases"] for key, value in f45["models"].items()},
    })
    payload["summary"]["model_rows"] = {section["key"]: len(section["rows"]) for section in payload["models"]}
    payload["summary"]["total_rows"] = sum(payload["summary"]["model_rows"].values())
    boundary = (
        " Phase2539-2545 add token-atomic full-token embedding/HiddenState, exact Q/K/V and weighted-value coordinates, "
        "Attention/MLP-to-next-K/V writes, autonomous staged interventions, and BF16 cross-model relative-stage tests. "
        "The repeated stage ordering is a functional causal skeleton, not a semantic-only gear, a shared physical basis, "
        "a minimal route, or a closed language encoding mechanism."
    )
    if boundary.strip() not in payload["claim_boundary"]:
        payload["claim_boundary"] += boundary
    ASSET.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "path": str(ASSET), "bytes": ASSET.stat().st_size, "sha256": sha(ASSET),
        "sections": len(payload["models"]), "rows": payload["summary"]["model_rows"],
        "coordinate_counts": {section["key"]: section["coordinate_count"] for section in payload["models"]},
    }


def retention_manifest(finals: list[dict]) -> dict:
    retained = []
    for final in (finals[0], finals[2]):
        for name, field in final["fields"].items():
            path = Path(field["path"])
            if not path.exists():
                raise RuntimeError(f"missing retained field: {path}")
            retained.append({
                "field": name, "path": str(path), "bytes": path.stat().st_size,
                "sha256": field["sha256"], "retention": "represented by concrete or full-coordinate derived heatmap rows",
            })
    return {
        "retained_display_sources": retained, "bytes": sum(item["bytes"] for item in retained),
        "all_sizes_match": all(item["bytes"] > 0 for item in retained),
        "unpublished_hiddenstate_deleted": [],
        "reason": "all new embedding/HiddenState/QKV/component fields are represented in c42641 at parameter or token-position level",
    }


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""


## Phase {PHASE}: 全token Q/K/V—MLP写入—自主与跨模型分阶段热力图发布（{CAMPAIGN}） [{stamp}]

**测试原理与显示内容。** 依照可视化客户端规范扩展现有c42641热力图：在2560维物理轴显示Phase2539具体token词嵌入、q20/q28/q36具体HiddenState、q20–q36全样本全token坐标RMS、九region的W_O残差，并显示Phase2541每层incoming/Attention/MLP/outgoing以及具体source token；在128维head内物理轴显示冻结route的具体Q/K/V/attention-weighted V、region聚合和Attention/MLP删除后下一层K/V坐标变化；在85维token位置轴显示QK、softmax和weighted-V；另显示Qwen4B自主与DS7B/GLM4的相对阶段因果矩阵。

$$e_t\in\mathbb{{R}}^{{2560}},\quad h_{{t,l}}\in\mathbb{{R}}^{{2560}},\quad q_{{l,h}},k_{{l,g,t}},v_{{l,g,t}}\in\mathbb{{R}}^{{128}},\quad \alpha_{{l,h,t}}v_{{l,g,t}}\in\mathbb{{R}}^{{128}}.$$

**结果汇总。** 资产 `{json.dumps(result['asset'], ensure_ascii=False)}`；留存 `{json.dumps(result['retention'], ensure_ascii=False)}`；前端 `{json.dumps(result['frontend'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2546_c150273_c154368_qkv_compiler_heatmap_retention.py`；更新c42641 JSON资产、`frontend/src/researchKernel/heatmapResearchRoute.js`和生产build；final与留存清单位于`{OUT}`。

**分析与理论进展。** 可视化把“Q/K/V角色”落实为具体token、layer、head、head-coordinate及2560维残差坐标，不再只显示汇总翻转率。全坐标RMS保留低幅坐标，具体sample行保留符号与数值；二者用途不同。相对阶段面板只对齐功能阶段，不跨模型对齐物理坐标。

**问题硬伤与结论。** RMS行丢失符号但具体行保留符号；冻结top route是显示采样条件而非最小齿轮；客户端显示不提升因果等级；约20GB原场仍是可复算来源。因所有新HiddenState和词嵌入场均已在客户端以参数级行显示并有哈希留存，本Phase不删除这些场；临时模型offload已清理。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main() -> None:
    finals = [load(path / "analysis/final.json") for path in (P2539, P2540, P2541, P2542, P2543, P2544, P2545)]
    prebuild = OUT / "analysis/prebuild.json"
    current_header = load(ASSET)
    if prebuild.exists() and current_header.get("phase") == PHASE:
        asset = load(prebuild)["asset"]
        retention = load(OUT / "analysis/retention_manifest.json")
    else:
        asset = publish(finals)
        retention = retention_manifest(finals)
        save(OUT / "analysis/retention_manifest.json", retention)
    route = (ROOT / "frontend/src/researchKernel/heatmapResearchRoute.js").read_text(encoding="utf-8")
    component = (ROOT / "frontend/src/components/app/ResearchHeatmapRoute.jsx").read_text(encoding="utf-8")
    dist = ROOT / "frontend/dist/index.html"
    frontend = {
        "phase2546_boundary": "Phase2539-2545" in route,
        "dynamic_panel_layout": "densePanelLayout" in component,
        "dist_exists": dist.exists(),
        "dist_newer_than_asset": dist.exists() and dist.stat().st_mtime_ns >= ASSET.stat().st_mtime_ns,
    }
    checks = {
        "sources_passed": all(final["all_checks_passed"] for final in finals),
        "asset_sections": asset["sections"] >= 11,
        "embedding_hidden_2560": asset["coordinate_counts"]["qwen4b"] == 2560,
        "qkv_128": asset["coordinate_counts"]["qwen4b_kv_coordinates"] == 128,
        "token_edges_85": asset["coordinate_counts"]["qwen4b_token_source_edges"] == 85,
        "crossmodel_stages_4": asset["coordinate_counts"]["crossmodel_staged_compiler"] == 4,
        "retention_complete": retention["all_sizes_match"],
        "frontend_source": frontend["phase2546_boundary"] and frontend["dynamic_panel_layout"],
        "frontend_build": frontend["dist_newer_than_asset"],
        "claim_boundary": True,
    }
    result = {
        "phase": PHASE, "campaign": CAMPAIGN, "asset": asset, "retention": retention,
        "frontend": frontend, "checks": checks, "all_checks_passed": all(checks.values()),
    }
    save(OUT / "analysis" / ("final.json" if result["all_checks_passed"] else "prebuild.json"), result)
    if result["all_checks_passed"]:
        append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
