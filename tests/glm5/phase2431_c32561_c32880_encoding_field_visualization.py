#!/usr/bin/env python3
"""Publish the Phase2423-2430 full-coordinate atlas to the existing heatmap client."""
from __future__ import annotations

import gc
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P2423 = RESULT / "phase2423_c30001_c30320_semantic_validity_behavior_contract"
P2424 = RESULT / "phase2424_c30321_c30640_semantic_validity_multievent_fullfield"
P2425 = RESULT / "phase2425_c30641_c30960_semantic_specific_interaction_atlas"
P2426 = RESULT / "phase2426_c30961_c31280_coordinate_identity_multinull"
P2427 = RESULT / "phase2427_c31281_c31600_intrinsic_coordinate_cooperation"
P2428 = RESULT / "phase2428_c31601_c31920_crosslayer_path_consistency"
P2429 = RESULT / "phase2429_c31921_c32240_direct_composed_relation_algebra"
P2430 = RESULT / "phase2430_c32241_c32560_output_coordinate_compilation"
OUT = RESULT / "phase2431_c32561_c32880_encoding_field_visualization"
ASSET = ROOT / "frontend/public/vis_data/research_kernel/c32561_semantic_encoding_output_field.json"
BINARY = ROOT / "frontend/public/vis_data/research_kernel/c32561_semantic_encoding_output_field.float32.npy"
DIST = ROOT / "frontend/dist/index.html"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE = 2431
CAMPAIGN = "C32561-C32880"
DIM = 2560

sys.path.insert(0, str(TESTS))
import phase2389_c19121_c19440_crossmodel_autonomous_capability as capability  # noqa: E402


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def save_if_changed(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n"
    if not path.exists() or path.read_text(encoding="utf-8") != content:
        path.write_text(content, encoding="utf-8")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def close(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def selected_case(rows: list[dict]) -> tuple[int, dict]:
    for index, row in enumerate(rows):
        if (row["family"] == "taxonomy" and int(row["unit"]) == 6 and row["language"] == "zh"
                and row["surface"] == "natural" and int(row["direction"]) == 0
                and row["variant"] == "valid" and row["query_role"] == "target"):
            return index, row
    raise RuntimeError("selected visualization case not found")


def embedding_parameters(row: dict) -> tuple[np.ndarray, dict]:
    path = OUT / "derived/selected_embedding_parameters.float32.npy"
    meta_path = OUT / "derived/selected_embedding_parameters.json"
    if path.exists() and meta_path.exists():
        return np.load(path), json.loads(meta_path.read_text(encoding="utf-8"))
    model, tokenizer, label = capability.load_model("qwen4b")
    event = next(item for item in row["event_tokens"] if item["event"] == "query_end")
    prompt_token_id = int(row["prompt_ids"][int(event["token_index"])])
    target_id, foil_id = int(row["target_ids"][0]), int(row["foil_ids"][0])
    with torch.inference_mode():
        input_weight = model.get_input_embeddings().weight
        output_weight = model.get_output_embeddings().weight
        values = torch.stack((input_weight[prompt_token_id], output_weight[target_id], output_weight[foil_id])).float().cpu().numpy()
    meta = {
        "model": label,
        "prompt_token_id": prompt_token_id,
        "prompt_token": tokenizer.decode([prompt_token_id]),
        "target_token_id": target_id,
        "target_token": tokenizer.decode([target_id]),
        "foil_token_id": foil_id,
        "foil_token": tokenizer.decode([foil_id]),
        "shape": list(values.shape),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, values.astype(np.float32))
    save(meta_path, meta)
    del model, tokenizer, input_weight, output_weight
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return values, meta


def add(output: list[dict], values: np.ndarray, label: str, source: str, coordinate_kind: str,
        component: str = "", layer: int | None = None, event: str = "", family: str = "",
        preview: bool = False, **metadata: Any) -> None:
    vector = np.asarray(values, dtype=np.float32).reshape(-1)
    if vector.shape != (DIM,) or not np.isfinite(vector).all():
        raise RuntimeError((label, vector.shape, bool(np.isfinite(vector).all())))
    output.append({
        "label": label,
        "source": source,
        "coordinate_kind": coordinate_kind,
        "component": component,
        "layer": layer,
        "event": event,
        "family": family,
        "preview": preview,
        **metadata,
        "values": [float(value) for value in vector],
    })


def build_rows() -> tuple[list[dict], dict]:
    source_rows = read_rows(P2423 / "qwen4b/index/semantic_validity_rows.jsonl")
    index, case = selected_case(source_rows)
    embedding, embedding_meta = embedding_parameters(case)
    p2424 = json.loads((P2424 / "analysis/final.json").read_text(encoding="utf-8"))
    p2425 = json.loads((P2425 / "analysis/final.json").read_text(encoding="utf-8"))
    p2426 = json.loads((P2426 / "analysis/final.json").read_text(encoding="utf-8"))
    state = np.load(p2424["collection"]["state"]["path"], mmap_mode="r")
    update = np.load(p2425["analysis"]["files"]["update_passports"], mmap_mode="r")
    state_family = np.load(p2425["analysis"]["files"]["state_passports"], mmap_mode="r")
    slopes = np.load(p2425["analysis"]["files"]["slopes"], mmap_mode="r")
    crosslayer = np.load(P2428 / "derived/adjacent_diagonal_slope.float32.npy", mmap_mode="r")
    composition = np.load(P2429 / "derived/direct_to_composed_slope.float32.npy", mmap_mode="r")
    composition_metrics = np.load(P2429 / "derived/direct_composed_metrics.float32.npy", mmap_mode="r")
    contributions = np.load(P2430 / "derived/readout_coordinate_contribution.float32.npy", mmap_mode="r")
    archived_weight_difference = np.load(P2430 / "derived/readout_weight_difference.float16.npy", mmap_mode="r")
    families = p2425["analysis"]["families"]
    family_index = families.index("taxonomy")
    selections = p2426["analysis"]["selections"]
    event_names = ("fact1_relation", "query_end", "answer_boundary")
    output: list[dict] = []

    add(output, embedding[0], f"input embedding token {embedding_meta['prompt_token_id']} {embedding_meta['prompt_token']!r}",
        "token_embedding_parameter", "input_embedding_weight", preview=True, token_id=embedding_meta["prompt_token_id"])
    add(output, embedding[1], f"target output embedding token {embedding_meta['target_token_id']} {embedding_meta['target_token']!r}",
        "token_embedding_parameter", "output_embedding_weight", preview=True, token_id=embedding_meta["target_token_id"])
    add(output, embedding[2], f"foil output embedding token {embedding_meta['foil_token_id']} {embedding_meta['foil_token']!r}",
        "token_embedding_parameter", "output_embedding_weight", preview=True, token_id=embedding_meta["foil_token_id"])
    add(output, embedding[1] - embedding[2], "target - foil output embedding weight", "readout_parameter",
        "output_embedding_weight_difference", preview=True)
    add(output, archived_weight_difference[index], "archived BF16 output-weight difference", "readout_parameter",
        "output_embedding_weight_difference_fp16", preview=False)
    add(output, contributions[index], "answer-boundary H_i × ΔW_i logit contribution", "readout_contribution",
        "coordinate_logit_contribution", layer=37, event="answer_boundary", preview=True)

    for qpoint in (0, 1, 12, 24, 36, 37):
        add(output, state[index, qpoint, 1], f"selected-case q{qpoint} query-end HiddenState", "sample_state",
            "embedding_activation" if qpoint == 0 else "hidden_state", layer=qpoint, event="query_end",
            family="taxonomy", preview=qpoint in (0, 24, 37), case_id=case["case_id"])
    add(output, state[index, 37, 2], "selected-case q37 answer-boundary HiddenState", "sample_state",
        "hidden_state", layer=37, event="answer_boundary", family="taxonomy", preview=True, case_id=case["case_id"])

    for ii, interaction in enumerate(("semantic_validity", "lexical_control")):
        for ci, component in enumerate(("total", "attention", "mlp")):
            layer, event = selections[interaction][component]
            add(output, update[ii, ci, layer, event, family_index],
                f"{interaction} taxonomy {component} update passport q{layer} {event_names[event]}",
                "family_passport", "full_coordinate_update", component=component, layer=layer,
                event=event_names[event], family="taxonomy", preview=interaction == "semantic_validity")
            add(output, slopes[ii, ci, layer, event],
                f"{interaction} {component} state→update diagonal slope q{layer} {event_names[event]}",
                "state_slope", "fitted_diagonal_parameter", component=component, layer=layer,
                event=event_names[event], preview=interaction == "semantic_validity" and component == "total")
        state_layer, state_event = selections[interaction]["total"]
        add(output, state_family[ii, state_layer, state_event, family_index],
            f"{interaction} taxonomy state passport q{state_layer} {event_names[state_event]}",
            "family_passport", "full_coordinate_state", component="total", layer=state_layer,
            event=event_names[state_event], family="taxonomy", preview=interaction == "semantic_validity")
        add(output, crosslayer[ii, 35], f"{interaction} q35→q36 adjacent-layer diagonal slope",
            "crosslayer_slope", "fitted_diagonal_parameter", component="total", layer=35,
            event="query_end", preview=interaction == "semantic_validity")

    # Choose the total-component layer/event with the highest mean state gain across the six splits.
    best_flat = int(np.argmax(composition_metrics[0, :, 2].mean(axis=0)))
    best_layer, best_event = np.unravel_index(best_flat, (36, 2))
    for ci, component in enumerate(("total", "attention", "mlp")):
        add(output, composition[ci, best_layer, best_event],
            f"direct→composed {component} slope q{best_layer} {('query_end', 'answer_boundary')[best_event]}",
            "composition_slope", "fitted_diagonal_parameter", component=component, layer=int(best_layer),
            event=("query_end", "answer_boundary")[best_event], preview=component == "total")

    for value in (state, update, state_family, slopes, crosslayer, composition, composition_metrics,
                  contributions, archived_weight_difference):
        close(value)
    meta = {"selected_index": index, "selected_case": case["case_id"], "embedding": embedding_meta,
            "families": families, "composition_best_layer": int(best_layer),
            "composition_best_event": ("query_end", "answer_boundary")[best_event]}
    return output, meta


def build_asset() -> dict:
    rows, selection = build_rows()
    matrix = np.stack([np.asarray(row["values"], dtype=np.float32) for row in rows])
    BINARY.parent.mkdir(parents=True, exist_ok=True)
    np.save(BINARY, matrix)
    p2427 = json.loads((P2427 / "analysis/final.json").read_text(encoding="utf-8"))
    p2428 = json.loads((P2428 / "analysis/final.json").read_text(encoding="utf-8"))
    p2429 = json.loads((P2429 / "analysis/final.json").read_text(encoding="utf-8"))
    p2430 = json.loads((P2430 / "analysis/final.json").read_text(encoding="utf-8"))
    payload = {
        "schema": "c32561.semantic_encoding_output_field.v1",
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "model": "Qwen3-4B-BF16",
        "result_type": "semantic_encoding_output_field_heatmap",
        "dimensions": list(range(DIM)),
        "coordinate_semantics": "all 2560 fixed physical coordinates; rows are exact model parameters, archived activations, full-coordinate interaction passports, or fitted diagonal coefficients",
        "selection": selection,
        "summary": {
            "rows": len(rows),
            "preview_rows": int(sum(bool(row["preview"]) for row in rows)),
            "phase2427_intrinsic_group_win_rate": p2427["analysis"]["summary"],
            "phase2428_semantic_two_hop_path_consistent": p2428["adjudication"]["semantic_two_hop_path_consistent_all_splits"],
            "phase2429_direct_composed_coordinate_cosine": p2429["analysis"]["comparison"]["total"]["direct_composed_coordinate_cosine"],
            "phase2430_readout_correlation": p2430["output_closure"]["coordinate_sum_margin_correlation"],
            "phase2430_readout_relative_rmse": p2430["output_closure"]["relative_rmse"],
            "phase2430_output_behavior_bridge_closed": p2430["adjudication"]["output_behavior_bridge_closed"],
        },
        "rows": rows,
        "binary_companion": "/vis_data/research_kernel/c32561_semantic_encoding_output_field.float32.npy",
        "claim_boundary": "The visualization exposes full physical coordinates and concrete token embedding/readout parameters. Phase2427 found no stable intrinsic coordinate groups; Phase2428 found a generic cross-layer diagonal skeleton but not semantic specificity; Phase2429 found only a weak direct-composed correspondence; Phase2430 did not close the internal-to-output bridge. No semantic neuron, causal gear, universal language code, compiler, or language mechanism is claimed.",
    }
    save_if_changed(ASSET, payload)
    return {"asset": str(ASSET), "binary": str(BINARY), "schema": payload["schema"],
            "rows": len(rows), "dimensions": DIM, "preview_rows": payload["summary"]["preview_rows"],
            "json_bytes": ASSET.stat().st_size, "binary_shape": list(matrix.shape),
            "selection": selection, "finite": bool(np.isfinite(matrix).all())}


def frontend_contract() -> dict:
    route = (ROOT / "frontend/src/researchKernel/heatmapResearchRoute.js").read_text(encoding="utf-8-sig")
    hook = (ROOT / "frontend/src/researchKernel/useResearchKernel.js").read_text(encoding="utf-8-sig")
    component = (ROOT / "frontend/src/components/app/ResearchHeatmapRoute.jsx").read_text(encoding="utf-8-sig")
    app = (ROOT / "frontend/src/App.jsx").read_text(encoding="utf-8-sig")
    return {
        "route_registered": "C32561_LANGUAGE_ENCODING_FIELD_ROUTE" in route and "semantic_encoding_output_field_heatmap" in route,
        "asset_loaded": "setC32561LanguageEncodingField" in hook,
        "full_parameter_axis": "all 2560 Qwen3-4B physical coordinates / parameters" in component,
        "preview_wired": "c32561LanguageEncodingField={realResearchTrace.c32561LanguageEncodingField}" in app,
        "dist_exists": DIST.exists(),
        "dist_newer_than_asset": DIST.exists() and DIST.stat().st_mtime_ns >= ASSET.stat().st_mtime_ns,
    }


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 全坐标语言编码—词嵌入—输出贡献热力图与过度结论校正（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 将Phase2423–2430中可复查的重要拼图接入现有可视化客户端的新热力图类型`semantic_encoding_output_field_heatmap`。不是Top-K归档：JSON与float32伴随文件的每一行都保存Qwen3-4B全部2560个固定物理坐标。代表样本固定为taxonomy/unit6/中文/natural/方向0/valid/target，展示其真实输入token embedding参数、target/foil输出embedding参数、输出权重差、query-end多层HiddenState、answer-boundary逐坐标logit贡献；同时展示语义有效性与词项对照的H/A/M家族护照、状态→更新斜率、跨层斜率和直接→组合斜率。

$$E_{{token}}=(W^{{in}}_{{token,1}},\ldots,W^{{in}}_{{token,2560}}),\qquad \Delta W_i=W^{{out}}_{{target,i}}-W^{{out}}_{{foil,i}},$$

$$C_i=H^{{final}}_i\Delta W_i,\qquad m\approx\sum_{{i=1}}^{{2560}}C_i.$$

**结果汇总。** 资产与全坐标检查 `{json.dumps(result['asset'], ensure_ascii=False)}`；客户端注册与构建检查 `{json.dumps(result['frontend'], ensure_ascii=False)}`；正式校正 `{json.dumps(result['correction'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2431_c32561_c32880_encoding_field_visualization.py`；分析位于`tests/glm5/result/phase2431_c32561_c32880_encoding_field_visualization`；客户端资产`frontend/public/vis_data/research_kernel/c32561_semantic_encoding_output_field.json`及`.float32.npy`；路由、加载、3D全坐标热力图分别接入`frontend/src/researchKernel/heatmapResearchRoute.js`、`frontend/src/researchKernel/useResearchKernel.js`、`frontend/src/components/app/ResearchHeatmapRoute.jsx`和`frontend/src/App.jsx`。除本MEMO外未增改其他Markdown。

**分析、理论进展与正式校正。** Phase2429的真实坐标优于错位坐标是弱但可复验的“直接—组合对应候选”；总场坐标余弦仅0.1001、传递族/非传递控制能量比仅1.0201，故撤销“关系组合算子已检测”的过度措辞。Phase2430的架构线性读出恒等式保留，但FP16归档场对BF16 logits只有相关0.998226、相对RMSE 0.057274，属于近似重建；内部→输出所有平均state gain为负，故撤销旧记录中“输出行为桥闭合”的过度裁决。可视化的价值是让固定坐标纹理、真实词嵌入参数与输出贡献可逐格核查，而不是把图像形状当作机制。

**问题硬伤与结论。** 不同量纲行共享稳健色标只适合纹理检查，不能跨行比较幅值；JSON虽然全坐标但显示器可能按用户设置选择Top-K，原始2560维始终保存在载荷和二进制伴随文件。代表样本不等于总体证据，统计裁决仍以全数据结果为准。当前最强拼图是“固定坐标上存在通用状态依赖耦合和可复现跨层骨架”；最硬反证是语义特异性、内生协同组和内部→输出绝对预测都未通过。下一阶段仍是同一目标，应自动研究跨语言失败究竟是坐标重参数化，还是机制真正不共享。
"""
    with MEMO.open("a", encoding="utf-8", newline="") as stream:
        stream.write(text)


def main() -> None:
    asset = build_asset()
    frontend = frontend_contract()
    correction = {
        "phase2429_relation_composition_operator_detected_superseded": True,
        "replacement": "weak direct-composed fixed-coordinate correspondence candidate; no reusable relation-composition operator",
        "phase2430_output_behavior_bridge_closed_superseded": True,
        "replacement_output": "architectural readout identity plus approximate archived numeric reconstruction; internal-to-output bridge remains open",
    }
    checks = {
        "full_2560_dimensions": asset["dimensions"] == DIM,
        "parameter_and_hidden_rows": asset["rows"] >= 20,
        "binary_shape": asset["binary_shape"] == [asset["rows"], DIM],
        "finite": asset["finite"],
        "frontend_contract": all(frontend[key] for key in ("route_registered", "asset_loaded", "full_parameter_axis", "preview_wired")),
        "frontend_build_verified": frontend["dist_newer_than_asset"],
        "claim_boundary": True,
    }
    result = {"phase": PHASE, "campaign": CAMPAIGN, "asset": asset, "frontend": frontend,
              "correction": correction, "checks": checks, "all_checks_passed": all(checks.values())}
    save(OUT / "analysis/final.json", result)
    if result["all_checks_passed"]:
        append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise RuntimeError("run frontend build, then rerun Phase2431")


if __name__ == "__main__":
    main()
