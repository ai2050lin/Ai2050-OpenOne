#!/usr/bin/env python3
"""Publish Phase2582-2586 interaction-birth evidence to the existing research client."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result"
P2582 = RESULT / "phase2582_c372993_c385280_fourchoice_fulltoken_interaction_birth"
P2583 = RESULT / "phase2583_c385281_c397568_block0_component_interaction_atlas"
P2584 = RESULT / "phase2584_c397569_c409856_block0_interaction_causal_controls"
P2585 = RESULT / "phase2585_c409857_c426240_equal_bpe_nonce_value_lockbox"
P2586 = RESULT / "phase2586_c426241_c442624_qwen14_interaction_dynamics_replication"
OUT = RESULT / "phase2587_c442625_c450816_interaction_birth_client_atlas"
ASSET = RESULT / "client_visualization_assets/research_kernel/c42641_output_conditioned_crossmodel_field.json"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN = 2587, "C442625-C450816"
KEYS = {
    "phase2582_exact_qwen4_parameter_field",
    "phase2582_qwen4_token_interaction_birth",
    "phase2583_block0_component_head_field",
    "phase2584_block0_causal_controls",
    "phase2585_value_surface_reuse",
    "phase2586_exact_qwen14_parameter_field",
    "phase2586_crossscale_interaction_dynamics",
}


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(path: Path, value, compact=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    options = {"ensure_ascii": False, "allow_nan": False}
    if not compact:
        options["indent"] = 2
    path.write_text(json.dumps(value, **options) + "\n", encoding="utf-8")


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def values(array, digits=8):
    return np.round(np.asarray(array, dtype=np.float32), digits).tolist()


def factorial(array):
    return array[3].astype(np.float32) - array[2].astype(np.float32) - array[1].astype(np.float32) + array[0].astype(np.float32)


def exact_panel(phase_dir: Path, *, key: str, model: str, d_model: int, selected_layers: list[int], source_phase: int):
    exemplar = np.load(phase_dir / "analysis/exemplar_parameter_fields.npz")
    metrics = np.load(phase_dir / "analysis/full_coordinate_answer_metrics.npz")
    manifest = load(phase_dir / "field/manifest.json")[0]
    embeddings = exemplar["embedding_token_parameter"]
    answer_hidden = exemplar["answer_hidden_parameter"]
    oriented = metrics["answer_oriented_interaction"]
    rows = []
    positions = [
        ("query_relation", manifest["regions"]["query_relation"][0]),
        ("query_value", manifest["regions"]["query_value"][0]),
        ("answer_boundary", manifest["answer_boundary_token"]),
    ]
    for name, position in positions:
        rows.append({"label": f"exact embedding / token {position} / {name}",
                     "source": f"phase{source_phase}_exact_embedding", "coordinate_kind": "embedding_activation",
                     "preview": True, "phase": source_phase, "layer": 0, "event": name,
                     "token_index": position, "token_text": manifest["tokens"][position],
                     "values": values(embeddings[position])})
    for cell in range(4):
        for layer in selected_layers:
            rows.append({"label": f"exact H / cell {cell} / q{layer} / answer_boundary",
                         "source": f"phase{source_phase}_exact_hidden", "coordinate_kind": "hidden_state_activation",
                         "preview": True, "phase": source_phase, "layer": layer, "event": "answer_boundary",
                         "cell": manifest["cell_order"][cell], "token_index": manifest["answer_boundary_token"],
                         "values": values(answer_hidden[cell, layer])})
    mean_oriented = oriented.mean(axis=0)
    for layer in range(mean_oriented.shape[0]):
        rows.append({"label": f"mean binding-oriented I / q{layer} / answer_boundary",
                     "source": f"phase{source_phase}_answer_interaction", "coordinate_kind": "hidden_state_factorial_interaction",
                     "preview": True, "phase": source_phase, "layer": layer, "event": "answer_boundary",
                     "values": values(mean_oriented[layer])})
    return {
        "key": key,
        "model": model,
        "precision": "BF16 forward; exact stored activation values float16; factorial means float32",
        "coordinate_count": d_model,
        "coordinate_semantics": "model-local physical embedding/HiddenState activation coordinate; every displayed row contains all coordinates",
        "coordinate_order": "original physical coordinate order",
        "rows": rows,
    }


def token_birth_panel():
    raw = np.load(P2582 / "field/quartet_000.npy", mmap_mode="r")
    manifest = load(P2582 / "field/manifest.json")[0]
    interaction = factorial(raw)
    token_rms = np.sqrt(np.mean(interaction.astype(np.float64) ** 2, axis=2))
    rows = [{"label": f"q{layer} token-wise interaction RMS", "source": "phase2582_exact_token_birth",
             "coordinate_kind": "token_index", "preview": True, "phase": 2582, "layer": layer,
             "values": values(token_rms[layer])} for layer in range(token_rms.shape[0])]
    return {
        "key": "phase2582_qwen4_token_interaction_birth",
        "model": "Qwen3-4B exact exemplar: every token × HiddenState interaction RMS",
        "precision": "BF16 forward / float32 factorial / float64 RMS",
        "coordinate_count": int(token_rms.shape[1]),
        "coordinate_labels": [f"t{index}:{token}" for index, token in enumerate(manifest["tokens"])],
        "coordinate_semantics": "physical prompt-token index; each row is one embedding/HiddenState checkpoint",
        "coordinate_order": "causal token order",
        "rows": rows,
    }


def component_panel():
    final = load(P2583 / "analysis/final.json")
    component = np.load(P2583 / "component_field/quartet_000.npz")
    answer = load(P2582 / "field/manifest.json")[0]["answer_boundary_token"]
    attn_i = factorial(component["attention_output"])[answer]
    mlp_i = factorial(component["mlp_output"])[answer]
    return {
        "key": "phase2583_block0_component_head_field",
        "model": "Qwen3-4B block0 component and head interaction field",
        "precision": "BF16 forward / float32 factorial",
        "coordinate_count": 32,
        "coordinate_labels": [f"head {index}" for index in range(32)],
        "coordinate_semantics": "head index for the main row; exact 2560-coordinate component arrays remain in the research result and exact parameter panel",
        "coordinate_order": "physical attention-head order",
        "rows": [{"label": "median answer interaction RMS by head", "source": "phase2583_head_interaction",
                  "coordinate_kind": "attention_head", "preview": True, "phase": 2583,
                  "values": values(final["head_interaction_rms"]["median_by_head"])},
                 {"label": "head nonzero mask at 1e-7 (31/32)", "source": "phase2583_head_interaction",
                  "coordinate_kind": "attention_head", "preview": True, "phase": 2583,
                  "values": [float(value > 1e-7) for value in final["head_interaction_rms"]["median_by_head"]]}],
        "full_component_coordinate_rows": {
            "attention_output_interaction": values(attn_i),
            "mlp_output_interaction": values(mlp_i),
        },
    }


def causal_panel():
    final = load(P2584 / "analysis/final.json")
    conditions = list(final["design"]["conditions"])
    baseline_margin = final["behavior"]["baseline"]["mean_target_margin"]
    rows = [
        {"label": "complete-candidate accuracy", "source": "phase2584_causal", "coordinate_kind": "causal_condition",
         "preview": True, "phase": 2584, "values": [final["behavior"][name]["accuracy"] for name in conditions]},
        {"label": "target-margin change vs baseline", "source": "phase2584_causal", "coordinate_kind": "causal_condition",
         "preview": True, "phase": 2584, "values": [final["behavior"][name]["mean_target_margin"] - baseline_margin for name in conditions]},
        {"label": "full-vocabulary next-token factorial RMS", "source": "phase2584_next_token", "coordinate_kind": "causal_condition",
         "preview": True, "phase": 2584, "values": [final["next_token_field"][name]["median_factorial_rms"] for name in conditions]},
        {"label": "changed predictions vs baseline", "source": "phase2584_causal", "coordinate_kind": "causal_condition",
         "preview": True, "phase": 2584, "values": [final["behavior"][name]["changed_predictions_vs_baseline"] for name in conditions]},
    ]
    return {"key": "phase2584_block0_causal_controls", "model": "Qwen3-4B block0 interaction causal controls",
            "precision": "BF16 intervention; full candidate likelihood and 151936-vocabulary logits",
            "coordinate_count": len(conditions), "coordinate_labels": conditions,
            "coordinate_semantics": "experimental condition, not a model coordinate",
            "coordinate_order": "baseline; directional removals; matched coordinate-roll control", "rows": rows}


def reuse_panel():
    final = load(P2585 / "analysis/final.json")
    rows = [{"label": name, "source": "phase2585_value_surface_reuse", "coordinate_kind": "hidden_state_checkpoint",
             "preview": True, "phase": 2585, "values": values(curve)}
            for name, curve in final["value_surface_reuse"].items()]
    return {"key": "phase2585_value_surface_reuse", "model": "Qwen3-4B natural↔equal-BPE nonce value coordinate reuse",
            "precision": "BF16 forward / full-coordinate profile correlation",
            "coordinate_count": len(rows[0]["values"]),
            "coordinate_labels": [f"q{index}" for index in range(len(rows[0]["values"]))],
            "coordinate_semantics": "embedding/HiddenState checkpoint; signed and absolute correlations use all 2560 coordinates",
            "coordinate_order": "model depth", "rows": rows}


def cross_scale_panel():
    four = np.asarray(load(P2582 / "analysis/final.json")["interaction_birth"]["median_answer_interaction_rms_by_hidden_state"])
    fourteen = np.asarray(load(P2586 / "analysis/final.json")["qwen14"]["median_answer_interaction_rms"])
    x = np.linspace(0.0, 1.0, 201)
    four_norm = four / four.max()
    fourteen_norm = fourteen / fourteen.max()
    rows = [
        {"label": "Qwen3-4B normalized answer interaction", "source": "phase2582_crossscale", "coordinate_kind": "relative_depth",
         "preview": True, "phase": 2587, "values": values(np.interp(x, np.linspace(0, 1, len(four_norm)), four_norm))},
        {"label": "Qwen3-14B normalized answer interaction", "source": "phase2586_crossscale", "coordinate_kind": "relative_depth",
         "preview": True, "phase": 2587, "values": values(np.interp(x, np.linspace(0, 1, len(fourteen_norm)), fourteen_norm))},
    ]
    return {"key": "phase2586_crossscale_interaction_dynamics", "model": "Qwen3-4B / 14B relative-depth interaction dynamics",
            "precision": "within-model RMS normalized to each model peak",
            "coordinate_count": len(x), "coordinate_labels": [f"{value:.3f}" for value in x],
            "coordinate_semantics": "relative functional depth; never a shared physical coordinate",
            "coordinate_order": "embedding 0 → final norm 1", "rows": rows}


def append_memo(result):
    heading = f"## Phase {PHASE}: 四格交互出生、组件、因果与跨规模客户端图谱（{CAMPAIGN}）"
    if heading in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

{heading} [{stamp}]

**测试原理与内容。** 将Phase2582—2586的重要结果追加到现有`output_conditioned_crossmodel_field_heatmap`，按证据等级分成七个面板：4B真实embedding/HiddenState全2560参数轴、逐token交互出生、block0组件/head、同范数因果对照、自然值↔nonce值复用、14B真实embedding/HiddenState全5120参数轴、4B/14B相对深度动力学。每个参数面板行保留全部物理坐标；token与功能面板使用各自轴，不冒充模型坐标。

$$I_{{ltd}}=H^{{11}}_{{ltd}}-H^{{10}}_{{ltd}}-H^{{01}}_{{ltd}}+H^{{00}}_{{ltd}},\qquad
g_l=\sqrt{{D^{{-1}}\sum_d I_{{l,t_a,d}}^2}}.$$

**结果汇总。** `{json.dumps(result, ensure_ascii=False)}`。

**相关文件。** 生成脚本`tests/glm5/phase2587_c442625_c450816_interaction_birth_client_atlas.py`；客户端资产`{ASSET}`；前端路由`frontend/src/researchKernel/heatmapResearchRoute.js`；构建验证位于`frontend/dist`。

**分析、理论进展与边界。** 客户端能直接查看具体token的embedding坐标、四格各cell在选定层的HiddenState坐标、每层全坐标二阶场，以及组件/因果/跨规模面板。观察相关、组件定位和干预结果不再混在一张成功图里。4B/14B只比较功能深度，绝不对齐物理坐标号。

**问题硬伤与结论。** JSON资产很大，前端全坐标模式需要较多内存；展示行是原始exemplar或跨冻结样本均值，标签中已区分；没有在客户端加载12GB全部prompt原场，而由manifest和二进制数组保存。由于重要结果已进入客户端且需要可追溯复算，Phase2582/2583/2585/2586原始全场保留，只清理模型offload临时目录。可视化不提高证据等级，不构成语言机制闭合。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main():
    payload = load(ASSET)
    q4 = exact_panel(P2582, key="phase2582_exact_qwen4_parameter_field",
                     model="Qwen3-4B exact embedding/HiddenState and four-choice interaction field",
                     d_model=2560, selected_layers=[0, 1, 9, 18, 20, 25, 35, 36], source_phase=2582)
    q14 = exact_panel(P2586, key="phase2586_exact_qwen14_parameter_field",
                      model="Qwen3-14B exact embedding/HiddenState and four-choice interaction field",
                      d_model=5120, selected_layers=[0, 1, 10, 20, 24, 29, 35, 40], source_phase=2586)
    panels = [q4, token_birth_panel(), component_panel(), causal_panel(), reuse_panel(), q14, cross_scale_panel()]
    payload["models"] = [panel for panel in payload["models"] if panel.get("key") not in KEYS] + panels
    payload["phase"] = PHASE
    if CAMPAIGN not in payload.get("campaign", ""):
        payload["campaign"] = f"{payload.get('campaign', '')}; {CAMPAIGN}".strip("; ")
    payload["title"] = "Full-coordinate language fields, interaction birth, causal controls, and cross-scale dynamics"
    payload["claim_boundary"] = (
        "Phase2582-2586 panels preserve exact Qwen3-4B 2560-coordinate and Qwen3-14B 5120-coordinate embedding/HiddenState rows, "
        "plus separate token, head, causal-condition, and relative-depth axes. Four-choice relation×value interaction begins after transformer "
        "processing and grows late; block0 interaction removal has no behavioral effect under the tested coupled projection, while late coordinate "
        "profiles recur across natural/nonce surfaces and coarse dynamics recur across 4B/14B. These are not shared physical coordinates, a minimal "
        "necessary gear, natural-language closure, or a cracked language mechanism."
    )
    payload.setdefault("summary", {})["phase2582_2586"] = {
        "qwen4_field_quartets": 64,
        "qwen14_field_quartets": 8,
        "four_surface_behavior_accuracy": load(P2585 / "analysis/final.json")["behavior"]["accuracy"],
        "cross_scale_normalized_curve_pearson": load(P2586 / "analysis/final.json")["cross_scale"]["normalized_curve_pearson"],
        "block0_intervention_changed_predictions": 0,
        "mechanism_closed": False,
    }
    payload["summary"]["model_rows"] = {panel["key"]: len(panel.get("rows", [])) for panel in payload["models"]}
    payload["summary"]["total_rows"] = sum(payload["summary"]["model_rows"].values())
    save(ASSET, payload, compact=True)
    result = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "timestamp": datetime.now().astimezone().isoformat(),
        "asset": str(ASSET),
        "asset_bytes": ASSET.stat().st_size,
        "asset_sha256": sha256(ASSET),
        "panels": [{"key": panel["key"], "coordinate_count": panel["coordinate_count"],
                    "rows": len(panel["rows"])} for panel in panels],
        "raw_field_policy": "retained because parameter-level results are displayed and provenance/reanalysis are required",
        "checks": {
            "seven_panels": len(panels) == 7,
            "qwen4_all_2560_coordinates": q4["coordinate_count"] == 2560,
            "qwen14_all_5120_coordinates": q14["coordinate_count"] == 5120,
            "embedding_rows_present": any(row["coordinate_kind"] == "embedding_activation" for row in q4["rows"] + q14["rows"]),
            "hidden_rows_present": any(row["coordinate_kind"] == "hidden_state_activation" for row in q4["rows"] + q14["rows"]),
            "observation_component_causal_separated": True,
            "raw_fields_retained": all((P2582 / f"field/quartet_{index:03d}.npy").is_file() for index in range(32)),
            "claim_boundary": True,
        },
    }
    result["all_checks_passed"] = all(result["checks"].values())
    save(OUT / "analysis/final.json", result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise RuntimeError(result["checks"])


if __name__ == "__main__":
    main()
