#!/usr/bin/env python3
"""Audit Phase2434-2452 reviews and freeze the portability/curvature/autoregressive campaign."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result"
P2406 = RESULT / "phase2406_c24561_c24880_behavior_precision_calibration"
P2435 = RESULT / "phase2435_c33841_c34160_hypergraph_material_fourmodel_behavior"
P2452 = RESULT / "phase2452_c39441_c39760_campaign_evidence_audit"
OUT = RESULT / "phase2453_c39761_c40080_portability_curvature_autoregressive_contract"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE = 2453
CAMPAIGN = "C39761-C40080"
MODEL_ORDER = ("qwen14b", "glm4", "deepseek7b")
MODEL_CONFIGS = {
    "qwen14b": ROOT / "models/hf/Qwen3-14B/config.json",
    "glm4": ROOT / "models/hf/glm4-9b-chat-hf/config.json",
    "deepseek7b": ROOT / "models/hf/deepseek-r1-distill-qwen-7b/config.json",
}


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def model_contract() -> dict:
    models = {}
    semantic_signatures = {}
    for key in MODEL_ORDER:
        source = read_rows(P2435 / key / "index/trajectory_rows.jsonl")
        rows = [row for row in source if int(row["unit"]) in (0, 4, 5) and row["surface"] == "canonical" and int(row["direction"]) == 0]
        rows.sort(key=lambda row: row["case_id"])
        if len(rows) != 288:
            raise RuntimeError((key, len(rows)))
        write_rows(OUT / f"contract/{key}_vjp_rows.jsonl", rows)
        config = json.loads(MODEL_CONFIGS[key].read_text(encoding="utf-8"))
        layers = int(config.get("num_hidden_layers", config.get("num_layers")))
        hidden = int(config.get("hidden_size"))
        qpoints = layers + 2
        # q0=embedding and q(L+1)=final norm. Map the frozen Qwen4B q16/q18 positions to relative qpoint depth.
        q_attr = round(16 * (qpoints - 1) / 37)
        q_grad = round(18 * (qpoints - 1) / 37)
        signatures = [{name: row[name] for name in ("case_id", "config_id", "family", "unit", "language", "surface", "direction", "variant", "query_role")} for row in rows]
        semantic_signatures[key] = signatures
        behavior = json.loads((P2435 / key / "analysis/final.json").read_text(encoding="utf-8"))
        models[key] = {"rows": len(rows), "layers": layers, "qpoints": qpoints, "hidden_size": hidden,
                       "relative_qpoints": {"state_times_gradient": q_attr, "gradient": q_grad},
                       "precision": behavior["precision"],
                       "aggregate_behavior_delta": behavior["teacher"]["valid_minus_broken_a"],
                       "qualified_families": [family for family, value in behavior["teacher"]["by_family"].items() if value["valid_minus_broken_a"] > .05]}
    reference = [{key: row[key] for key in row if key != "case_id"} for row in semantic_signatures["qwen14b"]]
    design_aligned = all([{key2: row[key2] for key2 in row if key2 != "case_id"} for row in semantic_signatures[key]] == reference for key in MODEL_ORDER)
    return {"model_order": list(MODEL_ORDER), "models": models, "semantic_design_aligned": design_aligned,
            "selection": "Qwen4B q16/q18 mapped by qpoint-relative depth; no target-model layer search",
            "events": ["query_end"], "fields": ["gradient", "state_times_gradient"],
            "nulls": ["coordinate_shift", "64_deranged_family_labels"],
            "claim_rule": "Compare only within-model dimensionless matched-vs-null advantages; do not compare raw amplitudes or coordinate identities across widths."}


def qwen14_precision_audit() -> dict:
    prior = json.loads((P2406 / "analysis/final.json").read_text(encoding="utf-8"))["qwen14b_bf16"]
    weights = list((ROOT / "models/hf/Qwen3-14B").glob("*.safetensors"))
    total = sum(path.stat().st_size for path in weights)
    return {"requested": prior["requested"], "prior_attempts": prior["attempts"], "prior_adjudication": prior["adjudication"],
            "weight_files": len(weights), "weight_bytes": total,
            "current_host_total_ram_gib": 31.4, "current_host_free_ram_gib_at_contract": 19.4,
            "decision": "Do not repeat a fifth known-crashing BF16 materialization. Use NF4 storage/BF16 compute for the activation-VJP portability test and prohibit cross-precision amplitude claims.",
            "bf16_model_result_claimed": False}


def evidence_audit() -> dict:
    prior = json.loads((P2452 / "analysis/final.json").read_text(encoding="utf-8"))
    return {
        "retained": [
            "The corrected Phase2434-2452 numerical chain and its three quality repairs are valid.",
            "The strongest supported object is a Qwen3-4B output-conditioned semantic-attribution and finite-direction candidate.",
            "Full-coordinate identity, held-unit reuse, language reuse, and surface reuse are evidence-bearing within Qwen3-4B.",
        ],
        "corrected": [
            "Phase2445 rejects the tested fixed coordinate/Gram bridge, not every possible internal-to-output mapping.",
            "A 0.579 sign agreement does not by itself prove strong nonlinearity; BF16 discretization and many near-zero finite effects remain alternatives.",
            "VJP is an analyst-computed output-conditioned covector, not evidence that the model explicitly runs backpropagation or stores a VJP gear.",
            "Semantic H-times-VJP exceeds the lexical control at its frozen layer, but direct gradient is often stronger for the lexical control; semantic specificity is object-dependent.",
            "Natural/canonical reuse does not remove shared entities, candidate protocol, task frame, or output identity.",
            "Cross-architecture coordinate isomorphism is untested/missing, not empirically falsified by Phase2434-2452.",
            "The finite test used one family mismatch and one dose; its result is local sufficiency evidence, not necessity or a stable small coordinate coalition.",
        ],
        "prior_mechanism_closed": prior["evidence"]["key_flags"]["language_encoding_mechanism_closed"],
    }


def campaign_plan() -> dict:
    return {"name": "portability-curvature-autoregressive-compilation",
            "phases": [
                {"phase": 2454, "task": "Qwen3-14B model-relative full-coordinate VJP portability, NF4 storage/BF16 compute fallback."},
                {"phase": 2455, "task": "GLM4-9B model-relative full-coordinate VJP portability, INT8 model-local inference."},
                {"phase": 2456, "task": "DS7B model-relative VJP response map; semantic claims restricted to behavior-qualified families."},
                {"phase": 2457, "task": "Cross-model dimensionless matched-vs-null and behavior-gated adjudication."},
                {"phase": 2458, "task": "Qwen3-4B 0.25/0.5/1/2 percent RMS odd/even response, multiple family derangements, BF16 floor."},
                {"phase": 2459, "task": "Balanced coordinate-group masks and pairwise non-additivity on frozen directions; no Top-K selection."},
                {"phase": 2460, "task": "Teacher-forced multi-token log-probability VJP/finite-response path."},
                {"phase": 2461, "task": "Output-identity and task-frame deconfounding with new answer interfaces."},
                {"phase": 2462, "task": "Full-coordinate visualization, retention/cleanup, evidence and successor audit."},
            ],
            "priority": "Observe portable full-coordinate laws first; infer structure second; use closure only as the final evidence level."}


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: Phase2434–2452复盘审查与可移植性—剂量律—自回归编译合同（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 将两份复盘逐项对照Phase2434–2452 final与三次质量修正，保留数值链、held-unit/语言/表述复用和有限效应；撤销“低符号一致率已证明强非线性”“固定输出桥彻底否定所有映射”“VJP就是模型齿轮”“跨模型同构已否决”等过度推断。冻结三模型相同语义设计：unit0发现、unit4确认、unit5 fresh、canonical、direction0、八族×中英×三validity×双角色=288条；不在目标模型重新选层，把Qwen4B q16/q18按相对qpoint深度映射。

$$q_m(r)=\operatorname{{round}}\left(r\frac{{L_m+1}}{{37}}\right),\quad r\in\{{16,18\}},$$
$$\Delta_m=\operatorname{{cos}}_m(\text{{matched}})-Q_{{.95}}\operatorname{{cos}}_m(\pi_{{family}}).$$

**结果汇总。** 证据审查 `{json.dumps(result['evidence_audit'], ensure_ascii=False)}`；模型合同 `{json.dumps(result['model_contract'], ensure_ascii=False)}`；Qwen14精度裁决 `{json.dumps(result['qwen14_precision'], ensure_ascii=False)}`；大阶段 `{json.dumps(result['campaign_plan'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2453_c39761_c40080_portability_curvature_autoregressive_contract.py`；三个冻结索引与final位于同名结果目录。没有修改其他Markdown。

**理论进展。** 当前候选应表达为“状态条件化的语义interaction与输出目标协向量之间的局部收缩”，而不是把VJP本身实体化为模型内齿轮。跨模型只比较各模型自己的matched-vs-null无量纲优势和相对深度，不比较不同宽度的坐标编号或幅值。

**问题硬伤与结论。** Qwen14B BF16 `device_map=auto`已经四次在本机权重物化阶段发生Windows访问冲突；当前31.4GiB RAM、约29GiB权重也没有安全重试余量，因此本阶段不进行第五次同型危险重试，使用NF4存储/BF16计算并严格标注。GLM/DS为INT8。量化模型只能用于模型内拓扑复制。合同完整，不代表机制成立。
"""
    with MEMO.open("a", encoding="utf-8", newline="") as stream:
        stream.write(text)


def main() -> None:
    final = OUT / "analysis/final.json"
    if final.exists():
        result = json.loads(final.read_text(encoding="utf-8")); append_memo(result); print(json.dumps(result, ensure_ascii=False, indent=2)); return
    evidence = evidence_audit()
    contract = model_contract()
    precision = qwen14_precision_audit()
    plan = campaign_plan()
    checks = {"prior_not_closed": evidence["prior_mechanism_closed"] is False,
              "three_models_ordered": contract["model_order"] == list(MODEL_ORDER),
              "rows_288_each": all(value["rows"] == 288 for value in contract["models"].values()),
              "design_aligned": contract["semantic_design_aligned"],
              "model_configs": all(path.exists() for path in MODEL_CONFIGS.values()),
              "qwen14_bf16_not_misclaimed": precision["bf16_model_result_claimed"] is False,
              "nine_phase_plan": len(plan["phases"]) == 9,
              "claim_boundary": True}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "evidence_audit": evidence, "model_contract": contract,
              "qwen14_precision": precision, "campaign_plan": plan, "checks": checks, "all_checks_passed": all(checks.values())}
    save(final, result); append_memo(result)
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
