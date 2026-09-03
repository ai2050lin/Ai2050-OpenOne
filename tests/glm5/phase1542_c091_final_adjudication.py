#!/usr/bin/env python3
"""Phase1542: final evidence adjudication for C091 and authorization of C092."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase1542_c091_final_adjudication"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1542 exists")
    phases = {
        phase: RESULT / name
        for phase, name in {
            1535: "phase1535_c091_external_human_material_and_analysis_adjudication",
            1536: "phase1536_c091_human_validated_chinese_relation_contract",
            1537: "phase1537_c091_behavior_only_qualification",
            1538: "phase1538_c091_behavior_gate_adjudication",
            1539: "phase1539_c091_canonical_all_state_capture",
            1540: "phase1540_c091_discovery_timing_atlas",
            1541: "phase1541_c091_dual_holdout_timing_validation",
        }.items()
    }
    audits = {phase: core.load(path / "audit/independent_final_audit.json") for phase, path in phases.items()}
    if not all(audit["all_checks_passed"] for audit in audits.values()):
        raise RuntimeError("C091 contains an unaudited phase")
    behavior = core.load(phases[1537] / "analysis/behavior_summary.json")
    scope = core.load(phases[1538] / "protocol/frozen_behavior_routes_and_hidden_scope.json")
    holdout = core.load(phases[1541] / "analysis/dual_holdout_summary.json")
    if scope["qualified_families"] != ["whole_part"] or holdout["status"] != "dual_holdout_gate_passed":
        raise RuntimeError("C091 final evidence preconditions failed")

    k267 = {
        "id": "K267",
        "grade": "E3-QWEN-HS-DESCRIPTIVE-BEHAVIOR-QUALIFIED",
        "title": "人类验证整体-部分材料上的前瞻晚层边界响应场",
        "claim": (
            "在冻结 Qwen3-4B、中文人类验证关系材料和通过行为资格的整体-部分询问中，"
            "两个预先区分的全维 Hidden-State 群体对象在答案边界 state31-32 出现，"
            "并在 discovery、confirmation、lockbox、具体/抽象材料及两种因果顺序中重复。"
        ),
        "not_claimed": [
            "纯整体-部分语义向量",
            "因果运输或必要/充分机制",
            "神经元组身份",
            "跨模型或跨语言不变量",
            "答案代码与任务终止已被排除",
            "新数学理论",
        ],
    }
    next_contract = {
        "campaign": "C092",
        "title": "整体-部分真值与答案代码正交化观察",
        "authorization": "run_phase1543_c092_truth_output_code_factorial_contract",
        "objective": "separate semantic truth, codebook instruction, and emitted answer token in embeddings-hidden-states-logits only",
        "frozen_constraints": {
            "model": "Qwen/Qwen3-4B BF16 CUDA",
            "materials": "C091 human-validated pairs with 30 true whole-part and 30 rank-matched false controls",
            "surfaces": ["prequery", "postquery"],
            "codebooks": ["native", "reversed"],
            "candidate_outputs": ["是", "否"],
            "analysis_scope": ["embeddings", "all_hidden_states", "candidate_logits"],
            "forbidden": ["attention", "MLP", "parameters", "gradients", "PCA", "TDA", "learned probes"],
            "behavior_before_hidden": True,
            "no_gate_changes_after_reveal": True,
        },
        "primary_decomposition": "H = S + y*T + c*C + (y*c)*A + epsilon",
        "decision_rule": "if both codebooks qualify, capture canonical all-state field and freeze discovery candidates before holdouts; otherwise retire only C092",
    }
    report = {
        "phase": 1542,
        "campaign": "C091",
        "status": "campaign_complete_with_k267",
        "audits": {str(phase): {"passed": value["passed"], "total": value["total"]} for phase, value in audits.items()},
        "behavior_scope": {
            "qualified": scope["qualified_families"],
            "retired": scope["retired_behavior_routes"],
            "global_behavior_accuracy": behavior["global"]["accuracy"],
        },
        "holdout_gate": {
            "passed": holdout["checks"]["all_holdout_gates_passed"],
            "min_centroid_cosine_to_discovery": min(row["centroid_cosine_to_discovery"] for row in holdout["results"]),
            "max_centroid_cosine_to_discovery": max(row["centroid_cosine_to_discovery"] for row in holdout["results"]),
        },
        "core_puzzle": k267,
        "theory_update": {
            "theory_name_unchanged": "条件化输出场闭合理论",
            "organizing_principle_unchanged": "复用-差分-条件化",
            "new_constraint": "a stable late-boundary response can coexist with unresolved semantic-truth versus answer-code identity",
            "new_mathematics_gate": "closed",
        },
        "hard_limits": [
            "one 4B model and one Chinese controlled interface",
            "only whole-part passed the binary behavior gate",
            "factorial controls are behavior-unqualified",
            "truth contrast retains lexical-family differences",
            "all internal findings are observational rather than intervention-based",
        ],
        "next_campaign": next_contract,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    core.save(OUT / "analysis/c091_final_adjudication.json", report)
    core.save(OUT / "theory/k267.json", k267)
    core.save(OUT / "protocol/next_campaign_authorization.json", next_contract)
    core.save(OUT / "analysis/final.json", {"phase": 1542, "campaign": "C091", "status": report["status"], "authorization": next_contract["authorization"]})
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
