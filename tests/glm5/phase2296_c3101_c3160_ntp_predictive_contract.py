#!/usr/bin/env python3
"""Freeze the NTP-aligned natural bilingual predictive-state campaign."""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
PARENT = RESULT / "phase2289_c2581_c2600_partition_lexicon_repair"
BEHAVIOR_PARENT = RESULT / "phase2290_c2601_c2700_qwen4b_natural_dynamic_field"
OUT = RESULT / "phase2296_c3101_c3160_ntp_predictive_contract"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
sys.path.insert(0, str(TESTS))

import model_utils  # noqa: E402
import phase1797_c263_c272_state_operator_common as compiler  # noqa: E402


PHASE = 2296
CAMPAIGN = "C3101-C3160"
FAMILIES = (
    "agent_patient", "attitude_event", "comparison_order",
    "location_binding", "possession_query", "relative_binding",
)
PARTITIONS = ("discovery", "confirmation", "fresh_confirmation", "fresh_lockbox")
QPOINTS_4B = (0, 1, 5, 10, 15, 20, 25, 30, 36, 37)
BEHAVIOR_GATE = 0.75
PREDICTIVE_GATES = {
    "lexical_sequence_accuracy": 0.75,
    "every_language_surface_partition_accuracy": 0.75,
    "lens_sign_accuracy": 0.75,
    "surface_equivalence_advantage": 0.0,
}
Q14_FAMILIES = ("agent_patient", "attitude_event", "location_binding")


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_rows(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def answer_ids(tokenizer, answer: str, language: str) -> list[int]:
    text = answer if language == "zh" else " " + answer
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        raise RuntimeError(("empty_answer", answer, language))
    return [int(value) for value in ids]


def compile_contract() -> tuple[list[dict], dict]:
    from transformers import AutoTokenizer

    rows = [row for row in read_rows(PARENT / "material/qwen_compiled.jsonl")
            if row["family"] in FAMILIES]
    tokenizer = AutoTokenizer.from_pretrained(
        model_utils.MODEL_CONFIGS["qwen3"]["path"], trust_remote_code=True,
        local_files_only=True, use_fast=False,
    )
    output = []
    first_token_collisions = []
    for row in rows:
        target_ids = answer_ids(tokenizer, row["correct_answer"], row["language"])
        wrong_ids = answer_ids(tokenizer, row["wrong_answer"], row["language"])
        if target_ids[0] == wrong_ids[0]:
            first_token_collisions.append(row["case_id"])
        output.append({
            **row,
            "ntp_prompt_ids": [int(value) for value in row["free_prompt_ids"]],
            "ntp_target_ids": target_ids,
            "ntp_wrong_ids": wrong_ids,
            "ntp_target_text": row["correct_answer"],
            "ntp_wrong_text": row["wrong_answer"],
            "ntp_interface": "natural_lexical_answer_boundary",
        })
    widths = [len(row["ntp_prompt_ids"]) for row in output]
    target_widths = Counter(len(row["ntp_target_ids"]) for row in output)
    balance = defaultdict(lambda: [0, 0])
    vocab = defaultdict(set)
    for row in output:
        balance[(row["family"], row["language"], row["surface"], row["partition"])][int(row["state"])] += 1
        vocab[row["partition"]].update((row["ntp_target_text"], row["ntp_wrong_text"]))
    overlaps = {}
    for i, left in enumerate(PARTITIONS):
        for right in PARTITIONS[i + 1:]:
            overlaps[f"{left}|{right}"] = sorted(vocab[left] & vocab[right])
    audit = {
        "rows": len(output),
        "families": list(FAMILIES),
        "prompt_width_min_max": [min(widths), max(widths)],
        "target_token_widths": dict(target_widths),
        "first_token_collision_count": len(first_token_collisions),
        "first_token_collision_cases": first_token_collisions,
        "state_balance_exact": all(value[0] == value[1] for value in balance.values()),
        "cross_partition_answer_overlap": overlaps,
        "human_blind_review": "NA_inherited_machine_audit_only",
    }
    return output, audit


def append_memo(result: dict) -> None:
    marker = f"## Phase {PHASE}:"
    if marker in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: NTP预测状态双接口大合同与证据纠偏（{CAMPAIGN}） [{stamp}]

**附件审查与测试原理。** “自回归训练使模型学习前缀对未来token竞争的条件状态”是正确起点；但训练目标本身不能推出 HiddenState 是完整未来的充分统计量、贝叶斯信念状态、Koopman不变子空间、信息流形测地线、拓扑空洞或波动方程。Phase2288-2295 的 `13/18`、`303/1332`、中层 `189` 与14B `4/5` 均是模型内预测证据，不是“浅层同构”“语义波”“偏微分过程”或因果齿轮。本期把下一研究对象冻结为同一语义材料的两个输出接口：原 A/B 账只作历史可比控制；自然词汇答案边界直接保存完整下一token分布、目标/错误续写的 teacher-forced 序列分数、全层边界状态和代表样本全部token状态。

**测试用例与公式。** 例如 `Leo Bell handed the atlas to Amina Arden. Who handed over the atlas?` 的自然目标是 `Leo Bell`，控制是 `Amina Arden`；状态翻转后目标随事实翻转，而不是交换 A/B 位置。训练目标和本期观测对象分别为：

$$
\mathcal L=-\sum_t\log p_\theta(x_{{t+1}}\mid x_{{\le t}}),
\qquad
\Pi_1(H)=\operatorname{{softmax}}(z(H)).
$$

完整多token候选只按 teacher forcing 计分：

$$
s(y_{{1:k}}\mid x)=\sum_{{r=1}}^k\log p_\theta(y_r\mid x,y_{{<r}}).
$$

冻结六个已通过 Phase2290 双行为门的构式、32单元、四分区、中英、两表面、真假两态，共 `{result['audit']['rows']}` 行；4B读出检查点 `{result['config']['qpoints_4b']}`，14B候选构式 `{result['config']['q14_families']}`。禁止以PCA、Top-K、余弦聚类、平均差分搬运或事后热点定义主结果。

**结果汇总与审计。** 材料审计 `{json.dumps(result['audit'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`；哈希 `{json.dumps(result['hashes'], ensure_ascii=False)}`。

**理论进展、问题硬伤与结论。** 本期没有运行模型。第一词token碰撞被逐行审计；序列正确性以完整答案分数判定，第一token margin只作局部观测。跨分区答案词汇并非完全不重叠，因为物体、地点和形容词表沿用旧材料；因此“新词汇锁箱”只对身份词成立，所有结论必须按分区分账。自然度仍无人类盲评。高等数学只在基础的概率—状态对应成立后作为诊断，不能预注册“必然发现流形或不变子空间”。脚本 `tests/glm5/phase2296_c3101_c3160_ntp_predictive_contract.py`；结果 `tests/glm5/result/phase2296_c3101_c3160_ntp_predictive_contract`。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def main() -> None:
    final_path = OUT / "analysis/final.json"
    if final_path.exists():
        result = json.loads(final_path.read_text(encoding="utf-8"))
        append_memo(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    parent = json.loads((BEHAVIOR_PARENT / "analysis/final.json").read_text(encoding="utf-8"))
    rows, audit = compile_contract()
    compiled = OUT / "material/ntp_natural_bilingual.jsonl"
    write_rows(compiled, rows)
    config = {
        "phase": PHASE, "campaign": CAMPAIGN,
        "frozen_before_new_model_execution": True,
        "source_phases": [2289, 2290, 2291, 2292, 2294],
        "families": list(FAMILIES), "partitions": list(PARTITIONS),
        "behavior_gate": BEHAVIOR_GATE, "predictive_gates": PREDICTIVE_GATES,
        "qpoints_4b": list(QPOINTS_4B), "q14_families": list(Q14_FAMILIES),
        "q14_selection": "fixed_from_prior_behavior_and_crossscale_evidence_before_ntp_run",
        "model_order": ["Qwen3-4B", "Qwen3-14B"],
        "claim_levels": ["observation", "prospective_prediction", "causal_only_if_pre_registered"],
        "forbidden_primary_methods": ["PCA", "Top-K", "cosine clustering", "mean-delta transport"],
        "advanced_math_policy": "diagnostic_only_after_basic_probability_state_checks",
    }
    save(OUT / "config/frozen_contract.json", config)
    checks = {
        "prior_six_families_dual_qualified": set(FAMILIES).issubset(parent["behavior"]["qualified_families"]),
        "row_count": len(rows) == len(FAMILIES) * 2 * 2 * 32 * 2,
        "state_balance": audit["state_balance_exact"],
        "all_answers_nonempty": all(row["ntp_target_ids"] and row["ntp_wrong_ids"] for row in rows),
        "all_first_tokens_discriminative": audit["first_token_collision_count"] == 0,
        "all_partitions_present": set(row["partition"] for row in rows) == set(PARTITIONS),
        "q14_frozen_before_ntp": set(Q14_FAMILIES).issubset(FAMILIES),
    }
    hashes = {"compiled": file_hash(compiled), "config": file_hash(OUT / "config/frozen_contract.json")}
    result = {
        "phase": PHASE, "campaign": CAMPAIGN,
        "timestamp": datetime.now().astimezone().isoformat(), "status": "closed",
        "audit": audit, "config": config, "checks": checks, "hashes": hashes,
        "all_checks_passed": all(checks.values()),
        "strict_conclusion": "The NTP-aligned dual-interface campaign is frozen; it corrects advanced-math overclaims and authorizes Qwen3-4B full-vocabulary observation only if every material check passes.",
        "next_authorization": "Collect Qwen3-4B lexical sequence behavior, full next-token logits, boundary trajectories, and representative all-token trajectories.",
    }
    save(final_path, result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
