#!/usr/bin/env python3
"""Freeze an eight-family raw-continuation and multi-step future campaign."""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P2306 = RESULT / "phase2306_c4021_c4160_corrected_surface_replication"
P2289 = RESULT / "phase2289_c2581_c2600_partition_lexicon_repair"
OUT = RESULT / "phase2309_c4321_c4440_multistep_future_contract"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
Q4_ROWS = P2306 / "material/corrected_declarative_continuation_bilingual.jsonl"
EXTRA_ROWS = P2289 / "material/qwen_compiled.jsonl"
sys.path.insert(0, str(TESTS))

import model_utils  # noqa: E402


PHASE = 2309
CAMPAIGN = "C4321-C4440"
FAMILIES = (
    "agent_patient",
    "attitude_event",
    "comparison_order",
    "location_binding",
    "possession_query",
    "relative_binding",
    "temporal_order",
    "taxonomy_chain",
)
LANGUAGES = ("en", "zh")
SURFACES = ("narrative", "dialogue")
PARTITIONS = ("discovery", "confirmation", "fresh_confirmation", "fresh_lockbox")
UNITS = 32
FUTURE_STEPS = 6
QPOINTS_4B = (0, 1, 5, 10, 15, 20, 25, 30, 36, 37)
BEHAVIOR_GATE = 0.75
FREE_IDENTITY_GATE = 0.50
LOCAL_LINEAR_GATES = {
    "sign_accuracy": 0.75,
    "median_relative_error": 0.30,
    "forward_reverse_order_rate": 0.75,
}
MODEL_ORDER = ("Qwen3-4B", "Qwen3-14B", "DeepSeek-7B")


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def write_rows(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def answer_ids(tokenizer, text: str, language: str) -> list[int]:
    encoded = tokenizer.encode((" " + text) if language == "en" else text,
                               add_special_tokens=False)
    if not encoded:
        raise RuntimeError(("empty_answer", language, text))
    return [int(value) for value in encoded]


def future_text(correct: str, wrong: str, language: str) -> tuple[str, str]:
    if language == "en":
        return f"{correct} rather than {wrong}.", f"{wrong} rather than {correct}."
    return f"{correct}，而不是{wrong}。", f"{wrong}，而不是{correct}。"


def temporal_prefix(row: dict, target_first: bool) -> str:
    target, wrong = row["correct_answer"], row["wrong_answer"]
    language, surface = row["language"], row["surface"]
    if language == "en":
        fact = (f"{target} arrived before {wrong}." if target_first else
                f"{wrong} arrived after {target}.")
        if surface == "dialogue":
            return f"A dated log states, '{fact}' According to the log, the first person to arrive was"
        return f"{fact} The first person to arrive was"
    fact = (f"{target}比{wrong}先到。" if target_first else
            f"{wrong}在{target}之后到达。")
    if surface == "dialogue":
        return f"一份有日期的记录写道：“{fact}”根据记录，先到的人是"
    return f"{fact}先到的人是"


def taxonomy_prefix(row: dict, target_first: bool) -> tuple[str, str, str]:
    obj = row["role_values"]["primary"]
    parent = row["role_values"]["secondary"]
    target = row["role_values"]["context"]
    paired_wrong = row["paired_top_category"]
    language, surface = row["language"], row["surface"]
    if language == "en":
        relation = (f"Every {parent} is {target}, and no {parent} is {paired_wrong}."
                    if target_first else
                    f"No {parent} is {paired_wrong}, and every {parent} is {target}.")
        facts = f"Every {obj} is a {parent}. {relation}"
        if surface == "dialogue":
            prefix = (f"A curator states, '{facts}' Under this closed catalog, the top category "
                      f"assigned to the {obj} is")
        else:
            prefix = (f"In this closed catalog, {facts} Therefore, the top category assigned "
                      f"to the {obj} is")
    else:
        relation = (f"每个{parent}都是{target}，并且没有{parent}属于{paired_wrong}。"
                    if target_first else
                    f"没有{parent}属于{paired_wrong}，并且每个{parent}都是{target}。")
        facts = f"每个{obj}都是{parent}。{relation}"
        if surface == "dialogue":
            prefix = f"管理员说明：“{facts}”按照这份封闭目录，{obj}所属的上位类别是"
        else:
            prefix = f"在这份封闭目录中，{facts}因此，{obj}所属的上位类别是"
    return prefix, target, paired_wrong


def compile_rows() -> tuple[list[dict], dict]:
    tokenizer = AutoTokenizer.from_pretrained(
        model_utils.MODEL_CONFIGS["qwen3"]["path"],
        trust_remote_code=True,
        local_files_only=True,
        use_fast=False,
    )
    rows: list[dict] = []
    for source in read_rows(Q4_ROWS):
        target_future, wrong_future = future_text(
            source["ntp_target_text"], source["ntp_wrong_text"], source["language"]
        )
        rows.append({
            **source,
            "case_id": source["case_id"] + "-future",
            "source_case_id": source["case_id"],
            "future_prompt": source["declarative_prefix"],
            "future_prompt_ids": [int(value) for value in tokenizer.encode(
                source["declarative_prefix"], add_special_tokens=False
            )],
            "future_target_text": target_future,
            "future_wrong_text": wrong_future,
            "future_target_ids": answer_ids(tokenizer, target_future, source["language"]),
            "future_wrong_ids": answer_ids(tokenizer, wrong_future, source["language"]),
            "future_identity_target_ids": answer_ids(
                tokenizer, source["ntp_target_text"], source["language"]
            ),
            "future_identity_wrong_ids": answer_ids(
                tokenizer, source["ntp_wrong_text"], source["language"]
            ),
            "future_steps_frozen": FUTURE_STEPS,
            "interface": "raw_declarative_multistep_continuation",
        })

    extra = [row for row in read_rows(EXTRA_ROWS)
             if row["family"] in ("temporal_order", "taxonomy_chain")]
    taxonomy_pairs: dict[tuple, dict[int, dict]] = defaultdict(dict)
    for row in extra:
        if row["family"] == "taxonomy_chain":
            taxonomy_pairs[(row["language"], row["surface"], int(row["unit"]))][int(row["state"])] = row
    for pair in taxonomy_pairs.values():
        if set(pair) != {0, 1}:
            raise RuntimeError(("taxonomy_state_pair", sorted(pair)))
        pair[0]["paired_top_category"] = pair[1]["role_values"]["context"]
        pair[1]["paired_top_category"] = pair[0]["role_values"]["context"]

    for source in extra:
        target_first = int(source["unit"]) % 2 == 0
        if source["family"] == "temporal_order":
            prefix = temporal_prefix(source, target_first)
            target, wrong = source["correct_answer"], source["wrong_answer"]
        else:
            prefix, target, wrong = taxonomy_prefix(source, target_first)
        target_future, wrong_future = future_text(target, wrong, source["language"])
        rows.append({
            **source,
            "case_id": source["case_id"] + "-future",
            "source_case_id": source["case_id"],
            "correct_answer": target,
            "wrong_answer": wrong,
            "ntp_target_text": target,
            "ntp_wrong_text": wrong,
            "future_prompt": prefix,
            "future_prompt_ids": [int(value) for value in tokenizer.encode(prefix, add_special_tokens=False)],
            "future_target_text": target_future,
            "future_wrong_text": wrong_future,
            "future_target_ids": answer_ids(tokenizer, target_future, source["language"]),
            "future_wrong_ids": answer_ids(tokenizer, wrong_future, source["language"]),
            "future_identity_target_ids": answer_ids(tokenizer, target, source["language"]),
            "future_identity_wrong_ids": answer_ids(tokenizer, wrong, source["language"]),
            "target_mention_order": "first" if target_first else "last",
            "source_fact_order_matched": True,
            "future_steps_frozen": FUTURE_STEPS,
            "interface": "raw_declarative_multistep_continuation",
        })

    rows.sort(key=lambda row: (
        FAMILIES.index(row["family"]),
        LANGUAGES.index(row["language"]),
        SURFACES.index(row["surface"]),
        int(row["unit"]),
        int(row["state"]),
    ))
    first_collision = [row["case_id"] for row in rows
                       if row["future_identity_target_ids"][0] == row["future_identity_wrong_ids"][0]]
    forbidden = [row["case_id"] for row in rows if any(
        marker in row["future_prompt"]
        for marker in ("?", "？", "Answer", "Options", "Reply with", "只回答", "回答：")
    )]
    replacement = [row["case_id"] for row in rows if "\ufffd" in canonical(row)]
    balance = defaultdict(lambda: Counter())
    for row in rows:
        key = (row["family"], row["language"], row["surface"], row["partition"])
        balance[key][f"state{int(row['state'])}"] += 1
        balance[key][row["target_mention_order"]] += 1
    paired_surface_failures = []
    groups = defaultdict(dict)
    for row in rows:
        groups[(row["family"], row["language"], int(row["unit"]), int(row["state"]))][row["surface"]] = row
    for key, pair in groups.items():
        if set(pair) != set(SURFACES) or len({r["target_mention_order"] for r in pair.values()}) != 1:
            paired_surface_failures.append(key)
    audit = {
        "rows": len(rows),
        "families": list(FAMILIES),
        "languages": dict(Counter(row["language"] for row in rows)),
        "surfaces": dict(Counter(row["surface"] for row in rows)),
        "partitions": dict(Counter(row["partition"] for row in rows)),
        "first_identity_token_collision_count": len(first_collision),
        "forbidden_marker_count": len(forbidden),
        "unicode_replacement_count": len(replacement),
        "balanced_cells": all(
            cell["state0"] == cell["state1"] and cell["first"] == cell["last"]
            for cell in balance.values()
        ),
        "paired_surface_order_failures": len(paired_surface_failures),
        "prompt_token_min_max": [
            min(len(row["future_prompt_ids"]) for row in rows),
            max(len(row["future_prompt_ids"]) for row in rows),
        ],
        "target_future_token_min_max": [
            min(len(row["future_target_ids"]) for row in rows),
            max(len(row["future_target_ids"]) for row in rows),
        ],
        "machine_semantic_audit": (
            "six inherited families retain Phase2306 corrected facts; temporal order uses explicit before/after; "
            "taxonomy uses a closed catalog with explicit positive and negative top-category premises"
        ),
        "machine_naturality_audit": "complete declarative facts and grammatical continuation cues",
        "independent_human_blind_review": "NA_not_run",
    }
    return rows, audit


def append_memo(result: dict) -> None:
    marker = f"## Phase {PHASE}:"
    if marker in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 八构式多步自然未来全坐标大合同（{CAMPAIGN}） [{stamp}]

**证据审查与过度结论修正。** Phase2296--2308 已经可靠建立完整下一 token 词表、全检查点 HiddenState 和最终输出逐坐标分账，但没有证明“动态信念流形”“全息干涉”“损失梯度在运行时推动每层更新”“知识吸引子”或“Koopman 模态”。约 800--1000 的有效参与数只说明在当前物理坐标基和指定输出对比中贡献较宽；正负代数相加也不自动成为物理干涉。前向推理的每层更新由冻结网络函数计算，不是运行时沿训练损失梯度下降。本阶段保留附件中关于完整未来、接口条件、非对角传动和跨模型功能比较的正确方向，但以基础计数、逐坐标分账和预定扰动为主，不预设高级数学结构。

**测试原理与材料。** 在模型加载前冻结八个构式族：施事--受事、态度--事件、比较次序、位置绑定、领属查询、关系从句绑定、时间先后和两跳分类链。每族包含中英双语、narrative/dialogue、32 个 unit、两种事实状态和四个隔离分区，共 `{result['audit']['rows']}` 行。六族继承 Phase2306 已修复的同 unit 同事实顺序材料；时间族用显式 before/after；分类族使用封闭目录并同时写明“属于目标上位类”和“不属于控制上位类”，避免用缺失事实推导否定。同一 unit 的两个表面保持相同目标提及顺序。典型目标未来不是单一名字，而是 `Leo Bell rather than Amina Arden.`；中文使用“正确答案，而不是错误答案。”。独立人类自然度盲评未运行，严格记为 NA。

$$
S(y_{{1:K}}\mid x)=\sum_{{k=1}}^K\log p(y_k\mid x,y_{{<k}}),
\qquad
\Pi_K(x)=\left\{{p(\cdot\mid x,y_{{<k}})\right\}}_{{k=1}}^K.
$$

全场和固定输出方向定义为：
$$
\mathcal H_i=\left(H_{{i,q,p,j}}\right)_{{q,p,j}},
\qquad
m_i^{{fix}}=z_{{a_i}}-z_{{b_i}},
\qquad
g_{{i,q}}=\nabla_{{H_{{i,q}}}}m_i^{{fix}}.
$$

其中同一 unit 的两种状态使用同一个有向 token 对比，不因正确答案交换而翻转读尺。后续局部扰动只检验一阶局部预测：
$$
m(H+\delta)-m(H)\approx g(H)^\top\delta.
$$

**冻结门槛、分流与停止条件。** 完整候选序列在 overall、语言、表面、分区和提及顺序切片的总分与长度均值准确率均不低于 `{BEHAVIOR_GATE}` 才称行为合格；自由身份命中率 `{FREE_IDENTITY_GATE}` 只作为更严格的路线资格之一，不阻断全量观察。Qwen3-4B 对全部八族采集；局部扰动只有 discovery/confirmation 预先达到符号准确率 `{LOCAL_LINEAR_GATES['sign_accuracy']}`、中位相对误差不高于 `{LOCAL_LINEAR_GATES['median_relative_error']}`、正向优于反向比例不低于 `{LOCAL_LINEAR_GATES['forward_reverse_order_rate']}` 的族才进入 fresh 裁决。跨模型按 `Qwen3-4B -> Qwen3-14B -> DeepSeek-7B` 依次运行 fresh 行为，不搬运坐标编号。单族失败只淘汰该分支；全部预定分支完成后结束。

**结果、审计与相关文件。** 本 Phase 不运行模型。材料审计 `{json.dumps(result['audit'], ensure_ascii=False)}`；冻结配置 `{json.dumps(result['config'], ensure_ascii=False)}`；哈希 `{json.dumps(result['hashes'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。脚本 `tests/glm5/phase2309_c4321_c4440_multistep_future_contract.py`；结果 `tests/glm5/result/phase2309_c4321_c4440_multistep_future_contract`。

**理论进展、问题硬伤与结论。** 新增的不是机制阳性，而是一个能区分首 token 竞争、完整多 token 未来、自由续写、表面顺序和跨模型功能的统一观察合同。硬伤包括研究者编写模板、没有独立人类盲评、分类链仍是人工封闭微世界、`rather than` 尾句带有统一输出形式，以及不同模型 tokenizer 不同。现有基础概率与逐坐标微分足够表达合同，没有证据要求新数学。下一步授权 Qwen3-4B BF16 非量化 CUDA 全量运行；所有族均保留观察，只有满足冻结资格的分支进入局部扰动。
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
    parent = json.loads((P2306 / "analysis/final.json").read_text(encoding="utf-8"))
    if not parent["all_checks_passed"]:
        raise RuntimeError("Phase2306 is not authorized")
    rows, audit = compile_rows()
    material_path = OUT / "material/multistep_future_bilingual.jsonl"
    write_rows(material_path, rows)
    config = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "frozen_before_model_load": True,
        "families": list(FAMILIES),
        "languages": list(LANGUAGES),
        "surfaces": list(SURFACES),
        "partitions": list(PARTITIONS),
        "units": UNITS,
        "future_steps": FUTURE_STEPS,
        "qpoints_qwen4b": list(QPOINTS_4B),
        "behavior_gate": BEHAVIOR_GATE,
        "free_identity_gate": FREE_IDENTITY_GATE,
        "local_linear_gates": LOCAL_LINEAR_GATES,
        "model_order": list(MODEL_ORDER),
        "route_policy": "route_level_elimination_not_campaign_hard_stop",
        "observation_policy": "capture_all_families_including_behavior_failures",
        "coordinate_policy": "original_order_all_coordinates_no_topk_no_pca",
        "advanced_math_policy": "not_primary; only after basic heldout residual demands it",
        "raw_cleanup_policy": "delete_only_after_verified_visual_derivative_and_analysis_ledger",
    }
    save(OUT / "config/frozen_contract.json", config)
    review = [{
        "case_id": row["case_id"],
        "naturalness_1_5": None,
        "semantic_unique_0_1": None,
        "reviewer": None,
    } for row in rows if row["partition"] == "fresh_lockbox"]
    write_rows(OUT / "external/human_blind_review_template.jsonl", review)
    hashes = {
        "material": file_hash(material_path),
        "config": file_hash(OUT / "config/frozen_contract.json"),
    }
    checks = {
        "parent_authorized": parent["all_checks_passed"],
        "row_count": len(rows) == len(FAMILIES) * len(LANGUAGES) * len(SURFACES) * UNITS * 2,
        "all_families": set(row["family"] for row in rows) == set(FAMILIES),
        "all_cells_balanced": audit["balanced_cells"],
        "same_unit_surface_order": audit["paired_surface_order_failures"] == 0,
        "no_identity_first_token_collision": audit["first_identity_token_collision_count"] == 0,
        "no_forbidden_markers": audit["forbidden_marker_count"] == 0,
        "unicode_intact": audit["unicode_replacement_count"] == 0,
        "human_review_honest_na": audit["independent_human_blind_review"] == "NA_not_run",
        "model_order_sequential": config["model_order"] == list(MODEL_ORDER),
        "no_advanced_math_assumed": config["advanced_math_policy"].startswith("not_primary"),
    }
    result = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "timestamp": datetime.now().astimezone().isoformat(),
        "status": "closed",
        "audit": audit,
        "config": config,
        "material": {"path": str(material_path.relative_to(ROOT)), "rows": len(rows)},
        "hashes": hashes,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "strict_conclusion": (
            "An eight-family bilingual multi-step raw-continuation campaign is frozen. "
            "No model, HiddenState, gradient, manifold, holographic, or causal result exists in this phase."
        ),
        "next_authorization": (
            "Run Qwen3-4B on every row; retain failures as observations and gate only later local perturbation branches."
        ),
    }
    save(final_path, result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
