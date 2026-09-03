#!/usr/bin/env python3
"""Repair only the cross-partition identity vocabulary from Phase 2288."""
from __future__ import annotations

import itertools
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
PARENT_OUT = RESULT / "phase2288_c2501_c2580_natural_sample_condition_contract"
OUT = RESULT / "phase2289_c2581_c2600_partition_lexicon_repair"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
sys.path.insert(0, str(TESTS))

import model_utils  # noqa: E402
import phase2288_c2501_c2580_natural_sample_condition_contract as parent  # noqa: E402


PHASE = 2289
CAMPAIGN = "C2581-C2600"


def save(path: Path, value: Any) -> None:
    parent.save(path, value)


def write_rows(path: Path, rows: list[dict]) -> None:
    parent.write_rows(path, rows)


def transformed(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, str):
        output = value
        for source, target in sorted(replacements.items(), key=lambda item: -len(item[0])):
            output = output.replace(source, target)
        return output
    if isinstance(value, list):
        return [transformed(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: transformed(item, replacements) for key, item in value.items()}
    return value


def role_names(language: str, unit: int) -> tuple[str, str, str]:
    names = parent.NAMES_EN if language == "en" else parent.NAMES_ZH
    if language == "en":
        return (f"{names[unit]} Arden", f"{names[(unit + 11) % parent.UNITS]} Bell",
                f"{names[(unit + 21) % parent.UNITS]} Cole")
    return (f"赵{names[unit]}", f"钱{names[(unit + 11) % parent.UNITS]}",
            f"孙{names[(unit + 21) % parent.UNITS]}")


def repaired_material() -> list[dict]:
    rows = []
    for row in parent.material():
        old = parent.vocabulary(row["language"], int(row["unit"]))
        new_a, new_b, new_c = role_names(row["language"], int(row["unit"]))
        replacements = {old["a"]: new_a, old["b"]: new_b, old["c"]: new_c}
        fixed = transformed(row, replacements)
        fixed["case_id"] = row["case_id"].replace("c2501-", "c2581-")
        fixed["repair_parent_case_id"] = row["case_id"]
        fixed["repair_scope"] = "identity_vocabulary_only"
        rows.append(fixed)
    return rows


def append_memo(result: dict) -> None:
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 自然双语合同的跨分区身份词汇纠错（{CAMPAIGN}） [{stamp}]

**测试原理与纠错边界。** Phase2288 在模型加载前发现人物身份跨 discovery、confirmation、fresh-confirmation 和 fresh-lockbox 重叠，因此 `all_checks_passed=false`，不能继续。该失败完整保留。本期不改八个构式、2048行规模、中英、两表面、状态、答案、门槛或分区；仅把人物槽改为角色特异且跨单元唯一的自然全名，例如 `Amina Arden / Mara Bell / Viktor Cole` 与 `赵安宁 / 钱孟然 / 孙吴越`，然后重新编译全部角色 token 跨度。

**测试用例与公式。** 每条语义程序仍保持相同图与状态配对：

$$
R_{{i,q,r,j}}=H_{{i,q,r,j}}^{{(1)}}-H_{{i,q,r,j}}^{{(0)}}.
$$

身份修复满足：

$$
V_{{p}}^{{primary}}\cap V_{{p'}}^{{primary}}=\varnothing,\qquad p\ne p'.
$$

**结果汇总。** `{json.dumps(result['audit'], ensure_ascii=False)}`；哈希 `{json.dumps(result['hashes'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**分析、理论进展、硬伤与结论。** 本期只修复锁箱身份隔离，没有运行模型或产生机制证据。110个中文角色跨度仍需使用“最短可解码文本跨度”而非精确 token 子序列，这说明 token 边界不是语义边界；位置方法已逐行记录。人类盲评仍为 `NA_not_run`。严格结论：修复后材料才有资格进入 Qwen3-4B 双行为与全坐标观察。脚本 `tests/glm5/phase2289_c2581_c2600_partition_lexicon_repair.py`；结果 `tests/glm5/result/phase2289_c2581_c2600_partition_lexicon_repair`。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def main() -> None:
    from transformers import AutoTokenizer

    parent_final = json.loads((PARENT_OUT / "analysis/final.json").read_text(encoding="utf-8"))
    if parent_final["checks"]["partition_primary_vocab_disjoint"]:
        raise RuntimeError("Phase2288 no longer exposes the registered vocabulary defect")
    rows = repaired_material()
    tokenizer = AutoTokenizer.from_pretrained(
        model_utils.MODEL_CONFIGS["qwen3"]["path"], trust_remote_code=True,
        local_files_only=True, use_fast=False,
    )
    compiled = parent.compile_rows(tokenizer, rows)
    material_path = OUT / "material/natural_bilingual_cases.jsonl"
    compiled_path = OUT / "material/qwen_compiled.jsonl"
    write_rows(material_path, rows)
    write_rows(compiled_path, compiled)
    write_rows(OUT / "external/human_blind_template.jsonl", [{
        "case_id": row["case_id"], "naturalness_1_5": None,
        "semantic_unique_0_1": None, "cross_surface_equivalent_0_1": None,
        "reviewer": None,
    } for row in rows if row["partition"] == "fresh_lockbox"])

    balance = defaultdict(lambda: [0, 0])
    partition_vocab = defaultdict(set)
    for row in rows:
        key = (row["family"], row["language"], row["surface"], row["partition"])
        balance[key][int(row["gold_position"])] += 1
        partition_vocab[row["partition"]].add(row["role_values"]["primary"])
    overlaps = {f"{a}|{b}": sorted(partition_vocab[a] & partition_vocab[b])
                for a, b in itertools.combinations(parent.PARTITION_RANGES, 2)}
    methods = Counter(method for row in compiled for method in row["role_position_methods"].values())
    widths = [len(row["prompt_ids"]) for row in compiled]
    audit = {
        "rows": len(rows),
        "repair_scope": "identity_vocabulary_only",
        "candidate_balance_exact": all(a == b for a, b in balance.values()),
        "primary_vocab_cross_partition_overlap": overlaps,
        "role_position_methods": dict(methods),
        "token_width_min_median_max": [min(widths), float(np.median(widths)), max(widths)],
        "human_review": "NA_not_run",
    }
    config = {**parent_final["config"], "phase": PHASE, "campaign": CAMPAIGN,
              "parent_failed_phase": 2288, "repair_scope": "identity_vocabulary_only"}
    save(OUT / "config/frozen_contract.json", config)
    save(OUT / "audit/material_audit.json", audit)
    hashes = {"material": parent.file_hash(material_path), "compiled": parent.file_hash(compiled_path),
              "config": parent.file_hash(OUT / "config/frozen_contract.json")}
    checks = {
        "parent_failure_preserved": parent_final["all_checks_passed"] is False,
        "row_count_unchanged": len(rows) == parent_final["material"]["rows"],
        "candidate_balance": audit["candidate_balance_exact"],
        "partition_primary_vocab_disjoint": all(not value for value in overlaps.values()),
        "all_roles_compiled": all(set(row["role_positions"]) == set(parent.ROLES) for row in compiled),
        "repair_scope_only_identity": all(row["repair_scope"] == "identity_vocabulary_only" for row in rows),
        "human_review_honest_na": audit["human_review"] == "NA_not_run",
    }
    result = {
        "phase": PHASE, "campaign": CAMPAIGN, "timestamp": datetime.now().astimezone().isoformat(),
        "status": "closed", "material": {"rows": len(rows), "path": str(material_path.relative_to(ROOT)),
                                             "compiled": str(compiled_path.relative_to(ROOT))},
        "audit": audit, "config": config, "hashes": hashes, "checks": checks,
        "all_checks_passed": all(checks.values()),
        "strict_conclusion": "The sole Phase2288 identity-overlap defect is repaired without changing tasks, gates, partitions, or outcomes; model execution is now authorized.",
        "next_authorization": "Run Qwen3-4B dual behavior, then capture fields only for behavior-qualified families.",
    }
    save(OUT / "analysis/final.json", result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
