#!/usr/bin/env python3
"""Audit Phase 2343-2350 evidence and classify Phase 2350 generation failures."""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase2351_c9121_c9240_evidence_generation_audit"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
CATALOG = ROOT / "frontend/public/research_data/current/language_encoding_catalog.json"
VIS = ROOT / "frontend/public/vis_data/research_kernel"
P2350 = RESULT / "phase2350_c8961_c9120_compositional_graph_natural_response_closure"
GENERATION = P2350 / "raw/lockbox_generation.jsonl"
PHASE = 2351
CAMPAIGN = "C9121-C9240"
ATTACHMENTS = (
    Path(r"C:\Users\Admin\.codex\attachments\128f335f-821e-4b81-91eb-5289128e936e\pasted-text.txt"),
    Path(r"C:\Users\Admin\.codex\attachments\7bbded01-6a8e-4cd8-b666-5be89756c1a2\pasted-text.txt"),
)
PHASE_DIRS = {
    2343: "phase2343_c8121_c8240_absolute_hidden_weight_baseline_audit",
    2344: "phase2344_c8241_c8360_bilingual_factorial_semantic_graph_contract",
    2345: "phase2345_c8361_c8480_qwen4b_bilingual_factorial_full_field",
    2346: "phase2346_c8481_c8600_factorial_coordinate_route_competition",
    2347: "phase2347_c8601_c8720_task_policy_formation_and_cleanup",
    2348: "phase2348_c8721_c8840_supported_task_continuous_coalition_causality",
    2349: "phase2349_c8841_c8960_crossmodel_supported_task_functional_atlas",
    2350: "phase2350_c8961_c9120_compositional_graph_natural_response_closure",
}
NEW_DATASETS = {
    "c8121_qwen4b_absolute_hidden_output_weight_control_passport": (2343, "C8121-C8240"),
    "c8361_qwen4b_bilingual_factorial_key_checkpoint_hiddenstate": (2345, "C8361-C8480"),
    "c8362_qwen4b_bilingual_factorial_key_checkpoint_contribution": (2345, "C8361-C8480"),
    "c8363_qwen4b_bilingual_factorial_reference_all_token_all_checkpoint": (2345, "C8361-C8480"),
    "c8481_qwen4b_bilingual_factorial_route_competition_passport": (2346, "C8481-C8600"),
    "c8601_qwen4b_task_policy_family_checkpoint_means": (2347, "C8601-C8720"),
    "c8602_qwen4b_task_policy_all_token_normalized_bins": (2347, "C8601-C8720"),
    "c8721_qwen4b_supported_task_continuous_family_coalitions": (2348, "C8721-C8840"),
    "c8841_qwen14b_supported_task_key_checkpoint_hiddenstate": (2349, "C8841-C8960"),
    "c8841_glm4_supported_task_key_checkpoint_hiddenstate": (2349, "C8841-C8960"),
    "c8841_deepseek7b_supported_task_key_checkpoint_hiddenstate": (2349, "C8841-C8960"),
    "c8961_qwen4b_compositional_natural_key_checkpoint_hiddenstate": (2350, "C8961-C9120"),
    "c8962_qwen4b_compositional_response_route_passport": (2350, "C8961-C9120"),
}

sys.path.insert(0, str(TESTS))
import phase2315_c5041_c5100_active_response_contract as io  # noqa: E402
import phase2319_c5321_c5400_active_response_atlas_cleanup as atlas  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def classify(row: dict) -> dict:
    generated = row["generated"].strip()
    target = row["target"].strip()
    first_line = generated.splitlines()[0].strip() if generated else ""
    found = re.search(r"(?:actor|item|class|domain)-\d{3}", generated)
    first_identifier = found.group(0) if found else ""
    if generated == target:
        category = "raw_exact"
    elif first_line == target:
        category = "correct_first_line_then_continues"
    elif first_identifier == target:
        category = "correct_identifier_embedded"
    elif re.fullmatch(r"\d{3}", first_line) and target.endswith(first_line):
        category = "answer_type_prefix_omitted"
    elif first_identifier:
        category = "wrong_identifier"
    else:
        category = "no_identifier"
    return {**row, "first_line": first_line, "first_identifier": first_identifier,
            "first_line_exact": first_line == target, "identifier_exact": first_identifier == target,
            "target_anywhere": target in generated, "category": category}


def generation_audit() -> dict:
    rows = [classify(row) for row in io.read_rows(GENERATION)]
    io.write_rows(OUT / "analysis/generation_error_taxonomy.jsonl", rows)
    metrics = {
        "rows": len(rows),
        "raw_exact": sum(row["exact"] for row in rows) / len(rows),
        "target_prefix": sum(row["prefix"] for row in rows) / len(rows),
        "first_line_exact": sum(row["first_line_exact"] for row in rows) / len(rows),
        "first_identifier_exact": sum(row["identifier_exact"] for row in rows) / len(rows),
        "target_anywhere": sum(row["target_anywhere"] for row in rows) / len(rows),
        "categories": dict(Counter(row["category"] for row in rows)),
    }
    cells = {}
    for factor in ("family", "language", "query"):
        groups = defaultdict(list)
        for row in rows:
            groups[row[factor]].append(row)
        cells[factor] = {key: {"rows": len(values),
                               "first_identifier_exact": sum(v["identifier_exact"] for v in values) / len(values),
                               "first_line_exact": sum(v["first_line_exact"] for v in values) / len(values)}
                         for key, values in sorted(groups.items())}
    return {"metrics": metrics, "cells": cells,
            "adjudication": "raw exact=0 is mainly a stop/continuation metric; semantic identifier accuracy is reported separately and never renamed exact generation"}


def phase_audit() -> dict:
    memo = MEMO.read_text(encoding="utf-8")
    records = {}
    for phase, directory in PHASE_DIRS.items():
        path = RESULT / directory / "analysis/final.json"
        value = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        records[str(phase)] = {"final_exists": path.exists(), "memo_heading_count": len(re.findall(
            rf"^## Phase {phase}:", memo, flags=re.MULTILINE)),
            "engineering_checks_passed": value.get("all_checks_passed")}
    return {"continuous": sorted(PHASE_DIRS) == list(range(2343, 2351)), "records": records,
            "all_final": all(v["final_exists"] for v in records.values()),
            "all_memo_once": all(v["memo_heading_count"] == 1 for v in records.values()),
            "all_engineering_checks": all(v["engineering_checks_passed"] for v in records.values())}


def attachment_audit() -> dict:
    return {
        "sources": [{"path": str(path), "sha256": sha256(path), "lines": len(path.read_text(encoding="utf-8").splitlines())}
                    for path in ATTACHMENTS],
        "retained": [
            "Phase2350 teacher-forced preference=1.0, prefix=0.912109375, raw exact=0 and causal bridge=false must be separated.",
            "The q35 compositional absolute-H factorial residual transfers across language, surface and query roles on the fresh lockbox, while row-sorting destroys coordinate identity.",
            "Teacher forcing, controlled identifiers, a common three-edge scaffold and prompt-boundary states do not establish graph composition or a causal mechanism.",
            "Phase2348 multiplicative deletion is vulnerable to RMSNorm rescaling and failed selectivity/rescue gates; a norm-preserving matched intervention is warranted.",
        ],
        "corrected_or_rejected": [
            "Rejected: Phase2346 showed all known representations were unstable. It passed its descriptive gate on the behavior-qualified supported-task subdomain; the all-task mechanism gate failed.",
            "Rejected: cross-language alignment continually failed. Phase2350 bilingual lockbox transfers were 0.75 and 1.0; earlier failures and later success have different material boundaries.",
            "Rejected: the data proves a conditional manifold, transient topology, category-theoretic equivalence, or new algebra. These remain hypotheses, not observations.",
            "Corrected: target-prefix success is not natural-generation closure, but raw exact=0 is also not proof that semantic generation failed; error taxonomy is required.",
            "Corrected: activation coordinates are HiddenState values, not model parameters; embedding/unembedding weights are parameters and must be labeled separately.",
        ],
        "evidence_status": "A coordinate-address-dependent observational response atlas exists in qualified domains; no selective causal gear or universal cross-model code has been established.",
    }


def repair_catalog_provenance() -> dict:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    changed = []
    for row in catalog.get("datasets", []):
        if row.get("id") in NEW_DATASETS:
            phase, campaign = NEW_DATASETS[row["id"]]
            if row.get("phase") != phase or row.get("campaign") != campaign:
                row["phase"], row["campaign"] = phase, campaign
                changed.append(row["id"])
    CATALOG.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metadata_changed = []
    for dataset_id, (phase, campaign) in NEW_DATASETS.items():
        path = VIS / f"{dataset_id}.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("phase") != phase or record.get("campaign") != campaign:
            record["phase"], record["campaign"] = phase, campaign
            path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            metadata_changed.append(dataset_id)
    return {"changed": changed, "metadata_changed": metadata_changed, "verified_ids": sorted(NEW_DATASETS),
            "all_present": all(any(row.get("id") == dataset_id for row in catalog.get("datasets", []))
                               for dataset_id in NEW_DATASETS)}


def verify_assets() -> list[dict]:
    results = []
    for dataset_id in sorted(NEW_DATASETS):
        metadata = VIS / f"{dataset_id}.json"
        record = json.loads(metadata.read_text(encoding="utf-8"))
        binary = ROOT / "frontend/public" / record["binary_url"].lstrip("/")
        results.append({"id": dataset_id, "metadata_exists": metadata.exists(), "binary_exists": binary.exists(),
                        "phase": record.get("phase"), "campaign": record.get("campaign"),
                        "coordinate_count": record.get("coordinate_count"),
                        "sha256": binary.exists() and sha256(binary) == record.get("binary_sha256")})
    return results


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 附件证据重审与自然生成零精确率的错误分类（{CAMPAIGN}） [{stamp}]

**测试原理、测试用例与公式。** 对两份附件逐项回查Phase2343–2350的`final.json`、MEMO和512条fresh-lockbox贪心输出。严格区分原始字符串完全相等、第一行相等、首个结构化标识符相等和仅目标前缀相等；例如`item-012\n\nThe text...`不是raw exact，但第一行和语义标识符均正确。证据分级仍为观察、复现、机制，后者不能由前者推出。

$$
A_{{raw}}=\frac1N\sum_i[\hat y_i=y_i],\quad
A_{{line}}=\frac1N\sum_i[\operatorname{{line}}_1(\hat y_i)=y_i],\quad
A_{{id}}=\frac1N\sum_i[\operatorname{{firstID}}(\hat y_i)=y_i].
$$

$$
\mathcal E_{{obs}}\supseteq\mathcal E_{{rep}}\supseteq\mathcal E_{{mech}},
\qquad \mathcal E_{{obs}}\not\Rightarrow\mathcal E_{{mech}}.
$$

**结果汇总。** 生成审计 `{json.dumps(result['generation'], ensure_ascii=False)}`。Phase连续性 `{json.dumps(result['phase_audit'], ensure_ascii=False)}`。附件裁决 `{json.dumps(result['attachment_audit'], ensure_ascii=False)}`。可视化来源修复 `{json.dumps(result['catalog_repair'], ensure_ascii=False)}`，核验 `{json.dumps(result['asset_verification'], ensure_ascii=False)}`。

**相关文件。** 脚本 `tests/glm5/phase2351_c9121_c9240_evidence_generation_audit.py`；结果 `tests/glm5/result/phase2351_c9121_c9240_evidence_generation_audit`；逐行错误表 `analysis/generation_error_taxonomy.jsonl`。

**理论进展、问题硬伤与结论。** raw exact为0首先暴露停止条件/续写格式问题，不能被改名成自然生成成功；同时也不能据此断言语义答案全错，必须看首行与首标识符。可以保留“q35组合族图谱依赖具体坐标地址且跨英中、表述、查询角色迁移”；不能保留“所有路线全面失败”“跨语言持续失败”“已发现条件流形/等价代数”。当前证据只到行为合格域中的描述性坐标图谱，Phase2348因果桥仍失败。

**下一大阶段冻结方案。** 连续执行：(1)多未来自然指令合同和生成token全坐标瞬态；(2)条件等价路线竞赛及未见词汇/表述锁箱；(3)保范数匹配干预、错族/错时刻/置乱控制与独立救援；(4)Qwen14B、GLM4、DS7B顺序功能复验；(5)可视化、清理和总审计。只在这些基础门出现稳定证据后尝试更高数学，不预设流形或范畴结构。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def main() -> None:
    final_path = OUT / "analysis/final.json"
    if final_path.exists():
        result = json.loads(final_path.read_text(encoding="utf-8")); append_memo(result)
        print(json.dumps(result, ensure_ascii=False, indent=2)); return
    generation = generation_audit()
    audit = phase_audit()
    attachments = attachment_audit()
    catalog = repair_catalog_provenance()
    verification = verify_assets()
    build = atlas.frontend_build()
    checks = {"generation_rows": generation["metrics"]["rows"] == 512,
              "phase_continuity": audit["continuous"] and audit["all_final"] and audit["all_memo_once"],
              "engineering_checks": audit["all_engineering_checks"], "catalog_ids": catalog["all_present"],
              "asset_hashes": all(row["sha256"] for row in verification), "frontend_build": build["passed"]}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "generation": generation, "phase_audit": audit,
              "attachment_audit": attachments, "catalog_repair": catalog, "asset_verification": verification,
              "frontend_build": build, "checks": checks, "all_checks_passed": all(checks.values())}
    save(final_path, result)
    if not result["all_checks_passed"]:
        raise RuntimeError(("phase2351_failed", checks))
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
