#!/usr/bin/env python3
"""Eight-family predicate-validity factorial with Qwen4B/Qwen14B behavior qualification."""
from __future__ import annotations

import gc
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase2423_c30001_c30320_semantic_validity_behavior_contract"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE = 2423
CAMPAIGN = "C30001-C30320"
VARIANTS = ("valid", "broken_a", "broken_b")
ROLES = ("source", "target")

sys.path.insert(0, str(TESTS))
import phase2389_c19121_c19440_crossmodel_autonomous_capability as capability  # noqa: E402
import phase2397_c21681_c22000_operation_behavior_token_calibration as behavior  # noqa: E402
import phase2405_c24241_c24560_deconfounded_operation_contract as contract  # noqa: E402
import phase2412_c26481_c26800_frozen_crossmodel_operator_replication as capture_loader  # noqa: E402


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def partition(unit: int, surface: str) -> str:
    controlled = surface in ("canonical", "paraphrase")
    if controlled:
        return "discovery" if unit < 6 else "fresh_unit_lockbox"
    return "template_lockbox" if unit < 6 else "joint_lockbox"


def compile_source() -> list[dict]:
    families = list(contract.FAMILIES)
    rows: list[dict] = []
    for fi, family in enumerate(families):
        wrong = {"broken_a": families[(fi + 1) % len(families)], "broken_b": families[(fi + 3) % len(families)]}
        for unit in range(8):
            for language in contract.LANGUAGES:
                entity_a, entity_b = contract.pairs(language, unit)
                for si, surface in enumerate(contract.SURFACES):
                    for direction in (0, 1):
                        source, target = (entity_a, entity_b) if direction == 0 else (entity_b, entity_a)
                        order = (fi + unit + si + direction) % 2
                        candidates = [source, target] if order == 0 else [target, source]
                        config_id = f"sv-{family}-u{unit}-{language}-{surface}-d{direction}"
                        for variant in VARIANTS:
                            fact_family = family if variant == "valid" else wrong[variant]
                            fact, spans = contract.render_fact(fact_family, language, surface, source, target)
                            for role in ROLES:
                                answer, foil = (source, target) if role == "source" else (target, source)
                                query = contract.prior.role_query(family, language, role)
                                prompt, events = contract.prior.prompt_with_events(language, [(fact, spans)], query, candidates)
                                rows.append({
                                    "case_id": f"{config_id}-{variant}-{role}", "config_id": config_id,
                                    "task": "semantic_validity_selection",
                                    "family": family, "fact_family": fact_family, "unit": unit,
                                    "language": language, "surface": surface,
                                    "surface_class": "controlled" if surface in ("canonical", "paraphrase") else "naturalized",
                                    "direction": direction, "variant": variant, "query_role": role,
                                    "candidate_order": order, "target_candidate_slot": candidates.index(answer),
                                    "partition": partition(unit, surface), "source": source, "target": target,
                                    "fact": fact, "query": query, "candidates": candidates,
                                    "answer": answer, "foil": foil, "prompt": prompt, "events": events,
                                })
    return rows


def material_audit(rows: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["config_id"]].append(row)
    exact = 0
    for group in grouped.values():
        keys = {(row["variant"], row["query_role"]) for row in group}
        same = all(row["source"] == group[0]["source"] and row["target"] == group[0]["target"] and
                   row["candidates"] == group[0]["candidates"] and row["candidate_order"] == group[0]["candidate_order"]
                   for row in group)
        exact += int(len(group) == 6 and len(keys) == 6 and same)
    return {
        "rows": len(rows), "configurations": len(grouped),
        "families": dict(Counter(row["family"] for row in rows)),
        "variants": dict(Counter(row["variant"] for row in rows)),
        "query_roles": dict(Counter(row["query_role"] for row in rows)),
        "languages": dict(Counter(row["language"] for row in rows)),
        "surfaces": dict(Counter(row["surface"] for row in rows)),
        "partitions": dict(Counter(row["partition"] for row in rows)),
        "six_row_exact_entity_candidate_blocks": exact,
        "unique_cases": len({row["case_id"] for row in rows}) == len(rows),
        "important_correction": "Entities/candidates/order are frozen; predicate phrase changes across validity variants. Phase2422 wording that the whole fact was frozen was too strong.",
    }


def summarize(rows: list[dict], scores: list[dict], metric: str) -> dict:
    joined = [{**row, **score, "metric": score[metric]} for row, score in zip(rows, scores)]
    def one(items: list[dict]) -> dict:
        x = np.asarray([item["metric"] for item in items], dtype=np.float64)
        if metric.endswith("margin"):
            return {"rows": len(items), "target_over_foil": float(np.mean(x > 0)), "mean_margin": float(np.mean(x))}
        return {"rows": len(items), "exact": float(np.mean(x)),
                "target_present": float(np.mean([item.get("target_present", False) for item in items]))}
    present = [variant for variant in VARIANTS if any(item["variant"] == variant for item in joined)]
    by_variant = {variant: one([item for item in joined if item["variant"] == variant]) for variant in present}
    by_variant_role = {variant: {role: one([item for item in joined if item["variant"] == variant and item["query_role"] == role])
                                  for role in ROLES} for variant in present}
    result = {"overall": one(joined), "by_variant": by_variant, "by_variant_role": by_variant_role}
    if set(VARIANTS).issubset(by_variant):
        key = "target_over_foil" if metric.endswith("margin") else "exact"
        result["valid_minus_broken_a"] = by_variant["valid"][key] - by_variant["broken_a"][key]
        result["broken_a_minus_broken_b"] = by_variant["broken_a"][key] - by_variant["broken_b"][key]
    return result


def run_model(key: str, source: list[dict]) -> dict:
    final = OUT / key / "analysis/final.json"
    if final.exists():
        payload = json.loads(final.read_text(encoding="utf-8"))
        rows = read_rows(OUT / key / "index/semantic_validity_rows.jsonl")
        teacher = read_rows(OUT / key / "behavior/teacher_scores.jsonl")
        lockbox = [row for row in rows if row["variant"] == "valid" and row["unit"] >= 6]
        generated = read_rows(OUT / key / "behavior/autonomous_lockbox.jsonl")
        payload["teacher"] = summarize(rows, teacher, "mean_logprob_margin")
        payload["autonomous"] = summarize(lockbox, generated, "exact")
        save(final, payload)
        return payload
    model, tokenizer, label = (capability.load_model(key) if key == "qwen4b" else capture_loader.load_for_capture(key))
    behavior.OUT = OUT
    try:
        index_path = OUT / key / "index/semantic_validity_rows.jsonl"
        if index_path.exists():
            rows = read_rows(index_path)
            calibration = json.loads((OUT / key / "analysis/token_calibration.json").read_text(encoding="utf-8"))
            if rows and "task" not in rows[0]:
                rows, calibration = behavior.compile_rows(tokenizer, source)
                write_rows(index_path, rows)
                save(OUT / key / "analysis/token_calibration.json", calibration)
        else:
            rows, calibration = behavior.compile_rows(tokenizer, source)
            write_rows(index_path, rows)
            save(OUT / key / "analysis/token_calibration.json", calibration)
        teacher, _ = behavior.score_rows(key, model, rows, 16 if key == "qwen4b" else 4)
        teacher_summary = summarize(rows, teacher, "mean_logprob_margin")
        lockbox = [row for row in rows if row["variant"] == "valid" and row["unit"] >= 6]
        generated, _ = behavior.generate_lockbox(key, model, tokenizer, lockbox, 8 if key == "qwen4b" else 3)
        autonomous_summary = summarize(lockbox, generated, "exact")
    finally:
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    checks = {
        "compiled_6144": calibration["rows"] == 6144,
        "monotonic_events": calibration["event_monotonic_rate"] == 1.0,
        "teacher_6144": len(teacher) == 6144,
        "autonomous_valid_lockbox_512": len(generated) == 512,
        "finite_teacher": math.isfinite(teacher_summary["overall"]["mean_margin"]),
    }
    result = {"model": key, "model_label": label,
              "precision": "BF16 weights" if key == "qwen4b" else "NF4 weights / BF16 compute; secondary capability qualification",
              "calibration": calibration, "teacher": teacher_summary, "autonomous": autonomous_summary,
              "checks": checks, "all_checks_passed": all(checks.values())}
    save(final, result)
    return result


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    q4, q14 = result["models"]["qwen4b"], result["models"]["qwen14b"]
    text = rf"""

## Phase {PHASE}: 八关系族谓词有效性双角色合同与Qwen4B/Qwen14B行为资格（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 建立八关系族×八unit×中英×四种整句表面×双方向的1024配置。每配置固定实体、候选和候选顺序，以正确谓词、循环偏移1的错误谓词A、循环偏移3的错误谓词B构造三种有效性，并分别询问source/target角色，共6144条。这样二阶差分会消去共同实体、候选槽和查询角色主效应；错误A—错误B提供同规模谓词词项基线。补正Phase2422合同措辞：冻结的是实体/候选/顺序，完整fact不能冻结，因为谓词有效性正是自变量。Qwen4B使用BF16权重；Qwen14B此前BF16权重`device_map=auto`多次造成Windows访问冲突，本Phase不重复不安全尝试，以NF4权重/BF16计算只作能力资格，绝不作幅度等价。

$$D_v=S_{{v,target}}-S_{{v,source}},\quad I_{{sem}}=D_{{valid}}-D_{{brokenA}},\quad I_{{lex}}=D_{{brokenA}}-D_{{brokenB}}.$$

$$\Delta_{{beh}}=P(margin>0\mid valid)-P(margin>0\mid brokenA).$$

**结果汇总。** 材料 `{json.dumps(result['material'], ensure_ascii=False)}`；Qwen4B教师强制 `{json.dumps(q4['teacher'], ensure_ascii=False)}`；Qwen4B自主锁箱 `{json.dumps(q4['autonomous'], ensure_ascii=False)}`；Qwen14B教师强制 `{json.dumps(q14['teacher'], ensure_ascii=False)}`；Qwen14B自主锁箱 `{json.dumps(q14['autonomous'], ensure_ascii=False)}`；精度台账 `{{"qwen4b": "{q4['precision']}", "qwen14b": "{q14['precision']}"}}`；裁决 `{json.dumps(result['adjudication'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2423_c30001_c30320_semantic_validity_behavior_contract.py`；6144条源材料、两模型独立token索引、教师强制分数、自主锁箱输出和final位于`tests/glm5/result/phase2423_c30001_c30320_semantic_validity_behavior_contract`。未修改其他Markdown。

**分析与理论进展。** 本Phase不是把“错误谓词”当作自然语言反例，而是构造同实体同槽的语义有效性最小对照。只有行为上模型能区分正确/错误谓词，后续场差异才有能力解释；若Qwen4B弱而Qwen14B强，Qwen4B场只能解释粗糙编码，不能据其阴性否定机制。双角色配对还为下一阶段逐坐标交互提供了不依赖搬运单条差分的重复样本图谱。

**问题硬伤与结论。** 不同关系谓词token长度未严格相同；错误谓词句子本身语法成立，只是与查询关系不一致，因此同时改变语义一致性与词项。brokenA—brokenB只能估计同规模词项替换，不能保证难度相同。教师强制margin和二候选自主答案均不是开放生成。Qwen14B量化权重只用于行为资格。是否存在语义专属坐标律必须由Phase2424–2428的完整场与多重零假设决定。
"""
    with MEMO.open("a", encoding="utf-8", newline="") as stream:
        stream.write(text)


def main() -> None:
    source_path = OUT / "material/semantic_validity_source.jsonl"
    if source_path.exists():
        source = read_rows(source_path)
        if source and "task" not in source[0]:
            source = compile_source()
            write_rows(source_path, source)
    else:
        source = compile_source()
        write_rows(source_path, source)
    material = material_audit(source)
    save(OUT / "material/material_audit.json", material)
    models = {key: run_model(key, source) for key in ("qwen4b", "qwen14b")}
    adjudication = {
        "qwen4b_behavior_qualified": models["qwen4b"]["teacher"]["valid_minus_broken_a"] > 0.05,
        "qwen14b_behavior_qualified": models["qwen14b"]["teacher"]["valid_minus_broken_a"] > 0.05,
        "autonomous_bridge_closed": False,
        "semantic_coordinate_operator_proven": False,
    }
    checks = {
        "material_6144": material["rows"] == 6144,
        "configurations_1024": material["configurations"] == 1024,
        "exact_six_row_blocks": material["six_row_exact_entity_candidate_blocks"] == 1024,
        "qwen4b_complete": models["qwen4b"]["all_checks_passed"],
        "qwen14b_complete": models["qwen14b"]["all_checks_passed"],
        "precision_labeled": models["qwen14b"]["precision"].startswith("NF4"),
        "claim_boundary": not adjudication["semantic_coordinate_operator_proven"],
    }
    result = {"phase": PHASE, "campaign": CAMPAIGN, "material": material, "models": models,
              "adjudication": adjudication, "checks": checks, "all_checks_passed": all(checks.values())}
    save(OUT / "analysis/final.json", result)
    append_memo(result)
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
