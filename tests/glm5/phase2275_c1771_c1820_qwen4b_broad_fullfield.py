#!/usr/bin/env python3
"""Run Qwen3-4B behavior and full-coordinate observation for Phase 2275."""
from __future__ import annotations

import gc
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
CONTRACT_OUT = RESULT / "phase2274_c1721_c1770_broad_construction_contract"
OUT = RESULT / "phase2275_c1771_c1820_qwen4b_broad_fullfield"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
sys.path.insert(0, str(TESTS))

import phase2266_c1469_c1504_qwen4b_independent_fullfield as capture_base  # noqa: E402
import phase2274_c1721_c1770_broad_construction_contract as contract  # noqa: E402


PHASE = 2275
CAMPAIGN = "C1771-C1820"
ROLE_FIELD = OUT / "raw/qwen3_4b_broad_role_field.float16.npy"
ROLE_INDEX = OUT / "raw/role_field_index.jsonl"
ROLE_PROGRESS = OUT / "raw/role_field_progress.json"
TOKEN_FIELD = OUT / "raw/qwen3_4b_broad_all_token_field.float16.npy"
TOKEN_INDEX = OUT / "raw/all_token_field_index.jsonl"
TOKEN_PROGRESS = OUT / "raw/all_token_field_progress.json"
TOKEN_UNITS = (16, 24)


def save(path: Path, value: Any) -> None:
    contract.save(path, value)


def load(path: Path) -> Any:
    return contract.load(path)


def read_rows(path: Path) -> list[dict]:
    return contract.read_rows(path)


def write_rows(path: Path, rows: list[dict]) -> None:
    contract.write_rows(path, rows)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def behavior_ledger(rows: list[dict], candidates: list[dict], generated: list[dict]) -> dict:
    c = {row["case_id"]: row for row in candidates}
    g = {row["case_id"]: row for row in generated}

    def summarize(subset: list[dict]) -> dict:
        ca = float(np.mean([c[row["case_id"]]["correct"] for row in subset]))
        ga = float(np.mean([g[row["case_id"]]["correct"] for row in subset]))
        return {"rows": len(subset), "candidate_accuracy": ca, "generation_accuracy": ga,
                "dual_qualified": min(ca, ga) >= contract.BEHAVIOR_GATE}

    family_groups: dict[str, list[dict]] = defaultdict(list)
    partition_groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    surface_groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    scheme_groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        family_groups[row["family"]].append(row)
        partition_groups[(row["family"], row["partition"])].append(row)
        surface_groups[(row["family"], row["surface"])].append(row)
        scheme_groups[(row["family"], int(row["output_scheme"]))].append(row)
    families = {key: summarize(value) for key, value in sorted(family_groups.items())}
    partitions = {"|".join(key): summarize(value) for key, value in sorted(partition_groups.items())}
    surfaces = {"|".join(key): summarize(value) for key, value in sorted(surface_groups.items())}
    schemes = {f"{key[0]}|{key[1]}": summarize(value) for key, value in sorted(scheme_groups.items())}
    qualified, qualification = [], {}
    for family in contract.FAMILIES:
        required = {
            "overall": families[family],
            "discovery": partitions[f"{family}|discovery"],
            "fresh_confirmation": partitions[f"{family}|fresh_confirmation"],
        }
        passed = all(value["dual_qualified"] for value in required.values())
        qualification[family] = {"qualified": passed, "required": required}
        if passed:
            qualified.append(family)
    aggregate = summarize(rows)
    return {**aggregate,
            "parsed_generation_fraction": float(np.mean([row["parsed"] is not None for row in generated])),
            "families": families, "partitions": partitions, "surfaces": surfaces,
            "output_schemes": schemes, "qualification_audit": qualification,
            "qualified_families": qualified}


def index_rows(rows: list[dict], candidates: list[dict], generated: list[dict]) -> list[dict]:
    c = {row["case_id"]: row for row in candidates}
    g = {row["case_id"]: row for row in generated}
    return [{
        "hidden_index": i,
        "case_id": row["case_id"],
        "family": row["family"],
        "language": row["language"],
        "unit": row["unit"],
        "surface": row["surface"],
        "state": row["state"],
        "partition": row["partition"],
        "fresh": row["fresh"],
        "role_positions": row["role_positions"],
        "prompt_length": len(row["prompt_ids"]),
        "prompt_ids": row["prompt_ids"],
        "factors": row["factors"],
        "output_scheme": row["output_scheme"],
        "gold_position": row["gold_position"],
        "true_code": row["true_code"],
        "false_code": row["false_code"],
        "candidate_correct": bool(c[row["case_id"]]["correct"]),
        "generation_correct": bool(g[row["case_id"]]["correct"]),
    } for i, row in enumerate(rows)]


def configure_capture_base() -> None:
    capture_base.ROLE_FIELD = ROLE_FIELD
    capture_base.ROLE_INDEX = ROLE_INDEX
    capture_base.ROLE_PROGRESS = ROLE_PROGRESS
    capture_base.TOKEN_FIELD = TOKEN_FIELD
    capture_base.TOKEN_INDEX = TOKEN_INDEX
    capture_base.TOKEN_PROGRESS = TOKEN_PROGRESS
    capture_base.TOKEN_UNITS = TOKEN_UNITS


def append_memo(result: dict) -> None:
    current = MEMO.read_text(encoding="utf-8-sig")
    if f"## Phase {PHASE}:" in current:
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    behavior = result["behavior"]
    text = rf"""

## Phase {PHASE}: Qwen3-4B 十六构式双行为与全坐标观察（{CAMPAIGN}） [{stamp}]

**测试原理与用例。** 严格使用 Phase2274 冻结的 3072 条英文材料，不修改句子、分区、答案码或门槛。候选 A/B 与精确自由生成分别记账；一个构式只有在总体、discovery 和 fresh confirmation 上两种准确率均不低于 0.75，才获得 HiddenState 观察资格。错误回答样本不后筛。每个合格构式保存全部样本的 embedding、36 个 block 后状态、final norm、六个预注册角色和全部 2560 个物理激活坐标；另外对 unit16 与 unit24 保存每个真实 prompt token 的同一完整场。

**公式。** 行为门与观察场为：

$$
Q_f=\prod_{{p\in\{{\mathrm{{all}},\mathrm{{discovery}},\mathrm{{fresh\ confirmation}}\}}}}
\mathbf 1\!\left[A^{{\mathrm{{cand}}}}_{{f,p}}\ge0.75\right]
\mathbf 1\!\left[A^{{\mathrm{{gen}}}}_{{f,p}}\ge0.75\right],
$$

$$
\mathcal F_f=\left\{{H_{{i,q,r,j}}\right\}}_{{i,q,r,j}},\qquad
\mathcal T_f=\left\{{H_{{i,q,t,j}}\right\}}_{{i,q,t,j}}.
$$

其中 $j$ 是运行时 HiddenState 激活坐标，不是权重参数；本期不读取 Attention、MLP、权重或梯度，也不使用 PCA、Top-K、余弦或差分搬运发现规律。

**结果汇总与门槛。** 总体候选准确率 `{behavior['candidate_accuracy']}`，精确自由生成 `{behavior['generation_accuracy']}`，解析率 `{behavior['parsed_generation_fraction']}`；16 类逐族结果 `{json.dumps(behavior['families'], ensure_ascii=False)}`；合格构式 `{json.dumps(behavior['qualified_families'], ensure_ascii=False)}`。六角色原场 `{json.dumps(result['role_field'], ensure_ascii=False)}`；全 token 原场 `{json.dumps(result['all_token_field'], ensure_ascii=False)}`；量化审计 `{json.dumps(result['quantization'], ensure_ascii=False)}`；哈希 `{json.dumps(result['hashes'], ensure_ascii=False)}`；工程检查 `{json.dumps(result['checks'], ensure_ascii=False)}`，总通过 `{result['all_checks_passed']}`。

**分析与理论进展。** 本期只建立观察资格与完整测量场。合格表示小模型能稳定执行该受控构式，不等于已经定位编码机制；不合格只把该构式内部结果记为 NA。三表面的分账允许下一阶段检查同一坐标规律是否跨直接句、释义和无关上下文保持；四套输出码允许检查晚层结果是否只是答案编译。理论主体和 RDC 不改名。

**问题、硬伤与结论。** 材料仍是研究者模板、英文单语、元语言代码输出且缺少独立人类盲评；float16 落盘与角色末 token 是测量近似；全场含错误回答样本，可能混入任务失败轨迹；同坐标可预测性仍可能来自一般残差流动力。严格结论：`{result['strict_conclusion']}` 下一步只能在冻结 discovery 上形成逐坐标模型，经 confirmation 与 fresh confirmation 固定后再揭示 fresh lockbox。

**相关文件。** 脚本 `tests/glm5/phase2275_c1771_c1820_qwen4b_broad_fullfield.py`；结果 `tests/glm5/result/phase2275_c1771_c1820_qwen4b_broad_fullfield`。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def run() -> dict:
    final = OUT / "analysis/final.json"
    if final.exists():
        result = load(final)
        append_memo(result)
        return result
    contract_result = load(CONTRACT_OUT / "analysis/final.json")
    if not contract_result["all_checks_passed"]:
        raise RuntimeError("Phase2274 contract is invalid")
    rows = read_rows(CONTRACT_OUT / "material/broad_construction_qwen_compiled.jsonl")
    candidate_path = OUT / "behavior/candidate.jsonl"
    generation_path = OUT / "behavior/generation.jsonl"
    model = None
    candidates: list[dict] = []
    generated: list[dict] = []
    try:
        model, tokenizer, device, placement = contract.previous.legacy.parent.model_base.qwen_model()
        if candidate_path.exists() and generation_path.exists():
            candidates, generated = read_rows(candidate_path), read_rows(generation_path)
        else:
            candidates = contract.previous.legacy.parent.model_base.behavior_base.batch_behavior(
                model, device, rows, batch_size=18)
            generated = capture_base.generation(model, tokenizer, device, rows)
            write_rows(candidate_path, candidates)
            write_rows(generation_path, generated)
        ledger = behavior_ledger(rows, candidates, generated)
        save(OUT / "behavior/ledger.json", ledger)
        qualified = set(ledger["qualified_families"])
        observed = [row for row in rows if row["family"] in qualified]
        observed_index = index_rows(observed, candidates, generated)
        configure_capture_base()
        role_field = capture_base.capture_role_field(model, device, observed, observed_index)
        token_rows = [row for row in observed if row["unit"] in TOKEN_UNITS]
        token_field = capture_base.capture_all_token_field(model, device, token_rows)
        quantization = contract.previous.legacy.parent.model_base.scope.parent.previous.model_base().quantization_audit(model)
    finally:
        if model is not None:
            contract.previous.legacy.parent.model_base.scope.parent.previous.model_base().release_bf16(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    hashes = {
        "candidate": file_hash(candidate_path),
        "generation": file_hash(generation_path),
        "role_field": file_hash(ROLE_FIELD) if ROLE_FIELD.exists() else None,
        "role_index": file_hash(ROLE_INDEX) if ROLE_INDEX.exists() else None,
        "token_field": file_hash(TOKEN_FIELD) if TOKEN_FIELD.exists() else None,
        "token_index": file_hash(TOKEN_INDEX) if TOKEN_INDEX.exists() else None,
    }
    checks = {
        "behavior_complete": len(candidates) == len(generated) == len(rows),
        "qualified_logic_exact": all(ledger["qualification_audit"][family]["qualified"] ==
                                     (family in set(ledger["qualified_families"])) for family in contract.FAMILIES),
        "role_shape": (not role_field.get("ran")) or role_field["shape"][1:] == [38, 6, 2560],
        "role_family_match": (not role_field.get("ran")) or
                             set(role_field["families"]) == set(ledger["qualified_families"]),
        "token_full_coordinates": (not token_field.get("ran")) or token_field["shape"][-1] == 2560,
        "token_family_match": (not token_field.get("ran")) or
                              set(token_field["families"]) == set(ledger["qualified_families"]),
        "finite_behavior": bool(np.isfinite(ledger["candidate_accuracy"]) and
                                np.isfinite(ledger["generation_accuracy"])),
    }
    strict = (f"{len(ledger['qualified_families'])}/{len(contract.FAMILIES)} families passed the frozen "
              "overall, discovery, and fresh-confirmation dual behavior gate; internal fields are observational only.")
    result = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "status": "closed",
        "timestamp": datetime.now().astimezone().isoformat(),
        "placement": placement,
        "quantization": quantization,
        "behavior": ledger,
        "role_field": role_field,
        "all_token_field": token_field,
        "hashes": hashes,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "strict_conclusion": strict,
        "next_authorization": "Run the frozen full-coordinate basic-structure tournament in partition order.",
    }
    save(final, result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


if __name__ == "__main__":
    run()
