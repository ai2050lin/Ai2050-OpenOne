#!/usr/bin/env python3
"""Model-relative output-conditioned semantic VJP portability worker and Qwen14B phase."""
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
P2453 = RESULT / "phase2453_c39761_c40080_portability_curvature_autoregressive_contract"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
MODEL_PHASE = {
    "qwen14b": (2454, "C40081-C40400", "phase2454_c40081_c40400_qwen14_output_conditioned_portability"),
    "glm4": (2455, "C40401-C40720", "phase2455_c40401_c40720_glm4_output_conditioned_portability"),
    "deepseek7b": (2456, "C40721-C41040", "phase2456_c40721_c41040_ds7b_output_conditioned_portability"),
}
FIELDS = ("gradient", "state_times_gradient")
INTERACTIONS = ("semantic_validity", "lexical_control")
VARIANTS = ("valid", "broken_a", "broken_b")
DIMENSION_RATIO_SHIFT = 791 / 2560

sys.path.insert(0, str(TESTS))
import phase2389_c19121_c19440_crossmodel_autonomous_capability as capability  # noqa: E402
import phase2390_c19441_c19760_qwen_semantic_lexical_fullfield as field_utils  # noqa: E402
import phase2412_c26481_c26800_frozen_crossmodel_operator_replication as cross_loader  # noqa: E402


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def close(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > 1e-30 else 0.0


def derangements(count: int, size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows = []
    while len(rows) < count:
        proposal = rng.permutation(size)
        if np.all(proposal != np.arange(size)):
            rows.append(proposal)
    return np.stack(rows)


def paths(key: str) -> dict[str, Path]:
    _, _, directory = MODEL_PHASE[key]
    out = RESULT / directory
    return {"out": out, "raw": out / "raw/output_conditioned_fields.float32.npy",
            "margin": out / "raw/live_margin.float32.npy", "progress": out / "raw/progress.json",
            "rows": out / "index/vjp_rows.jsonl", "passports": out / "derived/semantic_lexical_passports.float32.npy",
            "metrics": out / "derived/crosslanguage_multinull_metrics.float32.npy",
            "final": out / "analysis/final.json"}


def load_model(key: str):
    if key == "qwen14b":
        return cross_loader.load_for_capture(key)
    return capability.load_model(key)


def capture(key: str, rows: list[dict], contract: dict) -> dict:
    p = paths(key)
    p["raw"].parent.mkdir(parents=True, exist_ok=True)
    dimension = int(contract["hidden_size"])
    shape = (len(rows), 2, 2, dimension)  # row, frozen qpoint slot, field, coordinate
    fields = np.lib.format.open_memmap(p["raw"], mode="r+" if p["raw"].exists() else "w+", dtype=np.float32, shape=shape)
    margins = np.lib.format.open_memmap(p["margin"], mode="r+" if p["margin"].exists() else "w+", dtype=np.float32, shape=(len(rows),))
    completed = int(json.loads(p["progress"].read_text(encoding="utf-8"))["completed"]) if p["progress"].exists() else 0
    model = tokenizer = None
    qpoints = [int(contract["relative_qpoints"]["state_times_gradient"]), int(contract["relative_qpoints"]["gradient"])]
    captures: dict[int, torch.Tensor] = {}
    handles = []
    if completed < len(rows):
        model, tokenizer, label = load_model(key)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        modules = field_utils.modules(model)
        if len(modules) != int(contract["qpoints"]):
            raise RuntimeError((key, len(modules), contract["qpoints"]))

        def leaf_hook(_module, _inputs, result):
            tensor = result[0] if isinstance(result, tuple) else result
            if not tensor.requires_grad:
                tensor.requires_grad_(True)

        handles.append(modules[0].register_forward_hook(leaf_hook))
        for qpoint in qpoints:
            def field_hook(_module, _inputs, result, qpoint=qpoint):
                tensor = result[0] if isinstance(result, tuple) else result
                tensor.retain_grad(); captures[qpoint] = tensor
            handles.append(modules[qpoint].register_forward_hook(field_hook))
        device = model.get_input_embeddings().weight.device
    else:
        label = contract["precision"]
        device = None
    try:
        for index in range(completed, len(rows)):
            row = rows[index]
            ids = torch.tensor([row["prompt_ids"]], dtype=torch.long, device=device)
            mask = torch.ones_like(ids)
            positions = torch.arange(ids.shape[1], device=device)[None]
            captures.clear()
            with torch.enable_grad():
                result = model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, return_dict=True)
                target, foil = int(row["target_ids"][0]), int(row["foil_ids"][0])
                margin = result.logits[0, -1, target] - result.logits[0, -1, foil]
                margin.backward()
            token_index = {event["event"]: int(event["token_index"]) for event in row["event_tokens"]}["query_end"]
            for slot, qpoint in enumerate(qpoints):
                state = captures[qpoint][0, token_index].detach().float().cpu().numpy()
                gradient = captures[qpoint].grad[0, token_index].detach().float().cpu().numpy()
                fields[index, slot, 0] = gradient
                fields[index, slot, 1] = state * gradient
            margins[index] = float(margin.detach().float().cpu())
            fields.flush(); margins.flush()
            save(p["progress"], {"completed": index + 1, "shape": shape, "qpoints": qpoints})
            if (index + 1) % 16 == 0 or index + 1 == len(rows):
                print(f"[phase{MODEL_PHASE[key][0]} {key} VJP] {index + 1}/{len(rows)}", flush=True)
            del result, margin, ids, mask, positions
    finally:
        for handle in handles:
            handle.remove()
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        fields.flush(); margins.flush(); close(fields); close(margins)
    p["rows"].parent.mkdir(parents=True, exist_ok=True)
    p["rows"].write_text("".join(json.dumps({name: row[name] for name in ("case_id", "config_id", "family", "unit", "language", "surface", "direction", "variant", "query_role")}, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return {"field": str(p["raw"]), "margin": str(p["margin"]), "shape": list(shape), "qpoints": qpoints,
            "rows": len(rows), "dimension": dimension, "storage": "float32 all physical coordinates at two frozen relative qpoints",
            "bytes": p["raw"].stat().st_size + p["margin"].stat().st_size, "model_label": label,
            "precision": contract["precision"]}


def analyze(key: str, rows: list[dict], collection: dict, contract: dict) -> dict:
    p = paths(key)
    field = np.load(collection["field"], mmap_mode="r")
    families = sorted({row["family"] for row in rows})
    languages = ("en", "zh")
    units = (0, 4, 5)
    lookup = {(int(row["unit"]), row["family"], row["language"], row["variant"], row["query_role"]): index for index, row in enumerate(rows)}
    passports = np.zeros((2, 2, 3, 2, 2, 8, collection["dimension"]), dtype=np.float32)
    # interaction, field, unit, language, qpoint-slot, family, coordinate
    for unit_index, unit in enumerate(units):
        for language_index, language in enumerate(languages):
            for family_index, family in enumerate(families):
                variant_values = {}
                for variant in VARIANTS:
                    source = lookup[(unit, family, language, variant, "source")]
                    target = lookup[(unit, family, language, variant, "target")]
                    variant_values[variant] = np.asarray(field[target] - field[source], dtype=np.float32)
                passports[0, :, unit_index, language_index] = passports[0, :, unit_index, language_index]
                for slot in range(2):
                    for field_index in range(2):
                        passports[0, field_index, unit_index, language_index, slot, family_index] = variant_values["valid"][slot, field_index] - variant_values["broken_a"][slot, field_index]
                        passports[1, field_index, unit_index, language_index, slot, family_index] = variant_values["broken_a"][slot, field_index] - variant_values["broken_b"][slot, field_index]
    close(field)
    p["passports"].parent.mkdir(parents=True, exist_ok=True)
    np.save(p["passports"], passports)
    permutations = derangements(64, 8, 2454 + list(MODEL_PHASE).index(key))
    shift = max(1, round(collection["dimension"] * DIMENSION_RATIO_SHIFT))
    metrics = np.zeros((2, 2, 3, 6), dtype=np.float32)
    # interaction, field, unit, coord/shift/nullmean/nullq95/physicaladv/identityadv
    for interaction in range(2):
        for field_index in range(2):
            slot = 1 if field_index == 0 else 0
            for unit_index in range(3):
                en = passports[interaction, field_index, unit_index, 0, slot]
                zh = passports[interaction, field_index, unit_index, 1, slot]
                coordinate = np.mean([cosine(en[index], zh[index]) for index in range(8)])
                shifted = np.mean([cosine(en[index], np.roll(zh[index], shift)) for index in range(8)])
                nulls = np.asarray([np.mean([cosine(en[index], zh[permutation[index]]) for index in range(8)]) for permutation in permutations])
                q95 = float(np.quantile(nulls, .95))
                metrics[interaction, field_index, unit_index] = (coordinate, shifted, float(np.mean(nulls)), q95, coordinate - shifted, coordinate - q95)
    np.save(p["metrics"], metrics)
    qualified = [family for family in contract["qualified_families"] if family in families]
    qualified_indices = [families.index(family) for family in qualified]
    summary = {}
    split_names = ("discovery_unit0", "confirmation_unit4", "fresh_unit5")
    for interaction, interaction_name in enumerate(INTERACTIONS):
        summary[interaction_name] = {}
        for field_index, field_name in enumerate(FIELDS):
            summary[interaction_name][field_name] = {}
            slot = 1 if field_index == 0 else 0
            for unit_index, split in enumerate(split_names):
                values = metrics[interaction, field_index, unit_index]
                en = passports[interaction, field_index, unit_index, 0, slot]
                zh = passports[interaction, field_index, unit_index, 1, slot]
                qualified_cos = float(np.mean([cosine(en[index], zh[index]) for index in qualified_indices])) if qualified_indices else float("nan")
                summary[interaction_name][field_name][split] = {
                    "qpoint": collection["qpoints"][slot], "language_coordinate": float(values[0]),
                    "shift": float(values[1]), "family_null_mean": float(values[2]), "family_null_q95": float(values[3]),
                    "physical_advantage": float(values[4]), "family_identity_q95_advantage": float(values[5]),
                    "behavior_qualified_family_coordinate": qualified_cos, "behavior_qualified_family_count": len(qualified_indices)}
    semantic = summary["semantic_validity"]["state_times_gradient"]
    lexical = summary["lexical_control"]["state_times_gradient"]
    held = ("confirmation_unit4", "fresh_unit5")
    lockbox = all(semantic[split]["physical_advantage"] > 0 and semantic[split]["family_identity_q95_advantage"] > 0 for split in held)
    semantic_over_lexical = all(semantic[split]["language_coordinate"] > lexical[split]["language_coordinate"] for split in held)
    return {"families": families, "qualified_families": qualified, "coordinate_shift": shift, "derangements": 64,
            "summary": summary, "semantic_attribution_held_lockbox": lockbox,
            "semantic_attribution_exceeds_lexical_held": semantic_over_lexical,
            "passports": str(p["passports"]), "metrics": str(p["metrics"])}


def append_memo(key: str, result: dict) -> None:
    phase, campaign, _ = MODEL_PHASE[key]
    if f"## Phase {phase}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    model_title = {"qwen14b": "Qwen3-14B", "glm4": "GLM4-9B", "deepseek7b": "DS7B"}[key]
    text = rf"""

## Phase {phase}: {model_title}相对深度输出条件全坐标可移植性复现（{campaign}） [{stamp}]

**测试原理与测试用例。** 完全冻结Phase2453的288条八族中英canonical-direction0材料：unit0发现、unit4确认、unit5 fresh，三validity×双查询角色；目标模型不重新选层。采集Qwen4B q16/q18映射后的两个相对qpoint在query-end的全部物理坐标gradient与$H\odot g$。构造target−source后再作valid−brokenA语义interaction和brokenA−brokenB词项interaction；每模型内部比较同坐标中英、按维度比例错位和64个无固定点family置乱。

$$g_i=\partial(\ell_a-\ell_b)/\partial H_i,\qquad A_i=H_i g_i,$$
$$I_{{sem}}=(A_t-A_s)_{{valid}}-(A_t-A_s)_{{brokenA}}.$$

**结果汇总。** 采集 `{json.dumps(result['collection'], ensure_ascii=False)}`；模型内全坐标裁决 `{json.dumps(result['analysis']['summary'], ensure_ascii=False)}`；裁决 `{json.dumps(result['adjudication'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 主脚本`tests/glm5/phase2454_c40081_c40400_qwen14_output_conditioned_portability.py`；本模型冻结索引、原始全坐标场、双interaction护照、64置乱指标与final位于`tests/glm5/result/{MODEL_PHASE[key][2]}`。

**分析与理论进展。** 该Phase只问：输出条件语义归因的“模型内同坐标胜错位/错family”结构是否在冻结相对深度复现。跨模型宽度不同，不比较坐标编号、原始余弦或幅值。行为不合格family只报告输入响应，不升级为成功语义机制。

**问题硬伤与结论。** 精度为`{result['collection']['precision']}`；量化会改变局部梯度，不能与Qwen4B作幅度等价。单事件、单方向、共同候选协议仍在。即使通过也只能称模型内可移植候选，不能证明跨架构同构或条件齿轮闭合。
"""
    with MEMO.open("a", encoding="utf-8", newline="") as stream:
        stream.write(text)


def run(key: str) -> dict:
    phase, campaign, _ = MODEL_PHASE[key]
    p = paths(key)
    if p["final"].exists():
        result = json.loads(p["final"].read_text(encoding="utf-8")); append_memo(key, result); print(json.dumps(result, ensure_ascii=False, indent=2)); return result
    contract_all = json.loads((P2453 / "analysis/final.json").read_text(encoding="utf-8"))["model_contract"]
    contract = contract_all["models"][key]
    rows = read_rows(P2453 / f"contract/{key}_vjp_rows.jsonl")
    collection = capture(key, rows, contract)
    analysis = analyze(key, rows, collection, contract)
    adjudication = {"semantic_attribution_held_lockbox": analysis["semantic_attribution_held_lockbox"],
                    "semantic_attribution_exceeds_lexical_held": analysis["semantic_attribution_exceeds_lexical_held"],
                    "behavior_gated_portability_candidate": analysis["semantic_attribution_held_lockbox"] and analysis["semantic_attribution_exceeds_lexical_held"] and len(analysis["qualified_families"]) >= 2,
                    "cross_architecture_coordinate_isomorphism_proven": False,
                    "language_encoding_mechanism_closed": False}
    values = [value for interaction in analysis["summary"].values() for field in interaction.values() for split in field.values() for value in split.values() if isinstance(value, (int, float))]
    checks = {"phase": phase == result_phase(key), "rows_288": collection["rows"] == 288,
              "all_coordinates": collection["shape"] == [288, 2, 2, contract["hidden_size"]],
              "two_frozen_qpoints": collection["qpoints"] == [contract["relative_qpoints"]["state_times_gradient"], contract["relative_qpoints"]["gradient"]],
              "sixty_four_derangements": analysis["derangements"] == 64,
              "files": all(Path(path).exists() for path in (collection["field"], collection["margin"], analysis["passports"], analysis["metrics"])),
              "finite": all(math.isfinite(float(value)) for value in values),
              "precision_labeled": bool(collection["precision"]), "claim_boundary": not adjudication["language_encoding_mechanism_closed"]}
    result = {"phase": phase, "campaign": campaign, "model": key, "collection": collection,
              "analysis": analysis, "adjudication": adjudication, "checks": checks, "all_checks_passed": all(checks.values())}
    save(p["final"], result); append_memo(key, result)
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def result_phase(key: str) -> int:
    return MODEL_PHASE[key][0]


if __name__ == "__main__":
    run("qwen14b")
