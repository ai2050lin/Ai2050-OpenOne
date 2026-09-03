#!/usr/bin/env python3
"""Replicate output-conditioned VJP textures across valid/broken interactions and held units."""
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
P2435 = RESULT / "phase2435_c33841_c34160_hypergraph_material_fourmodel_behavior/qwen4b"
OUT = RESULT / "phase2448_c38001_c38480_vjp_semantic_multiunit_replication"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE = 2448
CAMPAIGN = "C38001-C38480"
UNITS = (3, 4, 5)
FIELDS = ("gradient", "state_times_gradient")
INTERACTIONS = ("semantic_validity", "lexical_control")
VARIANTS = ("valid", "broken_a", "broken_b")
SHIFT = 791
DIM = 2560

sys.path.insert(0, str(TESTS))
import phase2389_c19121_c19440_crossmodel_autonomous_capability as capability  # noqa: E402
import phase2390_c19441_c19760_qwen_semantic_lexical_fullfield as field_utils  # noqa: E402


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def close(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > 1e-30 else 0.0


def selected_rows() -> list[dict]:
    rows = read_rows(P2435 / "index/trajectory_rows.jsonl")
    selected = [row for row in rows if int(row["unit"]) in UNITS and row["surface"] == "natural"]
    selected.sort(key=lambda row: row["case_id"])
    return selected


def capture(rows: list[dict]) -> dict:
    raw = OUT / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    paths = {
        "gradient": raw / "query_margin_vjp.float32.npy",
        "state_times_gradient": raw / "query_margin_state_times_vjp.float32.npy",
        "margin": raw / "live_margin.float32.npy",
    }
    shape = (len(rows), 38, DIM)
    gradient = np.lib.format.open_memmap(paths["gradient"], mode="r+" if paths["gradient"].exists() else "w+", dtype=np.float32, shape=shape)
    contribution = np.lib.format.open_memmap(paths["state_times_gradient"], mode="r+" if paths["state_times_gradient"].exists() else "w+", dtype=np.float32, shape=shape)
    margins = np.lib.format.open_memmap(paths["margin"], mode="r+" if paths["margin"].exists() else "w+", dtype=np.float32, shape=(len(rows),))
    progress_path = raw / "progress.json"
    completed = int(json.loads(progress_path.read_text(encoding="utf-8"))["completed"]) if progress_path.exists() else 0
    model = tokenizer = None
    captures: dict[int, torch.Tensor] = {}
    handles = []
    modules = []
    device = None
    if completed < len(rows):
        model, tokenizer, _ = capability.load_model("qwen4b")
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        modules = field_utils.modules(model)
        for qpoint, module in enumerate(modules):
            def hook(_module, _inputs, result, qpoint=qpoint):
                tensor = result[0] if isinstance(result, tuple) else result
                if qpoint == 0 and not tensor.requires_grad:
                    tensor.requires_grad_(True)
                tensor.retain_grad()
                captures[qpoint] = tensor
            handles.append(module.register_forward_hook(hook))
        device = model.get_input_embeddings().weight.device
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
            event_map = {event["event"]: int(event["token_index"]) for event in row["event_tokens"]}
            token_index = event_map["query_end"]
            for qpoint in range(38):
                state = captures[qpoint][0, token_index].detach().float().cpu().numpy()
                grad = captures[qpoint].grad[0, token_index].detach().float().cpu().numpy()
                gradient[index, qpoint] = grad
                contribution[index, qpoint] = state * grad
            margins[index] = float(margin.detach().float().cpu())
            gradient.flush(); contribution.flush(); margins.flush()
            save(progress_path, {"completed": index + 1, "shape": shape, "event": "query_end"})
            if (index + 1) % 32 == 0 or index + 1 == len(rows):
                print(f"[phase2448 VJP] {index + 1}/{len(rows)}", flush=True)
            del result, margin, ids, mask, positions
    finally:
        for handle in handles:
            handle.remove()
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gradient.flush(); contribution.flush(); margins.flush()
        close(gradient); close(contribution); close(margins)
    write_rows(OUT / "index/vjp_rows.jsonl", [{key: row[key] for key in ("case_id", "config_id", "family", "unit", "language", "surface", "direction", "variant", "query_role", "answer", "foil")} for row in rows])
    return {"gradient": str(paths["gradient"]), "state_times_gradient": str(paths["state_times_gradient"]), "margin": str(paths["margin"]),
            "shape": list(shape), "rows": len(rows), "event": "query_end", "storage": "float32 full coordinates",
            "bytes": sum(path.stat().st_size for path in paths.values()), "inference": "Qwen3-4B BF16 CUDA; parameters frozen; exact first-token margin VJP"}


def derangements(count: int, size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows = []
    while len(rows) < count:
        proposal = rng.permutation(size)
        if np.all(proposal != np.arange(size)):
            rows.append(proposal)
    return np.stack(rows)


def build_passports(rows: list[dict], collection: dict) -> tuple[np.ndarray, list[str]]:
    families = sorted({row["family"] for row in rows})
    languages = ("en", "zh")
    lookup = {(int(row["unit"]), row["family"], row["language"], int(row["direction"]), row["variant"], row["query_role"]): index for index, row in enumerate(rows)}
    passports = np.zeros((2, 2, len(UNITS), 2, 38, len(families), DIM), dtype=np.float32)
    for field_index, field_name in enumerate(FIELDS):
        field = np.load(collection[field_name], mmap_mode="r")
        for unit_index, unit in enumerate(UNITS):
            for language_index, language in enumerate(languages):
                for family_index, family in enumerate(families):
                    variants = {}
                    for variant in VARIANTS:
                        directions = []
                        for direction in (0, 1):
                            source = lookup[(unit, family, language, direction, variant, "source")]
                            target = lookup[(unit, family, language, direction, variant, "target")]
                            directions.append(np.asarray(field[target] - field[source], dtype=np.float32))
                        variants[variant] = np.mean(directions, axis=0)
                    passports[0, field_index, unit_index, language_index, :, family_index] = variants["valid"] - variants["broken_a"]
                    passports[1, field_index, unit_index, language_index, :, family_index] = variants["broken_a"] - variants["broken_b"]
        close(field)
    derived = OUT / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    np.save(derived / "semantic_lexical_vjp_passports.float32.npy", passports)
    return passports, families


def analyze(rows: list[dict], collection: dict) -> dict:
    passports, families = build_passports(rows, collection)
    permutations = derangements(64, len(families), 2448)
    metrics = np.zeros((2, 2, len(UNITS), 38, 6), dtype=np.float32)
    # coordinate, shift791, family-null mean, family-null q95, physical advantage, identity-q95 advantage
    for interaction in range(2):
        for field in range(2):
            for unit in range(len(UNITS)):
                for qpoint in range(38):
                    en = passports[interaction, field, unit, 0, qpoint]
                    zh = passports[interaction, field, unit, 1, qpoint]
                    coordinate = np.mean([cosine(en[index], zh[index]) for index in range(len(families))])
                    shifted = np.mean([cosine(en[index], np.roll(zh[index], SHIFT)) for index in range(len(families))])
                    nulls = np.asarray([np.mean([cosine(en[index], zh[permutation[index]]) for index in range(len(families))]) for permutation in permutations])
                    q95 = float(np.quantile(nulls, .95))
                    metrics[interaction, field, unit, qpoint] = (coordinate, shifted, float(np.mean(nulls)), q95, coordinate - shifted, coordinate - q95)
    np.save(OUT / "derived/crosslanguage_multinull_metrics.float32.npy", metrics)
    split_names = ("discovery_unit3", "confirmation_unit4", "fresh_unit5")
    summary: dict[str, Any] = {}
    selections = {}
    for interaction, interaction_name in enumerate(INTERACTIONS):
        summary[interaction_name] = {}
        selections[interaction_name] = {}
        for field, field_name in enumerate(FIELDS):
            discovery = metrics[interaction, field, 0]
            selected = int(np.argmax(discovery[:, 4] + discovery[:, 5]))
            selections[interaction_name][field_name] = selected
            summary[interaction_name][field_name] = {}
            for unit_index, split in enumerate(split_names):
                values = metrics[interaction, field, unit_index, selected]
                summary[interaction_name][field_name][split] = {
                    "qpoint_frozen_from_unit3": selected,
                    "language_coordinate": float(values[0]),
                    "shift791": float(values[1]),
                    "family_null_mean": float(values[2]),
                    "family_null_q95": float(values[3]),
                    "physical_advantage": float(values[4]),
                    "family_identity_q95_advantage": float(values[5]),
                }
    semantic = summary["semantic_validity"]["state_times_gradient"]
    held = [semantic["confirmation_unit4"], semantic["fresh_unit5"]]
    lexical = summary["lexical_control"]["state_times_gradient"]
    semantic_lockbox = all(row["physical_advantage"] > 0 and row["family_identity_q95_advantage"] > 0 for row in held)
    semantic_specific = semantic_lockbox and all(semantic[split]["language_coordinate"] > lexical[split]["language_coordinate"] for split in ("confirmation_unit4", "fresh_unit5"))
    return {"families": families, "permutations": 64, "selection": selections, "summary": summary,
            "semantic_attribution_held_unit_lockbox": semantic_lockbox,
            "semantic_exceeds_lexical_on_held_units": semantic_specific,
            "passports": str(OUT / "derived/semantic_lexical_vjp_passports.float32.npy"),
            "metrics": str(OUT / "derived/crosslanguage_multinull_metrics.float32.npy")}


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 自动续研——输出条件VJP的valid/broken双交互三unit冻结复制（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 扩展Phase2447单一valid试验，采集unit3/4/5、自然表述、八语言族、中英、双方向、valid/broken-A/broken-B、双查询角色共576条；每条在query-end保存embedding、36 block与final norm的全部2560坐标梯度及$H\odot g$。先作target-role−source-role，再构造语义有效性交互$I_{{sem}}=D_{{valid}}-D_{{brokenA}}$和词项控制$I_{{lex}}=D_{{brokenA}}-D_{{brokenB}}$。只用unit3选择层，冻结后检验unit4、unit5；零假设为+791物理坐标错配和64个无固定点family置乱。

$$g_{{q,e,i}}=\frac{{\partial(\ell_a-\ell_b)}}{{\partial H_{{q,e,i}}}},\quad A_{{q,e,i}}=H_{{q,e,i}}g_{{q,e,i}},$$
$$I_{{sem}}=(A_t-A_s)_{{valid}}-(A_t-A_s)_{{brokenA}},\quad I_{{lex}}=(A_t-A_s)_{{brokenA}}-(A_t-A_s)_{{brokenB}}.$$

**结果汇总。** 采集 `{json.dumps(result['collection'], ensure_ascii=False)}`；冻结选择 `{json.dumps(result['analysis']['selection'], ensure_ascii=False)}`；双交互全结果 `{json.dumps(result['analysis']['summary'], ensure_ascii=False)}`；裁决 `{json.dumps(result['adjudication'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2448_c38001_c38480_vjp_semantic_multiunit_replication.py`；576条全层全坐标梯度/归因、索引、双交互护照和64置乱指标位于同名结果目录。

**分析与理论进展。** 该Phase将“某个输出目标选择了共享坐标”与“语义有效性本身选择了共享坐标”分开。只有$I_{{sem}}$的输出条件归因在两个未参与选层的unit同时超过物理错配和family置乱95%分位，并优于$I_{{lex}}$，才称为语义特异性VJP候选；否则Phase2447只说明输出任务模板共享读取路径。

**问题硬伤与结论。** VJP仍是局部一阶导数，不能替代有限扰动或必要性；自然表述固定，尚未完成跨surface复制。family只有8个，64置乱只是有限经验零假设。即使通过也只能升级为“输出条件语义归因纹理候选”，不能称为条件齿轮或编码机制闭合。
"""
    with MEMO.open("a", encoding="utf-8", newline="") as stream:
        stream.write(text)


def main() -> None:
    final = OUT / "analysis/final.json"
    if final.exists():
        result = json.loads(final.read_text(encoding="utf-8"))
        append_memo(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    rows = selected_rows()
    collection = capture(rows)
    analysis = analyze(rows, collection)
    adjudication = {
        "semantic_attribution_held_unit_lockbox": analysis["semantic_attribution_held_unit_lockbox"],
        "semantic_exceeds_lexical_on_held_units": analysis["semantic_exceeds_lexical_on_held_units"],
        "output_conditioned_semantic_attribution_candidate": analysis["semantic_attribution_held_unit_lockbox"] and analysis["semantic_exceeds_lexical_on_held_units"],
        "conditional_coordinate_gear_proven": False,
    }
    checks = {
        "rows_576": collection["rows"] == 576,
        "shape": collection["shape"] == [576, 38, DIM],
        "eight_families": len(analysis["families"]) == 8,
        "three_units": len(UNITS) == 3,
        "sixty_four_derangements": analysis["permutations"] == 64,
        "all_files": all(Path(path).exists() for path in (collection["gradient"], collection["state_times_gradient"], collection["margin"], analysis["passports"], analysis["metrics"])),
        "finite": all(math.isfinite(value) for interaction in analysis["summary"].values() for field in interaction.values() for split in field.values() for value in split.values()),
        "claim_boundary": not adjudication["conditional_coordinate_gear_proven"],
    }
    result = {"phase": PHASE, "campaign": CAMPAIGN, "collection": collection, "analysis": analysis,
              "adjudication": adjudication, "checks": checks, "all_checks_passed": all(checks.values())}
    save(final, result)
    append_memo(result)
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
