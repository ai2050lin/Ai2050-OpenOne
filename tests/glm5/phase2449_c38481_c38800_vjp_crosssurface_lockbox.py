#!/usr/bin/env python3
"""Cross-surface lockbox for the frozen output-conditioned semantic VJP candidate."""
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
P2448 = RESULT / "phase2448_c38001_c38480_vjp_semantic_multiunit_replication"
OUT = RESULT / "phase2449_c38481_c38800_vjp_crosssurface_lockbox"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE = 2449
CAMPAIGN = "C38481-C38800"
UNITS = (4, 5)
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
    selected = [row for row in rows if int(row["unit"]) in UNITS and row["surface"] == "canonical"]
    selected.sort(key=lambda row: row["case_id"])
    return selected


def capture(rows: list[dict]) -> dict:
    raw = OUT / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    paths = {"gradient": raw / "query_margin_vjp.float32.npy",
             "state_times_gradient": raw / "query_margin_state_times_vjp.float32.npy",
             "margin": raw / "live_margin.float32.npy"}
    shape = (len(rows), 38, DIM)
    fields = [np.lib.format.open_memmap(paths[name], mode="r+" if paths[name].exists() else "w+", dtype=np.float32, shape=shape) for name in FIELDS]
    margins = np.lib.format.open_memmap(paths["margin"], mode="r+" if paths["margin"].exists() else "w+", dtype=np.float32, shape=(len(rows),))
    progress_path = raw / "progress.json"
    completed = int(json.loads(progress_path.read_text(encoding="utf-8"))["completed"]) if progress_path.exists() else 0
    model = tokenizer = None
    modules = []
    captures: dict[int, torch.Tensor] = {}
    handles = []
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
            token_index = {event["event"]: int(event["token_index"]) for event in row["event_tokens"]}["query_end"]
            for qpoint in range(38):
                state = captures[qpoint][0, token_index].detach().float().cpu().numpy()
                grad = captures[qpoint].grad[0, token_index].detach().float().cpu().numpy()
                fields[0][index, qpoint] = grad
                fields[1][index, qpoint] = state * grad
            margins[index] = float(margin.detach().float().cpu())
            for field in fields:
                field.flush()
            margins.flush()
            save(progress_path, {"completed": index + 1, "shape": shape, "surface": "canonical", "event": "query_end"})
            if (index + 1) % 32 == 0 or index + 1 == len(rows):
                print(f"[phase2449 VJP] {index + 1}/{len(rows)}", flush=True)
            del result, margin, ids, mask, positions
    finally:
        for handle in handles:
            handle.remove()
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        for field in fields:
            field.flush(); close(field)
        margins.flush(); close(margins)
    index_path = OUT / "index/vjp_rows.jsonl"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text("".join(json.dumps({key: row[key] for key in ("case_id", "config_id", "family", "unit", "language", "surface", "direction", "variant", "query_role")}, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return {"gradient": str(paths["gradient"]), "state_times_gradient": str(paths["state_times_gradient"]), "margin": str(paths["margin"]),
            "shape": list(shape), "rows": len(rows), "surface": "canonical", "event": "query_end", "storage": "float32 full coordinates",
            "bytes": sum(path.stat().st_size for path in paths.values()), "inference": "Qwen3-4B BF16 CUDA; exact first-token margin VJP"}


def derangements(count: int, size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows = []
    while len(rows) < count:
        proposal = rng.permutation(size)
        if np.all(proposal != np.arange(size)):
            rows.append(proposal)
    return np.stack(rows)


def build_passports(rows: list[dict], collection: dict, families: list[str]) -> np.ndarray:
    lookup = {(int(row["unit"]), row["family"], row["language"], int(row["direction"]), row["variant"], row["query_role"]): index for index, row in enumerate(rows)}
    passports = np.zeros((2, 2, 2, 2, 38, 8, DIM), dtype=np.float32)
    for field_index, field_name in enumerate(FIELDS):
        field = np.load(collection[field_name], mmap_mode="r")
        for unit_index, unit in enumerate(UNITS):
            for language_index, language in enumerate(("en", "zh")):
                for family_index, family in enumerate(families):
                    variant_values = {}
                    for variant in VARIANTS:
                        directional = []
                        for direction in (0, 1):
                            source = lookup[(unit, family, language, direction, variant, "source")]
                            target = lookup[(unit, family, language, direction, variant, "target")]
                            directional.append(np.asarray(field[target] - field[source], dtype=np.float32))
                        variant_values[variant] = np.mean(directional, axis=0)
                    passports[0, field_index, unit_index, language_index, :, family_index] = variant_values["valid"] - variant_values["broken_a"]
                    passports[1, field_index, unit_index, language_index, :, family_index] = variant_values["broken_a"] - variant_values["broken_b"]
        close(field)
    path = OUT / "derived/canonical_semantic_lexical_vjp_passports.float32.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, passports)
    return passports


def null_stats(left: np.ndarray, right: np.ndarray, permutations: np.ndarray) -> tuple[float, float, float, float]:
    coordinate = float(np.mean([cosine(left[index], right[index]) for index in range(8)]))
    shift = float(np.mean([cosine(left[index], np.roll(right[index], SHIFT)) for index in range(8)]))
    nulls = np.asarray([np.mean([cosine(left[index], right[permutation[index]]) for index in range(8)]) for permutation in permutations])
    return coordinate, shift, float(np.mean(nulls)), float(np.quantile(nulls, .95))


def analyze(rows: list[dict], collection: dict) -> dict:
    previous = json.loads((P2448 / "analysis/final.json").read_text(encoding="utf-8"))
    families = previous["analysis"]["families"]
    selections = previous["analysis"]["selection"]
    natural = np.load(previous["analysis"]["passports"], mmap_mode="r")
    canonical = build_passports(rows, collection, families)
    permutations = derangements(64, 8, 2449)
    summary: dict[str, Any] = {}
    for interaction, interaction_name in enumerate(INTERACTIONS):
        summary[interaction_name] = {}
        for field, field_name in enumerate(FIELDS):
            qpoint = int(selections[interaction_name][field_name])
            summary[interaction_name][field_name] = {}
            for unit_index, unit in enumerate(UNITS):
                can_en = canonical[interaction, field, unit_index, 0, qpoint]
                can_zh = canonical[interaction, field, unit_index, 1, qpoint]
                crosslang = null_stats(can_en, can_zh, permutations)
                surface_rows = []
                for language in range(2):
                    nat = natural[interaction, field, unit_index + 1, language, qpoint]
                    can = canonical[interaction, field, unit_index, language, qpoint]
                    surface_rows.append(null_stats(nat, can, permutations))
                surface = np.mean(surface_rows, axis=0)
                summary[interaction_name][field_name][f"unit{unit}"] = {
                    "qpoint_frozen_from_natural_unit3": qpoint,
                    "canonical_crosslanguage_coordinate": crosslang[0],
                    "canonical_crosslanguage_shift791": crosslang[1],
                    "canonical_crosslanguage_family_null_q95": crosslang[3],
                    "surface_coordinate": float(surface[0]),
                    "surface_shift791": float(surface[1]),
                    "surface_family_null_q95": float(surface[3]),
                    "crosslanguage_physical_advantage": crosslang[0] - crosslang[1],
                    "crosslanguage_identity_q95_advantage": crosslang[0] - crosslang[3],
                    "surface_physical_advantage": float(surface[0] - surface[1]),
                    "surface_identity_q95_advantage": float(surface[0] - surface[3]),
                }
    close(natural)
    semantic = summary["semantic_validity"]["state_times_gradient"]
    lexical = summary["lexical_control"]["state_times_gradient"]
    lockbox = all(semantic[f"unit{unit}"][key] > 0 for unit in UNITS for key in ("crosslanguage_physical_advantage", "crosslanguage_identity_q95_advantage", "surface_physical_advantage", "surface_identity_q95_advantage"))
    semantic_exceeds_lexical = all(semantic[f"unit{unit}"]["surface_coordinate"] > lexical[f"unit{unit}"]["surface_coordinate"] for unit in UNITS)
    return {"families": families, "frozen_selection": selections, "permutations": 64, "summary": summary,
            "semantic_attribution_crosssurface_lockbox": lockbox,
            "semantic_surface_exceeds_lexical": semantic_exceeds_lexical,
            "canonical_passports": str(OUT / "derived/canonical_semantic_lexical_vjp_passports.float32.npy")}


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 自动续研——输出条件语义归因候选的跨表述锁箱（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** Phase2448仅在natural表述建立三unit候选。本Phase新增unit4/5 canonical表述、八族中英双方向、三variant双角色共384条query-end全层全2560坐标VJP与$H\odot g$。层号完全冻结自natural unit3；同时检验canonical内部中英同坐标、以及同unit同语言natural↔canonical表述复用，均对比+791坐标错配和64个family置乱95%分位。

$$R_{{surf}}=\frac1{{2|F|}}\sum_{{l,f}}\cos\left(I^{{natural}}_{{l,f}},I^{{canonical}}_{{l,f}}\right),\qquad
\Delta_{{null}}=R_{{surf}}-Q_{{0.95}}\!\left[R_{{surf}}^{{\pi(f)}}\right].$$

**结果汇总。** 采集 `{json.dumps(result['collection'], ensure_ascii=False)}`；冻结层 `{json.dumps(result['analysis']['frozen_selection'], ensure_ascii=False)}`；跨语言与跨表述 `{json.dumps(result['analysis']['summary'], ensure_ascii=False)}`；裁决 `{json.dumps(result['adjudication'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2449_c38481_c38800_vjp_crosssurface_lockbox.py`；384条原场、canonical双交互护照、索引和final位于同名结果目录；natural基线来自Phase2448，未重采样。

**分析与理论进展。** 该锁箱直接回答Phase2448阳性是否依赖natural提示模板。如果语义归因在冻结层跨表述仍保持相同物理坐标和family身份，而词项控制较弱，则证据从“跨语言复用”推进到“跨语言且跨表述的输出条件语义纹理复用”。

**问题硬伤与结论。** canonical/natural仍共享任务框架、候选格式和实体，不能排除更高层任务结构；VJP仍是局部一阶读出。通过只允许命名为跨表述输出条件归因候选，不等于有限扰动因果齿轮。
"""
    with MEMO.open("a", encoding="utf-8", newline="") as stream:
        stream.write(text)


def main() -> None:
    final = OUT / "analysis/final.json"
    if final.exists():
        result = json.loads(final.read_text(encoding="utf-8")); append_memo(result); print(json.dumps(result, ensure_ascii=False, indent=2)); return
    rows = selected_rows()
    collection = capture(rows)
    analysis = analyze(rows, collection)
    adjudication = {"semantic_attribution_crosssurface_lockbox": analysis["semantic_attribution_crosssurface_lockbox"],
                    "semantic_surface_exceeds_lexical": analysis["semantic_surface_exceeds_lexical"],
                    "crosssurface_output_conditioned_semantic_candidate": analysis["semantic_attribution_crosssurface_lockbox"] and analysis["semantic_surface_exceeds_lexical"],
                    "conditional_coordinate_gear_proven": False}
    checks = {"rows_384": collection["rows"] == 384, "shape": collection["shape"] == [384, 38, DIM],
              "eight_families": len(analysis["families"]) == 8, "two_held_units": len(UNITS) == 2,
              "sixty_four_derangements": analysis["permutations"] == 64,
              "all_files": all(Path(path).exists() for path in (collection["gradient"], collection["state_times_gradient"], collection["margin"], analysis["canonical_passports"])),
              "finite": all(math.isfinite(value) for interaction in analysis["summary"].values() for field in interaction.values() for unit in field.values() for value in unit.values()),
              "claim_boundary": not adjudication["conditional_coordinate_gear_proven"]}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "collection": collection, "analysis": analysis,
              "adjudication": adjudication, "checks": checks, "all_checks_passed": all(checks.values())}
    save(final, result); append_memo(result)
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
