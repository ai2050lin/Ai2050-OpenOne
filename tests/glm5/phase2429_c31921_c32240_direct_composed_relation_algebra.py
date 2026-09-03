#!/usr/bin/env python3
"""Direct-versus-composed relation algebra on four transitive and four negative-control families."""
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
OUT = TESTS / "result/phase2429_c31921_c32240_direct_composed_relation_algebra"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE = 2429
CAMPAIGN = "C31921-C32240"
VARIANTS = ("chain", "edge1", "edge2", "null", "direct")
TRANSITIVE = ("spatial", "temporal", "comparison", "taxonomy")
NONTRANSITIVE = ("preference", "ownership", "causal", "role_binding")
EVENTS = ("query_end", "answer_boundary")
SPLITS = ("confirmation", "fresh_unit", "template", "joint", "language", "family")
COMPONENTS = ("total", "attention", "mlp")

sys.path.insert(0, str(TESTS))
import phase2389_c19121_c19440_crossmodel_autonomous_capability as capability  # noqa: E402
import phase2397_c21681_c22000_operation_behavior_token_calibration as behavior  # noqa: E402
import phase2405_c24241_c24560_deconfounded_operation_contract as contract  # noqa: E402
import phase2411_c26161_c26480_crosslayer_composition_output_bridge as geometry  # noqa: E402
import phase2412_c26481_c26800_frozen_crossmodel_operator_replication as capture_loader  # noqa: E402
import phase2424_c30321_c30640_semantic_validity_multievent_fullfield as capture  # noqa: E402
import phase2425_c30641_c30960_semantic_specific_interaction_atlas as atlas  # noqa: E402


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def close(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def reduced_rows(rows: list[dict]) -> list[dict]:
    result = []
    for source_index, row in enumerate(rows):
        mapping = {event["event"]: int(event["token_index"]) for event in row["event_tokens"]}
        if not all(event in mapping for event in EVENTS):
            raise RuntimeError((row["case_id"], sorted(mapping)))
        item = {key: value for key, value in row.items() if key != "event_tokens"}
        item.update({"source_index": source_index, "event_names": list(EVENTS),
                     "event_token_indices": [mapping[event] for event in EVENTS]})
        result.append(item)
    return result


def partition(unit: int, surface: str) -> str:
    controlled = surface in ("canonical", "paraphrase")
    if controlled:
        return "discovery" if unit < 6 else "fresh_unit_lockbox"
    return "template_lockbox" if unit < 6 else "joint_lockbox"


def query(language: str, source: str) -> str:
    if language == "en":
        return f"Starting from {source}, use all connected records of the stated relation. Which candidate is the final reachable entity?"
    return f"从{source}开始，使用该关系的全部相连记录，最终可到达哪个候选实体？"


def distractors(language: str, unit: int, excluded: set[str]) -> list[str]:
    values = []
    for offset in range(1, 8):
        values.extend(contract.triples(language, (unit + offset) % 8))
    unique = []
    for value in values:
        if value not in excluded and value not in unique:
            unique.append(value)
    if len(unique) < 4:
        raise RuntimeError((language, unit, unique))
    return unique[:4]


def compile_source() -> list[dict]:
    rows = []
    families = list(contract.FAMILIES)
    for fi, family in enumerate(families):
        for unit in range(8):
            for language in contract.LANGUAGES:
                a, b, c = contract.triples(language, unit)
                for si, surface in enumerate(contract.SURFACES):
                    for direction in (0, 1):
                        source, middle, endpoint = (a, b, c) if direction == 0 else (c, b, a)
                        d1, d2, d3, d4 = distractors(language, unit, {source, middle, endpoint})
                        pairs = {
                            "chain": ((source, middle), (middle, endpoint)),
                            "edge1": ((source, middle), (d1, d2)),
                            "edge2": ((d1, d2), (middle, endpoint)),
                            "null": ((d1, d2), (d3, d4)),
                            "direct": ((source, endpoint), (d1, d2)),
                        }
                        order = (fi + unit + si + direction) % 2
                        candidates = [source, endpoint] if order == 0 else [endpoint, source]
                        config_id = f"alg-{family}-u{unit}-{language}-{surface}-d{direction}"
                        for variant in VARIANTS:
                            facts = [contract.render_fact(family, language, surface, left, right) for left, right in pairs[variant]]
                            prompt, events = contract.prior.prompt_with_events(language, facts, query(language, source), candidates)
                            rows.append({
                                "case_id": f"{config_id}-{variant}", "config_id": config_id,
                                "task": "direct_composed_relation", "family": family,
                                "family_type": "transitive" if family in TRANSITIVE else "nontransitive_control",
                                "unit": unit, "language": language, "surface": surface,
                                "surface_class": "controlled" if surface in ("canonical", "paraphrase") else "naturalized",
                                "direction": direction, "variant": variant, "candidate_order": order,
                                "target_candidate_slot": candidates.index(endpoint), "partition": partition(unit, surface),
                                "source": source, "middle": middle, "endpoint": endpoint,
                                "facts": [fact for fact, _ in facts], "query": query(language, source),
                                "candidates": candidates, "answer": endpoint, "foil": source,
                                "prompt": prompt, "events": events,
                            })
    return rows


def material_audit(rows: list[dict]) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["config_id"]].append(row)
    exact = sum(int(len(group) == 5 and {row["variant"] for row in group} == set(VARIANTS) and
                    all(row["candidates"] == group[0]["candidates"] and row["query"] == group[0]["query"] for row in group))
                for group in groups.values())
    return {"rows": len(rows), "configurations": len(groups), "variants": dict(Counter(row["variant"] for row in rows)),
            "families": dict(Counter(row["family"] for row in rows)), "family_types": dict(Counter(row["family_type"] for row in rows)),
            "languages": dict(Counter(row["language"] for row in rows)), "surfaces": dict(Counter(row["surface"] for row in rows)),
            "exact_five_variant_blocks": exact, "unique_cases": len({row["case_id"] for row in rows}) == len(rows)}


def behavior_summary(rows: list[dict], scores: list[dict], metric: str) -> dict:
    joined = [{**row, **score} for row, score in zip(rows, scores)]
    def one(items: list[dict]) -> dict:
        values = np.asarray([item[metric] for item in items], dtype=np.float64)
        if metric.endswith("margin"):
            return {"rows": len(items), "target_over_foil": float(np.mean(values > 0)), "mean_margin": float(np.mean(values))}
        return {"rows": len(items), "exact": float(np.mean(values)),
                "target_present": float(np.mean([item.get("target_present", False) for item in items]))}
    result = {}
    for family_type in ("transitive", "nontransitive_control"):
        chosen = [row for row in joined if row["family_type"] == family_type]
        result[family_type] = {variant: one([row for row in chosen if row["variant"] == variant])
                               for variant in sorted({row["variant"] for row in chosen})}
        if set(VARIANTS).issubset(result[family_type]):
            key = "target_over_foil" if metric.endswith("margin") else "exact"
            result[family_type]["chain_minus_null"] = result[family_type]["chain"][key] - result[family_type]["null"][key]
            result[family_type]["direct_minus_null"] = result[family_type]["direct"][key] - result[family_type]["null"][key]
    return result


def run_model(key: str, source: list[dict], collect: bool) -> tuple[dict, dict | None]:
    final = OUT / key / "analysis/behavior_final.json"
    collection_file = OUT / key / "analysis/collection.json"
    if final.exists() and (not collect or collection_file.exists()):
        payload = json.loads(final.read_text(encoding="utf-8"))
        payload["checks"].pop("autonomous_512", None)
        generated_path = OUT / key / "behavior/autonomous_lockbox.jsonl"
        if generated_path.exists():
            payload["checks"]["autonomous_256"] = len(read_rows(generated_path)) == 256
        payload["all_checks_passed"] = all(payload["checks"].values())
        save(final, payload)
        return payload, (json.loads(collection_file.read_text(encoding="utf-8")) if collect else None)
    model, tokenizer, label = (capability.load_model(key) if key == "qwen4b" else capture_loader.load_for_capture(key))
    behavior.OUT = OUT
    capture.OUT = OUT / key
    capture.EVENTS = EVENTS
    try:
        index_path = OUT / key / "index/direct_composed_rows.jsonl"
        if index_path.exists():
            rows = read_rows(index_path)
            calibration = json.loads((OUT / key / "analysis/token_calibration.json").read_text(encoding="utf-8"))
        else:
            compiled, calibration = behavior.compile_rows(tokenizer, source)
            rows = reduced_rows(compiled) if collect else compiled
            write_rows(index_path, rows); save(OUT / key / "analysis/token_calibration.json", calibration)
        teacher, _ = behavior.score_rows(key, model, rows, 4)
        teacher_summary = behavior_summary(rows, teacher, "mean_logprob_margin")
        lockbox = [row for row in rows if row["family_type"] == "transitive" and int(row["unit"]) >= 6 and row["variant"] in ("chain", "direct")]
        generated, _ = behavior.generate_lockbox(key, model, tokenizer, lockbox, 4 if key == "qwen4b" else 2)
        autonomous = behavior_summary(lockbox, generated, "exact")
        collection = capture.collect_events(model, rows, 4) if collect else None
    finally:
        del model, tokenizer; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    checks = {"compiled_5120": calibration["rows"] == 5120, "teacher_5120": len(teacher) == 5120,
              "autonomous_256": len(generated) == 256, "monotonic": calibration["event_monotonic_rate"] == 1.0,
              "collection_full": (not collect or collection["state"]["shape"] == [5120, 38, 2, 2560])}
    result = {"model": key, "label": label,
              "precision": "BF16 weights" if key == "qwen4b" else "NF4 weights / BF16 compute; behavior only",
              "calibration": calibration, "teacher": teacher_summary, "autonomous": autonomous,
              "checks": checks, "all_checks_passed": all(checks.values())}
    save(final, result)
    if collection is not None:
        save(collection_file, collection)
    return result, collection


def config_index(rows: list[dict]) -> tuple[list[dict], dict[str, np.ndarray]]:
    configs = sorted({row["config_id"] for row in rows})
    mapping = {(row["config_id"], row["variant"]): i for i, row in enumerate(rows)}
    meta, indices = [], {variant: [] for variant in VARIANTS}
    for config in configs:
        row = rows[mapping[(config, "chain")]]
        meta.append({key: row[key] for key in ("config_id", "family", "family_type", "unit", "language", "surface", "surface_class", "direction", "partition")})
        for variant in VARIANTS:
            indices[variant].append(mapping[(config, variant)])
    return meta, {key: np.asarray(value, dtype=np.int64) for key, value in indices.items()}


def interaction(field: np.ndarray, layer: int, event: int, index: dict) -> tuple[np.ndarray, np.ndarray]:
    value = {variant: np.asarray(field[index[variant], layer, event], dtype=np.float32) for variant in VARIANTS}
    composed = value["chain"] - value["edge1"] - value["edge2"] + value["null"]
    direct = value["direct"] - value["null"]
    return composed, direct


def subset_meta(meta: list[dict], chosen: np.ndarray) -> list[dict]:
    return [meta[int(i)] for i in chosen]


def analyze(rows: list[dict], collection: dict) -> dict:
    meta, index = config_index(rows)
    trans = np.flatnonzero([row["family_type"] == "transitive" for row in meta])
    nontrans = np.flatnonzero([row["family_type"] == "nontransitive_control" for row in meta])
    tmeta = subset_meta(meta, trans)
    families = np.asarray([row["family"] for row in tmeta], dtype=object)
    specs = atlas.split_specs(tmeta, families)
    state = np.load(collection["state"]["path"], mmap_mode="r")
    attention = np.load(collection["attention"]["path"], mmap_mode="r")
    mlp = np.load(collection["mlp"]["path"], mmap_mode="r")
    layers, events, dim = attention.shape[1:]
    (OUT / "derived").mkdir(parents=True, exist_ok=True)
    metrics = np.zeros((3, len(SPLITS), 5, layers, events), dtype=np.float32)
    energy = np.zeros((2, 3, layers, events), dtype=np.float32)
    geometry_metrics = np.zeros((3, 2, layers, events), dtype=np.float32)
    slopes = np.lib.format.open_memmap(OUT / "derived/direct_to_composed_slope.float32.npy", mode="w+", dtype=np.float32,
                                       shape=(3, layers, events, dim))
    full_train = specs["fresh_unit"][0]
    for layer in range(layers):
        for event in range(events):
            h_comp, h_direct = interaction(state, layer, event, index)
            a_comp, a_direct = interaction(attention, layer, event, index)
            m_comp, m_direct = interaction(mlp, layer, event, index)
            for ci, (comp, direct) in enumerate(((a_comp + m_comp, a_direct + m_direct), (a_comp, a_direct), (m_comp, m_direct))):
                energy[0, ci, layer, event] = float(np.mean(comp[trans] * comp[trans]))
                energy[1, ci, layer, event] = float(np.mean(comp[nontrans] * comp[nontrans]))
                fitted = atlas.fit(full_train, families, h_direct[trans], comp[trans])
                slopes[ci, layer, event] = fitted["slope"]
                family_names = sorted(set(families))
                comp_pass = np.stack([comp[trans][full_train[families[full_train] == family]].mean(0) for family in family_names])
                direct_pass = np.stack([direct[trans][full_train[families[full_train] == family]].mean(0) for family in family_names])
                geometry_metrics[ci, 0, layer, event] = geometry.correlation(geometry.geometry_vector(comp_pass), geometry.geometry_vector(direct_pass))
                geometry_metrics[ci, 1, layer, event] = float(np.mean([geometry.cosine(comp_pass[i], direct_pass[i]) for i in range(len(family_names))]))
                for si, split in enumerate(SPLITS):
                    train, test, conditioned = specs[split]
                    if split == "family":
                        values = atlas.family_holdout(tmeta, families, train, test, h_direct[trans], comp[trans])
                    else:
                        fit = atlas.fit(train, families, h_direct[trans], comp[trans], family_conditioned=conditioned)
                        global_p, family_p, state_p = atlas.predict(test, families, h_direct[trans], fit)
                        _, _, mismatch_p = atlas.predict(test, families, h_direct[trans], fit, mismatch=True)
                        values = atlas.gains(comp[trans][test], global_p, family_p, state_p, mismatch_p)
                    metrics[ci, si, :4, layer, event] = values
                    metrics[ci, si, 4, layer, event] = float(np.mean(comp[trans][test] ** 2))
        print(f"[phase2429 analysis] layer {layer + 1}/{layers}", flush=True)
    slopes.flush(); close(slopes)
    np.save(OUT / "derived/direct_composed_metrics.float32.npy", metrics)
    np.save(OUT / "derived/composition_energy.float32.npy", energy)
    np.save(OUT / "derived/direct_composed_geometry.float32.npy", geometry_metrics)
    summary = {component: {split: {"family_gain": float(metrics[ci, si, 1].mean()),
                                   "state_gain": float(metrics[ci, si, 2].mean()),
                                   "mismatch_gain": float(metrics[ci, si, 3].mean()),
                                   "physical_advantage": float((metrics[ci, si, 2] - metrics[ci, si, 3]).mean()),
                                   "composition_energy": float(metrics[ci, si, 4].mean())}
                           for si, split in enumerate(SPLITS)} for ci, component in enumerate(COMPONENTS)}
    comparison = {component: {"transitive_over_nontransitive_energy_ratio": float(energy[0, ci].mean() / max(energy[1, ci].mean(), 1e-30)),
                              "direct_composed_relation_geometry": float(geometry_metrics[ci, 0].mean()),
                              "direct_composed_coordinate_cosine": float(geometry_metrics[ci, 1].mean())}
                  for ci, component in enumerate(COMPONENTS)}
    files = {"slopes": str(OUT / "derived/direct_to_composed_slope.float32.npy"),
             "metrics": str(OUT / "derived/direct_composed_metrics.float32.npy"),
             "energy": str(OUT / "derived/composition_energy.float32.npy"),
             "geometry": str(OUT / "derived/direct_composed_geometry.float32.npy")}
    for value in (state, attention, mlp):
        close(value)
    return {"configurations": len(meta), "transitive_configurations": len(trans), "nontransitive_configurations": len(nontrans),
            "summary": summary, "comparison": comparison, "files": files}


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 直接关系—两边组合的全坐标代数与非传递负对照（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 对八族×八unit×中英×四表面×双方向1024配置构造五个严格同查询/同候选/双记录面板：完整链、仅边1、仅边2、全空连接、直接source→endpoint，共5120条。spatial/temporal/comparison/taxonomy为可传递目标族；preference/ownership/causal/role_binding为非传递负对照。Qwen4B与Qwen14B依次完成教师强制和256条自主锁箱；Qwen4B在query/answer两事件采集H/A/M全部2560坐标。用包含—排除提取双边联合项，与直接关系相对空连接的场比较。

$$I_{{comp}}=X_{{chain}}-X_{{edge1}}-X_{{edge2}}+X_{{null}},\qquad I_{{direct}}=X_{{direct}}-X_{{null}},$$

$$\hat I_{{comp,i}}^U=\bar I_{{comp,f,i}}^U+\beta_i(I_{{direct,i}}^H-\bar I_{{direct,f,i}}^H).$$

**结果汇总。** 材料 `{json.dumps(result['material'], ensure_ascii=False)}`；Qwen4B行为 `{json.dumps(result['models']['qwen4b'], ensure_ascii=False)}`；Qwen14B行为 `{json.dumps(result['models']['qwen14b'], ensure_ascii=False)}`；全坐标代数 `{json.dumps(result['analysis'], ensure_ascii=False)}`；裁决 `{json.dumps(result['adjudication'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2429_c31921_c32240_direct_composed_relation_algebra.py`；5120条材料、两模型token/行为、Qwen4B两事件H/A/M原始场、逐组件全坐标斜率和层事件指标位于`tests/glm5/result/phase2429_c31921_c32240_direct_composed_relation_algebra`。未修改其他Markdown。

**分析与理论进展。** 包含—排除不把某条文本差分搬运成机制，而是用1024个配置逐坐标积累“两个边共同连通才出现”的联合图谱。若可传递族行为合格，$I_{{comp}}$能从$I_{{direct}}$跨内容/模板/语言/家族预测，且非传递族联合能量较低，才支持有限参数复用关系组合规律。关系Gram一致而坐标映射失败仍只能叫功能几何，不叫代数闭合。

**问题硬伤与结论。** 自然语言中的causal等关系有时也可传递，负对照不是逻辑绝对；comparison与taxonomy的传递性依赖读法。五面板虽同双记录但具体token长度不同。包含—排除会放大量化误差，且联合项也可能反映连通性/检索难度。Qwen14B为NF4权重/BF16计算且只作行为资格。原始Qwen4B场留到Phase2431统一发布/清理。
"""
    with MEMO.open("a", encoding="utf-8", newline="") as stream:
        stream.write(text)


def main() -> None:
    final = OUT / "analysis/final.json"
    if final.exists():
        result = json.loads(final.read_text(encoding="utf-8")); append_memo(result)
        print(json.dumps(result, ensure_ascii=False, indent=2)); return
    source_path = OUT / "material/direct_composed_source.jsonl"
    source = read_rows(source_path) if source_path.exists() else compile_source()
    if not source_path.exists():
        write_rows(source_path, source)
    material = material_audit(source); save(OUT / "material/material_audit.json", material)
    q4, collection = run_model("qwen4b", source, True)
    q14, _ = run_model("qwen14b", source, False)
    rows = read_rows(OUT / "qwen4b/index/direct_composed_rows.jsonl")
    analysis = analyze(rows, collection)
    q14_behavior = q14["teacher"]["transitive"]["chain_minus_null"] > .05
    physical = all(analysis["summary"][component][split]["physical_advantage"] > 0
                   for component in COMPONENTS for split in SPLITS)
    negative = all(value["transitive_over_nontransitive_energy_ratio"] > 1 for value in analysis["comparison"].values())
    adjudication = {"qwen14_transitive_chain_behavior_qualified": q14_behavior,
                    "direct_to_composed_physical_law_all_components_splits": physical,
                    "transitive_composition_energy_exceeds_nontransitive_all_components": negative,
                    "relation_composition_operator_detected": q14_behavior and physical and negative,
                    "recursive_composition_mechanism_proven": False}
    checks = {"material_5120": material["rows"] == 5120 and material["configurations"] == 1024,
              "exact_five_variant_blocks": material["exact_five_variant_blocks"] == 1024,
              "qwen4b_complete": q4["all_checks_passed"], "qwen14b_complete": q14["all_checks_passed"],
              "full_coordinate_capture": collection["state"]["shape"] == [5120, 38, 2, 2560],
              "full_coordinate_derived": all(Path(path).exists() for path in analysis["files"].values()),
              "finite": all(math.isfinite(number) for value in analysis["comparison"].values() for number in value.values()),
              "raw_retained": all(Path(item["path"]).exists() for item in collection.values()),
              "claim_boundary": not adjudication["recursive_composition_mechanism_proven"]}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "material": material,
              "models": {"qwen4b": q4, "qwen14b": q14}, "collection": collection,
              "analysis": analysis, "adjudication": adjudication, "checks": checks,
              "all_checks_passed": all(checks.values())}
    save(final, result); append_memo(result)
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
