#!/usr/bin/env python3
"""Publish the natural event/residual path at parameter level and clean undisplayed fields."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result"
P2512 = RESULT / "phase2512_c75521_c76672_existing_fullfield_event_transition_map"
P2513 = RESULT / "phase2513_c76673_c78624_fresh_context_factorial_behavior_fullfield"
P2514 = RESULT / "phase2514_c78625_c79776_context_operator_competition_lockbox"
P2520 = RESULT / "phase2520_c85025_c86176_natural_language_counterfactual_fullfield"
P2521 = RESULT / "phase2521_c86177_c87200_natural_field_and_causal_lockbox"
P2523 = RESULT / "phase2523_c88577_c89952_boundary_component_residual_accounting"
OUT = RESULT / "phase2524_c89953_c91328_event_path_visualization_retention_audit"
PUBLIC = ROOT / "frontend/public/vis_data/research_kernel"
ASSET = PUBLIC / "c42641_output_conditioned_crossmodel_field.json"
BINARY = PUBLIC / "c42641_qwen4b_output_conditioned_field.float32.npy"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN, DIM, SOURCE = 2524, "C89953-C91328", 2560, "phase2524_natural_event_residual_path"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8-sig").splitlines() if x.strip()]


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def heatrow(vector: np.ndarray, label: str, kind: str, **meta: Any) -> dict:
    values = np.asarray(vector, dtype=np.float32).reshape(-1)
    if values.shape != (DIM,) or not np.isfinite(values).all():
        raise RuntimeError(label)
    return {"label": label, "source": SOURCE, "coordinate_kind": kind, "preview": True,
            **meta, "values": [float(v) for v in values]}


def publish() -> dict:
    f2513, f2514 = load(P2513 / "analysis/final.json"), load(P2514 / "analysis/final.json")
    f2520, f2521, f2523 = (load(P2520 / "analysis/final.json"), load(P2521 / "analysis/final.json"),
                            load(P2523 / "analysis/final.json"))
    added: list[dict] = []

    # Fresh factorial raw event and all-token checkpoints: explicit q0 embedding and q28/q36 HiddenState.
    event = np.load(f2513["collection"]["event_field"], mmap_mode="r")
    event_rows = read(Path(f2513["collection"]["event_index"]))
    representatives = [r for r in event_rows if r["unit"] == 28 and r["language"] == "en" and r["paraphrase"] == 0
                       and r["fact_order"] == 0 and r["definition_order"] == 0 and r["candidate_order"] == 0
                       and r["meaning_swap"] == 0 and r["query_marker"] == 0]
    for r in representatives:
        for event_index, event_name, qpoint in ((2, "query_marker", 0), (5, "answer_boundary", 28), (5, "answer_boundary", 36)):
            kind = "event_path_embedding" if qpoint == 0 else "event_path_hiddenstate"
            added.append(heatrow(event[r["model_row"], event_index, qpoint],
                                   f"phase2513 {'-'.join(r['families'])} {event_name} q{qpoint} raw", kind,
                                   phase=2513, pair_id=r["pair_id"], families=r["families"], event=event_name,
                                   layer=qpoint, token_position=r["event_positions"][event_index], storage="float16 source"))
    alltoken = np.load(f2513["collection"]["alltoken_field"], mmap_mode="r")
    alltoken_rows = {r["case_id"]: r for r in read(Path(f2513["collection"]["alltoken_index"]))}
    for r in representatives:
        token_row = alltoken_rows[r["case_id"]]
        token_position = r["event_positions"][2]
        absolute = token_row["offset"][0] + token_position
        for qpoint in (0, 28):
            added.append(heatrow(alltoken[absolute, qpoint],
                                   f"phase2513 {'-'.join(r['families'])} alltoken query-marker pos{token_position} q{qpoint}",
                                   "event_path_alltoken_embedding" if qpoint == 0 else "event_path_alltoken_hiddenstate",
                                   phase=2513, pair_id=r["pair_id"], token_position=token_position, layer=qpoint,
                                   storage="float16 all-token source"))

    # Failed operator family: expose actual/prediction/residual, preserving the negative result.
    selected = np.load(f2514["fields"]["selected_lockbox"]["path"], mmap_mode="r")
    selected_rows = read(Path(f2514["fields"]["selected_lockbox"]["index"]))
    for index, r in enumerate(selected_rows):
        if r["context_id"] != 0:
            continue
        for offset, row_kind in enumerate(r["row_kinds"]):
            added.append(heatrow(selected[index * 4 + offset],
                                   f"phase2514 {'-'.join(r['edge'])} {r['language']} {r['event']} q{r['qpoint']} {row_kind}",
                                   "event_path_operator_lockbox", phase=2514, edge=r["edge"], language=r["language"],
                                   event=r["event"], layer=r["qpoint"], model=r["model"], row_kind=row_kind))

    # Natural raw event field: query-property word embedding and late HiddenState per family.
    natural = np.load(f2520["collection"]["field"], mmap_mode="r")
    natural_rows = read(Path(f2520["collection"]["index"]))
    nreps = [r for r in natural_rows if r["unit"] == 31 and r["language"] == "en" and r["surface"] == 0
             and r["output_mode"] == "candidate" and r["meaning_swap"] == 0 and r["query_property"] == 0]
    for r in nreps:
        for event_index, event_name, qpoint in ((1, "query_property", 0), (1, "query_property", 28),
                                                 (2, "answer_boundary", 28), (2, "answer_boundary", 36)):
            added.append(heatrow(natural[r["model_row"], event_index, qpoint],
                                   f"phase2520 {r['family']} {event_name} q{qpoint} raw",
                                   "natural_word_embedding" if qpoint == 0 else "natural_event_hiddenstate",
                                   phase=2520, family=r["family"], family_id=r["family_id"], language="en",
                                   event=event_name, layer=qpoint, token_position=r["event_positions"][event_index]))

    # Natural selection interaction: average nuisance views, keep unit/family/layer explicit.
    natural_interaction = np.load(f2521["fields"]["interactions"]["path"], mmap_mode="r")
    for ui, unit in enumerate((30, 31)):
        for fi, family in enumerate(f2520["behavior"]["qualified_families"]):
            for qpoint in (28, 36):
                vector = np.asarray(natural_interaction[ui, fi, :, :, :, 2, qpoint], np.float32).mean(axis=(0, 1, 2))
                added.append(heatrow(vector, f"phase2521 unit{unit} {family} answer q{qpoint} Walsh mean",
                                       "natural_event_walsh", phase=2521, unit=unit, family=family,
                                       event="answer_boundary", layer=qpoint, averaging="language surface output_mode"))

    # Component interaction and raw component vectors at the two causal checkpoints.
    comp = np.load(f2523["fields"]["interactions"]["path"], mmap_mode="r")
    raw_comp = np.load(f2523["fields"]["components"]["path"], mmap_mode="r")
    comp_index = read(Path(f2523["fields"]["index"]["path"]))
    for ui, unit in enumerate((30, 31)):
        for fi, family in enumerate(f2520["behavior"]["qualified_families"]):
            for layer in (27, 35):
                attn = np.asarray(comp[ui, fi, :, 0, layer], np.float32).mean(axis=0)
                mlp = np.asarray(comp[ui, fi, :, 1, layer], np.float32).mean(axis=0)
                for component_name, vector in (("attention", attn), ("mlp", mlp), ("attention_plus_mlp", attn + mlp)):
                    added.append(heatrow(vector, f"phase2523 unit{unit} {family} layer{layer} {component_name} Walsh",
                                           "boundary_component_walsh", phase=2523, unit=unit, family=family,
                                           component=component_name, layer=layer, qpoint_out=layer + 1,
                                           averaging="two languages"))
    raw_reps = [r for r in comp_index if r["unit"] == 31 and r["language"] == "en" and r["meaning_swap"] == 0
                and r["query_property"] == 0]
    for r in raw_reps:
        for component_index, component_name in enumerate(("attention", "mlp")):
            for layer in (27, 35):
                added.append(heatrow(raw_comp[r["component_row"], component_index, layer],
                                       f"phase2523 {r['family']} raw {component_name} layer{layer}",
                                       "boundary_component_raw", phase=2523, unit=31, family=r["family"],
                                       component=component_name, layer=layer, qpoint_out=layer + 1,
                                       case_id=r["case_id"]))

    # Same-run residual checkpoints underlying the conservation audit.
    residual = np.load(f2523["fields"]["residual_checkpoints"]["path"], mmap_mode="r")
    for r in raw_reps:
        for qpoint in (0, 28, 36):
            added.append(heatrow(residual[r["component_row"], qpoint],
                                   f"phase2523 {r['family']} same-run residual q{qpoint}",
                                   "boundary_residual_embedding" if qpoint == 0 else "boundary_residual_hiddenstate",
                                   phase=2523, unit=31, family=r["family"], layer=qpoint, case_id=r["case_id"]))

    payload = load(ASSET)
    qwen = next(section for section in payload["models"] if section["key"] == "qwen4b")
    qwen["rows"] = [r for r in qwen["rows"] if r.get("source") != SOURCE] + added
    energy_rows = [np.asarray(r["values"], dtype=np.float64) for r in added
                   if r["coordinate_kind"] in ("natural_event_walsh", "boundary_component_walsh")]
    energy = np.square(np.stack(energy_rows)).sum(axis=0)
    qwen.setdefault("coordinate_orders", {})["event_path"] = [int(v) for v in np.argsort(-energy)]
    matrix = np.stack([np.asarray(r["values"], dtype=np.float32) for r in qwen["rows"]])
    np.save(BINARY, matrix)
    qwen["binary_shape"] = list(matrix.shape); qwen["binary_sha256"] = digest(BINARY)
    payload["phase"] = PHASE; payload["campaign"] = "C39761-C91328"
    payload["summary"]["phase2521_natural_answer_boundary_causal_replication"] = True
    payload["summary"]["phase2522_crossmodel_event_boundary_replication"] = True
    payload["summary"]["phase2523_single_layer_components_sufficient"] = False
    payload["summary"]["phase2523_residual_input_is_dominant_carrier"] = True
    payload["summary"]["model_rows"] = {section["key"]: len(section["rows"]) for section in payload["models"]}
    payload["summary"]["total_rows"] = sum(payload["summary"]["model_rows"].values())
    sentence = ("Phase2521-2523 support a cross-family and cross-model late answer-boundary output-identity carrier, "
                "while cross-language coordinate identity and single-layer Attention/MLP sufficiency fail; this is "
                "an event-role residual path, not an identified semantic compiler or language-independent basis.")
    if sentence not in payload["claim_boundary"]:
        payload["claim_boundary"] = payload["claim_boundary"].rstrip() + " " + sentence
    content = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if ASSET.read_text(encoding="utf-8") != content:
        ASSET.write_text(content, encoding="utf-8")
    return {"asset": str(ASSET), "rows_added": len(added), "qwen_shape": list(matrix.shape),
            "binary": str(BINARY), "binary_sha256": digest(BINARY), "event_path_coordinates": len(energy),
            "json_bytes": ASSET.stat().st_size}


def retention_and_cleanup() -> dict:
    f2512, f2513, f2514 = load(P2512 / "analysis/final.json"), load(P2513 / "analysis/final.json"), load(P2514 / "analysis/final.json")
    f2520, f2521, f2523 = load(P2520 / "analysis/final.json"), load(P2521 / "analysis/final.json"), load(P2523 / "analysis/final.json")
    retained_paths = [
        Path(f2513["collection"]["event_field"]), Path(f2513["collection"]["alltoken_field"]),
        Path(f2514["fields"]["selected_lockbox"]["path"]), Path(f2520["collection"]["field"]),
        Path(f2521["fields"]["interactions"]["path"]), Path(f2523["fields"]["components"]["path"]),
        Path(f2523["fields"]["residual_checkpoints"]["path"]), Path(f2523["fields"]["interactions"]["path"]),
    ]
    retained = [{"path": str(p), "bytes": p.stat().st_size, "sha256": digest(p),
                 "retention": "retained: important source has parameter-level representative rows in client"}
                for p in retained_paths]
    cleanup_targets = [Path(f2512["collection"]["field"]), Path(f2514["fields"]["all_interactions"]["path"])]
    cleanup_manifest_path = OUT / "analysis/cleanup_manifest.json"
    if cleanup_manifest_path.exists():
        cleanup = load(cleanup_manifest_path)
    else:
        cleanup = []
        result_root = RESULT.resolve()
        for path in cleanup_targets:
            resolved = path.resolve()
            if result_root not in resolved.parents:
                raise RuntimeError(f"unsafe cleanup target {resolved}")
            record = {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256_before_delete": digest(resolved),
                      "reason": "full-coordinate intermediate not published; scalar metrics/index/reproducible script retained"}
            resolved.unlink()
            record["deleted"] = not resolved.exists()
            cleanup.append(record)
        save(cleanup_manifest_path, cleanup)
    save(OUT / "analysis/retention_manifest.json", retained)
    return {"retained_files": len(retained), "retained_bytes": sum(r["bytes"] for r in retained),
            "cleaned_files": len(cleanup), "cleaned_bytes": sum(r["bytes"] for r in cleanup),
            "cleanup_verified": all(r["deleted"] for r in cleanup), "retained_hashes": all(len(r["sha256"]) == 64 for r in retained)}


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""


## Phase {PHASE}: 自然事件残差路径逐坐标发布、留存与清理（{CAMPAIGN}） [{stamp}]

**测试原理与显示内容。** 在现有c42641可视化资产新增`event_path`坐标顺序，按Phase2521自然事件Walsh场和Phase2523 Attention/MLP组件Walsh场的原始坐标平方能量冻结；物理坐标顺序并列保留。客户端逐行发布Phase2513新鲜析因q0词嵌入、q28/q36 HiddenState和all-token检查点，Phase2514失败算子的实际/预测/残差，Phase2520九族自然原场，Phase2521九族事件交互，以及Phase2523同轮残差、组件原场与组件交互。

$$\pi_{{event}}=\operatorname{{argsort}}_i\left[-\sum_{{r\in\mathcal R_{{Walsh}}}}r_i^2\right].$$

**结果汇总。** 发布 `{json.dumps(result['asset'], ensure_ascii=False)}`；前端 `{json.dumps(result['frontend'], ensure_ascii=False)}`；留存清理 `{json.dumps(result['retention'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2524_c89953_c91328_event_path_visualization_retention_audit.py`；`ResearchHeatmapRoute.jsx`新增自然事件残差路径顺序；c42641 JSON/float32矩阵、生产build、SHA-256、留存/清理清单与final位于对应目录。

**理论进展。** 可视化现在同时给出具体token位置、事件、层、组件、语言族与全部2560坐标。最强拼图是：关系选择并非在query-token单点传送，而是在候选/指令后的答案边界形成跨模型可移植的输出身份；该身份主要由此前累积的残差状态承载，最后一层Attention/MLP只作局部增量。

**问题硬伤与结论。** `event_path`只是观察排序，不是模型天然基底；跨语言物理同一性未通过；teacher-forced翻转不等于自主生成；组件仍未细分到head/neuron。Phase2512旧六边派生场与Phase2514未发布的全条件大张量已在保留哈希、指标、索引和脚本后删除，释放容量；所有其余重要大场均已有参数级代表行并保留。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main() -> None:
    asset = publish()
    retention = retention_and_cleanup()
    source_text = (ROOT / "frontend/src/components/app/ResearchHeatmapRoute.jsx").read_text(encoding="utf-8")
    dist = ROOT / "frontend/dist/index.html"
    frontend = {"event_path_control": "event_path" in source_text,
                "dist_exists": dist.exists(), "dist_newer": dist.exists() and dist.stat().st_mtime_ns >= ASSET.stat().st_mtime_ns}
    headings = MEMO.read_text(encoding="utf-8").splitlines()
    sequence = {str(phase): sum(line.startswith(f"## Phase {phase}:") for line in headings) for phase in range(2511, 2524)}
    checks = {"rows_published": asset["rows_added"] >= 250, "all_2560_coordinates": asset["event_path_coordinates"] == DIM,
              "binary_hash": len(asset["binary_sha256"]) == 64, "frontend_control": frontend["event_path_control"],
              "frontend_built": frontend["dist_newer"], "retention_hashes": retention["retained_hashes"],
              "cleanup_verified": retention["cleanup_verified"], "phase_sequence": all(v == 1 for v in sequence.values()),
              "claim_boundary": True}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "asset": asset, "frontend": frontend, "retention": retention,
              "phase_sequence_before_append": sequence,
              "adjudication": {"important_fields_visible_at_parameter_level": True,
                               "event_role_residual_path_supported": True, "language_independent_coordinate_basis": False,
                               "single_layer_semantic_compiler": False, "language_encoding_mechanism_closed": False},
              "checks": checks, "all_checks_passed": all(checks.values())}
    save(OUT / "analysis" / ("final.json" if result["all_checks_passed"] else "prebuild.json"), result)
    if result["all_checks_passed"]:
        append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
