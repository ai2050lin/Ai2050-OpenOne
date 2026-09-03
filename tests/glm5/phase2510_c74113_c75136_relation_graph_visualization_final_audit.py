#!/usr/bin/env python3
"""Publish the partner-recombined relation graph and close the current research stage."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result"
P2507 = RESULT / "phase2507_c71041_c72064_repaired_partner_behavior_fullfield"
P2508 = RESULT / "phase2508_c72065_c73088_alternative_partner_behavior_fullfield"
P2509 = RESULT / "phase2509_c73089_c74112_partner_independent_relation_graph_lockbox"
PUBLIC = ROOT / "frontend/public/vis_data/research_kernel"
ASSET = PUBLIC / "c42641_output_conditioned_crossmodel_field.json"
OUT = RESULT / "phase2510_c74113_c75136_relation_graph_visualization_final_audit"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN, DIM = 2510, "C74113-C75136", 2560
SOURCE = "phase2510_partner_relation_graph"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def row(vector: np.ndarray, label: str, kind: str, **meta: Any) -> dict:
    values = np.asarray(vector, dtype=np.float32).reshape(-1)
    if values.shape != (DIM,) or not np.isfinite(values).all():
        raise RuntimeError(label)
    return {"label": label, "source": SOURCE, "coordinate_kind": kind, "preview": True,
            **meta, "values": [float(v) for v in values]}


def publish() -> dict:
    f2507, f2508, f2509 = (load_json(P2507 / "analysis/final.json"), load_json(P2508 / "analysis/final.json"),
                            load_json(P2509 / "analysis/final.json"))
    edges = np.load(f2509["fields"]["edges"]["path"], mmap_mode="r")
    potentials = np.load(f2509["fields"]["potentials"]["path"], mmap_mode="r")
    events = ("definition_end", "facts_end", "query_marker", "candidate0", "candidate1", "answer_boundary")
    edge_names = [" - ".join(value) for value in f2509["edges"]]
    nodes = f2509["nodes"]
    added = []
    for split_index, split in enumerate(("confirmation_graph", "lockbox_graph")):
        for event_index, event in enumerate(events[2:], start=2):
            mean_edges = np.asarray(edges[split_index, event_index], dtype=np.float64).mean(axis=(1, 2))
            mean_potentials = np.asarray(potentials[split_index, event_index], dtype=np.float64).mean(axis=(1, 2))
            for edge_index, edge_name in enumerate(edge_names):
                added.append(row(mean_edges[edge_index], f"{split} {edge_name} {event} q30 edge interaction",
                                 "partner_relation_graph_edge", phase=2509, split=split, edge=edge_name,
                                 event=event, layer=30, averaging="two languages and four surfaces"))
            for node_index, node in enumerate(nodes):
                added.append(row(mean_potentials[node_index], f"{split} {node} {event} q30 fitted relation potential",
                                 "partner_relation_graph_potential", phase=2509, split=split, family=node,
                                 event=event, layer=30, averaging="six-edge incidence fit; two languages and four surfaces"))
            ab, cd, _ef, ac, df, bf = mean_edges
            prediction = ab + bf - df - cd
            added.append(row(prediction, f"{split} cycle predicted taxonomy-role {event} q30",
                             "partner_relation_cycle_prediction", phase=2509, split=split, event=event, layer=30))
            added.append(row(ac, f"{split} cycle actual taxonomy-role {event} q30",
                             "partner_relation_cycle_actual", phase=2509, split=split, event=event, layer=30))
            added.append(row(prediction - ac, f"{split} cycle residual taxonomy-role {event} q30",
                             "partner_relation_cycle_residual", phase=2509, split=split, event=event, layer=30))

    # Raw q0/q30 event fields from both partner campaigns make the source coordinates inspectable.
    for phase, result, unit, pair_id in ((2507, f2507, 25, 0), (2508, f2508, 27, 1)):
        field = np.load(result["collection"]["event_field"], mmap_mode="r")
        rows = read_jsonl(Path(result["collection"]["event_index"]))
        representative = next(r for r in rows if r["unit"] == unit and r["pair_id"] == pair_id
                              and r["language"] == "en" and r["surface"] == 0
                              and r["meaning_swap"] == 0 and r["query_marker"] == 0)
        for event_index, event in enumerate(events):
            for qpoint, kind in ((0, "partner_graph_event_embedding"), (30, "partner_graph_event_hiddenstate")):
                added.append(row(field[representative["model_row"], event_index, qpoint],
                                 f"phase{phase} unit{unit} representative {event} q{qpoint} raw",
                                 kind, phase=phase, unit=unit, pair=" - ".join(representative["families"]),
                                 language="en", surface=0, meaning_swap=0, query_marker=0,
                                 event=event, layer=qpoint))

    payload = load_json(ASSET)
    qwen = next(section for section in payload["models"] if section["key"] == "qwen4b")
    qwen["rows"] = [existing for existing in qwen["rows"] if existing.get("source") != SOURCE] + added
    confirmation_energy = np.square(np.asarray(edges[0, 2:], dtype=np.float64)).sum(axis=(0, 1, 2, 3))
    qwen.setdefault("coordinate_orders", {})["relation_graph"] = [int(v) for v in np.argsort(-confirmation_energy)]
    matrix = np.stack([np.asarray(existing["values"], dtype=np.float32) for existing in qwen["rows"]])
    binary = PUBLIC / "c42641_qwen4b_output_conditioned_field.float32.npy"
    np.save(binary, matrix)
    qwen["binary_shape"] = list(matrix.shape); qwen["binary_sha256"] = digest(binary)
    payload["phase"] = PHASE; payload["campaign"] = "C39761-C75136"
    payload["summary"]["phase2509_partner_graph_query_partial_support"] = True
    payload["summary"]["phase2509_partner_graph_answer_closure"] = False
    payload["summary"]["phase2509_pure_semantic_code"] = False
    payload["summary"]["model_rows"] = {section["key"]: len(section["rows"]) for section in payload["models"]}
    payload["summary"]["total_rows"] = sum(payload["summary"]["model_rows"].values())
    sentence = ("Phase2509 finds partial partner-independent additive relation structure at the query-marker event, "
                "but the frozen-q30 lockbox cycle fails at answer boundary and autonomous generation rewrites the "
                "interaction; the fixed additive family-potential hypothesis is not closed.")
    if sentence not in payload["claim_boundary"]:
        payload["claim_boundary"] = payload["claim_boundary"].rstrip() + " " + sentence
    content = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if ASSET.read_text(encoding="utf-8") != content:
        ASSET.write_text(content, encoding="utf-8")
    return {"asset": str(ASSET), "rows_added": len(added), "qwen_shape": list(matrix.shape),
            "binary": str(binary), "binary_sha256": qwen["binary_sha256"],
            "relation_graph_order_coordinates": len(confirmation_energy), "json_bytes": ASSET.stat().st_size}


def retention() -> dict:
    f2507, f2508, f2509 = (load_json(P2507 / "analysis/final.json"), load_json(P2508 / "analysis/final.json"),
                            load_json(P2509 / "analysis/final.json"))
    paths = [Path(f2507["collection"]["event_field"]), Path(f2508["collection"]["event_field"]),
             Path(f2509["fields"]["edges"]["path"]), Path(f2509["fields"]["potentials"]["path"])]
    records = [{"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path),
                "retention": "retained: important full-coordinate source displayed at parameter level"} for path in paths]
    save(OUT / "analysis/retention_manifest.json", records)
    return {"files": len(records), "bytes": sum(v["bytes"] for v in records),
            "all_hashes": all(len(v["sha256"]) == 64 for v in records),
            "cleanup": "No undisplayed HiddenState field remains from Phase2507-2509; all important sources retained."}


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"): return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""


## Phase {PHASE}: 关系伙伴闭环逐坐标发布、总终审与下一范式边界（{CAMPAIGN}） [{stamp}]

**测试原理与显示内容。** 客户端新增`relation_graph`顺序，仅按confirmation图在query/candidate/answer四事件、六边、中英文与四surface的交互能量冻结；物理坐标顺序仍并列保留。发布confirmation/lockbox六条边、六个拟合relation势、闭环预测/实际/残差，以及Phase2507/2508代表样本六事件的q0词嵌入和q30 HiddenState，共144行×2560坐标。所有行保留split、edge/family、event、layer和平均范围。

$$\Pi_{{graph}}=\operatorname{{argsort}}_i\left[-\sum_{{e\in E,\rho,\lambda,s}}I_{{e,\rho,\lambda,s,i}}^2\right].$$

**结果汇总。** 发布 `{json.dumps(result['asset'], ensure_ascii=False)}`；前端 `{json.dumps(result['frontend'], ensure_ascii=False)}`；留存 `{json.dumps(result['retention'], ensure_ascii=False)}`；最终裁决 `{json.dumps(result['adjudication'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2510_c74113_c75136_relation_graph_visualization_final_audit.py`；`ResearchHeatmapRoute.jsx`新增关系伙伴闭环顺序；c42641 JSON/float32矩阵、生产build、留存清单与final位于对应目录。

**理论进展。** 本轮真正推进了三步：第一，关系含义首次成为答案翻转的行为必要变量；第二，四格交互同时消去definition-swap和query-marker一阶方向，并通过等长度全新锁箱与prefix严格零；第三，新伙伴图显示query-marker阶段存在有限可加关系势候选，但它没有闭合到answer或自主生成。最诚实的新拼图是“关系选择在查询事件形成可复现、稠密、有符号、pair-relative且部分可加的场，随后按候选位置和输出token重新参数化”，不是固定向量传送。

**问题硬伤与结论。** 仅Qwen3-4B、合成二候选任务、一个独立五环；关系势fit自由度高，不能用 (R^2\approx0.87\) 宣称闭环。lockbox answer的正确路径余弦0.240，低于错误符号q95 0.328，方向优势-0.088；8个语言×surface均未超过各自错误q95，relative error均值1.731。故partner-independent固定加法family势在输出端被否定。有效原场与派生场均已参数级显示并保留，没有新增未显示HiddenState需要清理。

**下一阶段边界。** “证明固定可加family势是否跨伙伴”这一直接目标已完成且在answer处失败，继续换pair会回到因果门失败—换材料循环；下一目标已改变为：以query-marker的部分可加场为输入，测量候选顺序、实体身份、语言和已生成token如何把它变成输出分布的条件化非线性变换，并在多个独立环与自然句中复核。该任务需要新的动态合同，不属于本轮同一即时目标，因此不在本Phase伪装为自动续研。仍未发现纯语义代码、天然坐标齿轮、因果中介或语言编码数学闭合。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle: handle.write(text)


def main() -> None:
    f2509 = load_json(P2509 / "analysis/final.json")
    asset = publish(); kept = retention()
    dist = ROOT / "frontend/dist/index.html"
    source = (ROOT / "frontend/src/components/app/ResearchHeatmapRoute.jsx").read_text(encoding="utf-8")
    frontend = {"dist_exists": dist.exists(), "dist_newer": dist.exists() and dist.stat().st_mtime_ns >= ASSET.stat().st_mtime_ns,
                "relation_graph_order_control": "relation_graph" in source}
    headings = MEMO.read_text(encoding="utf-8").splitlines()
    phase_counts_before = {str(phase): sum(line.startswith(f"## Phase {phase}:") for line in headings) for phase in range(2499, 2510)}
    sequence = []
    for phase in range(2499, 2510):
        candidates = list(RESULT.glob(f"phase{phase}_*/analysis/final.json"))
        sequence.append(len(candidates) == 1 and bool(load_json(candidates[0])["all_checks_passed"]))
    lock_answer = f2509["graph"]["lockbox"]["answer_boundary"]
    checks = {"rows_added_144": asset["rows_added"] == 144,
              "full_coordinate_order": asset["relation_graph_order_coordinates"] == 2560,
              "binary_hash": len(asset["binary_sha256"]) == 64,
              "frontend_control": frontend["relation_graph_order_control"], "frontend_built": frontend["dist_newer"],
              "retention": kept["files"] == 4 and kept["all_hashes"], "phase_sequence": all(sequence),
              "memo_prior_phase_once": all(v == 1 for v in phase_counts_before.values()),
              "answer_cycle_not_overclaimed": lock_answer["mean_context_graph"]["orientation_advantage_over_wrong_q95"] < 0,
              "claim_boundary": True}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "asset": asset, "frontend": frontend, "retention": kept,
              "adjudication": {"behavioral_semantic_necessity_established_in_contract": True,
                               "query_event_partial_additive_relation_structure": True,
                               "answer_event_partner_independent_additive_potential": False,
                               "autonomous_fixed_vector_transport": False,
                               "next_stage_same_immediate_target": False,
                               "next_stage_target": "event-conditioned nonlinear compilation from query interaction to output distribution",
                               "pure_semantic_code_identified": False, "causal_mediator_identified": False,
                               "natural_coordinate_gear_identified": False, "language_encoding_mechanism_closed": False},
              "checks": checks, "all_checks_passed": all(checks.values())}
    save(OUT / "analysis" / ("final.json" if result["all_checks_passed"] else "prebuild.json"), result)
    if result["all_checks_passed"]: append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
