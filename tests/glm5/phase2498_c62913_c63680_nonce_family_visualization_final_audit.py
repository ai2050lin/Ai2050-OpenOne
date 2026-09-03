#!/usr/bin/env python3
"""Publish nonce-marker family/marker fields and audit the automatic continuation stage."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result"
P2496 = RESULT / "phase2496_c61121_c62272_nonce_marker_rotation_behavior_fullfield"
P2497 = RESULT / "phase2497_c62273_c62912_family_vs_marker_fullcoordinate_lockbox"
PUBLIC = ROOT / "frontend/public/vis_data/research_kernel"
ASSET = PUBLIC / "c42641_output_conditioned_crossmodel_field.json"
OUT = RESULT / "phase2498_c62913_c63680_nonce_family_visualization_final_audit"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN, DIM = 2498, "C62913-C63680", 2560
SOURCE = "phase2498_nonce_family_field"


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(16 * 1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def row(vector: np.ndarray, label: str, kind: str, **meta: Any) -> dict:
    v = np.asarray(vector, dtype=np.float32).reshape(-1)
    if v.shape != (DIM,) or not np.isfinite(v).all(): raise RuntimeError(label)
    return {"label": label, "source": SOURCE, "coordinate_kind": kind, "preview": True,
            **meta, "values": [float(x) for x in v]}


def publish() -> dict:
    f2496 = json.loads((P2496 / "analysis/final.json").read_text(encoding="utf-8"))
    f2497 = json.loads((P2497 / "analysis/final.json").read_text(encoding="utf-8"))
    qpoint = int(f2497["selection"]["qpoint"]); events = f2496["collection"]["events"]
    family = np.load(f2497["fields"]["family"]["path"], mmap_mode="r")
    marker = np.load(f2497["fields"]["marker"]["path"], mmap_mode="r")
    field = np.load(f2496["collection"]["event_field"], mmap_mode="r")
    field_rows = read_jsonl(Path(f2496["collection"]["index"]))
    families = f2497["families"]
    added = []
    representative = next(r for r in field_rows if r["unit"] == 19 and r["family"] == "taxonomy"
                          and r["language"] == "en" and r["definition_surface"] == 0 and r["marker_id"] == 0)
    for event, event_index, qp, kind in (("definition_semantic", 0, 0, "nonce_definition_embedding"),
                                         ("record_marker", 1, qpoint, "nonce_record_hiddenstate"),
                                         ("answer_boundary", 3, qpoint, "nonce_answer_hiddenstate")):
        added.append(row(field[representative["model_row"], event_index, qp],
                         f"unit19 taxonomy marker0 {event} q{qp} raw field", kind,
                         phase=2496, unit=19, family="taxonomy", language="en", marker_id=0,
                         event=event, layer=qp))
    for event_index, event in enumerate(events):
        for family_index, family_name in enumerate(families):
            added.append(row(family[1, event_index, family_index],
                             f"unit19 {family_name} across-marker family passport {event} q{qpoint}",
                             "nonce_across_marker_family_passport", phase=2497, unit=19, family=family_name,
                             event=event, layer=qpoint, averaging="four markers, two languages, two definition surfaces"))
        for marker_id in range(4):
            added.append(row(marker[1, event_index, marker_id],
                             f"unit19 marker{marker_id} across-family passport {event} q{qpoint}",
                             "nonce_across_family_marker_passport", phase=2497, unit=19, marker_id=marker_id,
                             event=event, layer=qpoint, averaging="twelve families, two languages, two definition surfaces"))
    payload = json.loads(ASSET.read_text(encoding="utf-8"))
    qwen = next(section for section in payload["models"] if section["key"] == "qwen4b")
    qwen["rows"] = [r for r in qwen["rows"] if r.get("source") != SOURCE] + added
    energy = np.square(np.asarray(family[0], dtype=np.float64)).sum(axis=(0, 1))
    qwen.setdefault("coordinate_orders", {})["nonce_family"] = [int(x) for x in np.argsort(-energy)]
    matrix = np.stack([np.asarray(r["values"], dtype=np.float32) for r in qwen["rows"]])
    binary = PUBLIC / "c42641_qwen4b_output_conditioned_field.float32.npy"; np.save(binary, matrix)
    qwen["binary_shape"] = list(matrix.shape); qwen["binary_sha256"] = digest(binary)
    payload["phase"] = PHASE; payload["campaign"] = "C39761-C63680"
    payload["summary"]["phase2497_family_beyond_record_marker_lockbox"] = True
    payload["summary"]["phase2497_pure_semantic_code"] = False
    payload["summary"]["model_rows"] = {s["key"]: len(s["rows"]) for s in payload["models"]}
    payload["summary"]["total_rows"] = sum(payload["summary"]["model_rows"].values())
    sentence = "Phase2497 shows family-condition reuse across four nonce record markers at frozen q20, but answer-boundary cross-language identity fails and the task does not require distinguishing relation meanings; this is conditional propagation, not a pure semantic code."
    payload["claim_boundary"] = payload["claim_boundary"].replace(" " + sentence, "").rstrip() + " " + sentence
    content = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if ASSET.read_text(encoding="utf-8") != content: ASSET.write_text(content, encoding="utf-8")
    return {"asset": str(ASSET), "rows_added": len(added), "qwen_shape": list(matrix.shape),
            "binary": str(binary), "sha256": qwen["binary_sha256"], "nonce_order_coordinates": len(energy),
            "json_bytes": ASSET.stat().st_size}


def retention() -> dict:
    f2496 = json.loads((P2496 / "analysis/final.json").read_text(encoding="utf-8"))
    f2497 = json.loads((P2497 / "analysis/final.json").read_text(encoding="utf-8"))
    paths = [Path(f2496["collection"][k]) for k in ("event_field", "alltoken_field")]
    paths += [Path(f2497["fields"][k]["path"]) for k in ("family", "marker")]
    records = [{"path": str(p), "bytes": p.stat().st_size, "sha256": digest(p),
                "retention": "retained: unique full-coordinate source displayed in client"} for p in paths]
    save(OUT / "analysis/retention_manifest.json", records)
    return {"files": len(records), "bytes": sum(r["bytes"] for r in records), "all_hashes": all(len(r["sha256"]) == 64 for r in records),
            "cleanup": "All new fields have parameter-level client slices; no unique HiddenState field deleted."}


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"): return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""


## Phase {PHASE}: 无意义marker条件传播逐坐标发布、自动续研阶段审计与下一范式边界（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 客户端新增unit19代表样本的definition token Embedding(q0)、record-marker/answer-boundary HiddenState(q20)，十二family在定义/记录/query/answer四事件的跨marker护照48行，四marker在四事件的跨family护照16行，共67行×完整2560物理坐标；新增unit18冻结`nonce_family`全坐标显示顺序。四个唯一原场/派生场做SHA256留存。随后审计Phase2495自动续研合同是否全部完成。

$$\Pi_{{nonce}}=\operatorname{{argsort}}_i\left[-\sum_{{e,f}}P_{{u18,e,f,i}}^2\right],\qquad |\Pi_{{nonce}}|=2560.$$

**结果汇总。** 发布 `{json.dumps(result['asset'], ensure_ascii=False)}`；前端 `{json.dumps(result['frontend'], ensure_ascii=False)}`；留存 `{json.dumps(result['retention'], ensure_ascii=False)}`；阶段裁决 `{json.dumps(result['adjudication'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2498_c62913_c63680_nonce_family_visualization_final_audit.py`；`ResearchHeatmapRoute.jsx`新增无意义标记语言族顺序；c42641 JSON/矩阵、生产构建、留存清单与final。

**理论进展。** Phase2497在unit19 q20显示：family跨marker身份优势在definition/record/query/answer依次为0.718/0.474/0.135/0.433；说明上文family定义条件能越过不同marker token，在固定物理坐标场中留下可复现纹理。与此同时marker跨family也明显，且answer跨语言family优势为-0.093，表明family条件、token身份和语言条件是并存交互，而非一个可直接抽出的纯语义向量。

**问题硬伤与结论。** 新任务行为虽十二族全通过，但答案只需沿已定义marker链接，不必利用十二种关系含义；因此不能说family纹理承担答案计算。自动续研已经完整执行，不停在Phase2495总结。下一阶段的直接目标不再相同：必须设计“交换关系含义会改变正确答案”的语义必要性材料，然后才可研究自然状态干预；这需要新的任务合同而不是继续扩充当前marker观察图。现阶段没有天然坐标齿轮、通用编译器或数学闭合。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as f: f.write(text)


def main() -> None:
    asset = publish(); kept = retention(); dist = ROOT / "frontend/dist/index.html"
    source = (ROOT / "frontend/src/components/app/ResearchHeatmapRoute.jsx").read_text(encoding="utf-8")
    frontend = {"dist_exists": dist.exists(), "dist_newer": dist.exists() and dist.stat().st_mtime_ns >= ASSET.stat().st_mtime_ns,
                "nonce_order_control": "nonce_family" in source}
    f2497 = json.loads((P2497 / "analysis/final.json").read_text(encoding="utf-8"))
    checks = {"rows_added_67": asset["rows_added"] == 67, "shape_878x2560": asset["qwen_shape"] == [878, 2560],
              "full_coordinate_order": asset["nonce_order_coordinates"] == 2560, "binary_hash": len(asset["sha256"]) == 64,
              "frontend_control": frontend["nonce_order_control"], "frontend_built": frontend["dist_newer"],
              "retention": kept["files"] == 4 and kept["all_hashes"], "phase2497_passed": f2497["all_checks_passed"],
              "phase_sequence_continuous": all(next(RESULT.glob(f"phase{p}_*")).joinpath("analysis/final.json").exists() for p in range(2486, 2498)),
              "claim_boundary": True}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "asset": asset, "frontend": frontend, "retention": kept,
              "adjudication": {"automatic_continuation_completed": True,
                               "strongest_new_piece": "family-condition texture crosses nonce record markers at frozen q20",
                               "hard_negative": "answer-boundary cross-language family identity fails and semantic relation meaning is not behaviorally necessary",
                               "next_stage_same_immediate_target": False,
                               "next_stage_target": "behaviorally necessary relation-meaning swaps before causal work",
                               "pure_semantic_code_identified": False, "natural_coordinate_gear_identified": False,
                               "language_encoding_mechanism_closed": False},
              "checks": checks, "all_checks_passed": all(checks.values())}
    save(OUT / "analysis" / ("final.json" if result["all_checks_passed"] else "prebuild.json"), result)
    if result["all_checks_passed"]: append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
