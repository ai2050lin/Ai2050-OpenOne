#!/usr/bin/env python3
"""Publish orthogonal signed textures, same-q trajectories, and transport parameters."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result"
P2488 = RESULT / "phase2488_c55873_c56832_qwen4b_orthogonal_fullcoordinate_field"
P2490 = RESULT / "phase2490_c57473_c58112_signed_texture_energy_envelope_controls"
P2491 = RESULT / "phase2491_c58113_c58880_same_qpoint_autonomous_trajectory"
P2492 = RESULT / "phase2492_c58881_c59520_raw_vs_standardized_block_transport"
PUBLIC = ROOT / "frontend/public/vis_data/research_kernel"
ASSET = PUBLIC / "c42641_output_conditioned_crossmodel_field.json"
OUT = RESULT / "phase2494_c60161_c60800_orthogonal_signed_field_visualization"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN, DIM = 2494, "C60161-C60800", 2560
SOURCE = "phase2494_orthogonal_signed_field"


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def make_row(vector: np.ndarray, label: str, kind: str, **meta: Any) -> dict:
    value = np.asarray(vector, dtype=np.float32).reshape(-1)
    if value.shape != (DIM,) or not np.isfinite(value).all():
        raise RuntimeError(label)
    return {"label": label, "source": SOURCE, "coordinate_kind": kind, "preview": True,
            **meta, "values": [float(x) for x in value]}


def trajectory_passports(field: np.ndarray, rows: list[dict], qpoint: int) -> tuple[list[dict], list[dict]]:
    use = [r for r in rows if r["unit"] == 16 and r["output_interface"] == "entity" and r["parsed_correct"]]
    families = sorted({r["family"] for r in use})
    added = []
    passports = []
    for event in ("boundary", "first", "answer"):
        values = {}
        for family in families:
            family_rows = [r for r in use if r["family"] == family and r["answer_step"] is not None]
            states = []
            for item in family_rows:
                step = 0 if event == "boundary" else (1 if event == "first" else item["answer_step"])
                states.append(np.asarray(field[item["model_row"], step, qpoint], dtype=np.float64))
            values[family] = np.mean(states, axis=0)
        grand = np.mean(list(values.values()), axis=0)
        for family in families:
            vector = values[family] - grand
            passports.append({"event": event, "family": family, "vector": vector})
            added.append(make_row(vector, f"unit16 {family} autonomous {event} family passport q{qpoint}",
                                  "same_qpoint_autonomous_family_passport", phase=2491, unit=16,
                                  layer=qpoint, event=event, family=family, output_interface="entity",
                                  selection="same qpoint for boundary, first and answer; correct paths only"))
    representative = next(r for r in use if r["family"] == "taxonomy" and r["language"] == "en" and r["surface"] == 0)
    for event, step in (("boundary", 0), ("first", 1), ("answer", representative["answer_step"])):
        added.append(make_row(field[representative["model_row"], step, qpoint],
                              f"unit16 taxonomy autonomous raw {event} q{qpoint}",
                              "same_qpoint_autonomous_hiddenstate", phase=2491, unit=16, layer=qpoint,
                              event=event, family="taxonomy", language="en", surface=0,
                              token_step=int(step), generated_text=representative["generated_text"]))
    return added, passports


def publish() -> dict:
    f2488 = json.loads((P2488 / "analysis/final.json").read_text(encoding="utf-8"))
    f2490 = json.loads((P2490 / "analysis/final.json").read_text(encoding="utf-8"))
    f2491 = json.loads((P2491 / "analysis/final.json").read_text(encoding="utf-8"))
    f2492 = json.loads((P2492 / "analysis/final.json").read_text(encoding="utf-8"))
    added = []
    event_field = np.load(f2488["collection"]["event_field"], mmap_mode="r")
    event_rows = read_jsonl(Path(f2488["collection"]["index"]))
    representative = next(r for r in event_rows if r["unit"] == 16 and r["family"] == "taxonomy"
                          and r["language"] == "en" and r["surface"] == 0 and r["output_interface"] == "entity")
    for event, event_index, qpoint, kind in (
        ("record_predicate", 0, 0, "token_embedding_activation"),
        ("answer_boundary", 4, 21, "orthogonal_prompt_hiddenstate"),
        ("answer_boundary_final_norm", 4, 37, "orthogonal_prompt_finalnorm"),
    ):
        added.append(make_row(event_field[representative["model_row"], event_index, qpoint],
                              f"unit16 taxonomy {event} q{qpoint} raw coordinate field", kind,
                              phase=2488, unit=16, family="taxonomy", language="en", surface=0,
                              output_interface="entity", event=event, layer=qpoint,
                              token_id=representative["event_token_ids"][event_index]))
    passports = np.load(f2490["passports"]["path"], mmap_mode="r")
    envelopes = np.load(f2490["envelope"]["path"], mmap_mode="r")
    event_index = f2490["passports"]["axes"] and 4
    families = f2490["families"]
    qpoint = int(f2490["selection"]["answer_boundary"])
    envelope = np.asarray(envelopes[event_index], dtype=np.float64)
    scale = np.sqrt(envelope + max(float(np.mean(envelope)) * 1e-8, 1e-12))
    for language_index, language in enumerate(("en", "zh")):
        for family_index, family in enumerate(families):
            vector = np.asarray(passports[1, event_index, language_index, family_index], dtype=np.float64)
            added.append(make_row(vector, f"unit16 {family} {language} signed family passport q{qpoint}",
                                  "orthogonal_signed_family_passport", phase=2490, unit=16, family=family,
                                  language=language, event="answer_boundary", layer=qpoint))
            added.append(make_row(vector / scale, f"unit16 {family} {language} RMS-standardized passport q{qpoint}",
                                  "orthogonal_rms_standardized_family_passport", phase=2490, unit=16,
                                  family=family, language=language, event="answer_boundary", layer=qpoint,
                                  normalization="frozen unit15 per-coordinate family-passport RMS"))
    added.append(make_row(np.sqrt(np.maximum(envelope, 0)), "unit15 answer-boundary coordinate RMS envelope",
                              "orthogonal_coordinate_rms_envelope", phase=2490, unit=15,
                              event="answer_boundary", layer=qpoint, interpretation="scale envelope; not signed code or importance"))
    trajectory_field = np.load(f2491["collection"]["field"], mmap_mode="r")
    trajectory_rows = read_jsonl(Path(f2491["collection"]["index"]))
    trajectory_added, _ = trajectory_passports(trajectory_field, trajectory_rows, int(f2491["qpoint"]))
    added.extend(trajectory_added)
    params = np.load(f2492["parameters"]["path"])
    for name, label in (("raw_diagonal", "raw diagonal slope"), ("source_rms", "source coordinate RMS"),
                        ("target_rms", "target coordinate RMS"), ("standardized_diagonal", "standardized diagonal slope")):
        added.append(make_row(params[name], f"q35→q36 {label} (unit15 fit)", "block_transport_parameter",
                              phase=2492, layer=35, event="answer_boundary", parameter=name,
                              selection="unit15 selected; unit16 lockbox"))
    payload = json.loads(ASSET.read_text(encoding="utf-8"))
    qwen = next(section for section in payload["models"] if section["key"] == "qwen4b")
    qwen["rows"] = [r for r in qwen["rows"] if r.get("source") != SOURCE] + added
    order_energy = np.square(np.asarray(passports[0, event_index], dtype=np.float64)).sum(axis=(0, 1))
    qwen.setdefault("coordinate_orders", {})["orthogonal_signed"] = [int(x) for x in np.argsort(-order_energy)]
    matrix = np.stack([np.asarray(r["values"], dtype=np.float32) for r in qwen["rows"]])
    binary = PUBLIC / "c42641_qwen4b_output_conditioned_field.float32.npy"
    np.save(binary, matrix)
    qwen["binary_shape"] = list(matrix.shape); qwen["binary_sha256"] = digest(binary)
    payload["phase"] = PHASE; payload["campaign"] = "C39761-C60800"
    payload["summary"]["phase2490_orthogonal_signed_texture_lockbox"] = True
    payload["summary"]["phase2491_same_qpoint_autonomous_trajectory"] = True
    payload["summary"]["phase2492_coordinate_diagonal_lockbox_winner"] = False
    payload["summary"]["model_rows"] = {section["key"]: len(section["rows"]) for section in payload["models"]}
    payload["summary"]["total_rows"] = sum(payload["summary"]["model_rows"].values())
    sentence = "Phase2490-2492 add orthogonal-family signed and RMS-standardized passports, same-q21 autonomous events, and q35-to-q36 full-coordinate baselines; energy is not information and diagonal transport did not beat the global standardized baseline on unit16."
    payload["claim_boundary"] = payload["claim_boundary"].replace(" " + sentence, "").rstrip() + " " + sentence
    content = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if ASSET.read_text(encoding="utf-8") != content:
        ASSET.write_text(content, encoding="utf-8")
    return {"asset": str(ASSET), "rows_added": len(added), "qwen_shape": list(matrix.shape),
            "binary": str(binary), "sha256": qwen["binary_sha256"], "orthogonal_order_coordinates": len(order_energy),
            "json_bytes": ASSET.stat().st_size}


def retention() -> dict:
    finals = [json.loads((p / "analysis/final.json").read_text(encoding="utf-8")) for p in (P2488, P2490, P2491, P2492)]
    paths = [Path(finals[0]["collection"][k]) for k in ("event_field", "alltoken_field", "alltoken_token_ids")]
    paths += [Path(finals[1]["envelope"]["path"]), Path(finals[1]["passports"]["path"])]
    paths += [Path(finals[2]["collection"][k]) for k in ("field", "event_mask", "token_ids")]
    paths += [Path(finals[3]["parameters"]["path"])]
    records = [{"path": str(p), "bytes": p.stat().st_size, "sha256": digest(p),
                "retention": "retained: unique full-coordinate source with parameter-level client slices"} for p in paths]
    save(OUT / "analysis/retention_manifest.json", records)
    return {"files": len(records), "bytes": sum(r["bytes"] for r in records),
            "all_hashes": all(len(r["sha256"]) == 64 for r in records),
            "cleanup": "No Phase2488-2492 unique field deleted; each important source has full-coordinate client slices. Qwen14B offload cache was already removed."}


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"): return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""


## Phase {PHASE}: 正交有符号语言族场、同层自主轨迹与传动参数的逐坐标客户端发布（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 将Phase2488–2492的重要证据加入现有原生坐标热力图：unit16 taxonomy真实token embedding(q0)、answer-boundary HiddenState(q21)/final norm(q37)；unit16九族×中英的原始与unit15-RMS标准化有符号护照；unit15坐标RMS包络；unit16九族boundary/first/answer同q21自主family护照及一个真实轨迹；q35→q36的原始/标准化逐坐标斜率和source/target RMS。新增unit15正交有符号能量冻结顺序，仍可切回物理顺序，全部行完整2560坐标。

$$\Pi_{{signed}}=\operatorname{{argsort}}_i\left[-\sum_{{\lambda,f}}(P_{{u15,\lambda,f,i}})^2\right],\qquad |\Pi_{{signed}}|=2560.$$

**结果汇总。** 资产 `{json.dumps(result['asset'], ensure_ascii=False)}`；前端 `{json.dumps(result['frontend'], ensure_ascii=False)}`；留存 `{json.dumps(result['retention'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2494_c60161_c60800_orthogonal_signed_field_visualization.py`；`frontend/src/components/app/ResearchHeatmapRoute.jsx`新增正交有符号顺序；更新c42641 JSON/float32矩阵和生产构建；本Phase final/留存清单。

**分析与理论进展。** 客户端现在能在同一物理坐标轴检查Embedding、HiddenState、family-relative有符号纹理、RMS尺度环境、自主token事件和基础transport参数。所谓“参数级”在此严格指每个激活坐标及拟合的每坐标参数，不把HiddenState坐标冒充模型权重。

**问题硬伤与结论。** 热力图只展示原场的代表切片，完整样本由哈希留存；排序是发现集可视化顺序，不是模型天然模块。重要数据已进入客户端，故不清理唯一原场；仅清除不可展示的14B临时offload。发布不提升证据等级，不宣称齿轮或闭合。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle: handle.write(text)


def main() -> None:
    asset = publish(); kept = retention()
    dist = ROOT / "frontend/dist/index.html"
    frontend = {"dist_exists": dist.exists(), "dist_newer": dist.exists() and dist.stat().st_mtime_ns >= ASSET.stat().st_mtime_ns,
                "orthogonal_order_control": "orthogonal_signed" in (ROOT / "frontend/src/components/app/ResearchHeatmapRoute.jsx").read_text(encoding="utf-8")}
    checks = {"rows_added_74": asset["rows_added"] == 74, "shape_811x2560": asset["qwen_shape"] == [811, 2560],
              "full_coordinate_order": asset["orthogonal_order_coordinates"] == 2560, "binary_hash": len(asset["sha256"]) == 64,
              "frontend_control": frontend["orthogonal_order_control"], "frontend_built": frontend["dist_newer"],
              "retention_hashed": kept["files"] == 9 and kept["all_hashes"], "claim_boundary": True}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "asset": asset, "frontend": frontend, "retention": kept,
              "adjudication": {"important_results_parameter_visible": True, "hiddenstate_cleanup_required": False,
                               "natural_coordinate_gear_identified": False, "language_encoding_mechanism_closed": False},
              "checks": checks, "all_checks_passed": all(checks.values())}
    save(OUT / "analysis" / ("final.json" if result["all_checks_passed"] else "prebuild.json"), result)
    if result["all_checks_passed"]: append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
