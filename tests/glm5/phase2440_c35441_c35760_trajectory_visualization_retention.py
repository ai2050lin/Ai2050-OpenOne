#!/usr/bin/env python3
"""Publish the Phase2435-2439 full-coordinate trajectory atlas and hash retained raw fields."""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result"
P2435 = RESULT / "phase2435_c33841_c34160_hypergraph_material_fourmodel_behavior/qwen4b"
P2436 = RESULT / "phase2436_c34161_c34480_qwen4b_hypergraph_fullfield"
P2437 = RESULT / "phase2437_c34481_c34800_signed_trajectory_atlas"
P2438 = RESULT / "phase2438_c34801_c35120_coordinate_event_group_tournament"
P2439 = RESULT / "phase2439_c35121_c35440_output_autonomous_bridge"
OUT = RESULT / "phase2440_c35441_c35760_trajectory_visualization_retention"
ASSET = ROOT / "frontend/public/vis_data/research_kernel/c32561_semantic_encoding_output_field.json"
BINARY = ROOT / "frontend/public/vis_data/research_kernel/c32561_semantic_encoding_output_field.float32.npy"
DIST = ROOT / "frontend/dist/index.html"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE = 2440
CAMPAIGN = "C35441-C35760"
DIM = 2560
TAG = "phase2440_trajectory_hypergraph"


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def save_if_changed(path: Path, value: Any) -> None:
    content = json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n"
    if not path.exists() or path.read_text(encoding="utf-8") != content:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def close(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def add(rows: list[dict], values: np.ndarray, label: str, source: str, coordinate_kind: str,
        layer: int | None = None, event: str = "", family: str = "", preview: bool = True,
        **metadata: Any) -> None:
    vector = np.asarray(values, dtype=np.float32).reshape(-1)
    if vector.shape != (DIM,) or not np.isfinite(vector).all():
        raise RuntimeError((label, vector.shape, bool(np.isfinite(vector).all())))
    rows.append({"label": label, "source": source, "coordinate_kind": coordinate_kind,
                 "component": metadata.pop("component", ""), "layer": layer, "event": event,
                 "family": family, "preview": preview, "campaign_tag": TAG, **metadata,
                 "values": [float(value) for value in vector]})


def selected(rows: list[dict]) -> tuple[int, dict]:
    for index, row in enumerate(rows):
        if (row["family"] == "taxonomy" and int(row["unit"]) == 5 and row["language"] == "zh"
                and row["surface"] == "natural" and int(row["direction"]) == 0
                and row["variant"] == "valid" and row["query_role"] == "target"):
            return index, row
    raise RuntimeError("phase2440 selected case not found")


def build_asset() -> dict:
    payload = json.loads(ASSET.read_text(encoding="utf-8"))
    old_rows = [row for row in payload["rows"] if row.get("campaign_tag") != TAG]
    source_rows = read_rows(P2435 / "index/trajectory_rows.jsonl")
    config_rows = read_rows(P2437 / "index/configurations.jsonl")
    row_index, case = selected(source_rows)
    config_index = next(index for index, row in enumerate(config_rows) if row["config_id"] == case["config_id"])
    families = json.loads((P2437 / "analysis/final.json").read_text(encoding="utf-8"))["analysis"]["families"]
    family_index = families.index("taxonomy")
    event_names = ("prefix_end", "operation_end", "argument_end", "context_end", "query_end",
                   "candidate1_end", "candidate2_end", "answer_boundary")
    event_state = np.load(P2436 / "raw/hypergraph_event_field.float16.npy", mmap_mode="r")
    signed_state = np.load(P2437 / "derived/signed_interaction_state.float16.npy", mmap_mode="r")
    passports = np.load(P2437 / "derived/signed_update_family_passports.float32.npy", mmap_mode="r")
    rms = np.load(P2437 / "derived/signed_update_coordinate_rms.float64.npy", mmap_mode="r")
    slopes = np.load(P2438 / "derived/discovery_diagonal_slopes.float32.npy", mmap_mode="r")
    contributions = np.load(P2439 / "derived/logit_lens_coordinate_contribution.float32.npy", mmap_mode="r")
    weight_difference = np.load(P2439 / "derived/target_foil_output_weight_difference.float32.npy", mmap_mode="r")
    output_interaction = np.load(P2439 / "derived/signed_output_contribution_interaction.float32.npy", mmap_mode="r")
    fresh_family = np.asarray([int(row["unit"]) == 5 and row["family"] == "taxonomy" for row in config_rows])
    new_rows: list[dict] = []
    for qpoint in (0, 12, 18, 24, 36, 37):
        add(new_rows, event_state[row_index, qpoint, 4], f"Phase2440 selected taxonomy q{qpoint} query-end state",
            "phase2436_event_fullfield", "embedding_activation" if qpoint == 0 else "hidden_state",
            layer=qpoint, event="query_end", family="taxonomy", preview=qpoint in (0, 18, 36, 37), case_id=case["case_id"])
    for qpoint in (0, 18, 36, 37):
        add(new_rows, event_state[row_index, qpoint, 7], f"Phase2440 selected taxonomy q{qpoint} answer-boundary state",
            "phase2436_event_fullfield", "embedding_activation" if qpoint == 0 else "hidden_state",
            layer=qpoint, event="answer_boundary", family="taxonomy", preview=qpoint in (18, 36, 37), case_id=case["case_id"])
    for ii, interaction in enumerate(("semantic_validity", "lexical_control")):
        add(new_rows, signed_state[ii, 18, 4, config_index], f"{interaction} selected-config state q18 query-end",
            "phase2437_signed_interaction", "full_coordinate_signed_state", layer=18, event="query_end",
            family="taxonomy", preview=True)
        for update, event in ((18, 4), (35, 4), (35, 6)):
            add(new_rows, passports[ii, 2, update, event, family_index],
                f"{interaction} fresh taxonomy update passport q{update} {event_names[event]}",
                "phase2437_fresh_update_passport", "full_coordinate_update", layer=update,
                event=event_names[event], family="taxonomy", preview=interaction == "semantic_validity")
        add(new_rows, slopes[ii, 18, 4], f"{interaction} state-to-update diagonal slope q18 query-end",
            "phase2438_coordinate_tournament", "fitted_diagonal_parameter", layer=18, event="query_end",
            preview=True)
        add(new_rows, rms[ii], f"{interaction} all-block/event/config coordinate RMS",
            "phase2437_coordinate_rms", "full_coordinate_rms", preview=False)
        add(new_rows, output_interaction[ii, 37, fresh_family].mean(axis=0),
            f"{interaction} fresh taxonomy final output-contribution interaction",
            "phase2439_output_interaction", "coordinate_logit_contribution", layer=37,
            event="answer_boundary", family="taxonomy", preview=True)
    add(new_rows, weight_difference[row_index], "selected target-minus-foil output embedding weight",
        "phase2439_output_weight", "output_embedding_weight_difference", preview=True, case_id=case["case_id"])
    for qpoint in (18, 36, 37):
        add(new_rows, contributions[row_index, qpoint], f"selected q{qpoint} H_i x delta-W_i logit contribution",
            "phase2439_parameter_readout", "coordinate_logit_contribution", layer=qpoint,
            event="answer_boundary", preview=True, case_id=case["case_id"])
    for value in (event_state, signed_state, passports, rms, slopes, contributions, weight_difference, output_interaction):
        close(value)
    rows = old_rows + new_rows
    matrix = np.stack([np.asarray(row["values"], dtype=np.float32) for row in rows])
    np.save(BINARY, matrix)
    p2437 = json.loads((P2437 / "analysis/final.json").read_text(encoding="utf-8"))
    p2438 = json.loads((P2438 / "analysis/final.json").read_text(encoding="utf-8"))
    p2439 = json.loads((P2439 / "analysis/final.json").read_text(encoding="utf-8"))
    payload.update({
        "phase": PHASE, "campaign": "C32561-C35760", "model": "Qwen3-4B-BF16",
        "coordinate_semantics": "all 2560 fixed physical coordinates; includes exact token embedding/readout parameters, embedding activation, event HiddenState, signed semantic/lexical interactions, family passports, coordinate RMS, fitted slopes, and output contributions",
        "rows": rows,
        "summary": {**payload.get("summary", {}), "phase2440_added_rows": len(new_rows),
                    "phase2437_semantic_lexical_energy_ratio": p2437["analysis"]["summary"]["semantic_to_lexical_energy_ratio"],
                    "phase2438_fresh_diagonal_gain": p2438["analysis"]["summary"]["semantic_validity"]["fresh_unit"]["diagonal"],
                    "phase2438_language_diagonal_gain": p2438["analysis"]["summary"]["semantic_validity"]["language"]["diagonal"],
                    "phase2439_readout_correlation": p2439["closure"]["correlation"],
                    "phase2439_autonomous_target_auc": p2439["autonomous"]["margin_target_present_auc"],
                    "phase2439_language_encoding_compiler_closed": p2439["adjudication"]["language_encoding_compiler_closed"]},
        "selection": {**payload.get("selection", {}), "phase2440_case": case["case_id"],
                      "phase2440_config": case["config_id"], "phase2440_best_basic_cell": [18, "query_end"]},
        "claim_boundary": "The client exposes every one of 2560 physical coordinates, including low-RMS coordinates. Fixed-coordinate and event identity improve local update prediction, but Chinese and held-family generalization fail and semantic interactions do not consistently beat lexical controls. The parameter-level readout identity is qualified; the internal-to-output compiler is not closed. No semantic neuron, universal coordinate gear, or cracked language mechanism is claimed."
    })
    save_if_changed(ASSET, payload)
    return {"asset": str(ASSET), "binary": str(BINARY), "schema": payload["schema"],
            "rows": len(rows), "added_rows": len(new_rows), "dimensions": DIM,
            "binary_shape": list(matrix.shape), "json_bytes": ASSET.stat().st_size,
            "binary_bytes": BINARY.stat().st_size, "finite": bool(np.isfinite(matrix).all()),
            "selected_case": case["case_id"]}


def sha256(path: Path, cache: dict) -> dict:
    stat = path.stat(); key = str(path)
    previous = cache.get(key, {})
    if previous.get("bytes") == stat.st_size and previous.get("mtime_ns") == stat.st_mtime_ns:
        return previous
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return {"path": key, "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": digest.hexdigest(),
            "retention": "retain_unique_full_coordinate_field"}


def retention_manifest() -> dict:
    path = OUT / "analysis/raw_retention_manifest.json"
    cache = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    targets = (P2436 / "raw/hypergraph_event_field.float16.npy",
               P2436 / "raw/fresh_valid_prompt_answer_all_token.float16.npy",
               P2437 / "derived/signed_interaction_state.float16.npy",
               P2439 / "derived/logit_lens_coordinate_contribution.float32.npy")
    records = {str(target): sha256(target, cache) for target in targets}
    save(path, records)
    return {"files": len(records), "bytes": sum(record["bytes"] for record in records.values()),
            "gib": sum(record["bytes"] for record in records.values()) / 1024 ** 3,
            "all_sha256": all(len(record["sha256"]) == 64 for record in records.values()),
            "manifest": str(path), "policy": "unique raw retained; failed/duplicate cache eligible for cleanup"}


def frontend_contract() -> dict:
    route = (ROOT / "frontend/src/researchKernel/heatmapResearchRoute.js").read_text(encoding="utf-8-sig")
    hook = (ROOT / "frontend/src/researchKernel/useResearchKernel.js").read_text(encoding="utf-8-sig")
    component = (ROOT / "frontend/src/components/app/ResearchHeatmapRoute.jsx").read_text(encoding="utf-8-sig")
    app = (ROOT / "frontend/src/App.jsx").read_text(encoding="utf-8-sig")
    return {"route_registered": "C32561_LANGUAGE_ENCODING_FIELD_ROUTE" in route and "C32561-C35760" in route,
            "asset_loaded": "setC32561LanguageEncodingField" in hook,
            "full_parameter_axis": "all 2560 Qwen3-4B physical coordinates / parameters" in component,
            "preview_range_updated": "C32561-C35760 embedding, HiddenState" in component,
            "preview_wired": "c32561LanguageEncodingField={realResearchTrace.c32561LanguageEncodingField}" in app,
            "dist_exists": DIST.exists(),
            "dist_newer_than_asset": DIST.exists() and DIST.stat().st_mtime_ns >= ASSET.stat().st_mtime_ns}


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 八族条件轨迹—词嵌入—HiddenState—输出贡献全坐标热力图与原场留存（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 将Phase2435–2439的重要结果追加到现有`semantic_encoding_output_field_heatmap`，每行完整保存Qwen3-4B全部2560固定物理坐标：代表样本q0 embedding activation、多层query/answer HiddenState、语义/词项有符号状态、fresh family更新护照、逐坐标RMS、对角斜率、真实target-minus-foil输出嵌入权重、逐坐标logit贡献和最终输出interaction。不是Top-K可视化。对四个不可替代的大场计算SHA256、字节数与mtime，冻结“唯一原场保留、失败/重复缓存可清理”策略。

$$V_{{r,j}},\ j=1,\ldots,2560;\qquad
\operatorname{{SHA256}}(F_k)=h_k,\quad F_k\in\{{H_{{event}},H_{{token}},I_{{signed}},C_{{output}}\}}.$$

**结果汇总。** 资产 `{json.dumps(result['asset'], ensure_ascii=False)}`；客户端/构建 `{json.dumps(result['frontend'], ensure_ascii=False)}`；留存清单 `{json.dumps(result['retention'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2440_c35441_c35760_trajectory_visualization_retention.py`；final与SHA256清单位于同名结果目录；全坐标资产为`frontend/public/vis_data/research_kernel/c32561_semantic_encoding_output_field.json`及`.float32.npy`；现有路由/加载/3D热力图继续由`frontend/src/researchKernel/heatmapResearchRoute.js`、`useResearchKernel.js`、`frontend/src/components/app/ResearchHeatmapRoute.jsx`和`frontend/src/App.jsx`承载。

**分析与理论进展。** 现在可以逐格对照“输入embedding参数/激活—事件HiddenState—有符号更新护照—拟合局部斜率—输出embedding权重—logit贡献”。客户端明确同时显示阳性和反证：fresh同坐标局部律存在，而跨语言、留family与内部→输出绝对增益失败，避免只展示漂亮纹理。

**问题硬伤与结论。** 热力图把不同量纲并列只用于纹理核查，不能跨行比较幅值；显示代表行不代替全样本裁决。JSON/二进制伴随资产保存全坐标，但4个原场仍大于客户端载荷，故以SHA256留存供复算而非全部浏览器加载。重要结果已经进入客户端，不清理唯一原场；只清理首轮唯一性失败产生的0.01GiB重复缓存。
"""
    with MEMO.open("a", encoding="utf-8", newline="") as stream:
        stream.write(text)


def main() -> None:
    asset = build_asset()
    retention = retention_manifest()
    frontend = frontend_contract()
    checks = {"full_2560": asset["dimensions"] == DIM and asset["binary_shape"] == [asset["rows"], DIM],
              "important_rows_added": asset["added_rows"] >= 20, "finite": asset["finite"],
              "frontend_source": all(frontend[key] for key in ("route_registered", "asset_loaded", "full_parameter_axis",
                                                                 "preview_range_updated", "preview_wired")),
              "frontend_built_after_asset": frontend["dist_newer_than_asset"],
              "raw_hashes": retention["files"] == 4 and retention["all_sha256"],
              "unique_raw_retained": all(Path(record["path"]).exists() for record in
                                         json.loads(Path(retention["manifest"]).read_text(encoding="utf-8")).values())}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "asset": asset, "frontend": frontend,
              "retention": retention, "checks": checks, "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        print(json.dumps(result, ensure_ascii=False, indent=2)); raise RuntimeError(checks)
    save(OUT / "analysis/final.json", result); append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
