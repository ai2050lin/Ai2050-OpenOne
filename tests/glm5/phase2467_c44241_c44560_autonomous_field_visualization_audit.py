#!/usr/bin/env python3
"""Extend the native-coordinate client field with Phase2465/2466 evidence."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result"
PUBLIC = ROOT / "frontend/public/vis_data/research_kernel"
ASSET = PUBLIC / "c42641_output_conditioned_crossmodel_field.json"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
OUT = RESULT / "phase2467_c44241_c44560_autonomous_field_visualization_audit"
PHASE, CAMPAIGN, DIM = 2467, "C44241-C44560", 2560
SOURCE_TAGS = {"phase2465_behavior_gated_output_identity", "phase2466_autonomous_path"}


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def save_if_changed(path: Path, value: Any) -> None:
    content = json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n"
    if not path.exists() or path.read_text(encoding="utf-8") != content:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def close(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def row(vector: np.ndarray, label: str, source: str, kind: str, preview: bool = False, **meta: Any) -> dict:
    value = np.asarray(vector, dtype=np.float32).reshape(-1)
    if value.shape != (DIM,) or not np.isfinite(value).all():
        raise RuntimeError((label, value.shape))
    return {
        "label": label,
        "source": source,
        "coordinate_kind": kind,
        "preview": preview,
        **meta,
        "values": [float(x) for x in value],
    }


def phase2465_rows() -> list[dict]:
    directory = next(RESULT.glob("phase2465_*"))
    final = json.loads((directory / "analysis/final.json").read_text(encoding="utf-8"))
    families = final["analysis"]["families"]
    passports = np.load(directory / "derived/entity_code_passports.float32.npy", mmap_mode="r")
    rows: list[dict] = []
    try:
        # [interaction, interface, unit, language, qpoint, field, family, coordinate]
        # Publish q16 HxVJP for both semantic and lexical controls. The complete q16/q18
        # tensor remains retained on disk and is linked through source metadata.
        for interaction, interaction_name in enumerate(("semantic_validity", "lexical_control")):
            for interface, interface_name in enumerate(("candidate_entity", "letter_code")):
                for language, language_name in enumerate(("en", "zh")):
                    for family, family_name in enumerate(families):
                        rows.append(row(
                            passports[interaction, interface, 1, language, 0, 1, family],
                            f"{interaction_name} HxVJP {interface_name} unit7 q16 {language_name} {family_name}",
                            "phase2465_behavior_gated_output_identity",
                            "state_times_gradient",
                            preview=(interaction == 0 and interface == 1 and language_name == "en" and family_name == "taxonomy"),
                            phase=2465,
                            unit=7,
                            layer=16,
                            event="prompt_query_end",
                            interaction=interaction_name,
                            interface=interface_name,
                            language=language_name,
                            family=family_name,
                            full_tensor="tests/glm5/result/phase2465_c43601_c43920_behavior_gated_output_identity_vjp/derived/entity_code_passports.float32.npy",
                        ))
    finally:
        close(passports)
    return rows


def phase2466_rows() -> list[dict]:
    directory = next(RESULT.glob("phase2466_*"))
    final = json.loads((directory / "analysis/final.json").read_text(encoding="utf-8"))
    families = final["analysis"]["families"]
    passports = np.load(directory / "derived/autonomous_path_passports.float32.npy", mmap_mode="r")
    fields = np.load(directory / "raw/autonomous_path_states.float16.npy", mmap_mode="r")
    index = read_jsonl(directory / "index/autonomous_rows.jsonl")
    rows: list[dict] = []
    events = ("prompt_query_end", "prompt_answer_boundary", "generated_token1")
    qpoints = (0, 16, 18, 37)
    try:
        # Full-family q18 passports expose the transition without reducing coordinates.
        # Both semantic and lexical controls are required because Phase2466 is not
        # semantic-specific.
        for interaction, interaction_name in enumerate(("semantic_validity", "lexical_control")):
            for interface, interface_name in enumerate(("candidate_entity", "letter_code")):
                for event, event_name in enumerate(events):
                    for language, language_name in enumerate(("en", "zh")):
                        for family, family_name in enumerate(families):
                            rows.append(row(
                                passports[interaction, interface, event, 2, language, family],
                                f"{interaction_name} autonomous {interface_name} {event_name} q18 {language_name} {family_name}",
                                "phase2466_autonomous_path",
                                "hidden_state_interaction",
                                preview=(interaction == 0 and interface == 1 and event_name == "generated_token1" and language_name == "en" and family_name == "taxonomy"),
                                phase=2466,
                                layer=18,
                                event=event_name,
                                interaction=interaction_name,
                                interface=interface_name,
                                language=language_name,
                                family=family_name,
                                autonomous=True,
                                full_tensor="tests/glm5/result/phase2466_c43921_c44240_autonomous_generated_path_fullfield/derived/autonomous_path_passports.float32.npy",
                            ))

        # Real activation examples retain q0/q16/q18/q37 for every measured event.
        # Select one predeclared taxonomy/en/valid/target row per interface only as a
        # client preview; the complete 192-row tensor remains retained and hashed.
        for interface_name in ("candidate_entity", "letter_code"):
            sample = next(i for i, item in enumerate(index) if item["interface"] == interface_name and item["family"] == "taxonomy" and item["language"] == "en" and item["variant"] == "valid" and item["query_role"] == "target")
            for event, event_name in enumerate(events):
                for qpoint, qpoint_name in enumerate(qpoints):
                    kind = "embedding_activation" if qpoint_name == 0 else "hidden_state"
                    rows.append(row(
                        fields[sample, event, qpoint],
                        f"actual {interface_name} {event_name} q{qpoint_name} taxonomy en",
                        "phase2466_autonomous_path",
                        kind,
                        preview=(interface_name == "letter_code" and event_name == "generated_token1" and qpoint_name == 18),
                        phase=2466,
                        layer=qpoint_name,
                        event=event_name,
                        interaction="raw_activation",
                        interface=interface_name,
                        language="en",
                        family="taxonomy",
                        autonomous=True,
                        full_tensor="tests/glm5/result/phase2466_c43921_c44240_autonomous_generated_path_fullfield/raw/autonomous_path_states.float16.npy",
                    ))
    finally:
        close(passports)
        close(fields)
    return rows


def publish_asset() -> dict:
    payload = json.loads(ASSET.read_text(encoding="utf-8"))
    qwen = next(section for section in payload["models"] if section["key"] == "qwen4b")
    original = [item for item in qwen["rows"] if item.get("source") not in SOURCE_TAGS]
    added_2465 = phase2465_rows()
    added_2466 = phase2466_rows()
    qwen["rows"] = original + added_2465 + added_2466
    binary = PUBLIC / "c42641_qwen4b_output_conditioned_field.float32.npy"
    matrix = np.stack([np.asarray(item["values"], dtype=np.float32) for item in qwen["rows"]])
    np.save(binary, matrix)
    digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    qwen["binary_shape"] = list(matrix.shape)
    qwen["binary_sha256"] = digest
    payload["phase"] = PHASE
    payload["campaign"] = "C39761-C44560"
    payload["title"] = "Cross-model native-coordinate output fields, interfaces, and autonomous generated path"
    payload["summary"]["phase2465_partial_output_identity_geometry"] = True
    payload["summary"]["phase2465_semantic_specific_output_identity"] = False
    payload["summary"]["phase2466_entity_autonomous_exact_rate"] = 0.0
    payload["summary"]["phase2466_code_autonomous_exact_rate"] = 0.40625
    payload["summary"]["phase2466_semantic_specific_autonomous_path"] = False
    payload["summary"]["model_rows"] = {section["key"]: len(section["rows"]) for section in payload["models"]}
    payload["summary"]["total_rows"] = sum(payload["summary"]["model_rows"].values())
    payload["claim_boundary"] = (
        "All rows retain complete native coordinates, but the client publishes only selected slices of the retained full tensors. "
        "Phase2466 uses actual greedy prefixes; candidate-entity exact rate is 0 and letter-code exact rate is 0.40625. "
        "Generated-path coordinate similarity is observational and is also present in lexical controls, so it is not a semantic-specific gear or a closed mechanism. "
        "Cross-architecture coordinate IDs and quantized amplitudes are never aligned or compared."
    )
    save_if_changed(ASSET, payload)
    return {
        "asset": str(ASSET),
        "json_bytes": ASSET.stat().st_size,
        "qwen4b_shape": list(matrix.shape),
        "qwen4b_sha256": digest,
        "phase2465_rows_added": len(added_2465),
        "phase2466_rows_added": len(added_2466),
        "total_rows": payload["summary"]["total_rows"],
    }


def retention() -> dict:
    targets = []
    for phase, names in (
        (2465, ("raw/entity_code_fields.float32.npy", "derived/entity_code_passports.float32.npy")),
        (2466, ("raw/autonomous_path_states.float16.npy", "derived/autonomous_path_passports.float32.npy")),
    ):
        directory = next(RESULT.glob(f"phase{phase}_*"))
        targets.extend(directory / name for name in names)
    records = []
    for path in targets:
        records.append({
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "retention": "retained: complete native-coordinate evidence; selected slices are published in the client asset",
        })
    save(OUT / "analysis/retention_manifest.json", records)
    return {
        "files": len(records),
        "bytes": sum(item["bytes"] for item in records),
        "all_hashes": all(len(item["sha256"]) == 64 for item in records),
        "cleanup": "No Phase2465/2466 field deleted because each is unique full-coordinate evidence represented by client slices.",
    }


def frontend() -> dict:
    component = (ROOT / "frontend/src/components/app/ResearchHeatmapRoute.jsx").read_text(encoding="utf-8-sig")
    dist = ROOT / "frontend/dist/index.html"
    return {
        "native_coordinate_panel": "buildC42641CrossmodelFieldData" in component and "panel.coordinateCount" in component,
        "dist_exists": dist.exists(),
        "dist_newer_than_asset": dist.exists() and dist.stat().st_mtime_ns >= ASSET.stat().st_mtime_ns,
    }


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""


## Phase {PHASE}: 行为门控VJP与自主生成路径的原生坐标热力图发布及留存审计（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 在Phase2462四模型原生坐标热力图中追加两类不能只保留均值的重要证据：Phase2465的unit7实体/代码接口语义与词项$H\odot g$完整q16坐标，以及Phase2466实体/代码实际贪心路径在prompt-query-end、answer-boundary、generated-token1三事件的q18语义/词项interaction。另为每个接口加入taxonomy/en/valid/target代表样本在三事件、q0/q16/q18/q37的真实Embedding与HiddenState全坐标行。客户端切片不压缩坐标，完整张量继续留存在结果目录并做SHA256审计。

$$\mathcal{{V}}=\operatorname{{span}}\left(H_q^{{event}}, I_{{sem,q18}}^{{interface,event}}, I_{{lex,q18}}^{{interface,event}}, (H\odot g)_{{q16}}^{{interface}}\right)\subset\mathbb{{R}}^{{2560}}.$$

**结果汇总。** 发布 `{json.dumps(result['asset'], ensure_ascii=False)}`；前端 `{json.dumps(result['frontend'], ensure_ascii=False)}`；留存 `{json.dumps(result['retention'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2467_c44241_c44560_autonomous_field_visualization_audit.py`；扩展资产`frontend/public/vis_data/research_kernel/c42641_output_conditioned_crossmodel_field.json`及Qwen4B float32二进制；final、原场SHA256清单位于同名结果目录。

**分析与理论进展。** 热力图现在可以逐坐标并列三种层次：静态输出条件VJP、真实生成前后的HiddenState、以及语义/词项对照的接口interaction。最值得保留的图谱不是“生成后仍有相关”这个均值，而是相关结构如何从query-end约0.85降到generated-token1约0.11，以及词项对照也出现相同量级的跨接口纹理。它把“输入共同模板”“输出接口”和“实际生成前缀”三者的分化位置直接暴露出来。

**问题硬伤与结论。** 热力图的q16/q18切片不是完整统计替代品；完整张量仍需由final中的全部qpoint裁决。Phase2466实体自主命中为0、字母代码仅0.40625，因而生成后纹理不能被命名为成功语言执行齿轮。Phase2465语义坐标复用不超过词项对照，Phase2466同样不具语义专属性。四个新全场均是独特证据并已通过客户端切片表示，故不删除；没有把跨模型坐标强行对齐。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def main() -> None:
    asset = publish_asset()
    kept = retention()
    front = frontend()
    checks = {
        "phase2465_rows_64": asset["phase2465_rows_added"] == 64,
        "phase2466_rows_216": asset["phase2466_rows_added"] == 216,
        "native_qwen_shape": asset["qwen4b_shape"] == [394, 2560],
        "binary_hash": len(asset["qwen4b_sha256"]) == 64,
        "frontend_source": front["native_coordinate_panel"],
        "frontend_built": front["dist_newer_than_asset"],
        "retained": kept["files"] == 4 and kept["all_hashes"],
    }
    result = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "asset": asset,
        "frontend": front,
        "retention": kept,
        "adjudication": {
            "important_fields_visible_at_parameter_level": True,
            "hiddenstate_cleanup_required": False,
            "reason": "unique full fields retained and represented by uncompressed-coordinate client slices",
            "language_encoding_mechanism_closed": False,
        },
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }
    destination = OUT / "analysis" / ("final.json" if result["all_checks_passed"] else "prebuild.json")
    save(destination, result)
    if result["all_checks_passed"]:
        append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
