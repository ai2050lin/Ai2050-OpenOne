#!/usr/bin/env python3
"""Publish Phase2470-2475 full-coordinate slices and audit retained fields."""
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
OUT = RESULT / "phase2476_c48641_c49280_fullcoordinate_visualization_retention"
PHASE, CAMPAIGN, DIM = 2476, "C48641-C49280", 2560
SOURCE_TAGS = {
    "phase2471_prompt_event_field",
    "phase2471_family_contrast",
    "phase2472_frozen_transport",
    "phase2475_autonomous_event_field",
    "phase2475_success_family_passport",
}


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


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            value.update(block)
    return value.hexdigest()


def row(vector: np.ndarray, label: str, source: str, kind: str, preview: bool = False, **meta: Any) -> dict:
    value = np.asarray(vector, dtype=np.float32).reshape(-1)
    if value.shape != (DIM,) or not np.isfinite(value).all():
        raise RuntimeError((label, value.shape, bool(np.isfinite(value).all())))
    return {
        "label": label,
        "source": source,
        "coordinate_kind": kind,
        "preview": preview,
        **meta,
        "values": [float(x) for x in value],
    }


def prompt_rows() -> list[dict]:
    directory = next(RESULT.glob("phase2471_*"))
    states = np.load(directory / "derived/event_states.float32.npy", mmap_mode="r")
    deltas = np.load(directory / "derived/event_layer_increments.float32.npy", mmap_mode="r")
    contrasts = np.load(directory / "derived/family_contrasts.float32.npy", mmap_mode="r")
    index = read_jsonl(directory / "index/event_rows.jsonl")
    final = json.loads((directory / "analysis/final.json").read_text(encoding="utf-8"))
    families = final["factor_accounting"]["levels"]["family"]
    events = ("statement_target", "candidate_target", "answer_boundary")
    rows: list[dict] = []
    try:
        for interface in ("code", "entity"):
            sample = next(i for i, item in enumerate(index) if item["unit"] == 10 and item["family"] == "taxonomy" and item["language"] == "en" and item["surface"] == 0 and item["output_interface"] == interface and item["parsed_correct"])
            for event, event_name in enumerate(events):
                for qpoint in (0, 10, 11, 25, 37):
                    rows.append(row(
                        states[sample, event, qpoint],
                        f"prompt actual {interface} {event_name} q{qpoint} taxonomy unit10 en",
                        "phase2471_prompt_event_field",
                        "embedding_activation" if qpoint == 0 else "hidden_state",
                        preview=(interface == "entity" and event_name == "answer_boundary" and qpoint in (0, 11, 25)),
                        phase=2471, unit=10, layer=qpoint, event=event_name, interface=interface,
                        language="en", family="taxonomy", coordinate_order="physical",
                        full_tensor="tests/glm5/result/phase2471_c45921_c46480_fullcoordinate_factor_atlas/derived/event_states.float32.npy",
                    ))
                rows.append(row(
                    deltas[sample, event, 10],
                    f"prompt actual {interface} {event_name} q10-to-q11 increment taxonomy unit10 en",
                    "phase2471_prompt_event_field", "layer_increment", False,
                    phase=2471, unit=10, layer_from=10, layer_to=11, event=event_name,
                    interface=interface, language="en", family="taxonomy", coordinate_order="physical",
                    full_tensor="tests/glm5/result/phase2471_c45921_c46480_fullcoordinate_factor_atlas/derived/event_layer_increments.float32.npy",
                ))

        # [unit, language, interface, family, event, qpoint, coordinate].  Language is
        # averaged only for this view; the complete tensor remains retained.
        for interface, interface_name in enumerate(("code", "entity")):
            for family, family_name in enumerate(families):
                for event, event_name in enumerate(events):
                    values = np.mean(contrasts[1, :, interface, family, event, 25], axis=0, dtype=np.float64)
                    rows.append(row(
                        values,
                        f"prompt family contrast {interface_name} {event_name} q25 {family_name} unit10 language-mean",
                        "phase2471_family_contrast", "family_contrast", family_name == "taxonomy" and event_name == "answer_boundary",
                        phase=2471, unit=10, layer=25, event=event_name, interface=interface_name,
                        language="mean(en,zh)", family=family_name, coordinate_order="physical",
                        full_tensor="tests/glm5/result/phase2471_c45921_c46480_fullcoordinate_factor_atlas/derived/family_contrasts.float32.npy",
                    ))
                increment = np.mean(contrasts[1, :, interface, family, 2, 11] - contrasts[1, :, interface, family, 2, 10], axis=0, dtype=np.float64)
                rows.append(row(
                    increment,
                    f"prompt family contrast increment {interface_name} answer_boundary q10-to-q11 {family_name} unit10",
                    "phase2471_family_contrast", "family_contrast_increment", False,
                    phase=2471, unit=10, layer_from=10, layer_to=11, event="answer_boundary",
                    interface=interface_name, language="mean(en,zh)", family=family_name, coordinate_order="physical",
                    full_tensor="tests/glm5/result/phase2471_c45921_c46480_fullcoordinate_factor_atlas/derived/family_contrasts.float32.npy",
                ))
    finally:
        close(states); close(deltas); close(contrasts)
    return rows


def transport_rows() -> list[dict]:
    directory = next(RESULT.glob("phase2472_*"))
    pooled = np.load(directory / "derived/frozen_pooled_diagonal_scale.float32.npy", mmap_mode="r")
    interfaces = np.load(directory / "derived/frozen_interface_diagonal_scale.float32.npy", mmap_mode="r")
    rows: list[dict] = []
    try:
        rows.append(row(
            pooled, "frozen pooled diagonal transport scale q10-to-q11", "phase2472_frozen_transport",
            "fitted_transport_parameter", True, phase=2472, layer_from=10, layer_to=11,
            interface="pooled", fit_split="unit9-en", lockbox="unit10-en/zh",
            full_tensor="tests/glm5/result/phase2472_c46481_c47040_coordinate_transport_competition/derived/frozen_pooled_diagonal_scale.float32.npy",
        ))
        for interface, interface_name in enumerate(("code", "entity")):
            rows.append(row(
                interfaces[interface], f"frozen {interface_name} diagonal transport scale q10-to-q11",
                "phase2472_frozen_transport", "fitted_transport_parameter", False,
                phase=2472, layer_from=10, layer_to=11, interface=interface_name,
                fit_split="unit9-en", lockbox="unit10-en/zh",
                full_tensor="tests/glm5/result/phase2472_c46481_c47040_coordinate_transport_competition/derived/frozen_interface_diagonal_scale.float32.npy",
            ))
    finally:
        close(pooled); close(interfaces)
    return rows


def autonomous_rows() -> list[dict]:
    directory = next(RESULT.glob("phase2475_*"))
    states = np.load(directory / "derived/aligned_autonomous_event_states.float32.npy", mmap_mode="r")
    passports = np.load(directory / "derived/success_family_event_passports.float32.npy", mmap_mode="r")
    index = read_jsonl(directory / "index/aligned_rows.jsonl")
    final = json.loads((directory / "analysis/final.json").read_text(encoding="utf-8"))
    families = final["collection"]["passports"]["families"]
    events = ("answer_boundary", "first_generated_token", "parsed_answer_token")
    selected = final["trajectory"]["discovery_selection"]["crossinterface"]
    rows: list[dict] = []
    try:
        for interface in ("code", "entity"):
            sample = next(i for i, item in enumerate(index) if item["unit"] == 10 and item["family"] == "taxonomy" and item["language"] == "en" and item["surface"] == 0 and item["output_interface"] == interface and item["parsed_correct"])
            for event, event_name in enumerate(events):
                for qpoint in (0, 10, 11, 21, 25, 37):
                    rows.append(row(
                        states[sample, event, qpoint],
                        f"autonomous actual {interface} {event_name} q{qpoint} taxonomy unit10 en",
                        "phase2475_autonomous_event_field",
                        "embedding_activation" if qpoint == 0 else "hidden_state",
                        preview=(interface == "entity" and event_name in ("answer_boundary", "parsed_answer_token") and qpoint in (0, 21)),
                        phase=2475, unit=10, layer=qpoint, event=event_name, interface=interface,
                        language="en", family="taxonomy", autonomous=True, parsed_correct=True,
                        full_tensor="tests/glm5/result/phase2475_c48001_c48640_autonomous_trajectory_adjudication/derived/aligned_autonomous_event_states.float32.npy",
                    ))

        # [unit, language, interface, family, event, qpoint, coordinate].  Publish the
        # frozen unit9-selected qpoint at every event, averaged over language only.
        for interface, interface_name in enumerate(("code", "entity")):
            for family, family_name in enumerate(families):
                for event, event_name in enumerate(events):
                    qpoint = int(selected[event_name])
                    values = np.mean(passports[1, :, interface, family, event, qpoint], axis=0, dtype=np.float64)
                    rows.append(row(
                        values,
                        f"successful autonomous family passport {interface_name} {event_name} q{qpoint} {family_name} unit10",
                        "phase2475_success_family_passport", "successful_autonomous_family_contrast",
                        family_name == "taxonomy",
                        phase=2475, unit=10, layer=qpoint, event=event_name, interface=interface_name,
                        language="mean(en,zh)", family=family_name, autonomous=True, parsed_correct=True,
                        selection="qpoint frozen on unit9; unit10 displayed",
                        full_tensor="tests/glm5/result/phase2475_c48001_c48640_autonomous_trajectory_adjudication/derived/success_family_event_passports.float32.npy",
                    ))
    finally:
        close(states); close(passports)
    return rows


def publish_asset() -> dict:
    payload = json.loads(ASSET.read_text(encoding="utf-8"))
    qwen = next(section for section in payload["models"] if section["key"] == "qwen4b")
    original = [item for item in qwen["rows"] if item.get("source") not in SOURCE_TAGS]
    phase2471 = prompt_rows()
    phase2472 = transport_rows()
    phase2475 = autonomous_rows()
    added = phase2471 + phase2472 + phase2475
    qwen["rows"] = original + added
    fingerprint = np.load(next(RESULT.glob("phase2471_*")) / "derived/discovery_coordinate_fingerprint_order.int32.npy")
    fingerprint = np.asarray(fingerprint, dtype=np.int64).reshape(-1)
    if sorted(fingerprint.tolist()) != list(range(DIM)):
        raise RuntimeError("frozen fingerprint order is not a permutation")
    qwen["coordinate_orders"] = {
        "physical": list(range(DIM)),
        "fingerprint": [int(value) for value in fingerprint],
    }
    qwen["coordinate_order_semantics"] = (
        "physical preserves model coordinate IDs; fingerprint is a unit9 discovery-set response ordering applied without dropping any coordinate and is not claimed to be a natural model basis"
    )
    binary = PUBLIC / "c42641_qwen4b_output_conditioned_field.float32.npy"
    matrix = np.stack([np.asarray(item["values"], dtype=np.float32) for item in qwen["rows"]])
    np.save(binary, matrix)
    binary_hash = digest(binary)
    qwen["binary_shape"] = list(matrix.shape)
    qwen["binary_sha256"] = binary_hash
    payload["phase"] = PHASE
    payload["campaign"] = "C39761-C49280"
    payload["title"] = "Native-coordinate prompt events, block increments, transport scales, and successful autonomous trajectories"
    payload["summary"]["phase2466_state_collapse_replicated"] = False
    payload["summary"]["phase2466_protocol_artifact_corrected"] = True
    payload["summary"]["phase2474_parsed_autonomous_accuracy"] = 0.94921875
    payload["summary"]["phase2475_successful_autonomous_family_texture_present"] = True
    payload["summary"]["phase2475_causal_gear_identified"] = False
    payload["summary"]["coordinate_orders"] = ["physical", "frozen_discovery_response_fingerprint"]
    payload["summary"]["model_rows"] = {section["key"]: len(section["rows"]) for section in payload["models"]}
    payload["summary"]["total_rows"] = sum(payload["summary"]["model_rows"].values())
    payload["claim_boundary"] = (
        "Phase2466 did not measure one autonomous trajectory across the compared events and its entity budget was exhausted by an Answer prefix, so the earlier state-collapse reading is withdrawn. "
        "Phase2474/2475 use actual greedy prefixes, semantic-event alignment, successful-path family passports, full 2560-coordinate slices, and unit9-selected/unit10 lockbox qpoints. "
        "Positive texture retention and frozen diagonal predictability are observational candidates, not sparse gears, causal mechanisms, unique encoding units, or a closed theory. "
        "Fingerprint order is only a full-coordinate display permutation learned on discovery data; cross-model coordinate IDs remain unaligned."
    )
    save_if_changed(ASSET, payload)
    return {
        "asset": str(ASSET), "json_bytes": ASSET.stat().st_size,
        "qwen4b_shape": list(matrix.shape), "qwen4b_sha256": binary_hash,
        "phase2471_rows_added": len(phase2471), "phase2472_rows_added": len(phase2472),
        "phase2475_rows_added": len(phase2475), "total_rows": payload["summary"]["total_rows"],
        "coordinate_orders_are_full_permutations": True,
    }


def retention() -> dict:
    paths = [
        next(RESULT.glob("phase2470_*")) / "raw/prompt_alltoken_allqpoint.float16.npy",
        next(RESULT.glob("phase2471_*")) / "derived/event_states.float32.npy",
        next(RESULT.glob("phase2471_*")) / "derived/event_layer_increments.float32.npy",
        next(RESULT.glob("phase2471_*")) / "derived/family_contrasts.float32.npy",
        next(RESULT.glob("phase2472_*")) / "derived/frozen_pooled_diagonal_scale.float32.npy",
        next(RESULT.glob("phase2472_*")) / "derived/frozen_interface_diagonal_scale.float32.npy",
        next(RESULT.glob("phase2474_*")) / "raw/autonomous_allqpoint_path.float16.npy",
        next(RESULT.glob("phase2475_*")) / "derived/aligned_autonomous_event_states.float32.npy",
        next(RESULT.glob("phase2475_*")) / "derived/success_family_event_passports.float32.npy",
    ]
    records = [{
        "path": str(path), "bytes": path.stat().st_size, "sha256": digest(path),
        "retention": "retained: unique full-coordinate evidence represented by parameter-level client slices",
    } for path in paths]
    save(OUT / "analysis/retention_manifest.json", records)
    return {
        "files": len(records), "bytes": sum(item["bytes"] for item in records),
        "all_hashes": all(len(item["sha256"]) == 64 for item in records),
        "cleanup": "No listed field deleted: each is unique full-coordinate evidence and parameter-level slices are published in the client.",
    }


def frontend() -> dict:
    component_path = ROOT / "frontend/src/components/app/ResearchHeatmapRoute.jsx"
    component = component_path.read_text(encoding="utf-8-sig")
    dist = ROOT / "frontend/dist/index.html"
    return {
        "native_coordinate_panel": "buildC42641CrossmodelFieldData" in component,
        "physical_and_fingerprint_controls": "冻结响应指纹顺序" in component and "coordinateOrderLabel" in component,
        "all_coordinate_mode": "全部参数" in component,
        "dist_exists": dist.exists(),
        "dist_newer_than_sources": dist.exists() and dist.stat().st_mtime_ns >= max(ASSET.stat().st_mtime_ns, component_path.stat().st_mtime_ns),
    }


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""


## Phase {PHASE}: 提示事件—跨层增量—成功自主轨迹的双顺序全坐标热力图与留存审计（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 将Phase2470–2475最重要的Qwen3-4B BF16证据追加到现有原生坐标客户端：（1）unit10 taxonomy真实提示样本在statement-target、candidate-target、answer-boundary三事件的q0 Embedding与q10/q11/q25/q37 HiddenState；（2）q10→q11真实层增量；（3）八个行为合格语言族在实体/代码接口的q25 family contrast及回答边界q10→q11 contrast增量；（4）unit9-en拟合的pooled/interface全2560坐标对角传动尺度；（5）真实成功自主生成的boundary、first-token、parsed-answer三事件激活与unit9选层、unit10锁箱family护照。客户端新增物理顺序和发现集冻结响应指纹顺序；两者均显示全部坐标，后者只重排、不压缩。

$$\pi_{{\mathrm{{fp}}}}=\operatorname{{argsort}}_d\,\Phi_d^{{(u9)}},\qquad \mathcal{{H}}^{{\mathrm{{view}}}}=\left(H_{{\pi(1)}},\ldots,H_{{\pi(2560)}}\right),\quad |\pi|=2560.$$

**结果汇总。** 资产 `{json.dumps(result['asset'], ensure_ascii=False)}`；前端 `{json.dumps(result['frontend'], ensure_ascii=False)}`；留存 `{json.dumps(result['retention'], ensure_ascii=False)}`；裁决 `{json.dumps(result['adjudication'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2476_c48641_c49280_fullcoordinate_visualization_retention.py`；前端`frontend/src/components/app/ResearchHeatmapRoute.jsx`；资产`frontend/public/vis_data/research_kernel/c42641_output_conditioned_crossmodel_field.json`、Qwen4B float32二进制；final与九个原场SHA256留存清单位于同名结果目录。

**分析与理论进展。** 现在可在同一客户端逐参数查看词嵌入、具体层HiddenState、层增量、family相对纹理、对角尺度以及成功生成轨迹，且能在物理坐标与冻结指纹顺序间切换。这让“哪些坐标共同响应、在何事件/层复用或分化”成为可审查对象；指纹顺序的价值是把分散纹理邻接展示，而不是宣称得到新基底。Phase2466的“0.85→0.11自主崩解”已被更正：那不是同一轨迹事件比较，实体预算又常被`Answer:`前缀耗尽；Phase2474/2475的真实路径没有复现该崩解。

**问题硬伤与结论。** 客户端只发布完整张量的有原则切片，不能替代磁盘上的全事件、全层统计；语言平均行可能隐藏语言差异，故原张量继续保留。冻结排序仍由unit9响应定义，不是模型自然坐标拓扑。成功路径纹理与有限对角可预测性均未证明因果齿轮、最小编码单位或机制闭合。九个独特全坐标场已有参数级可视化切片并完成SHA256审计，因此不清理；清理会不可逆地丢失目前最重要的低值坐标证据。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def main() -> None:
    asset = publish_asset()
    kept = retention()
    front = frontend()
    checks = {
        "phase2471_rows": asset["phase2471_rows_added"] == 100,
        "phase2472_rows": asset["phase2472_rows_added"] == 3,
        "phase2475_rows": asset["phase2475_rows_added"] == 84,
        "native_qwen_shape": asset["qwen4b_shape"] == [581, 2560],
        "binary_hash": len(asset["qwen4b_sha256"]) == 64,
        "full_coordinate_permutations": asset["coordinate_orders_are_full_permutations"],
        "frontend_source": all(front[key] for key in ("native_coordinate_panel", "physical_and_fingerprint_controls", "all_coordinate_mode")),
        "frontend_built": front["dist_newer_than_sources"],
        "retained": kept["files"] == 9 and kept["all_hashes"],
    }
    result = {
        "phase": PHASE, "campaign": CAMPAIGN, "asset": asset, "frontend": front, "retention": kept,
        "adjudication": {
            "phase2466_state_collapse_replicated": False,
            "important_fields_visible_at_parameter_level": True,
            "hiddenstate_cleanup_required": False,
            "frozen_fingerprint_is_model_natural_basis": False,
            "causal_coordinate_gear_identified": False,
            "language_encoding_mechanism_closed": False,
        },
        "checks": checks, "all_checks_passed": all(checks.values()),
    }
    save(OUT / "analysis" / ("final.json" if result["all_checks_passed"] else "prebuild.json"), result)
    if result["all_checks_passed"]:
        append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
