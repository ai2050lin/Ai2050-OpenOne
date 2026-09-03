#!/usr/bin/env python3
"""Publish chain node/trajectory fields and retain unique full-coordinate evidence."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result"
P2480 = RESULT / "phase2480_c51201_c51840_qualified_chain_fullcoordinate_field"
P2481 = RESULT / "phase2481_c51841_c52480_chain_node_edge_basic_atlas"
PUBLIC = ROOT / "frontend/public/vis_data/research_kernel"
ASSET = PUBLIC / "c42641_output_conditioned_crossmodel_field.json"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
OUT = RESULT / "phase2482_c52481_c53120_chain_visualization_retention_audit"
PHASE, CAMPAIGN, DIM = 2482, "C52481-C53120", 2560
SOURCE_TAGS = {"phase2481_chain_actual_nodes", "phase2481_chain_family_passport", "phase2481_chain_main_distractor"}


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def save_if_changed(path: Path, value: Any) -> None:
    content = json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n"
    if not path.exists() or path.read_text(encoding="utf-8") != content:
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text(content, encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def close(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None: mmap.close()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024): value.update(block)
    return value.hexdigest()


def row(vector: np.ndarray, label: str, source: str, kind: str, preview: bool = False, **meta: Any) -> dict:
    value = np.asarray(vector, dtype=np.float32).reshape(-1)
    if value.shape != (DIM,) or not np.isfinite(value).all(): raise RuntimeError((label, value.shape, bool(np.isfinite(value).all())))
    return {"label": label, "source": source, "coordinate_kind": kind, "preview": preview, **meta, "values": [float(x) for x in value]}


def chain_rows() -> list[dict]:
    final = json.loads((P2481 / "analysis/final.json").read_text(encoding="utf-8"))
    index = read_jsonl(P2481 / "index/node_event_rows.jsonl")
    prompt = np.load(final["collection"]["prompt_nodes"]["path"], mmap_mode="r")
    generated = np.load(final["collection"]["generated_nodes"]["path"], mmap_mode="r")
    prompt_pass = np.load(final["collection"]["passports"]["prompt_passports"], mmap_mode="r")
    generated_pass = np.load(final["collection"]["passports"]["generated_passports"], mmap_mode="r")
    paired = np.load(final["collection"]["passports"]["main_minus_distractor"], mmap_mode="r")
    families = ("causal", "handoff", "part_whole")
    rows: list[dict] = []
    try:
        sample = next(i for i, item in enumerate(index) if item["unit"] == 13 and item["family"] == "causal" and item["language"] == "en" and item["surface"] == 0 and item["answer_step"] is not None)
        for path_index, path_name in enumerate(("main", "distractor")):
            for node in range(4):
                for qpoint in (0, 4, 23, 36, 37):
                    rows.append(row(
                        prompt[sample, path_index, node, qpoint],
                        f"chain actual prompt {path_name} node{node} q{qpoint} causal unit13 en",
                        "phase2481_chain_actual_nodes", "embedding_activation" if qpoint == 0 else "hidden_state",
                        preview=path_name == "main" and node in (0, 3) and qpoint in (0, 4, 23),
                        phase=2481, unit=13, layer=qpoint, event=f"prompt_{path_name}_node{node}", family="causal", language="en",
                        full_tensor="tests/glm5/result/phase2481_c51841_c52480_chain_node_edge_basic_atlas/derived/prompt_main_distractor_node_states.float32.npy",
                    ))
        for node in range(4):
            for qpoint in (0, 1, 4, 36, 37):
                rows.append(row(
                    generated[sample, node, qpoint],
                    f"chain actual generated node{node} q{qpoint} causal unit13 en",
                    "phase2481_chain_actual_nodes", "embedding_activation" if qpoint == 0 else "hidden_state",
                    preview=node in (0, 3) and qpoint in (1, 4),
                    phase=2481, unit=13, layer=qpoint, event=f"generated_main_node{node}", family="causal", language="en", autonomous=True,
                    full_tensor="tests/glm5/result/phase2481_c51841_c52480_chain_node_edge_basic_atlas/derived/generated_main_node_states.float32.npy",
                ))
        for language, language_name in enumerate(("en", "zh")):
            for family, family_name in enumerate(families):
                for node in range(4):
                    rows.append(row(
                        prompt_pass[2, language, family, node, 4],
                        f"chain prompt family passport {family_name} node{node} q4 unit13 {language_name}",
                        "phase2481_chain_family_passport", "prompt_family_node_contrast",
                        preview=family_name == "causal" and node in (0, 3),
                        phase=2481, unit=13, layer=4, event=f"prompt_main_node{node}", family=family_name, language=language_name,
                        selection="q4 selected on unit12 for crosslanguage; unit13 lockbox displayed",
                        full_tensor="tests/glm5/result/phase2481_c51841_c52480_chain_node_edge_basic_atlas/derived/prompt_family_node_passports.float32.npy",
                    ))
                    rows.append(row(
                        generated_pass[2, language, family, node, 1],
                        f"chain generated family passport {family_name} node{node} q1 unit13 {language_name}",
                        "phase2481_chain_family_passport", "generated_family_node_contrast",
                        preview=family_name == "causal" and node in (0, 3),
                        phase=2481, unit=13, layer=1, event=f"generated_main_node{node}", family=family_name, language=language_name, autonomous=True,
                        selection="q1 selected on unit12 for prompt-to-generation; unit13 lockbox displayed",
                        full_tensor="tests/glm5/result/phase2481_c51841_c52480_chain_node_edge_basic_atlas/derived/generated_family_node_passports.float32.npy",
                    ))
                    rows.append(row(
                        paired[2, language, family, node, 23],
                        f"chain main-minus-distractor {family_name} node{node} q23 unit13 {language_name}",
                        "phase2481_chain_main_distractor", "main_minus_distractor_node_contrast",
                        preview=family_name == "causal" and node in (0, 3),
                        phase=2481, unit=13, layer=23, event=f"prompt_node{node}", family=family_name, language=language_name,
                        selection="q23 selected on unit12; unit13 lockbox displayed",
                        full_tensor="tests/glm5/result/phase2481_c51841_c52480_chain_node_edge_basic_atlas/derived/main_minus_distractor_node_contrasts.float32.npy",
                    ))
    finally:
        for value in (prompt, generated, prompt_pass, generated_pass, paired): close(value)
    return rows


def publish_asset() -> dict:
    payload = json.loads(ASSET.read_text(encoding="utf-8"))
    qwen = next(section for section in payload["models"] if section["key"] == "qwen4b")
    original = [item for item in qwen["rows"] if item.get("source") not in SOURCE_TAGS]
    added = chain_rows(); qwen["rows"] = original + added
    order = np.load(P2481 / "derived/chain_discovery_coordinate_order.int32.npy")
    if sorted(order.tolist()) != list(range(DIM)): raise RuntimeError("chain order is not a full permutation")
    qwen.setdefault("coordinate_orders", {})["chain_fingerprint"] = [int(value) for value in order]
    qwen["coordinate_order_semantics"] = (
        "physical preserves model IDs; fingerprint orders Phase2471 unit9 responses; chain_fingerprint orders Phase2481 unit12 chain-family energy; both are complete permutations and neither is claimed natural"
    )
    binary = PUBLIC / "c42641_qwen4b_output_conditioned_field.float32.npy"
    matrix = np.stack([np.asarray(item["values"], dtype=np.float32) for item in qwen["rows"]]); np.save(binary, matrix)
    qwen["binary_shape"] = list(matrix.shape); qwen["binary_sha256"] = digest(binary)
    payload["phase"] = PHASE; payload["campaign"] = "C39761-C53120"
    payload["title"] = "Native-coordinate language-family, autonomous-output, and multi-node knowledge-chain fields"
    payload["summary"]["phase2480_chain_behavior_accuracy"] = 0.7916666666666666
    payload["summary"]["phase2481_three_family_chain_texture_candidate"] = True
    payload["summary"]["phase2481_chain_mechanism_closed"] = False
    payload["summary"]["coordinate_orders"] = ["physical", "frozen_discovery_response_fingerprint", "frozen_knowledge_chain_fingerprint"]
    payload["summary"]["model_rows"] = {section["key"]: len(section["rows"]) for section in payload["models"]}
    payload["summary"]["total_rows"] = sum(payload["summary"]["model_rows"].values())
    payload["claim_boundary"] = (
        "Phase2481 publishes three-family, full-coordinate node/passport slices from behavior-qualified explicit chains. "
        "Unit12 selects qpoints and unit13 is displayed as lockbox, but only three family labels and two derangements exist; relation wording, node identity, externalized path memory, and unequal surface counts remain confounds. "
        "All coordinate orders are display permutations without coordinate deletion; no sparse gear, natural basis, typed internal graph, causal compiler, or closed language mechanism is claimed."
    )
    save_if_changed(ASSET, payload)
    return {"asset": str(ASSET), "json_bytes": ASSET.stat().st_size, "rows_added": len(added), "qwen_shape": list(matrix.shape), "sha256": qwen["binary_sha256"], "total_rows": payload["summary"]["total_rows"]}


def retention() -> dict:
    final2480 = json.loads((P2480 / "analysis/final.json").read_text(encoding="utf-8"))
    final2481 = json.loads((P2481 / "analysis/final.json").read_text(encoding="utf-8"))
    paths = [Path(final2480["collection"][name]["path"]) for name in ("prompt_field", "trajectory_field")]
    paths += [Path(final2481["collection"]["prompt_nodes"]["path"]), Path(final2481["collection"]["generated_nodes"]["path"])]
    paths += [Path(final2481["collection"]["passports"][name]) for name in ("prompt_passports", "generated_passports", "main_minus_distractor")]
    records = [{"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path), "retention": "retained: unique all-coordinate chain evidence with parameter-level client slices"} for path in paths]
    save(OUT / "analysis/retention_manifest.json", records)
    return {"files": len(records), "bytes": sum(row["bytes"] for row in records), "all_hashes": all(len(row["sha256"]) == 64 for row in records), "cleanup": "No unique Phase2480/2481 full field deleted."}


def frontend() -> dict:
    source_path = ROOT / "frontend/src/components/app/ResearchHeatmapRoute.jsx"
    source = source_path.read_text(encoding="utf-8-sig"); dist = ROOT / "frontend/dist/index.html"
    return {
        "chain_order_control": "知识链冻结指纹顺序" in source and "chain_fingerprint" in source,
        "full_coordinate_panel": "buildC42641CrossmodelFieldData" in source,
        "dist_newer": dist.exists() and dist.stat().st_mtime_ns >= max(source_path.stat().st_mtime_ns, ASSET.stat().st_mtime_ns),
    }


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"): return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""


## Phase {PHASE}: 知识链节点/生成/干扰全坐标热力图、链指纹顺序与留存审计（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 在Phase2476原生坐标客户端追加Phase2481三层证据：unit13 causal代表样本主链/干扰链四节点q0/q4/q23/q36/q37真实激活；同一样本四个实际生成节点q0/q1/q4/q36/q37；三family、双语言、四节点的q4 prompt family passport、q1 generated passport及q23 main-minus-distractor。新增unit12链family能量冻结顺序，仍保留全部2560坐标且可随时切回物理顺序。Phase2480/2481七个唯一原场逐文件SHA256审计。

$$\Pi_{{chain}}=\operatorname{{argsort}}_d\sum_{{\lambda,f,k}}(P_{{\lambda f k d}}^{{u12,q4}})^2,\qquad |\Pi_{{chain}}|=2560.$$

**结果汇总。** 资产 `{json.dumps(result['asset'], ensure_ascii=False)}`；前端 `{json.dumps(result['frontend'], ensure_ascii=False)}`；留存 `{json.dumps(result['retention'], ensure_ascii=False)}`；裁决 `{json.dumps(result['adjudication'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2482_c52481_c53120_chain_visualization_retention_audit.py`；前端`frontend/src/components/app/ResearchHeatmapRoute.jsx`；扩展后的c42641 JSON/float32资产及本Phase留存清单/final。

**分析与理论进展。** 客户端现在可把同一组三跳链按物理坐标、旧语言族指纹和新知识链指纹三种完整顺序观察，并逐参数并列输入节点、干扰节点、回答路径节点与family相对纹理。链指纹把分布式响应邻接显示，但不改变数值、不舍弃低值坐标，也不被命名为模型天然基底。

**问题硬伤与结论。** 只发布了完整原场的原则性切片；统计裁决仍以磁盘张量和final为准。三family和模板混杂使高余弦不能升级为普遍知识编码。七个场均为唯一长链全坐标证据且已在客户端显示参数级切片，因此保留，不清理。客户端生产构建通过后本续阶段完成，但语言机制仍未闭合。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle: handle.write(text)


def main() -> None:
    asset = publish_asset(); kept = retention(); front = frontend()
    checks = {
        "rows_added_132": asset["rows_added"] == 132,
        "qwen_shape": asset["qwen_shape"] == [713, 2560],
        "hash": len(asset["sha256"]) == 64,
        "frontend_source": front["chain_order_control"] and front["full_coordinate_panel"],
        "frontend_built": front["dist_newer"],
        "retained": kept["files"] == 7 and kept["all_hashes"],
    }
    result = {
        "phase": PHASE, "campaign": CAMPAIGN, "asset": asset, "frontend": front, "retention": kept,
        "adjudication": {"chain_fields_visible_at_parameter_level": True, "hiddenstate_cleanup_required": False, "natural_chain_coordinate_basis_identified": False, "language_encoding_mechanism_closed": False},
        "checks": checks, "all_checks_passed": all(checks.values()),
    }
    save(OUT / "analysis" / ("final.json" if result["all_checks_passed"] else "prebuild.json"), result)
    if result["all_checks_passed"]: append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
