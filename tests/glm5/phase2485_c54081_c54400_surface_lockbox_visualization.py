#!/usr/bin/env python3
"""Publish the causal surface lockbox and retain its full-coordinate fields."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result"
P2484 = RESULT / "phase2484_c53441_c54080_causal_surface_lockbox"
PUBLIC = ROOT / "frontend/public/vis_data/research_kernel"
ASSET = PUBLIC / "c42641_output_conditioned_crossmodel_field.json"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
OUT = RESULT / "phase2485_c54081_c54400_surface_lockbox_visualization"
PHASE, CAMPAIGN, DIM = 2485, "C54081-C54400", 2560
SOURCE = "phase2484_causal_surface_lockbox"


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024): value.update(block)
    return value.hexdigest()


def row(vector: np.ndarray, label: str, kind: str, **meta: Any) -> dict:
    value = np.asarray(vector, dtype=np.float32).reshape(-1)
    if value.shape != (DIM,) or not np.isfinite(value).all(): raise RuntimeError(label)
    return {"label": label, "source": SOURCE, "coordinate_kind": kind, "preview": True, **meta, "values": [float(x) for x in value]}


def publish() -> dict:
    final = json.loads((P2484 / "analysis/final.json").read_text(encoding="utf-8"))
    selection = final["analysis"]["unit12_selection"]
    arrays = {name: np.load(path, mmap_mode="r") for name, path in final["collection"]["fields"].items()}
    added = []
    try:
        selection_key = {"prompt": "prompt_main", "main_minus_distractor": "main_minus_distractor", "generated": "generated_main"}
        for metric, values in arrays.items():
            qpoint = int(selection[selection_key[metric]])
            for surface_index, surface in enumerate((0, 2)):
                for node in range(4):
                    added.append(row(
                        values[2, :, surface_index, node, qpoint].mean(axis=0, dtype=np.float64),
                        f"causal surface{surface} {metric} node{node} q{qpoint} unit13 language-mean",
                        "causal_surface_node_field", phase=2484, unit=13, layer=qpoint,
                        event=f"{metric}_node{node}", family="causal", surface=surface, language="mean(en,zh)",
                        selection="qpoint selected on unit12; unit13 lockbox displayed",
                        full_tensor=final["collection"]["fields"][metric],
                    ))
    finally:
        for value in arrays.values():
            mmap = getattr(value, "_mmap", None)
            if mmap is not None: mmap.close()
    payload = json.loads(ASSET.read_text(encoding="utf-8")); qwen = next(section for section in payload["models"] if section["key"] == "qwen4b")
    qwen["rows"] = [value for value in qwen["rows"] if value.get("source") != SOURCE] + added
    matrix = np.stack([np.asarray(value["values"], dtype=np.float32) for value in qwen["rows"]])
    binary = PUBLIC / "c42641_qwen4b_output_conditioned_field.float32.npy"; np.save(binary, matrix)
    qwen["binary_shape"] = list(matrix.shape); qwen["binary_sha256"] = digest(binary)
    payload["phase"] = PHASE; payload["campaign"] = "C39761-C54400"
    payload["summary"]["phase2484_causal_surface_same_node_reuse_candidate"] = True
    payload["summary"]["phase2484_energy_envelope_is_signed_code"] = False
    payload["summary"]["model_rows"] = {section["key"]: len(section["rows"]) for section in payload["models"]}
    payload["summary"]["total_rows"] = sum(payload["summary"]["model_rows"].values())
    sentence = "Phase2484 surface energy correlation is an unsigned coordinate-scale envelope; signed main-minus-distractor cosine is much lower and only one causal family is tested."
    boundary = payload["claim_boundary"].replace(" " + sentence, "").rstrip()
    payload["claim_boundary"] = boundary + " " + sentence
    content = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if ASSET.read_text(encoding="utf-8") != content:
        ASSET.write_text(content, encoding="utf-8")
    return {"asset": str(ASSET), "rows_added": len(added), "qwen_shape": list(matrix.shape), "sha256": qwen["binary_sha256"], "json_bytes": ASSET.stat().st_size}


def retention() -> dict:
    final = json.loads((P2484 / "analysis/final.json").read_text(encoding="utf-8")); records = []
    for path_text in final["collection"]["fields"].values():
        path = Path(path_text); records.append({"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path), "retention": "retained: important signed surface lockbox displayed at parameter level"})
    save(OUT / "analysis/retention_manifest.json", records)
    return {"files": len(records), "bytes": sum(value["bytes"] for value in records), "all_hashes": all(len(value["sha256"]) == 64 for value in records), "cleanup": "No Phase2484 field deleted because all three have parameter-level client slices."}


def frontend() -> dict:
    dist = ROOT / "frontend/dist/index.html"
    return {"dist_exists": dist.exists(), "dist_newer": dist.exists() and dist.stat().st_mtime_ns >= ASSET.stat().st_mtime_ns}


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"): return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""


## Phase {PHASE}: causal双表面有符号节点场的参数级发布与能量包络边界审计（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** Phase2484得到几乎为1的逐坐标平方能量相关，但unit13有符号main-minus-distractor同node余弦仅0.2733、胜错位node 0.2009。为避免只展示能量造成“已找到稳定编码”的错觉，客户端同时发布unit13、surface0/2、四节点的prompt-main(q16)、main-minus-distractor(q21)和generated-main(q16)语言均值全2560坐标，共24行；保留物理/旧指纹/链指纹三种完整顺序。三个原场做SHA256留存。

$$\rho(E_{{s0}},E_{{s2}})\approx1\;\not\Rightarrow\;\cos(D_{{s0}},D_{{s2}})\approx1,\qquad E_d=\sum D_d^2.$$

**结果汇总。** 资产 `{json.dumps(result['asset'], ensure_ascii=False)}`；前端 `{json.dumps(result['frontend'], ensure_ascii=False)}`；留存 `{json.dumps(result['retention'], ensure_ascii=False)}`；裁决 `{json.dumps(result['adjudication'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2485_c54081_c54400_surface_lockbox_visualization.py`；扩展c42641客户端资产、三个Phase2484全场及本Phase final/留存清单。

**分析与理论进展。** 新视图允许直接区分坐标幅值包络和带符号路径纹理。几乎一致的平方能量更可能说明q21存在稳定各向异性坐标尺度；真正携带相对方向的信息只表现为中等有符号余弦。这一修正阻止把“哪些坐标通常幅值大”误称为“知识链编码相同”。

**问题硬伤与结论。** 仍只有causal一个family和两个surface；语言均值视图不能替代原场。结果重要且已显示到参数级客户端，故不清理。它支持表面间存在部分同node/查询路径纹理，但不支持普遍链齿轮、天然基底或机制闭合。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle: handle.write(text)


def main() -> None:
    asset = publish(); kept = retention(); front = frontend()
    checks = {"rows24": asset["rows_added"] == 24, "qwen_shape": asset["qwen_shape"] == [737, 2560], "hash": len(asset["sha256"]) == 64, "frontend_built": front["dist_newer"], "retained": kept["files"] == 3 and kept["all_hashes"]}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "asset": asset, "frontend": front, "retention": kept, "adjudication": {"signed_surface_fields_visible": True, "energy_envelope_is_encoding_mechanism": False, "hiddenstate_cleanup_required": False, "language_encoding_mechanism_closed": False}, "checks": checks, "all_checks_passed": all(checks.values())}
    save(OUT / "analysis" / ("final.json" if result["all_checks_passed"] else "prebuild.json"), result)
    if result["all_checks_passed"]: append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
