#!/usr/bin/env python3
"""Publish cross-architecture behavior boundaries and close the campaign."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result"
P2271 = RESULT / "phase2271_c1625_c1664_cross_architecture_topology"
P2272 = RESULT / "phase2272_c1665_c1704_output_boundary_repair"
CONTRACT = RESULT / "phase2265_c1433_c1468_independent_bilingual_contract"
OUT = RESULT / "phase2273_c1705_c1720_cross_architecture_publication"
VIS = ROOT / "frontend/public/vis_data/research_kernel"
CATALOG = ROOT / "frontend/public/research_data/current/language_encoding_catalog.json"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"

PHASE = 2273
CAMPAIGN = "C1705-C1720"
MODELS = ("glm4", "deepseek7b")
STAGES = (("original", P2271), ("boundary_repaired", P2272))
FAMILIES = ("location_state", "property_state", "patient_binding",
            "temporal_order", "comparison_order")
METRICS = ("candidate_accuracy", "generation_accuracy", "parsed_generation_fraction", "dual_qualified")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def family_parse_fraction(stage_root: Path, model: str) -> dict[str, float]:
    materials = read_jsonl(CONTRACT / "material/independent_bilingual_cases.jsonl")
    family_by_id = {row["case_id"]: row["family"] for row in materials}
    generated = read_jsonl(stage_root / model / "behavior/generation.jsonl")
    output = {}
    for family in FAMILIES:
        rows = [row for row in generated if family_by_id[row["case_id"]] == family]
        output[family] = float(np.mean([row["parsed"] is not None for row in rows]))
    return output


def publish_behavior_boundary() -> dict:
    matrix, rows = [], []
    for stage, root in STAGES:
        aggregate = load_json(root / "analysis/final.json")
        for model in MODELS:
            ledger = aggregate["models"][model]["behavior"]
            parse = family_parse_fraction(root, model)
            for family in FAMILIES:
                family_result = ledger["families"][family]
                values = [family_result["candidate_accuracy"], family_result["generation_accuracy"],
                          parse[family], float(family_result["dual_qualified"])]
                rows.append({"row": len(rows), "stage": stage, "model": model,
                             "family": family, "metrics": list(METRICS),
                             "behavior_gate": 0.75,
                             "internal_hiddenstate_observed": False})
                matrix.append(values)
    binary = VIS / "c1705_cross_architecture_behavior_boundary.float16.npy"
    values = np.asarray(matrix, np.float16)
    np.save(binary, values)
    sha = file_hash(binary)
    metadata = {
        "schema": "ai2050.cross-architecture-behavior-boundary.v1",
        "generated_at": datetime.now().astimezone().isoformat(),
        "phase": PHASE, "campaign": CAMPAIGN,
        "title": "GLM4 / DS7B Cross-Architecture Behavior Boundary",
        "binary_url": "/vis_data/research_kernel/" + binary.name,
        "binary_shape": list(values.shape), "binary_sha256": sha,
        "columns": list(METRICS), "coordinate_count": len(METRICS), "rows": rows,
        "coordinate_semantics": "Columns are behavior qualification metrics, not activation coordinates or model parameters",
        "boundary": "No GLM4 or DS7B internal field was collected because fewer than two families passed the frozen dual-behavior gate.",
    }
    target = VIS / "c1705_cross_architecture_behavior_boundary.json"
    save_json(target, metadata)
    return {"id": "c1705_cross_architecture_behavior_boundary", "metadata": target,
            "binary": binary, "shape": list(values.shape), "sha256": sha, "rows": len(rows)}


def update_catalog(dataset: dict) -> dict:
    catalog = load_json(CATALOG)
    entry = {
        "id": dataset["id"], "title": "GLM4 / DS7B Cross-Architecture Behavior Boundary",
        "phase": PHASE, "campaign": CAMPAIGN, "model": "GLM4 / DeepSeek-7B",
        "source_path": "/vis_data/research_kernel/" + dataset["metadata"].name,
        "binary_path": "/vis_data/research_kernel/" + dataset["binary"].name,
        "source_schema": "ai2050.cross-architecture-behavior-boundary.v1",
        "coordinate_count": 4, "row_count": dataset["rows"],
        "claim_level": "behavior_qualification_boundary",
        "boundary": "Behavior metrics only; no internal activation field qualified.",
        "kinds": ["candidate_accuracy", "generation_accuracy", "parse_fraction", "dual_qualification"],
    }
    catalog["datasets"] = [entry] + [row for row in catalog.get("datasets", []) if row.get("id") != entry["id"]]
    catalog["generated_at"] = datetime.now().astimezone().isoformat()
    save_json(CATALOG, catalog)
    return {"entry_added": entry["id"], "dataset_count": len(catalog["datasets"])}


def verify_dataset(dataset: dict) -> dict:
    metadata = load_json(dataset["metadata"])
    value = np.load(dataset["binary"], mmap_mode="r")
    try:
        checks = {
            "shape_ok": list(value.shape) == metadata["binary_shape"] == dataset["shape"],
            "rows_ok": len(metadata["rows"]) == value.shape[0] == dataset["rows"],
            "finite": bool(np.isfinite(np.asarray(value, np.float32)).all()),
            "range_ok": bool(np.all((value >= 0) & (value <= 1))),
        }
    finally:
        mmap = getattr(value, "_mmap", None)
        if mmap is not None:
            mmap.close()
    checks["hash_ok"] = file_hash(dataset["binary"]) == metadata["binary_sha256"]
    return checks


def frontend_build() -> dict:
    npm = shutil.which("npm.cmd") or shutil.which("npm")
    if npm:
        command = [npm, "run", "build"]
    else:
        node = sorted(Path.home().glob(
            "AppData/Local/OpenAI/Codex/runtimes/cua_node/*/bin/node.exe"), reverse=True)[0]
        command = [str(node), str(ROOT / "frontend/node_modules/vite/bin/vite.js"), "build"]
    completed = subprocess.run(command, cwd=ROOT / "frontend", text=True, encoding="utf-8",
                               errors="replace", capture_output=True, timeout=600)
    return {"command": command, "returncode": completed.returncode,
            "stdout_tail": completed.stdout[-2000:], "stderr_tail": completed.stderr[-2000:],
            "passed": completed.returncode == 0}


def hidden_field_audit() -> dict:
    candidates = []
    for root in (P2271, P2272):
        for model in MODELS:
            path = root / model / "raw/relative_window_field.float16.npy"
            candidates.append({"path": str(path.relative_to(ROOT)), "exists": path.exists()})
    return {"files": candidates, "hidden_field_count": sum(row["exists"] for row in candidates),
            "bytes_deleted": 0, "reason": "No internal field was created because behavior qualification was insufficient."}


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 跨架构行为边界图谱发布与大阶段终审（C1705-C1720） [{stamp}]

**测试原理与用例。** 汇总 Phase2271 原始助手接口与 Phase2272 冻结边界修复，在 GLM4、DS7B、五个家族上形成 20 行行为边界矩阵。每行只含候选准确率、自由生成准确率、精确代码解析率和双行为资格四列。这个图用于解释为什么没有跨架构 HiddenState 结果；四列是行为度量，不是激活坐标或模型参数。凡未通过至少两个家族双行为门的模型，不创建、不展示虚假的 embedding/HiddenState 场。

**数学公式。** 对模型 $m$、接口阶段 $s$、家族 $f$ 定义行为边界向量：

$$
B_{{m,s,f}}=(A^{{\mathrm{{cand}}}}_{{m,s,f}},
A^{{\mathrm{{gen}}}}_{{m,s,f}},P^{{\mathrm{{parse}}}}_{{m,s,f}},Q_{{m,s,f}}),
$$

$$
Q_{{m,s,f}}=\mathbf 1\!\left[
\min(A^{{\mathrm{{cand}}}}_{{m,s,f}},A^{{\mathrm{{gen}}}}_{{m,s,f}})\ge 0.75
\right],
$$

并额外要求 discovery 与 fresh confirmation 同时满足该门，才允许内部观察。

**结果汇总。** 发布数据 `{json.dumps(result['dataset'], ensure_ascii=False)}`；验证 `{json.dumps(result['verification'], ensure_ascii=False)}`；原始场审计 `{json.dumps(result['hidden_field_audit'], ensure_ascii=False)}`；客户端构建 `{json.dumps(result['frontend_build'], ensure_ascii=False)}`；HTTP 检查由最终文件审计执行。Phase2271 的 GLM4 为“生成强、候选弱”，DS7B 为“强制推理导致代码接口失效”；Phase2272 后 GLM4 仅位置状态合格，DS7B 解析恢复但五家族仍均低于双门。

**分析与理论进展。** `{result['strict_conclusion']}` 因而，Qwen3-4B/14B 的模型本地逐坐标预测拓扑目前不能外推为跨架构普遍规律；更严格地说，它在 GLM4 与 DS7B 上没有获得合法的内部测试对象，而不是已经被反证。一个重要方法拼图是：候选 logit、自由生成、输出解析和内部场是不同对象，跨模型不能把某一输出接口当作统一测量尺。

**问题、硬伤与瓶颈。** 未得到 GLM4/DS7B 的 embedding/HiddenState 坐标图；边界修复是模型特异的；受控代码任务与自然语言能力仍有距离；GLM4 生成正确但候选分数偏低的原因没有被内部化解释；DS7B 关闭 think 后仍只到中等行为水平。继续为当前材料调提示会变成接口过拟合，因此本路线在这里停止，而总项目不停止。

**结论与下一大阶段。** 脚本：`tests/glm5/phase2273_c1705_c1720_cross_architecture_publication.py`；结果：`tests/glm5/result/phase2273_c1705_c1720_cross_architecture_publication`；图谱：`frontend/public/vis_data/research_kernel/c1705_*`。下一阶段仍服务于语言编码图谱，但目标和材料接口发生实质变化：应为每个架构先建立自然、模型适配且语义等价的行为接口，再冻结跨模型共同的外部语义对象；只有各模型分别合格后，比较相对层深、角色拓扑和模型本地全坐标规律。不能继续在本合同上追加第三种后缀。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def run() -> dict:
    final = OUT / "analysis/final.json"
    if final.exists():
        result = load_json(final)
        append_memo(result)
        return result
    p2271 = load_json(P2271 / "analysis/final.json")
    p2272 = load_json(P2272 / "analysis/final.json")
    dataset = publish_behavior_boundary()
    catalog = update_catalog(dataset)
    verification = verify_dataset(dataset)
    build = frontend_build()
    field_audit = hidden_field_audit()
    checks = {
        "phase2271_complete": p2271["all_checks_passed"],
        "phase2272_complete": p2272["all_checks_passed"],
        "behavior_matrix_verified": all(verification.values()),
        "no_unqualified_hidden_field": field_audit["hidden_field_count"] == 0,
        "frontend_build_passed": build["passed"],
    }
    result = {
        "phase": PHASE, "campaign": CAMPAIGN,
        "status": "closed" if all(checks.values()) else "audit_failed",
        "timestamp": datetime.now().astimezone().isoformat(),
        "dataset": {"id": dataset["id"], "shape": dataset["shape"],
                    "rows": dataset["rows"], "sha256": dataset["sha256"]},
        "verification": verification, "catalog": catalog,
        "frontend_build": build, "hidden_field_audit": field_audit,
        "checks": checks, "all_checks_passed": all(checks.values()),
        "strict_conclusion": ("GLM4 and DS7B did not yield a legal cross-architecture HiddenState topology test "
                              "under either frozen interface; this is missing qualification, not a negative "
                              "mechanism result."),
    }
    save_json(final, result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=True, indent=2), flush=True)
    return result


if __name__ == "__main__":
    run()
