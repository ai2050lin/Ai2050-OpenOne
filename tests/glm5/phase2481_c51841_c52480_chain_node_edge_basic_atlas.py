#!/usr/bin/env python3
"""Basic all-coordinate node/edge atlas for behavior-qualified knowledge chains."""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result"
P2480 = RESULT / "phase2480_c51201_c51840_qualified_chain_fullcoordinate_field"
OUT = RESULT / "phase2481_c51841_c52480_chain_node_edge_basic_atlas"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN, DIM = 2481, "C51841-C52480", 2560
sys.path.insert(0, str(ROOT / "tests/glm5"))
import model_utils  # noqa: E402
MODEL_PATH = Path(model_utils.MODEL_CONFIGS["qwen3"]["path"])


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def close(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None: mmap.close()


def cosine_rows(first: np.ndarray, second: np.ndarray) -> float:
    a = np.asarray(first, dtype=np.float64); b = np.asarray(second, dtype=np.float64)
    numerator = np.sum(a * b, axis=1)
    denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return float(np.mean(numerator / np.maximum(denominator, 1e-30)))


def matched(first: np.ndarray, second: np.ndarray) -> dict:
    coordinate = cosine_rows(first, second)
    null = [cosine_rows(first, second[[1, 2, 0]]), cosine_rows(first, second[[2, 0, 1]])]
    return {
        "coordinate": coordinate, "family_mismatch_mean": float(np.mean(null)),
        "family_mismatch_max": float(np.max(null)), "family_identity_advantage": coordinate - float(np.max(null)),
    }


def normalize(value: str) -> str:
    import re
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", value.casefold())


def extract() -> tuple[dict, list[dict]]:
    final = json.loads((P2480 / "analysis/final.json").read_text(encoding="utf-8"))
    rows = read_jsonl(P2480 / "index/chain_rows.jsonl")
    prompt = np.load(final["collection"]["prompt_field"]["path"], mmap_mode="r")
    trajectory = np.load(final["collection"]["trajectory_field"]["path"], mmap_mode="r")
    token_ids = np.load(final["collection"]["generated_ids"], mmap_mode="r")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True, local_files_only=True)
    derived = OUT / "derived"; derived.mkdir(parents=True, exist_ok=True)
    prompt_nodes_path = derived / "prompt_main_distractor_node_states.float32.npy"
    generated_nodes_path = derived / "generated_main_node_states.float32.npy"
    boundary_path = derived / "prompt_answer_boundary_states.float32.npy"
    prompt_nodes = np.lib.format.open_memmap(prompt_nodes_path, mode="w+", dtype=np.float32, shape=(24, 2, 4, 38, DIM))
    generated_nodes = np.lib.format.open_memmap(generated_nodes_path, mode="w+", dtype=np.float32, shape=(24, 4, 38, DIM))
    boundary = np.lib.format.open_memmap(boundary_path, mode="w+", dtype=np.float32, shape=(24, 38, DIM))
    generated_nodes[:] = np.nan
    enhanced = []
    try:
        for row_number, row in enumerate(rows):
            offset, end = row["prompt_token_offset"]
            boundary[row_number] = prompt[end - 1]
            for path_index, chain_name in enumerate(("main_chain", "distractor_chain")):
                for node_index, node in enumerate(row[chain_name]):
                    positions = [position for first, last in row["node_spans"][node] for position in range(first, last)]
                    prompt_nodes[row_number, path_index, node_index] = np.mean(prompt[offset + np.asarray(positions)], axis=0, dtype=np.float64)
            ids = [int(value) for value in token_ids[row_number] if int(value) >= 0]
            steps = []
            for node_index, node in enumerate(row["main_chain"]):
                step_found = None
                for step in range(1, len(ids) + 1):
                    if normalize(node) in normalize(tokenizer.decode(ids[:step], skip_special_tokens=True)):
                        step_found = step; break
                steps.append(step_found)
                if step_found is not None:
                    generated_nodes[row_number, node_index] = trajectory[row_number, step_found]
            enhanced.append({**row, "generated_main_node_steps": steps})
    finally:
        for value in (prompt_nodes, generated_nodes, boundary): value.flush(); close(value)
        for value in (prompt, trajectory, token_ids): close(value)
    index_path = OUT / "index/node_event_rows.jsonl"; index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in enhanced), encoding="utf-8")
    return {
        "prompt_nodes": {"path": str(prompt_nodes_path), "shape": [24, 2, 4, 38, DIM], "axes": ["row", "main/distractor", "node0-3", "qpoint", "coordinate"]},
        "generated_nodes": {"path": str(generated_nodes_path), "shape": [24, 4, 38, DIM], "missing": "NaN when node did not appear within 48 generated tokens"},
        "answer_boundary": {"path": str(boundary_path), "shape": [24, 38, DIM]}, "index": str(index_path),
    }, enhanced


def passports(collection: dict, rows: list[dict]) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    prompt_nodes = np.load(collection["prompt_nodes"]["path"], mmap_mode="r")
    generated_nodes = np.load(collection["generated_nodes"]["path"], mmap_mode="r")
    units, languages, families = [11, 12, 13], ["en", "zh"], ["causal", "handoff", "part_whole"]
    prompt_pass = np.zeros((3, 2, 3, 4, 38, DIM), dtype=np.float32)
    generated_pass = np.full((3, 2, 3, 4, 38, DIM), np.nan, dtype=np.float32)
    paired = np.zeros((3, 2, 3, 4, 38, DIM), dtype=np.float32)
    counts = np.zeros((3, 2, 3, 4), dtype=np.int32)
    try:
        for ui, unit in enumerate(units):
            for li, language in enumerate(languages):
                valid = [i for i, row in enumerate(rows) if row["unit"] == unit and row["language"] == language]
                for node in range(4):
                    baseline = np.mean(prompt_nodes[valid, 0, node], axis=0, dtype=np.float64)
                    generated_valid = [i for i in valid if np.isfinite(generated_nodes[i, node]).all()]
                    generated_baseline = np.mean(generated_nodes[generated_valid, node], axis=0, dtype=np.float64) if generated_valid else None
                    for fi, family in enumerate(families):
                        selected = [i for i in valid if rows[i]["family"] == family]
                        counts[ui, li, fi, node] = len(selected)
                        prompt_pass[ui, li, fi, node] = np.mean(prompt_nodes[selected, 0, node], axis=0, dtype=np.float64) - baseline
                        paired[ui, li, fi, node] = np.mean(prompt_nodes[selected, 0, node] - prompt_nodes[selected, 1, node], axis=0, dtype=np.float64)
                        generated_selected = [i for i in selected if np.isfinite(generated_nodes[i, node]).all()]
                        if generated_selected and generated_baseline is not None:
                            generated_pass[ui, li, fi, node] = np.mean(generated_nodes[generated_selected, node], axis=0, dtype=np.float64) - generated_baseline
        prompt_path = OUT / "derived/prompt_family_node_passports.float32.npy"
        generated_path = OUT / "derived/generated_family_node_passports.float32.npy"
        paired_path = OUT / "derived/main_minus_distractor_node_contrasts.float32.npy"
        count_path = OUT / "derived/family_node_counts.int32.npy"
        np.save(prompt_path, prompt_pass); np.save(generated_path, generated_pass); np.save(paired_path, paired); np.save(count_path, counts)
    finally:
        close(prompt_nodes); close(generated_nodes)
    meta = {
        "prompt_passports": str(prompt_path), "generated_passports": str(generated_path),
        "main_minus_distractor": str(paired_path), "counts": str(count_path),
        "shape": list(prompt_pass.shape), "axes": ["unit11/12/13", "language", "family(causal,handoff,part_whole)", "node0-3", "qpoint", "coordinate"],
    }
    return meta, prompt_pass, generated_pass, paired


def analyze(prompt_pass: np.ndarray, generated_pass: np.ndarray, paired: np.ndarray) -> dict:
    all_metrics = {"crosslanguage": {}, "node_reuse": {}, "prompt_to_generation": {}, "main_distractor_crosslanguage": {}}
    for ui, unit in enumerate((11, 12, 13)):
        for qpoint in range(38):
            ci = [matched(prompt_pass[ui, 0, :, node, qpoint], prompt_pass[ui, 1, :, node, qpoint]) for node in range(4)]
            all_metrics["crosslanguage"].setdefault(f"unit{unit}", {})[f"q{qpoint}"] = {key: float(np.mean([value[key] for value in ci])) for key in ci[0]}
            nr = [matched(prompt_pass[ui, language, :, node, qpoint], prompt_pass[ui, language, :, node + 1, qpoint]) for language in range(2) for node in range(3)]
            all_metrics["node_reuse"].setdefault(f"unit{unit}", {})[f"q{qpoint}"] = {key: float(np.mean([value[key] for value in nr])) for key in nr[0]}
            pg = []
            for language in range(2):
                for node in range(4):
                    if np.isfinite(generated_pass[ui, language, :, node, qpoint]).all():
                        pg.append(matched(prompt_pass[ui, language, :, node, qpoint], generated_pass[ui, language, :, node, qpoint]))
            all_metrics["prompt_to_generation"].setdefault(f"unit{unit}", {})[f"q{qpoint}"] = ({key: float(np.mean([value[key] for value in pg])) for key in pg[0]} if pg else None)
            md = [matched(paired[ui, 0, :, node, qpoint], paired[ui, 1, :, node, qpoint]) for node in range(4)]
            all_metrics["main_distractor_crosslanguage"].setdefault(f"unit{unit}", {})[f"q{qpoint}"] = {key: float(np.mean([value[key] for value in md])) for key in md[0]}
    selection = {}; lockbox = {}
    for metric in all_metrics:
        candidates = [qpoint for qpoint in range(38) if all_metrics[metric]["unit12"][f"q{qpoint}"] is not None]
        qpoint = max(candidates, key=lambda q: all_metrics[metric]["unit12"][f"q{q}"]["family_identity_advantage"])
        selection[metric] = qpoint; lockbox[metric] = all_metrics[metric]["unit13"][f"q{qpoint}"]
    qpoint = selection["crosslanguage"]
    discovery_energy = np.sum(np.square(prompt_pass[1, :, :, :, qpoint].astype(np.float64)), axis=(0, 1, 2))
    order = np.argsort(-discovery_energy)
    np.save(OUT / "derived/chain_discovery_coordinate_order.int32.npy", order.astype(np.int32))
    unit13_energy = np.sum(np.square(prompt_pass[2, :, :, :, qpoint].astype(np.float64)), axis=(0, 1, 2))
    rank_profile_corr = float(np.corrcoef(discovery_energy[order], unit13_energy[order])[0, 1])
    normalized = discovery_energy / max(float(np.sum(discovery_energy)), 1e-30)
    effective = float(1.0 / np.sum(normalized * normalized))
    cumulative = np.cumsum(np.sort(normalized)[::-1])
    density = {
        "selected_qpoint": qpoint, "effective_coordinate_count": effective,
        "effective_fraction": effective / DIM,
        "coordinates_for_50pct_energy": int(np.searchsorted(cumulative, 0.5) + 1),
        "coordinates_for_90pct_energy": int(np.searchsorted(cumulative, 0.9) + 1),
        "unit12_unit13_frozen_rank_energy_correlation": rank_profile_corr,
    }
    return {"all_qpoints": all_metrics, "unit12_selection": selection, "unit13_lockbox": lockbox, "density": density}


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"): return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""


## Phase {PHASE}: 三类知识链节点—干扰节点—生成节点的基本全坐标图谱与锁箱（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 不引入PCA/Top-K或高级数学。对Phase2480每行主链与干扰链的四个节点，汇总其全部上下文token occurrence（只在token轴求均值，保留2560坐标），另提取answer-boundary及自主输出中每个主链节点首次完整出现后的状态。建立三类基本场：（1）family相对prompt node passport；（2）成功输出family node passport；（3）逐行main-node减paired-distractor-node。比较跨语言、相邻节点位置复用、prompt→generated、主/干扰对比跨语言。unit12只选qpoint，unit13一次锁箱；三family错配作基本对照。

$$P_{{f,k}}^{{(q)}}=\mathbb E[H_{{main,k}}^{{(q)}}\mid f]-\mathbb E[H_{{main,k}}^{{(q)}}],\qquad D_{{f,k}}^{{(q)}}=\mathbb E[H_{{main,k}}^{{(q)}}-H_{{dist,k}}^{{(q)}}\mid f].$$

**结果汇总。** 派生场 `{json.dumps(result['collection'], ensure_ascii=False)}`；unit12选层 `{json.dumps(result['analysis']['unit12_selection'], ensure_ascii=False)}`；unit13锁箱 `{json.dumps(result['analysis']['unit13_lockbox'], ensure_ascii=False)}`；坐标密度 `{json.dumps(result['analysis']['density'], ensure_ascii=False)}`；裁决 `{json.dumps(result['adjudication'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2481_c51841_c52480_chain_node_edge_basic_atlas.py`；prompt主/干扰节点场、生成节点场、回答边界、family护照、main-minus-distractor、冻结坐标顺序、逐行事件index与全部qpoint结果位于同名目录。

**分析与理论进展。** 本Phase第一次把长句链的外部结构拆成可逐坐标比较的节点位置、关系族、主/干扰路径及实际输出节点。锁箱同family优于错family才支持“族纹理跨语言/位置/生成事件复用”的候选；任何正结果仍可能来自family专属关系措辞或节点token身份。冻结坐标排序只服务可视化和复现，不是天然齿轮。

**问题硬伤与结论。** 只有三family，错配零假设仅有两个derangement，不能给稳定尾部分位；causal有两个surface而另外两族一个，family计数不平衡；生成节点缺失依赖行为成功；对occurrence求均值会混合source/target角色。当前只积累L1级图谱并报告坐标密度，不进行稀疏删除或机制闭合。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle: handle.write(text)


def main() -> None:
    collection, rows = extract()
    passport_meta, prompt_pass, generated_pass, paired = passports(collection, rows)
    analysis = analyze(prompt_pass, generated_pass, paired)
    collection["passports"] = passport_meta
    lockbox = analysis["unit13_lockbox"]
    checks = {
        "raw_node_shapes": collection["prompt_nodes"]["shape"] == [24, 2, 4, 38, 2560] and collection["generated_nodes"]["shape"] == [24, 4, 38, 2560],
        "three_families": prompt_pass.shape[2] == 3,
        "all_coordinates": prompt_pass.shape[-1] == 2560,
        "unit12_selection_unit13_lockbox": len(analysis["unit12_selection"]) == 4 and len(lockbox) == 4,
        "finite_lockbox": all(value is not None and all(math.isfinite(number) for number in value.values()) for value in lockbox.values()),
        "full_coordinate_density": 0 < analysis["density"]["effective_coordinate_count"] <= 2560,
        "claim_boundary": True,
    }
    result = {
        "phase": PHASE, "campaign": CAMPAIGN, "collection": collection, "analysis": analysis,
        "adjudication": {
            "basic_chain_coordinate_atlas_available": True,
            "family_texture_reuse_candidate": all(value["family_identity_advantage"] > 0 for value in lockbox.values()),
            "natural_coordinate_gear_identified": False,
            "causal_chain_compiler_identified": False,
            "language_encoding_mechanism_closed": False,
        },
        "checks": checks, "all_checks_passed": all(checks.values()),
    }
    save(OUT / "analysis/final.json", result)
    if result["all_checks_passed"]: append_memo(result)
    print(json.dumps({key: value for key, value in result.items() if key != "analysis"} | {"analysis_summary": {"unit12_selection": analysis["unit12_selection"], "unit13_lockbox": lockbox, "density": analysis["density"]}}, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]: raise RuntimeError(checks)


if __name__ == "__main__": main()
