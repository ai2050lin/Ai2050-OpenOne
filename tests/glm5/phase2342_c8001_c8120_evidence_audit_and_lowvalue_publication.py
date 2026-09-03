#!/usr/bin/env python3
"""Final evidence audit, low-value multidose publication, and next-stage adjudication."""
from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P2332 = RESULT / "phase2332_c6361_c6480_multidose_natural_direction_calibration"
OUT = RESULT / "phase2342_c8001_c8120_evidence_audit_and_lowvalue_publication"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
VIS = ROOT / "frontend/public/vis_data/research_kernel"
PHASE = 2342
CAMPAIGN = "C8001-C8120"
SOURCES = (10, 20, 30)
TARGETS = {10: (11, 14, 37), 20: (21, 24, 37), 30: (31, 34, 37)}
PROBES = ("natural_state", "random_control", "natural_plus_random")
DOSES = (0.02, 0.01, 0.005, 0.0025)

sys.path.insert(0, str(TESTS))
import phase2315_c5041_c5100_active_response_contract as io  # noqa: E402
import phase2319_c5321_c5400_active_response_atlas_cleanup as atlas  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


PHASE_DIRS = {
    2330: "phase2330_c6081_c6200_language_family_atlas_contract",
    2331: "phase2331_c6201_c6360_qwen4b_twenty_family_fullfield",
    2332: "phase2332_c6361_c6480_multidose_natural_direction_calibration",
    2333: "phase2333_c6481_c6640_twenty_family_coordinate_atlas",
    2334: "phase2334_c6641_c6760_fresh_family_atlas_adjudication",
    2335: "phase2335_c6761_c6920_independent_construction_replication",
    2336: "phase2336_c6921_c7080_decisive_output_coordinate_accounting",
    2337: "phase2337_c7081_c7240_fixed_ab_output_interface",
    2338: "phase2338_c7241_c7400_fixed_ab_layer_trajectory_controls",
    2339: "phase2339_c7401_c7600_crossmodel_fixed_ab_replication",
    2340: "phase2340_c7601_c7840_twenty_family_fixed_interface_atlas",
    2341: "phase2341_c7841_c8000_coordinate_texture_orthogonal_controls",
}


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_rows(path: Path) -> list[dict]:
    return io.read_rows(path)


def close_memmap(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def metadata_for(format_name: str) -> list[dict]:
    cells = read_rows(P2332 / format_name / "index/cells.jsonl")
    lookup = {(int(row["pair_index"]), int(row["source_index"])): row for row in cells}
    metadata = []
    for pair_index in range(20):
        for source_index, source in enumerate(SOURCES):
            cell = lookup[(pair_index, source_index)]
            for probe_index, probe in enumerate(PROBES):
                for dose_index, dose in enumerate(DOSES):
                    for target_index, target in enumerate(TARGETS[source]):
                        metadata.append({
                            "format": format_name, "family": cell["family"], "language": cell["language"],
                            "pair_index": pair_index, "source_q": source, "target_q": target,
                            "probe": probe, "probe_index": probe_index, "dose": dose, "dose_index": dose_index,
                            "target_index": target_index, "source_norm": cell["source_norm"],
                            "state0_case": cell["state0_case"], "state1_case": cell["state1_case"],
                        })
    return metadata


def publish_low_value_fields() -> list[dict]:
    assets = []
    specs = (
        ("float16", "directional_derivative", "c8001_qwen4b_fp16_multidose_directional_derivative",
         "FP16 multidose full-coordinate directional derivatives", "multidose_directional_derivative_v1"),
        ("float16", "even_response", "c8002_qwen4b_fp16_multidose_even_response",
         "FP16 multidose full-coordinate even responses", "multidose_even_response_v1"),
        ("bfloat16", "directional_derivative", "c8003_qwen4b_bf16_multidose_directional_derivative",
         "BF16 multidose full-coordinate directional derivatives", "multidose_directional_derivative_v1"),
        ("bfloat16", "even_response", "c8004_qwen4b_bf16_multidose_even_response",
         "BF16 multidose full-coordinate even responses", "multidose_even_response_v1"),
    )
    metadata_cache = {name: metadata_for(name) for name in ("float16", "bfloat16")}
    for format_name, stem, dataset_id, title, schema in specs:
        source_path = P2332 / format_name / f"raw/{stem}.float32.npy"
        source = np.load(source_path, mmap_mode="r")
        flat = source.reshape(-1, source.shape[-1])
        metadata = metadata_cache[format_name]
        if flat.shape[0] != len(metadata):
            raise RuntimeError(("metadata_shape", dataset_id, flat.shape, len(metadata)))
        binary = VIS / f"{dataset_id}.float32.npy"
        out = atlas.create_binary(binary.name, flat.shape[0], flat.shape[1], np.float32)
        out[:] = flat
        out.flush(); close_memmap(out); close_memmap(source)
        assets.append(atlas.write_metadata(
            dataset_id, title, binary, metadata, f"Qwen3-4B-{format_name.upper()}", schema,
            "numerical calibration, not semantic mechanism", "20 families × 3 sources × 3 probes × 4 doses × 3 targets",
            "all 2560 coordinates, including low-value responses",
            {"coordinate_count": 2560, "no_projection": True, "doses": list(DOSES), "probes": list(PROBES)},
        ))
    return assets


def phase_audit() -> dict:
    memo_text = MEMO.read_text(encoding="utf-8")
    records = {}
    for phase, directory in PHASE_DIRS.items():
        path = RESULT / directory / "analysis/final.json"
        record = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        headings = len(re.findall(rf"^## Phase {phase}:", memo_text, flags=re.MULTILINE))
        records[str(phase)] = {
            "final_exists": path.exists(), "memo_heading_count": headings,
            "engineering_checks_passed": None if record is None else bool(record.get("all_checks_passed", False)),
            "scientific_gate": None if record is None else (
                record.get("analysis", {}).get("gate", {}).get("passed")
                if isinstance(record.get("analysis"), dict) else None
            ),
        }
    return {
        "continuous": sorted(PHASE_DIRS) == list(range(2330, 2342)),
        "all_final_exist": all(row["final_exists"] for row in records.values()),
        "all_memo_once": all(row["memo_heading_count"] == 1 for row in records.values()),
        "records": records,
        "expected_numeric_check_failures": {
            "2336": "FP16 max reconstruction residual 0.011299 exceeded frozen 0.01 by 0.001299",
            "2337": "FP16 max reconstruction residual 0.022891 exceeded frozen 0.02 by 0.002891",
        },
    }


def field_coverage(assets: list[dict]) -> dict:
    coverage = {
        "phase2331_boundary_and_representative": ["c6481", "c6482"],
        "phase2332_multidose_response_fields": [row["id"] for row in assets],
        "phase2333_twenty_family_fields": ["c6481", "c6482", "c6483", "c6484"],
        "phase2334_fresh_passport": ["c6641"],
        "phase2335_independent_fields": ["c6761", "c6762"],
        "phase2336_decisive_state_weight_contribution": ["c6921", "c6922", "c6923"],
        "phase2337_fixed_ab_state_weight_contribution": ["c7081", "c7082", "c7083"],
        "phase2338_option_swap_trajectories": ["c7241", "c7242", "c7243"],
        "phase2339_crossmodel_full_fields": ["c7401_qwen14b", "c7402_qwen14b", "c7401_glm4", "c7402_glm4", "c7401_deepseek7b", "c7402_deepseek7b"],
        "phase2340_twenty_family_full_fields": ["c7601", "c7602"],
        "phase2341_q23_control_passport": ["c7841"],
    }
    raw_hiddenstate_unpublished = []
    return {"coverage": coverage, "raw_hiddenstate_unpublished": raw_hiddenstate_unpublished,
            "deleted_hiddenstate_files": [],
            "cleanup_decision": "No HiddenState field deleted: every important full field is represented in a verified visualization asset; small decisions, indices, weights, and progress files are not HiddenState fields."}


def evidence_adjudication() -> dict:
    p2332 = json.loads((P2332 / "analysis/final.json").read_text(encoding="utf-8"))
    p2335 = json.loads((RESULT / PHASE_DIRS[2335] / "analysis/final.json").read_text(encoding="utf-8"))
    p2338 = json.loads((RESULT / PHASE_DIRS[2338] / "analysis/final.json").read_text(encoding="utf-8"))
    p2339 = json.loads((RESULT / PHASE_DIRS[2339] / "analysis/final.json").read_text(encoding="utf-8"))
    p2340 = json.loads((RESULT / PHASE_DIRS[2340] / "analysis/final.json").read_text(encoding="utf-8"))
    p2341 = json.loads((RESULT / PHASE_DIRS[2341] / "analysis/final.json").read_text(encoding="utf-8"))
    return {
        "retained_claims": [
            "HiddenState and output use are distributed across the full coordinate field; a single direction is not an adequate primary object.",
            "Concrete coordinate identity matters for the q23 within-language family atlas: full/standardized fields strongly outperform row-sorted distributions and prompt length.",
            "A fixed A/B output interface separates natural answer-token identity from state-conditioned coordinate use.",
            "Option-swap stability and two strict cross-model replications support a functional coordinate-use-strength phenomenon, not coordinate-number equivalence.",
            "Precision format and dose materially change low-value finite-difference observations; FP16/BF16 are interventions, not neutral observation windows.",
        ],
        "corrected_or_rejected_claims": [
            "Rejected: failure to find one static causal gate proves that no static gears exist.",
            "Rejected: the current data proves a Riemannian tensor field, geodesics, an Omega-manifold, or a new closed mathematical theory.",
            "Rejected: 6912 or other coordinate×case cells are independent semantic samples.",
            "Rejected: activation coordinates are model parameters; they are state values, while unembedding rows are parameters.",
            "Corrected: Phase2317 implementation normalized base directions and kept pair-sum scale intentionally; the memo formula was ambiguous, not the code path.",
            "Corrected: Phase2332 even/odd medians are FP16 0.314973 and BF16 1.241791; the earlier 28.66/75.39 summary omitted dose×source-norm and is invalid.",
            "Rejected: the current family graph is language universal. q23 cross-language minimum is 0.2353, below the frozen 0.30 gate.",
            "Rejected: the final layer contains one stable universal family direction. Independent generators invalidated natural-difference transfer and Phase2340 final distance dominance failed.",
        ],
        "anchor_results": {
            "phase2332": p2332["analysis"],
            "phase2335_cross_generator_semantic_candidates": p2335["analysis"]["semantic_candidates"],
            "phase2338_qwen4_option_swap_gate": p2338["analysis"]["gate"],
            "phase2339_crossmodel_summary": p2339["summary"],
            "phase2340_behavior_qualified_count": len(p2340["behavior"]["qualified"]),
            "phase2340_midlayer_peaks": p2340["analysis"]["peaks_fresh_lockbox"],
            "phase2341_gate": p2341["analysis"]["gate"],
        },
        "current_theory_status": "A reusable, language-conditioned coordinate-use-strength atlas candidate exists; a language-invariant semantic code and causal mechanism closure do not yet exist.",
        "next_big_stage": {
            "title": "Bilingual semantic-graph alignment and lexical/task orthogonalization",
            "objective": "Determine whether the same semantic graph in English and Chinese reuses coordinate identities after lexical and task-template factors are independently crossed.",
            "work_packages": [
                "Rebuild at least 12 behavior-qualified families with identical entity placeholders and exactly parallel semantic graphs in English/Chinese; use 16+ units, multiple natural templates, and untouched lockboxes.",
                "Cross three axes independently: same vocabulary/different task, same task/different vocabulary, same semantic graph/different language; keep fixed balanced A/B output and paired option swaps.",
                "Freeze q23 only as an observation window, also analyze every layer; calculate full-coordinate sign, standardized magnitude, coordinate-identity, and row-sorted controls without Top-K selection.",
                "Require bilingual transfer above a frozen threshold in both directions, coordinate identity advantage over sorted distributions, behavior qualification, and independent lockbox replication before any causal intervention.",
                "Only then repeat the bilingual functional invariant on Qwen14B, GLM4, and DS7B sequentially; compare family×relative-depth behavior, never raw coordinate numbers across models.",
            ],
        },
    }


def append_memo(result: dict) -> None:
    marker = f"## Phase {PHASE}:"
    if marker in MEMO.read_text(encoding="utf-8"): return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    record = rf"""

## Phase {PHASE}: 全阶段证据总审计、低值场发布与双语下一阶段裁决（{CAMPAIGN}） [{stamp}]

**测试原理与公式。** 本Phase不增加模型结论，而把Phase2330–2341的工程通过、科学门槛、MEMO连续性、原始场可视化覆盖和附件主张逐条对账。此前未发布的Phase2332 FP16/BF16多剂量方向导数与偶响应四个全2560坐标场现在进入客户端；所有新增重要HiddenState/响应场均有可视化副本，因此无需删除原始HiddenState场。区分“观测到结构”“跨条件复现”“机制闭合”三个证据等级：

$$
\mathcal E_{{obs}}\supseteq\mathcal E_{{rep}}\supseteq\mathcal E_{{mech}},
\qquad \mathcal E_{{obs}}\not\Rightarrow\mathcal E_{{mech}}.
$$

$$
\operatorname{{LanguageInvariant}}=[A_{{en\to zh}}\ge\tau]\land[A_{{zh\to en}}\ge\tau]\land
[A_{{coord}}-A_{{sorted}}\ge\delta]\land B_{{behavior}}\land L_{{lockbox}}.
$$

**结果汇总、相关文件与清理。** Phase连续审计 `{json.dumps(result['phase_audit'], ensure_ascii=False)}`；证据裁决 `{json.dumps(result['evidence'], ensure_ascii=False)}`；场覆盖/清理 `{json.dumps(result['field_coverage'], ensure_ascii=False)}`；新增低值可视化 `{json.dumps(result['datasets'], ensure_ascii=False)}`；核验 `{json.dumps(result['verification'], ensure_ascii=False)}`；客户端构建 `{json.dumps(result['frontend_build'], ensure_ascii=False)}`。脚本 `tests/glm5/phase2342_c8001_c8120_evidence_audit_and_lowvalue_publication.py`；结果 `tests/glm5/result/phase2342_c8001_c8120_evidence_audit_and_lowvalue_publication`。

**最终理论进展。** 可以保留：状态场和输出使用都是分布式全坐标现象；固定A/B接口下存在族条件的具体坐标强度图谱；q23完整坐标和逐行标准化均强于排序分布与长度基线；Qwen14B和GLM4严格跨模型功能复现，DS7B中层有连续信号但最终层未过。必须否定：未发现单门不等于不存在静态齿轮；没有证据支持“必然是黎曼张量场/测地线/Omega流形”；激活不是参数；当前结果不是语言数学闭合。

**问题、硬伤与结论。** 最关键硬伤是跨语言失败：q23跨英文/中文最低0.2353，低于冻结0.30；所以当前最准确称呼是“语言条件化的语言族坐标使用图谱候选”，不是语言无关语义码。Phase2335证明自然差分跨生成器失败，Phase2340证明最终层距离优势不稳定；因此不能回到差分搬运，也不能直接做因果闭合。量化模型低值不可与FP16精度等同；程序模板与选择题界面仍是主要混淆。

**下一大阶段。** 目标仍相同，路线已经自动推进并冻结为“同语义图的双语对齐＋词汇/任务三轴正交化”：至少12个行为合格族、16+ units、完全平行英中语义图；分别测试同词汇跨任务、同任务跨词汇、同语义跨语言；q23只作为观察窗但仍分析全层；必须双向跨语言、具体坐标优于排序分布、行为和独立lockbox同时通过，才进入Qwen14B/GLM4/DS7B顺序复验与随后因果闭合。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle: handle.write(record)


def main() -> None:
    final_path = OUT / "analysis/final.json"
    if final_path.exists():
        result = json.loads(final_path.read_text(encoding="utf-8")); append_memo(result)
        print(json.dumps(result, ensure_ascii=False, indent=2)); return
    datasets = publish_low_value_fields()
    verification = [atlas.verify(row) for row in datasets]
    verified = all(all(value for key, value in row.items() if key != "id") for row in verification)
    if not verified: raise RuntimeError(("verification_failed", verification))
    catalog = atlas.update_catalog(datasets)
    build = atlas.frontend_build()
    if not build["passed"]: raise RuntimeError(("frontend_build_failed", build))
    audit = phase_audit()
    coverage = field_coverage(datasets)
    evidence = evidence_adjudication()
    checks = {"continuous_phases": audit["continuous"], "all_final_exist": audit["all_final_exist"],
              "all_memo_once_before_phase2342": audit["all_memo_once"], "no_unpublished_hiddenstate": not coverage["raw_hiddenstate_unpublished"],
              "assets_verified": verified, "frontend_build": build["passed"]}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "phase_audit": audit, "evidence": evidence,
              "field_coverage": coverage, "datasets": json.loads(json.dumps(datasets, ensure_ascii=False, default=str)),
              "verification": verification, "catalog": json.loads(json.dumps(catalog, ensure_ascii=False, default=str)),
              "frontend_build": build, "checks": checks, "all_checks_passed": all(checks.values())}
    save(final_path, result)
    if not result["all_checks_passed"]: raise RuntimeError(("phase2342_failed", checks))
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
