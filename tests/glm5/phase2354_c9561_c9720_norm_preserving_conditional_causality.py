#!/usr/bin/env python3
"""Norm-preserving matched/wrong/permuted causal tournament for the q24 prompt atlas."""
from __future__ import annotations

import gc
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]; TESTS = ROOT / "tests/glm5"; RESULT = TESTS / "result"
P2352 = RESULT / "phase2352_c9241_c9400_natural_multifuture_transient_field"
P2353 = RESULT / "phase2353_c9401_c9560_conditional_equivalence_route_competition"
OUT = RESULT / "phase2354_c9561_c9720_norm_preserving_conditional_causality"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"; VIS = ROOT / "frontend/public/vis_data/research_kernel"
PASSPORT = VIS / "c9401_qwen4b_conditional_equivalence_prompt_passport.float32.npy"
OUTCOMES = OUT / "raw/intervention_outcomes.float32.npy"; INDEX = OUT / "index/intervention_rows.jsonl"
PROGRESS = OUT / "raw/progress.json"; PHASE = 2354; CAMPAIGN = "C9561-C9720"; DOSES = (0.02, 0.05, 0.10)

sys.path.insert(0, str(TESTS))
import model_utils  # noqa: E402
import phase2315_c5041_c5100_active_response_contract as io  # noqa: E402
import phase2316_c5101_c5160_qwen4b_active_baseline as baseline  # noqa: E402
import phase2319_c5321_c5400_active_response_atlas_cleanup as atlas  # noqa: E402

if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def close_memmap(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None: mmap.close()


def modules(model) -> list[Any]: return [model.model.embed_tokens, *list(model.model.layers), model.model.norm]


def derive_directions(rows: list[dict], labels: list[str]) -> dict[str, Any]:
    matrix = np.load(PASSPORT, mmap_mode="r"); n = len(rows); signed = matrix[3 * n:4 * n]
    discovery = {}; confirmation = {}
    grand_d = signed[[i for i, r in enumerate(rows) if r["partition"] == "discovery"]].mean(axis=0, dtype=np.float64)
    grand_c = signed[[i for i, r in enumerate(rows) if r["partition"] == "confirmation"]].mean(axis=0, dtype=np.float64)
    for label in labels:
        idx_d = [i for i, r in enumerate(rows) if r["family"] == label and r["partition"] == "discovery"]
        idx_c = [i for i, r in enumerate(rows) if r["family"] == label and r["partition"] == "confirmation"]
        discovery[label] = (signed[idx_d].mean(axis=0, dtype=np.float64) - grand_d).astype(np.float32)
        confirmation[label] = (signed[idx_c].mean(axis=0, dtype=np.float64) - grand_c).astype(np.float32)
    close_memmap(matrix)
    return {"discovery": discovery, "confirmation": confirmation,
            "grand_discovery": grand_d.astype(np.float32), "grand_confirmation": grand_c.astype(np.float32)}


def specs(qpoint: int) -> list[dict]:
    return ([{"name": "baseline", "dose": 0.0, "qpoint": qpoint}]
            + [{"name": "matched_suppress", "dose": d, "qpoint": qpoint} for d in DOSES]
            + [{"name": "wrong_family_suppress", "dose": 0.05, "qpoint": qpoint},
               {"name": "permuted_suppress", "dose": 0.05, "qpoint": qpoint},
               {"name": "random_equal_l2_suppress", "dose": 0.05, "qpoint": qpoint},
               {"name": "wrong_layer_suppress", "dose": 0.05, "qpoint": max(1, qpoint - 4)},
               {"name": "independent_rescue", "dose": 0.05, "qpoint": qpoint},
               {"name": "matched_invoke", "dose": 0.05, "qpoint": qpoint}])


def tangent_edit(vector: torch.Tensor, direction: torch.Tensor, signed_dose: float) -> torch.Tensor:
    original_norm = torch.linalg.vector_norm(vector).clamp_min(1e-8)
    direction = direction - torch.dot(direction, vector) / (original_norm * original_norm) * vector
    direction = direction / torch.linalg.vector_norm(direction).clamp_min(1e-8) * original_norm
    edited = vector + signed_dose * direction
    return edited / torch.linalg.vector_norm(edited).clamp_min(1e-8) * original_norm


def score_candidate(model, device, batch: list[dict], key: str, capture_context: dict, pad: int) -> np.ndarray:
    combined = [r["prompt_ids"] + r[key] for r in batch]; ids, mask, positions = baseline.pad_right(combined, device, pad)
    capture_context["ends"] = torch.tensor([len(r["prompt_ids"]) - 1 for r in batch], dtype=torch.long, device=device)
    capture_context["rows"] = batch
    output = model.model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, return_dict=True)
    scores = []
    for local, row in enumerate(batch):
        answer = row[key]; start = len(row["prompt_ids"]); pos = torch.arange(start - 1, start + len(answer) - 1, device=device)
        logits = model.lm_head(output.last_hidden_state[local, pos]).float(); token_ids = torch.tensor(answer, dtype=torch.long, device=device)
        scores.append(float(F.log_softmax(logits, dim=-1)[torch.arange(len(answer), device=device), token_ids].mean()))
    return np.asarray(scores, dtype=np.float32)


def collect(model, device, rows: list[dict], labels: list[str], directions: dict[str, Any], qpoint: int, batch_size: int = 12) -> dict:
    selected = [r for r in rows if r["partition"] == "fresh_lockbox" and r["surface"] == "natural" and r["state"] == 0]
    all_specs = specs(qpoint); expected = len(selected) * len(all_specs)
    OUTCOMES.parent.mkdir(parents=True, exist_ok=True)
    outcomes = np.lib.format.open_memmap(OUTCOMES, mode="w+", dtype=np.float32, shape=(expected, 4)); metadata = []
    wrong = {label: labels[(i + 1) % len(labels)] for i, label in enumerate(labels)}
    rng = np.random.default_rng(9561)
    permuted = {label: rng.permutation(directions["discovery"][label]).copy() for label in labels}
    random_dirs = {}
    for label in labels:
        value = rng.standard_normal(len(directions["discovery"][label])).astype(np.float32)
        value *= np.linalg.norm(directions["discovery"][label]) / max(np.linalg.norm(value), 1e-8); random_dirs[label] = value
    pad = int(model.config.pad_token_id or model.config.eos_token_id or 0); cursor = 0; context: dict[str, Any] = {}
    with torch.inference_mode():
        for spec_index, spec in enumerate(all_specs):
            context["spec"] = spec
            def hook(_module, _inputs, value):
                tensor = value[0] if isinstance(value, tuple) else value; edited = tensor.clone(); active = context["spec"]
                if active["name"] != "baseline":
                    for local, row in enumerate(context["rows"]):
                        pos = int(context["ends"][local]); family = row["family"]; vector = edited[local, pos].float()
                        direction = directions["discovery"][family]
                        if active["name"] == "wrong_family_suppress": direction = directions["discovery"][wrong[family]]
                        elif active["name"] == "permuted_suppress": direction = permuted[family]
                        elif active["name"] == "random_equal_l2_suppress": direction = random_dirs[family]
                        dose = float(active["dose"]); sign = 1.0 if active["name"] == "matched_invoke" else -1.0
                        vector = tangent_edit(vector, torch.from_numpy(direction).to(vector.device), sign * dose)
                        if active["name"] == "independent_rescue":
                            vector = tangent_edit(vector, torch.from_numpy(directions["confirmation"][family]).to(vector.device), dose)
                        edited[local, pos] = vector.to(edited.dtype)
                return (edited, *value[1:]) if isinstance(value, tuple) else edited
            handle = modules(model)[int(spec["qpoint"])].register_forward_hook(hook)
            try:
                for start in range(0, len(selected), batch_size):
                    batch = selected[start:start + batch_size]
                    good = score_candidate(model, device, batch, "target_ids", context, pad)
                    bad = score_candidate(model, device, batch, "wrong_ids", context, pad); margin = good - bad
                    outcomes[cursor:cursor + len(batch)] = np.stack([good, bad, margin, margin > 0], axis=1)
                    for row in batch:
                        metadata.append({"case_id": row["case_id"], "family": row["family"], "language": row["language"],
                                         "query": row["query"], "depth": row["depth"], "unit": row["unit"],
                                         "intervention": spec["name"], "dose": spec["dose"], "qpoint": spec["qpoint"],
                                         "atlas_qpoint": qpoint, "wrong_control_family": wrong[row["family"]]})
                    cursor += len(batch); outcomes.flush()
            finally: handle.remove()
            save(PROGRESS, {"completed_specs": spec_index + 1, "cursor": cursor}); print(f"[phase2354] {spec_index + 1}/{len(all_specs)} {spec['name']}", flush=True)
    outcomes.flush(); close_memmap(outcomes); io.write_rows(INDEX, metadata)
    return {"rows": len(selected), "specs": all_specs, "outcome_shape": [expected, 4], "norm_preserving": True}


def analyze(labels: list[str]) -> dict:
    rows = io.read_rows(INDEX); outcomes = np.load(OUTCOMES, mmap_mode="r"); by_case = defaultdict(dict)
    for i, row in enumerate(rows): by_case[row["case_id"]][(row["intervention"], float(row["dose"]))] = i
    spec_keys = sorted({key for values in by_case.values() for key in values}); aggregate = {}; families = {}
    for key in spec_keys:
        idx = np.asarray([v[key] for v in by_case.values()]); base = np.asarray([v[("baseline", 0.0)] for v in by_case.values()])
        delta = outcomes[idx, 2] - outcomes[base, 2]
        aggregate[f"{key[0]}:{key[1]:.2f}"] = {"rows": len(idx), "accuracy": float(np.mean(outcomes[idx, 3])),
                                                  "mean_margin": float(np.mean(outcomes[idx, 2])),
                                                  "mean_margin_delta": float(np.mean(delta)), "median_margin_delta": float(np.median(delta))}
    for family in labels:
        case_ids = [case for case, values in by_case.items() if rows[values[("baseline", 0.0)]]["family"] == family]
        families[family] = {}
        for key in spec_keys:
            idx = np.asarray([by_case[c][key] for c in case_ids]); base = np.asarray([by_case[c][("baseline", 0.0)] for c in case_ids])
            families[family][f"{key[0]}:{key[1]:.2f}"] = {"accuracy": float(np.mean(outcomes[idx, 3])),
                "mean_margin_delta": float(np.mean(outcomes[idx, 2] - outcomes[base, 2]))}
    d2 = aggregate["matched_suppress:0.02"]["mean_margin_delta"]; d5 = aggregate["matched_suppress:0.05"]["mean_margin_delta"]
    d10 = aggregate["matched_suppress:0.10"]["mean_margin_delta"]
    controls = [aggregate[k]["mean_margin_delta"] for k in ("wrong_family_suppress:0.05", "permuted_suppress:0.05", "random_equal_l2_suppress:0.05")]
    selective = sum(families[f]["matched_suppress:0.05"]["mean_margin_delta"] < min(
        families[f][k]["mean_margin_delta"] for k in ("wrong_family_suppress:0.05", "permuted_suppress:0.05", "random_equal_l2_suppress:0.05")) for f in labels)
    gate = {"baseline_behavior": aggregate["baseline:0.00"]["accuracy"] >= 0.75,
            "negative_monotonic_dose": d2 < 0 and d5 < d2 and d10 < d5,
            "matched_stronger_than_controls": d5 < min(controls),
            "matched_stronger_than_wrong_layer": d5 < aggregate["wrong_layer_suppress:0.05"]["mean_margin_delta"],
            "independent_rescue": aggregate["independent_rescue:0.05"]["mean_margin_delta"] > d5,
            "matched_invocation_positive": aggregate["matched_invoke:0.05"]["mean_margin_delta"] > 0,
            "family_selectivity_count": int(selective), "family_selectivity_pass": selective >= int(np.ceil(0.75 * len(labels)))}
    gate["causal_candidate_passed"] = all(v for k, v in gate.items() if k != "family_selectivity_count")
    close_memmap(outcomes); return {"aggregate": aggregate, "families": families, "gate": gate}


def publish(directions: dict[str, Any], labels: list[str], qpoint: int) -> dict:
    values = []; metadata = []
    for view in ("discovery", "confirmation"):
        for family in labels:
            values.append(directions[view][family]); metadata.append({"family": family, "view": f"{view}_signed_direction", "qpoint": qpoint})
    values = np.stack(values); dataset_id = "c9561_qwen4b_norm_preserving_family_directions"
    binary = VIS / f"{dataset_id}.float32.npy"; out = atlas.create_binary(binary.name, *values.shape, np.float32); out[:] = values
    out.flush(); close_memmap(out)
    return atlas.write_metadata(dataset_id, "Qwen3-4B independent family tangent directions", binary, metadata,
        "Qwen3-4B-FP16", "norm_preserving_family_directions_v1", "exploratory intervention directions; causal gate reported separately",
        "12 discovery-derived and 12 independent-confirmation-derived signed directions",
        "all 2560 coordinates retained; intervention projects each direction tangent to each sample state",
        {"phase": PHASE, "campaign": CAMPAIGN, "coordinate_count": 2560, "qpoint": qpoint, "no_topk": True})


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"): return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 保范数条件匹配干预—独立救援—控制锦标赛（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 使用Phase2353 q24有符号prompt图谱，仅由discovery训练族方向，confirmation独立生成救援方向；在384条fresh-lockbox、natural、state0样本上测试baseline、匹配抑制三剂量、错族、坐标置乱、等L2随机、错层、独立救援和匹配调用。每个方向先投影到样本状态的正交切向，再扰动并恢复原L2范数，避免Phase2348乘法删除直接改变RMSNorm前尺度。

$$
t_f=d_f-\frac{{d_f^\top h}}{{\|h\|_2^2}}h,\qquad
h'=\|h\|_2\frac{{h\pm\alpha\|h\|_2t_f/\|t_f\|_2}}{{\|h\pm\alpha\|h\|_2t_f/\|t_f\|_2\|_2}}.
$$

**结果汇总。** 采集 `{json.dumps(result['collection'], ensure_ascii=False)}`；聚合/逐族结果与门槛 `{json.dumps(result['analysis'], ensure_ascii=False)}`；方向热力图 `{json.dumps(result['dataset'], ensure_ascii=False)}`。

**相关文件。** 脚本 `tests/glm5/phase2354_c9561_c9720_norm_preserving_conditional_causality.py`；结果 `tests/glm5/result/phase2354_c9561_c9720_norm_preserving_conditional_causality`；客户端`c9561`。

**理论进展、问题硬伤与结论。** 该实验修复了尺度混淆，但“自然切空间/流形切向”仍是过度命名：这里只是欧氏空间中对当前向量的正交、保范数扰动。只有负向剂量响应、强于所有同层控制与错层、独立救援、正向调用和至少75%逐族选择性全部通过，才称局部因果候选；任何一项失败都不能宣布齿轮。另补记Phase2353：生成路线的最佳step=0，所有step>0最低准确率只到机会水平，因此生成瞬态门已纠正为失败。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle: handle.write(text)


def main() -> None:
    final_path = OUT / "analysis/final.json"
    if final_path.exists():
        result = json.loads(final_path.read_text(encoding="utf-8")); append_memo(result); print(json.dumps(result, ensure_ascii=False, indent=2)); return
    p2353 = json.loads((P2353 / "analysis/final.json").read_text(encoding="utf-8")); qpoint = int(p2353["prompt"]["selected_qpoint"])
    labels = list(p2353["labels"]); rows = io.read_rows(P2352 / "material/natural_multifuture_graphs.jsonl"); directions = derive_directions(rows, labels)
    freeze = {"frozen_before_model_load": True, "qpoint": qpoint, "doses": list(DOSES), "labels": labels,
              "gate": "dose+same-layer controls+wrong-layer+independent rescue+invocation+75% family selectivity"}
    save(OUT / "config/frozen_contract.json", freeze); model = None
    try:
        model, _tokenizer, device = model_utils.load_model("qwen3", dtype=torch.float16, use_8bit=False)
        collection = collect(model, device, rows, labels, directions, qpoint)
    finally:
        if model is not None: model_utils.release_model(model)
        del model; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    analysis = analyze(labels); dataset = publish(directions, labels, qpoint); verification = atlas.verify(dataset)
    verified = all(v for k, v in verification.items() if k != "id"); catalog = atlas.update_catalog([dataset]); build = atlas.frontend_build()
    checks = {"outcome_shape": collection["outcome_shape"] == [3840, 4], "asset": verified, "frontend_build": build["passed"]}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "freeze": freeze, "collection": collection, "analysis": analysis,
              "dataset": json.loads(json.dumps(dataset, ensure_ascii=False, default=str)), "verification": verification,
              "catalog": json.loads(json.dumps(catalog, ensure_ascii=False, default=str)), "frontend_build": build,
              "checks": checks, "all_checks_passed": all(checks.values())}
    save(final_path, result)
    if not result["all_checks_passed"]: raise RuntimeError(("phase2354_failed", checks))
    append_memo(result); print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
