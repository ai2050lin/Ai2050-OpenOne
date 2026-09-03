#!/usr/bin/env python3
"""Output-conditioned full-coordinate VJP pilot on 64 fresh multi-family rows."""
from __future__ import annotations

import gc
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P2435 = RESULT / "phase2435_c33841_c34160_hypergraph_material_fourmodel_behavior/qwen4b"
P2436 = RESULT / "phase2436_c34161_c34480_qwen4b_hypergraph_fullfield"
OUT = RESULT / "phase2447_c37681_c38000_output_conditioned_vjp_pilot"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE = 2447
CAMPAIGN = "C37681-C38000"
EVENTS = ("query_end", "answer_boundary")
FIELDS = ("gradient", "state_times_gradient")
MEASURES = ("language_coordinate", "language_shift791", "language_family_permuted", "energy")
SHIFT = 791

sys.path.insert(0, str(TESTS))
import phase2389_c19121_c19440_crossmodel_autonomous_capability as capability  # noqa: E402
import phase2390_c19441_c19760_qwen_semantic_lexical_fullfield as field_utils  # noqa: E402


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def close(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None: mmap.close()


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    a, b = np.asarray(left, dtype=np.float64).reshape(-1), np.asarray(right, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > 1e-30 else 0.0


def selected_rows() -> list[dict]:
    all_rows = read_rows(P2435 / "index/trajectory_rows.jsonl")
    return [row for row in all_rows if row["variant"] == "valid" and int(row["unit"]) == 5 and row["surface"] == "natural"]


def capture(rows: list[dict]) -> dict:
    raw = OUT / "raw"; raw.mkdir(parents=True, exist_ok=True)
    gradient_path = raw / "output_margin_vjp.float32.npy"
    contribution_path = raw / "output_margin_state_times_vjp.float32.npy"
    margin_path = raw / "live_margin.float32.npy"
    progress = raw / "progress.json"
    shape = (len(rows), 38, 2, 2560)
    gradients = np.lib.format.open_memmap(gradient_path, mode="r+" if gradient_path.exists() else "w+",
                                          dtype=np.float32, shape=shape)
    contributions = np.lib.format.open_memmap(contribution_path, mode="r+" if contribution_path.exists() else "w+",
                                              dtype=np.float32, shape=shape)
    margins = np.lib.format.open_memmap(margin_path, mode="r+" if margin_path.exists() else "w+",
                                       dtype=np.float32, shape=(len(rows),))
    completed = int(json.loads(progress.read_text(encoding="utf-8"))["completed"]) if progress.exists() else 0
    model = tokenizer = None
    if completed < len(rows):
        model, tokenizer, _ = capability.load_model("qwen4b")
        model.eval()
        for parameter in model.parameters(): parameter.requires_grad_(False)
        modules = field_utils.modules(model)
        captures: dict[int, torch.Tensor] = {}
        handles = []
        for qpoint, module in enumerate(modules):
            def hook(_module, _inputs, result, qpoint=qpoint):
                tensor = result[0] if isinstance(result, tuple) else result
                if qpoint == 0 and not tensor.requires_grad:
                    tensor.requires_grad_(True)
                tensor.retain_grad(); captures[qpoint] = tensor
            handles.append(module.register_forward_hook(hook))
        device = model.get_input_embeddings().weight.device
    else:
        modules = []; captures = {}; handles = []; device = None
    try:
        for index in range(completed, len(rows)):
            row = rows[index]
            ids = torch.tensor([row["prompt_ids"]], dtype=torch.long, device=device)
            mask = torch.ones_like(ids); positions = torch.arange(ids.shape[1], device=device)[None]
            captures.clear()
            with torch.enable_grad():
                result = model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, return_dict=True)
                target, foil = int(row["target_ids"][0]), int(row["foil_ids"][0])
                margin = result.logits[0, -1, target] - result.logits[0, -1, foil]
                margin.backward()
            event_map = {event["event"]: int(event["token_index"]) for event in row["event_tokens"]}
            token_indices = (event_map["query_end"], event_map["answer_boundary"])
            for qpoint in range(len(modules)):
                state = captures[qpoint][0, list(token_indices)].detach().float().cpu().numpy()
                grad = captures[qpoint].grad[0, list(token_indices)].detach().float().cpu().numpy()
                gradients[index, qpoint] = grad; contributions[index, qpoint] = state * grad
            margins[index] = float(margin.detach().float().cpu())
            gradients.flush(); contributions.flush(); margins.flush()
            save(progress, {"completed": index + 1, "shape": shape, "method": "parameters frozen; embedding output is VJP leaf"})
            if (index + 1) % 8 == 0 or index + 1 == len(rows):
                print(f"[phase2447 VJP] {index + 1}/{len(rows)}", flush=True)
            del result, margin, ids, mask, positions
    finally:
        for handle in handles: handle.remove()
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gradients.flush(); contributions.flush(); margins.flush(); close(gradients); close(contributions); close(margins)
    write_rows(OUT / "index/vjp_rows.jsonl", [{key: row[key] for key in ("case_id", "config_id", "family", "unit", "language",
                                                                                       "surface", "direction", "query_role", "answer", "foil")}
                                                   for row in rows])
    return {"gradient": str(gradient_path), "contribution": str(contribution_path), "margin": str(margin_path),
            "shape": list(shape), "rows": len(rows), "bytes": gradient_path.stat().st_size + contribution_path.stat().st_size + margin_path.stat().st_size,
            "inference": "Qwen3-4B BF16 CUDA, frozen parameters, exact first-token margin VJP", "storage": "float32 full coordinates"}


def pair_index(rows: list[dict]) -> tuple[list[dict], list[tuple[int, int]]]:
    configs = sorted({row["config_id"] for row in rows})
    lookup = {(row["config_id"], row["query_role"]): index for index, row in enumerate(rows)}
    meta, pairs = [], []
    for config in configs:
        source, target = lookup[(config, "source")], lookup[(config, "target")]
        meta.append({key: rows[source][key] for key in ("config_id", "family", "language", "direction")})
        pairs.append((source, target))
    return meta, pairs


def analyze(rows: list[dict], collection: dict) -> dict:
    gradient = np.load(collection["gradient"], mmap_mode="r")
    contribution = np.load(collection["contribution"], mmap_mode="r")
    margins = np.load(collection["margin"], mmap_mode="r")
    meta, pairs = pair_index(rows)
    families = sorted({row["family"] for row in meta})
    permutation = np.random.default_rng(2447).permutation(len(families))
    lookup = {(row["family"], row["language"], int(row["direction"])): index for index, row in enumerate(meta)}
    metrics = np.zeros((2, 38, 2, len(MEASURES)), dtype=np.float32)
    for fi, field in enumerate((gradient, contribution)):
        role_difference = np.stack([np.asarray(field[target] - field[source], dtype=np.float32) for source, target in pairs])
        for qpoint in range(38):
            for event in range(2):
                state = role_difference[:, qpoint, event]
                en = np.stack([np.mean([state[lookup[(family, "en", direction)]] for direction in (0, 1)], axis=0) for family in families])
                zh = np.stack([np.mean([state[lookup[(family, "zh", direction)]] for direction in (0, 1)], axis=0) for family in families])
                metrics[fi, qpoint, event] = (
                    np.mean([cosine(en[i], zh[i]) for i in range(8)]),
                    np.mean([cosine(en[i], np.roll(zh[i], SHIFT)) for i in range(8)]),
                    np.mean([cosine(en[i], zh[permutation[i]]) for i in range(8)]),
                    np.mean(state.astype(np.float64) ** 2))
    derived = OUT / "derived"; derived.mkdir(parents=True, exist_ok=True)
    np.save(derived / "vjp_crosslanguage_metrics.float32.npy", metrics)
    summary = {}
    for fi, field_name in enumerate(FIELDS):
        summary[field_name] = {}
        for event, event_name in enumerate(EVENTS):
            physical = metrics[fi, :, event, 0] - metrics[fi, :, event, 1]
            identity = metrics[fi, :, event, 0] - metrics[fi, :, event, 2]
            best = int(np.argmax(physical + identity))
            summary[field_name][event_name] = {"best_qpoint": best,
                                               "language_coordinate": float(metrics[fi, best, event, 0]),
                                               "shift791": float(metrics[fi, best, event, 1]),
                                               "family_permuted": float(metrics[fi, best, event, 2]),
                                               "physical_advantage": float(physical[best]),
                                               "family_identity_advantage": float(identity[best]),
                                               "energy": float(metrics[fi, best, event, 3])}
    archived = np.asarray(contribution[:, 37, 1], dtype=np.float64).sum(axis=1)
    truth = np.asarray(margins, dtype=np.float64)
    residual = archived - truth
    closure = {"correlation": float(np.corrcoef(archived, truth)[0, 1]),
               "relative_rmse": float(np.sqrt(np.mean(residual ** 2)) / max(np.sqrt(np.mean(truth ** 2)), 1e-30)),
               "max_abs": float(np.max(np.abs(residual)))}
    for value in (gradient, contribution, margins): close(value)
    return {"pairs": len(pairs), "families": families, "metrics": str(derived / "vjp_crosslanguage_metrics.float32.npy"),
            "summary": summary, "final_norm_answer_contribution_closure": closure}


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"): return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 自动续研——第一分歧token输出条件VJP的逐层全坐标试验（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 针对Phase2445“固定跨样本映射无法编译输出”的硬伤，改用逐样本输出条件链式法则。对64条fresh/valid/natural八族中英双方向双角色行，冻结Qwen3-4B全部参数，以第一分歧token目标—foil logit margin为标量；反向保存embedding、36 block、final norm在query-end与answer-boundary的全部2560梯度及$H_i\partial m/\partial H_i$，float32不裁剪。再构造同配置target-role−source-role，比较同family中英、+791坐标错配与family置乱。

$$g_{{q,t,i}}=\frac{{\partial(\ell_a-\ell_b)}}{{\partial H_{{q,t,i}}}},\qquad
A_{{q,t,i}}=H_{{q,t,i}}g_{{q,t,i}}.$$

**结果汇总。** 采集 `{json.dumps(result['collection'], ensure_ascii=False)}`；跨语言输出条件场 `{json.dumps(result['analysis']['summary'], ensure_ascii=False)}`；final-norm恒等核对 `{json.dumps(result['analysis']['final_norm_answer_contribution_closure'], ensure_ascii=False)}`；裁决 `{json.dumps(result['adjudication'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2447_c37681_c38000_output_conditioned_vjp_pilot.py`；64×38×2×2560梯度与state×gradient、逐行margin、索引和跨语言指标位于同名结果目录。

**分析与理论进展。** VJP不是用外部统计模型猜输出，而是询问这个具体token竞争通过真实计算图对每层每坐标有多敏感。final norm answer位置的$H_i g_i$应直接重建margin，是数值质量门；更早query位置的梯度反映后续层怎样从上下文读取信息。若其跨语言同坐标身份稳定，说明输出目标确实选择了共享物理坐标路径；若不稳定，则内部semantic passport与输出路径仍分离。

**问题硬伤与结论。** 梯度是局部一阶敏感度，不是必要性、有限扰动因果或完整贡献；$H_i g_i$在非线性中间层不满足逐层守恒。64条均为valid且同一unit，只是新范式试验，不足以证明语义特异性；必须在后续扩展valid/broken和multiunit后才可命名输出条件齿轮。
"""
    with MEMO.open("a", encoding="utf-8", newline="") as stream: stream.write(text)


def main() -> None:
    final = OUT / "analysis/final.json"
    if final.exists():
        result = json.loads(final.read_text(encoding="utf-8")); append_memo(result); print(json.dumps(result, ensure_ascii=False, indent=2)); return
    rows = selected_rows()
    collection = capture(rows)
    analysis = analyze(rows, collection)
    gradient_query = analysis["summary"]["gradient"]["query_end"]
    contribution_query = analysis["summary"]["state_times_gradient"]["query_end"]
    closure = analysis["final_norm_answer_contribution_closure"]
    adjudication = {"vjp_numeric_closure": closure["correlation"] > .995 and closure["relative_rmse"] < .08,
                    "query_gradient_crosslanguage_coordinate_candidate": gradient_query["physical_advantage"] > 0 and gradient_query["family_identity_advantage"] > 0,
                    "query_attribution_crosslanguage_coordinate_candidate": contribution_query["physical_advantage"] > 0 and contribution_query["family_identity_advantage"] > 0,
                    "output_conditioned_semantic_gear_proven": False}
    checks = {"rows_64": collection["rows"] == 64, "shape": collection["shape"] == [64, 38, 2, 2560],
              "pairs_32": analysis["pairs"] == 32, "eight_families": len(analysis["families"]) == 8,
              "numeric_closure": adjudication["vjp_numeric_closure"],
              "all_files": all(Path(path).exists() for path in (collection["gradient"], collection["contribution"], collection["margin"], analysis["metrics"])),
              "finite": all(math.isfinite(value) for field in analysis["summary"].values() for event in field.values() for value in event.values()),
              "claim_boundary": not adjudication["output_conditioned_semantic_gear_proven"]}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "collection": collection, "analysis": analysis,
              "adjudication": adjudication, "checks": checks, "all_checks_passed": all(checks.values())}
    save(final, result); append_memo(result)
    if not result["all_checks_passed"]: raise RuntimeError(checks)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
