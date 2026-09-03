#!/usr/bin/env python3
"""C655: full-coordinate upstream trajectory response transport without diagonal fitting."""
from __future__ import annotations

import gc
import itertools
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))

import phase2163_c629_model_specific_worker as loader
import phase2187_c651_c653_qwen14_prospective_concept_bridge as c651
import phase2188_c654_corrected_upstream_concept_bridge as c654

PHASE = 2189
CAMPAIGN = "C655"
OUT = TESTS / "result/phase2189_c655_full_trajectory_response_transport"
SOURCE = TESTS / "result/phase2187_c651_c653_qwen14_prospective_natural_concept_bridge"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
ROLES = c651.ROLES
Q_VALUES = (0, 8, 16, 25)
TARGET_Q = 33
GAIN_GATE = 0.02


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()]


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False) + "\n")


def close_mmap(array: Any) -> None:
    mmap = getattr(array, "_mmap", None)
    if mmap is not None:
        mmap.close()


def finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, list):
        return all(finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def freeze(rows: list[dict], behavior: list[dict]) -> None:
    for part in ("protocol", "material", "behavior", "raw", "analysis", "audit"):
        (OUT / part).mkdir(parents=True, exist_ok=True)
    protocol = {
        "phase": PHASE, "campaign": CAMPAIGN,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": "Qwen3-14B FP16 disk offload",
        "source_contract": "C651 frozen English material and behavior; C654 endpoint-leak correction",
        "object": "predict q33 boundary concept-pair response from the complete upstream trajectory, without coordinatewise regression",
        "trajectory": "q0/q8/q16/q25 x all six roles x all 5120 signed coordinates",
        "representations": {
            "raw": "unchanged full trajectory",
            "block_unit": "each checkpoint-role block divided by its own L2 norm; no coordinate removed",
            "increments": "q8-q0, q16-q8 and q25-q16 for every role and coordinate",
            "signed_sqrt": "sign(x)*sqrt(abs(x)) to retain sign while increasing low-value visibility",
        },
        "response_transport": ["nearest discovery trajectory", "inverse-distance barycenter of three discovery trajectories"],
        "selection": "discovery references only; confirmation selects representation and transport once; lockbox reveals once",
        "baselines": ["zero", "discovery mean target response", "direct q25-boundary response copy"],
        "gate": "selected method beats the strongest simple baseline by >=0.02 NRMSE on confirmation and lockbox",
        "causal": "only if prediction passes: predicted q33 response versus exact q33, no patch, wrong response and sign reversal; doses 0.5/1/1.5",
        "forbidden": "Attention/MLP/weights/gradients/Top-K/PCA/projection/post-unblind changes",
    }
    if not (OUT / "protocol/preregistration.json").exists():
        save(OUT / "protocol/preregistration.json", protocol)
    write_rows(OUT / "material/material.jsonl", rows)
    write_rows(OUT / "behavior/reused_behavior.jsonl", behavior)


def capture(model, device, compiled: list[dict], dim: int) -> np.memmap:
    modules = [model.model.embed_tokens, *list(model.model.layers), model.model.norm]
    path = OUT / "raw/all_role_field.float16.npy"
    field = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16,
                                     shape=(len(compiled), len(modules), len(ROLES), dim))
    captured: list[torch.Tensor] = []
    handles = [module.register_forward_hook(
        lambda _m, _a, output: captured.append(output[0] if isinstance(output, tuple) else output))
        for module in modules]
    try:
        for row_i, item in enumerate(compiled):
            ids = torch.tensor([item["prompt_ids"]], dtype=torch.long, device=device)
            mask = torch.ones_like(ids)
            pos = torch.arange(ids.shape[1], device=device)[None]
            captured.clear()
            with torch.inference_mode():
                model(input_ids=ids, attention_mask=mask, position_ids=pos,
                      use_cache=False, return_dict=True)
            for q, tensor in enumerate(captured):
                values = tensor[0].float().cpu().numpy().astype(np.float16)
                for role_i, role in enumerate(ROLES):
                    field[row_i, q, role_i] = values[int(item["role_positions"][role][-1])]
            print(f"[C655 trajectory field] {row_i + 1}/{len(compiled)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    field.flush()
    return field


def build_samples(partition: str, field: np.ndarray, row_index: dict[str, int],
                  behavior_map: dict[str, dict]) -> list[dict]:
    answer = []
    q_indices = np.asarray(Q_VALUES)
    boundary_i = ROLES.index("boundary")
    for pair_i, (a, b) in enumerate(c651.partition_pairs(partition)):
        a_case = c651.make_row(a, "en")["case_id"]
        b_case = c651.make_row(b, "en")["case_id"]
        if not all(behavior_map[case]["candidate_correct"] and
                   behavior_map[case]["generated_correct"] for case in (a_case, b_case)):
            continue
        ai, bi = row_index[a_case], row_index[b_case]
        trajectory = (field[bi, q_indices].astype(np.float32) -
                      field[ai, q_indices].astype(np.float32))
        target = (field[bi, TARGET_Q, boundary_i].astype(np.float32) -
                  field[ai, TARGET_Q, boundary_i].astype(np.float32))
        answer.append({"pair_index": pair_i, "a": a, "b": b,
                       "trajectory": trajectory, "target": target})
    return answer


def represent(trajectory: np.ndarray, kind: str) -> np.ndarray:
    value = trajectory.astype(np.float32, copy=True)
    if kind == "raw":
        return value.reshape(-1)
    if kind == "block_unit":
        norms = np.sqrt(np.square(value).sum(axis=-1, keepdims=True))
        return (value / np.maximum(norms, 1e-8)).reshape(-1)
    if kind == "increments":
        return np.diff(value, axis=0).reshape(-1)
    if kind == "signed_sqrt":
        return (np.sign(value) * np.sqrt(np.abs(value))).reshape(-1)
    raise ValueError(kind)


def distances(query: np.ndarray, references: np.ndarray) -> np.ndarray:
    scale = max(float(query.size), 1.0)
    return np.square(references - query[None]).sum(axis=1) / scale


def transport(query: np.ndarray, references: np.ndarray, targets: np.ndarray,
              mode: str) -> np.ndarray:
    dist = distances(query, references)
    if mode == "nearest":
        return targets[int(np.argmin(dist))]
    if mode == "barycenter3":
        order = np.argsort(dist)[:min(3, len(dist))]
        weights = 1.0 / np.maximum(dist[order], 1e-8)
        weights /= weights.sum()
        return (targets[order] * weights[:, None]).sum(0)
    raise ValueError(mode)


def nrmse(pred: list[np.ndarray], truth: list[np.ndarray]) -> float:
    num = sum(float(np.square(a - b).sum()) for a, b in zip(pred, truth))
    den = sum(float(np.square(b).sum()) for b in truth)
    return float(math.sqrt(num / max(den, 1e-12)))


def evaluate(rows: list[dict], train: list[dict], representation: str,
             mode: str) -> tuple[list[np.ndarray], dict]:
    refs = np.stack([represent(row["trajectory"], representation) for row in train])
    targets = np.stack([row["target"] for row in train])
    pred = [transport(represent(row["trajectory"], representation), refs, targets, mode)
            for row in rows]
    truth = [row["target"] for row in rows]
    baselines = {
        "zero": [np.zeros_like(row["target"]) for row in rows],
        "discovery_mean": [targets.mean(0) for _row in rows],
        "q25_boundary_copy": [row["trajectory"][-1, ROLES.index("boundary")] for row in rows],
    }
    metrics = {"selected": {"nrmse": nrmse(pred, truth)}}
    metrics.update({name: {"nrmse": nrmse(values, truth)} for name, values in baselines.items()})
    return pred, metrics


def append_memo(result: dict) -> None:
    existing = MEMO.read_text(encoding="utf-8-sig")
    if f"## Phase {PHASE}:" in existing:
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    protocol = load(OUT / "protocol/preregistration.json")
    formulas = r"""$$
\mathcal T_{ab}=\{H_b(q,r,j)-H_a(q,r,j)\}_{q\in\{0,8,16,25\},r,j}
$$

$$
\mathcal I_{ab}=\{D_{ab}(q_{k+1},r,j)-D_{ab}(q_k,r,j)\}_{k,r,j}
$$

$$
\widehat Y(x)=\sum_{i\in N_3(x)}
\frac{(d(x,x_i)+\epsilon)^{-1}}{\sum_{k\in N_3(x)}(d(x,x_k)+\epsilon)^{-1}}Y_i
$$

$$
\mathrm{gain}=\min(\mathrm{NRMSE}_{0},\mathrm{NRMSE}_{\mu},
\mathrm{NRMSE}_{q25-copy})-\mathrm{NRMSE}_{transport}
$$"""
    text = f"""

## Phase {PHASE}: 全坐标上游轨迹的响应运输检验（C655） [{stamp}]

**阶段目标。** C654排除了恒等泄漏，并否定了当前逐坐标对角/单检查点最近响应桥。本期保持同一48概念、同一Qwen3-14B和同一英语行为账本，把四个上游检查点、六个角色、全部5120坐标联合成一个轨迹对象，检验不预设坐标独立性的实例响应运输。

**冻结合同。**

```json
{json.dumps(protocol, ensure_ascii=False, indent=2)}
```

**测试用例与算法。** discovery仍含24概念，confirmation与lockbox各12概念。每个族内两概念形成一条轨迹差分。并行观察原始全轨迹、逐块单位化轨迹、相邻检查点增量轨迹和保留正负号的平方根轨迹；每种表示分别用最近发现实例或三个最近实例的逆距离响应重心预测`q33/boundary`。所有表示均保留每个物理坐标，不做Top-K或投影。

**数学公式。**

{formulas}

**详细结果。**

```json
{json.dumps(result, ensure_ascii=False, indent=2)}
```

**分析、理论进展、问题与硬伤。** 该算法允许一个样本的完整层-角色-坐标图样共同决定调用哪个已见响应，不再为每个坐标独立拟合斜率；但它仍是有限实例运输，不是唯一因果电路，也不能表达训练集中从未出现的新算子。样本仅有10条合格discovery概念对、6条confirmation和6条lockbox，受控单词翻译与单模型边界仍在，人类自然度审查仍为`NA`。

**相关文件。** 脚本：`tests/glm5/phase2189_c655_full_trajectory_response_transport.py`；结果：`{OUT.relative_to(ROOT)}`；预注册：`{(OUT / 'protocol/preregistration.json').relative_to(ROOT)}`。

**结论与下一步授权。** {result['strict_interpretation']} {result['next_authorization']}
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def main() -> None:
    final_path = OUT / "analysis/final.json"
    if final_path.exists():
        print(json.dumps(load(final_path), ensure_ascii=False, indent=2))
        return
    rows = [row for row in read_rows(SOURCE / "material/material.jsonl")
            if row["target_language"] == "en"]
    selected_ids = {row["case_id"] for row in rows}
    behavior = [row for row in read_rows(SOURCE / "behavior/behavior.jsonl")
                if row["case_id"] in selected_ids]
    behavior_map = {row["case_id"]: row for row in behavior}
    freeze(rows, behavior)

    model = None
    field = None
    causal_rows: list[dict] = []
    dose_rows: list[dict] = []
    try:
        model, tokenizer, device, placement, loader_name = loader.load_model("qwen3_14b")
        compiled = c651.previous.compile_rows(tokenizer, rows)
        write_rows(OUT / "material/compiled.jsonl", compiled)
        by_case = {row["case_id"]: row for row in compiled}
        row_index = {row["case_id"]: i for i, row in enumerate(compiled)}
        dim = int(model.model.embed_tokens.weight.shape[1])
        field = capture(model, device, compiled, dim)
        train = build_samples("discovery", field, row_index, behavior_map)
        confirmation = build_samples("confirmation", field, row_index, behavior_map)
        lockbox = build_samples("lockbox", field, row_index, behavior_map)

        tournament = []
        for representation, mode in itertools.product(
                ("raw", "block_unit", "increments", "signed_sqrt"),
                ("nearest", "barycenter3")):
            _pred, measured = evaluate(confirmation, train, representation, mode)
            simple = min(measured[name]["nrmse"]
                         for name in ("zero", "discovery_mean", "q25_boundary_copy"))
            tournament.append({"representation": representation, "mode": mode,
                               "train_rows": len(train), "confirmation_rows": len(confirmation),
                               "metrics": measured,
                               "gain_over_best_simple": simple - measured["selected"]["nrmse"]})
        winner = min(tournament, key=lambda row: row["metrics"]["selected"]["nrmse"])
        selection = {**winner, "frozen_before_lockbox": True}
        save(OUT / "analysis/confirmation_tournament.json", tournament)
        save(OUT / "protocol/confirmation_selection_frozen.json", selection)
        lock_pred, lock_metrics = evaluate(lockbox, train, winner["representation"], winner["mode"])
        lock_simple = min(lock_metrics[name]["nrmse"]
                          for name in ("zero", "discovery_mean", "q25_boundary_copy"))
        lock_gain = lock_simple - lock_metrics["selected"]["nrmse"]
        prediction_pass = winner["gain_over_best_simple"] >= GAIN_GATE and lock_gain >= GAIN_GATE
        save(OUT / "analysis/lockbox_prediction.json", {
            "selection": selection, "rows": len(lockbox), "metrics": lock_metrics,
            "gain_over_best_simple": lock_gain, "passed": prediction_pass})

        if prediction_pass:
            for pair_i, (sample, predicted) in enumerate(zip(lockbox, lock_pred)):
                a, b = sample["a"], sample["b"]
                a_case = c651.make_row(a, "en")["case_id"]
                b_case = c651.make_row(b, "en")["case_id"]
                item = c651.previous._eval_item(by_case[a_case], by_case[b_case])
                wrong = lock_pred[(pair_i + 1) % len(lock_pred)]
                modes = {
                    "zero": [],
                    "predicted_q33": c651.previous._patches(item, TARGET_Q, [("boundary", predicted)]),
                    "exact_q33": c651.previous._patches(item, TARGET_Q, [("boundary", sample["target"])]),
                    "wrong_response": c651.previous._patches(item, TARGET_Q, [("boundary", wrong)]),
                    "wrong_direction": c651.previous._patches(item, TARGET_Q, [("boundary", -predicted)]),
                }
                for mode, patches in modes.items():
                    generated = c651.previous.translation._patched_generate(
                        model, tokenizer, item, patches, max_new_tokens=10)
                    causal_rows.append({"pair_index": pair_i, "mode": mode, **generated})
                for dose in (0.5, 1.0, 1.5):
                    generated = c651.previous.translation._patched_generate(
                        model, tokenizer, item,
                        c651.previous._patches(item, TARGET_Q, [("boundary", predicted * dose)]),
                        max_new_tokens=10)
                    dose_rows.append({"pair_index": pair_i, "dose": dose, **generated})
                print(f"[C655 causal] {pair_i + 1}/{len(lockbox)}", flush=True)
        write_rows(OUT / "raw/causal_generation.jsonl", causal_rows)
        write_rows(OUT / "raw/dose_generation.jsonl", dose_rows)
        field.flush()
        close_mmap(field)
        field = None
    finally:
        if field is not None:
            field.flush()
            close_mmap(field)
        loader.release_model("qwen3_14b", model)
        gc.collect()

    causal_rates = {mode: c654.mean_or_none([
        row["correct"] for row in causal_rows if row["mode"] == mode])
        for mode in sorted({row["mode"] for row in causal_rows})}
    dose_rates = {str(dose): c654.mean_or_none([
        row["correct"] for row in dose_rows if row["dose"] == dose])
        for dose in (0.5, 1.0, 1.5)}
    predicted_causal = bool(causal_rows) and (
        (causal_rates.get("predicted_q33") or 0.0) >= 0.50 and
        (causal_rates.get("predicted_q33") or 0.0) - max(
            causal_rates.get("zero") or 0.0, causal_rates.get("wrong_response") or 0.0,
            causal_rates.get("wrong_direction") or 0.0) >= 0.25)

    field_path = OUT / "raw/all_role_field.float16.npy"
    cleanup = {"deleted": [], "retained": [], "bytes_deleted": 0}
    if field_path.exists():
        cleanup["bytes_deleted"] = field_path.stat().st_size
        cleanup["deleted"].append(str(field_path.relative_to(ROOT)))
        field_path.unlink()
    save(OUT / "audit/cleanup.json", cleanup)
    strict = ("Full-coordinate trajectory transport predicted and selectively called a new q33 concept response."
              if prediction_pass and predicted_causal else
              "Neither coordinatewise fitting nor the frozen full-trajectory exemplar transports jointly predicted and called the lockbox response.")
    next_auth = ("当前同一冻结对象的预注册桥路线已经穷尽；下一阶段若继续，必须改变经验对象为多词短语/多表面中的样本内概念-语言联合响应，而不是继续调当前距离或阈值。")
    result = {
        "phase": PHASE, "campaign": CAMPAIGN, "status": "closed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "placement": placement, "loader": loader_name,
        "field_shape_before_cleanup": [48, 42, 6, 5120],
        "sample_counts": {"discovery_pairs": len(train),
                          "confirmation_pairs": len(confirmation), "lockbox_pairs": len(lockbox)},
        "candidate_methods": len(tournament), "selection": selection,
        "lockbox_metrics": lock_metrics, "lockbox_gain_over_best_simple": lock_gain,
        "prospective_prediction_pass": prediction_pass,
        "causal_rates": causal_rates, "dose_rates": dose_rates,
        "predicted_state_causal_pass": predicted_causal,
        "visualization": "not_added; C654 already visualizes the identical full-coordinate English lockbox response",
        "cleanup": cleanup, "new_foundational_mathematics_gate": False,
        "strict_interpretation": strict, "next_authorization": next_auth,
    }
    result["checks"] = {
        "source_complete": len(rows) == len(behavior) == 48,
        "full_trajectory_used": all(row["train_rows"] == len(train) for row in tournament),
        "confirmation_frozen_before_lockbox": selection["frozen_before_lockbox"],
        "causal_complete_or_registered_na": bool(causal_rows) == prediction_pass,
        "undisplayed_field_cleaned": not field_path.exists(), "finite": finite(result),
    }
    result["all_checks_passed"] = all(result["checks"].values())
    save(final_path, result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
