#!/usr/bin/env python3
"""C654: correct C651 endpoint leakage and test an upstream English concept bridge."""
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

PHASE = 2188
CAMPAIGN = "C654"
OUT = TESTS / "result/phase2188_c654_corrected_upstream_concept_bridge"
SOURCE = TESTS / "result/phase2187_c651_c653_qwen14_prospective_natural_concept_bridge"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
CATALOG = c651.CATALOG
VISUAL = ROOT / "frontend/public/vis_data/research_kernel/c654_corrected_upstream_bridge_atlas.json"
VISUAL_BINARY = ROOT / "frontend/public/vis_data/research_kernel/c654_selected_english_response.float16.npy"
ROLES = c651.ROLES
BEHAVIOR_GATE = 0.80
GAIN_GATE = 0.02
Q_CANDIDATES = (0, 8, 16, 25)
TARGET_Q = 33


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


def mean_or_none(values: list[float | bool]) -> float | None:
    return float(np.mean(values)) if values else None


def freeze(rows: list[dict], behavior: list[dict]) -> None:
    for part in ("protocol", "material", "behavior", "raw", "analysis", "audit"):
        (OUT / part).mkdir(parents=True, exist_ok=True)
    protocol = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_material_sha256": load(SOURCE / "protocol/preregistration.json")["material_sha256"],
        "model": "Qwen3-14B FP16 disk offload",
        "object": "English natural-translation concept-pair response at q33 boundary predicted only from strictly earlier checkpoints",
        "material": "the 48 frozen C651 concepts; English target only; discovery 24, confirmation 12, lockbox 12",
        "behavior": "reuse the already revealed C651 candidate/free-generation ledger; every English partition must independently satisfy both >=0.80",
        "camera": "all 42 checkpoints x six roles x all 5120 signed activation coordinates; two selected full-token panels",
        "leakage_correction": {
            "forbidden_predictor": "q33 boundary, because it is identical to the prediction target",
            "identity_relabel": "q8 source cross-target identity is a same-prefix lexical control, not evidence of target-language concept transport",
            "allowed_predictors": "q0/q8/q16/q25 x source/query/instruction/boundary/output_marker/assistant",
        },
        "prediction": "discovery-only coordinatewise affine or nearest-response fit; confirmation selects once; lockbox reveals once",
        "baselines": ["zero", "discovery_mean", "copy_input_response"],
        "prediction_gate": "selected method improves the strongest simple baseline by >=0.02 NRMSE on both confirmation and lockbox",
        "causal_authorization": "only after prediction gate; exact upstream, predicted q33, exact q33 and matched wrong controls plus doses",
        "forbidden": "Attention/MLP/weights/gradients/Top-K/PCA/projection/post-unblind edits",
        "human_review": "NA_pending_external_review inherited from frozen material",
    }
    if not (OUT / "protocol/preregistration.json").exists():
        save(OUT / "protocol/preregistration.json", protocol)
    write_rows(OUT / "material/material.jsonl", rows)
    write_rows(OUT / "behavior/reused_behavior.jsonl", behavior)


def capture(model, device, compiled: list[dict], dim: int) -> tuple[np.memmap, list[dict]]:
    modules = [model.model.embed_tokens, *list(model.model.layers), model.model.norm]
    path = OUT / "raw/all_role_field.float16.npy"
    field = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16,
                                     shape=(len(compiled), len(modules), len(ROLES), dim))
    selected = {c651.make_row(record, "en")["case_id"]
                for record in c651.partition_pairs("lockbox")[0]}
    panel_dir = OUT / "raw/full_token_panel"
    panel_dir.mkdir(parents=True, exist_ok=True)
    captured: list[torch.Tensor] = []
    handles = [module.register_forward_hook(
        lambda _m, _a, output: captured.append(output[0] if isinstance(output, tuple) else output))
        for module in modules]
    ledger = []
    try:
        for row_i, item in enumerate(compiled):
            ids = torch.tensor([item["prompt_ids"]], dtype=torch.long, device=device)
            mask = torch.ones_like(ids)
            pos = torch.arange(ids.shape[1], device=device)[None]
            captured.clear()
            with torch.inference_mode():
                model(input_ids=ids, attention_mask=mask, position_ids=pos,
                      use_cache=False, return_dict=True)
            panel = None
            panel_path = None
            if item["case_id"] in selected:
                panel_path = panel_dir / f"row_{row_i:03d}.float16.npy"
                panel = np.lib.format.open_memmap(
                    panel_path, mode="w+", dtype=np.float16,
                    shape=(len(modules), len(item["prompt_ids"]), dim))
            for q, tensor in enumerate(captured):
                values = tensor[0].float().cpu().numpy().astype(np.float16)
                if panel is not None:
                    panel[q] = values
                for role_i, role in enumerate(ROLES):
                    field[row_i, q, role_i] = values[int(item["role_positions"][role][-1])]
            if panel is not None and panel_path is not None:
                panel.flush()
                ledger.append({"case_id": item["case_id"],
                               "path": str(panel_path.relative_to(ROOT)),
                               "shape": list(panel.shape), "bytes": panel_path.stat().st_size})
                close_mmap(panel)
            print(f"[C654 corrected field] {row_i + 1}/{len(compiled)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    field.flush()
    save(OUT / "raw/full_token_panel_ledger.json", ledger)
    return field, ledger


def samples(partition: str, q: int, role: str, field: np.ndarray,
            row_index: dict[str, int], behavior_map: dict[str, dict]) -> list[dict]:
    answer = []
    role_i = ROLES.index(role)
    boundary_i = ROLES.index("boundary")
    for pair_i, (a, b) in enumerate(c651.partition_pairs(partition)):
        a_case = c651.make_row(a, "en")["case_id"]
        b_case = c651.make_row(b, "en")["case_id"]
        if not all(behavior_map[case]["candidate_correct"] and
                   behavior_map[case]["generated_correct"] for case in (a_case, b_case)):
            continue
        ai, bi = row_index[a_case], row_index[b_case]
        answer.append({
            "pair_index": pair_i, "a": a, "b": b,
            "x": field[bi, q, role_i].astype(np.float32) - field[ai, q, role_i].astype(np.float32),
            "y": field[bi, TARGET_Q, boundary_i].astype(np.float32) - field[ai, TARGET_Q, boundary_i].astype(np.float32),
        })
    return answer


def fit_diagonal(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xm, ym = x.mean(0), y.mean(0)
    beta = ((x - xm) * (y - ym)).sum(0) / (np.square(x - xm).sum(0) + 1e-6)
    return xm.astype(np.float32), ym.astype(np.float32), beta.astype(np.float32)


def nearest(x: np.ndarray, xt: np.ndarray, yt: np.ndarray) -> np.ndarray:
    return yt[int(np.argmin(np.square(xt - x[None]).sum(1)))]


def nrmse(pred: list[np.ndarray], truth: list[np.ndarray]) -> float | None:
    if not truth:
        return None
    num = sum(float(np.square(a - b).sum()) for a, b in zip(pred, truth))
    den = sum(float(np.square(b).sum()) for b in truth)
    return float(math.sqrt(num / max(den, 1e-12)))


def predict(kind: str, rows: list[dict], model: tuple[np.ndarray, ...]) -> list[np.ndarray]:
    xm, ym, beta, xt, yt = model
    if kind == "diagonal":
        return [ym + beta * (row["x"] - xm) for row in rows]
    if kind == "nearest_response":
        return [nearest(row["x"], xt, yt) for row in rows]
    raise ValueError(kind)


def metrics(rows: list[dict], model: tuple[np.ndarray, ...]) -> dict:
    _xm, _ym, _beta, _xt, yt = model
    truth = [row["y"] for row in rows]
    candidates = {
        "diagonal": predict("diagonal", rows, model),
        "nearest_response": predict("nearest_response", rows, model),
        "zero": [np.zeros_like(row["y"]) for row in rows],
        "discovery_mean": [yt.mean(0) for _row in rows],
        "copy_input_response": [row["x"] for row in rows],
    }
    return {name: {"nrmse": nrmse(values, truth)} for name, values in candidates.items()}


def append_memo(result: dict) -> None:
    existing = MEMO.read_text(encoding="utf-8-sig")
    if f"## Phase {PHASE}:" in existing:
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    protocol = load(OUT / "protocol/preregistration.json")
    formulas = r"""$$
D_{ab}(q,r)=H_b(q,r)-H_a(q,r),\qquad q\in\{0,8,16,25\}
$$

$$
Y_{ab}=D_{ab}(33,\mathrm{boundary}),\qquad
(q,r)\ne(33,\mathrm{boundary})
$$

$$
\widehat Y_{ab}=\mu_Y+\beta\odot(D_{ab}(q,r)-\mu_X)
$$

$$
\mathrm{gain}=\min\{\mathrm{NRMSE}_{0},\mathrm{NRMSE}_{\mu},
\mathrm{NRMSE}_{copy}\}-\mathrm{NRMSE}_{selected}
$$"""
    text = f"""

## Phase {PHASE}: C651恒等泄漏纠偏与严格上游概念桥（C654） [{stamp}]

**阶段目标与证据重裁。** Phase2187的正式“未通过”结论保留，但其确认集最佳项`q33/boundary -> q33/boundary`是目标预测自身，近零NRMSE属于恒等泄漏；`q8/source`跨目标语言身份1.0也只是同一前缀中源词状态的词汇控制，不能命名为目标语言概念运输。本期不改旧账，复用冻结材料与已揭盲行为，只允许严格早于目标的检查点预测`q33/boundary`。

**冻结合同。**

```json
{json.dumps(protocol, ensure_ascii=False, indent=2)}
```

**测试用例与原理。** 对每个语言族内概念对，如锁箱`蓝莓 -> 甜瓜`、`电脑 -> 袋子`，先在`q0/q8/q16/q25`六个角色上取得英语自然翻译的全5120维差分，再预测同一概念对在`q33/boundary`的差分。24个discovery概念只拟合，12个confirmation概念一次选型，12个lockbox概念一次裁决。自由生成与候选行为沿用Phase2187原始账本，三个英语分区均独立通过双门。

**数学公式。**

{formulas}

**详细结果。**

```json
{json.dumps(result, ensure_ascii=False, indent=2)}
```

**分析、理论进展与硬伤。** 本期把“目标向量预测自身”和“相同前缀源词指纹”从翻译机制证据中剔除。全坐标对角模型仍假设坐标独立，最近响应仍是实例检索；二者通过也只能支持上游响应的前瞻可预测性。若锁箱不通过，淘汰的是这两种桥，不是概念翻译机制。材料仍是受控单词翻译，人类自然度审查仍为`NA`，模型仅Qwen3-14B，激活坐标不是参数权重。

**相关文件。** 脚本：`tests/glm5/phase2188_c654_corrected_upstream_concept_bridge.py`；结果：`{OUT.relative_to(ROOT)}`；预注册：`{(OUT / 'protocol/preregistration.json').relative_to(ROOT)}`；全坐标可视化：`{VISUAL.relative_to(ROOT)}`和`{VISUAL_BINARY.relative_to(ROOT)}`。

**结论与下一步授权。** {result['strict_interpretation']} {result['next_authorization']}
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def main() -> None:
    final_path = OUT / "analysis/final.json"
    if final_path.exists():
        print(json.dumps(load(final_path), ensure_ascii=False, indent=2))
        return

    source_material = read_rows(SOURCE / "material/material.jsonl")
    rows = [row for row in source_material if row["target_language"] == "en"]
    source_behavior = read_rows(SOURCE / "behavior/behavior.jsonl")
    selected_ids = {row["case_id"] for row in rows}
    behavior = [row for row in source_behavior if row["case_id"] in selected_ids]
    behavior_map = {row["case_id"]: row for row in behavior}
    freeze(rows, behavior)

    route_metrics = {}
    for partition in ("discovery", "confirmation", "lockbox"):
        values = [row for row in rows if row["partition"] == partition]
        candidate = float(np.mean([behavior_map[row["case_id"]]["candidate_correct"] for row in values]))
        generated = float(np.mean([behavior_map[row["case_id"]]["generated_correct"] for row in values]))
        route_metrics[partition] = {"rows": len(values), "candidate_accuracy": candidate,
                                    "generated_accuracy": generated,
                                    "qualified": candidate >= BEHAVIOR_GATE and generated >= BEHAVIOR_GATE}
    if not all(row["qualified"] for row in route_metrics.values()):
        raise RuntimeError(route_metrics)
    save(OUT / "behavior/route_metrics.json", route_metrics)

    leakage_audit = {
        "phase2187_selected_input": "D_ab(q33,boundary)",
        "phase2187_prediction_target": "D_ab(q33,boundary)",
        "same_tensor_by_definition": True,
        "near_zero_nrmse_is_mechanical_identity": True,
        "q8_source_cross_target_warning": "source token precedes target-language instruction; causal masking preserves the same-prefix lexical state",
        "corrected_candidates": [{"checkpoint": q, "role": role}
                                 for q, role in itertools.product(Q_CANDIDATES, ROLES)],
        "target": {"checkpoint": TARGET_Q, "role": "boundary"},
    }
    save(OUT / "audit/leakage_audit.json", leakage_audit)

    model = None
    causal_rows: list[dict] = []
    dose_rows: list[dict] = []
    field = None
    try:
        model, tokenizer, device, placement, loader_name = loader.load_model("qwen3_14b")
        compiled = c651.previous.compile_rows(tokenizer, rows)
        write_rows(OUT / "material/compiled.jsonl", compiled)
        by_case = {row["case_id"]: row for row in compiled}
        row_index = {row["case_id"]: i for i, row in enumerate(compiled)}
        dim = int(model.model.embed_tokens.weight.shape[1])
        field, panels = capture(model, device, compiled, dim)

        tournament = []
        fitted: dict[tuple[int, str], tuple[np.ndarray, ...]] = {}
        max_input_target_equal = []
        for q, role in itertools.product(Q_CANDIDATES, ROLES):
            train = samples("discovery", q, role, field, row_index, behavior_map)
            confirm = samples("confirmation", q, role, field, row_index, behavior_map)
            if not train or not confirm:
                continue
            xt = np.stack([row["x"] for row in train])
            yt = np.stack([row["y"] for row in train])
            xm, ym, beta = fit_diagonal(xt, yt)
            fitted[(q, role)] = (xm, ym, beta, xt, yt)
            measured = metrics(confirm, fitted[(q, role)])
            equality = max(float(np.max(np.abs(row["x"] - row["y"]))) for row in confirm)
            max_input_target_equal.append(equality)
            tournament.append({"checkpoint": q, "role": role,
                               "train_rows": len(train), "confirmation_rows": len(confirm),
                               "max_abs_input_target_difference": equality,
                               "metrics": measured})
        choices = [(row["metrics"][kind]["nrmse"], row, kind)
                   for row in tournament for kind in ("diagonal", "nearest_response")]
        _, winner, winner_kind = min(choices, key=lambda item: item[0])
        simple = min(winner["metrics"][kind]["nrmse"]
                     for kind in ("zero", "discovery_mean", "copy_input_response"))
        confirmation_gain = simple - winner["metrics"][winner_kind]["nrmse"]
        selection = {"checkpoint": winner["checkpoint"], "role": winner["role"],
                     "model": winner_kind, "confirmation": winner["metrics"],
                     "confirmation_gain_over_best_simple": confirmation_gain,
                     "frozen_before_lockbox": True}
        save(OUT / "analysis/confirmation_tournament.json", tournament)
        save(OUT / "protocol/confirmation_selection_frozen.json", selection)

        q, role = selection["checkpoint"], selection["role"]
        model_fit = fitted[(q, role)]
        lock = samples("lockbox", q, role, field, row_index, behavior_map)
        lock_metrics = metrics(lock, model_fit)
        lock_simple = min(lock_metrics[kind]["nrmse"]
                          for kind in ("zero", "discovery_mean", "copy_input_response"))
        lock_gain = lock_simple - lock_metrics[winner_kind]["nrmse"]
        prediction_pass = confirmation_gain >= GAIN_GATE and lock_gain >= GAIN_GATE
        save(OUT / "analysis/lockbox_prediction.json", {
            "selection": selection, "rows": len(lock), "metrics": lock_metrics,
            "gain_over_best_simple": lock_gain, "passed": prediction_pass})
        xm, ym, beta, xt, yt = model_fit
        np.savez(OUT / "raw/selected_bridge_model.npz", model=np.asarray([winner_kind]),
                 checkpoint=np.asarray([q]), role=np.asarray([ROLES.index(role)]),
                 x_mean=xm, y_mean=ym, beta=beta, x_train=xt, y_train=yt)

        if prediction_pass:
            wrong_q = {0: 8, 8: 16, 16: 25, 25: 16}[q]
            for pair_i, sample in enumerate(lock):
                a, b = sample["a"], sample["b"]
                a_case = c651.make_row(a, "en")["case_id"]
                b_case = c651.make_row(b, "en")["case_id"]
                item = c651.previous._eval_item(by_case[a_case], by_case[b_case])
                exact = sample["x"]
                exact_q33 = sample["y"]
                predicted = predict(winner_kind, [sample], model_fit)[0]
                wrong = lock[(pair_i + 1) % len(lock)]["x"]
                wrong_role = "query" if role != "query" else "instruction"
                modes = {
                    "zero": [],
                    "exact_upstream": c651.previous._patches(item, q, [(role, exact)]),
                    "predicted_q33": c651.previous._patches(item, TARGET_Q, [("boundary", predicted)]),
                    "exact_q33": c651.previous._patches(item, TARGET_Q, [("boundary", exact_q33)]),
                    "wrong_pair": c651.previous._patches(item, q, [(role, wrong)]),
                    "wrong_direction": c651.previous._patches(item, q, [(role, -exact)]),
                    "wrong_role": c651.previous._patches(item, q, [(wrong_role, exact)]),
                    "wrong_checkpoint": c651.previous._patches(item, wrong_q, [(role, exact)]),
                }
                for mode, patches in modes.items():
                    generated = c651.previous.translation._patched_generate(
                        model, tokenizer, item, patches, max_new_tokens=10)
                    causal_rows.append({"pair_index": pair_i, "a": a["concept_uid"],
                                        "b": b["concept_uid"], "mode": mode, **generated})
                for kind, vector, qv, rv in (
                    ("exact_upstream", exact, q, role),
                    ("predicted_q33", predicted, TARGET_Q, "boundary"),
                ):
                    for dose in (0.5, 1.0, 1.5):
                        generated = c651.previous.translation._patched_generate(
                            model, tokenizer, item,
                            c651.previous._patches(item, qv, [(rv, vector * dose)]),
                            max_new_tokens=10)
                        dose_rows.append({"pair_index": pair_i, "kind": kind,
                                          "dose": dose, **generated})
                print(f"[C654 causal] {pair_i + 1}/{len(lock)}", flush=True)
        write_rows(OUT / "raw/causal_generation.jsonl", causal_rows)
        write_rows(OUT / "raw/dose_generation.jsonl", dose_rows)

        pair = c651.partition_pairs("lockbox")[0]
        ai = row_index[c651.make_row(pair[0], "en")["case_id"]]
        bi = row_index[c651.make_row(pair[1], "en")["case_id"]]
        response = (field[bi].astype(np.float32) - field[ai].astype(np.float32)).astype(np.float16)
        VISUAL_BINARY.parent.mkdir(parents=True, exist_ok=True)
        np.save(VISUAL_BINARY, response)
        atlas = {
            "schema": "ai2050.corrected_upstream_concept_bridge.v1", "phase": PHASE,
            "campaign": CAMPAIGN, "model": "Qwen3-14B", "target_language": "en",
            "coordinates": dim, "coordinate_ids": list(range(dim)),
            "checkpoints": list(range(field.shape[1])), "roles": list(ROLES),
            "selected_pair": [pair[0], pair[1]], "response_shape": list(response.shape),
            "response": np.round(response.astype(np.float32), 6).tolist(),
            "binary_float16": "/vis_data/research_kernel/c654_selected_english_response.float16.npy",
            "selection": selection, "lockbox_prediction": lock_metrics,
            "full_coordinate": True, "no_topk": True,
            "warning": "activation coordinates are model-specific states, not parameter weights",
        }
        save(VISUAL, atlas)
        field.flush()
        close_mmap(field)
        field = None
    finally:
        if field is not None:
            field.flush()
            close_mmap(field)
        loader.release_model("qwen3_14b", model)
        gc.collect()

    causal_rates = {mode: mean_or_none([row["correct"] for row in causal_rows if row["mode"] == mode])
                    for mode in sorted({row["mode"] for row in causal_rows})}
    dose_rates = {f"{kind}@{dose}": mean_or_none([
        row["correct"] for row in dose_rows if row["kind"] == kind and row["dose"] == dose])
        for kind, dose in itertools.product(("exact_upstream", "predicted_q33"), (0.5, 1.0, 1.5))}
    predicted_causal = bool(causal_rows) and (
        (causal_rates.get("predicted_q33") or 0.0) >= 0.50 and
        (causal_rates.get("predicted_q33") or 0.0) - max(
            causal_rates.get("zero") or 0.0, causal_rates.get("wrong_pair") or 0.0,
            causal_rates.get("wrong_direction") or 0.0,
            causal_rates.get("wrong_role") or 0.0,
            causal_rates.get("wrong_checkpoint") or 0.0) >= 0.25)

    catalog = load(CATALOG)
    entry = {"id": "c654_corrected_upstream_bridge_atlas",
             "label": "C654 Corrected Upstream Concept Bridge",
             "path": "/vis_data/research_kernel/c654_corrected_upstream_bridge_atlas.json",
             "binary_path": "/vis_data/research_kernel/c654_selected_english_response.float16.npy",
             "phase": PHASE, "full_coordinate": True,
             "heatmap_type": "embedding_hiddenstate_full_coordinate"}
    datasets = catalog.setdefault("field_datasets", [])
    datasets[:] = [row for row in datasets if row.get("id") != entry["id"]]
    datasets.append(entry)
    catalog["generated_at"] = datetime.now(timezone.utc).isoformat()
    save(CATALOG, catalog)

    cleanup = {"deleted": [], "retained": [str(VISUAL.relative_to(ROOT)),
               str(VISUAL_BINARY.relative_to(ROOT))], "bytes_deleted": 0}
    ledger = load(OUT / "raw/full_token_panel_ledger.json")
    cleanup["retained"].extend(row["path"] for row in ledger)
    field_path = OUT / "raw/all_role_field.float16.npy"
    if field_path.exists():
        cleanup["bytes_deleted"] += field_path.stat().st_size
        cleanup["deleted"].append(str(field_path.relative_to(ROOT)))
        field_path.unlink()
    save(OUT / "audit/cleanup.json", cleanup)

    strict = ("The corrected strictly-upstream bridge passed prediction and the predicted q33 response was selectively callable."
              if prediction_pass and predicted_causal else
              "The C651 identity leak was removed; the corrected strictly-upstream bridge did not jointly pass prospective prediction and causal calling.")
    next_auth = ("若双门均通过，授权全新多词短语与关系组合翻译；否则保留全场规律，淘汰当前逐坐标与最近实例桥，下一阶段研究不预设坐标独立性的样本内关系映射。")
    result = {
        "phase": PHASE, "campaign": CAMPAIGN, "status": "closed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "route_metrics": route_metrics, "placement": placement, "loader": loader_name,
        "field_shape_before_cleanup": [48, 42, 6, 5120],
        "leakage_audit": leakage_audit, "candidate_count": len(tournament),
        "input_target_identity_found": any(value == 0.0 for value in max_input_target_equal),
        "selection": selection, "lockbox_rows": len(lock),
        "lockbox_metrics": lock_metrics, "lockbox_gain_over_best_simple": lock_gain,
        "prospective_prediction_pass": prediction_pass,
        "causal_rates": causal_rates, "dose_rates": dose_rates,
        "predicted_state_causal_pass": predicted_causal,
        "human_review": "NA_pending_external_review",
        "new_foundational_mathematics_gate": False,
        "visual": str(VISUAL.relative_to(ROOT)), "cleanup": cleanup,
        "strict_interpretation": strict, "next_authorization": next_auth,
    }
    result["checks"] = {
        "source_ledgers_complete": len(rows) == len(behavior) == 48,
        "all_english_partitions_qualified": all(row["qualified"] for row in route_metrics.values()),
        "endpoint_self_predictor_excluded": all(row["checkpoint"] < TARGET_Q for row in tournament),
        "confirmation_selection_frozen_before_lockbox": selection["frozen_before_lockbox"],
        "causal_complete_or_registered_na": bool(causal_rows) == prediction_pass,
        "visual_complete": VISUAL.exists() and VISUAL_BINARY.exists(),
        "cleanup_complete": not field_path.exists(), "finite": finite(result),
    }
    result["all_checks_passed"] = all(result["checks"].values())
    save(final_path, result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
