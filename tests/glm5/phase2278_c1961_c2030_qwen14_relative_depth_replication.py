#!/usr/bin/env python3
"""Prospective Qwen3-14B relative-depth replication for Phase 2278."""
from __future__ import annotations

import gc
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
CONTRACT_OUT = RESULT / "phase2274_c1721_c1770_broad_construction_contract"
SOURCE_OUT = RESULT / "phase2276_c1821_c1890_full_coordinate_structure_tournament"
OUT = RESULT / "phase2278_c1961_c2030_qwen14_relative_depth_replication"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
MODEL_ROOT = ROOT / "models/hf/Qwen3-14B"
OFFLOAD_ROOT = RESULT / "phase1118_qwen3_14b_fp16_offload_smoke/disk_offload_revision5"
sys.path.insert(0, str(TESTS))

import phase2266_c1469_c1504_qwen4b_independent_fullfield as capture_base  # noqa: E402
import phase2274_c1721_c1770_broad_construction_contract as contract  # noqa: E402
import phase2276_c1821_c1890_full_coordinate_structure_tournament as structure  # noqa: E402


PHASE = 2278
CAMPAIGN = "C1961-C2030"
FROZEN = {
    "patient_binding": {"q4": 11, "role": "query", "role_index": 4, "model": "own_affine", "q14": [10, 11, 12, 13, 14]},
    "relative_clause_binding": {"q4": 9, "role": "relation", "role_index": 2, "model": "piecewise_quartile", "q14": [8, 9, 10, 11, 12]},
    "location_state": {"q4": 11, "role": "query", "role_index": 4, "model": "own_affine", "q14": [10, 11, 12, 13, 14]},
}
QPOINTS = tuple(sorted({q for setting in FROZEN.values() for q in setting["q14"]} | {7, 9, 13}))
UNIT_PANEL = tuple([*range(0, 6), *range(12, 14), *range(16, 20), *range(24, 28)])
GENERATION_SURFACE = "direct"
GATES = structure.GATES
CONTROLS = structure.CONTROLS
FIELD = OUT / "raw/qwen3_14b_midlayer_role_field.float16.npy"
INDEX = OUT / "raw/midlayer_role_field_index.jsonl"
PROGRESS = OUT / "raw/midlayer_role_field_progress.json"
COMPILED = OUT / "material/qwen3_14b_midlayer_compiled.jsonl"
EPS = 1e-8


def save(path: Path, value: Any) -> None:
    contract.save(path, value)


def load(path: Path) -> Any:
    return contract.load(path)


def read_rows(path: Path) -> list[dict]:
    return contract.read_rows(path)


def write_rows(path: Path, rows: list[dict]) -> None:
    contract.write_rows(path, rows)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def device_map() -> dict:
    result = {"model.embed_tokens": 0}
    result.update({f"model.layers.{i}": 0 if i < 18 else "disk" for i in range(40)})
    result.update({"model.norm": "disk", "model.rotary_emb": "cpu", "lm_head": "disk"})
    return result


def parse_code(text: str, row: dict) -> str | None:
    clean = text.strip().lower()
    hits = []
    for code in (row["true_code"], row["false_code"]):
        match = re.search(rf"\b{re.escape(code.lower())}\b", clean)
        if match:
            hits.append((match.start(), code))
    return min(hits)[1] if hits else None


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_ROOT), local_files_only=True,
                                               trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(str(MODEL_ROOT), local_files_only=True, trust_remote_code=True)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, dtype=torch.float16, trust_remote_code=True)
    model.tie_weights()
    model = load_checkpoint_and_dispatch(
        model, checkpoint=str(MODEL_ROOT), device_map=device_map(),
        no_split_module_classes=list(model._no_split_modules), offload_folder=str(OFFLOAD_ROOT),
        offload_buffers=False, dtype=torch.float16, offload_state_dict=True,
        force_hooks=True, strict=True)
    model.eval()
    return model, tokenizer, torch.device("cuda:0")


def selected_modules(model) -> dict[int, torch.nn.Module]:
    output = {0: model.model.embed_tokens} if 0 in QPOINTS else {}
    for q in QPOINTS:
        if q > 0:
            output[q] = model.model.layers[q - 1]
    return output


def capture_candidate_field(model, tokenizer, device, rows: list[dict]) -> list[dict]:
    shape = (len(rows), len(QPOINTS), len(contract.ROLES), 5120)
    FIELD.parent.mkdir(parents=True, exist_ok=True)
    completed = 0
    candidates: list[dict] = []
    candidate_path = OUT / "behavior/candidate_midlayer.jsonl"
    if FIELD.exists() and PROGRESS.exists() and candidate_path.exists():
        progress = load(PROGRESS)
        if tuple(progress["shape"]) != shape:
            raise RuntimeError(("resume_shape", progress["shape"], shape))
        completed = int(progress["completed_rows"])
        candidates = read_rows(candidate_path)
        if len(candidates) != completed:
            raise RuntimeError(("candidate_resume", len(candidates), completed))
        field = np.lib.format.open_memmap(FIELD, mode="r+")
    else:
        field = np.lib.format.open_memmap(FIELD, mode="w+", dtype=np.float16, shape=shape)
        save(PROGRESS, {"shape": list(shape), "completed_rows": 0})
    captured: dict[int, torch.Tensor] = {}
    handles = []
    for q, module in selected_modules(model).items():
        def hook(_module, _args, output, q=q):
            captured[q] = output[0] if isinstance(output, tuple) else output
        handles.append(module.register_forward_hook(hook))
    pad = int(tokenizer.pad_token_id)
    try:
        for start in range(completed, len(rows), 6):
            batch = rows[start:start + 6]
            width = max(len(row["prompt_ids"]) for row in batch)
            ids = torch.full((len(batch), width), pad, dtype=torch.long, device=device)
            mask = torch.zeros_like(ids)
            for i, row in enumerate(batch):
                seq = row["prompt_ids"]
                ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
                mask[i, :len(seq)] = 1
            pos = mask.long().cumsum(-1) - 1
            pos.masked_fill_(mask == 0, 0)
            captured.clear()
            with torch.inference_mode():
                logits = model(input_ids=ids, attention_mask=mask, position_ids=pos,
                               use_cache=False, return_dict=True).logits
            if set(captured) != set(QPOINTS):
                raise RuntimeError(("captured_qpoints", sorted(captured), QPOINTS))
            for local_i, row in enumerate(batch):
                length = len(row["prompt_ids"])
                scores = [float(logits[local_i, length - 1, candidate[0]])
                          for candidate in row["candidate_ids"]]
                prediction = int(scores[1] > scores[0])
                candidates.append({"case_id": row["case_id"], "scores": scores,
                                   "prediction": prediction,
                                   "correct": prediction == int(row["gold_position"])})
                for q_i, q in enumerate(QPOINTS):
                    state = captured[q][local_i].float().cpu().numpy()
                    for role_i, role in enumerate(contract.ROLES):
                        field[start + local_i, q_i, role_i] = state[row["role_positions"][role][-1]]
            field.flush()
            write_rows(candidate_path, candidates)
            done = min(start + len(batch), len(rows))
            save(PROGRESS, {"shape": list(shape), "completed_rows": done})
            if start % 32 == 0:
                print(f"[Qwen3-14B candidate+field] {done}/{len(rows)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
        field.flush()
        del field
    return candidates


def generation(model, tokenizer, device, rows: list[dict]) -> list[dict]:
    path = OUT / "behavior/generation_midlayer.jsonl"
    output = read_rows(path) if path.exists() else []
    completed = len(output)
    pad = int(tokenizer.pad_token_id)
    for start in range(completed, len(rows), 8):
        batch = rows[start:start + 8]
        width = max(len(row["free_prompt_ids"]) for row in batch)
        ids = torch.full((len(batch), width), pad, dtype=torch.long, device=device)
        mask = torch.zeros_like(ids)
        for i, row in enumerate(batch):
            seq = row["free_prompt_ids"]
            ids[i, width - len(seq):] = torch.tensor(seq, dtype=torch.long, device=device)
            mask[i, width - len(seq):] = 1
        with torch.inference_mode():
            generated = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=6,
                                       do_sample=False, pad_token_id=pad,
                                       eos_token_id=tokenizer.eos_token_id)
        for i, row in enumerate(batch):
            new_ids = generated[i, width:].tolist()
            text = tokenizer.decode(new_ids, skip_special_tokens=True)
            parsed = parse_code(text, row)
            output.append({"case_id": row["case_id"], "text": text, "generated_ids": new_ids,
                           "parsed": parsed, "correct": parsed == row["correct_answer"]})
        write_rows(path, output)
        if start % 32 == 0:
            print(f"[Qwen3-14B generation] {min(start + len(batch), len(rows))}/{len(rows)}", flush=True)
    return output


def behavior_ledger(rows: list[dict], candidates: list[dict], generated: list[dict]) -> dict:
    c = {row["case_id"]: row for row in candidates}
    g = {row["case_id"]: row for row in generated}

    def summary(subset: list[dict]) -> dict:
        generated_subset = [row for row in subset if row["case_id"] in g]
        ca = float(np.mean([c[row["case_id"]]["correct"] for row in generated_subset]))
        ga = float(np.mean([g[row["case_id"]]["correct"] for row in generated_subset]))
        all_ca = float(np.mean([c[row["case_id"]]["correct"] for row in subset]))
        return {"rows": len(subset), "generation_panel_rows": len(generated_subset),
                "candidate_accuracy_all_surfaces": all_ca,
                "candidate_accuracy": ca, "generation_accuracy": ga,
                "dual_qualified": min(ca, ga) >= contract.BEHAVIOR_GATE}

    families = {}
    qualified = []
    for family in FROZEN:
        subset = [row for row in rows if row["family"] == family]
        partitions = {part: summary([row for row in subset if row["partition"] == part])
                      for part in ("discovery", "confirmation", "fresh_confirmation", "fresh_lockbox")}
        overall = summary(subset)
        passed = all(value["dual_qualified"] for value in
                     (overall, partitions["discovery"], partitions["fresh_confirmation"]))
        families[family] = {**overall, "partitions": partitions, "qualified": passed}
        if passed:
            qualified.append(family)
    return {**summary(rows), "candidate_rows": len(candidates), "generation_rows": len(generated),
            "parsed_generation_fraction": float(np.mean([row["parsed"] is not None for row in generated])),
            "families": families, "qualified_families": qualified}


def make_index(rows: list[dict]) -> list[dict]:
    return [{"hidden_index": i, "case_id": row["case_id"], "family": row["family"],
             "unit": int(row["unit"]), "surface": row["surface"], "state": int(row["state"]),
             "partition": row["partition"], "output_scheme": int(row["output_scheme"]),
             "role_positions": row["role_positions"]} for i, row in enumerate(rows)]


def pair_rows(index: list[dict]) -> list[dict]:
    groups: dict[tuple, dict[int, dict]] = defaultdict(dict)
    for row in index:
        groups[(row["family"], row["unit"], row["surface"], row["partition"])][row["state"]] = row
    output = []
    for key, states in sorted(groups.items()):
        if set(states) != {0, 1}:
            raise RuntimeError(("pair", key))
        output.append({"family": key[0], "unit": key[1], "surface": key[2], "partition": key[3],
                       "output_scheme": states[0]["output_scheme"],
                       "state0_index": states[0]["hidden_index"],
                       "state1_index": states[1]["hidden_index"]})
    return output


def qslot(q: int) -> int:
    return QPOINTS.index(q)


def arrays(field: np.ndarray, pairs: list[dict], q: int, role_i: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    i0 = np.asarray([row["state0_index"] for row in pairs], dtype=np.int64)
    i1 = np.asarray([row["state1_index"] for row in pairs], dtype=np.int64)
    h0 = np.asarray(field[i0, qslot(q), role_i], dtype=np.float32)
    h1 = np.asarray(field[i1, qslot(q), role_i], dtype=np.float32)
    return h0, h1, h1 - h0


def fit_bundle(field: np.ndarray, pairs: list[dict], q: int, role_i: int) -> dict[str, Any]:
    x, h1, y = arrays(field, pairs, q, role_i)
    own = structure.fit_affine(x, y)
    shift = max(1, len(pairs) // 3)
    shuffled = structure.fit_affine(x, np.roll(y, shift=shift, axis=0))
    previous_q = q - 1 if q > 0 and q - 1 in QPOINTS else q
    _, _, previous = arrays(field, pairs, previous_q, role_i)
    previous_model = structure.fit_affine(previous, y) if previous_q != q else (
        np.zeros(x.shape[1], np.float32), y.mean(axis=0).astype(np.float32))
    return {"mean": y.mean(axis=0).astype(np.float32),
            "mean_h1": h1.mean(axis=0).astype(np.float32),
            "own_affine": own, "piecewise_quartile": structure.fit_piecewise(x, y),
            "sign_interval": structure.fit_sign_interval(x, y),
            "shuffled_affine": shuffled, "previous_checkpoint_affine": previous_model,
            "previous_q": previous_q}


def predict_candidate(name: str, bundle: dict[str, Any], x: np.ndarray) -> np.ndarray:
    if name == "own_affine":
        return x * bundle[name][0] + bundle[name][1]
    if name == "piecewise_quartile":
        return structure.predict_piecewise(bundle[name], x)
    if name == "sign_interval":
        return structure.predict_sign_interval(bundle[name], x)
    raise KeyError(name)


def evaluate(field: np.ndarray, pairs: list[dict], q: int, role_i: int, model_name: str,
             own: dict[str, Any], wrong: dict[str, Any], shared: tuple[np.ndarray, np.ndarray],
             surface_means: dict[str, np.ndarray], scheme_means: dict[int, np.ndarray]) -> dict:
    x, _h1, y = arrays(field, pairs, q, role_i)
    previous_q = int(own["previous_q"])
    _, _, previous = arrays(field, pairs, previous_q, role_i)
    predictions = {
        "candidate": predict_candidate(model_name, own, x),
        "family_mean": np.broadcast_to(own["mean"], y.shape),
        "algebraic": own["mean_h1"] - x,
        "shared_affine": x * shared[0] + shared[1],
        "wrong_family_affine": x * wrong["own_affine"][0] + wrong["own_affine"][1],
        "shuffled_affine": x * own["shuffled_affine"][0] + own["shuffled_affine"][1],
        "previous_checkpoint_affine": previous * own["previous_checkpoint_affine"][0] + own["previous_checkpoint_affine"][1],
        "surface_mean": np.stack([surface_means[row["surface"]] for row in pairs]),
        "output_scheme_mean": np.stack([scheme_means[int(row["output_scheme"])] for row in pairs]),
    }
    errors = {name: np.mean(np.abs(prediction - y), axis=0).astype(np.float32)
              for name, prediction in predictions.items()}
    baseline = errors["family_mean"]
    gains = {name: float((errors[name].mean() - errors["candidate"].mean()) /
                         max(float(errors[name].mean()), EPS)) for name in CONTROLS}
    wins = {name: float(np.mean(errors["candidate"] < errors[name])) for name in CONTROLS}
    return {"pairs": len(pairs), "mae": float(errors["candidate"].mean()),
            "gain_over_family_mean": float((baseline.mean() - errors["candidate"].mean()) /
                                           max(float(baseline.mean()), EPS)),
            "coordinate_win_over_family_mean": float(np.mean(errors["candidate"] < baseline)),
            "gain_over_controls": gains, "coordinate_win_over_controls": wins,
            "minimum_control_gain": min(gains.values()), "minimum_control_win": min(wins.values()),
            "candidate_error": errors["candidate"]}


def passes(value: dict) -> bool:
    return bool(value["gain_over_family_mean"] >= GATES["gain_over_family_mean"] and
                value["coordinate_win_over_family_mean"] >= GATES["coordinate_win_over_family_mean"] and
                value["minimum_control_gain"] >= GATES["gain_over_each_control"] and
                value["minimum_control_win"] >= GATES["coordinate_win_over_each_control"])


def structure_tournament(field: np.ndarray, index: list[dict], qualified: list[str]) -> dict:
    pairs = pair_rows(index)
    by_family = {family: [row for row in pairs if row["family"] == family] for family in qualified}
    discovery = {family: [row for row in by_family[family] if row["partition"] == "discovery"]
                 for family in qualified}
    fits: dict[tuple[str, int], dict] = {}
    shared: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    surface_means: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    scheme_means: dict[tuple[str, int], dict[int, np.ndarray]] = {}
    for q in QPOINTS:
        pooled_x, pooled_y = [], []
        for family in qualified:
            role_i = int(FROZEN[family]["role_index"])
            x, _h1, y = arrays(field, discovery[family], q, role_i)
            pooled_x.append(x)
            pooled_y.append(y)
            fits[(family, q)] = fit_bundle(field, discovery[family], q, role_i)
            surface_means[(family, q)] = {surface: arrays(
                field, [row for row in discovery[family] if row["surface"] == surface], q, role_i)[2].mean(axis=0)
                for surface in contract.SURFACES}
            scheme_means[(family, q)] = {scheme: arrays(
                field, [row for row in discovery[family] if int(row["output_scheme"]) == scheme], q, role_i)[2].mean(axis=0)
                for scheme in range(len(contract.OUTPUT_SCHEMES))}
        shared[q] = structure.fit_affine(np.concatenate(pooled_x), np.concatenate(pooled_y))
    confirmation: dict[str, dict] = {}
    selected: dict[str, dict] = {}
    score_rows = []
    passports = np.full((len(FROZEN), 8, 5120), np.nan, dtype=np.float32)
    for family_i, family in enumerate(FROZEN):
        if family not in qualified:
            confirmation[family] = {"passes": False, "reason": "behavior_unqualified"}
            continue
        setting = FROZEN[family]
        wrong_family = qualified[(qualified.index(family) + 1) % len(qualified)]
        test = [row for row in by_family[family] if row["partition"] == "confirmation"]
        candidates = []
        for q in setting["q14"]:
            value = evaluate(field, test, q, int(setting["role_index"]), setting["model"],
                             fits[(family, q)], fits[(wrong_family, q)], shared[q],
                             surface_means[(family, q)], scheme_means[(family, q)])
            record = {k: v for k, v in value.items() if k != "candidate_error"}
            record.update({"family": family, "checkpoint": q, "relative_depth": q / 40.0,
                           "role": setting["role"], "model": setting["model"], "passes": passes(value)})
            score_rows.append(record)
            candidates.append((record, value))
        record, value = max(candidates, key=lambda item: (
            item[0]["minimum_control_gain"], item[0]["minimum_control_win"],
            item[0]["gain_over_family_mean"]))
        confirmation[family] = record
        selected[family] = {"checkpoint": record["checkpoint"], "role": setting["role"],
                            "role_index": setting["role_index"], "model": setting["model"],
                            "confirmation_passed": record["passes"]}
        bundle = fits[(family, record["checkpoint"])]
        if setting["model"] == "own_affine":
            passports[family_i, 0] = bundle["own_affine"][0]
            passports[family_i, 1] = bundle["own_affine"][1]
        elif setting["model"] == "piecewise_quartile":
            passports[family_i, :3] = bundle["piecewise_quartile"][0]
            passports[family_i, 3:7] = bundle["piecewise_quartile"][1]
        passports[family_i, 7] = value["candidate_error"]
    passport_path = OUT / "atlas/qwen14_selected_coordinate_passport.float32.npy"
    passport_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(passport_path, passports)
    write_rows(OUT / "analysis/confirmation_candidates.jsonl", score_rows)
    save(OUT / "protocol/frozen_q14_selections.json", selected)
    stages = {}
    for partition in ("fresh_confirmation", "fresh_lockbox"):
        summaries = {}
        for family in FROZEN:
            if family not in selected or not selected[family]["confirmation_passed"]:
                summaries[family] = {"passes": False, "reason": "confirmation_unqualified"}
                continue
            if partition == "fresh_lockbox" and not stages["fresh_confirmation"][family]["passes"]:
                summaries[family] = {"passes": False, "reason": "fresh_confirmation_unqualified"}
                continue
            setting = selected[family]
            q = int(setting["checkpoint"])
            wrong_family = qualified[(qualified.index(family) + 1) % len(qualified)]
            test = [row for row in by_family[family] if row["partition"] == partition]
            value = evaluate(field, test, q, int(setting["role_index"]), setting["model"],
                             fits[(family, q)], fits[(wrong_family, q)], shared[q],
                             surface_means[(family, q)], scheme_means[(family, q)])
            summaries[family] = {k: v for k, v in value.items() if k != "candidate_error"}
            summaries[family].update({"checkpoint": q, "role": setting["role"],
                                      "model": setting["model"], "passes": passes(value)})
        stages[partition] = summaries
    return {"confirmation": confirmation, "selected": selected,
            "fresh_confirmation": stages["fresh_confirmation"],
            "fresh_lockbox": stages["fresh_lockbox"],
            "lockbox_passed_families": [family for family, value in stages["fresh_lockbox"].items()
                                         if value.get("passes")],
            "passport": {"path": str(passport_path.relative_to(ROOT)), "shape": list(passports.shape)}}


def append_memo(result: dict) -> None:
    current = MEMO.read_text(encoding="utf-8-sig")
    if f"## Phase {PHASE}:" in current:
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: Qwen3-14B 相对层深全坐标前瞻复现（{CAMPAIGN}） [{stamp}]

**测试原理与用例。** 本期只迁移 Phase2276 在 Qwen3-4B fresh lockbox 通过、且位于中层的三个预测结构：受事绑定、关系从句绑定和位置状态；不迁移 Phase2277 已失败的坐标联盟，也不把 q0/q1 词汇局部效应混称为深层装配。材料使用 16 个预先分层的词汇单元、三种表面、真假两状态，共 `{result['material_rows']}` 行；候选行为覆盖全部表面，精确自由生成在预注册 direct 表面上独立记账。Qwen3-14B 使用 FP16 CUDA 加磁盘卸载顺序执行，在相对层深窗口内保存六个语义角色的全部 5120 个运行时激活坐标。4B 与 14B 只比较角色、相对层深、模型类型和门槛，不比较物理坐标编号。

**公式。** 四十层模型的相对深度为：

$$
\rho(q)=\frac{{q}}{{40}},\qquad q_{{14}}\in\mathcal W\!\left(\operatorname{{round}}\left(40\frac{{q_4}}{{36}}\right)\right).
$$

每个坐标只使用 discovery 拟合冻结的基础响应函数：

$$
\widehat R_j=f_{{f,q,r,j}}(H^0_j),\qquad
R_j=H^1_j-H^0_j.
$$

确认集只选择预注册窗口内的相对深度；fresh-confirmation 只负责授权，fresh-lockbox 才给最终裁决。通过要求相对家族均值的误差下降不低于 `0.03`、逐坐标胜率不低于 `0.55`，并且相对每个冻结控制的误差下降和逐坐标胜率分别不低于 `0.01`、`0.52`。

**结果汇总。** 行为：`{json.dumps(result['behavior'], ensure_ascii=False)}`。确认选择：`{json.dumps(result['structure']['confirmation'], ensure_ascii=False)}`。fresh-confirmation：`{json.dumps(result['structure']['fresh_confirmation'], ensure_ascii=False)}`。fresh-lockbox：`{json.dumps(result['structure']['fresh_lockbox'], ensure_ascii=False)}`。通过族：`{json.dumps(result['structure']['lockbox_passed_families'], ensure_ascii=False)}`。逐坐标护照：`{json.dumps(result['structure']['passport'], ensure_ascii=False)}`。哈希与检查：`{json.dumps(result['hashes'], ensure_ascii=False)}`、`{json.dumps(result['checks'], ensure_ascii=False)}`。

**分析与理论进展。** {result['strict_conclusion']} 通过只说明“构式条件下从基态坐标预测同坐标响应”的功能拓扑在更大同族模型上可复验；它不是相同神经元、相同物理坐标、参数规模因果律或唯一语义程序。q0/q1 阳性首先属于词汇和局部前缀条件；只有中层窗口阳性才支持较深的角色条件化响应装配，但仍需和输出准备、表面模板及一般残差动力学区分。

**问题、硬伤与瓶颈。** 这只是 4B 到 14B 的一个同模型族区间；训练数据和训练轨迹未控制；磁盘卸载只影响工程速度但扩大数值执行复杂度；材料仍是受控英文且无人类盲评；每坐标基础函数忽略跨坐标耦合；确认集选择相对深度带来有限的窗口内多重比较；完整原场虽未压缩，但当前判决仍是坐标误差汇总，不等于机制闭合。

**结论与下一步。** 下一阶段发布可观察的 exact-coordinate 护照、单坐标效应、候选掩码和相对层深地图；只保留进入客户端的数据派生物，清理未显示的大型原始 HiddenState 场。脚本 `tests/glm5/phase2278_c1961_c2030_qwen14_relative_depth_replication.py`；结果 `tests/glm5/result/phase2278_c1961_c2030_qwen14_relative_depth_replication`。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def run() -> dict:
    final = OUT / "analysis/final.json"
    if final.exists():
        result = load(final)
        append_memo(result)
        return result
    source = load(SOURCE_OUT / "analysis/final.json")
    if not set(FROZEN).issubset(source["lockbox_passed_families"]):
        raise RuntimeError(("source_selection_changed", source["lockbox_passed_families"], list(FROZEN)))
    raw_rows = [row for row in read_rows(CONTRACT_OUT / "material/broad_construction_cases.jsonl")
                if row["family"] in FROZEN and int(row["unit"]) in UNIT_PANEL]
    prereg = {"phase": PHASE, "campaign": CAMPAIGN, "frozen_before_model": True,
              "source_phase": 2276, "families": FROZEN, "qpoints": QPOINTS,
              "unit_panel": UNIT_PANEL, "generation_surface": GENERATION_SURFACE,
              "material_rows": len(raw_rows), "behavior_gate": contract.BEHAVIOR_GATE,
              "gates": GATES, "controls": list(CONTROLS),
              "partitions": {"discovery": list(range(0, 6)), "confirmation": list(range(12, 14)),
                             "fresh_confirmation": list(range(16, 20)), "fresh_lockbox": list(range(24, 28))},
              "forbidden": ["cross-model physical-coordinate alignment", "attention", "MLP", "weight",
                            "gradient", "PCA", "Top-K", "lockbox reselection"]}
    save(OUT / "protocol/preregistration.json", prereg)
    save(OUT / "protocol/aborted_full_seven_family_resource_audit.json", {
        "status": "aborted_before_any_family_level_behavior_or_hiddenstate_result",
        "observed_before_abort": "four per-case candidate scores only",
        "reason": "22 disk-offloaded layers made the seven-family all-row generation design computationally disproportionate",
        "replacement_frozen_scope": "three mid-layer families, 16 stratified units, all three candidate surfaces, direct-surface generation panel",
        "scientific_use_of_aborted_rows": "none",
    })
    model = None
    try:
        model, tokenizer, device = load_model()
        if COMPILED.exists():
            rows = read_rows(COMPILED)
        else:
            rows = contract.compile_rows(tokenizer, raw_rows)
            write_rows(COMPILED, rows)
        candidates = capture_candidate_field(model, tokenizer, device, rows)
        generation_rows = [row for row in rows if row["surface"] == GENERATION_SURFACE]
        generated = generation(model, tokenizer, device, generation_rows)
        behavior = behavior_ledger(rows, candidates, generated)
        save(OUT / "behavior/ledger.json", behavior)
        index = make_index(rows)
        write_rows(INDEX, index)
        field = np.load(FIELD, mmap_mode="r")
        result_structure = structure_tournament(field, index, behavior["qualified_families"])
        mmap = getattr(field, "_mmap", None)
        if mmap is not None:
            mmap.close()
    finally:
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    hashes = {"preregistration": file_hash(OUT / "protocol/preregistration.json"),
              "compiled": file_hash(COMPILED), "candidate": file_hash(OUT / "behavior/candidate_midlayer.jsonl"),
              "generation": file_hash(OUT / "behavior/generation_midlayer.jsonl"), "field": file_hash(FIELD),
              "passport": file_hash(OUT / "atlas/qwen14_selected_coordinate_passport.float32.npy")}
    checks = {"source_frozen": source["all_checks_passed"], "rows_exact": len(raw_rows) == 288,
              "field_shape": tuple(np.load(FIELD, mmap_mode="r").shape) ==
                             (288, len(QPOINTS), len(contract.ROLES), 5120),
              "behavior_complete": len(candidates) == len(raw_rows) and
                                   len(generated) == len(raw_rows) // len(contract.SURFACES),
              "lockbox_only_after_fresh_confirmation": all(
                  not value.get("passes") or result_structure["fresh_confirmation"][family].get("passes")
                  for family, value in result_structure["fresh_lockbox"].items()),
              "finite_passport": bool(not np.isinf(np.load(
                  OUT / "atlas/qwen14_selected_coordinate_passport.float32.npy")).any())}
    passed = result_structure["lockbox_passed_families"]
    result = {"phase": PHASE, "campaign": CAMPAIGN, "status": "closed",
              "timestamp": datetime.now().astimezone().isoformat(), "material_rows": len(raw_rows),
              "model": "Qwen3-14B", "precision": "float16", "placement": "cuda_disk_offload",
              "qpoints": QPOINTS, "behavior": behavior, "structure": result_structure,
              "hashes": hashes, "checks": checks, "all_checks_passed": all(checks.values()),
              "strict_conclusion": f"{len(passed)}/3 frozen Qwen3-4B mid-layer predictive topologies passed the Qwen3-14B relative-depth fresh lockbox; no physical coordinate identity or causal circuit is claimed.",
              "next_authorization": "Publish exact-coordinate atlases and clean non-published raw fields."}
    save(final, result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    run()
