#!/usr/bin/env python3
"""Prospectively replicate Qwen4 predictive topology in Qwen3-14B."""
from __future__ import annotations

import gc
import hashlib
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
CONTRACT_OUT = RESULT / "phase2265_c1433_c1468_independent_bilingual_contract"
Q4_OUT = RESULT / "phase2267_c1505_c1540_coordinate_model_tournament"
OUT = RESULT / "phase2269_c1577_c1600_qwen14_relative_topology_replication"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
sys.path.insert(0, str(TESTS))

import phase2163_c629_model_specific_worker as model_worker  # noqa: E402
import phase2265_c1433_c1468_independent_bilingual_contract as contract  # noqa: E402


PHASE = 2269
CAMPAIGNS = tuple(f"C{i}" for i in range(1577, 1601))
FAMILIES = ("location_state", "property_state", "patient_binding")
OFFSETS = (-2, -1, 0, 1, 2)
GAIN_GATE = 0.03
WIN_GATE = 0.55
EPS = 1e-8
FIELD = OUT / "raw/qwen3_14b_relative_window_field.float16.npy"
INDEX = OUT / "raw/field_index.jsonl"
PROGRESS = OUT / "raw/capture_progress.json"


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
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def close_mmap(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def parse_code(text: str, row: dict) -> str | None:
    clean = text.strip().lower()
    hits = []
    for code in (row["true_code"], row["false_code"]):
        match = re.search(rf"\b{re.escape(code.lower())}\b", clean)
        if match:
            hits.append((match.start(), code))
    return min(hits)[1] if hits else None


def generation(model, tokenizer, device, rows: list[dict]) -> list[dict]:
    pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    output = []
    for start in range(0, len(rows), 8):
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
                                       do_sample=False, pad_token_id=pad, eos_token_id=tokenizer.eos_token_id)
        for i, row in enumerate(batch):
            text = tokenizer.decode(generated[i, width:].tolist(), skip_special_tokens=True)
            parsed = parse_code(text, row)
            output.append({"case_id": row["case_id"], "text": text, "parsed": parsed,
                           "correct_answer": row["correct_answer"], "correct": parsed == row["correct_answer"]})
        if start % 48 == 0:
            print(f"[qwen14-generation] {min(start + len(batch), len(rows))}/{len(rows)}", flush=True)
    return output


def behavior(rows: list[dict], candidates: list[dict], generated: list[dict]) -> dict:
    c = {row["case_id"]: row for row in candidates}
    g = {row["case_id"]: row for row in generated}

    def summary(subset: list[dict]) -> dict:
        ca = float(np.mean([c[row["case_id"]]["correct"] for row in subset]))
        ga = float(np.mean([g[row["case_id"]]["correct"] for row in subset]))
        return {"rows": len(subset), "candidate_accuracy": ca, "generation_accuracy": ga,
                "dual_qualified": min(ca, ga) >= contract.BEHAVIOR_GATE}

    families, partitions = {}, {}
    for family in FAMILIES:
        families[family] = summary([row for row in rows if row["family"] == family])
        for partition in ("discovery", "confirmation", "fresh_confirmation", "fresh_lockbox"):
            partitions[f"{family}|{partition}"] = summary(
                [row for row in rows if row["family"] == family and row["partition"] == partition])
    qualified = [family for family in FAMILIES if families[family]["dual_qualified"] and
                 partitions[f"{family}|discovery"]["dual_qualified"] and
                 partitions[f"{family}|fresh_confirmation"]["dual_qualified"]]
    return {"rows": len(rows), "families": families, "partitions": partitions,
            "qualified_families": qualified,
            "candidate_accuracy": float(np.mean([row["correct"] for row in candidates])),
            "generation_accuracy": float(np.mean([row["correct"] for row in generated])),
            "parsed_generation_fraction": float(np.mean([row["parsed"] is not None for row in generated]))}


def relative_center(q4: int, q14_layers: int) -> int:
    return max(1, min(q14_layers, int(round(q4 / 36.0 * q14_layers))))


def capture(model, device, rows: list[dict], settings: dict) -> dict:
    base = model.model
    dim = int(base.embed_tokens.weight.shape[1])
    shape = (len(rows), len(OFFSETS), len(contract.ROLES), dim)
    FIELD.parent.mkdir(parents=True, exist_ok=True)
    completed = 0
    if FIELD.exists() and PROGRESS.exists():
        progress = load(PROGRESS)
        if tuple(progress["shape"]) != shape:
            raise RuntimeError(("resume_shape", progress["shape"], shape))
        completed = int(progress["completed_rows"])
        field = np.lib.format.open_memmap(FIELD, mode="r+")
    else:
        field = np.lib.format.open_memmap(FIELD, mode="w+", dtype=np.float16, shape=shape)
        save(PROGRESS, {"shape": list(shape), "completed_rows": 0})
    checkpoints = sorted({q for setting in settings.values() for q in setting["q14_window"]})
    captured: dict[int, torch.Tensor] = {}

    def make_hook(q: int):
        def hook(_module, _args, output):
            captured[q] = output[0] if isinstance(output, tuple) else output
        return hook

    handles = [base.layers[q - 1].register_forward_hook(make_hook(q)) for q in checkpoints]
    index_rows = []
    try:
        row_i = completed
        while row_i < len(rows):
            batch_size = 1
            while (batch_size < 6 and row_i + batch_size < len(rows)
                   and rows[row_i]["family"] == rows[row_i + batch_size]["family"]
                   and len(rows[row_i]["prompt_ids"]) == len(rows[row_i + batch_size]["prompt_ids"])):
                batch_size += 1
            batch = rows[row_i:row_i + batch_size]
            ids = torch.tensor([row["prompt_ids"] for row in batch], dtype=torch.long, device=device)
            mask = torch.ones_like(ids)
            pos = mask.long().cumsum(-1) - 1
            captured.clear()
            with torch.inference_mode():
                base(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False, return_dict=True)
            for local_i, row in enumerate(batch):
                window = settings[row["family"]]["q14_window"]
                for wi, q in enumerate(window):
                    values = captured[q][local_i]
                    for ri, role in enumerate(contract.ROLES):
                        field[row_i + local_i, wi, ri] = values[row["role_positions"][role][-1]].float().cpu().numpy().astype(np.float16)
                index_rows.append({"hidden_index": row_i + local_i, "case_id": row["case_id"],
                                   "family": row["family"], "language": row["language"],
                                   "unit": row["unit"], "surface": row["surface"], "state": row["state"],
                                   "partition": row["partition"], "q4_checkpoint": settings[row["family"]]["q4_checkpoint"],
                                   "q14_window": window, "roles": list(contract.ROLES)})
            row_i += batch_size
            field.flush()
            save(PROGRESS, {"shape": list(shape), "completed_rows": row_i})
            if row_i % 16 <= batch_size - 1:
                print(f"[qwen14-field] {row_i}/{len(rows)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
        field.flush()
        close_mmap(field)
    if INDEX.exists() and completed:
        prior = read_rows(INDEX)
        by_id = {row["case_id"]: row for row in prior}
        for row in index_rows:
            by_id[row["case_id"]] = row
        index_rows = [by_id[row["case_id"]] for row in rows]
    write_rows(INDEX, index_rows)
    return {"ran": True, "path": str(FIELD.relative_to(ROOT)), "shape": list(shape),
            "offsets": list(OFFSETS), "roles": list(contract.ROLES), "all_physical_coordinates": True}


def build_pairs(index: list[dict]) -> dict[str, dict[str, list[tuple[int, int, tuple]]]]:
    groups: dict[tuple, dict[int, int]] = {}
    for row in index:
        key = (row["family"], row["language"], int(row["unit"]), row["surface"], row["partition"])
        groups.setdefault(key, {})[int(row["state"])] = int(row["hidden_index"])
    output = {}
    for key, states in groups.items():
        family, language, unit, surface, partition = key
        output.setdefault(family, {}).setdefault(partition, []).append((states[0], states[1], (language, unit, surface)))
    for family in output:
        for partition in output[family]:
            output[family][partition].sort(key=lambda item: item[2])
    return output


def arrays(field, pairs, family, partition, wi, ri):
    cell = pairs[family][partition]
    h0 = np.asarray(field[[row[0] for row in cell], wi, ri], np.float32)
    h1 = np.asarray(field[[row[1] for row in cell], wi, ri], np.float32)
    return h0, h1, h1 - h0


def fit_affine(x, y):
    mx, my = x.mean(0), y.mean(0)
    dx = x - mx
    a = np.mean(dx * (y - my), 0) / np.maximum(np.mean(dx * dx, 0), EPS)
    return a.astype(np.float32), (my - a * mx).astype(np.float32)


def shuffled(h0, h1, labels):
    perm = np.arange(len(labels))
    groups = defaultdict(list)
    for i, (language, _unit, surface) in enumerate(labels):
        groups[(language, surface)].append(i)
    for members in groups.values():
        for position, member in enumerate(members):
            perm[member] = members[(position + 1) % len(members)]
    return h1[perm] - h0


def fit_models(field, pairs, families):
    shape = (len(families), len(OFFSETS), len(contract.ROLES), field.shape[-1])
    own_a, own_b = np.empty(shape, np.float32), np.empty(shape, np.float32)
    shuf_a, shuf_b = np.empty(shape, np.float32), np.empty(shape, np.float32)
    mean_r, mean_h1 = np.empty(shape, np.float32), np.empty(shape, np.float32)
    shared_a = np.empty((len(OFFSETS), len(contract.ROLES), field.shape[-1]), np.float32)
    shared_b = np.empty_like(shared_a)
    for wi in range(len(OFFSETS)):
        for ri in range(len(contract.ROLES)):
            px, py = [], []
            for fi, family in enumerate(families):
                h0, h1, response = arrays(field, pairs, family, "discovery", wi, ri)
                own_a[fi, wi, ri], own_b[fi, wi, ri] = fit_affine(h0, response)
                shuf_a[fi, wi, ri], shuf_b[fi, wi, ri] = fit_affine(
                    h0, shuffled(h0, h1, [row[2] for row in pairs[family]["discovery"]]))
                mean_r[fi, wi, ri], mean_h1[fi, wi, ri] = response.mean(0), h1.mean(0)
                px.append(h0); py.append(response)
            shared_a[wi, ri], shared_b[wi, ri] = fit_affine(np.concatenate(px), np.concatenate(py))
    return {"own_a": own_a, "own_b": own_b, "shuf_a": shuf_a, "shuf_b": shuf_b,
            "mean_r": mean_r, "mean_h1": mean_h1, "shared_a": shared_a, "shared_b": shared_b}


def pred(a, b, x):
    return a * x + b


def evaluate(field, pairs, families, models, family, partition, wi, ri, wrong_fi=None):
    fi = families.index(family)
    h0, _h1, truth = arrays(field, pairs, family, partition, wi, ri)
    own = pred(models["own_a"][fi, wi, ri], models["own_b"][fi, wi, ri], h0)
    controls = {
        "mean": np.broadcast_to(models["mean_r"][fi, wi, ri], truth.shape),
        "algebraic": models["mean_h1"][fi, wi, ri] - h0,
        "shared": pred(models["shared_a"][wi, ri], models["shared_b"][wi, ri], h0),
        "shuffled": pred(models["shuf_a"][fi, wi, ri], models["shuf_b"][fi, wi, ri], h0),
    }
    own_err = np.mean(np.abs(own - truth), 0)
    if wrong_fi is None:
        candidates = []
        for other in range(len(families)):
            if other == fi:
                continue
            wrong = pred(models["own_a"][other, wi, ri], models["own_b"][other, wi, ri], h0)
            candidates.append((float(np.mean(np.abs(wrong - truth))), other))
        wrong_fi = min(candidates)[1]
    controls["wrong"] = pred(models["own_a"][wrong_fi, wi, ri], models["own_b"][wrong_fi, wi, ri], h0)
    result = {"rows": len(truth), "mae": float(own_err.mean()), "wrong_family": families[wrong_fi]}
    gains, wins = [], []
    for name, value in controls.items():
        err = np.mean(np.abs(value - truth), 0)
        mae = float(err.mean())
        gain = float((mae - result["mae"]) / max(mae, EPS))
        win = float(np.mean(own_err < err))
        result[f"{name}_mae"], result[f"gain_vs_{name}"], result[f"win_vs_{name}"] = mae, gain, win
        gains.append(gain); wins.append(win)
    result["min_control_gain"], result["min_control_win"] = min(gains), min(wins)
    result["passes"] = bool(result["min_control_gain"] >= GAIN_GATE and result["min_control_win"] >= WIN_GATE)
    return result, wrong_fi


def replicate(field, index, families, settings):
    pairs = build_pairs(index)
    models = fit_models(field, pairs, families)
    decisions = []
    atlas_rows, atlas_labels = [], []
    for family in families:
        choices = []
        for wi in range(len(OFFSETS)):
            for ri in range(len(contract.ROLES)):
                metric, wrong = evaluate(field, pairs, families, models, family, "confirmation", wi, ri)
                choices.append((metric["min_control_gain"], metric["min_control_win"], -metric["mae"], wi, ri, wrong, metric))
        chosen = max(choices)
        wi, ri, wrong_fi, confirmation = chosen[3], chosen[4], chosen[5], chosen[6]
        fresh, _ = evaluate(field, pairs, families, models, family, "fresh_confirmation", wi, ri, wrong_fi)
        lock = None
        if fresh["passes"]:
            lock, _ = evaluate(field, pairs, families, models, family, "fresh_lockbox", wi, ri, wrong_fi)
        q14 = settings[family]["q14_window"][wi]
        decisions.append({"family": family, "offset": OFFSETS[wi], "q14_checkpoint": q14,
                          "relative_depth": q14 / settings[family]["q14_layers"],
                          "role": contract.ROLES[ri], "wrong_family": families[wrong_fi],
                          "confirmation": confirmation, "fresh_confirmation": fresh,
                          "lockbox_revealed": bool(fresh["passes"]), "lockbox": lock,
                          "replicated": bool(lock and lock["passes"])})
        partition = "fresh_lockbox" if lock is not None else "fresh_confirmation"
        fi = families.index(family)
        h0, _h1, truth = arrays(field, pairs, family, partition, wi, ri)
        values = {
            "truth_mean_abs": np.mean(np.abs(truth), 0),
            "own_abs_error": np.mean(np.abs(pred(models["own_a"][fi, wi, ri], models["own_b"][fi, wi, ri], h0) - truth), 0),
            "mean_abs_error": np.mean(np.abs(models["mean_r"][fi, wi, ri] - truth), 0),
            "algebraic_abs_error": np.mean(np.abs((models["mean_h1"][fi, wi, ri] - h0) - truth), 0),
        }
        for metric, row in values.items():
            atlas_labels.append({"row": len(atlas_rows), "family": family, "partition": partition,
                                 "q14_checkpoint": q14, "role": contract.ROLES[ri], "metric": metric})
            atlas_rows.append(row.astype(np.float16))
    atlas = np.stack(atlas_rows)
    atlas_path = OUT / "atlas/qwen14_relative_topology_coordinates.float16.npy"
    atlas_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(atlas_path, atlas)
    labels_path = OUT / "atlas/qwen14_relative_topology_coordinates.rows.jsonl"
    write_rows(labels_path, atlas_labels)
    return decisions, atlas_path, labels_path, atlas


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: Qwen3-14B相对层深全坐标前瞻复现（C1577-C1600） [{stamp}]

**测试原理与用例。** Phase2268没有严格双向因果家族，因此本期不声称复现因果。根据Phase2267锁箱最小控制增益，在运行14B前冻结三个不同类型的最强预测家族：位置状态、属性状态和受事绑定。使用Phase2265同一套独立干净材料，每族256行；先要求总体、discovery和fresh confirmation双行为通过。把4B候选检查点按相对深度映射到14B，并预先保存中心±2层、六角色、全部5120物理坐标。14B自己的confirmation在30个层角色格中选择候选，fresh confirmation过门后才揭示fresh lockbox；物理坐标编号不跨模型对齐。

**公式。**

$$
q_{{14}}=\operatorname{{round}}\!\left(\frac{{q_4}}{{36}}L_{{14}}\right)+\delta,\quad
\delta\in\{{-2,-1,0,1,2\}},\qquad
\widehat R_{{i,j}}=a_{{f,j}}H^0_{{i,j}}+b_{{f,j}}.
$$

模型竞赛仍同时要求相对家族均值、纯代数、三族共享、错配和错家族的MAE增益不低于0.03，逐坐标胜率不低于0.55。使用全5120坐标，不做Top-K、PCA或余弦筛选。

**结果汇总。** 14B行为 `{json.dumps(result['behavior'], ensure_ascii=False)}`；冻结相对窗口 `{json.dumps(result['settings'], ensure_ascii=False)}`；全坐标场 `{json.dumps(result['field'], ensure_ascii=False)}`；家族裁决 `{json.dumps(result['decisions'], ensure_ascii=False)}`；正式复现家族 `{json.dumps(result['replicated_families'], ensure_ascii=False)}`；哈希 `{json.dumps(result['hashes'], ensure_ascii=False)}`；工程检查 `{result['checks']}`，总通过 `{result['all_checks_passed']}`。

**分析、理论进展与边界。** `{result['strict_conclusion']}` 跨规模复现只比较相对层深、角色类型和“模型自身坐标上的条件预测胜过控制”，不比较相同坐标编号，也不证明共享权重、电路或新数学。14B与4B同属Qwen3架构，仍不是异架构普适性。

**问题、硬伤、结论与相关文件。** 只前瞻冻结三族和±2层窗口；候选仍由30格选择；模型磁盘卸载会降低速度但不改数值合同；材料无人类盲评、元语言输出码和float16写盘仍是限制。脚本 `tests/glm5/phase2269_c1577_c1600_qwen14_relative_topology_replication.py`；结果 `tests/glm5/result/phase2269_c1577_c1600_qwen14_relative_topology_replication`。下一步发布4B/14B全坐标图谱和4B全token场，并清理未展示的原始样本场。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def run() -> dict:
    final = OUT / "analysis/final.json"
    if final.exists():
        return load(final)
    q4 = load(Q4_OUT / "analysis/final.json")
    q4_decisions = {row["family"]: row for row in q4["decisions"]}
    raw = [row for row in read_rows(CONTRACT_OUT / "material/independent_bilingual_cases.jsonl") if row["family"] in FAMILIES]
    freeze = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "phase": PHASE,
              "families": list(FAMILIES), "selection_basis": "top three Phase2267 lockbox minimum-control gains across distinct family types",
              "offsets": list(OFFSETS), "gates": {"gain": GAIN_GATE, "win": WIN_GATE},
              "frozen_before_qwen14": True}
    save(OUT / "protocol/preregistration.json", freeze)
    model = None
    try:
        model, tokenizer, device, placement, loader = model_worker.load_model("qwen3_14b")
        compiled = contract.compile_rows(tokenizer, raw)
        write_rows(OUT / "material/qwen14_compiled.jsonl", compiled)
        candidate_path = OUT / "behavior/candidate.jsonl"
        generation_path = OUT / "behavior/generation.jsonl"
        if candidate_path.exists() and generation_path.exists():
            candidates, generated = read_rows(candidate_path), read_rows(generation_path)
        else:
            candidates = contract.legacy.parent.model_base.behavior_base.batch_behavior(model, device, compiled, batch_size=8)
            generated = generation(model, tokenizer, device, compiled)
            write_rows(candidate_path, candidates); write_rows(generation_path, generated)
        ledger = behavior(compiled, candidates, generated)
        layers = len(model.model.layers)
        settings = {}
        for family in FAMILIES:
            q4_checkpoint = int(q4_decisions[family]["checkpoint"])
            center = relative_center(q4_checkpoint, layers)
            window = [max(1, min(layers, center + offset)) for offset in OFFSETS]
            settings[family] = {"q4_checkpoint": q4_checkpoint, "q4_role": q4_decisions[family]["role"],
                                "q14_layers": layers, "q14_center": center, "q14_window": window}
        save(OUT / "protocol/frozen_relative_windows.json", settings)
        observed = [row for row in compiled if row["family"] in ledger["qualified_families"]]
        field_info = capture(model, device, observed, settings) if observed else {"ran": False, "reason": "no_behavior_qualified_family"}
    finally:
        model_worker.release_model("qwen3_14b", model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if field_info.get("ran"):
        field = np.load(FIELD, mmap_mode="r")
        index = read_rows(INDEX)
        try:
            decisions, atlas_path, labels_path, atlas = replicate(field, index, ledger["qualified_families"], settings)
        finally:
            close_mmap(field)
    else:
        decisions, atlas_path, labels_path, atlas = [], None, None, np.empty((0, 5120), np.float16)
    replicated = [row["family"] for row in decisions if row["replicated"]]
    checks = {
        "three_families_only": len(raw) == 768 and set(row["family"] for row in raw) == set(FAMILIES),
        "behavior_complete": ledger["rows"] == 768,
        "relative_windows_frozen": set(settings) == set(FAMILIES) and all(len(value["q14_window"]) == 5 for value in settings.values()),
        "field_matches_qualification": (not field_info.get("ran")) or field_info["shape"][0] == len(observed),
        "all_coordinates": (not field_info.get("ran")) or field_info["shape"][-1] == 5120,
        "decisions_complete": len(decisions) == len(ledger["qualified_families"]),
        "ordered_reveal": all(row["lockbox_revealed"] == row["fresh_confirmation"]["passes"] for row in decisions),
        "atlas_all_coordinates": atlas.shape[-1] == 5120,
    }
    hashes = {"preregistration": file_hash(OUT / "protocol/preregistration.json"),
              "windows": file_hash(OUT / "protocol/frozen_relative_windows.json"),
              "candidate": file_hash(OUT / "behavior/candidate.jsonl"), "generation": file_hash(OUT / "behavior/generation.jsonl"),
              "field": file_hash(FIELD) if FIELD.exists() else None, "index": file_hash(INDEX) if INDEX.exists() else None,
              "atlas": file_hash(atlas_path) if atlas_path else None, "atlas_rows": file_hash(labels_path) if labels_path else None}
    strict = (f"{len(replicated)}/{len(FAMILIES)} preregistered families replicated a model-local coordinate predictor in Qwen3-14B; "
              "this is relative-depth/role-topology replication, not coordinate identity or causality.")
    result = {"phase": PHASE, "campaigns": list(CAMPAIGNS), "status": "closed",
              "timestamp": datetime.now().astimezone().isoformat(), "loader": loader, "placement": placement,
              "behavior": ledger, "settings": settings, "field": field_info, "decisions": decisions,
              "replicated_families": replicated,
              "atlas": {"path": str(atlas_path.relative_to(ROOT)) if atlas_path else None,
                        "rows": str(labels_path.relative_to(ROOT)) if labels_path else None,
                        "shape": list(atlas.shape), "all_coordinates": True},
              "hashes": hashes, "checks": checks, "all_checks_passed": all(checks.values()),
              "strict_conclusion": strict,
              "next_authorization": "Publish Qwen4/Qwen14 coordinate atlases and Qwen4 token field, then clean undisplayed raw fields."}
    save(final, result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


if __name__ == "__main__":
    run()
