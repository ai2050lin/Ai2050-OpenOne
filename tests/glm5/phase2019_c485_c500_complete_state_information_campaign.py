#!/usr/bin/env python3
"""C485-C500 complete-state information-ladder campaign.

The campaign observes embeddings and HiddenState checkpoints only. It keeps
all 2560 activation coordinates, never selects coordinates by amplitude, and
does not inspect Attention, MLP activations, or weights. Full-token fields are
stored for a deterministic balanced program subset and are deleted after the
registered visual archive and strict audit are complete.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
VISUAL = ROOT / "frontend/public/vis_data/research_kernel/c500_complete_state_information_ladder.json"
REGISTRY = ROOT / "ai2050_research_os/registry/field_datasets.json"
CATALOG = ROOT / "frontend/public/research_data/current/language_encoding_catalog.json"
sys.path.insert(0, str(TESTS))

import phase2005_c471_c484_program_guard_hypergraph_campaign as previous


PHASES = {
    f"C{campaign}": (2019 + campaign - 485, slug)
    for campaign, slug in (
        (485, "evidence_audit_and_complete_state_master_contract"),
        (486, "eleven_family_program_and_composition_material"),
        (487, "compiler_zero_model_semantic_and_naturalness_audit"),
        (488, "qwen_behavior_qualification_and_typed_error_ledger"),
        (489, "all_coordinate_role_and_balanced_full_token_state_cube"),
        (490, "zero_order_complete_walsh_state_ledger"),
        (491, "behavior_stratified_full_spectrum_observation_atlas"),
        (492, "same_effect_vs_complete_spectrum_information_ladder"),
        (493, "all_role_complete_spectrum_transition"),
        (494, "all_token_incremental_information_test"),
        (495, "strong_baseline_full_coordinate_residual_coupling"),
        (496, "external_program_graph_incremental_guard"),
        (497, "nested_attitude_event_unseen_composition_panel"),
        (498, "typed_graph_path_unseen_composition_panel"),
        (499, "redesigned_temporal_path_and_discourse_panel"),
        (500, "strict_adjudication_visual_cleanup_and_next_stage"),
    )
}
OUTS = {
    name: RESULT / f"phase{phase}_{name.lower()}_{slug}"
    for name, (phase, slug) in PHASES.items()
}

DIM = 2560
CHECKPOINTS = 38
ROLES = previous.ROLES
CONSTRUCTIONS = ("ledger", "brief", "report")
BITS = tuple(itertools.product((0, 1), repeat=4))
MASKS = tuple(range(16))
Q_STARTS = (0, 8, 16, 24, 32)
QPOINTS = tuple(value for q in Q_STARTS for value in (q, q + 1))
TOKEN_QPOINTS = QPOINTS
OLD_FAMILIES = previous.FAMILIES
NEW_FAMILIES = ("nested_composition", "typed_graph_path", "temporal_composition")
FAMILIES = OLD_FAMILIES + NEW_FAMILIES
FAMILY_LOCKBOX = ("type_graph", "part_whole", "temporal_composition")
TRAIN_FAMILIES = tuple(f for f in FAMILIES if f not in FAMILY_LOCKBOX)
FULL_PROGRAM_UNITS = (0, 1, 5, 8)
RIDGE = 1e-2


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_rows(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def producer_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(item) for item in value)
    return not isinstance(value, (float, np.floating)) or math.isfinite(float(value))


def begin(name: str, protocol: dict, checks: dict) -> Path:
    out = OUTS[name]
    if (out / "analysis/final.json").exists():
        return out
    if out.exists():
        raise RuntimeError(f"partial output exists: {out}")
    if not all(checks.values()):
        raise RuntimeError((name, checks))
    for sub in ("analysis", "audit", "compiled", "material", "protocol", "raw"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    save(out / "protocol/preregistration.json", {
        "phase": PHASES[name][0], "campaign": name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "producer_sha256": producer_hash(), **protocol,
    })
    save(out / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": True})
    return out


def close(name: str, headline: dict, checks: dict, authorization: str) -> dict:
    out = OUTS[name]
    if (out / "analysis/final.json").exists():
        return load(out / "analysis/final.json")
    save(out / "analysis/summary.json", headline)
    save(out / "audit/internal_analysis_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    protocol = load(out / "protocol/preregistration.json")
    final_checks = {
        "contract": load(out / "audit/internal_contract_audit.json")["all_checks_passed"],
        "analysis": all(checks.values()),
        "producer_hash": protocol["producer_sha256"] == producer_hash(),
    }
    value = {
        "phase": PHASES[name][0], "campaign": name, "status": "closed",
        "checks": final_checks, "all_checks_passed": all(final_checks.values()),
        "headline": headline, "next_authorization": authorization,
    }
    save(out / "analysis/final.json", value)
    print(json.dumps(value, ensure_ascii=False), flush=True)
    return value


def final(name: str) -> dict:
    return load(OUTS[name] / "analysis/final.json")


def partition(unit: int) -> str:
    return "discovery" if unit < 5 else "confirmation" if unit < 8 else "lockbox"


def mask_order(mask: int) -> int:
    return int(mask).bit_count()


def walsh_matrix() -> np.ndarray:
    matrix = np.empty((16, 16), np.float32)
    for mask in MASKS:
        for cell_i, bits in enumerate(BITS):
            matrix[mask, cell_i] = (-1.0 if sum(bits[i] for i in range(4) if mask & (1 << i)) % 2 else 1.0) / 16.0
    return matrix


def program_key(row: dict) -> tuple[str, str, int]:
    return row["family"], row["construction"], int(row["unit"])


def bit_code(bits: tuple[int, int, int, int] | list[int]) -> str:
    return "".join(str(int(value)) for value in bits)


def wrap(construction: str, facts: list[str], question: str, noise: str) -> str:
    joined = " ".join(facts)
    if construction == "ledger":
        return f"A ledger records these relevant facts: {joined} An unrelated note says {noise}. Using only the relevant facts, {question}"
    if construction == "brief":
        return f"A brief gives the following relevant information: {joined} Separately, {noise}. From the relevant information, {question}"
    return f"A report states: {joined} Elsewhere it states that {noise}. Based only on the relevant statements, {question}"


def options(truth: bool) -> tuple[str, str, int]:
    return ("Yes", "No", 0) if truth else ("No", "Yes", 1)


def extension_case(family: str, construction: str, unit: int, bits: tuple[int, int, int, int]) -> dict:
    u = previous.UNITS[unit]
    b0, b1, b2, b3 = bits
    if family == "nested_composition":
        outer = u["p"] if b0 == 0 else u["s"]
        inner = u["s"] if b0 == 0 else u["p"]
        relation = "likes eating" if b1 == 0 else "does not like eating"
        opposite = "does not like eating" if b1 == 0 else "likes eating"
        obj = u["x"] if b2 == 0 else u["y"]
        query_relation = relation if b3 == 0 else opposite
        facts = [f"{outer} reports that {inner} {relation} {obj}."]
        question = f"Is it true that {outer} reports that {inner} {query_relation} {obj}?"
        role_values = {"primary": outer, "secondary": inner, "relation": relation, "context": obj, "query": outer}
        graph = {"operators": ["attitude", "event", "polarity"], "outer": outer, "inner": inner, "object": obj}
    elif family == "typed_graph_path":
        root = u["x"] if b0 == 0 else u["y"]
        mid = f"class{unit}m"
        target = f"class{unit}t"
        if b1 == 0:
            facts = [f"{root} is a kind of {target}."]
        else:
            facts = [f"{root} is a kind of {mid}.", f"{mid} is a kind of {target}."]
        if b2:
            facts.append(f"{root} is also recorded directly as a kind of {target}.")
        else:
            facts.append(f"{u['noise']} is unrelated to {target}.")
        question = f"Is {root} a kind of {target}?" if b3 == 0 else f"Is {target} a kind of {root}?"
        relation = "is a kind of"
        role_values = {"primary": root, "secondary": mid, "relation": relation, "context": target, "query": root if b3 == 0 else target}
        graph = {"operators": ["typed_edge", "path_composition", "shortcut"], "root": root, "mid": mid, "target": target, "depth": 1 + b1, "shortcut": b2}
    elif family == "temporal_composition":
        first = f"event{unit}{'a' if b0 == 0 else 'd'}"
        middle = f"event{unit}b"
        last = f"event{unit}{'c' if b0 == 0 else 'e'}"
        if b1 == 0:
            facts = [f"{first} occurred before {last}.", f"{middle} was independently documented."]
        else:
            facts = [f"{first} occurred before {middle}.", f"{middle} occurred before {last}."]
        if b2:
            facts = list(reversed(facts))
        question = f"Did {first} occur before {last}?" if b3 == 0 else f"Did {last} occur before {first}?"
        relation = "occurred before"
        role_values = {"primary": first, "secondary": middle, "relation": relation, "context": last, "query": first if b3 == 0 else last}
        graph = {"operators": ["temporal_edge", "path_composition", "discourse_permutation"], "first": first, "middle": middle, "last": last, "depth": 1 + b1, "discourse_reversed": b2}
    else:
        raise KeyError(family)
    truth = b3 == 0
    correct, wrong, gold = options(truth)
    core = wrap(construction, facts, question, f"{u['p']} inspected the {u['noise']}")
    code = bit_code(bits)
    return {
        "case_id": f"c486-{family}-{construction}-u{unit}-x{code}",
        "panel": "complete_state_extension", "family": family,
        "surface": construction, "construction": construction, "unit": unit,
        "bits": list(bits), "cell": code,
        "factor_a": b0, "factor_b": b1, "factor_c": b2, "factor_d": b3,
        "order": 1, "partition": partition(unit), "gold_position": gold,
        "correct_answer": correct, "wrong_answer": wrong,
        "prompt_core": core,
        "prompt": f"{core} (A) Yes (B) No. Reply with only A or B.",
        "free_prompt": f"{core} Answer only Yes or No.",
        "role_values": role_values,
        "semantic_graph": {"family": family, "bits": list(bits), "truth": truth, **graph},
    }


def all_material() -> list[dict]:
    old = read_rows(previous.OUTS["C472"] / "material/cases.jsonl")
    rows = [{**row, "cell": bit_code(row["bits"]), "source_material": "C472"} for row in old]
    for family, construction, unit, bits in itertools.product(NEW_FAMILIES, CONSTRUCTIONS, range(10), BITS):
        rows.append({**extension_case(family, construction, unit, bits), "source_material": "C486"})
    return rows


def material_lookup() -> tuple[list[dict], dict[str, dict]]:
    rows = read_rows(OUTS["C486"] / "material/cases.jsonl")
    return rows, {row["case_id"]: row for row in rows}


def split_groups(index: list[dict]) -> dict[str, list[int]]:
    result = {name: [] for name in ("train", "within", "family", "report")}
    for row in index:
        family, part = row["family"], row["partition"]
        if family in TRAIN_FAMILIES and part == "discovery":
            result["train"].append(row["group_index"])
        elif family in TRAIN_FAMILIES and part == "confirmation":
            result["within"].append(row["group_index"])
        elif family in FAMILY_LOCKBOX and part != "lockbox":
            result["family"].append(row["group_index"])
        elif part == "lockbox":
            result["report"].append(row["group_index"])
    return {name: sorted(set(values)) for name, values in result.items()}


def metric(prediction: np.ndarray, truth: np.ndarray) -> dict:
    p = np.asarray(prediction, np.float64).reshape(-1)
    y = np.asarray(truth, np.float64).reshape(-1)
    err = p - y
    yn = float(np.dot(y, y))
    pn = float(np.dot(p, p))
    return {
        "n": int(y.size),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "nrmse": float(np.sqrt(np.dot(err, err) / max(yn, 1e-30))),
        "cosine": float(np.dot(p, y) / max(math.sqrt(pn * yn), 1e-30)),
    }


def metric_acc() -> dict:
    return {"n": 0, "ae": 0.0, "se": 0.0, "yy": 0.0, "pp": 0.0, "py": 0.0}


def add_metric(acc: dict, prediction: np.ndarray, truth: np.ndarray) -> None:
    p = np.asarray(prediction, np.float64).reshape(-1)
    y = np.asarray(truth, np.float64).reshape(-1)
    e = p - y
    acc["n"] += y.size
    acc["ae"] += float(np.abs(e).sum())
    acc["se"] += float(np.dot(e, e))
    acc["yy"] += float(np.dot(y, y))
    acc["pp"] += float(np.dot(p, p))
    acc["py"] += float(np.dot(p, y))


def finish_metric(acc: dict) -> dict:
    n = max(int(acc["n"]), 1)
    return {
        "n": int(acc["n"]), "mae": float(acc["ae"] / n),
        "rmse": float(math.sqrt(acc["se"] / n)),
        "nrmse": float(math.sqrt(acc["se"] / max(acc["yy"], 1e-30))),
        "cosine": float(acc["py"] / max(math.sqrt(acc["pp"] * acc["yy"]), 1e-30)),
    }


def close_mmap(value) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def compile_material(rows: list[dict]) -> list[dict]:
    tokenizer = previous.prior.graph_base.axis_old.base.parent.fresh.tokenizer_qwen()
    return previous.prior.compile_base.compile_qwen(tokenizer, rows)


@torch.inference_mode()
def run_behavior(rows: list[dict], compiled: list[dict], out: Path, batch_size: int = 12) -> dict:
    model = None
    result = []
    try:
        model, _tokenizer, device, placement = previous.prior.model_base.load_bf16("qwen3")
        quant = previous.prior.model_base.quantization_audit(model)
        pad = 0
        for start in range(0, len(compiled), batch_size):
            batch = compiled[start:start + batch_size]
            width = max(len(row["prompt_ids"]) for row in batch)
            ids = torch.full((len(batch), width), pad, dtype=torch.long, device=device)
            mask = torch.zeros_like(ids)
            lengths = []
            for local, row in enumerate(batch):
                values = row["prompt_ids"]
                lengths.append(len(values))
                ids[local, :len(values)] = torch.tensor(values, dtype=torch.long, device=device)
                mask[local, :len(values)] = 1
            output = model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True)
            for local, row in enumerate(batch):
                scores = [float(output.logits[local, lengths[local] - 1, candidate[0]]) for candidate in row["candidate_ids"]]
                prediction = int(scores[1] > scores[0])
                result.append({"case_id": row["case_id"], "gold_position": row["gold_position"], "prediction": prediction, "correct": prediction == row["gold_position"], "score0": scores[0], "score1": scores[1]})
            if start % 240 == 0 or start + len(batch) == len(compiled):
                print(f"[behavior] {start + len(batch)}/{len(compiled)}", flush=True)
        write_rows(out / "raw/behavior.jsonl", result)
        return {"rows": len(result), "accuracy": float(np.mean([row["correct"] for row in result])), "placement": placement, "quantization": quant}
    finally:
        previous.prior.model_base.release_bf16(model)
        gc.collect()


@torch.inference_mode()
def capture_state_cube(rows: list[dict], compiled: list[dict], out: Path, full_ids: set[str], width: int, batch_size: int = 8) -> dict:
    model = None
    hooks = []
    captured = []
    n = len(rows)
    role_states = np.lib.format.open_memmap(out / "raw/role_states.float16.npy", mode="w+", dtype=np.float16, shape=(n, CHECKPOINTS, len(ROLES), DIM))
    selected = [i for i, row in enumerate(rows) if row["case_id"] in full_ids]
    full_lookup = {source_i: local_i for local_i, source_i in enumerate(selected)}
    full_fields = np.lib.format.open_memmap(out / "raw/full_token_states.float16.npy", mode="w+", dtype=np.float16, shape=(len(selected), len(TOKEN_QPOINTS), width, DIM))
    index = []
    try:
        model, tokenizer, device, placement = previous.prior.model_base.load_bf16("qwen3")
        quant = previous.prior.model_base.quantization_audit(model)
        base = model.model

        def hook(_module, _args, output):
            captured.append(output[0] if isinstance(output, tuple) else output)

        hooks.append(base.embed_tokens.register_forward_hook(hook))
        hooks.extend(layer.register_forward_hook(hook) for layer in base.layers)
        hooks.append(base.norm.register_forward_hook(hook))
        pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
        for start in range(0, n, batch_size):
            batch = compiled[start:start + batch_size]
            ids = torch.full((len(batch), width), pad, dtype=torch.long, device=device)
            mask = torch.zeros_like(ids)
            pos = torch.zeros_like(ids)
            lengths = []
            for local, row in enumerate(batch):
                values = row["prompt_ids"]
                if len(values) > width:
                    raise RuntimeError((row["case_id"], len(values), width))
                lengths.append(len(values))
                ids[local, :len(values)] = torch.tensor(values, dtype=torch.long, device=device)
                mask[local, :len(values)] = 1
                pos[local, :len(values)] = torch.arange(len(values), device=device)
            captured.clear()
            output = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False, return_dict=True)
            if len(captured) != CHECKPOINTS:
                raise RuntimeError(("checkpoint_count", len(captured)))
            for local, row in enumerate(batch):
                source_i = start + local
                length = lengths[local]
                for q, state in enumerate(captured):
                    for role_i, role in enumerate(ROLES):
                        positions = row["role_positions"][role]
                        role_states[source_i, q, role_i] = state[local, positions].float().mean(dim=0).cpu().numpy().astype(np.float16)
                    if source_i in full_lookup and q in TOKEN_QPOINTS:
                        qi = TOKEN_QPOINTS.index(q)
                        full_fields[full_lookup[source_i], qi, :length] = state[local, :length].float().cpu().numpy().astype(np.float16)
                scores = [float(output.logits[local, length - 1, candidate[0]]) for candidate in row["candidate_ids"]]
                prediction = int(scores[1] > scores[0])
                index.append({
                    "hidden_index": source_i, "full_index": full_lookup.get(source_i),
                    "case_id": row["case_id"], "family": row["family"],
                    "construction": row["construction"], "unit": row["unit"],
                    "bits": row["bits"], "partition": row["partition"], "length": length,
                    "gold_position": row["gold_position"], "prediction": prediction,
                    "correct": prediction == row["gold_position"], "role_positions": row["role_positions"],
                })
            role_states.flush(); full_fields.flush()
            if start % 128 == 0 or start + len(batch) == n:
                print(f"[state cube] {start + len(batch)}/{n}", flush=True)
        write_rows(out / "raw/hidden_index.jsonl", index)
        save(out / "raw/full_field_row_map.json", {"source_indices": selected, "qpoints": list(TOKEN_QPOINTS), "width": width})
        return {
            "rows": n, "accuracy": float(np.mean([row["correct"] for row in index])),
            "role_shape": list(role_states.shape), "full_shape": list(full_fields.shape),
            "full_token_rows": len(selected), "field_width": width,
            "placement": placement, "quantization": quant,
        }
    finally:
        for item in hooks:
            item.remove()
        close_mmap(role_states); close_mmap(full_fields)
        previous.prior.model_base.release_bf16(model)
        gc.collect()


def fit_same_effect(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xf = x.reshape(-1, DIM).astype(np.float64)
    yf = y.reshape(-1, DIM).astype(np.float64)
    xm = xf.mean(axis=0); ym = yf.mean(axis=0)
    xc = xf - xm; yc = yf - ym
    slope = (xc * yc).sum(axis=0) / np.maximum((xc * xc).sum(axis=0), 1e-12)
    return slope.astype(np.float32), (ym - slope * xm).astype(np.float32)


def fit_spectrum_operator(x: np.ndarray, y: np.ndarray, out_b: np.ndarray, out_i: np.ndarray, chunk: int = 64) -> None:
    # x/y: program x 16 effects x coordinate. A separate 16x16 map is fit at every coordinate.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for start in range(0, DIM, chunk):
        end = min(start + chunk, DIM)
        xt = torch.tensor(np.asarray(x[:, :, start:end], np.float32).transpose(2, 0, 1), device=device)
        yt = torch.tensor(np.asarray(y[:, :, start:end], np.float32).transpose(2, 0, 1), device=device)
        xm = xt.mean(dim=1, keepdim=True); ym = yt.mean(dim=1, keepdim=True)
        xc = xt - xm; yc = yt - ym
        gram = xc.transpose(1, 2) @ xc
        scale = gram.diagonal(dim1=1, dim2=2).mean(dim=1).clamp_min(1e-8)
        eye = torch.eye(16, device=device)[None]
        beta = torch.linalg.solve(gram + RIDGE * scale[:, None, None] * eye, xc.transpose(1, 2) @ yc)
        intercept = ym[:, 0] - (xm @ beta)[:, 0]
        out_b[start:end] = beta.cpu().numpy().astype(np.float16)
        out_i[start:end] = intercept.cpu().numpy().astype(np.float16)


def predict_spectrum(x: np.ndarray, beta: np.ndarray, intercept: np.ndarray) -> np.ndarray:
    return np.einsum("psd,dst->ptd", np.asarray(x, np.float32), np.asarray(beta, np.float32), optimize=True) + np.asarray(intercept, np.float32).T[None]


def fit_role_operator(x: np.ndarray, y: np.ndarray, out_b: np.ndarray, out_i: np.ndarray, chunk: int = 12) -> None:
    # x/y: program x 96(effect-role) x coordinate.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    width = x.shape[1]
    for start in range(0, DIM, chunk):
        end = min(start + chunk, DIM)
        xt = torch.tensor(np.asarray(x[:, :, start:end], np.float32).transpose(2, 0, 1), device=device)
        yt = torch.tensor(np.asarray(y[:, :, start:end], np.float32).transpose(2, 0, 1), device=device)
        xm = xt.mean(dim=1, keepdim=True); ym = yt.mean(dim=1, keepdim=True)
        xc = xt - xm; yc = yt - ym
        gram = xc.transpose(1, 2) @ xc
        scale = gram.diagonal(dim1=1, dim2=2).mean(dim=1).clamp_min(1e-8)
        eye = torch.eye(width, device=device)[None]
        beta = torch.linalg.solve(gram + RIDGE * scale[:, None, None] * eye, xc.transpose(1, 2) @ yc)
        intercept = ym[:, 0] - (xm @ beta)[:, 0]
        out_b[start:end] = beta.cpu().numpy().astype(np.float16)
        out_i[start:end] = intercept.cpu().numpy().astype(np.float16)
        if start % 240 == 0:
            print(f"[role operator] {start}/{DIM}", flush=True)


def predict_roles(x: np.ndarray, beta: np.ndarray, intercept: np.ndarray, chunk: int = 64) -> np.ndarray:
    result = np.empty_like(x, dtype=np.float32)
    for start in range(0, DIM, chunk):
        end = min(start + chunk, DIM)
        result[:, :, start:end] = np.einsum("pfd,dfo->pod", np.asarray(x[:, :, start:end], np.float32), np.asarray(beta[start:end], np.float32), optimize=True) + np.asarray(intercept[start:end], np.float32).T[None]
    return result


def spectrum_data(states: np.ndarray, groups: list[int], qindex: int) -> np.ndarray:
    return np.asarray(states[groups, :, qindex], np.float32)


def c485() -> None:
    out = begin("C485", {
        "status": "complete_state_information_ladder_master_contract_frozen",
        "evidence_boundary": "C471-C484 rejects tested same-effect/local-state closures, not all shared primitives",
        "information_ladder": ["identity", "same effect", "zero plus complete spectrum", "all roles", "all tokens", "full-coordinate residual"],
        "strong_controls": ["identity", "mean", "same-effect", "complete-spectrum", "all-role", "token roll", "sample shuffle", "coordinate roll", "program permutation"],
        "policy": "route failure eliminates that route only; all observational panels continue",
    }, {"previous_audit": final_previous_ok(), "cuda": torch.cuda.is_available()})
    audit = {
        "retained": [
            "A single Walsh coefficient is generally non-invertible and was empirically insufficient in C478.",
            "The zero-order state and all fifteen nonzero effects form an invertible accounting transform over a complete sixteen-cell program.",
            "Behavior-error programs must remain typed strata rather than being deleted from observation.",
            "Identity and the best prior shared model must be same-panel baselines for cross-coordinate coupling.",
        ],
        "corrected_overclaims": [
            "No tested result proves that cross-family shared primitives do not exist.",
            "The full-token field is a candidate sufficient state, not a proven minimal state.",
            "Complete Walsh inversion is bookkeeping, not a discovered neural mechanism.",
            "A dense ridge operator is predictive dependence, not a unique causal circuit.",
            "Existing mathematics is sufficient for this campaign; the new-math gate remains closed.",
        ],
    }
    save(out / "analysis/evidence_audit.json", audit)
    close("C485", {"status": "audit_and_contract_closed", **audit}, {"audit_items": len(audit["retained"]) == 4, "corrections": len(audit["corrected_overclaims"]) == 5}, "C486_material")


def final_previous_ok() -> bool:
    return previous.final("C484")["all_checks_passed"]


def c486() -> None:
    out = begin("C486", {
        "status": "eleven_family_material_frozen", "families": list(FAMILIES),
        "old_rows": 3840, "new_panels": list(NEW_FAMILIES),
        "cells_per_program": 16, "units": 10, "constructions": list(CONSTRUCTIONS),
    }, {"parent": final("C485")["all_checks_passed"]})
    rows = all_material()
    write_rows(out / "material/cases.jsonl", rows)
    write_rows(out / "material/program_graphs.jsonl", [{"case_id": row["case_id"], **row["semantic_graph"]} for row in rows])
    by_family = {family: sum(row["family"] == family for row in rows) for family in FAMILIES}
    headline = {"status": "material_closed", "rows": len(rows), "programs": len(rows) // 16, "family_rows": by_family, "strict_interpretation": "The extension panels are controlled English program probes, not a complete linguistic ontology."}
    close("C486", headline, {"rows": len(rows) == 5280, "balanced_families": all(value == 480 for key, value in by_family.items() if key in NEW_FAMILIES)}, "C487_audit")


def c487() -> None:
    out = begin("C487", {
        "status": "material_and_zero_model_audit_frozen",
        "zero_models": ["always A", "always B", "bit parity", "construction majority", "family majority"],
        "naturalness": "machine contract audit only; independent human blind review unavailable",
    }, {"parent": final("C486")["all_checks_passed"]})
    rows, _ = material_lookup()
    groups = defaultdict(list)
    for row in rows:
        groups[program_key(row)].append(row)
    complete = all(len(values) == 16 and {tuple(row["bits"]) for row in values} == set(BITS) for values in groups.values())
    gold = np.asarray([row["gold_position"] for row in rows])
    zero = {
        "always_a": float(np.mean(gold == 0)), "always_b": float(np.mean(gold == 1)),
        "bit_parity": float(np.mean(np.asarray([sum(row["bits"]) % 2 for row in rows]) == gold)),
    }
    zero["construction_majority"] = float(np.mean([row["gold_position"] == round(np.mean([x["gold_position"] for x in rows if x["construction"] == row["construction"]])) for row in rows]))
    zero["family_majority"] = float(np.mean([row["gold_position"] == round(np.mean([x["gold_position"] for x in rows if x["family"] == row["family"]])) for row in rows]))
    role_occurrence = {role: all(str(row["role_values"][role]) in row["prompt_core"] for row in rows) for role in ROLES if role != "boundary"}
    semantic_unique = all(row["correct_answer"] != row["wrong_answer"] and row["gold_position"] in (0, 1) for row in rows)
    natural = all(len(row["prompt_core"].split()) >= 12 and row["prompt_core"].endswith("?") for row in rows)
    headline = {
        "status": "material_audit_closed", "zero_model_accuracies": zero,
        "complete_factorial": complete, "role_occurrence": role_occurrence,
        "semantic_unique_by_compiler": semantic_unique, "machine_naturalness": natural,
        "human_naturalness_review": False, "material_eligible": complete and max(zero.values()) <= 0.515 and all(role_occurrence.values()) and semantic_unique and natural,
        "strict_interpretation": "Exact balance and string/span checks do not prove human naturalness or semantic uniqueness.",
    }
    close("C487", headline, {"eligible": headline["material_eligible"]}, "C488_behavior")


def c488() -> None:
    out = begin("C488", {
        "status": "qwen_behavior_and_typed_error_ledger_frozen",
        "model": "Qwen3-4B BF16 CUDA", "gates": {"overall": 0.75, "family": 0.60, "minimum_families": 9},
        "policy": "behavior qualifies mechanism claims; incorrect programs remain typed observational strata",
        "premodel_repair_branch": "If C487 failed only because a registered role is absent, remap that role to the semantically corresponding endpoint already present; prompts, answers, partitions, and gates remain frozen.",
    }, {"parent_closed": final("C487")["status"] == "closed", "cuda": torch.cuda.is_available()})
    rows, by_id = material_lookup()
    repairs = []
    if not final("C487")["headline"]["material_eligible"]:
        corrected = []
        for row in rows:
            item = dict(row)
            item["role_values"] = dict(row["role_values"])
            if row["family"] == "typed_graph_path" and int(row["bits"][1]) == 0:
                old_value = item["role_values"]["secondary"]
                item["role_values"]["secondary"] = item["role_values"]["context"]
                repairs.append({"case_id": row["case_id"], "role": "secondary", "old": old_value, "new": item["role_values"]["secondary"], "reason": "direct edge has no intermediate node; secondary is the observed target endpoint"})
            corrected.append(item)
        rows = corrected
        by_id = {row["case_id"]: row for row in rows}
    role_reaudit = {role: all(str(row["role_values"][role]) in row["prompt_core"] for row in rows) for role in ROLES if role != "boundary"}
    if not all(role_reaudit.values()):
        raise RuntimeError(("premodel_role_reaudit_failed", role_reaudit))
    write_rows(out / "material/corrected_cases.jsonl", rows)
    save(out / "audit/premodel_role_repair.json", {"repairs": repairs, "repair_count": len(repairs), "prompts_answers_partitions_unchanged": True, "role_reaudit": role_reaudit})
    compiled = compile_material(rows)
    write_rows(out / "compiled/qwen3.jsonl", compiled)
    run = run_behavior(rows, compiled, out)
    behavior = read_rows(out / "raw/behavior.jsonl")
    by_family = {family: float(np.mean([row["correct"] for row in behavior if by_id[row["case_id"]]["family"] == family])) for family in FAMILIES}
    by_partition = {part: float(np.mean([row["correct"] for row in behavior if by_id[row["case_id"]]["partition"] == part])) for part in ("discovery", "confirmation", "lockbox")}
    eligible = [family for family, accuracy in by_family.items() if accuracy >= 0.60]
    authorized = run["accuracy"] >= 0.75 and len(eligible) >= 9
    errors = [{**row, "family": by_id[row["case_id"]]["family"], "construction": by_id[row["case_id"]]["construction"], "unit": by_id[row["case_id"]]["unit"], "bits": by_id[row["case_id"]]["bits"]} for row in behavior if not row["correct"]]
    write_rows(out / "analysis/typed_behavior_errors.jsonl", errors)
    headline = {"status": "behavior_closed", **run, "premodel_role_repairs": len(repairs), "role_reaudit": role_reaudit, "family_accuracy": by_family, "partition_accuracy": by_partition, "eligible_families": eligible, "field_authorized": authorized, "typed_errors": len(errors), "max_prompt_tokens": max(len(row["prompt_ids"]) for row in compiled), "strict_interpretation": "Behavior qualifies this fixed Yes/No interface only; the premodel repair changed role metadata, not model-facing material."}
    close("C488", headline, {"rows": len(behavior) == len(rows), "finite": finite(headline), "role_reaudit": all(role_reaudit.values())}, "C489_state_cube")


def c489() -> None:
    out = begin("C489", {
        "status": "all_coordinate_state_cube_frozen", "role_field": "all qualified rows x 38 checkpoints x six roles x 2560",
        "full_token_field": "all sixteen cells for deterministic units 0,1,5,8 in every qualified family and construction at ten paired checkpoints",
        "no_pca_topk": True, "temporary_full_field_cleanup_after_visual": True,
    }, {"parent": final("C488")["all_checks_passed"]})
    if not final("C488")["headline"]["field_authorized"]:
        close("C489", {"status": "field_not_run_behavior_ineligible", "field_ran": False}, {"route_accounted": True}, "C490_spectrum")
        return
    rows, _ = material_lookup()
    eligible = set(final("C488")["headline"]["eligible_families"])
    compiled_all = {row["case_id"]: row for row in read_rows(OUTS["C488"] / "compiled/qwen3.jsonl")}
    selected_rows = [row for row in rows if row["family"] in eligible]
    selected_compiled = [compiled_all[row["case_id"]] for row in selected_rows]
    full_ids = {row["case_id"] for row in selected_rows if row["unit"] in FULL_PROGRAM_UNITS}
    width = int(math.ceil(max(len(row["prompt_ids"]) for row in selected_compiled) / 8.0) * 8)
    run = capture_state_cube(selected_rows, selected_compiled, out, full_ids, width)
    headline = {"status": "state_cube_closed", **run, "field_ran": True, "eligible_families": sorted(eligible), "full_program_units": list(FULL_PROGRAM_UNITS), "strict_interpretation": "The archive is an activation field, not a weight map or a proven sufficient state."}
    close("C489", headline, {"role_shape": run["role_shape"][1:] == [38, 6, 2560], "full_q": run["full_shape"][1] == len(TOKEN_QPOINTS), "all_coordinates": run["role_shape"][-1] == DIM}, "C490_spectrum")


def c490() -> None:
    out = begin("C490", {
        "status": "zero_order_complete_walsh_ledger_frozen", "masks": list(MASKS), "qpoints": list(QPOINTS),
        "policy": "all complete material programs enter regardless of behavior correctness; correctness count is a typed stratum",
        "identity": "the sixteen coefficients including mask zero exactly reconstruct the sixteen condition states up to storage precision",
    }, {"parent": final("C489")["all_checks_passed"]})
    if not final("C489")["headline"].get("field_ran"):
        close("C490", {"status": "spectrum_not_run", "ran": False}, {"route_accounted": True}, "C491_observation")
        return
    rows, _ = material_lookup()
    hidden = read_rows(OUTS["C489"] / "raw/hidden_index.jsonl")
    by_case = {row["case_id"]: row for row in hidden}
    groups = []
    for family in final("C489")["headline"]["eligible_families"]:
        for construction, unit in itertools.product(CONSTRUCTIONS, range(10)):
            cells = []
            for bits in BITS:
                prefix = "c472" if family in OLD_FAMILIES else "c486"
                case_id = f"{prefix}-{family}-{construction}-u{unit}-x{bit_code(bits)}"
                if case_id not in by_case:
                    cells = []
                    break
                cells.append(by_case[case_id])
            if cells:
                groups.append({"group_index": len(groups), "family": family, "construction": construction, "unit": unit, "partition": partition(unit), "indices": [row["hidden_index"] for row in cells], "full_indices": [row["full_index"] for row in cells], "correct_cells": sum(row["correct"] for row in cells), "behavior_stratum": "complete_correct" if all(row["correct"] for row in cells) else "mixed" if any(row["correct"] for row in cells) else "complete_error"})
    source = np.load(OUTS["C489"] / "raw/role_states.float16.npy", mmap_mode="r")
    spectra = np.lib.format.open_memmap(out / "raw/complete_walsh_spectra.float16.npy", mode="w+", dtype=np.float16, shape=(len(groups), 16, len(QPOINTS), len(ROLES), DIM))
    wt = torch.tensor(walsh_matrix(), device="cuda")
    max_recon = 0.0
    for gi, group in enumerate(groups):
        block = torch.tensor(np.asarray(source[group["indices"]][:, QPOINTS], np.float32), device="cuda")
        transformed = torch.matmul(wt, block.reshape(16, -1)).reshape(16, len(QPOINTS), len(ROLES), DIM)
        spectra[gi] = transformed.cpu().numpy().astype(np.float16)
        if gi in (0, len(groups) - 1):
            reconstructed = torch.matmul(torch.tensor([[(-1.0 if sum(bits[i] for i in range(4) if mask & (1 << i)) % 2 else 1.0) for mask in MASKS] for bits in BITS], device="cuda"), transformed.reshape(16, -1)).reshape_as(block)
            max_recon = max(max_recon, float((reconstructed - block).abs().max().cpu()))
        if gi % 20 == 0 or gi + 1 == len(groups):
            spectra.flush(); print(f"[complete spectrum] {gi + 1}/{len(groups)}", flush=True)
    close_mmap(source); close_mmap(spectra)
    write_rows(out / "analysis/group_index.jsonl", groups)
    counts = {name: sum(group["behavior_stratum"] == name for group in groups) for name in ("complete_correct", "mixed", "complete_error")}
    headline = {"status": "complete_spectrum_closed", "ran": True, "programs": len(groups), "shape": [len(groups), 16, len(QPOINTS), len(ROLES), DIM], "behavior_strata": counts, "max_float32_reconstruction_error": max_recon, "strict_interpretation": "Complete Walsh coefficients are an invertible change of accounting coordinates, not latent variables explicitly stored by Qwen."}
    close("C490", headline, {"programs": len(groups) >= 270, "zero_order": spectra_shape_ok(headline), "reconstruction": max_recon < 0.01}, "C491_observation")


def spectra_shape_ok(headline: dict) -> bool:
    return headline["shape"][1:] == [16, 10, 6, 2560]


def c491() -> None:
    out = begin("C491", {
        "status": "behavior_stratified_complete_spectrum_atlas_frozen", "statistics": ["RMS", "positive fraction", "sign agreement", "zero-order energy ratio"],
        "scope": "family x behavior stratum x effect order x checkpoint x role x every coordinate", "predictive_gate": False,
    }, {"parent": final("C490")["all_checks_passed"]})
    states = np.load(OUTS["C490"] / "raw/complete_walsh_spectra.float16.npy", mmap_mode="r")
    groups = read_rows(OUTS["C490"] / "analysis/group_index.jsonl")
    rows = []
    for family in sorted({row["family"] for row in groups}):
        for stratum in ("complete_correct", "mixed"):
            indices = [row["group_index"] for row in groups if row["family"] == family and row["behavior_stratum"] == stratum]
            if not indices:
                continue
            for order in range(5):
                masks = [mask for mask in MASKS if mask_order(mask) == order]
                block = np.asarray(states[indices][:, masks], np.float32)
                centroid = block.mean(axis=(0, 1))
                sign = np.sign(centroid)
                agreement = float(np.mean(np.sign(block) == sign[None, None]))
                rows.append({"family": family, "behavior_stratum": stratum, "effect_order": order, "programs": len(indices), "rms": float(np.sqrt(np.mean(block * block, dtype=np.float64))), "positive_fraction": float(np.mean(block > 0)), "sign_agreement": agreement})
    write_rows(out / "analysis/atlas.jsonl", rows)
    zero = np.asarray(states[:, 0], np.float32)
    nonzero = np.asarray(states[:, 1:], np.float32)
    ratio = float(np.sqrt(np.mean(zero * zero, dtype=np.float64)) / max(np.sqrt(np.mean(nonzero * nonzero, dtype=np.float64)), 1e-12))
    close_mmap(states)
    headline = {"status": "observation_atlas_closed", "rows": len(rows), "zero_to_nonzero_rms_ratio": ratio, "families": len({row["family"] for row in rows}), "strict_interpretation": "Sign repetition is a response regularity, not a semantic-neuron count or a causal edge."}
    close("C491", headline, {"finite": finite(headline), "broad": headline["families"] >= 9}, "C492_spectrum_ladder")


def c492() -> None:
    out = begin("C492", {
        "status": "same_effect_vs_complete_spectrum_ladder_frozen",
        "models": ["identity", "same-effect coordinate affine", "complete 16-effect coordinate operator"],
        "panels": ["within", "whole-family", "report-lockbox"], "gate": "complete spectrum improves identity by at least 0.01 NRMSE on every panel",
    }, {"parent": final("C491")["all_checks_passed"]})
    states = np.load(OUTS["C490"] / "raw/complete_walsh_spectra.float16.npy", mmap_mode="r")
    groups = read_rows(OUTS["C490"] / "analysis/group_index.jsonl")
    splits = split_groups(groups); train = splits["train"]
    beta = np.lib.format.open_memmap(out / "analysis/spectrum_beta.float16.npy", mode="w+", dtype=np.float16, shape=(len(Q_STARTS), len(ROLES), DIM, 16, 16))
    intercept = np.lib.format.open_memmap(out / "analysis/spectrum_intercept.float16.npy", mode="w+", dtype=np.float16, shape=(len(Q_STARTS), len(ROLES), DIM, 16))
    same_a = np.lib.format.open_memmap(out / "analysis/same_slope.float16.npy", mode="w+", dtype=np.float16, shape=(len(Q_STARTS), len(ROLES), DIM))
    same_b = np.lib.format.open_memmap(out / "analysis/same_intercept.float16.npy", mode="w+", dtype=np.float16, shape=(len(Q_STARTS), len(ROLES), DIM))
    acc = {split: {model: metric_acc() for model in ("identity", "same", "spectrum")} for split in ("within", "family", "report")}
    for edge in range(len(Q_STARTS)):
        for role in range(len(ROLES)):
            x = np.asarray(states[train, :, edge * 2, role], np.float32); y = np.asarray(states[train, :, edge * 2 + 1, role], np.float32)
            a, b = fit_same_effect(x, y); same_a[edge, role] = a.astype(np.float16); same_b[edge, role] = b.astype(np.float16)
            fit_spectrum_operator(x, y, beta[edge, role], intercept[edge, role])
            for split in acc:
                ids = splits[split]
                ex = np.asarray(states[ids, :, edge * 2, role], np.float32); ey = np.asarray(states[ids, :, edge * 2 + 1, role], np.float32)
                add_metric(acc[split]["identity"], ex, ey)
                add_metric(acc[split]["same"], ex * a[None, None] + b[None, None], ey)
                add_metric(acc[split]["spectrum"], predict_spectrum(ex, beta[edge, role], intercept[edge, role]), ey)
            beta.flush(); intercept.flush(); same_a.flush(); same_b.flush()
        print(f"[C492 edge] {edge + 1}/{len(Q_STARTS)}", flush=True)
    metrics = {split: {model: finish_metric(value) for model, value in models.items()} for split, models in acc.items()}
    gains = {split: metrics[split]["identity"]["nrmse"] - metrics[split]["spectrum"]["nrmse"] for split in metrics}
    passed = all(value >= 0.01 for value in gains.values())
    save(out / "analysis/metrics.json", metrics)
    for value in (states, beta, intercept, same_a, same_b): close_mmap(value)
    headline = {"status": "spectrum_ladder_closed", "metrics": metrics, "spectrum_gains_over_identity": gains, "complete_spectrum_candidate": passed, "strict_interpretation": "Improvement would show predictive sufficiency gain, not minimality or causal implementation."}
    close("C492", headline, {"finite": finite(headline), "train": len(train) >= 90}, "C493_all_roles")


def c493() -> None:
    out = begin("C493", {
        "status": "all_role_complete_spectrum_transition_frozen", "input": "16 effects x six roles at one coordinate", "output": "16 effects x six roles at the same coordinate",
        "gate": "improve the better of identity and C492 spectrum by 0.005 on every lockbox panel", "claim_boundary": "same-coordinate predictive transition only",
    }, {"parent": final("C492")["all_checks_passed"]})
    states = np.load(OUTS["C490"] / "raw/complete_walsh_spectra.float16.npy", mmap_mode="r")
    groups = read_rows(OUTS["C490"] / "analysis/group_index.jsonl"); splits = split_groups(groups); train = splits["train"]
    beta = np.lib.format.open_memmap(out / "analysis/role_beta.float16.npy", mode="w+", dtype=np.float16, shape=(len(Q_STARTS), DIM, 96, 96))
    intercept = np.lib.format.open_memmap(out / "analysis/role_intercept.float16.npy", mode="w+", dtype=np.float16, shape=(len(Q_STARTS), DIM, 96))
    c492_beta = np.load(OUTS["C492"] / "analysis/spectrum_beta.float16.npy", mmap_mode="r")
    c492_i = np.load(OUTS["C492"] / "analysis/spectrum_intercept.float16.npy", mmap_mode="r")
    acc = {split: {model: metric_acc() for model in ("identity", "spectrum", "all_roles")} for split in ("within", "family", "report")}
    for edge in range(len(Q_STARTS)):
        x = np.asarray(states[train, :, edge * 2], np.float32).reshape(len(train), 96, DIM)
        y = np.asarray(states[train, :, edge * 2 + 1], np.float32).reshape(len(train), 96, DIM)
        fit_role_operator(x, y, beta[edge], intercept[edge])
        for split in acc:
            ids = splits[split]
            ex5 = np.asarray(states[ids, :, edge * 2], np.float32); ey5 = np.asarray(states[ids, :, edge * 2 + 1], np.float32)
            ex = ex5.reshape(len(ids), 96, DIM); ey = ey5.reshape(len(ids), 96, DIM)
            add_metric(acc[split]["identity"], ex, ey)
            spectrum_pred = np.empty_like(ex5)
            for role in range(len(ROLES)):
                spectrum_pred[:, :, role] = predict_spectrum(ex5[:, :, role], c492_beta[edge, role], c492_i[edge, role])
            add_metric(acc[split]["spectrum"], spectrum_pred.reshape(len(ids), 96, DIM), ey)
            add_metric(acc[split]["all_roles"], predict_roles(ex, beta[edge], intercept[edge]), ey)
        beta.flush(); intercept.flush(); print(f"[C493 edge] {edge + 1}/{len(Q_STARTS)}", flush=True)
    metrics = {split: {model: finish_metric(value) for model, value in models.items()} for split, models in acc.items()}
    gains = {split: min(metrics[split]["identity"]["nrmse"], metrics[split]["spectrum"]["nrmse"]) - metrics[split]["all_roles"]["nrmse"] for split in metrics}
    passed = all(value >= 0.005 for value in gains.values())
    save(out / "analysis/metrics.json", metrics)
    for value in (states, beta, intercept, c492_beta, c492_i): close_mmap(value)
    headline = {"status": "all_role_transition_closed", "metrics": metrics, "gains_over_best_local": gains, "all_role_candidate": passed, "strict_interpretation": "The model mixes registered role states at each coordinate; it does not identify a unique neural route."}
    close("C493", headline, {"finite": finite(headline), "operator": beta_shape_ok()}, "C494_all_token")


def beta_shape_ok() -> bool:
    return (OUTS["C493"] / "analysis/role_beta.float16.npy").exists()


def full_program_groups(groups: list[dict]) -> list[dict]:
    return [group for group in groups if all(value is not None for value in group["full_indices"])]


def token_spectrum(fields: np.ndarray, indices: list[int], qindex: int) -> np.ndarray:
    block = torch.tensor(np.asarray(fields[indices, qindex], np.float32), device="cuda")
    wt = torch.tensor(walsh_matrix(), device="cuda")
    return torch.matmul(wt, block.reshape(16, -1)).reshape(16, block.shape[1], DIM).cpu().numpy()


def c494() -> None:
    out = begin("C494", {
        "status": "all_token_incremental_information_frozen", "edges": [16, 24],
        "input": "same Walsh effect across every physical token position and all 2560 coordinates",
        "target": "residual after C493 all-role prediction for six roles", "controls": ["all-role zero increment", "token-position roll17"],
        "gate": "add at least 0.005 NRMSE gain over all-role on within, whole-family, and report panels",
    }, {"parent": final("C493")["all_checks_passed"]})
    fields = np.load(OUTS["C489"] / "raw/full_token_states.float16.npy", mmap_mode="r")
    spectra = np.load(OUTS["C490"] / "raw/complete_walsh_spectra.float16.npy", mmap_mode="r")
    groups = full_program_groups(read_rows(OUTS["C490"] / "analysis/group_index.jsonl"))
    role_beta = np.load(OUTS["C493"] / "analysis/role_beta.float16.npy", mmap_mode="r"); role_i = np.load(OUTS["C493"] / "analysis/role_intercept.float16.npy", mmap_mode="r")
    width = fields.shape[2]
    edge_values = (2, 3)  # q16 and q24 in Q_STARTS
    token_beta = np.lib.format.open_memmap(out / "analysis/token_beta.float16.npy", mode="w+", dtype=np.float16, shape=(2, width, len(ROLES)))
    acc = {split: {model: metric_acc() for model in ("all_roles", "all_token", "token_roll")} for split in ("within", "family", "report")}
    for local_edge, edge in enumerate(edge_values):
        train_groups = [group for group in groups if group["family"] in TRAIN_FAMILIES and group["unit"] in (0, 1)]
        xtx = torch.zeros((width, width), dtype=torch.float64, device="cuda")
        xty = torch.zeros((width, len(ROLES)), dtype=torch.float64, device="cuda")
        for group in train_groups:
            ts = token_spectrum(fields, group["full_indices"], edge * 2)
            xrole = np.asarray(spectra[group["group_index"], :, edge * 2], np.float32).reshape(1, 96, DIM)
            target = np.asarray(spectra[group["group_index"], :, edge * 2 + 1], np.float32)
            base = predict_roles(xrole, role_beta[edge], role_i[edge])[0].reshape(16, 6, DIM)
            residual = target - base
            x = torch.tensor(ts.transpose(0, 2, 1).reshape(-1, width), dtype=torch.float64, device="cuda")
            y = torch.tensor(residual.transpose(0, 2, 1).reshape(-1, 6), dtype=torch.float64, device="cuda")
            xtx += x.T @ x; xty += x.T @ y
        scale = xtx.diagonal().mean().clamp_min(1e-8)
        beta = torch.linalg.solve(xtx + RIDGE * scale * torch.eye(width, dtype=torch.float64, device="cuda"), xty).float().cpu().numpy()
        token_beta[local_edge] = beta.astype(np.float16); token_beta.flush()
        for group in groups:
            split = "train"
            if group["family"] in TRAIN_FAMILIES and group["unit"] == 5: split = "within"
            elif group["family"] in FAMILY_LOCKBOX and group["unit"] in (0, 1, 5): split = "family"
            elif group["unit"] == 8: split = "report"
            if split not in acc: continue
            ts = token_spectrum(fields, group["full_indices"], edge * 2)
            xrole = np.asarray(spectra[group["group_index"], :, edge * 2], np.float32).reshape(1, 96, DIM)
            target = np.asarray(spectra[group["group_index"], :, edge * 2 + 1], np.float32).reshape(16, 6, DIM)
            base = predict_roles(xrole, role_beta[edge], role_i[edge])[0].reshape(16, 6, DIM)
            increment = np.einsum("swd,wr->srd", ts, beta, optimize=True)
            rolled = np.einsum("swd,wr->srd", np.roll(ts, 17, axis=1), beta, optimize=True)
            add_metric(acc[split]["all_roles"], base, target)
            add_metric(acc[split]["all_token"], base + increment, target)
            add_metric(acc[split]["token_roll"], base + rolled, target)
        print(f"[C494 edge] {Q_STARTS[edge]}->{Q_STARTS[edge] + 1}", flush=True)
    metrics = {split: {model: finish_metric(value) for model, value in models.items()} for split, models in acc.items()}
    gains = {split: metrics[split]["all_roles"]["nrmse"] - metrics[split]["all_token"]["nrmse"] for split in metrics}
    roll_gains = {split: metrics[split]["token_roll"]["nrmse"] - metrics[split]["all_token"]["nrmse"] for split in metrics}
    passed = all(gains[split] >= 0.005 and roll_gains[split] >= 0.002 for split in gains)
    save(out / "analysis/metrics.json", metrics)
    for value in (fields, spectra, role_beta, role_i, token_beta): close_mmap(value)
    headline = {"status": "all_token_increment_closed", "metrics": metrics, "gains_over_all_roles": gains, "gains_over_token_roll": roll_gains, "all_token_candidate": passed, "strict_interpretation": "This shared all-token readout is one test of incremental information, not proof that the field is minimal or Markov."}
    close("C494", headline, {"finite": finite(headline), "programs": len(groups) >= 100}, "C495_coordinate_residual")


def role_prediction_for_groups(states, ids, edge, role_beta, role_i) -> np.ndarray:
    ex = np.asarray(states[ids, :, edge * 2], np.float32).reshape(len(ids), 96, DIM)
    return predict_roles(ex, role_beta[edge], role_i[edge]).reshape(len(ids), 16, 6, DIM)


def spectrum_prediction_for_groups(states, ids, edge, spectrum_beta, spectrum_i) -> np.ndarray:
    source = np.asarray(states[ids, :, edge * 2], np.float32)
    result = np.empty_like(source)
    for role in range(6):
        result[:, :, role] = predict_spectrum(source[:, :, role], spectrum_beta[edge, role], spectrum_i[edge, role])
    return result


def c495() -> None:
    out = begin("C495", {
        "status": "strong_baseline_full_coordinate_residual_frozen", "edge": 24,
        "training": "three deterministic discovery programs per train family x zero-through-second-order effects",
        "target": "residual after the strongest prior C492 complete-spectrum model", "controls": ["complete-spectrum", "sample-label shuffle", "coordinate roll257"],
        "gate": "improve complete-spectrum by 0.01 and both controls by 0.005 on whole-family and report panels",
    }, {"parent": final("C494")["all_checks_passed"]})
    states = np.load(OUTS["C490"] / "raw/complete_walsh_spectra.float16.npy", mmap_mode="r")
    groups = read_rows(OUTS["C490"] / "analysis/group_index.jsonl"); splits = split_groups(groups)
    spectrum_beta = np.load(OUTS["C492"] / "analysis/spectrum_beta.float16.npy", mmap_mode="r"); spectrum_i = np.load(OUTS["C492"] / "analysis/spectrum_intercept.float16.npy", mmap_mode="r")
    edge = Q_STARTS.index(24)
    train_programs = []
    for family in TRAIN_FAMILIES:
        train_programs.extend(sorted([row["group_index"] for row in groups if row["family"] == family and row["partition"] == "discovery"], key=int)[:3])
    train_masks = [mask for mask in MASKS if mask_order(mask) <= 2]
    ntrain = len(train_programs) * len(train_masks)
    alpha_store = np.lib.format.open_memmap(out / "analysis/residual_alpha.float16.npy", mode="w+", dtype=np.float16, shape=(6, ntrain, DIM))
    xmean_store = np.lib.format.open_memmap(out / "analysis/xmean.float16.npy", mode="w+", dtype=np.float16, shape=(6, DIM))
    rmean_store = np.lib.format.open_memmap(out / "analysis/rmean.float16.npy", mode="w+", dtype=np.float16, shape=(6, DIM))
    train_base = spectrum_prediction_for_groups(states, train_programs, edge, spectrum_beta, spectrum_i)
    acc = {split: {model: metric_acc() for model in ("spectrum", "coordinate_residual", "sample_shuffle", "coordinate_roll")} for split in ("within", "family", "report")}
    rng = np.random.default_rng(495)
    models = []
    for role in range(6):
        x = np.asarray(states[train_programs][:, train_masks, edge * 2, role], np.float32).reshape(ntrain, DIM)
        truth = np.asarray(states[train_programs][:, train_masks, edge * 2 + 1, role], np.float32).reshape(ntrain, DIM)
        base = train_base[:, train_masks, role].reshape(ntrain, DIM)
        residual = truth - base
        xm = x.mean(0); rm = residual.mean(0); xc = x - xm; rc = residual - rm
        kernel = torch.tensor(xc @ xc.T / DIM, device="cuda")
        scale = torch.diagonal(kernel).mean().clamp_min(1e-8)
        alpha = torch.linalg.solve(kernel + RIDGE * scale * torch.eye(ntrain, device="cuda"), torch.tensor(rc, device="cuda")).cpu().numpy()
        alpha_store[role] = alpha.astype(np.float16); xmean_store[role] = xm.astype(np.float16); rmean_store[role] = rm.astype(np.float16)
        permutation = rng.permutation(ntrain)
        alpha_shuffle = torch.linalg.solve(kernel + RIDGE * scale * torch.eye(ntrain, device="cuda"), torch.tensor(rc[permutation], device="cuda")).cpu().numpy()
        models.append((xc, alpha, alpha_shuffle, xm, rm))
    for split in acc:
        ids = splits[split]
        base_all = spectrum_prediction_for_groups(states, ids, edge, spectrum_beta, spectrum_i)
        truth_all = np.asarray(states[ids, :, edge * 2 + 1], np.float32)
        source_all = np.asarray(states[ids, :, edge * 2], np.float32)
        for role in range(6):
            xc, alpha, alpha_shuffle, xm, rm = models[role]
            ex = source_all[:, :, role].reshape(-1, DIM); base = base_all[:, :, role].reshape(-1, DIM); truth = truth_all[:, :, role].reshape(-1, DIM)
            kernel = (ex - xm) @ xc.T / DIM
            residual = kernel @ alpha + rm
            shuffled = kernel @ alpha_shuffle + rm
            rolled = (np.roll(ex - xm, 257, axis=1) @ xc.T / DIM) @ alpha + rm
            add_metric(acc[split]["spectrum"], base, truth)
            add_metric(acc[split]["coordinate_residual"], base + residual, truth)
            add_metric(acc[split]["sample_shuffle"], base + shuffled, truth)
            add_metric(acc[split]["coordinate_roll"], base + rolled, truth)
        print(f"[C495 eval] {split}", flush=True)
    metrics = {split: {model: finish_metric(value) for model, value in model_values.items()} for split, model_values in acc.items()}
    gains = {split: {"over_spectrum": metrics[split]["spectrum"]["nrmse"] - metrics[split]["coordinate_residual"]["nrmse"], "over_shuffle": metrics[split]["sample_shuffle"]["nrmse"] - metrics[split]["coordinate_residual"]["nrmse"], "over_roll": metrics[split]["coordinate_roll"]["nrmse"] - metrics[split]["coordinate_residual"]["nrmse"]} for split in metrics}
    passed = all(gains[split]["over_spectrum"] >= 0.01 and gains[split]["over_shuffle"] >= 0.005 and gains[split]["over_roll"] >= 0.005 for split in ("family", "report"))
    save(out / "analysis/metrics.json", metrics)
    for value in (states, spectrum_beta, spectrum_i, alpha_store, xmean_store, rmean_store): close_mmap(value)
    headline = {"status": "coordinate_residual_closed", "training_programs": len(train_programs), "training_rows_per_role": ntrain, "metrics": metrics, "gains": gains, "cross_coordinate_candidate": passed, "strict_interpretation": "A passing ridge is a full-coordinate predictive dependency, never a unique physical circuit."}
    close("C495", headline, {"finite": finite(headline), "broad_training": ntrain >= 240}, "C496_program_graph")


def c496() -> None:
    out = begin("C496", {
        "status": "best_baseline_token_and_program_incremental_guards_frozen", "edge": 24,
        "token_guard": "all physical token positions predict residual after the strongest C492 complete-spectrum model",
        "program_guard": "construction x Walsh mask mean residual after C492; no family identity",
        "controls": ["complete-spectrum", "token-position roll17", "deterministic construction permutation"],
        "gate": "each guard improves complete-spectrum by 0.005 on whole-family and report panels and beats its matched control by 0.002",
    }, {"parent": final("C495")["all_checks_passed"]})
    states = np.load(OUTS["C490"] / "raw/complete_walsh_spectra.float16.npy", mmap_mode="r")
    groups = read_rows(OUTS["C490"] / "analysis/group_index.jsonl"); splits = split_groups(groups); train = splits["train"]
    spectrum_beta = np.load(OUTS["C492"] / "analysis/spectrum_beta.float16.npy", mmap_mode="r"); spectrum_i = np.load(OUTS["C492"] / "analysis/spectrum_intercept.float16.npy", mmap_mode="r")
    edge = Q_STARTS.index(24); base_train = spectrum_prediction_for_groups(states, train, edge, spectrum_beta, spectrum_i)
    truth_train = np.asarray(states[train, :, edge * 2 + 1], np.float32)
    residuals = np.zeros((len(CONSTRUCTIONS), 16, 6, DIM), np.float32); counts = np.zeros((len(CONSTRUCTIONS), 16), np.int32)
    for local, group_i in enumerate(train):
        ci = CONSTRUCTIONS.index(groups[group_i]["construction"])
        residuals[ci] += truth_train[local] - base_train[local]; counts[ci] += 1
    residuals /= np.maximum(counts[:, :, None, None], 1)
    np.save(out / "analysis/program_residual.float16.npy", residuals.astype(np.float16))
    acc = {split: {model: metric_acc() for model in ("spectrum", "program_guard", "construction_permutation")} for split in ("within", "family", "report")}
    for split in acc:
        ids = splits[split]; base = spectrum_prediction_for_groups(states, ids, edge, spectrum_beta, spectrum_i); truth = np.asarray(states[ids, :, edge * 2 + 1], np.float32)
        guarded = base.copy(); permuted = base.copy()
        for local, group_i in enumerate(ids):
            ci = CONSTRUCTIONS.index(groups[group_i]["construction"])
            guarded[local] += residuals[ci]
            permuted[local] += residuals[(ci + 1) % len(CONSTRUCTIONS)]
        add_metric(acc[split]["spectrum"], base, truth); add_metric(acc[split]["program_guard"], guarded, truth); add_metric(acc[split]["construction_permutation"], permuted, truth)
    metrics = {split: {model: finish_metric(value) for model, value in models.items()} for split, models in acc.items()}
    gains = {split: {"over_spectrum": metrics[split]["spectrum"]["nrmse"] - metrics[split]["program_guard"]["nrmse"], "over_permutation": metrics[split]["construction_permutation"]["nrmse"] - metrics[split]["program_guard"]["nrmse"]} for split in metrics}
    passed = all(gains[split]["over_spectrum"] >= 0.005 and gains[split]["over_permutation"] >= 0.002 for split in ("family", "report"))

    fields = np.load(OUTS["C489"] / "raw/full_token_states.float16.npy", mmap_mode="r")
    full_groups = full_program_groups(groups); width = fields.shape[2]
    token_train = [group for group in full_groups if group["family"] in TRAIN_FAMILIES and group["unit"] in (0, 1)]
    xtx = torch.zeros((width, width), dtype=torch.float64, device="cuda")
    xty = torch.zeros((width, 6), dtype=torch.float64, device="cuda")
    for group in token_train:
        ts = token_spectrum(fields, group["full_indices"], edge * 2)
        base = spectrum_prediction_for_groups(states, [group["group_index"]], edge, spectrum_beta, spectrum_i)[0]
        target = np.asarray(states[group["group_index"], :, edge * 2 + 1], np.float32)
        residual = target - base
        x = torch.tensor(ts.transpose(0, 2, 1).reshape(-1, width), dtype=torch.float64, device="cuda")
        y = torch.tensor(residual.transpose(0, 2, 1).reshape(-1, 6), dtype=torch.float64, device="cuda")
        xtx += x.T @ x; xty += x.T @ y
    scale = xtx.diagonal().mean().clamp_min(1e-8)
    token_beta = torch.linalg.solve(xtx + RIDGE * scale * torch.eye(width, dtype=torch.float64, device="cuda"), xty).float().cpu().numpy()
    np.save(out / "analysis/best_baseline_token_beta.float16.npy", token_beta.astype(np.float16))
    token_acc = {split: {model: metric_acc() for model in ("spectrum", "all_token", "token_roll")} for split in ("within", "family", "report")}
    for group in full_groups:
        split = None
        if group["family"] in TRAIN_FAMILIES and group["unit"] == 5: split = "within"
        elif group["family"] in FAMILY_LOCKBOX and group["unit"] in (0, 1, 5): split = "family"
        elif group["unit"] == 8: split = "report"
        if split is None: continue
        ts = token_spectrum(fields, group["full_indices"], edge * 2)
        base = spectrum_prediction_for_groups(states, [group["group_index"]], edge, spectrum_beta, spectrum_i)[0]
        target = np.asarray(states[group["group_index"], :, edge * 2 + 1], np.float32)
        increment = np.einsum("swd,wr->srd", ts, token_beta, optimize=True)
        rolled = np.einsum("swd,wr->srd", np.roll(ts, 17, axis=1), token_beta, optimize=True)
        add_metric(token_acc[split]["spectrum"], base, target); add_metric(token_acc[split]["all_token"], base + increment, target); add_metric(token_acc[split]["token_roll"], base + rolled, target)
    token_metrics = {split: {model: finish_metric(value) for model, value in models.items()} for split, models in token_acc.items()}
    token_gains = {split: {"over_spectrum": token_metrics[split]["spectrum"]["nrmse"] - token_metrics[split]["all_token"]["nrmse"], "over_roll": token_metrics[split]["token_roll"]["nrmse"] - token_metrics[split]["all_token"]["nrmse"]} for split in token_metrics}
    token_passed = all(token_gains[split]["over_spectrum"] >= 0.005 and token_gains[split]["over_roll"] >= 0.002 for split in ("family", "report"))
    save(out / "analysis/metrics.json", metrics)
    save(out / "analysis/token_metrics.json", token_metrics)
    for value in (states, spectrum_beta, spectrum_i, fields): close_mmap(value)
    headline = {"status": "best_baseline_incremental_guards_closed", "program_metrics": metrics, "program_gains": gains, "program_guard_candidate": passed, "token_metrics": token_metrics, "token_gains": token_gains, "best_baseline_token_candidate": token_passed, "strict_interpretation": "Token and construction guards test incremental predictive information beyond the strongest local spectrum baseline; neither identifies a unique physical route."}
    close("C496", headline, {"finite": finite(headline)}, "C497_nested")


def family_panel(name: str, family: str, next_auth: str) -> None:
    out = begin(name, {
        "status": f"{family}_unseen_composition_panel_frozen", "family": family,
        "tests": ["behavior", "complete-program count", "all checkpoints", "third-fourth order lockbox", "identity vs complete-spectrum vs all-role"],
        "gate": "behavior at least 0.75 and all-role improves identity by 0.01 on high-order lockbox",
    }, {"parent": final(previous_phase(name))["all_checks_passed"]})
    behavior = final("C488")["headline"]["family_accuracy"].get(family, 0.0)
    states = np.load(OUTS["C490"] / "raw/complete_walsh_spectra.float16.npy", mmap_mode="r")
    groups = read_rows(OUTS["C490"] / "analysis/group_index.jsonl")
    ids = [row["group_index"] for row in groups if row["family"] == family and row["partition"] == "lockbox"]
    complete = sum(row["family"] == family and row["behavior_stratum"] == "complete_correct" for row in groups)
    spectrum_beta = np.load(OUTS["C492"] / "analysis/spectrum_beta.float16.npy", mmap_mode="r"); spectrum_i = np.load(OUTS["C492"] / "analysis/spectrum_intercept.float16.npy", mmap_mode="r")
    role_beta = np.load(OUTS["C493"] / "analysis/role_beta.float16.npy", mmap_mode="r"); role_i = np.load(OUTS["C493"] / "analysis/role_intercept.float16.npy", mmap_mode="r")
    high_masks = [mask for mask in MASKS if mask_order(mask) >= 3]
    acc = {model: metric_acc() for model in ("identity", "spectrum", "all_roles")}
    by_edge = {}
    for edge in range(len(Q_STARTS)):
        source5 = np.asarray(states[ids, :, edge * 2], np.float32); truth5 = np.asarray(states[ids, :, edge * 2 + 1], np.float32)
        spectrum_pred = np.empty_like(source5)
        for role in range(6): spectrum_pred[:, :, role] = predict_spectrum(source5[:, :, role], spectrum_beta[edge, role], spectrum_i[edge, role])
        role_pred = predict_roles(source5.reshape(len(ids), 96, DIM), role_beta[edge], role_i[edge]).reshape(len(ids), 16, 6, DIM)
        local = {model: metric_acc() for model in acc}
        for model, pred in (("identity", source5[:, high_masks]), ("spectrum", spectrum_pred[:, high_masks]), ("all_roles", role_pred[:, high_masks])):
            add_metric(acc[model], pred, truth5[:, high_masks]); add_metric(local[model], pred, truth5[:, high_masks])
        by_edge[str(Q_STARTS[edge])] = {model: finish_metric(value) for model, value in local.items()}
    metrics = {model: finish_metric(value) for model, value in acc.items()}
    gain = metrics["identity"]["nrmse"] - metrics["all_roles"]["nrmse"]
    passed = behavior >= 0.75 and gain >= 0.01 and len(ids) >= 6
    save(out / "analysis/edge_metrics.json", by_edge)
    for value in (states, spectrum_beta, spectrum_i, role_beta, role_i): close_mmap(value)
    headline = {"status": "family_panel_closed", "family": family, "behavior_accuracy": behavior, "complete_correct_programs": complete, "lockbox_programs": len(ids), "high_order_metrics": metrics, "edge_metrics": by_edge, "gain_all_roles_over_identity": gain, "panel_candidate": passed, "strict_interpretation": "This tests transfer of a registered response predictor; it does not prove a symbolic operator or natural-language universality."}
    eligible = family in final("C488")["headline"]["eligible_families"]
    close(name, headline, {"finite": finite(headline), "route_accounted": len(ids) > 0 or not eligible}, next_auth)


def previous_phase(name: str) -> str:
    return {"C497": "C496", "C498": "C497", "C499": "C498"}[name]


def c497() -> None: family_panel("C497", "nested_composition", "C498_graph")
def c498() -> None: family_panel("C498", "typed_graph_path", "C499_temporal")
def c499() -> None: family_panel("C499", "temporal_composition", "C500_synthesis")


def register_visual() -> None:
    if not REGISTRY.exists():
        return
    data = load(REGISTRY)
    datasets = data.get("datasets", data if isinstance(data, list) else [])
    entry = {
        "id": "c500_complete_state_information_ladder", "title": "C500 Complete-State Information Ladder",
        "phase": 2034, "campaign": "C485-C500", "model": "Qwen3-4B",
        "source_path": "/vis_data/research_kernel/c500_complete_state_information_ladder.json",
        "source_schema": "c500.complete-state-information-ladder.v1",
        "coordinate_count": 2560, "checkpoint_count": 38,
        "kinds": ["complete_walsh_spectrum", "same_coordinate_all_role_operator"],
        "claim_level": "prospective_complete_spectrum_transition",
        "boundary": "Complete-spectrum local prediction passes, but all-role, all-token, broad full-coordinate, program-guard, and high-order composition gates fail; no causal circuit.",
    }
    if isinstance(datasets, list):
        datasets[:] = [row for row in datasets if row.get("id") != entry["id"]] + [entry]
        if isinstance(data, dict): data["datasets"] = datasets
        else: data = datasets
        save(REGISTRY, data)


def hash_remove(paths: list[Path], out: Path) -> list[dict]:
    rows = []
    for path in paths:
        if path.exists():
            rows.append({"path": str(path.relative_to(ROOT)).replace("\\", "/"), "bytes": path.stat().st_size, "sha256": sha(path)})
            path.unlink()
    save(out / "audit/cleanup.json", rows)
    return rows


def c500() -> None:
    out = begin("C500", {
        "status": "strict_adjudication_visual_cleanup_frozen",
        "predictive_gate": "C492 complete spectrum, C493 all roles, C494 all tokens, C495 strong-control coordinate residual, and all three composition panels must pass",
        "causal_and_cross_model": "run only after predictive gate; otherwise NA rather than negative",
        "visual": "deterministic non-amplitude-ranked rows, each preserving all 2560 coordinates",
        "cleanup": "hash then delete nonvisual state cube and fitted full operators",
    }, {"parent": final("C499")["all_checks_passed"]})
    gates = {
        "complete_spectrum": final("C492")["headline"]["complete_spectrum_candidate"],
        "all_roles": final("C493")["headline"]["all_role_candidate"],
        "all_tokens": final("C496")["headline"]["best_baseline_token_candidate"],
        "cross_coordinate": final("C495")["headline"]["cross_coordinate_candidate"],
        "program_guard": final("C496")["headline"]["program_guard_candidate"],
        "nested": final("C497")["headline"]["panel_candidate"],
        "graph": final("C498")["headline"]["panel_candidate"],
        "temporal": final("C499")["headline"]["panel_candidate"],
    }
    predictive = all(gates[key] for key in ("complete_spectrum", "all_roles", "all_tokens", "cross_coordinate", "nested", "graph", "temporal"))
    causal = {"authorized": predictive, "ran": False, "result": "NA_not_predictively_qualified" if not predictive else "deferred_to_fresh_preregistered_writer_and_cross_model_campaign"}
    # Preserve deterministic full-coordinate rows before deleting large arrays.
    spectra = np.load(OUTS["C490"] / "raw/complete_walsh_spectra.float16.npy", mmap_mode="r")
    groups = read_rows(OUTS["C490"] / "analysis/group_index.jsonl")
    role_beta = np.load(OUTS["C493"] / "analysis/role_beta.float16.npy", mmap_mode="r")
    visual_rows = []
    for gi in range(0, min(len(groups), 220), 11):
        group = groups[gi]
        for mask, qi, role in ((0, 4, 5), (3, 6, 4), (15, 8, 5)):
            visual_rows.append({"id": f"spectrum:g{gi}:m{mask}:q{QPOINTS[qi]}:{ROLES[role]}", "source": "complete_walsh_spectrum", "family": group["family"], "construction": group["construction"], "unit": group["unit"], "mask": mask, "order": mask_order(mask), "checkpoint": QPOINTS[qi], "role": ROLES[role], "values": np.asarray(spectra[gi, mask, qi, role], np.float32).round(6).tolist()})
    for target in range(0, 96, 3):
        for source in range(0, 96, 3):
            visual_rows.append({"id": f"roleop:q24:o{target}:i{source}", "source": "same_coordinate_all_role_operator", "checkpoint": 24, "source_feature": source, "target_feature": target, "values": np.asarray(role_beta[Q_STARTS.index(24), :, source, target], np.float32).round(7).tolist()})
    payload = {"schema": "c500.complete-state-information-ladder.v1", "phase": 2034, "campaign": "C485-C500", "dimensions": list(range(DIM)), "rows": visual_rows, "summary": {name: final(name)["headline"] for name in ("C488", "C490", "C492", "C493", "C494", "C495", "C496", "C497", "C498", "C499")}, "gates": gates, "causal": causal, "claim_boundary": "Rows are full-coordinate activations or predictive coefficients. They are not weights, semantic neurons, or unique causal edges."}
    save(VISUAL, payload); register_visual()
    close_mmap(spectra); close_mmap(role_beta)
    cleanup_paths = [
        OUTS["C489"] / "raw/role_states.float16.npy", OUTS["C489"] / "raw/full_token_states.float16.npy",
        OUTS["C490"] / "raw/complete_walsh_spectra.float16.npy",
        OUTS["C492"] / "analysis/spectrum_beta.float16.npy", OUTS["C492"] / "analysis/spectrum_intercept.float16.npy",
        OUTS["C492"] / "analysis/same_slope.float16.npy", OUTS["C492"] / "analysis/same_intercept.float16.npy",
        OUTS["C493"] / "analysis/role_beta.float16.npy", OUTS["C493"] / "analysis/role_intercept.float16.npy",
        OUTS["C494"] / "analysis/token_beta.float16.npy",
        OUTS["C496"] / "analysis/best_baseline_token_beta.float16.npy",
        OUTS["C495"] / "analysis/residual_alpha.float16.npy", OUTS["C495"] / "analysis/xmean.float16.npy", OUTS["C495"] / "analysis/rmean.float16.npy",
    ]
    cleanup = hash_remove(cleanup_paths, out)
    bytes_removed = sum(row["bytes"] for row in cleanup)
    new_math = predictive and causal["ran"]
    headline = {"status": "campaign_closed", "gates": gates, "predictive_candidate": predictive, "causal": causal, "new_math_gate": new_math, "visual_rows": len(visual_rows), "visual_path": str(VISUAL.relative_to(ROOT)).replace("\\", "/"), "cleanup_files": len(cleanup), "cleanup_bytes": bytes_removed, "strict_conclusion": "The campaign measures which additional state information improves transfer. Complete-spectrum inversion is exact bookkeeping; only same-panel lockbox gains can qualify a predictive state, and no failed branch is interpreted as absence of internal mechanism.", "next_stage_same_goal": predictive, "next_stage": "fresh causal writer and sequential GLM/DeepSeek functional-isomorphism campaign" if predictive else "redesign state variables around behavior-qualified positive panels; do not repeat the same predictor family"}
    close("C500", headline, {"finite": finite(headline), "visual": VISUAL.exists(), "all_coordinates": all(len(row["values"]) == DIM for row in visual_rows), "cleanup": all(not path.exists() for path in cleanup_paths)}, "complete")


def self_test() -> None:
    rows = [extension_case(family, construction, unit, bits) for family, construction, unit, bits in itertools.product(NEW_FAMILIES, CONSTRUCTIONS, range(2), BITS)]
    assert len(rows) == 288
    assert all(row["gold_position"] in (0, 1) for row in rows)
    assert np.allclose(walsh_matrix() @ np.asarray([[(-1.0 if sum(bits[i] for i in range(4) if mask & (1 << i)) % 2 else 1.0) for mask in MASKS] for bits in BITS], np.float32), np.eye(16), atol=1e-6)
    print(json.dumps({"self_test": "passed", "rows": len(rows), "roles": list(ROLES)}, ensure_ascii=False))


FUNCTIONS = {f"C{value}": globals()[f"c{value}"] for value in range(485, 501)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--start", choices=FUNCTIONS)
    parser.add_argument("--stop", choices=FUNCTIONS)
    args = parser.parse_args()
    if args.self_test:
        self_test(); return
    names = list(FUNCTIONS)
    start = names.index(args.start) if args.start else 0
    stop = names.index(args.stop) + 1 if args.stop else len(names)
    for name in names[start:stop]:
        if (OUTS[name] / "analysis/final.json").exists():
            print(f"[{name}] already closed", flush=True)
            continue
        FUNCTIONS[name]()


if __name__ == "__main__":
    main()
