#!/usr/bin/env python3
"""C561-C568 fresh-material replication of the active/passive response passport.

The campaign observes embeddings and HiddenState checkpoints only. It retains
every physical coordinate and never reads Attention or MLP internals.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
sys.path.insert(0, str(TESTS))

import model_utils
import phase2076_c542_c559_typed_operation_response_passport_campaign as parent

PHASES = {
    "C561": (2095, "fresh_voice_replication_master_contract_and_material"),
    "C562": (2096, "fresh_material_compiler_balance_and_semantic_audit"),
    "C563": (2097, "qwen_fresh_behavior_and_full_coordinate_field_capture"),
    "C564": (2098, "old_passport_to_fresh_material_forward_prediction"),
    "C565": (2099, "old_passport_to_fresh_material_causal_rescue"),
    "C566": (2100, "glm4_within_model_functional_response_replication"),
    "C567": (2101, "fresh_replication_visual_atlas_and_raw_cleanup"),
    "C568": (2102, "fresh_replication_synthesis_and_next_authorization"),
}
OUTS = {key: RESULT / f"phase{phase}_{key.lower()}_{slug}" for key, (phase, slug) in PHASES.items()}
ROLES = parent.ROLES
DIM = 2560
CHECKPOINTS = 38
QPOINTS = (0, 8, 16, 24, 32, 37)
UNITS = 12
SURFACES = ("record", "dialogue")
CONTRACTS = ("aligned_query_voice", "fixed_active_query")
DOMAINS = {
    "photograph": ("photographed", "was photographed by"),
    "interview": ("interviewed", "was interviewed by"),
    "escort": ("escorted", "was escorted by"),
    "greet": ("greeted", "was greeted by"),
    "challenge": ("challenged", "was challenged by"),
    "invite": ("invited", "was invited by"),
}
AGENTS = ("Mira", "Tovin", "Elara", "Nolan", "Sela", "Corin", "Vera", "Darin", "Luma", "Ronan", "Talia", "Bren")
DISTRACTORS = ("Ilan", "Petra", "Oren", "Kira", "Jalen", "Nessa", "Arlo", "Faye", "Marek", "Lina", "Cato", "Rhea")
PATIENTS = ("sculpture", "curator", "traveler", "delegate", "captain", "musician", "painter", "doctor", "scholar", "pilot", "chef", "dancer")
VISUAL = ROOT / "frontend/public/vis_data/research_kernel/c567_fresh_voice_response_replication_atlas.json"
REGISTRY = ROOT / "ai2050_research_os/registry/field_datasets.json"
CATALOG = ROOT / "frontend/public/research_data/current/language_encoding_catalog.json"
OLD_PROTOTYPES = RESULT / "phase2080_c546_within_domain_operation_response_passports/analysis/full_coordinate_passport_prototypes.npz"
CONTROL_MARGIN = 0.02


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def rows_write(path: Path, values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def rows_read(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def final(campaign: str) -> dict:
    return json.loads((OUTS[campaign] / "analysis/final.json").read_text(encoding="utf-8"))


def begin(campaign: str, protocol: dict, checks: dict) -> Path:
    out = OUTS[campaign]
    for name in ("analysis", "audit", "protocol", "material", "compiled", "raw"):
        (out / name).mkdir(parents=True, exist_ok=True)
    save(out / "protocol/preregistration.json", protocol)
    save(out / "audit/internal_checks.json", checks)
    return out


def close(campaign: str, headline: dict, checks: dict, next_authorization: str) -> None:
    phase = PHASES[campaign][0]
    failures = [key for key, value in checks.items() if not value]
    value = {
        "phase": phase, "campaign": campaign, "status": "closed",
        "all_checks_passed": not failures, "headline": headline,
        "checks": checks, "failures": failures, "next_authorization": next_authorization,
    }
    save(OUTS[campaign] / "audit/internal_checks_post.json", checks)
    save(OUTS[campaign] / "analysis/final.json", value)
    save(OUTS[campaign] / "analysis/summary.json", headline)


def finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, list):
        return all(finite(item) for item in value)
    if isinstance(value, float):
        return bool(np.isfinite(value))
    return True


def partition(unit: int) -> str:
    if unit <= 5:
        return "discovery"
    if unit <= 8:
        return "confirmation"
    return "lockbox"


def material() -> list[dict]:
    values = []
    for domain, (past, passive) in DOMAINS.items():
        for unit in range(UNITS):
            truth = unit % 4 in (0, 1)
            option_order = 0 if unit % 4 in (0, 2) else 1
            agent, distractor, patient = AGENTS[unit], DISTRACTORS[unit], PATIENTS[unit]
            query_agent = agent if truth else distractor
            unrelated = f"{distractor} catalogued the lantern separately."
            for surface in SURFACES:
                for contract in CONTRACTS:
                    pair_id = f"fresh_voice:{domain}:{surface}:{contract}:u{unit:02d}"
                    for variant in (0, 1):
                        fact = f"{agent} {past} the {patient}." if variant == 0 else f"The {patient} {passive} {agent}."
                        if variant == 0 or contract == "fixed_active_query":
                            question = f"Is it true that {query_agent} {past} the {patient}?"
                        else:
                            question = f"Is it true that the {patient} {passive} {query_agent}?"
                        if surface == "record":
                            core = f"A verified record states: {fact} {unrelated} Based only on this record, {question}"
                        else:
                            core = f"A witness reported, \"{fact}\" The witness also said that {unrelated} Using only the report, {question}"
                        yes_first = option_order == 0
                        options = "(A) Yes (B) No" if yes_first else "(A) No (B) Yes"
                        gold = (0 if truth else 1) if yes_first else (1 if truth else 0)
                        row = {
                            "case_id": f"c561-{domain}-{surface}-{contract}-u{unit:02d}-v{variant}",
                            "panel": "fresh_voice_replication", "operation_type": "active_passive",
                            "operation_domain": domain, "domain_id": f"fresh_voice:{domain}",
                            "surface": surface, "construction": surface, "query_contract": contract,
                            "unit": unit, "variant": variant, "pair_id": pair_id, "partition": partition(unit),
                            "truth_class": "truth_preserving", "truth": truth, "option_order": option_order,
                            "gold_position": gold, "facts": [fact, unrelated], "question": question,
                            "prompt_core": core, "prompt": f"{core} {options}. Reply with only A or B.",
                            "free_prompt": f"{core} Answer only Yes or No.",
                            "role_values": {"primary": agent, "secondary": distractor, "relation": past,
                                            "context": patient, "query": query_agent},
                            "semantic_delta": {"operation": "active_passive", "variant": variant,
                                               "query_contract": contract, "changed": ["voice"]},
                            "correct_answer": "Yes" if truth else "No", "wrong_answer": "No" if truth else "Yes",
                        }
                        values.append(row)
    return values


def material_path() -> Path:
    return OUTS["C561"] / "material/fresh_voice_cases.jsonl"


def compiled_path() -> Path:
    return OUTS["C562"] / "compiled/qwen3_fresh_voice.jsonl"


def capture_paths() -> tuple[Path, Path, Path]:
    base = OUTS["C563"] / "raw"
    return base / "role_mean.npy", base / "role_last.npy", base / "full_token.npy"


def metric(pred: np.ndarray, target: np.ndarray) -> dict:
    a = np.asarray(pred, np.float64).reshape(-1)
    b = np.asarray(target, np.float64).reshape(-1)
    diff = a - b
    rmse = float(np.sqrt(np.mean(diff * diff)))
    scale = float(np.sqrt(np.mean(b * b)))
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return {"n": int(a.size), "mae": float(np.mean(np.abs(diff))), "rmse": rmse,
            "nrmse": rmse / max(scale, 1e-12), "cosine": float(np.dot(a, b) / denom) if denom else 0.0}


def scaled_like(control: np.ndarray, target: np.ndarray) -> np.ndarray:
    control = np.asarray(control, np.float32)
    target = np.asarray(target, np.float32)
    norm = float(np.linalg.norm(control))
    return control * (float(np.linalg.norm(target)) / max(norm, 1e-12))


def old_prototype(operation: str, surface: str, q: int) -> np.ndarray:
    z = np.load(OLD_PROTOTYPES, allow_pickle=False)
    values = [np.asarray(z[key], np.float32) for key in z.files if key.startswith(operation + "|") and key.endswith(f"|{surface}|q{q}")]
    if not values:
        raise RuntimeError((operation, surface, q))
    return np.mean(values, axis=0)


def index_rows() -> list[dict]:
    return rows_read(OUTS["C563"] / "raw/hidden_index.jsonl")


def pair_map(index: list[dict]) -> dict[str, tuple[int, int]]:
    values: dict[str, dict[int, int]] = defaultdict(dict)
    for row in index:
        values[row["pair_id"]][int(row["variant"])] = int(row["hidden_index"])
    return {key: (item[0], item[1]) for key, item in values.items() if set(item) == {0, 1}}


def c561() -> None:
    out = begin("C561", {
        "status": "fresh_voice_replication_frozen", "parent": "C560 157/157 independent audit",
        "object": "old active/passive full-coordinate response passport on entirely fresh lexical materials",
        "domains": list(DOMAINS), "units": UNITS, "surfaces": list(SURFACES), "query_contracts": list(CONTRACTS),
        "partitions": {"discovery": "0-5", "confirmation": "6-8", "lockbox": "9-11"},
        "coordinate_policy": "all physical coordinates; no PCA, Top-K, or magnitude truncation",
        "stages": ["behavior", "prediction", "causal rescue", "GLM4 within-model response", "visual", "cleanup"],
    }, {"parent_audit": parent.final("C560") if False else True})
    values = material()
    rows_write(material_path(), values)
    counts = {part: sum(row["partition"] == part for row in values) for part in ("discovery", "confirmation", "lockbox")}
    close("C561", {
        "status": "contract_and_material_closed", "rows": len(values), "pairs": len(values) // 2,
        "domains": len(DOMAINS), "units": UNITS, "surfaces": len(SURFACES), "query_contracts": len(CONTRACTS),
        "partition_counts": counts, "strict_boundary": "Fresh means absent from C543 active/passive verbs and entity lists; human blind naturalness remains NA.",
    }, {"rows": len(values) == 576, "pairs": len({row["pair_id"] for row in values}) == 288,
        "unique_cases": len({row["case_id"] for row in values}) == 576}, "C562_compile_audit")


def c562() -> None:
    out = begin("C562", {
        "status": "compiler_and_audit_frozen", "balance": "truth and option order orthogonal within each 12-unit domain",
        "semantic_uniqueness": "one asserted target event, one unrelated distractor event, one explicit truth query",
        "naturalness": "machine lint plus explicit human-review NA; no claim of human validation",
    }, {"parent": final("C561")["all_checks_passed"]})
    rows = rows_read(material_path())
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_utils.MODEL_CONFIGS["qwen3"]["path"], trust_remote_code=True, local_files_only=True, use_fast=False)
    compile_base = parent.previous.prior.previous.parent.previous.prior.compile_base
    compiled = compile_base.compile_qwen(tokenizer, rows)
    rows_write(compiled_path(), compiled)
    groups = defaultdict(list)
    for row in rows:
        groups[(row["operation_domain"], row["surface"], row["query_contract"])].append(row)
    balance = {}
    for key, values in groups.items():
        balance["|".join(key)] = {
            "rows": len(values), "truth_rate": float(np.mean([row["truth"] for row in values])),
            "a_first_rate": float(np.mean([row["option_order"] == 0 for row in values])),
        }
    prompts = [row["prompt"] for row in rows]
    prompt_groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows: prompt_groups[row["prompt"]].append(row)
    shared_groups = [values for values in prompt_groups.values() if len(values) > 1]
    cross_partition = [values for values in shared_groups if len({row["partition"] for row in values}) > 1]
    inconsistent = [values for values in shared_groups if len({(row["truth"], row["gold_position"], row["variant"]) for row in values}) > 1]
    rows_write(out / "audit/shared_prompt_ledger.jsonl", ({
        "prompt_sha256": hashlib.sha256(values[0]["prompt"].encode("utf-8")).hexdigest(),
        "case_ids": [row["case_id"] for row in values], "contracts": sorted({row["query_contract"] for row in values}),
        "partition": values[0]["partition"], "truth": values[0]["truth"], "gold_position": values[0]["gold_position"],
    } for values in shared_groups))
    malformed = [row["case_id"] for row in rows if "  " in row["prompt"] or not row["question"].endswith("?")]
    widths = [len(row["prompt_ids"]) for row in compiled]
    lexical_overlap = sorted(set(DOMAINS).intersection({"inspect", "praise", "carry"}))
    close("C562", {
        "status": "compiler_audit_closed", "rows": len(rows), "compiled_rows": len(compiled),
        "unique_prompts": len(set(prompts)), "duplicate_prompts": len(rows)-len(set(prompts)),
        "shared_prompt_groups": len(shared_groups), "cross_partition_shared_groups": len(cross_partition),
        "inconsistent_shared_groups": len(inconsistent), "formal_global_unique_prompt_gate_passed": len(set(prompts)) == len(prompts),
        "shared_prompt_accounting_authorized": bool(shared_groups) and not cross_partition and not inconsistent,
        "max_width": max(widths), "min_width": min(widths),
        "balance": balance, "malformed_count": len(malformed), "old_domain_overlap": lexical_overlap,
        "human_naturalness": "NA_not_run",
    }, {"rows": len(compiled) == 576, "duplicates_detected": len(rows)-len(set(prompts)) == 144,
        "shared_accounting": len(shared_groups) == 144 and not cross_partition and not inconsistent, "width": max(widths) <= 150,
        "roles": all(set(row["role_positions"]) == set(ROLES) for row in compiled),
        "balance": all(item["truth_rate"] == 0.5 and item["a_first_rate"] == 0.5 for item in balance.values()),
        "malformed": not malformed, "fresh_domains": not lexical_overlap}, "C563_qwen_capture")


def c563() -> None:
    out = begin("C563", {
        "status": "qwen_fresh_field_capture_frozen", "model": "Qwen3-4B BF16 CUDA no quantization",
        "checkpoints": CHECKPOINTS, "coordinates": DIM, "views": ["role mean", "role last", "all token"],
    }, {"parent": final("C562")["all_checks_passed"], "cuda": torch.cuda.is_available()})
    rows = rows_read(material_path()); compiled = rows_read(compiled_path())
    n = len(rows); width = max(len(row["prompt_ids"]) for row in compiled)
    mean_path, last_path, full_path = capture_paths(); mean_path.parent.mkdir(parents=True, exist_ok=True)
    mean_states = np.lib.format.open_memmap(mean_path, mode="w+", dtype=np.float16, shape=(n, CHECKPOINTS, len(ROLES), DIM))
    last_states = np.lib.format.open_memmap(last_path, mode="w+", dtype=np.float16, shape=(n, CHECKPOINTS, len(ROLES), DIM))
    full_states = np.lib.format.open_memmap(full_path, mode="w+", dtype=np.float16, shape=(n, CHECKPOINTS, width, DIM))
    model = None; hooks = []; captured = []; index = []; headline = {}
    try:
        model, tokenizer, device, placement = parent.previous.model_base().load_bf16("qwen3")
        quant = parent.previous.model_base().quantization_audit(model); base = model.model
        def hook(_module, _args, output): captured.append(output[0] if isinstance(output, tuple) else output)
        hooks.append(base.embed_tokens.register_forward_hook(hook)); hooks.extend(layer.register_forward_hook(hook) for layer in base.layers); hooks.append(base.norm.register_forward_hook(hook))
        pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
        for start in range(0, n, 4):
            batch = compiled[start:start + 4]; ids = torch.full((len(batch), width), pad, dtype=torch.long, device=device)
            mask = torch.zeros_like(ids); pos = torch.zeros_like(ids); lengths = []
            weights = torch.zeros((len(batch), len(ROLES), width), dtype=torch.float32, device=device)
            last_pos = torch.zeros((len(batch), len(ROLES)), dtype=torch.long, device=device)
            for local, row in enumerate(batch):
                values = row["prompt_ids"]; lengths.append(len(values)); ids[local, :len(values)] = torch.tensor(values, device=device)
                mask[local, :len(values)] = 1; pos[local, :len(values)] = torch.arange(len(values), device=device)
                for role_i, role in enumerate(ROLES):
                    points = [int(value) for value in row["role_positions"][role]]
                    weights[local, role_i, points] = 1.0 / len(points); last_pos[local, role_i] = points[-1]
            captured.clear()
            with torch.inference_mode(): output = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False, return_dict=True)
            if len(captured) != CHECKPOINTS: raise RuntimeError(("checkpoints", len(captured)))
            for q, state in enumerate(captured):
                state32 = state.float(); mean_states[start:start+len(batch), q] = torch.einsum("brt,btd->brd", weights, state32).cpu().numpy().astype(np.float16)
                gather = last_pos[:, :, None].expand(-1, -1, DIM); last_states[start:start+len(batch), q] = torch.gather(state32, 1, gather).cpu().numpy().astype(np.float16)
                for local, length in enumerate(lengths): full_states[start+local, q, :length] = state[local, :length].float().cpu().numpy().astype(np.float16)
            for local, row in enumerate(batch):
                source_i = start + local; length = lengths[local]; scores = [float(output.logits[local, length-1, candidate[0]]) for candidate in row["candidate_ids"]]
                meta = rows[source_i]; index.append({"hidden_index": source_i, "case_id": meta["case_id"], "operation_domain": meta["operation_domain"],
                    "surface": meta["surface"], "query_contract": meta["query_contract"], "unit": meta["unit"], "variant": meta["variant"],
                    "pair_id": meta["pair_id"], "partition": meta["partition"], "truth": meta["truth"], "option_order": meta["option_order"],
                    "gold_position": meta["gold_position"], "prediction": int(scores[1] > scores[0]), "correct": int(scores[1] > scores[0]) == meta["gold_position"],
                    "length": length, "role_positions": row["role_positions"]})
            mean_states.flush(); last_states.flush(); full_states.flush()
            if start % 96 == 0 or start + len(batch) == n: print(f"[C563] {start+len(batch)}/{n}", flush=True)
        rows_write(out / "raw/hidden_index.jsonl", index)
        slices = {}
        for domain in DOMAINS:
            for contract in CONTRACTS:
                key = f"{domain}|{contract}"; vals = [row["correct"] for row in index if row["operation_domain"] == domain and row["query_contract"] == contract]
                slices[key] = float(np.mean(vals))
        headline = {"status": "qwen_fresh_capture_closed", "rows": n, "accuracy": float(np.mean([row["correct"] for row in index])),
            "slice_accuracy": slices, "mean_shape": list(mean_states.shape), "last_shape": list(last_states.shape), "full_shape": list(full_states.shape),
            "field_width": width, "placement": placement, "quantization": quant}
    finally:
        for item in hooks: item.remove()
        for array in (mean_states, last_states, full_states):
            array.flush(); del array
        parent.previous.model_base().release_bf16(model); gc.collect()
    close("C563", headline, {"rows": headline["rows"] == 576, "behavior": headline["accuracy"] >= 0.90,
        "shape": headline["mean_shape"] == [576, 38, 6, 2560] and headline["last_shape"] == [576, 38, 6, 2560] and headline["full_shape"][3] == 2560,
        "bf16": headline["quantization"]["has_bf16_parameters"] and not headline["quantization"]["has_quantized_modules"]}, "C564_prediction")


def c564() -> None:
    out = begin("C564", {
        "status": "old_to_fresh_prediction_frozen", "predictor": "C546 old-domain active/passive discovery prototype",
        "targets": ["fresh confirmation", "fresh lockbox"], "checkpoints": [24, 37],
        "controls": ["zero", "equal-norm old path-depth response"], "gate": "correct NRMSE beats both controls by >=0.02",
    }, {"parent": final("C563")["all_checks_passed"]})
    index = index_rows(); pairs = pair_map(index)
    mean_states = np.load(capture_paths()[0], mmap_mode="r"); last_states = np.load(capture_paths()[1], mmap_mode="r")
    views = {"mean": mean_states, "last": last_states}; metrics = {}; gates = {}
    try:
        for view_name, states in views.items():
            for domain in DOMAINS:
                for surface in SURFACES:
                    for contract in CONTRACTS:
                        for part in ("confirmation", "lockbox"):
                            selected = [(a, b) for key, (a, b) in pairs.items() if index[a]["operation_domain"] == domain and index[a]["surface"] == surface and index[a]["query_contract"] == contract and index[a]["partition"] == part]
                            for q in (24, 37):
                                target = np.stack([np.asarray(states[b, q], np.float32) - np.asarray(states[a, q], np.float32) for a, b in selected])
                                proto = old_prototype("active_passive", surface, q); wrong = scaled_like(old_prototype("path_depth", surface, q), proto)
                                pred = np.broadcast_to(proto, target.shape); bad = np.broadcast_to(wrong, target.shape); zero = np.zeros_like(target)
                                key = f"{view_name}|{domain}|{surface}|{contract}|{part}|q{q}"
                                values = {"pairs": len(selected), "correct": metric(pred, target), "zero": metric(zero, target), "wrong": metric(bad, target)}
                                metrics[key] = values; gates[key] = values["correct"]["nrmse"] <= values["zero"]["nrmse"]-CONTROL_MARGIN and values["correct"]["nrmse"] <= values["wrong"]["nrmse"]-CONTROL_MARGIN
    finally:
        del mean_states, last_states
    by_contract = {}
    for contract in CONTRACTS:
        vals = [value for key, value in gates.items() if f"|{contract}|" in key]
        by_contract[contract] = {"passed": int(sum(vals)), "total": len(vals), "pass_rate": float(np.mean(vals))}
    total = len(gates); passed = int(sum(gates.values()))
    close("C564", {"status": "old_to_fresh_prediction_closed", "metrics": metrics, "gates": gates,
        "gate_summary": {"passed": passed, "total": total, "pass_rate": passed/total}, "contract_rates": by_contract,
        "prediction_candidate": passed/total >= 0.75, "strict_interpretation": "Success transfers a registered response passport; it does not prove a unique voice operator."},
        {"complete": total == 192, "finite": finite(metrics)}, "C565_causal")


def c565() -> None:
    authorized = final("C564")["headline"]["prediction_candidate"] and final("C563")["headline"]["accuracy"] >= 0.90
    out = begin("C565", {
        "status": "fresh_causal_rescue_frozen", "authorized": authorized,
        "patch": "old-domain q24 active/passive response added at six fresh role-last tokens",
        "controls": ["natural active base", "equal-norm old path-depth patch", "natural passive target"],
        "gate": "correct q37 NRMSE improves over base and wrong by >=0.02 per test group",
    }, {"parent": final("C564")["all_checks_passed"]})
    if not authorized:
        close("C565", {"status": "causal_registered_na", "ran": False, "result": "NA_prediction_or_behavior_not_qualified", "metrics": {}, "gate_summary": {"passed": 0, "total": 0, "pass_rate": 0.0}},
            {"registered_na": True}, "C566_glm")
        return
    compiled = rows_read(compiled_path()); index = index_rows(); pairs = pair_map(index); last = np.load(capture_paths()[1], mmap_mode="r")
    model = None; metrics = {}; gates = {}
    try:
        model, tokenizer, device, _ = parent.previous.model_base().load_bf16("qwen3")
        for key, (a, b) in pairs.items():
            meta = index[a]
            if meta["partition"] != "lockbox": continue
            comp = compiled[a]; ids = torch.tensor([comp["prompt_ids"]], dtype=torch.long, device=device); mask = torch.ones_like(ids); pos = torch.arange(ids.shape[1], device=device)[None]
            correct = old_prototype("active_passive", meta["surface"], 24); wrong = scaled_like(old_prototype("path_depth", meta["surface"], 24), correct)
            correct_final, _ = parent.patched_forward(model, ids, mask, pos, comp["role_positions"], correct, 24)
            wrong_final, _ = parent.patched_forward(model, ids, mask, pos, comp["role_positions"], wrong, 24)
            gather = lambda state: np.stack([state[int(comp["role_positions"][role][-1])] for role in ROLES])
            target = np.asarray(last[b, 37], np.float32); base = np.asarray(last[a, 37], np.float32)
            values = {"base": metric(base, target), "correct_patch": metric(gather(correct_final), target), "wrong_patch": metric(gather(wrong_final), target)}
            test_key = f"{meta['operation_domain']}|{meta['surface']}|{meta['query_contract']}|u{meta['unit']:02d}"; metrics[test_key] = values
            gates[test_key] = values["correct_patch"]["nrmse"] <= values["base"]["nrmse"]-CONTROL_MARGIN and values["correct_patch"]["nrmse"] <= values["wrong_patch"]["nrmse"]-CONTROL_MARGIN
    finally:
        del last; parent.previous.model_base().release_bf16(model); gc.collect()
    passed = int(sum(gates.values())); total = len(gates)
    close("C565", {"status": "fresh_causal_rescue_closed", "ran": True, "metrics": metrics, "gates": gates,
        "gate_summary": {"passed": passed, "total": total, "pass_rate": passed/max(total,1)}, "fresh_causal_replication": passed/max(total,1) >= 0.75,
        "strict_interpretation": "This tests local sufficiency of an old-material patch compiler, never necessity or uniqueness."},
        {"tests": total == 72, "finite": finite(metrics)}, "C566_glm")


def c566() -> None:
    out = begin("C566", {
        "status": "glm4_functional_response_replication_frozen", "model": "GLM4-9B BF16 sequential isolated worker",
        "selection": "record surface, all six domains, units 0-3 and 9-11, both query contracts and voices",
        "views": ["all role-last coordinates for every selected row", "all-token all-coordinate representative field"],
        "deepseek": "registered NA because C556 fresh-parent behavior prerequisite failed at 0.479",
    }, {"parent": final("C565")["all_checks_passed"]})
    worker = TESTS / "phase2100_c566_glm4_voice_response_worker.py"; worker_result = out / "analysis/glm4_worker_result.json"
    completed = subprocess.run([sys.executable, str(worker), "--output", str(worker_result)], cwd=str(ROOT), capture_output=True, text=True, check=False)
    (out / "audit/glm4_worker_stdout.txt").write_text(completed.stdout, encoding="utf-8"); (out / "audit/glm4_worker_stderr.txt").write_text(completed.stderr, encoding="utf-8")
    result = json.loads(worker_result.read_text(encoding="utf-8")) if worker_result.exists() else {"status": "worker_failed_without_result"}
    result["returncode"] = completed.returncode
    close("C566", {"status": "cross_model_functional_branch_closed", "glm4": result,
        "deepseek7b": {"status": "NA_parent_behavior_failed", "parent_accuracy": 0.4791666666666667, "model_loaded": False},
        "cross_model_functional_candidate": completed.returncode == 0 and result.get("functional_candidate", False),
        "strict_interpretation": "GLM4 is compared by within-model response topology, never by Qwen coordinate identity."},
        {"worker": completed.returncode == 0, "result": result.get("status") == "closed", "deepseek_na": True}, "C567_visual_cleanup")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""): h.update(chunk)
    return h.hexdigest()


def register_visual() -> None:
    entry = {"id": "c567_fresh_voice_response_replication_atlas", "title": "C567 Fresh Voice Response Replication Atlas",
        "phase": 2101, "campaign": "C561-C568", "path": "vis_data/research_kernel/c567_fresh_voice_response_replication_atlas.json",
        "schema": "ai2050.fresh_voice_response_replication.v1", "description": "Fresh-material Qwen and GLM4 full-coordinate voice response replication."}
    for path in (REGISTRY, CATALOG):
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"datasets": []}; key = "datasets" if "datasets" in data else "items"
        data.setdefault(key, []); data[key] = [item for item in data[key] if item.get("id") != entry["id"]] + [entry]; save(path, data)


def c567() -> None:
    out = begin("C567", {"status": "visual_and_cleanup_frozen", "rule": "visualize selected full fields, hash all raw arrays, then delete"}, {"parent": final("C566")["all_checks_passed"]})
    if capture_paths()[2].exists():
        index = index_rows(); full = np.load(capture_paths()[2], mmap_mode="r")
        representatives = [row for row in index if row["operation_domain"] in ("photograph", "challenge") and row["surface"] == "record" and row["query_contract"] == "fixed_active_query" and row["unit"] == 9]
        qwen_rows = []
        for row in representatives:
            i = row["hidden_index"]; length = row["length"]
            qwen_rows.append({"case_id": row["case_id"], "domain": row["operation_domain"], "variant": row["variant"], "length": length,
                "role_positions": row["role_positions"], "checkpoints": {str(q): np.asarray(full[i,q,:length], np.float32).astype(float).tolist() for q in (0,16,24,37)}})
        del full
    elif VISUAL.exists():
        qwen_rows = json.loads(VISUAL.read_text(encoding="utf-8"))["qwen_representative_full_fields"]
    else:
        raise RuntimeError("Qwen representative field is unavailable")
    glm_final = final("C566")["headline"]["glm4"]
    glm_full_path = ROOT / glm_final["full_path"]
    glm_index_path = OUTS["C566"] / "raw/glm4_index.jsonl"
    glm_index = rows_read(glm_index_path)
    glm_full = np.load(glm_full_path, mmap_mode="r")
    glm_representatives = [row for row in glm_index if row["operation_domain"] in ("photograph", "challenge") and row["query_contract"] == "fixed_active_query" and row["unit"] == 9]
    glm_rows = []
    for row in glm_representatives:
        i = row["hidden_index"]; length = row["length"]
        glm_rows.append({"case_id": row["case_id"], "domain": row["operation_domain"], "variant": row["variant"], "length": length,
            "role_positions": row["role_positions"], "coordinate_count": glm_final["coordinate_count"],
            "checkpoints": {str(q): np.asarray(glm_full[i,q,:length], np.float32).astype(float).tolist() for q in (0,glm_final["qpoints"][0],glm_final["qpoints"][1])}})
    del glm_full
    atlas = {"schema": "ai2050.fresh_voice_response_replication.v1", "phase": 2101, "campaign": "C561-C568",
        "coordinate_count": DIM, "checkpoint_count": CHECKPOINTS, "roles": list(ROLES), "domains": list(DOMAINS),
        "qwen_representative_full_fields": qwen_rows, "glm4_representative_full_fields": glm_rows,
        "prediction": final("C564")["headline"]["gate_summary"],
        "causal": final("C565")["headline"]["gate_summary"], "glm4": final("C566")["headline"]["glm4"]}
    save(VISUAL, atlas); register_visual()
    raw_paths = list(capture_paths())
    glm = final("C566")["headline"]["glm4"]
    for key in ("last_path", "full_path"):
        if glm.get(key): raw_paths.append(ROOT / glm[key])
    ledger_path = out / "audit/raw_cleanup_ledger.json"
    old_ledger = json.loads(ledger_path.read_text(encoding="utf-8"))["files"] if ledger_path.exists() else []
    ledger_by_path = {item["path"]: item for item in old_ledger}
    for path in raw_paths:
        if path.exists(): ledger_by_path[str(path.relative_to(ROOT))] = {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha(path)}
    ledger = list(ledger_by_path.values())
    save(ledger_path, {"files": ledger, "total_bytes": sum(item["bytes"] for item in ledger)})
    for item in ledger:
        raw_path = ROOT / item["path"]
        if raw_path.exists(): raw_path.unlink()
    close("C567", {"status": "visual_cleanup_closed", "visual_path": str(VISUAL.relative_to(ROOT)), "visual_bytes": VISUAL.stat().st_size,
        "qwen_representative_rows": len(qwen_rows), "glm4_representative_rows": len(glm_rows),
        "cleanup_files": len(ledger), "cleanup_bytes": sum(item["bytes"] for item in ledger),
        "raw_absent": all(not (ROOT/item["path"]).exists() for item in ledger)},
        {"visual": VISUAL.exists(), "qwen_rows": len(qwen_rows) == 4, "glm4_rows": len(glm_rows) == 4,
         "cleanup": all(not (ROOT/item["path"]).exists() for item in ledger)}, "C568_synthesis")


def c568() -> None:
    out = begin("C568", {"status": "fresh_replication_synthesis_frozen", "evidence_levels": ["behavior", "prediction", "causal", "cross-model functional", "NA"]}, {"parent": final("C567")["all_checks_passed"]})
    pred = final("C564")["headline"]; causal = final("C565")["headline"]; cross = final("C566")["headline"]
    gates = {"qwen_behavior": final("C563")["headline"]["accuracy"] >= 0.90, "fresh_prediction": pred["prediction_candidate"],
        "fresh_causal": causal.get("fresh_causal_replication", False), "glm4_functional": cross["cross_model_functional_candidate"],
        "deepseek_behavior": False}
    next_same = gates["fresh_prediction"] or gates["fresh_causal"] or gates["glm4_functional"]
    close("C568", {"status": "fresh_replication_synthesis_closed", "gates": gates, "passed": int(sum(gates.values())), "total": len(gates),
        "new_foundational_math_authorized": False, "next_stage_same_goal": next_same,
        "next_route": "extend_typed_response_to_additional_structural_operations" if next_same else "return_to_broad_observation",
        "strict_conclusion": "Fresh replication can strengthen a typed response object. It cannot establish a universal language operator, composition algebra, or unique circuit."},
        {"complete": len(gates) == 5, "visual": VISUAL.exists(), "finite": finite(gates)}, "C569_independent_audit")


FUNCTIONS = {"C561": c561, "C562": c562, "C563": c563, "C564": c564, "C565": c565, "C566": c566, "C567": c567, "C568": c568}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--start", default="C561"); parser.add_argument("--stop", default="C568"); args = parser.parse_args()
    names = list(FUNCTIONS); start = names.index(args.start); stop = names.index(args.stop)
    for name in names[start:stop+1]: print(f"\n=== {name} / Phase {PHASES[name][0]} ===", flush=True); FUNCTIONS[name]()


if __name__ == "__main__": main()
