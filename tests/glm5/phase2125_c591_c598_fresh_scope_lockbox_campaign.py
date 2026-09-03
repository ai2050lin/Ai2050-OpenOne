#!/usr/bin/env python3
"""Fresh lexical/construction lockbox for the C571-C590 response candidates.

The script reads only token embeddings and HiddenStates. It never reads
attention or MLP internals and never uses PCA, Top-K, or magnitude filtering.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
import math
import shutil
import subprocess
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
VISUAL = ROOT / "frontend/public/vis_data/research_kernel/c598_fresh_scope_lockbox_atlas.json"
REGISTRY = ROOT / "ai2050_research_os/registry/field_datasets.json"
CATALOG = ROOT / "frontend/public/research_data/current/language_encoding_catalog.json"
sys.path.insert(0, str(TESTS))

import model_utils
import phase1797_c263_c272_state_operator_common as compiler
import phase2105_c571_c589_scope_program_algebra_campaign as previous


PHASES = {
    f"C{c}": (2125 + c - 591, slug)
    for c, slug in (
        (591, "recovery_amendment_and_fresh_lockbox_contract"),
        (592, "fresh_lexical_construction_and_query_switch_material"),
        (593, "fresh_qwen_behavior_qualification"),
        (594, "fresh_qwen_all_token_all_coordinate_capture"),
        (595, "frozen_passport_composition_and_dynamics_transfer"),
        (596, "output_changing_query_switch_causal_specificity"),
        (597, "qwen14_scale_response_topology"),
        (598, "parameter_atlas_cleanup_and_campaign_synthesis"),
    )
}
OUTS = {name: RESULT / f"phase{phase}_{name.lower()}_{slug}" for name, (phase, slug) in PHASES.items()}
ROLES = previous.ROLES
QPOINTS = previous.QPOINTS
DIM = previous.DIM
CHECKPOINTS = previous.CHECKPOINTS
CONTROL_MARGIN = previous.CONTROL_MARGIN
GATE = previous.PREDICTION_GATE
UNITS = 12

FRESH_NAMES_A = ("Selwin", "Mirel", "Tovin", "Elara", "Neris", "Calder", "Viora", "Hadren", "Ilyra", "Jorvan", "Keira", "Lucan", "Maelis", "Nolan", "Odette", "Pavel", "Riona", "Stellan")
FRESH_NAMES_B = ("Osmund", "Lysette", "Perrin", "Sabine", "Torren", "Una", "Valric", "Wrenna", "Xavian", "Ysolde", "Zorin", "Amara", "Bastien", "Celia", "Dorian", "Evelin", "Florian", "Greta")
FRESH_OBJECTS_A = ("acacia", "begonia", "cypress", "foxglove", "hemlock", "lavender", "myrtle", "primrose", "saffron", "tulip", "verbena", "wisteria", "anise", "bamboo", "clover", "dogwood", "eucalyptus", "freesia")
FRESH_OBJECTS_B = ("basalt", "ceramic", "copper", "garnet", "obsidian", "silk", "topaz", "willow", "agate", "burlap", "crystal", "dolomite", "enamel", "felt", "graphite", "hemp", "jasper", "lacquer")
FRESH_MIDDLES = tuple(f"bridgeunit{i:02d}" for i in range(18))
FRESH_TARGETS = tuple(f"terminalunit{i:02d}" for i in range(18))
FRESH_NOISES = ("protractor", "hygrometer", "periscope", "gyroscope", "manometer", "telescope", "seismograph", "rangefinder", "densitometer", "galvanometer", "interferometer", "magnetometer", "photometer", "radiometer", "sonometer", "thermocouple", "waveguide", "clinometer")


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


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(v) for v in value)
    if isinstance(value, (float, np.floating)):
        return math.isfinite(float(value))
    return True


def begin(name: str, protocol: dict, checks: dict) -> Path:
    out = OUTS[name]
    out.mkdir(parents=True, exist_ok=True)
    save(out / "protocol/preregistration.json", {
        "phase": PHASES[name][0], "campaign": name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "producer_sha256": sha(Path(__file__)), **protocol,
    })
    save(out / "audit/internal_checks.json", checks)
    if not all(bool(v) for v in checks.values()):
        raise RuntimeError((name, checks))
    return out


def close(name: str, headline: dict, checks: dict, authorization: str) -> dict:
    out = OUTS[name]
    save(out / "analysis/summary.json", headline)
    save(out / "audit/internal_checks_post.json", checks)
    value = {"phase": PHASES[name][0], "campaign": name, "status": "closed", "all_checks_passed": all(bool(v) for v in checks.values()), "headline": headline, "next_authorization": authorization}
    save(out / "analysis/final.json", value)
    if not value["all_checks_passed"]:
        raise RuntimeError((name, checks))
    return value


def final(name: str) -> dict:
    return load(OUTS[name] / "analysis/final.json")


def old_final(c: int) -> dict:
    return previous.final(f"C{c}")


def material_path() -> Path:
    return OUTS["C592"] / "material/fresh_scope_lockbox.jsonl"


def compiled_path() -> Path:
    return OUTS["C593"] / "compiled/qwen3_fresh_scope_lockbox.jsonl"


def behavior_path() -> Path:
    return OUTS["C593"] / "behavior/qwen3_behavior.jsonl"


def qualified_path() -> Path:
    return OUTS["C593"] / "behavior/qualified_slices.json"


def capture_path() -> Path:
    return OUTS["C594"] / "raw/qwen3_role_mean_states.float16.npy"


def capture_last_path() -> Path:
    return OUTS["C594"] / "raw/qwen3_role_last_states.float16.npy"


def capture_index_path() -> Path:
    return OUTS["C594"] / "raw/hidden_index.jsonl"


def shard_dir() -> Path:
    return OUTS["C594"] / "raw/qwen3_full_token_shards"


def fresh_partition(unit: int) -> str:
    if unit < 6:
        return "discovery"
    if unit < 9:
        return "confirmation"
    return "lockbox"


def fresh_wrap(surface: str, facts: list[str], question: str) -> str:
    body = " ".join(facts)
    if surface == "record":
        return f"The following archive entries are authoritative: {body} Considering only these entries, {question}"
    return f"Coordinator: Here are the accepted statements: {body} Auditor: From those statements alone, {question}"


def install_fresh_lexicon() -> None:
    previous.NAMES_A = FRESH_NAMES_A
    previous.NAMES_B = FRESH_NAMES_B
    previous.OBJECTS_A = FRESH_OBJECTS_A
    previous.OBJECTS_B = FRESH_OBJECTS_B
    previous.MIDDLES = FRESH_MIDDLES
    previous.TARGETS = FRESH_TARGETS
    previous.NOISES = FRESH_NOISES
    previous.wrap = fresh_wrap


def query_switch_case(domain: str, surface: str, unit: int, variant: int) -> dict:
    values = previous.values(unit)
    a, b, x, y, noise = (values[k] for k in ("a", "b", "x", "y", "noise"))
    relation = previous.relation_words(domain)[0]
    query_object = x if variant else y
    row = previous.make_row(
        case_id=f"c592-query-switch-{domain}-{surface}-u{unit:02d}-v{variant}",
        panel="query_switch_causal", family="query_object_switch", domain=domain,
        surface=surface, unit=unit, cell=f"v{variant}", variant=variant,
        facts=[f"{a} {relation} {x}.", f"{b} registered the {noise} independently."],
        question=f"Is it true that {a} {relation} {query_object}?", truth=bool(variant),
        roles={"primary": a, "secondary": b, "relation": relation, "context": x, "query": query_object},
        semantic_graph={"input_type": "evidence_query_program", "output_type": "binary_truth", "scope": "query_object", "invariants": ["evidence", "output_protocol", "entity_roles"], "changed": ["query_object", "truth"], "family": "query_object_switch", "domain": domain},
        order_offset=list(("inspect", "praise", "carry")).index(domain),
    )
    row["partition"] = fresh_partition(unit)
    return row


def make_fresh_material() -> list[dict]:
    install_fresh_lexicon()
    rows = [row for row in previous.material() if row["unit"] < UNITS]
    for row in rows:
        row["case_id"] = row["case_id"].replace("c572-", "c592-")
        row["partition"] = fresh_partition(int(row["unit"]))
        row["construction"] = "fresh_archive_" + row["construction"]
        row["source_lineage"] = "fresh_lexicon_and_fresh_wrapper"
    for domain, surface, unit, variant in itertools.product(("inspect", "praise", "carry"), previous.SURFACES, range(UNITS), (0, 1)):
        rows.append(query_switch_case(domain, surface, unit, variant))
    return rows


def metric(prediction: np.ndarray, truth: np.ndarray) -> dict:
    return previous.metric(prediction, truth)


def scaled_like(control: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return previous.scaled_like(control, reference)


def role_bundle(states: np.ndarray, row: dict, q: int) -> np.ndarray:
    return np.asarray(states[int(row["hidden_index"]), q], np.float32)


def pair_rows(index: list[dict], panel: str, family: str) -> list[tuple[dict, dict]]:
    groups: dict[tuple, dict[int, dict]] = defaultdict(dict)
    for row in index:
        if row["panel"] == panel and row["family"] == family and row.get("variant") is not None:
            groups[(row["operation_domain"], row["surface"], row["unit"])][int(row["variant"])] = row
    return [(v[0], v[1]) for v in groups.values() if set(v) == {0, 1}]


def factorial_groups(index: list[dict], panel: str) -> list[tuple[tuple, dict[str, dict]]]:
    groups: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for row in index:
        if row["panel"] == panel:
            groups[(row["operation_domain"], row["surface"], row["unit"])][row["cell"]] = row
    return sorted(groups.items())


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as values:
        return {key: np.asarray(values[key], np.float32) for key in values.files}


def save_npz(path: Path, values: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **values)


def c591() -> None:
    audit = load(RESULT / "phase2124_c590_scope_program_campaign_independent_audit/analysis/final.json")
    amendment = load(RESULT / "phase2124_c590_scope_program_campaign_independent_audit/audit/post_crash_storage_recovery_amendment.json")
    out = begin("C591", {
        "status": "fresh_lockbox_master_contract_frozen",
        "parent_authorization": audit["next_authorization"],
        "frozen_models": ["Qwen3-4B BF16 CUDA", "Qwen3-14B FP16 CUDA-plus-disk"],
        "object": "fresh lexical and wrapper transfer of C571 response, composition and dynamics candidates plus an output-changing query switch",
        "data_policy": "all token embeddings, all post-block states, final norm and all physical coordinates; no attention/MLP, PCA, Top-K or magnitude threshold",
        "gates": {"behavior_slice": 0.75, "prediction_margin": CONTROL_MARGIN, "candidate_pass_rate": GATE},
        "failure_policy": "route-level accounting; a failed family does not stop unrelated families",
    }, {"c590_passed": audit["all_checks_passed"], "same_goal_authorized": audit["headline"]["route"]["same_exact_goal"], "recovery_explicit": amendment["status"] == "post_crash_storage_recovery_amendment"})
    headline = {"status": "recovery_amendment_and_fresh_contract_closed", "c590_checks": audit["headline"]["checks_total"], "historical_hash_mismatch_phases": amendment["changed_before_current_hash"], "provenance_limitation_retained": True, "planned_units": UNITS, "planned_partitions": {"discovery": 6, "confirmation": 3, "lockbox": 3}, "strict_interpretation": "The recovery amendment documents a storage implementation change; it is not retrospective preregistration."}
    save(out / "analysis/evidence_audit.json", {"c590": audit, "recovery_amendment": amendment})
    close("C591", headline, {"audit": audit["all_checks_passed"], "lineage": bool(amendment["historical_producer_hashes"])}, "C592_material")


def c592() -> None:
    out = begin("C592", {"status": "fresh_material_frozen", "lexicon": "18 unused proper-name/object/noise tuples; first 12 used", "construction": "archive and coordinator/auditor wrappers absent from C572", "panels": "all C572 panels plus output-changing query switch", "human_naturalness": "NA_not_run; machine syntax and semantic ledger only"}, {"parent": final("C591")["all_checks_passed"]})
    rows = make_fresh_material()
    write_rows(material_path(), rows)
    old_rows = read_rows(previous.material_path())
    old_prompts = {row["prompt"] for row in old_rows}
    prompts = [row["prompt"] for row in rows]
    duplicate_groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        duplicate_groups[row["prompt"]].append(row)
    shared = [items for items in duplicate_groups.values() if len(items) > 1]
    cross_partition = [items for items in shared if len({row["partition"] for row in items}) > 1]
    inconsistent = [items for items in shared if len({(row["truth"], row["gold_position"]) for row in items}) > 1]
    slices = defaultdict(int)
    for row in rows:
        slices[f"{row['panel']}|{row['family']}|{row['operation_domain']}"] += 1
    malformed = [row["case_id"] for row in rows if "  " in row["prompt"] or not row["question"].endswith("?")]
    headline = {"status": "fresh_material_closed", "rows": len(rows), "unique_cases": len({r["case_id"] for r in rows}), "unique_prompts": len(set(prompts)), "shared_prompt_groups": len(shared), "cross_partition_shared_groups": len(cross_partition), "inconsistent_shared_groups": len(inconsistent), "old_prompt_overlap": sum(p in old_prompts for p in prompts), "panel_counts": {panel: sum(r["panel"] == panel for r in rows) for panel in sorted({r["panel"] for r in rows})}, "partition_counts": {part: sum(r["partition"] == part for r in rows) for part in ("discovery", "confirmation", "lockbox")}, "slice_counts": dict(slices), "malformed": len(malformed), "human_naturalness": "NA_not_run", "examples": {panel: next(r["prompt"] for r in rows if r["panel"] == panel) for panel in sorted({r["panel"] for r in rows})}}
    close("C592", headline, {"nonempty": bool(rows), "unique": len({r["case_id"] for r in rows}) == len(rows), "fresh_prompts": not any(p in old_prompts for p in prompts), "no_cross_partition_duplicates": not cross_partition, "consistent_duplicates": not inconsistent, "syntax": not malformed, "query_switch": any(r["panel"] == "query_switch_causal" for r in rows)}, "C593_behavior")


def c593() -> None:
    out = begin("C593", {"status": "fresh_qwen_behavior_frozen", "model": "Qwen3-4B BF16 CUDA", "behavior_before_hiddenstate": True, "gate": "each panel-family-domain slice >=0.75; only qualified slices may be captured"}, {"parent": final("C592")["all_checks_passed"], "cuda": torch.cuda.is_available()})
    rows = read_rows(material_path())
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_utils.MODEL_CONFIGS["qwen3"]["path"], trust_remote_code=True, local_files_only=True, use_fast=False)
    compiled = compiler.compile_qwen(tokenizer, rows)
    write_rows(compiled_path(), compiled)
    model = None
    behavior = []
    try:
        model, tokenizer, device, placement = previous.parent.previous.model_base().load_bf16("qwen3")
        pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
        for start in range(0, len(compiled), 12):
            batch = compiled[start:start + 12]
            width = max(len(r["prompt_ids"]) for r in batch)
            ids = torch.full((len(batch), width), pad, dtype=torch.long, device=device)
            mask = torch.zeros_like(ids)
            for i, row in enumerate(batch):
                seq = row["prompt_ids"]
                ids[i, :len(seq)] = torch.tensor(seq, device=device)
                mask[i, :len(seq)] = 1
            pos = mask.long().cumsum(-1) - 1
            pos.masked_fill_(mask == 0, 0)
            with torch.inference_mode():
                logits = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False, return_dict=True).logits
            for i, row in enumerate(batch):
                length = len(row["prompt_ids"])
                scores = [float(logits[i, length - 1, candidate[0]]) for candidate in row["candidate_ids"]]
                pred = int(scores[1] > scores[0])
                source = rows[start + i]
                behavior.append({"case_id": row["case_id"], "panel": source["panel"], "family": source["family"], "operation_domain": source["operation_domain"], "surface": source["surface"], "unit": source["unit"], "partition": source["partition"], "variant": source.get("variant"), "gold_position": row["gold_position"], "prediction": pred, "correct": pred == row["gold_position"], "candidate_scores": scores})
            if start % 360 == 0 or start + len(batch) == len(compiled):
                print(f"[C593 behavior] {start + len(batch)}/{len(compiled)}", flush=True)
    finally:
        previous.parent.previous.model_base().release_bf16(model)
        gc.collect()
    write_rows(behavior_path(), behavior)
    slices = {}
    for key, items in itertools.groupby(sorted(behavior, key=lambda r: (r["panel"], r["family"], r["operation_domain"])), key=lambda r: (r["panel"], r["family"], r["operation_domain"])):
        values = list(items)
        accuracy = float(np.mean([v["correct"] for v in values]))
        slices["|".join(key)] = {"rows": len(values), "accuracy": accuracy, "qualified": accuracy >= 0.75}
    qualified = sorted(key for key, value in slices.items() if value["qualified"])
    save(qualified_path(), {"gate": 0.75, "qualified": qualified, "slices": slices})
    headline = {"status": "fresh_behavior_closed", "rows": len(rows), "compiled_rows": len(compiled), "behavior_accuracy": float(np.mean([r["correct"] for r in behavior])), "qualified_slices": len(qualified), "total_slices": len(slices), "slices": slices, "max_width": max(len(r["prompt_ids"]) for r in compiled), "placement": placement, "human_naturalness": "NA_not_run"}
    close("C593", headline, {"complete": len(behavior) == len(rows), "compiled": len(compiled) == len(rows), "qualified": bool(qualified), "finite": finite(headline)}, "C594_capture")


def c594() -> None:
    out = begin("C594", {"status": "fresh_qwen_full_field_capture_frozen", "model": "Qwen3-4B BF16 CUDA", "selection": "only C593-qualified slices", "tensor": "sample x embedding+36 blocks+final norm x all prompt tokens x all 2560 coordinates", "storage": "24-row bounded shards plus all-coordinate role mean and role-last tensors"}, {"parent": final("C593")["all_checks_passed"], "cuda": torch.cuda.is_available()})
    rows = read_rows(material_path())
    compiled_all = read_rows(compiled_path())
    behavior = {r["case_id"]: r for r in read_rows(behavior_path())}
    qualified = set(load(qualified_path())["qualified"])
    selected = [(row, comp) for row, comp in zip(rows, compiled_all) if f"{row['panel']}|{row['family']}|{row['operation_domain']}" in qualified]
    selected.sort(key=lambda item: len(item[1]["prompt_ids"]))
    n = len(selected)
    estimated = sum(CHECKPOINTS * len(comp["prompt_ids"]) * DIM * 2 for _, comp in selected) + 2 * n * CHECKPOINTS * len(ROLES) * DIM * 2
    free_before = shutil.disk_usage(RESULT).free
    if free_before < estimated + (8 << 30):
        raise RuntimeError({"free": free_before, "estimated": estimated, "required_headroom": 8 << 30})
    shard_dir().mkdir(parents=True, exist_ok=True)
    capture_path().parent.mkdir(parents=True, exist_ok=True)
    mean_states = np.lib.format.open_memmap(capture_path(), mode="w+", dtype=np.float16, shape=(n, CHECKPOINTS, len(ROLES), DIM))
    last_states = np.lib.format.open_memmap(capture_last_path(), mode="w+", dtype=np.float16, shape=(n, CHECKPOINTS, len(ROLES), DIM))
    model = None
    hooks = []
    captured = []
    index = []
    ledger = []
    headline = {}
    try:
        model, tokenizer, device, placement = previous.parent.previous.model_base().load_bf16("qwen3")
        quant = previous.parent.previous.model_base().quantization_audit(model)
        base = model.model
        def hook(_module, _args, output):
            captured.append(output[0] if isinstance(output, tuple) else output)
        hooks.append(base.embed_tokens.register_forward_hook(hook))
        hooks.extend(layer.register_forward_hook(hook) for layer in base.layers)
        hooks.append(base.norm.register_forward_hook(hook))
        pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
        for shard_start in range(0, n, 24):
            items = selected[shard_start:shard_start + 24]
            width = max(len(comp["prompt_ids"]) for _, comp in items)
            shard_id = shard_start // 24
            shard_path = shard_dir() / f"shard_{shard_id:04d}.float16.npy"
            shard = np.lib.format.open_memmap(shard_path, mode="w+", dtype=np.float16, shape=(len(items), CHECKPOINTS, width, DIM))
            for local_start in range(0, len(items), 4):
                batch = items[local_start:local_start + 4]
                ids = torch.full((len(batch), width), pad, dtype=torch.long, device=device)
                mask = torch.zeros_like(ids)
                weights = torch.zeros((len(batch), len(ROLES), width), dtype=torch.float32, device=device)
                last_pos = torch.zeros((len(batch), len(ROLES)), dtype=torch.long, device=device)
                lengths = []
                for i, (_row, comp) in enumerate(batch):
                    seq = comp["prompt_ids"]
                    lengths.append(len(seq))
                    ids[i, :len(seq)] = torch.tensor(seq, device=device)
                    mask[i, :len(seq)] = 1
                    for role_i, role in enumerate(ROLES):
                        points = [int(v) for v in comp["role_positions"][role]]
                        weights[i, role_i, points] = 1.0 / len(points)
                        last_pos[i, role_i] = points[-1]
                pos = mask.long().cumsum(-1) - 1
                pos.masked_fill_(mask == 0, 0)
                captured.clear()
                with torch.inference_mode():
                    model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False, return_dict=True)
                if len(captured) != CHECKPOINTS:
                    raise RuntimeError((len(captured), CHECKPOINTS))
                for q, state in enumerate(captured):
                    state32 = state.float()
                    target_slice = slice(shard_start + local_start, shard_start + local_start + len(batch))
                    mean_states[target_slice, q] = torch.einsum("brt,btd->brd", weights, state32).cpu().numpy().astype(np.float16)
                    gather = last_pos[:, :, None].expand(-1, -1, DIM)
                    last_states[target_slice, q] = torch.gather(state32, 1, gather).cpu().numpy().astype(np.float16)
                    for i, length in enumerate(lengths):
                        shard[local_start + i, q, :length] = state[i, :length].float().cpu().numpy().astype(np.float16)
                for i, (row, comp) in enumerate(batch):
                    index.append({"hidden_index": shard_start + local_start + i, "shard": shard_path.name, "shard_index": local_start + i, "case_id": row["case_id"], "panel": row["panel"], "family": row["family"], "operation_domain": row["operation_domain"], "surface": row["surface"], "unit": row["unit"], "partition": row["partition"], "cell": row["cell"], "variant": row.get("variant"), "factors": row.get("factors", {}), "truth": row["truth"], "length": lengths[i], "role_positions": comp["role_positions"], "behavior_correct": behavior[row["case_id"]]["correct"]})
            shard.flush()
            del shard
            mean_states.flush()
            last_states.flush()
            ledger.append({"shard": shard_path.name, "rows": len(items), "width": width, "bytes": shard_path.stat().st_size})
            print(f"[C594 capture] {min(shard_start + len(items), n)}/{n} shard={shard_id:04d}", flush=True)
        write_rows(capture_index_path(), index)
        save(out / "raw/shard_ledger.json", ledger)
        raw_bytes = sum(v["bytes"] for v in ledger) + capture_path().stat().st_size + capture_last_path().stat().st_size
        headline = {"status": "fresh_qwen_full_field_closed", "rows": n, "qualified_slices": len(qualified), "role_mean_shape": list(mean_states.shape), "role_last_shape": list(last_states.shape), "full_token_shards": len(ledger), "raw_bytes": raw_bytes, "estimated_bytes": estimated, "free_disk_before": free_before, "behavior_correct_rate": float(np.mean([r["behavior_correct"] for r in index])), "placement": placement, "quantization": quant}
    finally:
        for handle in hooks:
            handle.remove()
        mean_states.flush()
        last_states.flush()
        del mean_states, last_states
        previous.parent.previous.model_base().release_bf16(model)
        gc.collect()
    close("C594", headline, {"rows": headline["rows"] == n and n > 0, "shape": headline["role_mean_shape"] == [n, 38, 6, 2560], "shards": headline["full_token_shards"] > 0, "bf16": headline["quantization"]["has_bf16_parameters"], "index": len(read_rows(capture_index_path())) == n}, "C595_transfer")


def c595() -> None:
    out = begin("C595", {"status": "frozen_old_candidate_transfer_frozen", "training": "all response passports and coordinate dynamics are frozen from C575/C580/C581/C587/C588", "test": "C592 unseen words and unseen wrappers only", "controls": ["zero response", "equal-norm wrong family/effect", "identity dynamics", "old target mean"], "gate": "correct NRMSE beats both registered controls by >=0.02; family candidate pass rate >=0.75"}, {"parent": final("C594")["all_checks_passed"]})
    index = read_rows(capture_index_path())
    fresh = np.load(capture_path(), mmap_mode="r")
    old_states = np.load(previous.capture_path(), mmap_mode="r")
    old_index = previous.index_rows()
    old_atomic = load_npz(previous.OUTS["C575"] / "analysis/discovery_atomic_prototypes.npz")
    observed = {}
    metrics = {}
    gates = {}
    dynamic = {}
    try:
        for family in previous.ATOMIC_SPECS:
            pairs_all = pair_rows(index, "atomic", family)
            for domain, surface, q in itertools.product(previous.ATOMIC_SPECS[family], previous.SURFACES, (16, 24, 37)):
                pairs = [p for p in pairs_all if p[0]["operation_domain"] == domain and p[0]["surface"] == surface]
                if not pairs:
                    continue
                truth = np.stack([role_bundle(fresh, b, q) - role_bundle(fresh, a, q) for a, b in pairs])
                key = f"{family}|{domain}|{surface}|q{q}"
                proto = old_atomic[key]
                wrong_family = next(name for name in previous.ATOMIC_SPECS if name != family)
                wrong_key = next(k for k in old_atomic if k.startswith(wrong_family + "|") and k.endswith(f"|{surface}|q{q}"))
                wrong = scaled_like(old_atomic[wrong_key], proto)
                value = {"samples": len(pairs), "correct": metric(np.broadcast_to(proto, truth.shape), truth), "zero": metric(np.zeros_like(truth), truth), "wrong": metric(np.broadcast_to(wrong, truth.shape), truth), "wrong_family": wrong_family}
                metrics[key] = value
                gates[key] = value["correct"]["nrmse"] <= value["zero"]["nrmse"] - CONTROL_MARGIN and value["correct"]["nrmse"] <= value["wrong"]["nrmse"] - CONTROL_MARGIN
                observed[key] = truth.mean(axis=0).astype(np.float32)

            old_train = previous.pair_rows(old_index, "atomic", family, "discovery")
            fresh_test = pairs_all
            for q0, q1 in ((16, 24), (24, 37)):
                xtr = np.stack([previous.pair_delta(old_states, p, q0) for p in old_train])
                ytr = np.stack([previous.pair_delta(old_states, p, q1) for p in old_train])
                xte = np.stack([role_bundle(fresh, b, q0) - role_bundle(fresh, a, q0) for a, b in fresh_test])
                yte = np.stack([role_bundle(fresh, b, q1) - role_bundle(fresh, a, q1) for a, b in fresh_test])
                beta, _ = previous.fit_coordinate_model(xtr, ytr)
                pred = previous.predict_coordinate_model(xte, beta)
                mean = np.broadcast_to(ytr.mean(axis=0), yte.shape)
                value = {"coordinate_affine": metric(pred, yte), "identity": metric(xte, yte), "old_target_mean": metric(mean, yte), "samples": len(fresh_test)}
                value["gate"] = value["coordinate_affine"]["nrmse"] <= value["identity"]["nrmse"] - CONTROL_MARGIN and value["coordinate_affine"]["nrmse"] <= value["old_target_mean"]["nrmse"] - CONTROL_MARGIN
                dynamic[f"{family}|q{q0}->q{q1}"] = value

        panel_specs = (
            ("discourse_voice_composition", previous.composition_effect, "interaction", previous.OUTS["C580"] / "analysis/composition_interaction_prototypes.npz"),
            ("path_paraphrase_composition", previous.composition_effect, "interaction", previous.OUTS["C580"] / "analysis/composition_interaction_prototypes.npz"),
            ("nested_attitude_flagship", previous.nested_effect, "interaction", previous.OUTS["C587"] / "analysis/nested_attitude_effect_prototypes.npz"),
        )
        composition = {}
        for panel, effect_fn, label, proto_path in panel_specs:
            book = load_npz(proto_path)
            for (domain, surface, _unit), cells in factorial_groups(index, panel):
                if len(cells) != 4:
                    continue
                for q in (16, 24, 37):
                    key = f"{panel}|{domain}|q{q}" if panel != "nested_attitude_flagship" else f"{domain}|{surface}|{label}|q{q}"
                    if key not in book:
                        continue
                    target = effect_fn(fresh, cells, q)[label]
                    composition.setdefault(key, []).append(target)
        composition_metrics = {}
        for key, values in composition.items():
            target = np.stack(values)
            if key.startswith("nested"):
                continue
            source_path = previous.OUTS["C587"] / "analysis/nested_attitude_effect_prototypes.npz" if "|record|" in key or "|dialogue|" in key else previous.OUTS["C580"] / "analysis/composition_interaction_prototypes.npz"
            book = load_npz(source_path)
            proto = book[key]
            value = {"samples": len(values), "correct": metric(np.broadcast_to(proto, target.shape), target), "zero": metric(np.zeros_like(target), target)}
            value["gate"] = value["correct"]["nrmse"] <= value["zero"]["nrmse"] - CONTROL_MARGIN
            composition_metrics[key] = value
    finally:
        previous.close_mmap(fresh)
        previous.close_mmap(old_states)
        del fresh, old_states
    save_npz(out / "analysis/fresh_observed_atomic_responses.npz", observed)
    family_summary = {family: {"passed": sum(v for k, v in gates.items() if k.startswith(family + "|")), "total": sum(k.startswith(family + "|") for k in gates)} for family in previous.ATOMIC_SPECS}
    for value in family_summary.values():
        value["pass_rate"] = value["passed"] / max(value["total"], 1)
        value["candidate"] = value["pass_rate"] >= GATE
    dynamic_summary = {"passed": sum(v["gate"] for v in dynamic.values()), "total": len(dynamic)}
    composition_summary = {"passed": sum(v["gate"] for v in composition_metrics.values()), "total": len(composition_metrics)}
    headline = {"status": "fresh_transfer_closed", "atomic_metrics": metrics, "atomic_gates": gates, "family_summary": family_summary, "fresh_atomic_candidates": [k for k, v in family_summary.items() if v["candidate"]], "dynamic_metrics": dynamic, "dynamic_summary": dynamic_summary, "composition_metrics": composition_metrics, "composition_summary": composition_summary, "strict_interpretation": "Passing transfers a frozen experimental response law to new words and wrappers. It does not identify a stored semantic variable or a unique physical circuit."}
    close("C595", headline, {"atomic": bool(metrics), "dynamic": bool(dynamic), "finite": finite(headline)}, "C596_causal_specificity")


def c596() -> None:
    out = begin("C596", {"status": "output_changing_query_switch_causal_test_frozen", "eligible_object": "fresh query-object switch only", "intervention": "add signed discovery q24 role-last response to the opposite lockbox query state", "directions": ["false-to-true", "true-to-false"], "controls": ["zero patch", "equal-norm old surface-response patch"], "readouts": ["q37 role-last state NRMSE", "A/B target output"], "gate": "state improves by >=0.02 over both controls and output changes to target"}, {"parent": final("C595")["all_checks_passed"]})
    rows = {r["case_id"]: r for r in read_rows(material_path())}
    compiled = {r["case_id"]: r for r in read_rows(compiled_path())}
    index = read_rows(capture_index_path())
    states = np.load(capture_last_path(), mmap_mode="r")
    pairs = pair_rows(index, "query_switch_causal", "query_object_switch")
    if not pairs:
        previous.close_mmap(states)
        del states
        close("C596", {"status": "query_switch_causal_registered_na", "metrics": {}, "gates": {}, "direction_summary": {}, "output_changing_sufficiency_candidate": False, "reason": "query-switch behavior slice did not qualify for HiddenState capture"}, {"registered_na": True}, "C597_qwen14")
        return
    old_book = load_npz(previous.OUTS["C575"] / "analysis/discovery_atomic_prototypes.npz")
    protos = {}
    for domain, surface in itertools.product(("inspect", "praise", "carry"), previous.SURFACES):
        train = [p for p in pairs if p[0]["operation_domain"] == domain and p[0]["surface"] == surface and p[0]["unit"] < 6]
        if train:
            protos[(domain, surface)] = np.stack([role_bundle(states, b, 24) - role_bundle(states, a, 24) for a, b in train]).mean(axis=0)
    model = None
    metrics = {}
    gates = {}
    try:
        model, tokenizer, device, _placement = previous.parent.previous.model_base().load_bf16("qwen3")
        wrong_base = old_book[next(k for k in old_book if k.startswith("discourse_permutation|") and k.endswith("|record|q24"))]
        for left, right in pairs:
            if left["unit"] < 9:
                continue
            proto = protos[(left["operation_domain"], left["surface"])]
            wrong = scaled_like(wrong_base, proto)
            for source, target, signed, direction in ((left, right, proto, "false_to_true"), (right, left, -proto, "true_to_false")):
                comp = compiled[source["case_id"]]
                target_comp = compiled[target["case_id"]]
                if comp["candidate_ids"] != target_comp["candidate_ids"]:
                    raise RuntimeError("query-switch candidate identity mismatch")
                ids = torch.tensor([comp["prompt_ids"]], dtype=torch.long, device=device)
                mask = torch.ones_like(ids)
                pos = torch.arange(ids.shape[1], device=device)[None]
                zero_state, zero_logits = previous.parent.patched_forward(model, ids, mask, pos, comp["role_positions"], np.zeros_like(signed), 24)
                correct_state, correct_logits = previous.parent.patched_forward(model, ids, mask, pos, comp["role_positions"], signed, 24)
                wrong_state, wrong_logits = previous.parent.patched_forward(model, ids, mask, pos, comp["role_positions"], wrong if direction == "false_to_true" else -wrong, 24)
                gather = lambda state: np.stack([state[int(comp["role_positions"][role][-1])] for role in ROLES])
                target_state = role_bundle(states, target, 37)
                candidate_ids = comp["candidate_ids"]
                target_gold = rows[target["case_id"]]["gold_position"]
                pred = lambda logits: int(float(logits[candidate_ids[1][0]]) > float(logits[candidate_ids[0][0]]))
                value = {"source": source["case_id"], "target": target["case_id"], "zero": metric(gather(zero_state), target_state), "correct": metric(gather(correct_state), target_state), "wrong": metric(gather(wrong_state), target_state), "zero_prediction": pred(zero_logits), "correct_prediction": pred(correct_logits), "wrong_prediction": pred(wrong_logits), "target_gold": target_gold}
                key = f"{left['operation_domain']}|{left['surface']}|u{left['unit']}|{direction}"
                metrics[key] = value
                gates[key] = value["correct"]["nrmse"] <= value["zero"]["nrmse"] - CONTROL_MARGIN and value["correct"]["nrmse"] <= value["wrong"]["nrmse"] - CONTROL_MARGIN and value["correct_prediction"] == target_gold
    finally:
        previous.parent.previous.model_base().release_bf16(model)
        previous.close_mmap(states)
        del states
        gc.collect()
    by_direction = {direction: {"passed": sum(v for k, v in gates.items() if k.endswith(direction)), "total": sum(k.endswith(direction) for k in gates)} for direction in ("false_to_true", "true_to_false")}
    for value in by_direction.values():
        value["pass_rate"] = value["passed"] / max(value["total"], 1)
    headline = {"status": "query_switch_causal_specificity_closed", "metrics": metrics, "gates": gates, "direction_summary": by_direction, "output_changing_sufficiency_candidate": all(v["pass_rate"] >= GATE for v in by_direction.values()), "strict_interpretation": "A pass is sufficient state guidance for this compiled query switch. It does not establish necessity, uniqueness, or general semantic editing."}
    close("C596", headline, {"tests": bool(metrics), "finite": finite(headline), "directions": all(v["total"] > 0 for v in by_direction.values())}, "C597_qwen14")


def c597() -> None:
    out = begin("C597", {"status": "qwen14_scale_panel_frozen", "model": "local Qwen3-14B FP16, 18 layers CUDA and 22 layers plus norm/head disk-offloaded", "subset": "six atomic families, first domain, record surface, units 0-7 and both variants", "behavior_policy": "no HiddenState capture unless model-specific behavior >=0.75", "coordinate_policy": "all token/checkpoint/5120 coordinates; model-relative topology only"}, {"parent": final("C596")["all_checks_passed"], "model_exists": (ROOT / "models/hf/Qwen3-14B").exists(), "cuda": torch.cuda.is_available()})
    pre_scale_cleanup = []
    legacy_shards = previous.full_shard_dir()
    if legacy_shards.exists():
        resolved = legacy_shards.resolve()
        if not str(resolved).lower().startswith(str(ROOT.resolve()).lower()):
            raise RuntimeError(f"cleanup target escaped workspace: {resolved}")
        size = sum(path.stat().st_size for path in legacy_shards.rglob("*") if path.is_file())
        shutil.rmtree(legacy_shards)
        pre_scale_cleanup.append({"path": str(legacy_shards.relative_to(ROOT)), "bytes": size, "reason": "C589 visualization already contains the derived full-coordinate panels; bulk shards are not directly displayed"})
    worker = TESTS / "phase2131_c597_qwen14_fresh_scope_worker.py"
    result_path = out / "analysis/qwen14_worker_result.json"
    completed = subprocess.run([sys.executable, str(worker), "--material", str(material_path()), "--output", str(result_path)], cwd=str(ROOT), capture_output=True, text=True, check=False)
    (out / "audit/qwen14_stdout.txt").parent.mkdir(parents=True, exist_ok=True)
    (out / "audit/qwen14_stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (out / "audit/qwen14_stderr.txt").write_text(completed.stderr, encoding="utf-8")
    result = load(result_path) if result_path.exists() else {"status": "worker_failed_without_result"}
    result["returncode"] = completed.returncode
    coverage = {"qwen3_4b": old_final(573)["headline"]["behavior_accuracy"], "glm4": old_final(586)["headline"]["models"]["glm4"].get("behavior_accuracy"), "deepseek7b": old_final(586)["headline"]["models"]["deepseek7b"].get("behavior_accuracy"), "qwen3_14b": result.get("behavior_accuracy")}
    headline = {"status": "qwen14_scale_panel_closed", "qwen14": result, "four_model_behavior_coverage": coverage, "pre_scale_cleanup": pre_scale_cleanup, "strict_interpretation": "Qwen3-14B is an independently frozen scale panel. Its physical coordinate numbers are never equated with Qwen3-4B, GLM4 or DeepSeek."}
    close("C597", headline, {"worker_returned": completed.returncode in (0, 1, 2), "result": result_path.exists(), "finite": finite(result)}, "C598_visual_cleanup")


def register_visual() -> None:
    entry = {"id": "c598_fresh_scope_lockbox_atlas", "title": "C598 Fresh Scope Lockbox Full-Parameter Atlas", "phase": 2132, "campaign": "C591-C598", "path": "vis_data/research_kernel/c598_fresh_scope_lockbox_atlas.json", "schema": "ai2050.fresh_scope_lockbox_atlas.v1", "description": "Fresh-word response transfer, output-changing state guidance, and exact token-by-checkpoint coordinate field."}
    for path in (REGISTRY, CATALOG):
        if not path.exists():
            continue
        data = load(path)
        container = data.setdefault("datasets", []) if isinstance(data, dict) else data
        if not any(item.get("id") == entry["id"] for item in container):
            container.append(entry)
            save(path, data)


def c598() -> None:
    out = begin("C598", {"status": "parameter_atlas_cleanup_synthesis_frozen", "visual": "one exact all-token field with embedding and every HiddenState coordinate plus full-coordinate response panels", "cleanup": "after visual extraction remove undisplayed full-token bulk; retain role tensors, indices, compiled rows and derived metrics", "theory_name": "Conditional Output Field Closure Theory", "organizing_principle": "Reuse-Difference-Conditioning (RDC)"}, {"parent": final("C597")["all_checks_passed"]})
    index = read_rows(capture_index_path())
    compiled = {r["case_id"]: r for r in read_rows(compiled_path())}
    representative = next(r for r in index if r["panel"] == "atomic" and r["family"] == "fact_voice_fixed_query" and r["surface"] == "record" and r["unit"] == 9 and r["variant"] == 0)
    shard = np.load(shard_dir() / representative["shard"], mmap_mode="r")
    exact = np.asarray(shard[representative["shard_index"], :, :representative["length"]], np.float16)
    token_ids = compiled[representative["case_id"]]["prompt_ids"]
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_utils.MODEL_CONFIGS["qwen3"]["path"], trust_remote_code=True, local_files_only=True, use_fast=False)
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    transfer = final("C595")["headline"]
    causal = final("C596")["headline"]
    q14 = final("C597")["headline"]["qwen14"]
    observed = load_npz(OUTS["C595"] / "analysis/fresh_observed_atomic_responses.npz")
    visual_observed = {key: value.tolist() for key, value in observed.items() if key.endswith("|q37")}
    atlas = {"schema": "ai2050.fresh_scope_lockbox_atlas.v1", "phase": 2132, "campaign": "C591-C598", "coordinate_policy": "all signed physical coordinates; no PCA, Top-K or magnitude threshold", "qwen3_4b": {"coordinates": 2560, "checkpoints": ["embedding"] + [f"block_{i:02d}_post" for i in range(36)] + ["final_norm"], "representative": {"case_id": representative["case_id"], "token_ids": token_ids, "tokens": tokens, "shape": list(exact.shape), "states": exact.tolist()}, "fresh_q37_role_response_fields": visual_observed}, "qwen3_14b": {"coordinates": q14.get("coordinates"), "checkpoints": q14.get("checkpoints"), "representative_full_coordinates": q14.get("representative_full_coordinates", {})}, "transfer_summary": transfer["family_summary"], "dynamic_summary": transfer["dynamic_summary"], "composition_summary": transfer["composition_summary"], "causal_summary": causal["direction_summary"], "warnings": ["Exact fields are observations, not named semantic neurons.", "The fresh construction has machine lint only; human blind naturalness was not run.", "Cross-model physical coordinates are not aligned."]}
    save(VISUAL, atlas)
    del shard, exact
    register_visual()
    cleanup_targets = [previous.full_shard_dir(), shard_dir(), previous.OUTS["C586"] / "raw/glm4/full_token_states.float16.npy"]
    q14_raw = q14.get("raw_path")
    if q14_raw:
        cleanup_targets.append(ROOT / q14_raw)
    removed = []
    for target in cleanup_targets:
        resolved = target.resolve()
        if not str(resolved).lower().startswith(str(ROOT.resolve()).lower()):
            raise RuntimeError(f"cleanup target escaped workspace: {resolved}")
        if target.is_dir():
            size = sum(p.stat().st_size for p in target.rglob("*") if p.is_file())
            shutil.rmtree(target)
            removed.append({"path": str(target.relative_to(ROOT)), "bytes": size})
        elif target.exists():
            size = target.stat().st_size
            target.unlink()
            removed.append({"path": str(target.relative_to(ROOT)), "bytes": size})
    fresh_candidates = transfer["fresh_atomic_candidates"]
    empirical = {"fresh_atomic_transfer": bool(fresh_candidates), "fresh_dynamic_transfer": transfer["dynamic_summary"]["passed"] > 0, "fresh_composition_transfer": transfer["composition_summary"]["passed"] > 0, "output_changing_guidance": causal["output_changing_sufficiency_candidate"], "qwen14_model_internal": q14.get("functional_candidate", False)}
    headline = {"status": "fresh_lockbox_visual_cleanup_synthesis_closed", "visual": str(VISUAL.relative_to(ROOT)), "visual_bytes": VISUAL.stat().st_size, "exact_parameter_shape": atlas["qwen3_4b"]["representative"]["shape"], "fresh_atomic_candidates": fresh_candidates, "empirical_gates": empirical, "cleanup": removed, "bytes_removed": sum(v["bytes"] for v in removed), "retained_role_mean": capture_path().exists(), "retained_role_last": capture_last_path().exists(), "new_foundational_mathematics_authorized": False, "strict_conclusion": "This fresh lockbox can confirm or reject reusable scope-indexed response laws. Even a broad pass does not establish a unique coordinate circuit or require new foundational mathematics."}
    close("C598", headline, {"visual": VISUAL.exists() and VISUAL.stat().st_size > 0, "embedding_and_hidden": atlas["qwen3_4b"]["representative"]["shape"][0] == 38, "all_coordinates": atlas["qwen3_4b"]["representative"]["shape"][2] == 2560, "role_tensors": capture_path().exists() and capture_last_path().exists(), "finite": finite(headline)}, "C599_independent_audit")


FUNCTIONS = {name: globals()[name.lower()] for name in PHASES}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", choices=list(PHASES), default="C591")
    parser.add_argument("--stop", choices=list(PHASES), default="C598")
    args = parser.parse_args()
    names = list(PHASES)
    start = names.index(args.start)
    stop = names.index(args.stop)
    if stop < start:
        raise SystemExit("--stop precedes --start")
    for name in names[start:stop + 1]:
        print(f"\n=== {name} / Phase {PHASES[name][0]} ===", flush=True)
        FUNCTIONS[name]()


if __name__ == "__main__":
    main()
