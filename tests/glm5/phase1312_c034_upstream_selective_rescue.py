#!/usr/bin/env python3
"""Phase1312: terminal upstream typed block/rescue experiment for C034."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "tests/glm5"
sys.path.insert(0, str(T))
from phase1023_fp16_utils import load_fp16, quantization_audit, release_fp16  # noqa: E402

PHASE = 1312
CAMPAIGN = "C034"
SCRIPT = Path(__file__).resolve()
AUDITOR = T / "phase1312_c034_upstream_selective_rescue_audit.py"
PARENT = T / "result/phase1311_c034_upstream_type_trajectory"
CONTRACT = T / "result/phase1309_c034_typed_response_camera_contract"
MATERIAL = CONTRACT / "material/frozen_typed_response_pairs.jsonl"
OUT = T / "result/phase1312_c034_upstream_selective_rescue"
PROTOCOL = OUT / "protocol/preregistration.json"
MANIFEST = OUT / "protocol/frozen_rescue_manifest.jsonl"
PRE = OUT / "audit/independent_preaudit.json"
POST = OUT / "audit/independent_final_audit.json"
ARRAYS = OUT / "raw/rescue_arrays.npz"
META = OUT / "raw/run_metadata.json"
SUMMARY = OUT / "analysis/rescue_summary.json"
FINAL = OUT / "analysis/final.json"
COMPLETE = OUT / "protocol/formal_run_complete.json"

PARTITIONS = ("confirmation", "holdout")
ATTRS = ("color", "material", "location", "size", "shape", "status")
SURFACES = ("catalog_prose", "inventory_ledger")
ARMS = ("baseline", "block_only", "correct_cross_surface", "matched_null_cross_surface",
        "wrong_attribute_cross_surface", "self_retention")
ROLE = "query_end"
BATCH = 2
EPS = 1e-12


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_rows(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for value in values:
            f.write(canonical(value) + "\n")


def identity_order(pair: dict[str, Any], identities: list[int]) -> list[dict[str, Any]]:
    result = []
    for identity in identities:
        match = [state for state in pair["states"] if state["gold_position"] == identity]
        if len(match) != 1:
            raise RuntimeError("identity alignment is not unique")
        result.append(match[0])
    return result


def build_manifest() -> list[dict[str, Any]]:
    trajectory = [x for x in rows(PARENT / "protocol/frozen_trajectory_manifest.jsonl") if x["partition"] in PARTITIONS]
    material = rows(MATERIAL)
    lookup = {(x["partition"], x["profile_index"], x["attribute"], x["surface"], x["panel"]): x for x in material}
    result = []
    for x in trajectory:
        null_pair = lookup[(x["partition"], x["profile_index"], x["attribute"], x["opposite_surface"], "matched_null")]
        identities = x["identity_positions"]
        result.append({
            "case_key": x["case_key"], "partition": x["partition"], "profile_index": x["profile_index"],
            "attribute": x["attribute"], "wrong_attribute": x["wrong_attribute"],
            "target_surface": x["anchor_surface"], "donor_surface": x["opposite_surface"],
            "identity_positions": identities,
            "target_state0": x["target_states"][0], "target_state1": x["target_states"][1],
            "correct_state0": x["same_attribute_states"][0], "correct_state1": x["same_attribute_states"][1],
            "wrong_state0": x["wrong_attribute_states"][0], "wrong_state1": x["wrong_attribute_states"][1],
            "null_state0": null_pair["states"][0], "null_state1": null_pair["states"][1],
            "source_keys": {"trajectory": x["case_key"], "null": null_pair["group_id"]},
        })
    return result


def preregister(force: bool) -> None:
    parent_final = load(PARENT / "analysis/final.json")
    parent_audit = load(PARENT / "audit/independent_final_audit.json")
    selected = load(PARENT / "analysis/selected_cell.json")["selected_cell"]
    if parent_final.get("authorization") != "phase1312_upstream_selective_rescue_only" or not parent_audit.get("all_checks_passed"):
        raise RuntimeError("Phase1311 did not authorize Phase1312")
    if selected != {"role": "query_end", "depth": 14, "cell_index": 1,
                    "discovery_metric": selected["discovery_metric"]}:
        raise RuntimeError("selected cell drift")
    if OUT.exists() and not force:
        raise RuntimeError(f"{OUT} already exists")
    if OUT.exists():
        shutil.rmtree(OUT)
    manifest = build_manifest()
    write_rows(MANIFEST, manifest)
    contract = load(CONTRACT / "protocol/preregistration.json")
    block_depth = int(selected["depth"])
    rescue_depth = block_depth + 1
    timeless = {
        "phase": PHASE, "campaign": CAMPAIGN, "schema_version": "phase1312.c034.upstream_typed_rescue.v1",
        "model": "qwen3-4b-fp16-cuda-no-quantization", "formal_run_budget": 1,
        "runtime": {"compiler": "right_padding", "global_fixed_length": True, "case_batch": BATCH},
        "object": "upstream query-end typed block/rescue at the Phase1311 frozen cell",
        "partitions": list(PARTITIONS), "role": ROLE, "block_depth": block_depth,
        "rescue_depth": rescue_depth, "arms": list(ARMS),
        "manifest": {"sha256": sha(MANIFEST), "case_count": len(manifest),
                     "partition_counts": {p: sum(x["partition"] == p for x in manifest) for p in PARTITIONS},
                     "surface_counts": {s: sum(x["target_surface"] == s for x in manifest) for s in SURFACES}},
        "block": "at layer14 query-end replace target identity1 residual by same-target identity0 residual",
        "rescue": "at layer15 query-end add opposite-surface correct-attribute identity1-minus-identity0 residual",
        "controls": {
            "block_only": "no layer15 delta", "matched_null": "opposite-surface matched-null state1-minus-state0 delta",
            "wrong_attribute": "opposite-surface next-attribute identity-aligned delta",
            "self_retention": "replace target identity1 layer14 residual by its own frozen layer14 residual",
        },
        "readout": "target identity1-minus-identity0 candidate margin at answer boundary",
        "aggregation": {
            "partition": "baseline, block, correct-rescue, and retention gates separately in confirmation and holdout",
            "global": "recovery median, correct/null gain ratio, and pairwise correct-over-controls wins over all 192 cases",
        },
        "thresholds": contract["causal"]["thresholds"],
        "success_authorization": "close_c034_with_upstream_typed_rescue_candidate",
        "failure_authorization": "close_c034_at_upstream_rescue_boundary",
        "claim_scope": contract["causal"]["claim_scope"],
        "hard_stops": ["No discovery partition", "No depth, role, donor, wrong-attribute, or threshold change",
                       "No head, MLP, neuron, or subspace search", "No second formal model run",
                       "C034 closes after Phase1312 regardless of verdict"],
        "dependencies": {
            "parent_protocol": sha(PARENT / "protocol/preregistration.json"),
            "parent_manifest": sha(PARENT / "protocol/frozen_trajectory_manifest.jsonl"),
            "parent_arrays": sha(PARENT / "raw/trajectory_arrays.npz"),
            "parent_selected": sha(PARENT / "analysis/selected_cell.json"),
            "parent_final": sha(PARENT / "analysis/final.json"),
            "parent_audit": sha(PARENT / "audit/independent_final_audit.json"),
            "contract": sha(CONTRACT / "protocol/preregistration.json"),
            "material": sha(MATERIAL), "manifest": sha(MANIFEST),
        },
        "source_hashes": {"main": sha(SCRIPT), "auditor": sha(AUDITOR)}, "model_weights_loaded": False,
    }
    protocol = {**timeless, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "protocol_digest": digest(timeless)}
    save(PROTOCOL, protocol)
    print(canonical({"cases": len(manifest), "block": block_depth, "rescue": rescue_depth,
                     "digest": protocol["protocol_digest"]}))


def make_batch(states: list[dict[str, Any]], raw_max: int, pad: int, device: torch.device):
    ids = torch.full((len(states), raw_max), pad, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for i, state in enumerate(states):
        n = len(state["ids"])
        ids[i, :n] = torch.tensor(state["ids"], dtype=torch.long, device=device)
        mask[i, :n] = 1
    return ids, mask, mask.cumsum(-1) - 1


def candidate_scores(model: Any, hidden: torch.Tensor, candidate_ids: list[int]) -> torch.Tensor:
    ids = torch.tensor(candidate_ids, dtype=torch.long, device=hidden.device)
    return model.lm_head.weight[ids] @ model.model.norm(hidden)


def summarize(margins: np.ndarray, answers: np.ndarray, metadata: list[dict[str, Any]], th: dict[str, float]):
    baseline, blocked, correct, null, wrong = [margins[:, i] for i in range(5)]
    correct_gain, null_gain, wrong_gain = correct - blocked, null - blocked, wrong - blocked
    denominator = baseline - blocked
    recovery = correct_gain / np.where(np.abs(denominator) > EPS, denominator, np.nan)
    partitions = {}
    for partition in PARTITIONS:
        idx = [i for i, x in enumerate(metadata) if x["partition"] == partition]
        partitions[partition] = {
            "baseline_accuracy": float(np.mean(answers[idx, 0])),
            "blocked_target_identity_accuracy": float(np.mean(answers[idx, 1])),
            "correct_rescue_accuracy": float(np.mean(answers[idx, 2])),
            "self_retention_accuracy": float(np.mean(answers[idx, 5])),
        }
    metrics = {
        "baseline_accuracy": float(np.mean(answers[:, 0])),
        "blocked_target_identity_accuracy": float(np.mean(answers[:, 1])),
        "correct_rescue_accuracy": float(np.mean(answers[:, 2])),
        "self_retention_accuracy": float(np.mean(answers[:, 5])),
        "correct_gain_median": float(np.median(correct_gain)),
        "null_gain_median": float(np.median(null_gain)),
        "wrong_attribute_gain_median": float(np.median(wrong_gain)),
        "recovery_fraction_median": float(np.nanmedian(recovery)),
        "valid_recovery_fraction": float(np.mean(np.isfinite(recovery))),
        "correct_over_null_margin_ratio": float(np.median(correct_gain)) / max(abs(float(np.median(null_gain))), EPS),
        "pairwise_correct_win_fraction": float(np.mean(correct_gain > np.maximum(null_gain, wrong_gain))),
        "natural_retention": float(np.mean(answers[:, [0, 5]])),
    }
    gates = {
        "finite": bool(np.isfinite(margins).all()),
        "recovery_defined": metrics["valid_recovery_fraction"] == 1.0,
        "recovery": metrics["recovery_fraction_median"] >= th["recovery_fraction_median_min"],
        "null_ratio": metrics["correct_over_null_margin_ratio"] >= th["correct_over_null_margin_ratio_min"],
        "pairwise_correct_win": metrics["pairwise_correct_win_fraction"] >= th["pairwise_correct_win_fraction_min"],
        "natural_retention": metrics["natural_retention"] >= th["natural_retention_min"],
    }
    for partition in PARTITIONS:
        cell = partitions[partition]
        gates[f"{partition}_baseline"] = cell["baseline_accuracy"] >= th["baseline_accuracy_min"]
        gates[f"{partition}_blocked"] = cell["blocked_target_identity_accuracy"] <= th["blocked_target_identity_accuracy_max"]
        gates[f"{partition}_correct_rescue"] = cell["correct_rescue_accuracy"] >= th["correct_rescue_accuracy_min"]
        gates[f"{partition}_retention"] = cell["self_retention_accuracy"] >= th["natural_retention_min"]
    return {"metrics": metrics, "partitions": partitions, "gates": gates, "all_gates_passed": all(gates.values())}


@torch.inference_mode()
def run() -> None:
    protocol, pre = load(PROTOCOL), load(PRE)
    if pre.get("authorization") != "run_phase1312_once" or not pre.get("all_checks_passed"):
        raise RuntimeError("independent preaudit did not authorize the run")
    if any(path.exists() for path in (ARRAYS, META, SUMMARY, FINAL, COMPLETE)):
        raise RuntimeError("formal run budget already consumed")
    manifest = rows(MANIFEST)
    model = None
    started = time.time()
    try:
        model, tokenizer, device, placement = load_fp16("qwen3")
        qa = quantization_audit(model)
        if qa["has_quantized_modules"] or not qa["has_fp16_parameters"]:
            raise RuntimeError(qa)
        keys = ("target_state0", "target_state1", "correct_state0", "correct_state1",
                "null_state0", "null_state1", "wrong_state0", "wrong_state1")
        raw_max = max(len(x[key]["ids"]) for x in manifest for key in keys)
        pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
        supports = "logits_to_keep" in inspect.signature(model.forward).parameters
        margins = np.empty((len(manifest), len(ARMS)), np.float32)
        answers = np.empty((len(manifest), len(ARMS)), np.bool_)
        block_depth, rescue_depth = protocol["block_depth"], protocol["rescue_depth"]
        block_layer, rescue_layer = model.model.layers[block_depth], model.model.layers[rescue_depth]

        for start in range(0, len(manifest), BATCH):
            group = manifest[start:start + BATCH]
            reference = [x[key] for x in group for key in keys]
            ids, mask, pos = make_batch(reference, raw_max, pad, device)
            kw = {"input_ids": ids, "attention_mask": mask, "position_ids": pos,
                  "use_cache": False, "output_hidden_states": True, "return_dict": True}
            if supports:
                kw["logits_to_keep"] = 1
            out = model(**kw)
            blocks, selfs, correct_deltas, null_deltas, wrong_deltas = [], [], [], [], []
            for local, x in enumerate(group):
                offset = 8 * local
                qpos = [reference[offset + j]["positions"][ROLE] for j in range(8)]
                blocks.append(out.hidden_states[block_depth][offset, qpos[0]].clone())
                selfs.append(out.hidden_states[block_depth][offset + 1, qpos[1]].clone())
                h = [out.hidden_states[rescue_depth][offset + j, qpos[j]].clone() for j in range(2, 8)]
                correct_deltas.append(h[1] - h[0])
                null_deltas.append(h[3] - h[2])
                wrong_deltas.append(h[5] - h[4])
            del out

            targets, block_rows, block_pos, block_vecs = [], [], [], []
            rescue_rows, rescue_pos, rescue_vecs = [], [], []
            for local, x in enumerate(group):
                base = len(targets)
                targets.extend([x["target_state1"]] * len(ARMS))
                q = x["target_state1"]["positions"][ROLE]
                for arm in (1, 2, 3, 4):
                    block_rows.append(base + arm); block_pos.append(q); block_vecs.append(blocks[local])
                block_rows.append(base + 5); block_pos.append(q); block_vecs.append(selfs[local])
                for arm, vector in ((2, correct_deltas[local]), (3, null_deltas[local]), (4, wrong_deltas[local])):
                    rescue_rows.append(base + arm); rescue_pos.append(q); rescue_vecs.append(vector)
            tids, tmask, tpos = make_batch(targets, raw_max, pad, device)
            br = torch.tensor(block_rows, dtype=torch.long, device=device)
            bp = torch.tensor(block_pos, dtype=torch.long, device=device)
            bv = torch.stack(block_vecs)
            rr = torch.tensor(rescue_rows, dtype=torch.long, device=device)
            rp = torch.tensor(rescue_pos, dtype=torch.long, device=device)
            rv = torch.stack(rescue_vecs)

            def block_hook(_module: Any, args: tuple[Any, ...]):
                hidden = args[0].clone(); hidden[br, bp] = bv
                return (hidden,) + args[1:]

            def rescue_hook(_module: Any, args: tuple[Any, ...]):
                hidden = args[0].clone(); hidden[rr, rp] = hidden[rr, rp] + rv
                return (hidden,) + args[1:]

            h0 = block_layer.register_forward_pre_hook(block_hook)
            h1 = rescue_layer.register_forward_pre_hook(rescue_hook)
            try:
                tkw = {"input_ids": tids, "attention_mask": tmask, "position_ids": tpos,
                       "use_cache": False, "output_hidden_states": True, "return_dict": True}
                if supports:
                    tkw["logits_to_keep"] = 1
                target_out = model(**tkw)
            finally:
                h1.remove(); h0.remove()
            final_hidden = target_out.hidden_states[-1]
            for local, x in enumerate(group):
                base = len(ARMS) * local
                identity0, identity1 = x["identity_positions"]
                apos = x["target_state1"]["positions"]["answer_boundary"]
                for arm in range(len(ARMS)):
                    scores = candidate_scores(model, final_hidden[base + arm, apos], x["target_state1"]["candidate_ids"])
                    margins[start + local, arm] = float((scores[identity1] - scores[identity0]).float().item())
                    answers[start + local, arm] = int(torch.argmax(scores).item()) == identity1
            del target_out

        metadata = [{k: x[k] for k in ("case_key", "partition", "profile_index", "attribute", "wrong_attribute", "target_surface", "donor_surface")} for x in manifest]
        analysis = summarize(margins, answers, metadata, protocol["thresholds"])
        authorization = ("close_c034_with_upstream_typed_rescue_candidate" if analysis["all_gates_passed"]
                         else "close_c034_at_upstream_rescue_boundary")
        ARRAYS.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(ARRAYS, identity1_minus_identity0_margin=margins, target_identity_correct=answers)
        save(META, {"phase": PHASE, "campaign": CAMPAIGN, "protocol_digest": protocol["protocol_digest"],
                    "array_sha256": sha(ARRAYS), "manifest_sha256": sha(MANIFEST), "case_metadata": metadata,
                    "model_audit": qa, "placement": placement, "runtime_seconds": time.time() - started,
                    "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0})
        save(SUMMARY, {**analysis, "phase": PHASE, "campaign": CAMPAIGN, "authorization": authorization})
        save(FINAL, {"phase": PHASE, "campaign": CAMPAIGN,
                     "verdict": "upstream_typed_rescue_qualified" if analysis["all_gates_passed"] else "upstream_typed_rescue_gate_failed",
                     "all_gates_passed": analysis["all_gates_passed"], "authorization": authorization,
                     "c034_closed": True, "protocol_digest": protocol["protocol_digest"]})
        save(COMPLETE, {"completed_at_utc": datetime.now(timezone.utc).isoformat(), "formal_runs_consumed": 1,
                        "protocol_digest": protocol["protocol_digest"]})
        print(canonical({"metrics": analysis["metrics"], "gates": analysis["gates"], "authorization": authorization}))
    finally:
        if model is not None:
            release_fp16(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preregister", "run"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    preregister(args.force) if args.command == "preregister" else run()
