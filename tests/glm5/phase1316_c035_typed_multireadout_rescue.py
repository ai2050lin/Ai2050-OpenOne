#!/usr/bin/env python3
"""Phase1316: typed multi-readout rescue after the frozen C035 multisite cut."""
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

PHASE = 1316
CAMPAIGN = "C035"
SCRIPT = Path(__file__).resolve()
AUDITOR = T / "phase1316_c035_typed_multireadout_rescue_audit.py"
PARENT = T / "result/phase1315_c035_multisite_position_cut"
CONTRACT = T / "result/phase1313_c035_semantic_position_cut_contract"
MATERIAL = CONTRACT / "material/frozen_position_cut_pairs.jsonl"
OUT = T / "result/phase1316_c035_typed_multireadout_rescue"
PROTOCOL = OUT / "protocol/preregistration.json"
MANIFEST = OUT / "protocol/frozen_typed_rescue_manifest.jsonl"
PRE = OUT / "audit/independent_preaudit.json"
POST = OUT / "audit/independent_final_audit.json"
ARRAYS = OUT / "raw/typed_rescue_arrays.npz"
META = OUT / "raw/run_metadata.json"
SUMMARY = OUT / "analysis/typed_rescue_summary.json"
FINAL = OUT / "analysis/final.json"
COMPLETE = OUT / "protocol/formal_run_complete.json"

PARTITIONS = ("confirmation", "holdout")
ATTRS = ("temperature", "texture", "origin", "condition", "category", "priority")
SURFACES = ("registry_prose", "registry_ledger")
ROLES = ("query_attribute", "query_value", "query_end", "record_entities", "record_queried_values")
ARMS = ("baseline", "block_only", "self_retention") + tuple(f"active_{a}" for a in ATTRS) + tuple(f"null_{a}" for a in ATTRS)
BLOCK_DEPTH = 14
RESCUE_DEPTH = 15
EPS = 1e-12


def canonical(v: Any) -> str:
    return json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(v: Any) -> str:
    return hashlib.sha256(canonical(v).encode()).hexdigest()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_rows(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for value in values:
            f.write(canonical(value) + "\n")


def oriented_states(pair: dict[str, Any], identities: list[int]) -> list[dict[str, Any]]:
    indexed = {state["gold_position"]: state for state in pair["states"]}
    if set(indexed) != set(identities):
        raise RuntimeError("donor identity transition cannot be aligned")
    return [indexed[identities[0]], indexed[identities[1]]]


def build_manifest() -> list[dict[str, Any]]:
    index = {(x["partition"], x["profile_index"], x["attribute"], x["surface"], x["panel"]): x
             for x in rows(MATERIAL)}
    result = []
    for partition in PARTITIONS:
        for profile in range(6):
            for target_surface in SURFACES:
                donor_surface = SURFACES[1 - SURFACES.index(target_surface)]
                for receiver_attribute in ATTRS:
                    receiver = index[(partition, profile, receiver_attribute, target_surface, "active")]
                    identities = receiver["identity_positions"]
                    active_donors = {}
                    null_donors = {}
                    for donor_attribute in ATTRS:
                        active = index[(partition, profile, donor_attribute, donor_surface, "active")]
                        null = index[(partition, profile, donor_attribute, donor_surface, "matched_null")]
                        active_donors[donor_attribute] = oriented_states(active, identities)
                        null_donors[donor_attribute] = null["states"]
                    result.append({
                        "case_key": f"{partition}|p{profile:02d}|{target_surface}|{receiver_attribute}",
                        "partition": partition, "profile_index": profile, "receiver_attribute": receiver_attribute,
                        "target_surface": target_surface, "donor_surface": donor_surface,
                        "identity_positions": identities, "receiver_states": receiver["states"],
                        "active_donors": active_donors, "null_donors": null_donors,
                    })
    return result


def preregister(force: bool) -> None:
    if load(PARENT / "analysis/final.json").get("authorization") != "phase1316_typed_rescue_only":
        raise RuntimeError("Phase1315 did not authorize typed rescue")
    if not load(PARENT / "audit/independent_final_audit.json").get("all_checks_passed"):
        raise RuntimeError("Phase1315 audit failed")
    if OUT.exists() and not force:
        raise RuntimeError(f"{OUT} already exists")
    if OUT.exists():
        shutil.rmtree(OUT)
    manifest = build_manifest()
    write_rows(MANIFEST, manifest)
    contract = load(CONTRACT / "protocol/preregistration.json")
    timeless = {
        "phase": PHASE, "campaign": CAMPAIGN, "schema_version": "phase1316.c035.typed_multireadout_rescue.v1",
        "model": "qwen3-4b-fp16-cuda-no-quantization", "formal_run_budget": 1,
        "partitions": list(PARTITIONS), "attributes": list(ATTRS), "surfaces": list(SURFACES),
        "block_depth": BLOCK_DEPTH, "rescue_depth": RESCUE_DEPTH, "roles": list(ROLES), "arms": list(ARMS),
        "block": "replace receiver state1 full-registered role residuals at layer14 input with aligned receiver state0 residuals",
        "rescue": "at layer15 input, add role-wise opposite-surface donor state1-minus-state0 residuals aligned to the receiver identity direction",
        "readout_family": "for every receiver attribute, compare all six active donor attributes and all six matched-null donors",
        "thresholds": contract["typed_rescue"]["thresholds"],
        "gate_application": "all thresholds, except global finiteness, must pass separately in confirmation and holdout",
        "manifest": {"sha256": sha(MANIFEST), "case_count": len(manifest)},
        "success_authorization": "close_c035_with_typed_multisite_rescue_candidate",
        "failure_authorization": "close_c035_with_multisite_dependence_without_type_selectivity",
        "hard_stops": [
            "No discovery partition", "No role, depth, donor, arm, threshold, material, or readout change",
            "No component or position search", "No second formal model run", "C035 closes after this phase regardless of verdict",
        ],
        "claim_scope": "typed cross-surface rescue for one frozen Qwen3 and controlled registry task; not minimal mediation or cross-model invariance",
        "dependencies": {"parent_protocol": sha(PARENT / "protocol/preregistration.json"),
                         "parent_manifest": sha(PARENT / "protocol/frozen_cut_manifest.jsonl"),
                         "parent_arrays": sha(PARENT / "raw/cut_arrays.npz"),
                         "parent_final": sha(PARENT / "analysis/final.json"),
                         "parent_audit": sha(PARENT / "audit/independent_final_audit.json"),
                         "contract": sha(CONTRACT / "protocol/preregistration.json"),
                         "material": sha(MATERIAL), "manifest": sha(MANIFEST)},
        "source_hashes": {"main": sha(SCRIPT), "auditor": sha(AUDITOR)},
    }
    save(PROTOCOL, {**timeless, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "protocol_digest": digest(timeless)})
    print(canonical({"cases": len(manifest), "arms": len(ARMS)}))


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


def summarize(margins: np.ndarray, answers: np.ndarray, metadata: list[dict[str, Any]], th: dict[str, float]) -> dict[str, Any]:
    active = margins[:, 3:9]
    null = margins[:, 9:15]
    active_answers = answers[:, 3:9]
    null_answers = answers[:, 9:15]
    attr_index = np.array([ATTRS.index(x["receiver_attribute"]) for x in metadata])
    correct = active[np.arange(len(metadata)), attr_index]
    correct_answers = active_answers[np.arange(len(metadata)), attr_index]
    correct_gain = correct - margins[:, 1]
    denominator = margins[:, 0] - margins[:, 1]
    recovery = correct_gain / np.where(np.abs(denominator) > EPS, denominator, np.nan)
    wrong_gain_rows = []
    wrong_answer_rows = []
    for i, ai in enumerate(attr_index):
        mask = np.arange(len(ATTRS)) != ai
        wrong_gain_rows.append(active[i, mask] - margins[i, 1])
        wrong_answer_rows.append(active_answers[i, mask])
    wrong_gain = np.stack(wrong_gain_rows)
    wrong_answers = np.stack(wrong_answer_rows)
    null_gain = null - margins[:, 1, None]
    partitions = {}
    gates = {"finite": bool(np.isfinite(margins).all())}
    for partition in PARTITIONS:
        idx = np.array([i for i, x in enumerate(metadata) if x["partition"] == partition])
        cell = {
            "baseline_accuracy": float(np.mean(answers[idx, 0])),
            "block_accuracy": float(np.mean(answers[idx, 1])),
            "self_retention": float(np.mean(answers[idx, 2])),
            "correct_rescue_accuracy": float(np.mean(correct_answers[idx])),
            "correct_recovery_fraction_median": float(np.nanmedian(recovery[idx])),
            "valid_recovery_fraction": float(np.mean(np.isfinite(recovery[idx]))),
            "own_attribute_win_fraction": float(np.mean(correct_gain[idx] > np.max(wrong_gain[idx], axis=1))),
            "wrong_attribute_exclusion_fraction": float(1.0 - np.mean(wrong_answers[idx])),
            "null_exclusion_fraction": float(1.0 - np.mean(null_answers[idx])),
            "correct_gain_median": float(np.median(correct_gain[idx])),
            "max_wrong_gain_median": float(np.median(np.max(wrong_gain[idx], axis=1))),
            "max_null_gain_median": float(np.median(np.max(null_gain[idx], axis=1))),
        }
        partitions[partition] = cell
        gates[f"{partition}_recovery_defined"] = cell["valid_recovery_fraction"] == 1.0
        gates[f"{partition}_correct_accuracy"] = cell["correct_rescue_accuracy"] >= th["correct_rescue_accuracy_min"]
        gates[f"{partition}_recovery"] = cell["correct_recovery_fraction_median"] >= th["correct_recovery_fraction_median_min"]
        gates[f"{partition}_own_win"] = cell["own_attribute_win_fraction"] >= th["own_attribute_win_fraction_min"]
        gates[f"{partition}_wrong_exclusion"] = cell["wrong_attribute_exclusion_fraction"] >= th["wrong_attribute_exclusion_fraction_min"]
        gates[f"{partition}_null_exclusion"] = cell["null_exclusion_fraction"] >= th["null_exclusion_fraction_min"]
        gates[f"{partition}_self"] = cell["self_retention"] >= th["self_retention_min"]
    metrics = {
        "baseline_accuracy": float(np.mean(answers[:, 0])), "block_accuracy": float(np.mean(answers[:, 1])),
        "self_retention": float(np.mean(answers[:, 2])), "correct_rescue_accuracy": float(np.mean(correct_answers)),
        "correct_recovery_fraction_median": float(np.nanmedian(recovery)),
        "own_attribute_win_fraction": float(np.mean(correct_gain > np.max(wrong_gain, axis=1))),
        "wrong_attribute_exclusion_fraction": float(1.0 - np.mean(wrong_answers)),
        "null_exclusion_fraction": float(1.0 - np.mean(null_answers)),
        "correct_gain_median": float(np.median(correct_gain)),
        "max_wrong_gain_median": float(np.median(np.max(wrong_gain, axis=1))),
        "max_null_gain_median": float(np.median(np.max(null_gain, axis=1))),
    }
    return {"metrics": metrics, "partitions": partitions, "gates": gates, "all_gates_passed": all(gates.values())}


@torch.inference_mode()
def run() -> None:
    protocol, pre = load(PROTOCOL), load(PRE)
    if pre.get("authorization") != "run_phase1316_once" or not pre.get("all_checks_passed"):
        raise RuntimeError("independent preaudit did not authorize run")
    if any(path.exists() for path in (ARRAYS, META, SUMMARY, FINAL, COMPLETE)):
        raise RuntimeError("formal run already consumed")
    manifest = rows(MANIFEST)
    model = None
    started = time.time()
    try:
        model, tokenizer, device, placement = load_fp16("qwen3")
        qa = quantization_audit(model)
        if qa["has_quantized_modules"] or not qa["has_fp16_parameters"]:
            raise RuntimeError(qa)
        all_states = []
        for item in manifest:
            all_states.extend(item["receiver_states"])
            for attr in ATTRS:
                all_states.extend(item["active_donors"][attr])
                all_states.extend(item["null_donors"][attr])
        raw_max = max(len(x["ids"]) for x in all_states)
        pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
        supports = "logits_to_keep" in inspect.signature(model.forward).parameters
        margins = np.empty((len(manifest), len(ARMS)), np.float32)
        answers = np.empty((len(manifest), len(ARMS)), np.bool_)
        block_layer = model.model.layers[BLOCK_DEPTH]
        rescue_layer = model.model.layers[RESCUE_DEPTH]
        for item_index, item in enumerate(manifest):
            references = list(item["receiver_states"])
            for attr in ATTRS:
                references.extend(item["active_donors"][attr])
                references.extend(item["null_donors"][attr])
            ids, mask, pos = make_batch(references, raw_max, pad, device)
            kw = {"input_ids": ids, "attention_mask": mask, "position_ids": pos,
                  "use_cache": False, "output_hidden_states": True, "return_dict": True}
            if supports:
                kw["logits_to_keep"] = 1
            ref_out = model(**kw)
            h14, h15 = ref_out.hidden_states[BLOCK_DEPTH], ref_out.hidden_states[RESCUE_DEPTH]
            receiver0, receiver1 = item["receiver_states"]
            target_positions = sorted({p for role in ROLES for p in receiver1["positions"][role]})
            targets = [receiver1 for _ in ARMS]
            block_rows, block_pos, block_vecs = [], [], []
            rescue_rows, rescue_pos, rescue_vecs = [], [], []
            for arm_index in range(1, len(ARMS)):
                source = h14[1] if arm_index == 2 else h14[0]
                for position in target_positions:
                    block_rows.append(arm_index); block_pos.append(position); block_vecs.append(source[position].clone())
            reference_offset = 2
            for attr_index, attr in enumerate(ATTRS):
                for donor_kind, arm_index, donor_pair_offset in (
                    ("active", 3 + attr_index, reference_offset),
                    ("null", 9 + attr_index, reference_offset + 2),
                ):
                    donor_states = item[f"{donor_kind}_donors"][attr]
                    for role in ROLES:
                        target_role_positions = receiver1["positions"][role]
                        donor0_positions = donor_states[0]["positions"][role]
                        donor1_positions = donor_states[1]["positions"][role]
                        if not (len(target_role_positions) == len(donor0_positions) == len(donor1_positions)):
                            raise RuntimeError("role cardinality drift")
                        for tp, p0, p1 in zip(target_role_positions, donor0_positions, donor1_positions):
                            rescue_rows.append(arm_index); rescue_pos.append(tp)
                            rescue_vecs.append((h15[donor_pair_offset + 1, p1] - h15[donor_pair_offset, p0]).clone())
                reference_offset += 4
            del ref_out
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
            gold = receiver1["gold_position"]
            answer_position = receiver1["positions"]["answer_boundary"][0]
            for arm_index in range(len(ARMS)):
                scores = candidate_scores(model, final_hidden[arm_index, answer_position], receiver1["candidate_ids"])
                nongold = torch.cat((scores[:gold], scores[gold + 1:]))
                margins[item_index, arm_index] = float((scores[gold] - torch.max(nongold)).item())
                answers[item_index, arm_index] = int(torch.argmax(scores).item()) == gold
            del target_out
        metadata = [{k: x[k] for k in ("case_key", "partition", "profile_index", "receiver_attribute",
                                        "target_surface", "donor_surface")} for x in manifest]
        analysis = summarize(margins, answers, metadata, protocol["thresholds"])
        authorization = ("close_c035_with_typed_multisite_rescue_candidate" if analysis["all_gates_passed"]
                         else "close_c035_with_multisite_dependence_without_type_selectivity")
        ARRAYS.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(ARRAYS, gold_minus_max_nongold_margin=margins, gold_correct=answers)
        save(META, {"phase": PHASE, "campaign": CAMPAIGN, "protocol_digest": protocol["protocol_digest"],
                    "array_sha256": sha(ARRAYS), "manifest_sha256": sha(MANIFEST), "case_metadata": metadata,
                    "model_audit": qa, "placement": placement, "runtime_seconds": time.time() - started,
                    "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0})
        save(SUMMARY, {**analysis, "phase": PHASE, "campaign": CAMPAIGN, "authorization": authorization})
        save(FINAL, {"phase": PHASE, "campaign": CAMPAIGN,
                     "verdict": "typed_multisite_rescue_qualified" if analysis["all_gates_passed"] else "typed_multisite_rescue_gate_failed",
                     "all_gates_passed": analysis["all_gates_passed"], "authorization": authorization,
                     "c035_closed": True, "protocol_digest": protocol["protocol_digest"]})
        save(COMPLETE, {"completed_at_utc": datetime.now(timezone.utc).isoformat(), "formal_runs_consumed": 1,
                        "protocol_digest": protocol["protocol_digest"]})
        print(canonical({"metrics": analysis["metrics"], "partitions": analysis["partitions"],
                         "authorization": authorization}))
    finally:
        if model is not None:
            release_fp16(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preregister", "run"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    preregister(args.force) if args.command == "preregister" else run()
