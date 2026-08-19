#!/usr/bin/env python3
"""Phase1315: frozen multisite semantic-position cut in Qwen3."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "tests/glm5"
sys.path.insert(0, str(T))
from phase1023_fp16_utils import load_fp16, quantization_audit, release_fp16  # noqa: E402

PHASE = 1315
CAMPAIGN = "C035"
SCRIPT = Path(__file__).resolve()
AUDITOR = T / "phase1315_c035_multisite_position_cut_audit.py"
PARENT = T / "result/phase1314_c035_qwen3_behavior"
CONTRACT = T / "result/phase1313_c035_semantic_position_cut_contract"
MATERIAL = CONTRACT / "material/frozen_position_cut_pairs.jsonl"
OUT = T / "result/phase1315_c035_multisite_position_cut"
PROTOCOL = OUT / "protocol/preregistration.json"
MANIFEST = OUT / "protocol/frozen_cut_manifest.jsonl"
PRE = OUT / "audit/independent_preaudit.json"
POST = OUT / "audit/independent_final_audit.json"
ARRAYS = OUT / "raw/cut_arrays.npz"
META = OUT / "raw/run_metadata.json"
SUMMARY = OUT / "analysis/cut_summary.json"
FINAL = OUT / "analysis/final.json"
COMPLETE = OUT / "protocol/formal_run_complete.json"

PARTITIONS = ("confirmation", "holdout")
ARMS = ("baseline", "query_end_only", "query_bundle", "record_bundle", "full_registered", "self_retention")
ROLE_SETS = {
    "query_end_only": ("query_end",),
    "query_bundle": ("query_attribute", "query_value", "query_end"),
    "record_bundle": ("record_entities", "record_queried_values"),
    "full_registered": ("query_attribute", "query_value", "query_end", "record_entities", "record_queried_values"),
}
DEPTH = 14
BATCH = 2
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


def union_positions(state: dict[str, Any], roles: tuple[str, ...]) -> list[int]:
    return sorted({position for role in roles for position in state["positions"][role]})


def build_manifest() -> list[dict[str, Any]]:
    result = []
    for pair in rows(MATERIAL):
        if pair["partition"] not in PARTITIONS or pair["panel"] != "active":
            continue
        state0, state1 = pair["states"]
        result.append({
            "case_key": pair["pair_key"], "partition": pair["partition"],
            "profile_index": pair["profile_index"], "attribute": pair["attribute"],
            "surface": pair["surface"], "identity_positions": pair["identity_positions"],
            "state0": state0, "state1": state1,
            "position_sets": {name: union_positions(state1, roles) for name, roles in ROLE_SETS.items()},
        })
    return result


def preregister(force: bool) -> None:
    if load(PARENT / "analysis/final.json").get("authorization") != "phase1315_multisite_cut_only":
        raise RuntimeError("Phase1314 did not authorize cut")
    if not load(PARENT / "audit/independent_final_audit.json").get("all_checks_passed"):
        raise RuntimeError("Phase1314 audit failed")
    if OUT.exists() and not force:
        raise RuntimeError(f"{OUT} already exists")
    if OUT.exists():
        shutil.rmtree(OUT)
    manifest = build_manifest()
    write_rows(MANIFEST, manifest)
    contract = load(CONTRACT / "protocol/preregistration.json")
    timeless = {
        "phase": PHASE, "campaign": CAMPAIGN, "schema_version": "phase1315.c035.multisite_cut.v1",
        "model": "qwen3-4b-fp16-cuda-no-quantization", "formal_run_budget": 1,
        "depth": DEPTH, "partitions": list(PARTITIONS), "arms": list(ARMS),
        "role_sets": {key: list(value) for key, value in ROLE_SETS.items()},
        "operation": "at layer14 input, replace target active-state1 residuals at each registered position with aligned active-state0 residuals",
        "margin": "state1 gold candidate score minus maximum non-gold candidate score",
        "thresholds": contract["position_cut"]["thresholds"],
        "manifest": {"sha256": sha(MANIFEST), "case_count": len(manifest),
                     "partition_counts": dict(Counter(x["partition"] for x in manifest))},
        "success_authorization": "phase1316_typed_rescue_only",
        "failure_authorization": "close_c035_at_registered_cut_boundary",
        "claim_scope": "dependence on the frozen layer14 semantic-position set, not minimality, redundancy, or semantic purity",
        "hard_stops": [
            "No discovery partition", "No role, set, depth, arm, threshold, or material change",
            "No head, MLP, neuron, subspace, layer, or window search", "No second formal model run",
            "Failure closes C035 without typed rescue",
        ],
        "dependencies": {"parent_protocol": sha(PARENT / "protocol/preregistration.json"),
                         "parent_final": sha(PARENT / "analysis/final.json"),
                         "parent_audit": sha(PARENT / "audit/independent_final_audit.json"),
                         "contract": sha(CONTRACT / "protocol/preregistration.json"),
                         "material": sha(MATERIAL), "manifest": sha(MANIFEST)},
        "source_hashes": {"main": sha(SCRIPT), "auditor": sha(AUDITOR)},
    }
    save(PROTOCOL, {**timeless, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "protocol_digest": digest(timeless)})
    print(canonical({"cases": len(manifest), "depth": DEPTH}))


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
    drops = margins[:, 0, None] - margins
    partitions = {}
    gates = {"finite": bool(np.isfinite(margins).all())}
    for partition in PARTITIONS:
        idx = [i for i, x in enumerate(metadata) if x["partition"] == partition]
        cell = {
            "baseline_accuracy": float(np.mean(answers[idx, 0])),
            "query_end_accuracy": float(np.mean(answers[idx, 1])),
            "query_bundle_accuracy": float(np.mean(answers[idx, 2])),
            "record_bundle_accuracy": float(np.mean(answers[idx, 3])),
            "full_cut_accuracy": float(np.mean(answers[idx, 4])),
            "self_retention": float(np.mean(answers[idx, 5])),
            "query_end_margin_drop_median": float(np.median(drops[idx, 1])),
            "full_cut_margin_drop_median": float(np.median(drops[idx, 4])),
        }
        cell["full_over_qend_drop_ratio"] = cell["full_cut_margin_drop_median"] / max(
            abs(cell["query_end_margin_drop_median"]), EPS)
        partitions[partition] = cell
        gates[f"{partition}_baseline"] = cell["baseline_accuracy"] >= th["baseline_accuracy_min"]
        gates[f"{partition}_self"] = cell["self_retention"] >= th["self_retention_min"]
        gates[f"{partition}_full_accuracy"] = cell["full_cut_accuracy"] <= th["full_cut_accuracy_max"]
        gates[f"{partition}_full_drop"] = cell["full_cut_margin_drop_median"] >= th["full_cut_margin_drop_median_min"]
        gates[f"{partition}_ratio"] = cell["full_over_qend_drop_ratio"] >= th["full_over_qend_drop_ratio_min"]
    metrics = {
        "baseline_accuracy": float(np.mean(answers[:, 0])),
        "arm_accuracy": {arm: float(np.mean(answers[:, i])) for i, arm in enumerate(ARMS)},
        "margin_median": {arm: float(np.median(margins[:, i])) for i, arm in enumerate(ARMS)},
        "margin_drop_median": {arm: float(np.median(drops[:, i])) for i, arm in enumerate(ARMS)},
    }
    return {"metrics": metrics, "partitions": partitions, "gates": gates, "all_gates_passed": all(gates.values())}


@torch.inference_mode()
def run() -> None:
    protocol, pre = load(PROTOCOL), load(PRE)
    if pre.get("authorization") != "run_phase1315_once" or not pre.get("all_checks_passed"):
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
        raw_max = max(len(x[key]["ids"]) for x in manifest for key in ("state0", "state1"))
        pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
        supports = "logits_to_keep" in inspect.signature(model.forward).parameters
        margins = np.empty((len(manifest), len(ARMS)), np.float32)
        answers = np.empty((len(manifest), len(ARMS)), np.bool_)
        layer = model.model.layers[DEPTH]
        for start in range(0, len(manifest), BATCH):
            group = manifest[start:start + BATCH]
            references = [x[key] for x in group for key in ("state0", "state1")]
            ids, mask, pos = make_batch(references, raw_max, pad, device)
            kw = {"input_ids": ids, "attention_mask": mask, "position_ids": pos,
                  "use_cache": False, "output_hidden_states": True, "return_dict": True}
            if supports:
                kw["logits_to_keep"] = 1
            reference_out = model(**kw)
            h14 = reference_out.hidden_states[DEPTH]
            targets = [x["state1"] for x in group for _ in ARMS]
            replace_rows, replace_pos, replace_vecs = [], [], []
            for local, x in enumerate(group):
                base = local * len(ARMS)
                source0 = h14[2 * local]
                source1 = h14[2 * local + 1]
                for arm_index, arm in enumerate(ARMS):
                    if arm == "baseline":
                        continue
                    positions = x["position_sets"]["full_registered"] if arm == "self_retention" else x["position_sets"][arm]
                    source = source1 if arm == "self_retention" else source0
                    for position in positions:
                        replace_rows.append(base + arm_index)
                        replace_pos.append(position)
                        replace_vecs.append(source[position].clone())
            del reference_out
            tids, tmask, tpos = make_batch(targets, raw_max, pad, device)
            rr = torch.tensor(replace_rows, dtype=torch.long, device=device)
            rp = torch.tensor(replace_pos, dtype=torch.long, device=device)
            rv = torch.stack(replace_vecs)

            def hook(_module: Any, args: tuple[Any, ...]):
                hidden = args[0].clone()
                hidden[rr, rp] = rv
                return (hidden,) + args[1:]

            handle = layer.register_forward_pre_hook(hook)
            try:
                tkw = {"input_ids": tids, "attention_mask": tmask, "position_ids": tpos,
                       "use_cache": False, "output_hidden_states": True, "return_dict": True}
                if supports:
                    tkw["logits_to_keep"] = 1
                target_out = model(**tkw)
            finally:
                handle.remove()
            final_hidden = target_out.hidden_states[-1]
            for local, x in enumerate(group):
                state = x["state1"]
                gold = state["gold_position"]
                answer_position = state["positions"]["answer_boundary"][0]
                for arm_index in range(len(ARMS)):
                    row = local * len(ARMS) + arm_index
                    scores = candidate_scores(model, final_hidden[row, answer_position], state["candidate_ids"])
                    nongold = torch.cat((scores[:gold], scores[gold + 1:]))
                    margins[start + local, arm_index] = float((scores[gold] - torch.max(nongold)).item())
                    answers[start + local, arm_index] = int(torch.argmax(scores).item()) == gold
            del target_out
        metadata = [{k: x[k] for k in ("case_key", "partition", "profile_index", "attribute", "surface")} for x in manifest]
        analysis = summarize(margins, answers, metadata, protocol["thresholds"])
        authorization = "phase1316_typed_rescue_only" if analysis["all_gates_passed"] else "close_c035_at_registered_cut_boundary"
        ARRAYS.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(ARRAYS, gold_minus_max_nongold_margin=margins, gold_correct=answers)
        save(META, {"phase": PHASE, "campaign": CAMPAIGN, "protocol_digest": protocol["protocol_digest"],
                    "array_sha256": sha(ARRAYS), "manifest_sha256": sha(MANIFEST), "case_metadata": metadata,
                    "model_audit": qa, "placement": placement, "runtime_seconds": time.time() - started,
                    "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0})
        save(SUMMARY, {**analysis, "phase": PHASE, "campaign": CAMPAIGN, "authorization": authorization})
        save(FINAL, {"phase": PHASE, "campaign": CAMPAIGN,
                     "verdict": "registered_multisite_cut_qualified" if analysis["all_gates_passed"] else "registered_multisite_cut_gate_failed",
                     "all_gates_passed": analysis["all_gates_passed"], "authorization": authorization,
                     "c035_closed": not analysis["all_gates_passed"], "protocol_digest": protocol["protocol_digest"]})
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
