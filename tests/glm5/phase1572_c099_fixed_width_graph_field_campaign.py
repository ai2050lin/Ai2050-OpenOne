#!/usr/bin/env python3
"""Phase1572 / C099: fixed-width correction and completion of the graph field.

C098 is closed at its preregistered numeric gate.  C099 changes exactly one
execution variable before rerunning the model: every batch has the same global
physical width.  Material, factors, partitions, thresholds and claim scope are
inherited without mutation.
"""
from __future__ import annotations

import argparse
import inspect
import json
import math
import shutil
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
C098 = RESULT / "phase1571_c098_observation_first_graph_campaign"
OUT = RESULT / "phase1572_c099_fixed_width_graph_field_campaign"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
import phase1571_c098_observation_first_graph_campaign as base
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16

PHASE = 1572
CAMPAIGN = "C099"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def c098_failure_diagnostic() -> dict[str, Any]:
    index = core.rows(C098 / "raw/all_token_field_index.jsonl")
    field = np.load(C098 / "raw/all_token_all_state_field.float16.npy", mmap_mode="r")
    by_unit: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in index:
        by_unit[row["unit_id"]].append(row)
    units = []
    global_max = 0.0
    for unit_id, rows in by_unit.items():
        reference = rows[0]
        positions = reference["token_start"] + np.asarray(reference["role_positions"]["target_pre"])
        ref = np.asarray(field[:, positions, :], dtype=np.float32)
        maximum = 0.0
        for row in rows[1:]:
            points = row["token_start"] + np.asarray(row["role_positions"]["target_pre"])
            value = np.asarray(field[:, points, :], dtype=np.float32)
            maximum = max(maximum, float(np.max(np.abs(value - ref))))
        global_max = max(global_max, maximum)
        if maximum > 0:
            units.append({
                "unit_id": unit_id,
                "causal_prefix_max_abs": maximum,
                "physical_widths": sorted({row["token_count"] for row in rows}),
            })
    return {
        "status": "C098_closed_at_preregistered_numeric_gate",
        "failed_gate": "causal_prefix_effect_max_abs <= 1e-6",
        "observed_global_max_abs": global_max,
        "affected_unit_count": len(units),
        "unaffected_unit_count": len(by_unit) - len(units),
        "affected_units": units,
        "diagnosis": "all nonzero units have a one-token physical-width split; role spans decode correctly",
        "hidden_structure_unblinded": False,
    }


def prepare() -> None:
    if (OUT / "protocol/preregistration.json").exists():
        raise RuntimeError("C099 already prepared")
    old = core.load(C098 / "protocol/preregistration.json")
    pre = core.load(C098 / "audit/pre_model_semantic_naturalness_zero_model_audit.json")
    diagnostic = c098_failure_diagnostic()
    if not pre["all_checks_passed"] or diagnostic["affected_unit_count"] != 6:
        raise RuntimeError((pre, diagnostic))
    core.save(C098 / "analysis/capture_failure.json", diagnostic)
    core.save(C098 / "analysis/final.json", {
        "phase": 1571,
        "campaign": "C098",
        "status": "closed_at_preregistered_numeric_execution_gate",
        "failed_gate": diagnostic["failed_gate"],
        "hidden_structure_analyzed": False,
        "authorization": "run_C099_same_material_fixed_global_width",
    })
    for relative in (
        "material/frozen_graph_units.jsonl",
        "material/frozen_cases.jsonl",
        "material/frozen_test_examples.jsonl",
        "compiled/qwen3_active.jsonl",
    ):
        destination = OUT / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(C098 / relative, destination)
    protocol = json.loads(json.dumps(old))
    protocol.update({
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "schema": "c099.fixed_width_observation_first_graph_field.v1",
        "parent_failure": diagnostic,
        "single_changed_variable": "all batches right-padded to frozen global width 210",
        "created_at_utc": now(),
        "authorization": "run_phase1572_fixed_width_capture",
    })
    protocol["execution"]["fixed_global_sequence_length"] = max(
        len(row["prompt_ids"]) for row in core.rows(OUT / "compiled/qwen3_active.jsonl")
    )
    protocol["material"] = {
        **protocol["material"],
        "unit_sha256": core.sha(OUT / "material/frozen_graph_units.jsonl"),
        "case_sha256": core.sha(OUT / "material/frozen_cases.jsonl"),
        "compiled_sha256": core.sha(OUT / "compiled/qwen3_active.jsonl"),
        "identity_to_C098": all(
            core.sha(OUT / relative) == core.sha(C098 / relative)
            for relative in (
                "material/frozen_graph_units.jsonl",
                "material/frozen_cases.jsonl",
                "compiled/qwen3_active.jsonl",
            )
        ),
    }
    protocol.pop("contract_sha256", None)
    protocol.pop("producer_sha256", None)
    protocol["contract_sha256"] = core.digest(protocol)
    protocol["producer_sha256"] = core.sha(Path(__file__))
    checks = {
        "C098_formally_closed": core.load(C098 / "analysis/final.json")["status"] == "closed_at_preregistered_numeric_execution_gate",
        "premodel_inherited": pre["all_checks_passed"],
        "material_identity": protocol["material"]["identity_to_C098"],
        "same_case_count": protocol["material"]["case_count"] == old["material"]["case_count"] == 1152,
        "same_factors": protocol["factors"] == old["factors"],
        "same_partitions": protocol["partitions"] == old["partitions"],
        "same_thresholds": all(
            protocol["execution"][key] == old["execution"][key]
            for key in ("repeat_hidden_max_abs", "repeat_logit_max_abs", "causal_prefix_effect_max_abs", "code_before_visible_effect_max_abs")
        ),
        "single_execution_change": protocol["execution"]["fixed_global_sequence_length"] == 210,
        "no_hidden_analysis_before_freeze": not diagnostic["hidden_structure_unblinded"],
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/pre_model_correction_audit.json", {
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
        "C098_failure": diagnostic,
    })
    print(json.dumps({"checks": checks, "contract_sha256": protocol["contract_sha256"], "authorization": protocol["authorization"]}, indent=2))


def fixed_batch(rows: list[dict[str, Any]], pad: int, device: torch.device, width: int):
    ids = torch.full((len(rows), width), pad, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    lengths = []
    for index, row in enumerate(rows):
        values = torch.tensor(row["prompt_ids"], dtype=torch.long, device=device)
        if len(values) > width:
            raise RuntimeError((row["case_id"], len(values), width))
        ids[index, : len(values)] = values
        mask[index, : len(values)] = 1
        lengths.append(len(values))
    position_ids = mask.cumsum(-1) - 1
    position_ids.masked_fill_(mask == 0, 0)
    return ids, mask, position_ids, lengths


@torch.inference_mode()
def forward(model, rows: list[dict[str, Any]], pad: int, device: torch.device, width: int):
    ids, mask, positions, lengths = fixed_batch(rows, pad, device, width)
    kwargs = {
        "input_ids": ids,
        "attention_mask": mask,
        "position_ids": positions,
        "use_cache": False,
        "output_hidden_states": True,
        "return_dict": True,
    }
    if "logits_to_keep" in inspect.signature(model.forward).parameters:
        kwargs["logits_to_keep"] = 1
    output = model(**kwargs)
    return output, ids, mask, positions, lengths


def capture() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    audit = core.load(OUT / "audit/pre_model_correction_audit.json")
    if protocol["authorization"] != "run_phase1572_fixed_width_capture" or not audit["all_checks_passed"]:
        raise RuntimeError("C099 authorization missing")
    if protocol["producer_sha256"] != core.sha(Path(__file__)):
        raise RuntimeError("C099 producer changed after freeze")
    compiled = core.rows(OUT / "compiled/qwen3_active.jsonl")
    total_tokens = sum(len(row["prompt_ids"]) for row in compiled)
    offsets = []
    cursor = 0
    for row in compiled:
        offsets.append((cursor, cursor + len(row["prompt_ids"])))
        cursor += len(row["prompt_ids"])
    raw_path = OUT / "raw/all_token_all_state_field.float16.npy"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    field = np.lib.format.open_memmap(raw_path, mode="w+", dtype=np.float16, shape=(base.STATES, total_tokens, base.DIM))
    model = None
    index = []
    first_repeat = None
    finite = True
    try:
        model, tok, device, placement = load_bf16("qwen3")
        quant = quantization_audit(model)
        pad = int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        width = protocol["execution"]["fixed_global_sequence_length"]
        for start in range(0, len(compiled), protocol["execution"]["batch_size"]):
            batch = compiled[start:start + protocol["execution"]["batch_size"]]
            output, ids, mask, positions, lengths = forward(model, batch, pad, device, width)
            logits = output.logits[:, -1].float()
            blocks = []
            scores_batch = []
            for local, row in enumerate(batch):
                block = torch.stack([hidden[local, : lengths[local]] for hidden in output.hidden_states], dim=0)
                finite = finite and bool(torch.isfinite(block).all())
                cpu = block.to(dtype=torch.float16, device="cpu").numpy()
                left, right = offsets[start + local]
                field[:, left:right, :] = cpu
                scores = [float(logits[local, candidate[0]]) for candidate in row["candidate_ids"]]
                if start == 0:
                    blocks.append(cpu.copy())
                    scores_batch.append(scores)
                prediction = int(scores[1] > scores[0])
                index.append({
                    "row_index": start + local,
                    **{key: row[key] for key in ("case_id", "unit_id", "family", "world", "unit_index", "partition", "surface", "x", "y", "branch", "code", "codebook", "truth", "output_yes", "gold_position")},
                    "prediction": prediction,
                    "correct": prediction == row["gold_position"],
                    "scores": scores,
                    "token_start": left,
                    "token_end": right,
                    "token_count": right - left,
                    "role_positions": row["role_positions"],
                })
            if start == 0:
                first_repeat = (batch, blocks, scores_batch)
            if (start // protocol["execution"]["batch_size"] + 1) % 24 == 0:
                print(f"[phase1572] captured {start + len(batch)}/{len(compiled)} cases", flush=True)
            del output, ids, mask, positions, logits, blocks
        field.flush()
        if first_repeat is None:
            raise RuntimeError("repeat missing")
        batch, old_blocks, old_scores = first_repeat
        output, ids, mask, positions, lengths = forward(model, batch, pad, device, width)
        repeat_hidden = 0.0
        repeat_logits = 0.0
        logits = output.logits[:, -1].float()
        for local, row in enumerate(batch):
            again = torch.stack([hidden[local, : lengths[local]] for hidden in output.hidden_states], dim=0).to(dtype=torch.float16, device="cpu").numpy()
            repeat_hidden = max(repeat_hidden, float(np.max(np.abs(again.astype(np.float32) - old_blocks[local].astype(np.float32)))))
            for candidate_index, candidate in enumerate(row["candidate_ids"]):
                repeat_logits = max(repeat_logits, abs(float(logits[local, candidate[0]]) - old_scores[local][candidate_index]))
    finally:
        field.flush()
        del field
        if model is not None:
            release_bf16(model)
    core.write_rows(OUT / "raw/all_token_field_index.jsonl", index)
    field = np.load(raw_path, mmap_mode="r")
    by_unit: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in index:
        by_unit[row["unit_id"]].append(row)
    causal_prefix = 0.0
    code_previsible = 0.0
    for rows in by_unit.values():
        reference = rows[0]
        points = reference["token_start"] + np.asarray(reference["role_positions"]["target_pre"])
        ref = np.asarray(field[:, points, :], dtype=np.float32)
        for row in rows[1:]:
            points = row["token_start"] + np.asarray(row["role_positions"]["target_pre"])
            value = np.asarray(field[:, points, :], dtype=np.float32)
            causal_prefix = max(causal_prefix, float(np.max(np.abs(value - ref))))
        for x in (1, -1):
            for y in (1, -1):
                standard = next(row for row in rows if (row["x"], row["y"], row["branch"], row["code"]) == (x, y, 1, 1))
                reversed_code = next(row for row in rows if (row["x"], row["y"], row["branch"], row["code"]) == (x, y, 1, -1))
                for role in ("target_post", "query_target"):
                    left = np.asarray(field[:, standard["token_start"] + np.asarray(standard["role_positions"][role]), :], dtype=np.float32)
                    right = np.asarray(field[:, reversed_code["token_start"] + np.asarray(reversed_code["role_positions"][role]), :], dtype=np.float32)
                    code_previsible = max(code_previsible, float(np.max(np.abs(left - right))))
    checks = {
        "shape": list(field.shape) == [base.STATES, total_tokens, base.DIM],
        "coverage": len(index) == 1152 and index[-1]["token_end"] == total_tokens,
        "finite": finite and all(math.isfinite(value) for row in index for value in row["scores"]),
        "repeat_hidden": repeat_hidden <= protocol["execution"]["repeat_hidden_max_abs"],
        "repeat_logits": repeat_logits <= protocol["execution"]["repeat_logit_max_abs"],
        "causal_prefix": causal_prefix <= protocol["execution"]["causal_prefix_effect_max_abs"],
        "code_previsible": code_previsible <= protocol["execution"]["code_before_visible_effect_max_abs"],
        "fixed_width": protocol["execution"]["fixed_global_sequence_length"] == 210,
        "bf16_nonquantized": quant["has_bf16_parameters"] and not quant["has_quantized_modules"],
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    behavior = {
        "global_accuracy": float(np.mean([row["correct"] for row in index])),
        "global_balanced_accuracy": base.ba([row["output_yes"] for row in index], [row["prediction"] == 0 for row in index]),
        "by_world": {world: float(np.mean([row["correct"] for row in index if row["world"] == world])) for world in base.WORLDS},
        "by_family": {family: float(np.mean([row["correct"] for row in index if row["family"] == family])) for family in base.FAMILIES},
        "by_code": {base.CODEBOOKS[code]["name"]: float(np.mean([row["correct"] for row in index if row["code"] == code])) for code in (1, -1)},
        "by_partition": {partition: float(np.mean([row["correct"] for row in index if row["partition"] == partition])) for partition in base.PARTITIONS},
        "stratum": "descriptive; behavior does not stop C099",
    }
    report = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "status": "fixed_width_all_token_all_state_capture_complete",
        "shape": list(field.shape),
        "total_real_tokens": total_tokens,
        "bytes": raw_path.stat().st_size,
        "raw_sha256": core.sha(raw_path),
        "index_sha256": core.sha(OUT / "raw/all_token_field_index.jsonl"),
        "numeric": {"repeat_hidden_max_abs": repeat_hidden, "repeat_logit_max_abs": repeat_logits, "causal_prefix_max_abs": causal_prefix, "code_previsible_max_abs": code_previsible},
        "behavior": behavior,
        "runtime": {"placement": placement, "quantization": quant},
        "checks": checks,
        "finished_at_utc": now(),
        "authorization": "run_phase1572_analysis",
    }
    core.save(OUT / "analysis/capture_summary.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "runtime"}, ensure_ascii=False, indent=2))


def analyze() -> None:
    report = core.load(OUT / "analysis/capture_summary.json")
    if report["authorization"] != "run_phase1572_analysis" or not all(report["checks"].values()):
        raise RuntimeError("C099 capture authorization missing")
    base.OUT = OUT
    base.PHASE = PHASE
    base.CAMPAIGN = CAMPAIGN
    base.analyze()


def export() -> None:
    base.OUT = OUT
    base.PHASE = PHASE
    base.CAMPAIGN = CAMPAIGN
    summary = core.load(OUT / "analysis/c098_graph_field_summary.json")
    if summary["authorization"] != "export_c098_graph_walsh_heatmap":
        core.save(OUT / "analysis/visualization_decision.json", {"important": False, "reason": "frozen importance threshold not reached"})
        return
    # Reuse the frozen renderer, then rename the canonical/client assets to the
    # corrected campaign identity without changing any measured values.
    base.export_heatmap()
    old_canonical = OUT / "visualization/c098_graph_walsh_heatmap.json"
    old_client = ROOT / "frontend/public/vis_data/research_kernel/c098_graph_walsh_heatmap.json"
    asset = core.load(old_canonical)
    asset.update({"phase": PHASE, "campaign": CAMPAIGN, "title": "C099 Fixed-Width Directed Graph Path Walsh Field"})
    canonical = OUT / "visualization/c099_graph_walsh_heatmap.json"
    client = ROOT / "frontend/public/vis_data/research_kernel/c099_graph_walsh_heatmap.json"
    core.save(canonical, asset)
    client.parent.mkdir(parents=True, exist_ok=True)
    client.write_bytes(canonical.read_bytes())
    if old_canonical.exists():
        old_canonical.unlink()
    if old_client.exists():
        old_client.unlink()
    decision = {
        "important": True,
        "asset": str(canonical.relative_to(ROOT)),
        "client": str(client.relative_to(ROOT)),
        "rows": len(asset["rows"]),
        "coordinates": len(asset["dimensions"]),
        "sha256": core.sha(canonical),
        "client_identity": core.sha(canonical) == core.sha(client),
    }
    core.save(OUT / "analysis/visualization_decision.json", decision)
    print(json.dumps(decision, indent=2))


def finalize() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    capture_report = core.load(OUT / "analysis/capture_summary.json")
    summary = core.load(OUT / "analysis/c098_graph_field_summary.json")
    visualization = core.load(OUT / "analysis/visualization_decision.json")
    checks = {
        "C098_closed": core.load(C098 / "analysis/final.json")["status"] == "closed_at_preregistered_numeric_execution_gate",
        "producer_frozen": protocol["producer_sha256"] == core.sha(Path(__file__)),
        "material_identity": protocol["material"]["identity_to_C098"],
        "capture": all(capture_report["checks"].values()),
        "raw_hash": core.sha(OUT / "raw/all_token_all_state_field.float16.npy") == capture_report["raw_sha256"],
        "walsh_hash": core.sha(OUT / "raw/focus_role_walsh_coefficients.float32.npy") == summary["walsh"]["sha256"],
        "support_count": len(core.rows(OUT / "analysis/discovery_top64_supports.jsonl")) == 3 * 4 * 37 * 4,
        "holdout_count": len(core.rows(OUT / "analysis/dual_holdout_xy_validation.jsonl")) == 3 * 4 * 37 * 4 * 2,
        "design_null": len(core.rows(OUT / "analysis/c097_shared_cell_design_null.jsonl")) == 12,
        "all_token_scan": summary["all_token_scan_rows"] > 0,
        "visualization": (not visualization["important"]) or visualization["client_identity"],
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    final = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "status": "fixed_width_graph_observation_major_stage_complete",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
        "result": summary,
        "visualization": visualization,
        "theory": {
            "name": "conditional output field closure theory",
            "principle": "reuse-difference-conditioning (RDC)",
            "formula": "H_{l+1,t}=T_l(H_{l,<=t};phi,eta); C_S=2^-4 sum_z chi_S(z)H(z)",
            "graph": "embedding identity -> directed local graph -> repeated-target/query code-invariant path response -> code-conditioned boundary response -> output competition",
            "math_status": "finite differences and conditional dynamics suffice for this observation; no new mathematics is established",
        },
        "next_authorization": "C100 observation-first non-transitive composition breadth if C099 yields a repeated path object; otherwise retain C099 as a scoped negative boundary.",
        "finished_at_utc": now(),
    }
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps(final, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "capture", "analyze", "export", "finalize", "all"))
    args = parser.parse_args()
    if args.stage in ("prepare", "all"):
        prepare()
    if args.stage in ("capture", "all"):
        capture()
    if args.stage in ("analyze", "all"):
        analyze()
    if args.stage in ("export", "all"):
        export()
    if args.stage in ("finalize", "all"):
        finalize()


if __name__ == "__main__":
    main()
