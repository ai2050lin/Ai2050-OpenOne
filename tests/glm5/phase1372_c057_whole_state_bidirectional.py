#!/usr/bin/env python3
"""Phase1372: frozen whole-Hidden-State sufficiency and necessity for C057."""
from __future__ import annotations

import argparse
import inspect
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16

PHASE, CAMPAIGN = 1372, "C057"
CONTRACT = TESTS / "result/phase1369_c057_independent_relation_campaign_contract"
BEHAVIOR = TESTS / "result/phase1370_c057_qwen_behavior_qualification"
CAMERA = TESTS / "result/phase1371_c057_bidirectional_mediation_camera"
OUT = TESTS / "result/phase1372_c057_whole_state_bidirectional"
MODEL = "qwen3"
DONOR_KEYS = ("clean_true", "corrupt_false", "wrong_identity_true", "status_true")
LAYOUT = (
    "suff_self", "suff_correct", "suff_wrong", "suff_status",
    "necessity_self", "necessity_corrupt", "necessity_wrong", "necessity_status",
)
DONOR_ROWS = (1, 0, 2, 3, 0, 1, 2, 3)


def parents() -> dict:
    final = core.load(CAMERA / "analysis/final.json")
    audit = core.load(CAMERA / "audit/independent_final_audit.json")
    if final.get("authorization") != "run_phase1372_c057_whole_state_bidirectional" or not audit.get("all_checks_passed"):
        raise RuntimeError("Phase1371 did not authorize natural causal reveal")
    return core.load(CONTRACT / "protocol/preregistration.json")


def prepare() -> None:
    protocol = parents()
    target = OUT / "protocol/execution_manifest.json"
    if target.exists():
        raise RuntimeError("Phase1372 manifest already exists")
    pairs = core.rows(BEHAVIOR / "material/eligible_pairs.jsonl")
    if len(pairs) != protocol["material"]["eligible_case_target"]:
        raise RuntimeError("eligible pair count changed")
    manifest = {
        "phase": PHASE, "campaign": CAMPAIGN,
        "contract_sha256": protocol["contract_sha256"],
        "camera_final_sha256": core.sha(CAMERA / "analysis/final.json"),
        "camera_audit_sha256": core.sha(CAMERA / "audit/independent_final_audit.json"),
        "model": MODEL, "precision": "bfloat16-no-quantization",
        "paths": protocol["paths"], "gate": protocol["bidirectional"],
        "layout": list(LAYOUT), "donor_rows": list(DONOR_ROWS),
        "rows_per_case": 20, "case_count": len(pairs),
        "case_ids": [row["pair_id"] for row in pairs],
        "all_paths_run_once": True, "post_reveal_changes_forbidden": True,
        "allowed_observables": protocol["allowed_observables"],
        "forbidden": protocol["forbidden"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    core.save(target, manifest)
    print(json.dumps(manifest, indent=2))


def make_batch(rows: list[dict], pad: int, device: torch.device):
    width = max(len(row["prompt_ids"]) for row in rows)
    ids = torch.full((len(rows), width), int(pad), dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    offsets = []
    for index, row in enumerate(rows):
        value = torch.tensor(row["prompt_ids"], dtype=torch.long, device=device)
        offset = width - len(value)
        offsets.append(offset)
        ids[index, offset:] = value
        mask[index, offset:] = 1
    positions = mask.cumsum(-1) - 1
    positions.masked_fill_(mask == 0, 0)
    return ids, mask, positions, offsets


def copy_role(value, original, target_index: int, target: dict, target_offset: int,
              source_index: int, source: dict, source_offset: int, role: str) -> None:
    target_points = [target_offset + p for p in target["role_positions"][role]]
    source_points = [source_offset + p for p in source["role_positions"][role]]
    if len(target_points) != len(source_points):
        raise RuntimeError("role span mismatch")
    value[target_index, target_points] = original[source_index, source_points]


def role_state(output, layer: int, row_index: int, row: dict, offset: int, role: str) -> torch.Tensor:
    points = [offset + p for p in row["role_positions"][role]]
    return output.hidden_states[layer][row_index, points].float().flatten()


def margin(output, row_index: int, row: dict) -> float:
    logits = output.logits[row_index, -1].float()
    return float(logits[row["candidate_ids"][0][0]] - logits[row["candidate_ids"][1][0]])


def project(value: torch.Tensor, origin: torch.Tensor, target: torch.Tensor) -> float:
    direction = target - origin
    return float(torch.dot(value - origin, direction) / (torch.dot(direction, direction) + 1e-12))


def path_summary(records: list[dict], path_name: str, path: dict, gate: dict) -> tuple[dict, dict, bool]:
    values = [row for row in records if row["path"] == path_name]
    checkpoints, checkpoint_checks = {}, {}
    for cp in path["checkpoints"]:
        key = f'{cp["role"]}@{cp["layer"]}'
        suff_correct = [row["suff_projection"][key]["correct"] for row in values]
        suff_wrong = [row["suff_projection"][key]["wrong"] for row in values]
        suff_status = [row["suff_projection"][key]["status"] for row in values]
        suff_adv = [c - max(w, s) for c, w, s in zip(suff_correct, suff_wrong, suff_status)]
        suff_win = [c > max(w, s) for c, w, s in zip(suff_correct, suff_wrong, suff_status)]
        # The completed formal run serialized this arm as `correct` although
        # the frozen layout names it `necessity_corrupt`.  The row index and
        # intervention are correct; accept the legacy label without rewriting raw data.
        necessity_key = "corrupt" if "corrupt" in values[0]["necessity_projection"][key] else "correct"
        necessity = [row["necessity_projection"][key][necessity_key] for row in values]
        self_l2 = [max(row["self_checkpoint_relative_l2"][key].values()) for row in values]
        metric = {
            "suff_correct_projection_median": statistics.median(suff_correct),
            "suff_advantage_median": statistics.median(suff_adv),
            "suff_win_fraction": sum(suff_win) / len(suff_win),
            "necessity_corrupt_projection_median": statistics.median(necessity),
            "self_relative_l2_max": max(self_l2),
        }
        checkpoints[key] = metric
        checkpoint_checks[key] = {
            "suff_projection": metric["suff_correct_projection_median"] >= gate["suff_projection_median_min"],
            "suff_advantage": metric["suff_advantage_median"] >= gate["suff_control_advantage_median_min"],
            "suff_win": metric["suff_win_fraction"] >= gate["suff_control_win_min"],
            "necessity_projection": metric["necessity_corrupt_projection_median"] >= gate["necessity_projection_median_min"],
            "self": metric["self_relative_l2_max"] <= gate["self_checkpoint_relative_l2_max"],
        }
    suff_correct = [row["suff_output_gain"]["correct"] for row in values]
    suff_wrong = [row["suff_output_gain"]["wrong"] for row in values]
    suff_status = [row["suff_output_gain"]["status"] for row in values]
    suff_adv = [c - max(w, s) for c, w, s in zip(suff_correct, suff_wrong, suff_status)]
    suff_win = [c > max(w, s) for c, w, s in zip(suff_correct, suff_wrong, suff_status)]
    nec_corrupt = [row["necessity_output_damage"]["corrupt"] for row in values]
    nec_status = [row["necessity_output_damage"]["status"] for row in values]
    nec_adv = [c - s for c, s in zip(nec_corrupt, nec_status)]
    nec_win = [c > s for c, s in zip(nec_corrupt, nec_status)]
    output = {
        "suff_correct_gain_median": statistics.median(suff_correct),
        "suff_advantage_median": statistics.median(suff_adv),
        "suff_win_fraction": sum(suff_win) / len(suff_win),
        "necessity_corrupt_damage_median": statistics.median(nec_corrupt),
        "necessity_direction_fraction": sum(v > 0 for v in nec_corrupt) / len(nec_corrupt),
        "necessity_over_status_median": statistics.median(nec_adv),
        "necessity_over_status_win_fraction": sum(nec_win) / len(nec_win),
        "self_max_abs_diff": max(max(abs(row["suff_output_gain"]["self"]),
                                     abs(row["necessity_output_damage"]["self"])) for row in values),
    }
    output_checks = {
        "suff_gain": output["suff_correct_gain_median"] >= gate["suff_output_gain_median_min"],
        "suff_advantage": output["suff_advantage_median"] >= gate["suff_output_advantage_median_min"],
        "suff_win": output["suff_win_fraction"] >= gate["suff_output_win_min"],
        "necessity_damage": output["necessity_corrupt_damage_median"] >= gate["necessity_output_damage_median_min"],
        "necessity_direction": output["necessity_direction_fraction"] >= gate["necessity_direction_fraction_min"],
        "necessity_over_status": output["necessity_over_status_median"] >= gate["necessity_over_status_median_min"],
        "necessity_over_status_win": output["necessity_over_status_win_fraction"] >= gate["necessity_over_status_win_min"],
        "self": output["self_max_abs_diff"] <= gate["self_output_max_abs_diff"],
    }
    qualified = all(all(item.values()) for item in checkpoint_checks.values()) and all(output_checks.values())
    return {"count": len(values), "checkpoints": checkpoints, "output": output}, \
        {"checkpoints": checkpoint_checks, "output": output_checks}, qualified


@torch.inference_mode()
def run() -> None:
    manifest = core.load(OUT / "protocol/execution_manifest.json")
    if (OUT / "analysis/qwen3_bidirectional_summary.json").exists():
        raise RuntimeError("Phase1372 run already exists")
    cases = core.rows(BEHAVIOR / "material/eligible_pairs.jsonl")
    compiled = {row["case_id"]: row for row in core.rows(CONTRACT / "compiled/qwen3_active.jsonl")}
    compiled.update({row["case_id"]: row for row in core.rows(CONTRACT / "compiled/qwen3_status.jsonl")})
    model = None
    try:
        model, tok, device, placement = load_bf16(MODEL)
        quant = quantization_audit(model)
        pad = int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        supports = "logits_to_keep" in inspect.signature(model.forward).parameters
        records, delta_meta, delta_values = [], [], []
        paths = list(manifest["paths"].items())
        for case_index, case in enumerate(cases):
            donors = {key: compiled[case[key]] for key in DONOR_KEYS}
            clean, corrupt = donors["clean_true"], donors["corrupt_false"]
            rows = [donors[key] for key in DONOR_KEYS]
            for _name, _path in paths:
                rows.extend([corrupt] * 4)
                rows.extend([clean] * 4)
            ids, mask, positions, offsets = make_batch(rows, pad, device)
            handles = []
            try:
                for layer in sorted({path["source"]["layer"] for _name, path in paths}):
                    selected = [(i, name, path) for i, (name, path) in enumerate(paths)
                                if path["source"]["layer"] == layer]

                    def hook(_module, args, selected_paths=selected):
                        original = args[0]
                        value = original.clone()
                        for path_index, _name, path in selected_paths:
                            base = 4 + path_index * 8
                            for local, source_index in enumerate(DONOR_ROWS):
                                target_index = base + local
                                copy_role(value, original, target_index, rows[target_index], offsets[target_index],
                                          source_index, rows[source_index], offsets[source_index], path["source"]["role"])
                        return (value,) + args[1:]

                    handles.append(model.model.layers[layer].register_forward_pre_hook(hook))
                kwargs = {"input_ids": ids, "attention_mask": mask, "position_ids": positions,
                          "use_cache": False, "output_hidden_states": True, "return_dict": True}
                if supports:
                    kwargs["logits_to_keep"] = 1
                output = model(**kwargs)
            finally:
                for handle in handles:
                    handle.remove()

            clean_margin, corrupt_margin = margin(output, 0, clean), margin(output, 1, corrupt)
            family3_clean = role_state(output, 3, 0, clean, offsets[0], "family")
            family3_corrupt = role_state(output, 3, 1, corrupt, offsets[1], "family")
            delta_values.append((family3_clean - family3_corrupt).cpu())
            delta_meta.append({key: case[key] for key in
                               ("pair_id", "partition", "surface", "target_family", "wrong_family", "family_pair", "direction")})
            for path_index, (path_name, path) in enumerate(paths):
                base = 4 + path_index * 8
                suff_gain = {name.split("_", 1)[1]: margin(output, base + i, corrupt) - corrupt_margin
                             for i, name in enumerate(LAYOUT[:4])}
                nec_damage = {name.split("_", 1)[1]: clean_margin - margin(output, base + 4 + i, clean)
                              for i, name in enumerate(LAYOUT[4:])}
                suff_projection, nec_projection, self_l2 = {}, {}, {}
                for cp in path["checkpoints"]:
                    key, layer, role = f'{cp["role"]}@{cp["layer"]}', cp["layer"], cp["role"]
                    clean_state = role_state(output, layer, 0, clean, offsets[0], role)
                    corrupt_state = role_state(output, layer, 1, corrupt, offsets[1], role)
                    suff_projection[key] = {}
                    nec_projection[key] = {}
                    for i, label in enumerate(("self", "correct", "wrong", "status")):
                        suff_state = role_state(output, layer, base + i, corrupt, offsets[base + i], role)
                        nec_state = role_state(output, layer, base + 4 + i, clean, offsets[base + 4 + i], role)
                        suff_projection[key][label] = project(suff_state, corrupt_state, clean_state)
                        nec_projection[key][label] = project(nec_state, clean_state, corrupt_state)
                    suff_self = role_state(output, layer, base, corrupt, offsets[base], role)
                    nec_self = role_state(output, layer, base + 4, clean, offsets[base + 4], role)
                    self_l2[key] = {
                        "suff": float((suff_self - corrupt_state).norm() / (corrupt_state.norm() + 1e-12)),
                        "necessity": float((nec_self - clean_state).norm() / (clean_state.norm() + 1e-12)),
                    }
                records.append({
                    **delta_meta[-1], "path": path_name,
                    "clean_margin": clean_margin, "corrupt_margin": corrupt_margin,
                    "suff_output_gain": suff_gain, "necessity_output_damage": nec_damage,
                    "suff_projection": suff_projection, "necessity_projection": nec_projection,
                    "self_checkpoint_relative_l2": self_l2,
                })
            if (case_index + 1) % 24 == 0:
                print(json.dumps({"bidirectional_cases": case_index + 1, "total": len(cases)}), flush=True)
            del output, ids, mask, positions

        core.write_rows(OUT / "raw/qwen3_whole_state_bidirectional.jsonl", records)
        torch.save({"metadata": delta_meta, "family3_clean_minus_corrupt": torch.stack(delta_values)},
                   OUT / "raw/family3_source_deltas.pt")
        metrics, checks, qualified = {}, {}, {}
        for name, path in paths:
            metrics[name], checks[name], qualified[name] = path_summary(records, name, path, manifest["gate"])
        summary = {
            "phase": PHASE, "campaign": CAMPAIGN, "model": MODEL,
            "path_metrics": metrics, "path_checks": checks, "path_qualified": qualified,
            "qualified_paths": [name for name, value in qualified.items() if value],
            "runtime": {"placement": placement, "quantization": quant,
                        "all_finite": all(math.isfinite(v) for row in records
                                          for v in list(row["suff_output_gain"].values()) +
                                          list(row["necessity_output_damage"].values())),
                        "finished_at_utc": datetime.now(timezone.utc).isoformat()},
            "claim_boundary": "Qwen-specific conditional whole-state bidirectional causal role on 288 behavior-correct independent cases",
        }
        core.save(OUT / "analysis/qwen3_bidirectional_summary.json", summary)
        print(json.dumps(summary, indent=2))
    finally:
        if model is not None:
            release_bf16(model)


def finalize() -> None:
    summary = core.load(OUT / "analysis/qwen3_bidirectional_summary.json")
    early = bool(summary["path_qualified"].get("family_early"))
    final = {
        "phase": PHASE, "campaign": CAMPAIGN,
        "path_qualified": summary["path_qualified"],
        "authorization": "run_phase1373_c057_early_path_mediation" if early
                         else "close_c057_without_early_bidirectional_qualification",
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps(final, indent=2))


def analyze_existing() -> None:
    """Aggregate the completed raw run after the postprocessing-only label erratum."""
    manifest = core.load(OUT / "protocol/execution_manifest.json")
    target = OUT / "analysis/qwen3_bidirectional_summary.json"
    if target.exists():
        raise RuntimeError("Phase1372 analysis already exists")
    records = core.rows(OUT / "raw/qwen3_whole_state_bidirectional.jsonl")
    if len(records) != manifest["case_count"] * len(manifest["paths"]):
        raise RuntimeError("formal raw record count mismatch")
    metrics, checks, qualified = {}, {}, {}
    for name, path in manifest["paths"].items():
        metrics[name], checks[name], qualified[name] = path_summary(records, name, path, manifest["gate"])
    erratum = {
        "phase": PHASE, "campaign": CAMPAIGN,
        "kind": "postprocessing_semantic_key_erratum",
        "formal_model_run_repeated": False,
        "raw_artifacts_rewritten": False,
        "frozen_layout": "necessity_corrupt",
        "serialized_projection_key": "correct",
        "resolution": "read serialized `correct` projection as the already-executed necessity_corrupt arm",
        "thresholds_or_routes_changed": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    core.save(OUT / "audit/postprocessing_label_erratum.json", erratum)
    summary = {
        "phase": PHASE, "campaign": CAMPAIGN, "model": MODEL,
        "path_metrics": metrics, "path_checks": checks, "path_qualified": qualified,
        "qualified_paths": [name for name, value in qualified.items() if value],
        "runtime": {"formal_model_run_completed": True, "formal_model_run_repeated": False,
                    "all_finite": all(math.isfinite(v) for row in records
                                      for v in list(row["suff_output_gain"].values()) +
                                      list(row["necessity_output_damage"].values())),
                    "postprocessing_finished_at_utc": datetime.now(timezone.utc).isoformat()},
        "erratum": erratum,
        "claim_boundary": "Qwen-specific conditional whole-state bidirectional causal role on 288 behavior-correct independent cases",
    }
    core.save(target, summary)
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run", "analyze", "finalize"))
    command = parser.parse_args().command
    if command == "prepare":
        prepare()
    elif command == "run":
        run()
    elif command == "analyze":
        analyze_existing()
    else:
        finalize()


if __name__ == "__main__":
    main()
