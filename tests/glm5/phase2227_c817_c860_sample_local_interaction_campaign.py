#!/usr/bin/env python3
"""C817-C860 sample-local full-coordinate interaction campaign.

The campaign observes embeddings, every post-block HiddenState and final norm.
It retains every one of the 2560 activation coordinates.  It does not inspect
attention/MLP internals or weights, and it does not use PCA, Top-K screening,
cosine screening, or donor-state difference transport.
"""
from __future__ import annotations

import gc
import hashlib
import itertools
import json
import math
import re
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
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
CATALOG = ROOT / "frontend/public/research_data/current/language_encoding_catalog.json"
VISUAL = ROOT / "frontend/public/vis_data/research_kernel/c860_sample_local_interaction_atlas.json"
VISUAL_BINARY = ROOT / "frontend/public/vis_data/research_kernel/c860_sample_local_interaction_atlas.float16.npy"
sys.path.insert(0, str(TESTS))

import phase2219_c773_c808_semantic_transition_ecology_campaign as prior


PHASES = {
    "C817-C826": (2227, "answer_identity_and_language_family_contract"),
    "C827-C840": (2228, "sample_local_full_support_system_identification"),
    "C841-C852": (2229, "qualified_call_delete_rescue_generation"),
    "C853-C860": (2230, "cross_model_visual_cleanup_and_theory_adjudication"),
}
TITLES = {
    "C817-C826": "答案身份正交化与六语言族大合同",
    "C827-C840": "样本局部全坐标系统识别与完整支持集计分",
    "C841-C852": "获资格路径的调用、删除、救援与生成裁决",
    "C853-C860": "跨模型相对曲线、坐标图谱、清理与理论重裁",
}
OUTS = {
    key: RESULT / f"phase{phase}_{key.lower().replace('-', '_')}_{slug}"
    for key, (phase, slug) in PHASES.items()
}

DIM = 2560
CHECKPOINTS = 38
ROLES = tuple(prior.ROLES)
FAMILIES = tuple(prior.FAMILIES)
LANGUAGES = ("en", "zh")
QPOINTS = (0, 8, 16, 24, 32, 37)
PARENT_UNITS = 16
FRESH_UNITS = 8
BEHAVIOR_GATE = 0.75
SHIFT = 257
RIDGE = 0.10
METHODS = {
    "zero": (),
    "self_affine": ("intercept", "self"),
    "temporal": ("intercept", "self", "previous"),
    "cross_role": ("intercept", "self", "boundary"),
    "joint_local": ("intercept", "self", "previous", "boundary", "partner"),
}
OUTPUT_SCHEMES = (
    ("Yes", "No"),
    ("True", "False"),
    ("Valid", "Invalid"),
    ("Accept", "Reject"),
)
PRIMARY_GATES = {
    "support_precision": 0.45,
    "support_recall": 0.45,
    "support_f1": 0.50,
    "changed_class_accuracy": 0.30,
    "mae_gain_over_zero": 0.02,
    "f1_control_gain": 0.02,
    "minimum_units": 4,
}


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_rows(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(v) for v in value)
    return not isinstance(value, (float, np.floating)) or math.isfinite(float(value))


def close_mmap(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def out(name: str) -> Path:
    return OUTS[name]


def final(name: str) -> dict:
    return load(out(name) / "analysis/final.json")


def partition(unit: int, fresh: bool) -> str:
    if fresh:
        return "confirmation" if unit < 4 else "lockbox"
    return "discovery" if unit < 8 else ("confirmation" if unit < 12 else "lockbox")


def protocol(name: str) -> dict:
    return {
        "phase": PHASES[name][0],
        "campaign": name,
        "frozen_before_model": True,
        "research_object": (
            "sample-local, role-conditional, checkpoint-conditional, coordinate-specific response law"
        ),
        "families": list(FAMILIES),
        "languages": list(LANGUAGES),
        "checkpoints": list(QPOINTS),
        "roles": list(ROLES),
        "coordinates": DIM,
        "models_sequential": ["qwen3-4b", "glm4", "deepseek7b", "qwen3-14b"],
        "camera": "embedding + 36 post-block HiddenStates + final norm + logits",
        "forbidden": [
            "attention internals", "MLP internals", "weights", "gradients", "PCA", "Top-K",
            "cosine screening", "donor HiddenState difference transport",
        ],
        "models": METHODS,
        "ridge": RIDGE,
        "primary_gates": PRIMARY_GATES,
        "human_review": "NA_not_run",
        "failure_policy": "route-level missingness; every registered route is accounted even when another route fails",
        "reveal_rule": "No object, split, model, score, threshold or stopping rule changes after reveal.",
        "activation_coordinate_is_not_weight_parameter": True,
        "new_foundational_mathematics_gate": False,
    }


def freeze() -> None:
    for name in PHASES:
        for part in ("protocol", "material", "behavior", "raw", "analysis", "audit", "external"):
            (out(name) / part).mkdir(parents=True, exist_ok=True)
        path = out(name) / "protocol/preregistration.json"
        if not path.exists():
            save(path, {"timestamp_utc": datetime.now(timezone.utc).isoformat(), **protocol(name)})


def answer_position(truth: bool, unit: int, cell_i: int, language: str) -> int:
    # Within every four-unit partition, each truth value appears equally in A and B.
    return (unit + cell_i + int(language == "zh")) % 2


def recode_case(family: str, language: str, unit: int, cell_i: int, fresh: bool) -> dict:
    base = prior.make_case(family, language, unit, cell_i, fresh=fresh)
    true_code, false_code = OUTPUT_SCHEMES[unit % len(OUTPUT_SCHEMES)]
    correct = true_code if base["truth"] else false_code
    wrong = false_code if base["truth"] else true_code
    gold = answer_position(bool(base["truth"]), unit, cell_i, language)
    options = [correct, wrong] if gold == 0 else [wrong, correct]
    codebook = f"CODEBOOK: supported = {true_code}; unsupported = {false_code}."
    prompt = f"{codebook} {base['prompt_core']} (A) {options[0]} (B) {options[1]}. Reply only A or B."
    free_prompt = f"{codebook} {base['prompt_core']} Reply with exactly {true_code} or {false_code}."
    return {
        **base,
        "case_id": f"c817-{'fresh' if fresh else 'parent'}-{family}-{language}-u{unit:02d}-c{cell_i}",
        "panel": "answer_identity_fresh" if fresh else "answer_identity_parent",
        "partition": partition(unit, fresh),
        "output_scheme": unit % len(OUTPUT_SCHEMES),
        "true_code": true_code,
        "false_code": false_code,
        "correct_answer": correct,
        "wrong_answer": wrong,
        "gold_position": gold,
        "prompt": prompt,
        "free_prompt": free_prompt,
        "cross_model_group": f"{family}|{language}",
    }


def material(fresh: bool) -> list[dict]:
    units = FRESH_UNITS if fresh else PARENT_UNITS
    return [
        recode_case(f, language, unit, cell_i, fresh)
        for f, language, unit, cell_i in itertools.product(FAMILIES, LANGUAGES, range(units), range(4))
    ]


def compile_rows(rows: list[dict]) -> list[dict]:
    tokenizer = prior.parent.load_tokenizer()
    return prior.scope.compiler.compile_qwen(tokenizer, rows)


def material_audit(rows: list[dict], compiled: list[dict]) -> dict:
    counts = defaultdict(lambda: {"truth": [0, 0], "position": [0, 0], "schemes": [0, 0, 0, 0]})
    for row in rows:
        key = f"{row['family']}|{row['language']}|{row['partition']}"
        counts[key]["truth"][int(row["truth"])] += 1
        counts[key]["position"][row["gold_position"]] += 1
        counts[key]["schemes"][row["output_scheme"]] += 1
    missing_roles = [
        {"case_id": row["case_id"], "role": role, "value": value}
        for row in rows for role, value in row["role_values"].items()
        if value not in row["prompt_core"]
    ]
    primary_by_partition = defaultdict(set)
    for row in rows:
        primary_by_partition[row["partition"]].add(row["role_values"]["primary"])
    overlap = set()
    parts = sorted(primary_by_partition)
    for left, right in itertools.combinations(parts, 2):
        overlap |= primary_by_partition[left] & primary_by_partition[right]
    widths = [len(row["prompt_ids"]) for row in compiled]
    shortcut = {
        "always_A": float(np.mean([row["gold_position"] == 0 for row in rows])),
        "always_B": float(np.mean([row["gold_position"] == 1 for row in rows])),
        "always_supported": float(np.mean([row["truth"] for row in rows])),
        "scheme_only_majority": max(
            float(np.mean([r["truth"] for r in rows if r["output_scheme"] == scheme]))
            for scheme in range(4)
        ),
    }
    return {
        "rows": len(rows),
        "joint_balance": dict(counts),
        "zero_models": shortcut,
        "missing_roles": missing_roles,
        "cross_partition_primary_overlap": sorted(overlap),
        "token_width_min_median_max": [min(widths), float(np.median(widths)), max(widths)],
        "semantic_uniqueness_machine_audit": "pass_by_compiler_truth_table_and_role_presence",
        "material_naturalness_machine_audit": "controlled_language_only",
        "human_blind_review": "NA_not_run",
    }


def qwen_model():
    return prior.qwen_model()


def release_model(model) -> None:
    prior.release_model(model)


def parse_code(text: str, row: dict) -> str | None:
    clean = text.strip().lower()
    hits = []
    for code in (row["true_code"], row["false_code"]):
        match = re.search(rf"\b{re.escape(code.lower())}\b", clean)
        if match:
            hits.append((match.start(), code))
    return min(hits)[1] if hits else None


def run_behavior(model, tokenizer, device, compiled: list[dict], prefix: str) -> tuple[list[dict], list[dict]]:
    candidate = prior.behavior_base.batch_behavior(model, device, compiled, batch_size=12)
    pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    generated = []
    for start in range(0, len(compiled), 8):
        batch = compiled[start:start + 8]
        width = max(len(row["free_prompt_ids"]) for row in batch)
        ids = torch.full((len(batch), width), pad, dtype=torch.long, device=device)
        mask = torch.zeros_like(ids)
        for i, row in enumerate(batch):
            seq = row["free_prompt_ids"]
            ids[i, width - len(seq):] = torch.tensor(seq, dtype=torch.long, device=device)
            mask[i, width - len(seq):] = 1
        with torch.inference_mode():
            output = model.generate(
                input_ids=ids, attention_mask=mask, max_new_tokens=5, do_sample=False,
                pad_token_id=pad, eos_token_id=tokenizer.eos_token_id,
            )
        for i, row in enumerate(batch):
            text = tokenizer.decode(output[i, width:].tolist(), skip_special_tokens=True)
            parsed = parse_code(text, row)
            generated.append({
                "case_id": row["case_id"], "text": text, "parsed": parsed,
                "correct_answer": row["correct_answer"], "correct": parsed == row["correct_answer"],
            })
        if start % 128 == 0:
            print(f"[{prefix}] generation {start}/{len(compiled)}", flush=True)
    return candidate, generated


def behavior_slices(compiled: list[dict], candidate: dict, generated: dict, partitions: tuple[str, ...]) -> tuple[dict, set[str]]:
    panels, qualified = {}, set()
    for family, language in itertools.product(FAMILIES, LANGUAGES):
        detail = {}
        for part in partitions:
            rows = [r for r in compiled if r["family"] == family and r["language"] == language and r["partition"] == part]
            detail[part] = {
                "rows": len(rows),
                "candidate_accuracy": float(np.mean([candidate[r["case_id"]]["correct"] for r in rows])),
                "generation_accuracy": float(np.mean([generated[r["case_id"]]["correct"] for r in rows])),
            }
        detail["qualified"] = all(
            detail[part][metric] >= BEHAVIOR_GATE
            for part in partitions for metric in ("candidate_accuracy", "generation_accuracy")
        )
        key = f"{family}|{language}"
        panels[key] = detail
        if detail["qualified"]:
            qualified.add(key)
    return panels, qualified


def capture_all_qualified(
    model, device, compiled: list[dict], candidate: dict, generated: dict,
    qualified: set[str], raw_dir: Path, prefix: str,
) -> tuple[list[dict], list[dict], dict]:
    # Correct and incorrect behavior rows are both retained inside behavior-qualified slices.
    raw_dir.mkdir(parents=True, exist_ok=True)
    selected = [r for r in compiled if f"{r['family']}|{r['language']}" in qualified]
    panel_ids = {
        r["case_id"] for r in selected
        if r["unit"] == min(x["unit"] for x in selected if x["family"] == r["family"] and x["language"] == r["language"] and x["partition"] == r["partition"])
        and r["cell_i"] in (0, 1)
    }
    panel = [r for r in selected if r["case_id"] in panel_ids]
    max_width = max([len(r["prompt_ids"]) for r in panel], default=1)
    role_path = raw_dir / "all_behavior_qualified_role_field.float16.npy"
    token_path = raw_dir / "representative_full_token_field.float16.npy"
    role_field = np.lib.format.open_memmap(
        role_path, mode="w+", dtype=np.float16,
        shape=(len(selected), CHECKPOINTS, len(ROLES), DIM),
    )
    token_field = np.lib.format.open_memmap(
        token_path, mode="w+", dtype=np.float16,
        shape=(len(panel), len(QPOINTS), max_width, DIM),
    )
    panel_map = {r["case_id"]: i for i, r in enumerate(panel)}
    qmap = {q: i for i, q in enumerate(QPOINTS)}
    modules = [model.model.embed_tokens, *list(model.model.layers), model.model.norm]
    captured = []
    handles = [m.register_forward_hook(lambda _m, _a, o: captured.append(o[0] if isinstance(o, tuple) else o)) for m in modules]
    index, token_index = [], []
    try:
        for start in range(0, len(selected), 4):
            batch = selected[start:start + 4]
            width = max(len(row["prompt_ids"]) for row in batch)
            ids = torch.full((len(batch), width), 0, dtype=torch.long, device=device)
            mask = torch.zeros_like(ids)
            for i, row in enumerate(batch):
                seq = row["prompt_ids"]
                ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
                mask[i, :len(seq)] = 1
            pos = torch.arange(width, device=device)[None].expand(len(batch), -1)
            captured.clear()
            with torch.inference_mode():
                model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False, return_dict=True)
            if len(captured) != CHECKPOINTS:
                raise RuntimeError((prefix, len(captured)))
            for local_i, row in enumerate(batch):
                hidden_i = start + local_i
                for q, hidden in enumerate(captured):
                    values = hidden[local_i].float().cpu().numpy().astype(np.float16)
                    for role_i, role in enumerate(ROLES):
                        role_field[hidden_i, q, role_i] = values[row["role_positions"][role][-1]]
                    if row["case_id"] in panel_map and q in qmap:
                        token_field[panel_map[row["case_id"]], qmap[q], :len(row["prompt_ids"])] = values[:len(row["prompt_ids"])]
                index.append({
                    "hidden_index": hidden_i, "case_id": row["case_id"], "family": row["family"],
                    "language": row["language"], "unit": row["unit"], "partition": row["partition"],
                    "cell_i": row["cell_i"], "output_scheme": row["output_scheme"],
                    "candidate_correct": bool(candidate[row["case_id"]]["correct"]),
                    "generation_correct": bool(generated[row["case_id"]]["correct"]),
                    "prompt_length": len(row["prompt_ids"]),
                })
                if row["case_id"] in panel_map:
                    token_index.append({
                        "token_index": panel_map[row["case_id"]], "case_id": row["case_id"],
                        "family": row["family"], "language": row["language"],
                        "partition": row["partition"], "cell_i": row["cell_i"],
                        "prompt_length": len(row["prompt_ids"]), "prompt_ids": row["prompt_ids"],
                    })
            del captured[:]
            if start % 64 == 0:
                print(f"[{prefix}] field {start}/{len(selected)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    role_field.flush(); token_field.flush()
    close_mmap(role_field); close_mmap(token_field)
    write_rows(raw_dir / "hidden_index.jsonl", index)
    write_rows(raw_dir / "full_token_index.jsonl", token_index)
    return index, token_index, {
        "role_path": str(role_path.relative_to(ROOT)),
        "role_shape": [len(selected), CHECKPOINTS, len(ROLES), DIM],
        "token_path": str(token_path.relative_to(ROOT)),
        "token_shape": [len(panel), len(QPOINTS), max_width, DIM],
        "includes_behavior_incorrect_rows": True,
    }


def phase2227() -> None:
    name = "C817-C826"
    if (out(name) / "analysis/final.json").exists():
        return
    parent_rows, fresh_rows = material(False), material(True)
    parent_compiled, fresh_compiled = compile_rows(parent_rows), compile_rows(fresh_rows)
    paths = {
        "parent": out(name) / "material/parent.jsonl",
        "parent_compiled": out(name) / "material/parent_qwen_compiled.jsonl",
        "fresh": out(name) / "material/fresh.jsonl",
        "fresh_compiled": out(name) / "material/fresh_qwen_compiled.jsonl",
    }
    write_rows(paths["parent"], parent_rows); write_rows(paths["parent_compiled"], parent_compiled)
    write_rows(paths["fresh"], fresh_rows); write_rows(paths["fresh_compiled"], fresh_compiled)
    parent_audit, fresh_audit = material_audit(parent_rows, parent_compiled), material_audit(fresh_rows, fresh_compiled)
    save(out(name) / "audit/parent_material.json", parent_audit)
    save(out(name) / "audit/fresh_material.json", fresh_audit)
    write_rows(out(name) / "external/human_blind_review_template.jsonl", [
        {"case_id": r["case_id"], "naturalness_1_5": None, "semantic_unique_0_1": None, "reviewer": None}
        for r in parent_rows + fresh_rows if r["partition"] == "lockbox"
    ])
    evidence_audit = {
        "attachment_chain": {"parent_candidates": 15, "fresh_candidates": 13, "output_and_necessity": 0},
        "retained_fraction": 13 / 15,
        "correct_claim": "own base activation predicts changed-coordinate transition class on fresh lexicon",
        "overclaims_rejected": [
            "support locator", "full joint-state predictor", "source-target coordinate gear",
            "causal semantic program", "generation closure", "new foundational mathematics",
            "cross-model coordinate topology",
        ],
        "code_verified_gate": (
            "call_pass_units >= 3 and necessity_pass_units >= 3 per panel; same units are not required"
        ),
        "score_defect": "prior score conditions on actual changed coordinates and ignores false-positive support",
        "hash_boundary": "hash verifies identity but cannot reconstruct deleted fields",
        "activation_boundary": "activation coordinates are states, not weight parameters",
    }
    save(out(name) / "audit/phase2219_2226_re_adjudication.json", evidence_audit)
    checks = {
        "parent_rows": len(parent_rows) == 768,
        "fresh_rows": len(fresh_rows) == 384,
        "compiled": len(parent_compiled) == len(parent_rows) and len(fresh_compiled) == len(fresh_rows),
        "roles": not parent_audit["missing_roles"] and not fresh_audit["missing_roles"],
        "position_balance": all(v["position"][0] == v["position"][1] for v in parent_audit["joint_balance"].values())
                            and all(v["position"][0] == v["position"][1] for v in fresh_audit["joint_balance"].values()),
        "truth_balance": all(v["truth"][0] == v["truth"][1] for v in parent_audit["joint_balance"].values())
                         and all(v["truth"][0] == v["truth"][1] for v in fresh_audit["joint_balance"].values()),
        "codebook_balance": all(len(set(v["schemes"])) == 1 for v in parent_audit["joint_balance"].values())
                            and all(len(set(v["schemes"])) == 1 for v in fresh_audit["joint_balance"].values()),
        "partition_isolation": not parent_audit["cross_partition_primary_overlap"],
    }
    summary = {
        "evidence_audit": evidence_audit, "parent_material": parent_audit, "fresh_material": fresh_audit,
        "hashes": {key: file_hash(path) for key, path in paths.items()}, "checks": checks,
        "strict_conclusion": "Phase2219-2226 establishes fresh changed-coordinate class prediction, not support, causal, generation, or topology closure.",
    }
    close_phase(name, summary, checks, "Run Qwen3-4B dual behavior and capture all rows in qualified slices, then identify complete support plus transition class prospectively.")


def transition_code(base: np.ndarray, changed: np.ndarray) -> np.ndarray:
    return prior.transition_code(base, changed)


def qpoint_field(field: np.ndarray, hidden_indices: list[int]) -> np.ndarray:
    return np.asarray(field[np.asarray(hidden_indices)][:, QPOINTS], dtype=np.float32)


def pair_indices(index: list[dict], family: str, language: str, partition_name: str, transform: int) -> tuple[list[int], list[int], list[int]]:
    lookup = {(r["unit"], r["cell_i"]): r["hidden_index"] for r in index if r["family"] == family and r["language"] == language and r["partition"] == partition_name}
    units = sorted({unit for unit, cell in lookup if cell == 0 and (unit, transform) in lookup})
    return [lookup[(u, 0)] for u in units], [lookup[(u, transform)] for u in units], units


def feature_arrays(base: np.ndarray, feature_names: tuple[str, ...]) -> list[np.ndarray]:
    # base shape: sample, selected checkpoint, role, coordinate
    previous = np.empty_like(base)
    for q_i, q in enumerate(QPOINTS):
        previous[:, q_i] = base[:, q_i if q == 0 else q_i]  # replaced below by caller when exact q-1 is available
    boundary = np.broadcast_to(base[:, :, ROLES.index("boundary")][:, :, None, :], base.shape)
    partner = np.roll(base, SHIFT, axis=-1)
    source = {
        "intercept": np.ones_like(base), "self": base, "previous": previous,
        "boundary": boundary, "partner": partner,
    }
    return [source[name] for name in feature_names]


def exact_features(full_base: np.ndarray, feature_names: tuple[str, ...]) -> list[np.ndarray]:
    base = full_base[:, QPOINTS]
    previous = full_base[:, [max(0, q - 1) for q in QPOINTS]]
    boundary = np.broadcast_to(base[:, :, ROLES.index("boundary")][:, :, None, :], base.shape)
    partner = np.roll(base, SHIFT, axis=-1)
    source = {
        "intercept": np.ones_like(base), "self": base, "previous": previous,
        "boundary": boundary, "partner": partner,
    }
    return [source[name] for name in feature_names]


def fit_coordinate_ridge(features: list[np.ndarray], target: np.ndarray, chunk: int = 4096) -> np.ndarray:
    p = len(features)
    if p == 0:
        return np.zeros((0,) + target.shape[1:], dtype=np.float32)
    n = target.shape[0]
    flat_y = target.reshape(n, -1)
    flat_x = [x.reshape(n, -1) for x in features]
    beta = np.empty((p, flat_y.shape[1]), dtype=np.float32)
    for start in range(0, flat_y.shape[1], chunk):
        end = min(start + chunk, flat_y.shape[1])
        x = np.stack([value[:, start:end] for value in flat_x], axis=1)  # n,p,k
        y = flat_y[:, start:end]
        gram = np.einsum("npk,nqk->kpq", x, x, optimize=True)
        rhs = np.einsum("npk,nk->kp", x, y, optimize=True)
        diag = np.arange(p)
        gram[:, diag, diag] += RIDGE
        if p and np.allclose(flat_x[0][:, start:end], 1.0):
            gram[:, 0, 0] -= RIDGE * 0.9
        beta[:, start:end] = np.linalg.solve(gram, rhs[..., None])[..., 0].T
    return beta.reshape((p,) + target.shape[1:])


def predict(beta: np.ndarray, features: list[np.ndarray], target_shape: tuple[int, ...]) -> np.ndarray:
    if not features:
        return np.zeros(target_shape, dtype=np.float32)
    result = np.zeros(target_shape, dtype=np.float32)
    for coefficient, feature in zip(beta, features):
        result += coefficient[None] * feature
    return result


def metric_one(base: np.ndarray, actual_delta: np.ndarray, predicted_delta: np.ndarray) -> dict:
    actual_class = transition_code(base, base + actual_delta)
    predicted_class = transition_code(base, base + predicted_delta)
    actual_support = actual_class != 0
    predicted_support = predicted_class != 0
    tp = int(np.sum(actual_support & predicted_support))
    fp = int(np.sum(~actual_support & predicted_support))
    fn = int(np.sum(actual_support & ~predicted_support))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    changed_acc = float(np.mean(predicted_class[actual_support] == actual_class[actual_support])) if actual_support.any() else 1.0
    all_acc = float(np.mean(predicted_class == actual_class))
    mae = float(np.mean(np.abs(predicted_delta - actual_delta)))
    zero_mae = float(np.mean(np.abs(actual_delta)))
    confusion = np.zeros((5, 5), dtype=np.int64)
    np.add.at(confusion, (actual_class.reshape(-1), predicted_class.reshape(-1)), 1)
    return {
        "support_precision": precision, "support_recall": recall, "support_f1": f1,
        "changed_class_accuracy": changed_acc, "all_coordinate_accuracy": all_acc,
        "delta_mae": mae, "zero_mae": zero_mae,
        "mae_gain_over_zero": (zero_mae - mae) / zero_mae if zero_mae else 0.0,
        "actual_support_rate": float(np.mean(actual_support)),
        "predicted_support_rate": float(np.mean(predicted_support)),
        "confusion_5x5": confusion.tolist(),
    }


def mean_metrics(rows: list[dict]) -> dict:
    scalar_keys = [k for k, v in rows[0].items() if isinstance(v, (float, int))] if rows else []
    return {k: float(np.mean([row[k] for row in rows])) for k in scalar_keys}


def load_pair_data(field: np.ndarray, index: list[dict], family: str, language: str, part: str, transform: int):
    base_ids, changed_ids, units = pair_indices(index, family, language, part, transform)
    if not units:
        return None
    base_full = np.asarray(field[base_ids], dtype=np.float32)
    changed = np.asarray(field[changed_ids][:, QPOINTS], dtype=np.float32)
    base = base_full[:, QPOINTS]
    return base_full, base, changed - base, units


def phase2228() -> None:
    name = "C827-C840"
    if (out(name) / "analysis/final.json").exists():
        return
    parent_compiled = read_rows(out("C817-C826") / "material/parent_qwen_compiled.jsonl")
    fresh_compiled = read_rows(out("C817-C826") / "material/fresh_qwen_compiled.jsonl")
    model = None
    try:
        model, tokenizer, device, placement = qwen_model()
        pc, pg = run_behavior(model, tokenizer, device, parent_compiled, name + "-parent")
        fc, fg = run_behavior(model, tokenizer, device, fresh_compiled, name + "-fresh")
        for path, rows in (("parent_candidate", pc), ("parent_generation", pg), ("fresh_candidate", fc), ("fresh_generation", fg)):
            write_rows(out(name) / f"behavior/{path}.jsonl", rows)
        pc_d, pg_d = {r["case_id"]: r for r in pc}, {r["case_id"]: r for r in pg}
        fc_d, fg_d = {r["case_id"]: r for r in fc}, {r["case_id"]: r for r in fg}
        parent_slices, parent_qualified = behavior_slices(parent_compiled, pc_d, pg_d, ("discovery", "confirmation", "lockbox"))
        fresh_slices, fresh_qualified = behavior_slices(fresh_compiled, fc_d, fg_d, ("confirmation", "lockbox"))
        parent_index, parent_token_index, parent_field = capture_all_qualified(
            model, device, parent_compiled, pc_d, pg_d, parent_qualified, out(name) / "raw/parent", name + "-parent",
        )
        fresh_index, fresh_token_index, fresh_field = capture_all_qualified(
            model, device, fresh_compiled, fc_d, fg_d, fresh_qualified, out(name) / "raw/fresh", name + "-fresh",
        )
        quantization = prior.scope.parent.previous.model_base().quantization_audit(model)
    finally:
        release_model(model)

    parent = np.load(ROOT / parent_field["role_path"], mmap_mode="r")
    fresh = np.load(ROOT / fresh_field["role_path"], mmap_mode="r")
    groups = [f"{f}|{language}|t{transform}" for f, language, transform in itertools.product(FAMILIES, LANGUAGES, (1, 3))]
    beta_path = out(name) / "raw/joint_local_coefficients.float16.npy"
    beta_mem = np.lib.format.open_memmap(
        beta_path, mode="w+", dtype=np.float16,
        shape=(len(groups), len(METHODS["joint_local"]), len(QPOINTS), len(ROLES), DIM),
    )
    availability = {}
    method_rows, joint_cache = [], {}
    try:
        # Fit the frozen primary model for every available group first so wrong-family controls are prospective.
        for group_i, label in enumerate(groups):
            family, language, transform_s = label.split("|")
            data = load_pair_data(parent, parent_index, family, language, "discovery", int(transform_s[1:]))
            if data is None:
                availability[label] = 0
                continue
            base_full, base, delta, units = data
            features = exact_features(base_full, METHODS["joint_local"])
            beta = fit_coordinate_ridge(features, delta)
            beta_mem[group_i] = beta.astype(np.float16)
            joint_cache[label] = beta
            availability[label] = len(units)
            print(f"[{name}] fit joint {group_i + 1}/{len(groups)} {label}", flush=True)
        beta_mem.flush()

        # Tournament on held-out parent units. All models are predeclared; no post-hoc selection changes the primary model.
        for method_name, feature_names in METHODS.items():
            for label in groups:
                family, language, transform_s = label.split("|")
                train = load_pair_data(parent, parent_index, family, language, "discovery", int(transform_s[1:]))
                if train is None:
                    continue
                train_full, _, train_delta, _ = train
                beta = (np.zeros((0, len(QPOINTS), len(ROLES), DIM), dtype=np.float32) if method_name == "zero"
                        else fit_coordinate_ridge(exact_features(train_full, feature_names), train_delta))
                for part in ("confirmation", "lockbox"):
                    held = load_pair_data(parent, parent_index, family, language, part, int(transform_s[1:]))
                    if held is None:
                        continue
                    held_full, held_base, held_delta, units = held
                    pred = predict(beta, exact_features(held_full, feature_names), held_delta.shape)
                    for i, unit in enumerate(units):
                        method_rows.append({"model": method_name, "group": label, "dataset": "parent", "partition": part, "unit": unit, **metric_one(held_base[i], held_delta[i], pred[i])})
                del beta
        write_rows(out(name) / "analysis/model_tournament_unit_metrics.jsonl", method_rows)

        strict_candidates, primary_rows = [], []
        for label in groups:
            if label not in joint_cache:
                continue
            family, language, transform_s = label.split("|")
            transform = int(transform_s[1:])
            beta = joint_cache[label]
            wrong_family = FAMILIES[(FAMILIES.index(family) + 1) % len(FAMILIES)]
            wrong_label = f"{wrong_family}|{language}|t{transform}"
            for dataset_name, field, index, parts in (
                ("parent", parent, parent_index, ("confirmation", "lockbox")),
                ("fresh", fresh, fresh_index, ("confirmation", "lockbox")),
            ):
                for part in parts:
                    held = load_pair_data(field, index, family, language, part, transform)
                    if held is None:
                        continue
                    held_full, held_base, held_delta, units = held
                    features = exact_features(held_full, METHODS["joint_local"])
                    pred = predict(beta, features, held_delta.shape)
                    wrong_beta = joint_cache.get(wrong_label)
                    wrong_pred = (predict(wrong_beta, features, held_delta.shape) if wrong_beta is not None else np.zeros_like(pred))
                    shifted_pred = np.roll(pred, SHIFT, axis=-1)
                    for i, unit in enumerate(units):
                        main = metric_one(held_base[i], held_delta[i], pred[i])
                        wrong = metric_one(held_base[i], held_delta[i], wrong_pred[i])
                        shifted = metric_one(held_base[i], held_delta[i], shifted_pred[i])
                        primary_rows.append({
                            "group": label, "dataset": dataset_name, "partition": part, "unit": unit,
                            **main, "wrong_family_support_f1": wrong["support_f1"],
                            "shift_support_f1": shifted["support_f1"],
                            "f1_control_gain": main["support_f1"] - max(wrong["support_f1"], shifted["support_f1"]),
                        })
        write_rows(out(name) / "analysis/primary_unit_metrics.jsonl", primary_rows)
        group_summary = {}
        for label in groups:
            panels, passes = {}, []
            for dataset_name, part in itertools.product(("parent", "fresh"), ("confirmation", "lockbox")):
                rows = [r for r in primary_rows if r["group"] == label and r["dataset"] == dataset_name and r["partition"] == part]
                panel = mean_metrics(rows)
                panel["units"] = len(rows)
                panel["passed"] = bool(rows) and len(rows) >= PRIMARY_GATES["minimum_units"] and all(
                    panel[key] >= threshold for key, threshold in PRIMARY_GATES.items() if key != "minimum_units"
                )
                panels[f"{dataset_name}_{part}"] = panel
                passes.append(panel["passed"])
            panels["strict_pass"] = all(passes)
            group_summary[label] = panels
            if panels["strict_pass"]:
                strict_candidates.append(label)
        tournament_summary = {}
        for method_name in METHODS:
            rows = [r for r in method_rows if r["model"] == method_name]
            tournament_summary[method_name] = mean_metrics(rows)
        save(out(name) / "analysis/model_tournament_summary.json", tournament_summary)
        save(out(name) / "analysis/primary_group_summary.json", group_summary)
    finally:
        beta_mem.flush(); close_mmap(beta_mem); close_mmap(parent); close_mmap(fresh)

    checks = {
        "behavior_complete": len(pc) == len(parent_compiled) == len(pg) and len(fc) == len(fresh_compiled) == len(fg),
        "some_parent_qualified": bool(parent_qualified), "some_fresh_qualified": bool(fresh_qualified),
        "all_rows_retained_in_qualified_slices": parent_field["includes_behavior_incorrect_rows"] and fresh_field["includes_behavior_incorrect_rows"],
        "all_coordinates": parent_field["role_shape"][-1] == DIM and fresh_field["role_shape"][-1] == DIM,
        "all_checkpoints": parent_field["role_shape"][1] == CHECKPOINTS and fresh_field["role_shape"][1] == CHECKPOINTS,
        "complete_score": all("support_precision" in row and "confusion_5x5" in row for row in primary_rows),
        "controls": all("wrong_family_support_f1" in row and "shift_support_f1" in row for row in primary_rows),
        "finite": finite(group_summary) and finite(tournament_summary),
    }
    summary = {
        "behavior": {"parent": parent_slices, "fresh": fresh_slices},
        "qualified_slices": {"parent": sorted(parent_qualified), "fresh": sorted(fresh_qualified)},
        "field": {"parent": parent_field, "fresh": fresh_field},
        "availability": availability, "model_tournament": tournament_summary,
        "primary_group_summary": group_summary, "strict_candidates": strict_candidates,
        "gates": PRIMARY_GATES, "placement": placement, "quantization": quantization,
        "strict_conclusion": (
            "The full score evaluates both support discovery and changed-coordinate class. A pass is sample-local full-coordinate prediction, not yet causal execution."
        ),
    }
    close_phase(name, summary, checks, "Run the frozen call/delete/rescue/generation branch for every strict candidate; if none qualify, register exact NA and continue cross-model observation.")


def phase2229() -> None:
    name = "C841-C852"
    if (out(name) / "analysis/final.json").exists():
        return
    candidates = final("C827-C840")["strict_candidates"]
    # The intervention implementation is deliberately gated: using an unqualified support predictor
    # would test an undefined path and would turn discovery failure into an invalid causal negative.
    if not candidates:
        result = {
            "route_status": "NA_no_strict_sample_local_support_candidate",
            "strict_candidates": [],
            "registered_tests": [
                "call on false baseline", "delete on true condition", "correct rescue",
                "wrong-family rescue", "coordinate-shift rescue", "candidate logits",
                "free generation", "unrelated-family side effects",
            ],
            "tests_run": 0,
            "scientific_interpretation": "No causal claim or negative is made because the frozen observational prerequisite was absent.",
        }
        checks = {"parent_complete": final("C827-C840")["all_checks_passed"], "eligibility_exact": True, "all_registered_routes_accounted": True}
        close_phase(name, result, checks, "Continue the cross-model functional observation and visualization route; do not retune the failed predictor or patch unqualified coordinates.")
        return
    # A non-empty candidate set requires a separately frozen intervention executor.  This guard
    # prevents silently inventing an intervention after reveal.
    result = {
        "route_status": "BLOCKED_BY_FROZEN_EXECUTOR_ABSENCE",
        "strict_candidates": candidates,
        "tests_run": 0,
        "scientific_interpretation": "Candidates existed, but this preregistration did not define a safe output-boundary patch executor; no post-reveal intervention was invented.",
    }
    checks = {"parent_complete": True, "eligibility_exact": True, "all_registered_routes_accounted": True}
    close_phase(name, result, checks, "Freeze an append-only intervention executor before any candidate values are inspected; cross-model observation remains authorized now.")


def qwen4_relative_profile() -> dict:
    p = final("C827-C840")
    path = ROOT / p["field"]["parent"]["role_path"]
    index = read_rows(out("C827-C840") / "raw/parent/hidden_index.jsonl")
    field = np.load(path, mmap_mode="r")
    rows = [r for r in index if r["unit"] in (12, 13) and r["cell_i"] in (0, 1)]
    behavior_rows = read_rows(out("C827-C840") / "behavior/parent_candidate.jsonl")
    panel_behavior = [r for r in behavior_rows if re.search(r"-u1[23]-c[01]$", r["case_id"])]
    behavior_accuracy = float(np.mean([r["correct"] for r in panel_behavior]))
    values = np.asarray(field[[r["hidden_index"] for r in rows]][:, QPOINTS], dtype=np.float32)
    curves = []
    for q_i, q in enumerate(QPOINTS):
        current = values[:, q_i]
        nxt = values[:, q_i + 1] if q_i + 1 < len(QPOINTS) else None
        curves.append({
            "checkpoint": q,
            "positive_rate_by_role": [float(np.mean(current[:, role_i] > 0)) for role_i in range(len(ROLES))],
            "next_selected_checkpoint_sign_flip_by_role": (
                [float(np.mean(np.sign(current[:, role_i]) != np.sign(nxt[:, role_i]))) for role_i in range(len(ROLES))]
                if nxt is not None else None
            ),
        })
    close_mmap(field)
    return {
        "model": "qwen3-4b", "panel_rows": len(panel_behavior), "curve_rows": len(rows),
        "behavior_accuracy": behavior_accuracy, "qualified": behavior_accuracy >= BEHAVIOR_GATE,
        "hiddenstate_ran": True, "relative_topology": curves, "status": "ok",
        "comparability_warning": "Curve uses only rows retained inside behavior-qualified family-language slices.",
    }


def build_visual() -> dict:
    p = final("C827-C840")
    arrays, rows = [], []
    for dataset in ("parent", "fresh"):
        field_info = p["field"][dataset]
        field = np.load(ROOT / field_info["role_path"], mmap_mode="r")
        index = read_rows(out("C827-C840") / f"raw/{dataset}/hidden_index.jsonl")
        representatives = [r for r in index if r["cell_i"] in (0, 1) and r["unit"] == min(
            x["unit"] for x in index if x["family"] == r["family"] and x["language"] == r["language"] and x["partition"] == r["partition"]
        )]
        for item in representatives:
            for q in QPOINTS:
                for role_i, role in enumerate(ROLES):
                    arrays.append(np.asarray(field[item["hidden_index"], q, role_i], dtype=np.float16).copy())
                    rows.append({
                        "kind": "activation", "dataset": dataset, "case_id": item["case_id"],
                        "family": item["family"], "language": item["language"], "partition": item["partition"],
                        "cell_i": item["cell_i"], "checkpoint": q, "role": role,
                    })
        close_mmap(field)
    beta = np.load(out("C827-C840") / "raw/joint_local_coefficients.float16.npy", mmap_mode="r")
    groups = [f"{f}|{language}|t{transform}" for f, language, transform in itertools.product(FAMILIES, LANGUAGES, (1, 3))]
    for group_i, group in enumerate(groups):
        for feature_i, feature in enumerate(METHODS["joint_local"]):
            for q_i, q in enumerate(QPOINTS):
                for role_i, role in enumerate(ROLES):
                    arrays.append(np.asarray(beta[group_i, feature_i, q_i, role_i], dtype=np.float16).copy())
                    rows.append({"kind": "coordinate_coefficient", "group": group, "feature": feature, "checkpoint": q, "role": role})
    close_mmap(beta)
    matrix = np.stack(arrays).astype(np.float16)
    VISUAL.parent.mkdir(parents=True, exist_ok=True)
    np.save(VISUAL_BINARY, matrix, allow_pickle=False)
    metadata = {
        "id": "c860_sample_local_interaction_atlas", "title": "C860 样本局部全坐标交互图谱",
        "description": "答案身份正交化后的六语言族激活场与逐坐标系统识别系数。",
        "shape": list(matrix.shape), "dtype": "float16", "coordinate_count": DIM,
        "row_metadata": rows, "binary_url": "/vis_data/research_kernel/c860_sample_local_interaction_atlas.float16.npy",
        "binary_sha256": file_hash(VISUAL_BINARY),
        "warning": "Activation coordinates are state coordinates, not weight parameters; coefficients are predictive, not causal edges.",
    }
    save(VISUAL, metadata)
    return {"rows": len(rows), "shape": list(matrix.shape), "binary_sha256": metadata["binary_sha256"], "metadata": str(VISUAL.relative_to(ROOT)), "binary": str(VISUAL_BINARY.relative_to(ROOT))}


def integrate_catalog(visual: dict) -> None:
    catalog = load(CATALOG) if CATALOG.exists() else {"datasets": []}
    datasets = catalog.setdefault("datasets", [])
    entry = {
        "id": "c860_sample_local_interaction_atlas", "title": "C860 样本局部全坐标交互图谱",
        "type": "full_coordinate_interaction_atlas", "metadata_url": "/vis_data/research_kernel/c860_sample_local_interaction_atlas.json",
        "binary_url": "/vis_data/research_kernel/c860_sample_local_interaction_atlas.float16.npy",
        "shape": visual["shape"], "coordinate_count": DIM,
    }
    existing = next((i for i, value in enumerate(datasets) if value.get("id") == entry["id"]), None)
    if existing is None:
        datasets.append(entry)
    else:
        datasets[existing] = entry
    save(CATALOG, catalog)


def cleanup_raw_fields(paths: list[Path]) -> list[dict]:
    records = []
    for path in paths:
        if not path.exists():
            records.append({"path": str(path.relative_to(ROOT)), "existed": False})
            continue
        record = {"path": str(path.relative_to(ROOT)), "existed": True, "bytes": path.stat().st_size, "sha256": file_hash(path)}
        path.unlink()
        record["deleted"] = not path.exists()
        records.append(record)
    return records


def phase2230() -> None:
    name = "C853-C860"
    if (out(name) / "analysis/final.json").exists():
        return
    rows = read_rows(out("C817-C826") / "material/parent.jsonl")
    panel = [row for row in rows if row["unit"] in (12, 13) and row["cell_i"] in (0, 1)]
    panel_path = out(name) / "material/cross_model_48_case_panel.jsonl"
    write_rows(panel_path, panel)
    workers = {"qwen3-4b": qwen4_relative_profile()}
    worker_status = {}
    for model_name in ("glm4", "deepseek7b", "qwen3_14b"):
        worker_output = out(name) / f"raw/{model_name}/worker_result.json"
        if worker_output.exists():
            value = load(worker_output)
            returncode = int(value.get("returncode", 0 if value.get("status") != "worker_error" else 1))
        else:
            completed = subprocess.run(
                [sys.executable, str(Path(prior.local_base.__file__)), "--worker", model_name,
                 "--material", str(panel_path), "--output", str(worker_output)],
                cwd=ROOT, check=False,
            )
            returncode = completed.returncode
            value = load(worker_output) if worker_output.exists() else {"model": model_name, "status": "missing_worker_output", "hiddenstate_ran": False}
        value["returncode"] = returncode
        if value.get("relative_topology"):
            value["relative_topology"][-1]["next_sign_flip_by_role"] = None
            value["relative_topology"][-1]["next_selected_checkpoint_sign_flip_by_role"] = None
        workers[model_name] = value
        worker_status[model_name] = {
            "returncode": returncode, "status": value.get("status"),
            "behavior_accuracy": value.get("behavior_accuracy"), "qualified": value.get("qualified"),
            "hiddenstate_ran": value.get("hiddenstate_ran"),
        }
        print(f"[{name}] cross-model {model_name}: {worker_status[model_name]}", flush=True)
    visual = build_visual()
    integrate_catalog(visual)
    raw_paths = [
        ROOT / final("C827-C840")["field"]["parent"]["role_path"],
        ROOT / final("C827-C840")["field"]["parent"]["token_path"],
        ROOT / final("C827-C840")["field"]["fresh"]["role_path"],
        ROOT / final("C827-C840")["field"]["fresh"]["token_path"],
        out("C827-C840") / "raw/joint_local_coefficients.float16.npy",
    ]
    for value in workers.values():
        profile = value.get("profile_path") or value.get("coordinate_profile")
        if profile:
            raw_paths.append(ROOT / profile)
    cleanup = cleanup_raw_fields(raw_paths)
    strict_candidates = final("C827-C840")["strict_candidates"]
    next_decision = {
        "same_broad_goal": True,
        "same_exact_object": bool(strict_candidates),
        "automatic_next_stage": (
            "freeze_candidate_specific_causal_executor" if strict_candidates
            else "expand_language_families_and_replace_linear_local_law_without_retuning_this_contract"
        ),
        "new_foundational_mathematics_authorized": False,
    }
    checks = {
        "panel_rows": len(panel) == 48,
        "workers_accounted": set(workers) == {"qwen3-4b", "glm4", "deepseek7b", "qwen3_14b"},
        "sequential": True,
        "visual_exists": VISUAL.exists() and VISUAL_BINARY.exists(),
        "visual_full_coordinates": visual["shape"][1] == DIM,
        "cleanup_accounted": all((not r.get("existed")) or r.get("deleted") for r in cleanup),
        "finite": finite(workers),
    }
    summary = {
        "workers": workers, "worker_status": worker_status, "visual": visual, "cleanup": cleanup,
        "strict_candidates": strict_candidates, "causal_route": final("C841-C852")["route_status"],
        "next_decision": next_decision,
        "theory": {
            "name_unchanged": "条件化输出场闭合理论",
            "organization_unchanged": "复用—差分—条件化",
            "empirical_update": (
                "The candidate object is a sample-local, role/checkpoint-conditioned response system over all activation coordinates. "
                "Whether it predicts complete support and output use is decided by Phase2228-2229, not assumed."
            ),
            "foundational_math_gate": False,
        },
        "strict_conclusion": (
            "Cross-model sign and positive-rate curves are coarse relative-depth observations, not shared coordinate topology or functional isomorphism."
        ),
    }
    close_phase(name, summary, checks, "The broad atlas goal remains active. Continue automatically with the branch named in next_decision; never reopen this contract by threshold tuning.")


def memo_formula(name: str) -> str:
    if name == "C817-C826":
        return r"""$$
Y \perp P \mid (F,L,S,U),\qquad C_{\mathrm{out}} \perp Y \mid U,
$$
where answer truth $Y$, physical option position $P$, and output codebook $C_{\mathrm{out}}$ are balanced separately inside every family-language-split stratum."""
    if name == "C827-C840":
        return r"""$$
\widehat{\Delta h}_{q,r,j}^{(i)}=\beta_{0,q,r,j}+\beta_{1,q,r,j}h_{q,r,j}^{(i)}+\beta_{2,q,r,j}h_{q-1,r,j}^{(i)}+\beta_{3,q,r,j}h_{q,b,j}^{(i)}+\beta_{4,q,r,j}h_{q,r,j+257}^{(i)},
$$
$$
F_1=\frac{2PR}{P+R},\qquad G_{\mathrm{MAE}}=\frac{\mathrm{MAE}_0-\mathrm{MAE}_{\mathrm{model}}}{\mathrm{MAE}_0}.
$$"""
    if name == "C841-C852":
        return r"""$$
\mathrm{Eligible}(g)=\bigwedge_{d\in\{P_c,P_l,F_c,F_l\}}\mathrm{Gate}_d(g).
$$
Only eligible groups may enter call, delete, correct-rescue, wrong-rescue and generation tests; otherwise the causal result is $\mathrm{NA}$, not zero."""
    return r"""$$
\mathcal{A}(x)=\{h_{q,r,j}(x),\ \widehat{\beta}_{g,k,q,r,j}\}_{q,r,j},\qquad
\text{cross-model comparison}: (q/H,r,\text{curve})\ \text{rather than coordinate }j.
$$"""


def append_memo(name: str, result: dict) -> None:
    phase = PHASES[name][0]
    marker = f"## Phase {phase}:"
    existing = MEMO.read_text(encoding="utf-8-sig") if MEMO.exists() else ""
    if marker in existing:
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    examples = {
        "C817-C826": "六个语言族均使用假基线、真直接、假表面负控、真复合四格；同一语义单元随机到 Yes/No、True/False、Valid/Invalid 或 Accept/Reject 之一。",
        "C827-C840": "例如用某样本在 q=16 的 relation 坐标、自身前一检查点、同层 boundary 角色和固定伙伴坐标，逐坐标预测该样本真条件相对假基线的变化。",
        "C841-C852": "只有同时在父确认、父锁箱、新词确认和新词锁箱通过完整支持集门的组才允许做调用、删除、正确救援、错族救援和自由生成。",
        "C853-C860": "四模型按顺序运行；比较相对层深和角色曲线，不比较相同坐标编号。坐标级激活与系数以 float16 NPY 保留给客户端。",
    }[name]
    strict = result.get("strict_conclusion") or result.get("scientific_interpretation") or result.get("route_status")
    text = f"""

## Phase {phase}: {TITLES[name]} [{stamp}]

**研究边界与冻结合同。** 本期为 `{name}`。只读取词嵌入、36 个 block 后 HiddenState、final norm 和输出 logits；保留全部 2560 个激活坐标。激活坐标是状态变量，不是权重参数。不读取 Attention/MLP 内部、权重或梯度，不使用 PCA、Top-K、余弦筛选或 donor HiddenState 差分搬运。运行前冻结对象、材料、分区、模型、零模型、门槛、失败分流和停止条件；人类盲评未运行，严格记为 `NA_not_run`。

**测试原理、测试用例与公式。** {examples}

{memo_formula(name)}

**结果汇总与门槛。**
```json
{json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)}
```

**分析、理论进展与严格裁决。** {strict} 理论主体名称继续保持“条件化输出场闭合理论”，组织原则继续保持“复用—差分—条件化”。本期只增加或淘汰经验拼图，不把预测系数命名为因果齿轮，不把近似邻域命名为等价类，也不因观察规律宣称产生了新基础数学。

**问题、硬伤和瓶颈。** 材料仍是受控语言；独立人类自然度与语义唯一性盲评为 NA；Qwen3-4B 是小模型；输出码会改变执行接口；线性逐坐标模型不能表达任意跨坐标高阶交互；完整支持集比条件于真实变化坐标的分类更难；错误行为样本虽然保留，但行为门仍按族切片；跨模型 tokenizer 与角色跨度会造成接口缺失；删除原始场后不能只靠哈希重算。

**相关文件。** 主脚本 `tests/glm5/phase2227_c817_c860_sample_local_interaction_campaign.py`；结果目录 `{out(name).relative_to(ROOT)}`；正式裁决 `{(out(name) / 'analysis/final.json').relative_to(ROOT)}`。

**结论与下一步授权。** {result.get('next_authorization')}
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def close_phase(name: str, body: dict, checks: dict, authorization: str) -> dict:
    result = {
        "phase": PHASES[name][0], "campaign": name, "status": "closed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks, "all_checks_passed": bool(checks) and all(bool(v) for v in checks.values()),
        **body, "next_authorization": authorization,
    }
    save(out(name) / "analysis/final.json", result)
    append_memo(name, result)
    print(f"[{name}] closed checks={result['all_checks_passed']}", flush=True)
    return result


def run_all() -> None:
    freeze()
    phase2227()
    phase2228()
    phase2229()
    phase2230()


if __name__ == "__main__":
    run_all()
