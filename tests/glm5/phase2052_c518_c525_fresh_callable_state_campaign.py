#!/usr/bin/env python3
"""C518-C525 fresh-vocabulary callable-state campaign.

Only token embeddings and HiddenState checkpoints are observed. Every one of
the 2560 activation coordinates is retained. Attention, MLP activations,
weights, PCA, and Top-K coordinate selection are forbidden.
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
VISUAL = ROOT / "frontend/public/vis_data/research_kernel/c525_fresh_callable_state_atlas.json"
REGISTRY = ROOT / "ai2050_research_os/registry/field_datasets.json"
CATALOG = ROOT / "frontend/public/research_data/current/language_encoding_catalog.json"
sys.path.insert(0, str(TESTS))

import phase2035_c501_c516_embedding_conditioned_state_campaign as previous


PHASES = {
    f"C{campaign}": (2052 + campaign - 518, slug)
    for campaign, slug in (
        (518, "fresh_vocabulary_callable_state_master_contract_and_material"),
        (519, "fresh_material_audit_and_qwen_behavior"),
        (520, "old_fresh_three_family_full_coordinate_capture"),
        (521, "fresh_vocabulary_high_order_panel_replication"),
        (522, "single_sample_role_bundle_transition_tournament"),
        (523, "all_checkpoint_autonomous_role_bundle_rollout"),
        (524, "single_token_intervention_eligibility_adjudication"),
        (525, "visual_cleanup_and_extended_campaign_synthesis"),
    )
}
OUTS = {name: RESULT / f"phase{phase}_{name.lower()}_{slug}" for name, (phase, slug) in PHASES.items()}

DIM = 2560
CHECKPOINTS = 38
ROLES = previous.ROLES
ROLE_INDEX = previous.ROLE_INDEX
FAMILIES = ("nested_composition", "typed_graph_path", "temporal_composition")
CONSTRUCTIONS = previous.CONSTRUCTIONS
BITS = previous.BITS
EDGES = previous.EDGES
QPOINTS = tuple(range(CHECKPOINTS))
RIDGE = 1e-2

PARENT_AUDIT = RESULT / "phase2051_c517_embedding_conditioned_state_campaign_independent_audit/audit/independent_audit.json"
OLD_ROWS_PATH = previous.OLD_CASES
OLD_COMPILED_PATH = previous.OLD_COMPILED

FRESH_UNITS = (
    {"p": "Ulric", "s": "Vessa", "x": "willow", "y": "yarrow", "noise": "astrolabe"},
    {"p": "Weylan", "s": "Xara", "x": "zinnia", "y": "acacia", "noise": "chronometer"},
    {"p": "Yorin", "s": "Zela", "x": "bamboo", "y": "clover", "noise": "spectrometer"},
    {"p": "Arven", "s": "Brena", "x": "dogwood", "y": "eucalyptus", "noise": "micrometer"},
    {"p": "Cedric", "s": "Daria", "x": "fir", "y": "gardenia", "noise": "planimeter"},
    {"p": "Emric", "s": "Freya", "x": "hawthorn", "y": "ivy", "noise": "tachometer"},
    {"p": "Gareth", "s": "Helena", "x": "jasmine", "y": "kudzu", "noise": "anemometer"},
    {"p": "Isen", "s": "Jessa", "x": "laurel", "y": "magnolia", "noise": "pyrometer"},
    {"p": "Kael", "s": "Liora", "x": "nandina", "y": "oleander", "noise": "spherometer"},
    {"p": "Marek", "s": "Nadia", "x": "peony", "y": "rosemary", "noise": "viscometer"},
)


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


def metric_acc() -> dict:
    return previous.metric_acc()


def add_metric(acc: dict, prediction: np.ndarray, truth: np.ndarray) -> None:
    previous.add_metric(acc, prediction, truth)


def finish_metric(acc: dict) -> dict:
    return previous.finish_metric(acc)


def vector_metric(prediction: np.ndarray, truth: np.ndarray) -> dict:
    acc = metric_acc()
    add_metric(acc, prediction, truth)
    return finish_metric(acc)


def partition(unit: int) -> str:
    return previous.partition(unit)


def wrap(construction: str, facts: list[str], question: str, noise: str) -> str:
    return previous.parent.wrap(construction, facts, question, noise)


def options(truth: bool) -> tuple[str, str, int]:
    return previous.parent.options(truth)


def fresh_case(family: str, construction: str, unit: int, bits: tuple[int, int, int, int]) -> dict:
    u = FRESH_UNITS[unit]
    b0, b1, b2, b3 = bits
    if family == "nested_composition":
        outer = u["p"] if b0 == 0 else u["s"]
        inner = u["s"] if b0 == 0 else u["p"]
        relation = "enjoys tasting" if b1 == 0 else "does not enjoy tasting"
        opposite = "does not enjoy tasting" if b1 == 0 else "enjoys tasting"
        obj = u["x"] if b2 == 0 else u["y"]
        query_relation = relation if b3 == 0 else opposite
        facts = [f"{outer} reports that {inner} {relation} {obj}."]
        question = f"Is it true that {outer} reports that {inner} {query_relation} {obj}?"
        roles = {"primary": outer, "secondary": inner, "relation": relation, "context": obj, "query": outer}
        graph = {"operators": ["attitude", "event", "polarity"], "outer": outer, "inner": inner, "object": obj}
    elif family == "typed_graph_path":
        root = u["x"] if b0 == 0 else u["y"]
        mid = f"freshclass{unit}m"
        target = f"freshclass{unit}t"
        facts = ([f"{root} is a member of {target}."] if b1 == 0 else [f"{root} is a member of {mid}.", f"{mid} is a member of {target}."])
        facts.append(f"{root} is also recorded directly as a member of {target}." if b2 else f"{u['noise']} is unrelated to {target}.")
        question = f"Is {root} a member of {target}?" if b3 == 0 else f"Is {target} a member of {root}?"
        relation = "is a member of"
        roles = {"primary": root, "secondary": mid, "relation": relation, "context": target, "query": root if b3 == 0 else target}
        graph = {"operators": ["typed_edge", "path_composition", "shortcut"], "depth": 1 + b1, "shortcut": b2}
    elif family == "temporal_composition":
        first = f"fresh{unit}{'alpha' if b0 == 0 else 'delta'}"
        middle = f"fresh{unit}beta"
        last = f"fresh{unit}{'gamma' if b0 == 0 else 'epsilon'}"
        facts = ([f"{first} happened earlier than {last}.", f"{middle} was independently logged."] if b1 == 0 else [f"{first} happened earlier than {middle}.", f"{middle} happened earlier than {last}."])
        if b2:
            facts = list(reversed(facts))
        question = f"Did {first} happen earlier than {last}?" if b3 == 0 else f"Did {last} happen earlier than {first}?"
        relation = "happened earlier than"
        roles = {"primary": first, "secondary": middle, "relation": relation, "context": last, "query": first if b3 == 0 else last}
        graph = {"operators": ["temporal_edge", "path_composition", "discourse_permutation"], "depth": 1 + b1, "discourse_reversed": b2}
    else:
        raise KeyError(family)
    truth = b3 == 0
    correct, wrong, gold = options(truth)
    core = wrap(construction, facts, question, f"{u['p']} inspected the {u['noise']}")
    code = "".join(str(int(value)) for value in bits)
    return {
        "case_id": f"c518-{family}-{construction}-u{unit}-x{code}",
        "panel": "fresh_callable_state", "family": family,
        "surface": construction, "construction": construction, "unit": unit,
        "bits": list(bits), "cell": code, "partition": partition(unit),
        "gold_position": gold, "correct_answer": correct, "wrong_answer": wrong,
        "prompt_core": core,
        "prompt": f"{core} (A) Yes (B) No. Reply with only A or B.",
        "free_prompt": f"{core} Answer only Yes or No.",
        "role_values": roles,
        "semantic_graph": {"family": family, "bits": list(bits), "truth": truth, **graph},
        "vocabulary_panel": "fresh_C518",
    }


def fresh_material() -> list[dict]:
    return [fresh_case(family, construction, unit, bits) for family, construction, unit, bits in itertools.product(FAMILIES, CONSTRUCTIONS, range(10), BITS)]


def fresh_rows() -> list[dict]:
    return read_rows(OUTS["C518"] / "material/fresh_cases.jsonl")


def combined_rows_compiled() -> tuple[list[dict], list[dict]]:
    old_rows = [row for row in read_rows(OLD_ROWS_PATH) if row["family"] in FAMILIES]
    old_compiled = [row for row in read_rows(OLD_COMPILED_PATH) if row["family"] in FAMILIES]
    return old_rows + fresh_rows(), old_compiled + read_rows(OUTS["C519"] / "compiled/qwen3_fresh.jsonl")


def build_groups(rows: list[dict], index: list[dict]) -> list[dict]:
    hidden = {row["case_id"]: int(row["hidden_index"]) for row in index}
    grouped = defaultdict(dict)
    for row in rows:
        source = "fresh" if row["case_id"].startswith("c518-") else "old"
        grouped[(source, row["family"], row["construction"], int(row["unit"]))][tuple(row["bits"])] = hidden[row["case_id"]]
    result = []
    for key, cells in sorted(grouped.items()):
        if set(cells) != set(BITS):
            raise RuntimeError((key, len(cells)))
        source, family, construction, unit = key
        result.append({"source": source, "family": family, "construction": construction, "unit": unit, "indices": [cells[tuple(bits)] for bits in BITS]})
    return result


def group_cube(states: np.ndarray, groups: list[dict], q: int, role_i: int) -> np.ndarray:
    ids = np.asarray([row["indices"] for row in groups], np.int64)
    return np.asarray(states[ids, q, role_i], np.float32)


def fit_bundle(x: np.ndarray, y: np.ndarray, chunk: int = 128) -> tuple[np.ndarray, np.ndarray]:
    # x/y: samples x roles x coordinates. Per coordinate, map six input roles
    # to all six target-role writes without dropping any coordinate.
    beta = np.empty((DIM, len(ROLES), len(ROLES)), np.float32)
    intercept = np.empty((DIM, len(ROLES)), np.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for start in range(0, DIM, chunk):
        end = min(start + chunk, DIM)
        xt = torch.tensor(np.asarray(x[:, :, start:end], np.float32).transpose(2, 0, 1), device=device)
        yt = torch.tensor(np.asarray(y[:, :, start:end], np.float32).transpose(2, 0, 1), device=device)
        xm = xt.mean(dim=1, keepdim=True)
        ym = yt.mean(dim=1, keepdim=True)
        xc = xt - xm
        yc = yt - ym
        gram = xc.transpose(1, 2) @ xc
        scale = gram.diagonal(dim1=1, dim2=2).mean(dim=1).clamp_min(1e-8)
        eye = torch.eye(len(ROLES), device=device)[None]
        b = torch.linalg.solve(gram + RIDGE * scale[:, None, None] * eye, xc.transpose(1, 2) @ yc)
        i = ym[:, 0] - (xm @ b)[:, 0]
        beta[start:end] = b.cpu().numpy()
        intercept[start:end] = i.cpu().numpy()
    return beta, intercept


def predict_bundle(x: np.ndarray, beta: np.ndarray, intercept: np.ndarray) -> np.ndarray:
    return np.einsum("srd,dro->sod", np.asarray(x, np.float32), beta, optimize=True) + intercept.T[None]


def fit_role_diag(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    slope = np.empty((len(ROLES), DIM), np.float32)
    intercept = np.empty_like(slope)
    for role_i in range(len(ROLES)):
        slope[role_i], intercept[role_i] = previous.fit_diag(x[:, role_i], y[:, role_i])
    return slope, intercept


def c518() -> None:
    audit = load(PARENT_AUDIT)
    out = begin("C518", {
        "status": "fresh_callable_state_master_contract_frozen",
        "parent_result": "graph and temporal high-order panel candidates; nested negative control; no single-sample predictor",
        "material": "3 families x 10 entirely fresh lexical units x 3 surfaces x 16 complete cells",
        "routes": ["fresh panel replication", "single-sample role bundle", "all-adjacent-checkpoint rollout", "single-token intervention eligibility"],
        "gates": {"fresh_panel_gain": 0.01, "single_sample_gain": 0.01, "rollout_gain": 0.02},
        "route_policy": "nested failure does not stop graph or temporal routes; panel failure does not stop single-sample observation",
        "forbidden": ["Attention", "MLP", "weights", "PCA", "Top-K", "post-reveal threshold changes"],
    }, {"parent_audit": audit["status"] == "passed", "cuda": torch.cuda.is_available()})
    rows = fresh_material()
    write_rows(out / "material/fresh_cases.jsonl", rows)
    family_counts = {family: sum(row["family"] == family for row in rows) for family in FAMILIES}
    close("C518", {
        "status": "contract_and_material_closed", "rows": len(rows), "family_counts": family_counts,
        "partitions": {part: sum(row["partition"] == part for row in rows) for part in ("discovery", "confirmation", "lockbox")},
        "strict_interpretation": "Fresh vocabulary and paraphrased relation phrases test prospective transfer; they are still controlled English micro-programs.",
    }, {"rows": len(rows) == 1440, "families": set(family_counts.values()) == {480}, "unique": len({row["case_id"] for row in rows}) == 1440}, "C519_audit_behavior")


def c519() -> None:
    out = begin("C519", {
        "status": "fresh_material_audit_behavior_frozen", "model": "local Qwen3 BF16 CUDA",
        "behavior_gate": {"global": 0.80, "family": 0.65},
        "observation_policy": "behavior errors retained as typed strata",
    }, {"parent": final("C518")["all_checks_passed"]})
    rows = fresh_rows()
    compiled = previous.parent.compile_material(rows)
    write_rows(out / "compiled/qwen3_fresh.jsonl", compiled)
    max_width = max(len(row["prompt_ids"]) for row in compiled)
    balance = {family: float(np.mean([row["gold_position"] == 0 for row in rows if row["family"] == family])) for family in FAMILIES}
    behavior = previous.parent.run_behavior(rows, compiled, out)
    raw = read_rows(out / "raw/behavior.jsonl")
    by_id = {row["case_id"]: row for row in rows}
    family_accuracy = {family: float(np.mean([row["correct"] for row in raw if by_id[row["case_id"]]["family"] == family])) for family in FAMILIES}
    close("C519", {
        "status": "audit_behavior_closed", **behavior, "max_prompt_tokens": max_width,
        "family_first_position_rate": balance, "family_accuracy": family_accuracy,
        "eligible_families": [family for family, value in family_accuracy.items() if value >= 0.65],
        "field_authorized": True,
    }, {"rows": behavior["rows"] == 1440, "width": max_width <= 128, "balance": set(balance.values()) == {0.5}, "finite": finite(behavior)}, "C520_capture")


def c520() -> None:
    out = begin("C520", {
        "status": "old_fresh_three_family_capture_frozen", "model": "local Qwen3 BF16 CUDA",
        "state": "q0 embedding, all 36 block outputs, final norm; six roles; all 2560 coordinates",
        "full_token_subset": "balanced lockbox complete cells for old and fresh panels",
    }, {"parent": final("C519")["all_checks_passed"], "cuda": torch.cuda.is_available()})
    rows, compiled = combined_rows_compiled()
    full_ids = {row["case_id"] for row in rows if row["unit"] == 8 and row["construction"] == "ledger"}
    width = max(len(row["prompt_ids"]) for row in compiled)
    headline = previous.parent.capture_state_cube(rows, compiled, out, full_ids, width, batch_size=8)
    close("C520", {"status": "capture_closed", **headline, "old_rows": 1440, "fresh_rows": 1440}, {
        "rows": headline["rows"] == 2880,
        "role_shape": headline["role_shape"] == [2880, 38, 6, 2560],
        "full_rows": headline["full_token_rows"] == 96,
        "finite": finite(headline),
    }, "C521_panel_replication")


def c521() -> None:
    out = begin("C521", {
        "status": "fresh_high_order_panel_replication_frozen",
        "train": "all 30 old-vocabulary complete programs per family",
        "test": "all 30 fresh-vocabulary complete programs per family",
        "models": ["identity", "shared channel diagonal", "family channel diagonal"],
        "high_order": "five Walsh masks of order 3 or 4",
        "gate": "family model improves shared NRMSE >= 0.01 separately per family",
    }, {"parent": final("C520")["all_checks_passed"]})
    rows, _ = combined_rows_compiled()
    index = read_rows(OUTS["C520"] / "raw/hidden_index.jsonl")
    states = np.load(OUTS["C520"] / "raw/role_states.float16.npy", mmap_mode="r")
    groups = build_groups(rows, index)
    old = [row for row in groups if row["source"] == "old"]
    fresh = [row for row in groups if row["source"] == "fresh"]
    basis = previous.orthonormal_walsh()
    masks = [mask for mask in range(16) if mask.bit_count() >= 3]
    acc = {family: {name: metric_acc() for name in ("identity", "shared", "family")} for family in FAMILIES}
    context = {}
    for q0, q1 in EDGES:
        for role_i, role in enumerate(ROLES):
            sx = np.einsum("ab,pbd->pad", basis, group_cube(states, old, q0, role_i), optimize=True)
            sy = np.einsum("ab,pbd->pad", basis, group_cube(states, old, q1, role_i), optimize=True)
            shared_s, shared_i = previous.fit_channel_diag(sx, sy)
            for family in FAMILIES:
                train = [row for row in old if row["family"] == family]
                test = [row for row in fresh if row["family"] == family]
                fx = np.einsum("ab,pbd->pad", basis, group_cube(states, train, q0, role_i), optimize=True)
                fy = np.einsum("ab,pbd->pad", basis, group_cube(states, train, q1, role_i), optimize=True)
                fs, fi = previous.fit_channel_diag(fx, fy)
                tx = np.einsum("ab,pbd->pad", basis, group_cube(states, test, q0, role_i), optimize=True)
                ty = np.einsum("ab,pbd->pad", basis, group_cube(states, test, q1, role_i), optimize=True)
                predictions = {"identity": tx[:, masks], "shared": (tx * shared_s + shared_i)[:, masks], "family": (tx * fs + fi)[:, masks]}
                key = f"{family}:q{q0}_q{q1}:{role}"
                context[key] = {name: vector_metric(pred, ty[:, masks]) for name, pred in predictions.items()}
                for name, pred in predictions.items():
                    add_metric(acc[family][name], pred, ty[:, masks])
    metrics = {family: {name: finish_metric(value) for name, value in models.items()} for family, models in acc.items()}
    gains = {family: metrics[family]["shared"]["nrmse"] - metrics[family]["family"]["nrmse"] for family in FAMILIES}
    candidates = {family: value >= 0.01 for family, value in gains.items()}
    save(out / "analysis/context_metrics.json", context)
    del states
    close("C521", {
        "status": "fresh_panel_replication_closed", "metrics": metrics, "family_conditioned_gains": gains,
        "family_candidates": candidates,
        "strict_interpretation": "A pass is prospective transfer of a complete-panel response rule, not a single-sentence callable mechanism.",
    }, {"finite": finite(metrics), "families": len(metrics) == 3, "contexts": len(context) == 90}, "C522_single_sample")


def c522() -> None:
    out = begin("C522", {
        "status": "single_sample_role_bundle_tournament_frozen",
        "train": "all individual old-vocabulary cells",
        "test": "all individual fresh-vocabulary cells",
        "target": "six-role local write H(q+1)-H(q)",
        "models": ["zero write", "family mean", "family same-role diagonal", "shared six-role bundle", "family six-role bundle"],
        "gate": "shared role bundle improves family mean by >= 0.01 for all three families and aggregate",
    }, {"parent": final("C521")["all_checks_passed"]})
    rows, _ = combined_rows_compiled()
    index = read_rows(OUTS["C520"] / "raw/hidden_index.jsonl")
    states = np.load(OUTS["C520"] / "raw/role_states.float16.npy", mmap_mode="r")
    by_id = {row["case_id"]: row for row in rows}
    old_ids = [row["hidden_index"] for row in index if not row["case_id"].startswith("c518-")]
    fresh_by_family = {family: [row["hidden_index"] for row in index if row["case_id"].startswith("c518-") and row["family"] == family] for family in FAMILIES}
    old_by_family = {family: [row["hidden_index"] for row in index if not row["case_id"].startswith("c518-") and row["family"] == family] for family in FAMILIES}
    shared_beta = np.empty((len(EDGES), DIM, 6, 6), np.float32)
    shared_intercept = np.empty((len(EDGES), DIM, 6), np.float32)
    family_beta = np.empty((3, len(EDGES), DIM, 6, 6), np.float32)
    family_intercept = np.empty((3, len(EDGES), DIM, 6), np.float32)
    acc = {family: {name: metric_acc() for name in ("zero", "mean", "diag", "shared_bundle", "family_bundle")} for family in FAMILIES}
    context = {}
    for edge_i, (q0, q1) in enumerate(EDGES):
        x_all = np.asarray(states[old_ids, q0], np.float32)
        y_all = np.asarray(states[old_ids, q1], np.float32) - x_all
        sb, si = fit_bundle(x_all, y_all)
        shared_beta[edge_i], shared_intercept[edge_i] = sb, si
        for family_i, family in enumerate(FAMILIES):
            train = old_by_family[family]
            test = fresh_by_family[family]
            x = np.asarray(states[train, q0], np.float32)
            y = np.asarray(states[train, q1], np.float32) - x
            fb, fi = fit_bundle(x, y)
            ds, di = fit_role_diag(x, y)
            family_beta[family_i, edge_i], family_intercept[family_i, edge_i] = fb, fi
            tx = np.asarray(states[test, q0], np.float32)
            ty = np.asarray(states[test, q1], np.float32) - tx
            predictions = {
                "zero": np.zeros_like(ty),
                "mean": np.broadcast_to(y.mean(axis=0), ty.shape),
                "diag": tx * ds[None] + di[None],
                "shared_bundle": predict_bundle(tx, sb, si),
                "family_bundle": predict_bundle(tx, fb, fi),
            }
            context[f"{family}:q{q0}_q{q1}"] = {name: vector_metric(pred, ty) for name, pred in predictions.items()}
            for name, pred in predictions.items():
                add_metric(acc[family][name], pred, ty)
    np.savez_compressed(out / "analysis/role_bundle_models.npz", shared_beta=shared_beta.astype(np.float16), shared_intercept=shared_intercept.astype(np.float16), family_beta=family_beta.astype(np.float16), family_intercept=family_intercept.astype(np.float16), families=np.asarray(FAMILIES))
    metrics = {family: {name: finish_metric(value) for name, value in models.items()} for family, models in acc.items()}
    aggregate = {name: metric_acc() for name in next(iter(acc.values()))}
    for family in FAMILIES:
        for name in aggregate:
            for key in aggregate[name]:
                aggregate[name][key] += acc[family][name][key]
    aggregate_done = {name: finish_metric(value) for name, value in aggregate.items()}
    shared_gains = {family: metrics[family]["mean"]["nrmse"] - metrics[family]["shared_bundle"]["nrmse"] for family in FAMILIES}
    shared_gains["aggregate"] = aggregate_done["mean"]["nrmse"] - aggregate_done["shared_bundle"]["nrmse"]
    family_gains = {family: metrics[family]["mean"]["nrmse"] - metrics[family]["family_bundle"]["nrmse"] for family in FAMILIES}
    shared_candidate = all(value >= 0.01 for value in shared_gains.values())
    boundary_core = context["typed_graph_path:q24_q25"]["shared_bundle"]["nrmse"] < context["typed_graph_path:q24_q25"]["mean"]["nrmse"] - 0.01 and context["temporal_composition:q24_q25"]["shared_bundle"]["nrmse"] < context["temporal_composition:q24_q25"]["mean"]["nrmse"] - 0.01
    save(out / "analysis/context_metrics.json", context)
    del states
    close("C522", {
        "status": "single_sample_tournament_closed", "metrics": metrics, "aggregate": aggregate_done,
        "shared_bundle_gains": shared_gains, "family_bundle_gains": family_gains,
        "shared_single_sample_candidate": shared_candidate, "q24_q25_graph_temporal_context_candidate": boundary_core,
        "strict_interpretation": "The role bundle is callable from one registered sample but still uses externally aligned role spans; it is not an autonomous parser or a unique circuit.",
    }, {"finite": finite(metrics), "contexts": len(context) == 15, "model_shape": list(shared_beta.shape) == [5, 2560, 6, 6]}, "C523_rollout")


def c523() -> None:
    out = begin("C523", {
        "status": "all_adjacent_checkpoint_rollout_frozen",
        "train": "old vocabulary, all 37 adjacent transitions",
        "test": "fresh vocabulary autonomous rollout from q0 role bundle",
        "models": ["shared role bundle", "family role bundle"],
        "readouts": [1, 8, 16, 24, 32, 37],
        "gate": "autonomous shared final-state NRMSE improves persistence by >= 0.02 in all families",
    }, {"parent": final("C522")["all_checks_passed"]})
    rows, _ = combined_rows_compiled()
    index = read_rows(OUTS["C520"] / "raw/hidden_index.jsonl")
    states = np.load(OUTS["C520"] / "raw/role_states.float16.npy", mmap_mode="r")
    old_ids = [row["hidden_index"] for row in index if not row["case_id"].startswith("c518-")]
    fresh_by_family = {family: [row["hidden_index"] for row in index if row["case_id"].startswith("c518-") and row["family"] == family] for family in FAMILIES}
    trajectories = {}
    aggregate = {family: {name: metric_acc() for name in ("persistence", "shared", "family")} for family in FAMILIES}
    # Fit and immediately apply each transition, avoiding a large retained model archive.
    current = {family: {"shared": np.asarray(states[ids, 0], np.float32), "family": np.asarray(states[ids, 0], np.float32), "origin": np.asarray(states[ids, 0], np.float32)} for family, ids in fresh_by_family.items()}
    readouts = {1, 8, 16, 24, 32, 37}
    for q in range(37):
        x_all = np.asarray(states[old_ids, q], np.float32)
        y_all = np.asarray(states[old_ids, q + 1], np.float32) - x_all
        sb, si = fit_bundle(x_all, y_all)
        family_models = {}
        for family in FAMILIES:
            train = [row["hidden_index"] for row in index if not row["case_id"].startswith("c518-") and row["family"] == family]
            fx = np.asarray(states[train, q], np.float32)
            fy = np.asarray(states[train, q + 1], np.float32) - fx
            family_models[family] = fit_bundle(fx, fy)
        for family in FAMILIES:
            fb, fi = family_models[family]
            current[family]["shared"] = current[family]["shared"] + predict_bundle(current[family]["shared"], sb, si)
            current[family]["family"] = current[family]["family"] + predict_bundle(current[family]["family"], fb, fi)
            if q + 1 in readouts:
                truth = np.asarray(states[fresh_by_family[family], q + 1], np.float32)
                metrics = {
                    "persistence": vector_metric(current[family]["origin"], truth),
                    "shared": vector_metric(current[family]["shared"], truth),
                    "family": vector_metric(current[family]["family"], truth),
                }
                trajectories[f"{family}:q{q + 1}"] = metrics
                if q + 1 == 37:
                    for name, pred in (("persistence", current[family]["origin"]), ("shared", current[family]["shared"]), ("family", current[family]["family"])):
                        add_metric(aggregate[family][name], pred, truth)
        print(f"[rollout] q{q + 1}/q37", flush=True)
    final_metrics = {family: {name: finish_metric(value) for name, value in models.items()} for family, models in aggregate.items()}
    gains = {family: final_metrics[family]["persistence"]["nrmse"] - final_metrics[family]["shared"]["nrmse"] for family in FAMILIES}
    candidate = all(value >= 0.02 for value in gains.values())
    save(out / "analysis/trajectory_metrics.json", trajectories)
    del states
    close("C523", {
        "status": "rollout_closed", "final_metrics": final_metrics, "shared_final_gains_over_persistence": gains,
        "autonomous_shared_candidate": candidate,
        "strict_interpretation": "Teacher-forced one-step fit and autonomous rollout are different claims; only the latter can support a trajectory law.",
    }, {"finite": finite(final_metrics), "readouts": len(trajectories) == 18, "families": len(final_metrics) == 3}, "C524_causal_eligibility")


def c524() -> None:
    out = begin("C524", {
        "status": "single_token_intervention_eligibility_frozen",
        "requirements": ["single-sample local candidate", "autonomous trajectory candidate", "exact token-state intervention compiler", "wrong-map controls"],
        "rule": "role-span averages cannot be written back as exact multi-token states; only boundary-token path could qualify after both predictive gates",
    }, {"parent": final("C523")["all_checks_passed"]})
    local = final("C522")["headline"]
    rollout = final("C523")["headline"]
    authorized = bool(local["shared_single_sample_candidate"] and local["q24_q25_graph_temporal_context_candidate"] and rollout["autonomous_shared_candidate"])
    causal = {"authorized": authorized, "ran": False, "result": "NA_predictive_qualification_failed" if not authorized else "NA_exact_boundary_executor_not_preregistered_in_parent_contract"}
    close("C524", {
        "status": "causal_eligibility_closed", "causal": causal,
        "measurement_to_intervention_boundary": "multi-token role averages are observer states, not exact writable token states",
        "strict_interpretation": "No patch is run unless prediction and intervention representation are both qualified before reveal.",
    }, {"authorization_rule": not causal["ran"], "finite": True}, "C525_visual_cleanup")


def register_visual() -> None:
    if REGISTRY.exists():
        registry = load(REGISTRY)
        item = {"id": "c525_fresh_callable_state_atlas", "title": "C525 Fresh Callable State Atlas", "phase": 2059, "campaign": "C518-C525", "path": "vis_data/research_kernel/c525_fresh_callable_state_atlas.json", "kind": "fresh_role_bundle_coordinate_atlas", "coordinates": 2560}
        datasets = registry.setdefault("datasets", [])
        datasets[:] = [row for row in datasets if row.get("id") != item["id"]]
        datasets.append(item)
        save(REGISTRY, registry)
    if CATALOG.exists():
        catalog = load(CATALOG)
        item = {"id": "c525_fresh_callable_state_atlas", "title": "C525 Fresh Callable State Atlas", "url": "/vis_data/research_kernel/c525_fresh_callable_state_atlas.json", "phase": 2059, "full_coordinate": True}
        datasets = catalog.setdefault("field_datasets", [])
        datasets[:] = [row for row in datasets if row.get("id") != item["id"]]
        datasets.append(item)
        save(CATALOG, catalog)


def c525() -> None:
    out = begin("C525", {
        "status": "visual_cleanup_synthesis_frozen",
        "visual": "old/fresh q0, q24, and q24->q25 all-coordinate role vectors for all three families",
        "cleanup": "hash and delete C520 role/full-token fields after visual archive",
    }, {"parent": final("C524")["all_checks_passed"]})
    rows, _ = combined_rows_compiled()
    by_id = {row["case_id"]: row for row in rows}
    index = read_rows(OUTS["C520"] / "raw/hidden_index.jsonl")
    state_path = OUTS["C520"] / "raw/role_states.float16.npy"
    full_path = OUTS["C520"] / "raw/full_token_states.float16.npy"
    states = np.load(state_path, mmap_mode="r")
    visual_rows = []
    for source in ("old", "fresh"):
        for family in FAMILIES:
            candidates = [row for row in index if row["family"] == family and (row["case_id"].startswith("c518-") == (source == "fresh")) and row["partition"] == "lockbox"]
            row = sorted(candidates, key=lambda item: item["case_id"])[0]
            i = row["hidden_index"]
            for role_i, role in enumerate(ROLES):
                q0 = np.asarray(states[i, 0, role_i], np.float32)
                q24 = np.asarray(states[i, 24, role_i], np.float32)
                q25 = np.asarray(states[i, 25, role_i], np.float32)
                visual_rows.append({"source": source, "family": family, "case_id": row["case_id"], "role": role, "embedding_q0": q0.tolist(), "state_q24": q24.tolist(), "write_q24_q25": (q25 - q24).tolist()})
    visual = {
        "schema": "ai2050.fresh_callable_state_atlas.v1", "phase": 2059, "campaign": "C518-C525", "coordinate_count": DIM,
        "panel_replication": final("C521")["headline"], "single_sample": final("C522")["headline"], "rollout": final("C523")["headline"], "causal": final("C524")["headline"], "rows": visual_rows,
    }
    save(VISUAL, visual)
    register_visual()
    del states
    gc.collect()
    cleanup = []
    for path in (state_path, full_path):
        if path.exists():
            cleanup.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha(path)})
    save(out / "audit/raw_field_cleanup_ledger.json", {"files": cleanup, "total_bytes": sum(row["bytes"] for row in cleanup)})
    for row in cleanup:
        (ROOT / row["path"]).unlink()
    gates = {
        "fresh_nested_panel": final("C521")["headline"]["family_candidates"]["nested_composition"],
        "fresh_graph_panel": final("C521")["headline"]["family_candidates"]["typed_graph_path"],
        "fresh_temporal_panel": final("C521")["headline"]["family_candidates"]["temporal_composition"],
        "single_sample": final("C522")["headline"]["shared_single_sample_candidate"],
        "autonomous_rollout": final("C523")["headline"]["autonomous_shared_candidate"],
        "causal": final("C524")["headline"]["causal"]["ran"],
    }
    close("C525", {
        "status": "extended_campaign_closed", "gates": gates,
        "visual_path": str(VISUAL.relative_to(ROOT)).replace("\\", "/"), "visual_rows": len(visual_rows),
        "visual_coordinate_values": len(visual_rows) * 3 * DIM,
        "cleanup_files": len(cleanup), "cleanup_bytes": sum(row["bytes"] for row in cleanup),
        "raw_fields_absent": not state_path.exists() and not full_path.exists(), "new_math_gate": False,
        "strict_conclusion": "Fresh panel transfer and single-sample callable-state prediction are adjudicated separately. No panel-level regularity is promoted to an autonomous mechanism without role-bundle and rollout qualification.",
    }, {"visual": VISUAL.exists() and len(visual_rows) == 36, "coordinates": all(len(row["state_q24"]) == DIM for row in visual_rows), "cleanup": not state_path.exists() and not full_path.exists(), "finite": finite(gates)}, "C526_independent_audit")


FUNCTIONS = {"C518": c518, "C519": c519, "C520": c520, "C521": c521, "C522": c522, "C523": c523, "C524": c524, "C525": c525}


def self_test() -> None:
    rows = fresh_material()
    assert len(rows) == 1440
    assert len({row["case_id"] for row in rows}) == 1440
    assert all(sum(row["family"] == family for row in rows) == 480 for family in FAMILIES)
    print(json.dumps({"self_test": "passed", "rows": len(rows)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--start", default="C518")
    parser.add_argument("--stop", default="C525")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    names = list(FUNCTIONS)
    start, stop = names.index(args.start), names.index(args.stop)
    for name in names[start:stop + 1]:
        FUNCTIONS[name]()


if __name__ == "__main__":
    main()
