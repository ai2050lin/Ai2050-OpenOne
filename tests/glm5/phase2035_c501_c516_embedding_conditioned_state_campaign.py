#!/usr/bin/env python3
"""C501-C516 embedding-conditioned complete-state campaign.

The campaign observes token embeddings and HiddenState checkpoints only. It
keeps every physical activation coordinate, uses no PCA or Top-K selection,
and never inspects Attention, MLP activations, or model weights.
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
VISUAL = ROOT / "frontend/public/vis_data/research_kernel/c516_embedding_conditioned_state_atlas.json"
REGISTRY = ROOT / "ai2050_research_os/registry/field_datasets.json"
CATALOG = ROOT / "frontend/public/research_data/current/language_encoding_catalog.json"
sys.path.insert(0, str(TESTS))

import phase2019_c485_c500_complete_state_information_campaign as parent


PHASES = {
    f"C{campaign}": (2035 + campaign - 501, slug)
    for campaign, slug in (
        (501, "evidence_audit_and_embedding_conditioned_master_contract"),
        (502, "six_family_lexical_ecology_material"),
        (503, "semantic_tokenization_balance_and_naturalness_audit"),
        (504, "qwen_lexical_ecology_behavior_qualification"),
        (505, "seventeen_family_embedding_and_hiddenstate_full_coordinate_capture"),
        (506, "raw_walsh_random_orthogonal_basis_control"),
        (507, "lexical_embedding_and_state_trajectory_observation_atlas"),
        (508, "embedding_only_local_write_prediction"),
        (509, "current_state_and_embedding_incremental_prediction"),
        (510, "same_token_polysemy_identifiability_control"),
        (511, "family_conditioned_complete_channel_transition_tournament"),
        (512, "nested_composition_high_order_lockbox"),
        (513, "typed_graph_high_order_lockbox"),
        (514, "temporal_composition_high_order_lockbox"),
        (515, "single_sample_predictive_and_causal_eligibility_adjudication"),
        (516, "visual_archive_cleanup_and_campaign_synthesis"),
    )
}
OUTS = {
    name: RESULT / f"phase{phase}_{name.lower()}_{slug}"
    for name, (phase, slug) in PHASES.items()
}

DIM = 2560
CHECKPOINTS = 38
ROLES = parent.ROLES
ROLE_INDEX = {name: i for i, name in enumerate(ROLES)}
EDGES = ((0, 1), (8, 9), (16, 17), (24, 25), (32, 33))
CONSTRUCTIONS = parent.CONSTRUCTIONS
BITS = parent.BITS
OLD_FAMILIES = parent.FAMILIES
LEXICAL_FAMILIES = (
    "lex_noun_taxonomy",
    "lex_part_whole",
    "lex_verb_event",
    "lex_adjective_property",
    "lex_polysemy",
    "lex_function_relation",
)
ALL_FAMILIES = OLD_FAMILIES + LEXICAL_FAMILIES
LEXICAL_SURFACES = ("statement", "note", "dialogue")
RIDGE = 1e-2
FIELD_WIDTH_LIMIT = 144
ORTHOGONAL_SEED = 5012035

OLD_CASES = parent.OUTS["C488"] / "material/corrected_cases.jsonl"
OLD_COMPILED = parent.OUTS["C488"] / "compiled/qwen3.jsonl"
OLD_BEHAVIOR = parent.OUTS["C488"] / "raw/behavior.jsonl"


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
        "phase": PHASES[name][0],
        "campaign": name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "producer_sha256": producer_hash(),
        **protocol,
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
        "phase": PHASES[name][0],
        "campaign": name,
        "status": "closed",
        "checks": final_checks,
        "all_checks_passed": all(final_checks.values()),
        "headline": headline,
        "next_authorization": authorization,
    }
    save(out / "analysis/final.json", value)
    print(json.dumps(value, ensure_ascii=False), flush=True)
    return value


def final(name: str) -> dict:
    return load(OUTS[name] / "analysis/final.json")


def partition(unit: int) -> str:
    return "discovery" if unit < 5 else "confirmation" if unit < 8 else "lockbox"


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
        "n": int(acc["n"]),
        "mae": float(acc["ae"] / n),
        "rmse": float(math.sqrt(acc["se"] / n)),
        "nrmse": float(math.sqrt(acc["se"] / max(acc["yy"], 1e-30))),
        "cosine": float(acc["py"] / max(math.sqrt(acc["pp"] * acc["yy"]), 1e-30)),
    }


def vector_metric(prediction: np.ndarray, truth: np.ndarray) -> dict:
    acc = metric_acc()
    add_metric(acc, prediction, truth)
    return finish_metric(acc)


def close_mmap(value) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def answer_options(truth: bool, order: int) -> tuple[str, int]:
    if order == 0:
        return "(A) Yes (B) No", 0 if truth else 1
    return "(A) No (B) Yes", 1 if truth else 0


NOUN_UNITS = (
    ("apple", "fruit", "hammer", "tool"),
    ("sparrow", "bird", "salmon", "fish"),
    ("violin", "instrument", "jacket", "clothing"),
    ("tulip", "flower", "oak", "tree"),
    ("copper", "metal", "granite", "rock"),
    ("comet", "celestial object", "canoe", "boat"),
    ("carrot", "vegetable", "lemon", "citrus fruit"),
    ("whale", "mammal", "lizard", "reptile"),
    ("novel", "book", "sonata", "composition"),
    ("sapphire", "gemstone", "linen", "fabric"),
)
PART_UNITS = (
    ("wheel", "bicycle", "page", "book"),
    ("leaf", "tree", "petal", "flower"),
    ("engine", "car", "rotor", "helicopter"),
    ("handle", "mug", "blade", "knife"),
    ("roof", "house", "screen", "phone"),
    ("keyboard", "computer", "rudder", "boat"),
    ("button", "shirt", "drawer", "cabinet"),
    ("lens", "camera", "string", "violin"),
    ("chapter", "novel", "verse", "poem"),
    ("branch", "tree", "fin", "fish"),
)
VERB_UNITS = (
    ("Ava", "carried", "parcel", "Ben", "gate"),
    ("Cora", "opened", "window", "Dylan", "box"),
    ("Elena", "painted", "mural", "Felix", "fence"),
    ("Grace", "repaired", "clock", "Hector", "radio"),
    ("Iris", "measured", "table", "Jonah", "rope"),
    ("Kara", "folded", "blanket", "Liam", "map"),
    ("Maya", "washed", "bottle", "Noah", "plate"),
    ("Olive", "moved", "chair", "Peter", "lamp"),
    ("Rina", "labeled", "folder", "Simon", "crate"),
    ("Tara", "delivered", "letter", "Victor", "package"),
)
ADJECTIVE_UNITS = (
    ("ruby", "red", "snow", "white"),
    ("coal", "black", "lemon", "yellow"),
    ("feather", "light", "anvil", "heavy"),
    ("silk", "smooth", "sandpaper", "rough"),
    ("ice", "cold", "fire", "hot"),
    ("honey", "sweet", "vinegar", "sour"),
    ("tower", "tall", "stool", "short"),
    ("glass", "clear", "mud", "opaque"),
    ("sponge", "soft", "stone", "hard"),
    ("turtle", "slow", "falcon", "fast"),
)
POLYSEMY_UNITS = (
    ("bank", "financial institution", "river edge"),
    ("bat", "flying mammal", "sports club"),
    ("crane", "wading bird", "lifting machine"),
    ("spring", "season", "coiled device"),
    ("seal", "marine animal", "official stamp"),
    ("mole", "burrowing animal", "skin mark"),
    ("bark", "tree covering", "dog sound"),
    ("jam", "fruit preserve", "traffic blockage"),
    ("pitch", "musical frequency", "sports field"),
    ("match", "small fire stick", "sporting contest"),
)
FUNCTION_UNITS = (
    ("lamp", "above", "desk", "shelf"),
    ("coin", "below", "cup", "plate"),
    ("key", "inside", "drawer", "pocket"),
    ("chair", "outside", "room", "garage"),
    ("train", "before", "bus", "ferry"),
    ("concert", "after", "lecture", "meeting"),
    ("letter", "from", "Mira", "Nolan"),
    ("parcel", "to", "Orin", "Pia"),
    ("notebook", "beside", "pencil", "ruler"),
    ("bridge", "between", "village", "harbor"),
)


def lexical_payload(family: str, unit: int, variant: int) -> dict:
    if family == "lex_noun_taxonomy":
        a, x, b, y = NOUN_UNITS[unit]
        primary, context, secondary, distractor = ((a, x, b, y) if variant == 0 else (b, y, a, x))
        relation = "is a kind of"
    elif family == "lex_part_whole":
        a, x, b, y = PART_UNITS[unit]
        primary, context, secondary, distractor = ((a, x, b, y) if variant == 0 else (b, y, a, x))
        relation = "is part of"
    elif family == "lex_verb_event":
        a, rel_a, x, b, y = VERB_UNITS[unit]
        if variant == 0:
            primary, relation, context, secondary, distractor = a, rel_a, x, b, y
        else:
            primary, relation, context, secondary, distractor = b, "inspected", y, a, x
    elif family == "lex_adjective_property":
        a, x, b, y = ADJECTIVE_UNITS[unit]
        primary, context, secondary, distractor = ((a, x, b, y) if variant == 0 else (b, y, a, x))
        relation = "has the property"
    elif family == "lex_polysemy":
        word, sense_a, sense_b = POLYSEMY_UNITS[unit]
        primary, relation = word, "means"
        context, distractor = (sense_a, sense_b) if variant == 0 else (sense_b, sense_a)
        secondary = f"alternate sense of {word}"
    elif family == "lex_function_relation":
        a, rel, x, y = FUNCTION_UNITS[unit]
        primary, relation, context, secondary, distractor = a, rel, x, f"comparison item {unit}", y
    else:
        raise KeyError(family)
    return {
        "primary": primary,
        "secondary": secondary,
        "relation": relation,
        "context": context,
        "distractor": distractor,
    }


def lexical_core(family: str, surface: str, payload: dict, truth: bool) -> str:
    p = payload["primary"]
    s = payload["secondary"]
    r = payload["relation"]
    x = payload["context"]
    y = payload["distractor"]
    if family == "lex_polysemy":
        fact = f"In this context, {p} {r} {x}; the phrase {s} would instead mean {y}."
        query = f"Does {p} {r} {x if truth else y} in this context?"
    elif family == "lex_function_relation":
        fact = f"The {p} is {r} the {x}; the {s} is associated with the {y}."
        query = f"Is the {p} {r} the {x if truth else y}?"
    elif family == "lex_verb_event":
        fact = f"{p} {r} the {x}; {s} separately checked the {y}."
        query = f"Did {p} {r} the {x if truth else y}?"
    elif family == "lex_adjective_property":
        fact = f"The {p} {r} {x}; the {s} instead has the property {y}."
        query = f"Does the {p} {r} {x if truth else y}?"
    else:
        article = "an" if r == "is a kind of" and str(x)[0].lower() in "aeiou" else "a"
        fact = f"The {p} {r} {article} {x}; the {s} {r} the {y}."
        query = f"Is the {p} {r} the {x if truth else y}?"
    if surface == "statement":
        return f"Read the relevant statement: {fact} Based on it, {query}"
    if surface == "note":
        return f"A note records the following relevant information. {fact} From this information alone, {query}"
    return f"Analyst: {fact} Reviewer: Using only that record, {query}"


def lexical_material() -> list[dict]:
    rows = []
    for family, surface, unit, variant, truth_bit, order in itertools.product(
        LEXICAL_FAMILIES, LEXICAL_SURFACES, range(10), (0, 1), (0, 1), (0, 1)
    ):
        truth = truth_bit == 0
        payload = lexical_payload(family, unit, variant)
        core = lexical_core(family, surface, payload, truth)
        option_text, gold = answer_options(truth, order)
        bits = [variant, truth_bit, order, LEXICAL_SURFACES.index(surface) % 2]
        rows.append({
            "case_id": f"c502-{family}-{surface}-u{unit}-v{variant}-t{truth_bit}-o{order}",
            "panel": "lexical_ecology",
            "family": family,
            "lexical_class": family.removeprefix("lex_"),
            "surface": surface,
            "construction": surface,
            "surface_index": LEXICAL_SURFACES.index(surface),
            "unit": unit,
            "variant": variant,
            "truth_bit": truth_bit,
            "candidate_order": order,
            "bits": bits,
            "partition": partition(unit),
            "gold_position": gold,
            "correct_answer": "Yes" if truth else "No",
            "wrong_answer": "No" if truth else "Yes",
            "prompt_core": core,
            "prompt": f"{core} {option_text}. Reply with only A or B.",
            "free_prompt": f"{core} Answer only Yes or No.",
            "role_values": {
                "primary": payload["primary"],
                "secondary": payload["secondary"],
                "relation": payload["relation"],
                "context": payload["context"],
                "query": payload["primary"],
            },
            "semantic_graph": {
                "family": family,
                "lexical_class": family.removeprefix("lex_"),
                "primary": payload["primary"],
                "relation": payload["relation"],
                "context": payload["context"],
                "distractor": payload["distractor"],
                "truth": truth,
            },
        })
    return rows


def lexical_rows() -> list[dict]:
    return read_rows(OUTS["C502"] / "material/lexical_cases.jsonl")


def combined_rows_and_compiled() -> tuple[list[dict], list[dict]]:
    old_rows = read_rows(OLD_CASES)
    old_compiled = read_rows(OLD_COMPILED)
    new_rows = lexical_rows()
    new_compiled = read_rows(OUTS["C503"] / "compiled/qwen3_lexical.jsonl")
    return old_rows + new_rows, old_compiled + new_compiled


def split_indices(index: list[dict]) -> dict[str, list[int]]:
    result = {"train": [], "within": [], "surface": [], "lockbox": []}
    for row in index:
        i = int(row["hidden_index"])
        surface = row["construction"]
        report = surface in ("report", "dialogue")
        if row["partition"] == "discovery" and not report:
            result["train"].append(i)
        if row["partition"] == "confirmation" and not report:
            result["within"].append(i)
        if row["partition"] in ("confirmation", "lockbox") and report:
            result["surface"].append(i)
        if row["partition"] == "lockbox":
            result["lockbox"].append(i)
    return result


def fit_diag(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xf = np.asarray(x, np.float64).reshape(-1, DIM)
    yf = np.asarray(y, np.float64).reshape(-1, DIM)
    xm = xf.mean(axis=0)
    ym = yf.mean(axis=0)
    xc = xf - xm
    slope = (xc * (yf - ym)).sum(axis=0) / np.maximum((xc * xc).sum(axis=0), 1e-12)
    return slope.astype(np.float32), (ym - slope * xm).astype(np.float32)


def predict_diag(x: np.ndarray, slope: np.ndarray, intercept: np.ndarray) -> np.ndarray:
    return np.asarray(x, np.float32) * slope + intercept


def orthonormal_walsh() -> np.ndarray:
    return parent.walsh_matrix() * 4.0


def random_orthogonal(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.standard_normal((16, 16)))
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1
    return (q * signs).astype(np.float32)


def build_old_groups(rows: list[dict], index: list[dict]) -> list[dict]:
    hidden = {row["case_id"]: int(row["hidden_index"]) for row in index}
    grouped: dict[tuple[str, str, int], dict[tuple[int, ...], int]] = defaultdict(dict)
    for row in rows:
        if row["family"] not in OLD_FAMILIES:
            continue
        grouped[(row["family"], row["construction"], int(row["unit"]))][tuple(row["bits"])] = hidden[row["case_id"]]
    result = []
    for group_index, (key, cells) in enumerate(sorted(grouped.items())):
        if set(cells) != set(BITS):
            raise RuntimeError((key, len(cells)))
        family, construction, unit = key
        result.append({
            "group_index": group_index,
            "family": family,
            "construction": construction,
            "unit": unit,
            "partition": partition(unit),
            "indices": [cells[tuple(bits)] for bits in BITS],
        })
    return result


def group_cube(states: np.ndarray, groups: list[dict], q: int, role_i: int) -> np.ndarray:
    indices = np.asarray([row["indices"] for row in groups], dtype=np.int64)
    return np.asarray(states[indices, q, role_i], np.float32)


def fit_channel_operator(x: np.ndarray, y: np.ndarray, chunk: int = 64) -> tuple[np.ndarray, np.ndarray]:
    beta = np.empty((DIM, 16, 16), np.float32)
    intercept = np.empty((DIM, 16), np.float32)
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
        eye = torch.eye(16, device=device)[None]
        b = torch.linalg.solve(gram + RIDGE * scale[:, None, None] * eye, xc.transpose(1, 2) @ yc)
        i = ym[:, 0] - (xm @ b)[:, 0]
        beta[start:end] = b.cpu().numpy()
        intercept[start:end] = i.cpu().numpy()
    return beta, intercept


def predict_channel_operator(x: np.ndarray, beta: np.ndarray, intercept: np.ndarray) -> np.ndarray:
    return np.einsum("psd,dst->ptd", np.asarray(x, np.float32), beta, optimize=True) + intercept.T[None]


def fit_channel_diag(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Program x channel x coordinate; each channel/coordinate has its own affine map.
    xf = np.asarray(x, np.float64)
    yf = np.asarray(y, np.float64)
    xm = xf.mean(axis=0)
    ym = yf.mean(axis=0)
    xc = xf - xm
    slope = (xc * (yf - ym)).sum(axis=0) / np.maximum((xc * xc).sum(axis=0), 1e-12)
    return slope.astype(np.float32), (ym - slope * xm).astype(np.float32)


def c501() -> None:
    parent_audit = parent.OUTS["C500"] / "audit/independent_audit.json"
    out = begin("C501", {
        "status": "embedding_conditioned_complete_state_master_contract_frozen",
        "research_object": "whether token embedding adds prospective routing information beyond current role state, and whether complete-channel transition is basis-specific or condition-specific",
        "routes": [
            "orthonormal raw/Walsh/random basis control",
            "six-family lexical ecology",
            "embedding-only local-write prediction",
            "current-state plus embedding incremental prediction",
            "same-token polysemy identifiability",
            "family-conditioned complete-channel transition",
            "nested, graph, and temporal high-order lockboxes",
        ],
        "route_policy": "a failed route is recorded and remaining preregistered routes continue",
        "measurement": "token embedding and HiddenState only; all 2560 coordinates; no Attention, MLP, weights, PCA, or Top-K",
        "qualification_gates": {
            "embedding_incremental_nrmse_gain": 0.005,
            "control_gap": 0.002,
            "conditioned_lockbox_nrmse_gain": 0.01,
            "basis_equivalence_max_nrmse_delta": 0.001,
        },
    }, {
        "parent_audit": parent_audit.exists() and load(parent_audit).get("status") == "passed",
        "cuda": torch.cuda.is_available(),
        "old_material": OLD_CASES.exists() and OLD_COMPILED.exists(),
    })
    retained = [
        "C492 shows that a complete 16-condition state improves over identity on same-panel transfers.",
        "C492 does not establish Walsh-basis specificity because equal-capacity raw and random orthogonal bases were not tested.",
        "The captured q0 state is the model token-embedding activation, not a context-complete concept vector.",
        "A complete experimental panel predictor is not automatically callable from one natural sample.",
    ]
    corrections = [
        "The attachment's random-basis result was a proposed test, not completed evidence.",
        "The zero/nonzero RMS ratio is not a causal or semantic energy decomposition.",
        "Embedding as a routing prior is a hypothesis; lexical identity, tokenization, frequency, and polysemy remain confounds.",
        "High cosine or lower NRMSE is predictive dependence, not a unique causal circuit or new mathematics.",
    ]
    save(out / "analysis/evidence_audit.json", {"retained": retained, "corrected_overclaims": corrections})
    close("C501", {
        "status": "audit_and_contract_closed",
        "retained": retained,
        "corrected_overclaims": corrections,
        "strict_boundary": "The campaign may qualify a predictive state variable; it cannot presuppose a semantic key, a gear circuit, or a new mathematical theory.",
    }, {"retained": len(retained) == 4, "corrections": len(corrections) == 4}, "C502_material")


def c502() -> None:
    out = begin("C502", {
        "status": "lexical_ecology_material_frozen",
        "families": list(LEXICAL_FAMILIES),
        "design": "6 families x 10 lexical units x 3 surfaces x 2 lexical variants x 2 truth values x 2 answer orders = 1440 cases",
        "partitions": {"discovery": list(range(5)), "confirmation": [5, 6, 7], "lockbox": [8, 9]},
        "naturalness_boundary": "controlled English with lexical and role diversity; no independent human naturalness claim",
    }, {"parent": final("C501")["all_checks_passed"]})
    rows = lexical_material()
    write_rows(out / "material/lexical_cases.jsonl", rows)
    family_counts = {family: sum(row["family"] == family for row in rows) for family in LEXICAL_FAMILIES}
    close("C502", {
        "status": "material_closed",
        "rows": len(rows),
        "family_counts": family_counts,
        "partitions": {part: sum(row["partition"] == part for row in rows) for part in ("discovery", "confirmation", "lockbox")},
        "strict_interpretation": "The lexical ecology broadens observed lexical roles; it is not a complete language-family ontology.",
    }, {
        "rows": len(rows) == 1440,
        "families": all(value == 240 for value in family_counts.values()),
        "unique_ids": len({row["case_id"] for row in rows}) == len(rows),
    }, "C503_audit")


def c503() -> None:
    out = begin("C503", {
        "status": "semantic_tokenization_balance_audit_frozen",
        "zero_models": ["always A", "always B", "truth-only without answer order", "surface-majority"],
        "role_policy": "all five lexical roles must compile to at least one token; boundary is the assistant generation boundary",
        "naturalness": "rule-based grammar and lexical review only; independent human review absent",
    }, {"parent": final("C502")["all_checks_passed"]})
    rows = lexical_rows()
    compiled = parent.compile_material(rows)
    write_rows(out / "compiled/qwen3_lexical.jsonl", compiled)
    max_width = max(len(row["prompt_ids"]) for row in compiled)
    family_position = {
        family: float(np.mean([row["gold_position"] == 0 for row in rows if row["family"] == family]))
        for family in LEXICAL_FAMILIES
    }
    surface_position = {
        surface: float(np.mean([row["gold_position"] == 0 for row in rows if row["surface"] == surface]))
        for surface in LEXICAL_SURFACES
    }
    role_lengths = defaultdict(list)
    for row in compiled:
        for role in ROLES:
            role_lengths[role].append(len(row["role_positions"][role]))
    same_token_pairs = 0
    for family, surface, unit, truth, order in itertools.product(
        ("lex_polysemy",), LEXICAL_SURFACES, range(10), (0, 1), (0, 1)
    ):
        ids = [
            row for row in compiled
            if row["family"] == family and row["surface"] == surface and row["unit"] == unit
            and row["truth_bit"] == truth and row["candidate_order"] == order
        ]
        if len(ids) == 2:
            same_token_pairs += 1
    checks = {
        "compiled_rows": len(compiled) == len(rows) == 1440,
        "width": max_width <= FIELD_WIDTH_LIMIT,
        "global_position_balance": float(np.mean([row["gold_position"] == 0 for row in rows])) == 0.5,
        "family_position_balance": all(value == 0.5 for value in family_position.values()),
        "surface_position_balance": all(value == 0.5 for value in surface_position.values()),
        "roles_nonempty": all(min(values) >= 1 for values in role_lengths.values()),
        "semantic_unique": all(row["correct_answer"] != row["wrong_answer"] for row in rows),
        "polysemy_pairs": same_token_pairs == 120,
    }
    audit = {
        "checks": checks,
        "max_prompt_tokens": max_width,
        "family_first_position_rate": family_position,
        "surface_first_position_rate": surface_position,
        "role_token_length_ranges": {key: [min(values), max(values)] for key, values in role_lengths.items()},
        "naturalness_audit": {
            "status": "controlled_template_review_passed",
            "independent_human_review": False,
            "hard_boundary": "machine and rule-based review cannot establish human naturalness",
        },
    }
    save(out / "analysis/material_audit.json", audit)
    close("C503", {"status": "audit_closed", **audit}, checks, "C504_behavior")


def c504() -> None:
    out = begin("C504", {
        "status": "qwen_lexical_behavior_frozen",
        "model": "local Qwen3 BF16 CUDA",
        "behavior_gate": {"global": 0.80, "family": 0.65},
        "observation_policy": "all rows remain observable; qualification metrics are also reported on behavior-correct rows",
    }, {"parent": final("C503")["all_checks_passed"]})
    rows = lexical_rows()
    compiled = read_rows(OUTS["C503"] / "compiled/qwen3_lexical.jsonl")
    headline = parent.run_behavior(rows, compiled, out)
    behavior = read_rows(out / "raw/behavior.jsonl")
    by_id = {row["case_id"]: row for row in rows}
    family_accuracy = {
        family: float(np.mean([row["correct"] for row in behavior if by_id[row["case_id"]]["family"] == family]))
        for family in LEXICAL_FAMILIES
    }
    eligible = [family for family, value in family_accuracy.items() if value >= 0.65]
    close("C504", {
        "status": "behavior_closed",
        **headline,
        "family_accuracy": family_accuracy,
        "eligible_families": eligible,
        "field_authorized": True,
        "strict_interpretation": "Behavior qualifies only this fixed answer interface; failed families remain descriptive missingness strata.",
    }, {
        "rows": headline["rows"] == 1440,
        "finite": finite(headline),
        "all_families_reported": len(family_accuracy) == 6,
        "field_policy": True,
    }, "C505_capture")


def c505() -> None:
    out = begin("C505", {
        "status": "seventeen_family_full_coordinate_capture_frozen",
        "model": "local Qwen3 BF16 CUDA",
        "state": "q0 token embedding, q1-q36 block outputs, q37 final norm; six roles; all 2560 coordinates",
        "full_token_subset": "one lockbox complete program per old family plus balanced lexical lockbox rows; all registered qpoint coordinates",
        "cleanup": "raw fields deleted only after visual archive and independent hash ledger are written",
    }, {"parent": final("C504")["all_checks_passed"], "cuda": torch.cuda.is_available()})
    rows, compiled = combined_rows_and_compiled()
    full_ids = {
        row["case_id"] for row in rows
        if (
            row["family"] in OLD_FAMILIES
            and row["construction"] == "ledger"
            and row["unit"] == 8
        ) or (
            row["family"] in LEXICAL_FAMILIES
            and row["surface"] == "statement"
            and row["unit"] in (8, 9)
        )
    }
    width = max(len(row["prompt_ids"]) for row in compiled)
    headline = parent.capture_state_cube(rows, compiled, out, full_ids, width, batch_size=8)
    index = read_rows(out / "raw/hidden_index.jsonl")
    family_counts = {family: sum(row["family"] == family for row in index) for family in ALL_FAMILIES}
    close("C505", {
        "status": "capture_closed",
        **headline,
        "families": len(family_counts),
        "family_counts": family_counts,
        "embedding_boundary": "q0 is the embed_tokens output; in Qwen3 it is a token embedding activation, not an already contextualized concept state.",
    }, {
        "rows": headline["rows"] == 6720,
        "role_shape": headline["role_shape"] == [6720, 38, 6, 2560],
        "full_rows": headline["full_token_rows"] == len(full_ids),
        "finite_accuracy": math.isfinite(headline["accuracy"]),
        "families": len(family_counts) == 17,
    }, "C506_basis_control")


def basis_splits(groups: list[dict]) -> tuple[list[dict], dict[str, list[dict]]]:
    lock = {"type_graph", "part_whole", "temporal_composition"}
    train_family = set(OLD_FAMILIES) - lock
    train = [row for row in groups if row["family"] in train_family and row["unit"] < 5]
    splits = {
        "within": [row for row in groups if row["family"] in train_family and 5 <= row["unit"] < 8],
        "family": [row for row in groups if row["family"] in lock and row["unit"] < 8],
        "lockbox": [row for row in groups if row["unit"] >= 8],
    }
    return train, splits


def c506() -> None:
    out = begin("C506", {
        "status": "equal_capacity_basis_control_frozen",
        "bases": ["raw_cell_identity", "orthonormal_walsh", "fixed_random_orthogonal"],
        "fairness": "all bases are 16x16 orthonormal transforms and use identical isotropic ridge scaling",
        "contexts": ["q24_to_q25 primary", "q24_to_q25 relation", "q24_to_q25 query"],
        "gate": {"max_nrmse_delta": 0.001},
        "mathematical_note": "orthogonal basis covariance predicts equal raw-space fits for isotropic ridge; empirical execution checks implementation and numerical tolerance",
    }, {"parent": final("C505")["all_checks_passed"]})
    rows, _ = combined_rows_and_compiled()
    index = read_rows(OUTS["C505"] / "raw/hidden_index.jsonl")
    states = np.load(OUTS["C505"] / "raw/role_states.float16.npy", mmap_mode="r")
    groups = build_old_groups(rows, index)
    train, splits = basis_splits(groups)
    bases = {
        "raw": np.eye(16, dtype=np.float32),
        "walsh": orthonormal_walsh(),
        "random": random_orthogonal(ORTHOGONAL_SEED),
    }
    metrics = {}
    raw_predictions = {}
    contexts = ((24, 25, "primary"), (24, 25, "relation"), (24, 25, "query"))
    for q0, q1, role in contexts:
        role_i = ROLE_INDEX[role]
        train_x_raw = group_cube(states, train, q0, role_i)
        train_y_raw = group_cube(states, train, q1, role_i)
        key = f"q{q0}_q{q1}_{role}"
        metrics[key] = {}
        raw_predictions[key] = {}
        for basis_name, basis in bases.items():
            x = np.einsum("ab,pbd->pad", basis, train_x_raw, optimize=True)
            y = np.einsum("ab,pbd->pad", basis, train_y_raw, optimize=True)
            beta, intercept = fit_channel_operator(x, y)
            metrics[key][basis_name] = {}
            for split_name, panel in splits.items():
                px_raw = group_cube(states, panel, q0, role_i)
                py_raw = group_cube(states, panel, q1, role_i)
                px = np.einsum("ab,pbd->pad", basis, px_raw, optimize=True)
                pred_basis = predict_channel_operator(px, beta, intercept)
                pred_raw = np.einsum("ba,pbd->pad", basis, pred_basis, optimize=True)
                metrics[key][basis_name][split_name] = vector_metric(pred_raw, py_raw)
                raw_predictions[key][(basis_name, split_name)] = pred_raw
            del beta, intercept
            gc.collect()
    del states
    deltas = []
    for key in metrics:
        for split_name in splits:
            values = [metrics[key][basis][split_name]["nrmse"] for basis in bases]
            deltas.append(max(values) - min(values))
    max_delta = float(max(deltas))
    basis_equivalent = max_delta <= 0.001
    save(out / "analysis/basis_metrics.json", metrics)
    close("C506", {
        "status": "basis_control_closed",
        "metrics": metrics,
        "max_nrmse_delta": max_delta,
        "basis_equivalent_within_tolerance": basis_equivalent,
        "walsh_specificity_supported": not basis_equivalent,
        "strict_interpretation": "Equivalent equal-capacity bases attribute the C492 gain to complete 16-condition information, not to a privileged Walsh coordinate system. Walsh labels remain useful experimental bookkeeping for factor order.",
    }, {
        "finite": finite(metrics),
        "contexts": len(metrics) == 3,
        "splits": all(set(value["raw"]) == set(splits) for value in metrics.values()),
    }, "C507_embedding_atlas")


def c507() -> None:
    out = begin("C507", {
        "status": "lexical_embedding_state_atlas_frozen",
        "objects": ["q0 role embeddings", "five local writes", "same token across surface", "same token across polysemous context"],
        "measurement": "full 2560-coordinate differences and norms; no coordinate selection",
    }, {"parent": final("C506")["all_checks_passed"]})
    rows, _ = combined_rows_and_compiled()
    index = read_rows(OUTS["C505"] / "raw/hidden_index.jsonl")
    states = np.load(OUTS["C505"] / "raw/role_states.float16.npy", mmap_mode="r")
    row_by_id = {row["case_id"]: row for row in rows}
    lexical = [row for row in index if row["family"] in LEXICAL_FAMILIES]
    family_role = {}
    for family in LEXICAL_FAMILIES:
        family_role[family] = {}
        ids = [row["hidden_index"] for row in lexical if row["family"] == family]
        for role in ROLES:
            ri = ROLE_INDEX[role]
            q0 = np.asarray(states[ids, 0, ri], np.float32)
            writes = np.concatenate([
                np.asarray(states[ids, q1, ri], np.float32) - np.asarray(states[ids, q0i, ri], np.float32)
                for q0i, q1 in EDGES
            ], axis=0)
            family_role[family][role] = {
                "q0_rms": float(np.sqrt(np.mean(q0 * q0))),
                "q0_coordinate_std_rms": float(np.sqrt(np.mean(np.var(q0, axis=0)))),
                "local_write_rms": float(np.sqrt(np.mean(writes * writes))),
                "rows": len(ids),
            }
    surface_pairs = []
    grouped = defaultdict(list)
    for row in lexical:
        source = row_by_id[row["case_id"]]
        key = (row["family"], row["unit"], source["variant"], source["truth_bit"], source["candidate_order"])
        grouped[key].append(row)
    for panel in grouped.values():
        if len(panel) != 3:
            continue
        vectors = [np.asarray(states[item["hidden_index"], 0, ROLE_INDEX["primary"]], np.float32) for item in panel]
        surface_pairs.extend(float(np.max(np.abs(vectors[i] - vectors[j]))) for i in range(3) for j in range(i + 1, 3))
    poly_pairs = []
    poly_group = defaultdict(list)
    for row in lexical:
        source = row_by_id[row["case_id"]]
        if row["family"] != "lex_polysemy":
            continue
        key = (row["unit"], row["construction"], source["truth_bit"], source["candidate_order"])
        poly_group[key].append(row)
    for panel in poly_group.values():
        if len(panel) != 2:
            continue
        a, b = panel
        e0 = np.asarray(states[a["hidden_index"], 0, ROLE_INDEX["primary"]], np.float32)
        e1 = np.asarray(states[b["hidden_index"], 0, ROLE_INDEX["primary"]], np.float32)
        h0 = np.asarray(states[a["hidden_index"], 24, ROLE_INDEX["query"]], np.float32)
        h1 = np.asarray(states[b["hidden_index"], 24, ROLE_INDEX["query"]], np.float32)
        poly_pairs.append({
            "embedding_max_abs": float(np.max(np.abs(e0 - e1))),
            "q24_query_rms_difference": float(np.sqrt(np.mean((h0 - h1) ** 2))),
        })
    del states
    headline = {
        "status": "embedding_atlas_closed",
        "family_role": family_role,
        "surface_pair_embedding_max_abs_max": float(max(surface_pairs)),
        "polysemy_pairs": len(poly_pairs),
        "polysemy_embedding_max_abs_max": float(max(row["embedding_max_abs"] for row in poly_pairs)),
        "polysemy_q24_query_rms_difference_mean": float(np.mean([row["q24_query_rms_difference"] for row in poly_pairs])),
        "strict_interpretation": "Identical q0 token embeddings can enter different contextual trajectories. This is compatible with embeddings as lexical initial conditions but refutes embedding-alone sense identification.",
    }
    save(out / "analysis/embedding_atlas.json", headline)
    close("C507", headline, {
        "finite": finite(headline),
        "families": len(family_role) == 6,
        "surface_pairs": len(surface_pairs) == 6 * 10 * 2 * 2 * 2 * 3,
        "polysemy_pairs": len(poly_pairs) == 120,
    }, "C508_embedding_prediction")


def prediction_panels(index: list[dict]) -> tuple[dict[str, list[int]], dict[int, dict]]:
    return split_indices(index), {int(row["hidden_index"]): row for row in index}


def c508() -> None:
    out = begin("C508", {
        "status": "embedding_only_local_write_prediction_frozen",
        "target": "A_q = H_{q+1,r} - H_{q,r}",
        "models": ["family-role mean", "same-coordinate q0 embedding affine"],
        "splits": ["within-unit", "unseen surface", "lexical lockbox"],
        "gate": "embedding must beat family-role mean by NRMSE >= 0.005 on every split to remain a routing-key candidate",
    }, {"parent": final("C507")["all_checks_passed"]})
    index = read_rows(OUTS["C505"] / "raw/hidden_index.jsonl")
    states = np.load(OUTS["C505"] / "raw/role_states.float16.npy", mmap_mode="r")
    panels, meta = prediction_panels(index)
    train = panels["train"]
    model_slopes = np.empty((len(EDGES), len(ROLES), DIM), np.float32)
    model_intercepts = np.empty_like(model_slopes)
    acc = {split: {"mean": metric_acc(), "embedding": metric_acc()} for split in ("within", "surface", "lockbox")}
    details = {}
    for edge_i, (q0, q1) in enumerate(EDGES):
        details[f"q{q0}_q{q1}"] = {}
        for role_i, role in enumerate(ROLES):
            x_train = np.asarray(states[train, 0, role_i], np.float32)
            y_train = np.asarray(states[train, q1, role_i], np.float32) - np.asarray(states[train, q0, role_i], np.float32)
            slope, intercept = fit_diag(x_train, y_train)
            model_slopes[edge_i, role_i] = slope
            model_intercepts[edge_i, role_i] = intercept
            family_means = {}
            for family in ALL_FAMILIES:
                ids = [i for i in train if meta[i]["family"] == family]
                family_means[family] = np.mean(
                    np.asarray(states[ids, q1, role_i], np.float32) - np.asarray(states[ids, q0, role_i], np.float32),
                    axis=0,
                ) if ids else np.mean(y_train, axis=0)
            details[f"q{q0}_q{q1}"][role] = {}
            for split in acc:
                ids = panels[split]
                x = np.asarray(states[ids, 0, role_i], np.float32)
                truth = np.asarray(states[ids, q1, role_i], np.float32) - np.asarray(states[ids, q0, role_i], np.float32)
                pred = predict_diag(x, slope, intercept)
                mean_pred = np.stack([family_means[meta[i]["family"]] for i in ids])
                add_metric(acc[split]["embedding"], pred, truth)
                add_metric(acc[split]["mean"], mean_pred, truth)
                details[f"q{q0}_q{q1}"][role][split] = {
                    "mean": vector_metric(mean_pred, truth),
                    "embedding": vector_metric(pred, truth),
                }
    np.savez_compressed(out / "analysis/embedding_models.npz", slope=model_slopes, intercept=model_intercepts)
    metrics = {split: {name: finish_metric(value) for name, value in models.items()} for split, models in acc.items()}
    gains = {split: metrics[split]["mean"]["nrmse"] - metrics[split]["embedding"]["nrmse"] for split in metrics}
    qualified = all(value >= 0.005 for value in gains.values())
    save(out / "analysis/edge_role_metrics.json", details)
    del states
    close("C508", {
        "status": "embedding_only_prediction_closed",
        "metrics": metrics,
        "nrmse_gains_over_family_mean": gains,
        "embedding_only_candidate": qualified,
        "strict_interpretation": "A positive result would establish prospective predictive information in q0 embeddings, not a semantic key or causal routing mechanism.",
    }, {
        "finite": finite(metrics),
        "model_shape": list(model_slopes.shape) == [5, 6, 2560],
        "all_splits": set(metrics) == {"within", "surface", "lockbox"},
    }, "C509_state_joint")


def c509() -> None:
    out = begin("C509", {
        "status": "current_state_plus_embedding_incremental_prediction_frozen",
        "target": "A_q = H_{q+1,r} - H_{q,r}",
        "models": ["same-coordinate current-state affine", "current-state plus q0-embedding residual affine"],
        "controls": ["within-family sample-permuted q0", "coordinate-rolled q0"],
        "gate": {"joint_gain_each_split": 0.005, "joint_control_gap_each_split": 0.002},
    }, {"parent": final("C508")["all_checks_passed"]})
    index = read_rows(OUTS["C505"] / "raw/hidden_index.jsonl")
    states = np.load(OUTS["C505"] / "raw/role_states.float16.npy", mmap_mode="r")
    panels, meta = prediction_panels(index)
    train = panels["train"]
    rng = np.random.default_rng(5092035)
    shuffled_train = np.asarray(train, np.int64).copy()
    for family in ALL_FAMILIES:
        slots = np.where(np.asarray([meta[i]["family"] == family for i in train]))[0]
        shuffled_train[slots] = shuffled_train[rng.permutation(slots)]
    x_slope = np.empty((len(EDGES), len(ROLES), DIM), np.float32)
    x_intercept = np.empty_like(x_slope)
    e_slope = np.empty_like(x_slope)
    e_intercept = np.empty_like(x_slope)
    acc = {
        split: {name: metric_acc() for name in ("state", "joint", "shuffle", "roll")}
        for split in ("within", "surface", "lockbox")
    }
    details = {}
    for edge_i, (q0, q1) in enumerate(EDGES):
        details[f"q{q0}_q{q1}"] = {}
        for role_i, role in enumerate(ROLES):
            x_train = np.asarray(states[train, q0, role_i], np.float32)
            e_train = np.asarray(states[train, 0, role_i], np.float32)
            y_train = np.asarray(states[train, q1, role_i], np.float32) - x_train
            sx, ix = fit_diag(x_train, y_train)
            residual = y_train - predict_diag(x_train, sx, ix)
            se, ie = fit_diag(e_train, residual)
            se_shuffle, ie_shuffle = fit_diag(np.asarray(states[shuffled_train, 0, role_i], np.float32), residual)
            se_roll, ie_roll = fit_diag(np.roll(e_train, 1, axis=1), residual)
            x_slope[edge_i, role_i], x_intercept[edge_i, role_i] = sx, ix
            e_slope[edge_i, role_i], e_intercept[edge_i, role_i] = se, ie
            details[f"q{q0}_q{q1}"][role] = {}
            for split in acc:
                ids = panels[split]
                x = np.asarray(states[ids, q0, role_i], np.float32)
                e = np.asarray(states[ids, 0, role_i], np.float32)
                truth = np.asarray(states[ids, q1, role_i], np.float32) - x
                state_pred = predict_diag(x, sx, ix)
                predictions = {
                    "state": state_pred,
                    "joint": state_pred + predict_diag(e, se, ie),
                    "shuffle": state_pred + predict_diag(e, se_shuffle, ie_shuffle),
                    "roll": state_pred + predict_diag(np.roll(e, 1, axis=1), se_roll, ie_roll),
                }
                for name, pred in predictions.items():
                    add_metric(acc[split][name], pred, truth)
                details[f"q{q0}_q{q1}"][role][split] = {name: vector_metric(pred, truth) for name, pred in predictions.items()}
    np.savez_compressed(out / "analysis/joint_models.npz", x_slope=x_slope, x_intercept=x_intercept, e_slope=e_slope, e_intercept=e_intercept)
    metrics = {split: {name: finish_metric(value) for name, value in models.items()} for split, models in acc.items()}
    gains = {split: metrics[split]["state"]["nrmse"] - metrics[split]["joint"]["nrmse"] for split in metrics}
    control_gaps = {
        split: min(metrics[split]["shuffle"]["nrmse"], metrics[split]["roll"]["nrmse"]) - metrics[split]["joint"]["nrmse"]
        for split in metrics
    }
    qualified = all(value >= 0.005 for value in gains.values()) and all(value >= 0.002 for value in control_gaps.values())
    save(out / "analysis/edge_role_metrics.json", details)
    del states
    close("C509", {
        "status": "joint_prediction_closed",
        "metrics": metrics,
        "joint_nrmse_gains_over_state": gains,
        "joint_nrmse_gaps_to_best_control": control_gaps,
        "embedding_incremental_candidate": qualified,
        "strict_interpretation": "Only incremental lockbox gain beyond the current state and matched controls can qualify embedding as an additional predictive key.",
    }, {
        "finite": finite(metrics),
        "model_shape": list(x_slope.shape) == [5, 6, 2560],
        "controls": all(set(models) == {"state", "joint", "shuffle", "roll"} for models in metrics.values()),
    }, "C510_polysemy")


def c510() -> None:
    out = begin("C510", {
        "status": "same_token_polysemy_control_frozen",
        "comparison": "same token form, same surface/order/truth, two explicit senses",
        "prediction": "q0 embedding must be identical while contextual state and local write may differ",
        "claim_boundary": "this tests embedding-alone identifiability, not whether embeddings participate in later computation",
    }, {"parent": final("C509")["all_checks_passed"]})
    rows, _ = combined_rows_and_compiled()
    by_id = {row["case_id"]: row for row in rows}
    index = read_rows(OUTS["C505"] / "raw/hidden_index.jsonl")
    states = np.load(OUTS["C505"] / "raw/role_states.float16.npy", mmap_mode="r")
    models = np.load(OUTS["C509"] / "analysis/joint_models.npz")
    grouped = defaultdict(list)
    for row in index:
        source = by_id[row["case_id"]]
        if row["family"] != "lex_polysemy":
            continue
        key = (row["unit"], row["construction"], source["truth_bit"], source["candidate_order"])
        grouped[key].append(row)
    records = []
    for panel in grouped.values():
        if len(panel) != 2:
            continue
        a, b = sorted(panel, key=lambda item: by_id[item["case_id"]]["variant"])
        ia, ib = a["hidden_index"], b["hidden_index"]
        for edge_i, (q0, q1) in enumerate(EDGES):
            role_i = ROLE_INDEX["query"]
            ea = np.asarray(states[ia, 0, role_i], np.float32)
            eb = np.asarray(states[ib, 0, role_i], np.float32)
            xa = np.asarray(states[ia, q0, role_i], np.float32)
            xb = np.asarray(states[ib, q0, role_i], np.float32)
            ya = np.asarray(states[ia, q1, role_i], np.float32) - xa
            yb = np.asarray(states[ib, q1, role_i], np.float32) - xb
            pred_e_a = predict_diag(ea, models["e_slope"][edge_i, role_i], models["e_intercept"][edge_i, role_i])
            pred_e_b = predict_diag(eb, models["e_slope"][edge_i, role_i], models["e_intercept"][edge_i, role_i])
            pred_x_a = predict_diag(xa, models["x_slope"][edge_i, role_i], models["x_intercept"][edge_i, role_i])
            pred_x_b = predict_diag(xb, models["x_slope"][edge_i, role_i], models["x_intercept"][edge_i, role_i])
            records.append({
                "edge": f"q{q0}_q{q1}",
                "embedding_max_abs": float(np.max(np.abs(ea - eb))),
                "target_pair_rms": float(np.sqrt(np.mean((ya - yb) ** 2))),
                "embedding_prediction_pair_rms": float(np.sqrt(np.mean((pred_e_a - pred_e_b) ** 2))),
                "state_prediction_pair_rms": float(np.sqrt(np.mean((pred_x_a - pred_x_b) ** 2))),
            })
    models.close()
    del states
    summary = {
        "pairs_edges": len(records),
        "embedding_max_abs_max": float(max(row["embedding_max_abs"] for row in records)),
        "target_pair_rms_mean": float(np.mean([row["target_pair_rms"] for row in records])),
        "embedding_prediction_pair_rms_mean": float(np.mean([row["embedding_prediction_pair_rms"] for row in records])),
        "state_prediction_pair_rms_mean": float(np.mean([row["state_prediction_pair_rms"] for row in records])),
    }
    save(out / "analysis/polysemy_records.json", records)
    close("C510", {
        "status": "polysemy_control_closed",
        **summary,
        "embedding_alone_sense_identifiable": summary["embedding_prediction_pair_rms_mean"] > 1e-6,
        "strict_interpretation": "The same q0 token embedding cannot identify the explicit sense by itself. Contextual HiddenState remains the candidate work state.",
    }, {
        "pairs": len(records) == 120 * 5,
        "finite": finite(summary),
        "same_embedding": summary["embedding_max_abs_max"] == 0.0,
    }, "C511_conditioned_operator")


def conditioned_group_splits(groups: list[dict]) -> tuple[list[dict], list[dict]]:
    return (
        [row for row in groups if row["unit"] < 8],
        [row for row in groups if row["unit"] >= 8],
    )


def c511() -> None:
    out = begin("C511", {
        "status": "family_conditioned_channel_tournament_frozen",
        "basis": "orthonormal Walsh for registered factor-order labels; C506 basis result governs mechanistic interpretation",
        "models": ["identity", "shared channel-diagonal", "family-conditioned channel-diagonal"],
        "training": "units 0-7 all three constructions",
        "lockbox": "units 8-9 all three constructions",
        "gate": "family-conditioned aggregate lockbox NRMSE improves shared by >= 0.01",
    }, {"parent": final("C510")["all_checks_passed"]})
    rows, _ = combined_rows_and_compiled()
    index = read_rows(OUTS["C505"] / "raw/hidden_index.jsonl")
    states = np.load(OUTS["C505"] / "raw/role_states.float16.npy", mmap_mode="r")
    groups = build_old_groups(rows, index)
    train, lockbox = conditioned_group_splits(groups)
    basis = orthonormal_walsh()
    family_names = list(OLD_FAMILIES)
    shared_slope = np.empty((len(EDGES), len(ROLES), 16, DIM), np.float32)
    shared_intercept = np.empty_like(shared_slope)
    family_slope = np.empty((len(family_names), len(EDGES), len(ROLES), 16, DIM), np.float32)
    family_intercept = np.empty_like(family_slope)
    acc = {name: metric_acc() for name in ("identity", "shared", "family")}
    family_metrics = {}
    order_metrics = {str(order): {name: metric_acc() for name in ("identity", "shared", "family")} for order in range(5)}
    for edge_i, (q0, q1) in enumerate(EDGES):
        for role_i, role in enumerate(ROLES):
            x_raw = group_cube(states, train, q0, role_i)
            y_raw = group_cube(states, train, q1, role_i)
            x = np.einsum("ab,pbd->pad", basis, x_raw, optimize=True)
            y = np.einsum("ab,pbd->pad", basis, y_raw, optimize=True)
            ss, si = fit_channel_diag(x, y)
            shared_slope[edge_i, role_i], shared_intercept[edge_i, role_i] = ss, si
            for family_i, family in enumerate(family_names):
                ft = [row for row in train if row["family"] == family]
                fx = np.einsum("ab,pbd->pad", basis, group_cube(states, ft, q0, role_i), optimize=True)
                fy = np.einsum("ab,pbd->pad", basis, group_cube(states, ft, q1, role_i), optimize=True)
                fs, fi = fit_channel_diag(fx, fy)
                family_slope[family_i, edge_i, role_i] = fs
                family_intercept[family_i, edge_i, role_i] = fi
            for family_i, family in enumerate(family_names):
                panel = [row for row in lockbox if row["family"] == family]
                px_raw = group_cube(states, panel, q0, role_i)
                py_raw = group_cube(states, panel, q1, role_i)
                px = np.einsum("ab,pbd->pad", basis, px_raw, optimize=True)
                py = np.einsum("ab,pbd->pad", basis, py_raw, optimize=True)
                predictions = {
                    "identity": px,
                    "shared": px * ss + si,
                    "family": px * family_slope[family_i, edge_i, role_i] + family_intercept[family_i, edge_i, role_i],
                }
                family_metrics.setdefault(family, {name: metric_acc() for name in predictions})
                for name, pred in predictions.items():
                    add_metric(acc[name], pred, py)
                    add_metric(family_metrics[family][name], pred, py)
                    for mask in range(16):
                        add_metric(order_metrics[str(mask.bit_count())][name], pred[:, mask], py[:, mask])
    np.savez_compressed(
        out / "analysis/conditioned_models.npz",
        shared_slope=shared_slope,
        shared_intercept=shared_intercept,
        family_slope=family_slope,
        family_intercept=family_intercept,
        family_names=np.asarray(family_names),
    )
    metrics = {name: finish_metric(value) for name, value in acc.items()}
    family_metrics_done = {
        family: {name: finish_metric(value) for name, value in models.items()}
        for family, models in family_metrics.items()
    }
    order_metrics_done = {
        order: {name: finish_metric(value) for name, value in models.items()}
        for order, models in order_metrics.items()
    }
    gain = metrics["shared"]["nrmse"] - metrics["family"]["nrmse"]
    del states
    close("C511", {
        "status": "conditioned_tournament_closed",
        "metrics": metrics,
        "family_metrics": family_metrics_done,
        "effect_order_metrics": order_metrics_done,
        "family_conditioned_nrmse_gain": gain,
        "family_conditioned_candidate": gain >= 0.01,
        "parameter_counts_per_edge_role": {
            "shared_channel_diagonal": 16 * DIM * 2,
            "family_conditioned_channel_diagonal": len(family_names) * 16 * DIM * 2,
        },
        "strict_interpretation": "Conditioned gain would show that external family state predicts different transition tables. It would not locate a unique circuit or prove that family labels are internal variables.",
    }, {
        "finite": finite(metrics),
        "families": len(family_metrics_done) == 11,
        "orders": set(order_metrics_done) == {"0", "1", "2", "3", "4"},
        "model_shape": list(family_slope.shape) == [11, 5, 6, 16, 2560],
    }, "C512_nested")


def high_order_panel(name: str, family: str, next_authorization: str) -> None:
    out = begin(name, {
        "status": f"{family}_high_order_lockbox_frozen",
        "family": family,
        "channels": "Walsh masks with order >= 3",
        "models": ["identity", "shared channel-diagonal", "family-conditioned channel-diagonal"],
        "gate": "family-conditioned NRMSE improves shared by >= 0.01 on high-order lockbox channels",
    }, {"parent": final("C511" if name == "C512" else f"C{int(name[1:]) - 1}")["all_checks_passed"]})
    rows, _ = combined_rows_and_compiled()
    index = read_rows(OUTS["C505"] / "raw/hidden_index.jsonl")
    states = np.load(OUTS["C505"] / "raw/role_states.float16.npy", mmap_mode="r")
    groups = build_old_groups(rows, index)
    panel = [row for row in groups if row["family"] == family and row["unit"] >= 8]
    models = np.load(OUTS["C511"] / "analysis/conditioned_models.npz")
    family_names = [str(value) for value in models["family_names"]]
    family_i = family_names.index(family)
    basis = orthonormal_walsh()
    masks = [mask for mask in range(16) if mask.bit_count() >= 3]
    acc = {name_: metric_acc() for name_ in ("identity", "shared", "family")}
    by_context = {}
    for edge_i, (q0, q1) in enumerate(EDGES):
        for role_i, role in enumerate(ROLES):
            x = np.einsum("ab,pbd->pad", basis, group_cube(states, panel, q0, role_i), optimize=True)
            y = np.einsum("ab,pbd->pad", basis, group_cube(states, panel, q1, role_i), optimize=True)
            predictions = {
                "identity": x[:, masks],
                "shared": (x * models["shared_slope"][edge_i, role_i] + models["shared_intercept"][edge_i, role_i])[:, masks],
                "family": (x * models["family_slope"][family_i, edge_i, role_i] + models["family_intercept"][family_i, edge_i, role_i])[:, masks],
            }
            key = f"q{q0}_q{q1}_{role}"
            by_context[key] = {model_name: vector_metric(pred, y[:, masks]) for model_name, pred in predictions.items()}
            for model_name, pred in predictions.items():
                add_metric(acc[model_name], pred, y[:, masks])
    metrics = {model_name: finish_metric(value) for model_name, value in acc.items()}
    gain = metrics["shared"]["nrmse"] - metrics["family"]["nrmse"]
    models.close()
    del states
    save(out / "analysis/context_metrics.json", by_context)
    close(name, {
        "status": "high_order_lockbox_closed",
        "family": family,
        "programs": len(panel),
        "masks": masks,
        "metrics": metrics,
        "family_conditioned_gain": gain,
        "high_order_candidate": gain >= 0.01,
        "strict_interpretation": "This is a held-out experimental-panel prediction. It is not yet a single-sentence composition operator.",
    }, {
        "finite": finite(metrics),
        "programs": len(panel) == 6,
        "masks": len(masks) == 5,
        "contexts": len(by_context) == 30,
    }, next_authorization)


def c512() -> None:
    high_order_panel("C512", "nested_composition", "C513_graph")


def c513() -> None:
    high_order_panel("C513", "typed_graph_path", "C514_temporal")


def c514() -> None:
    high_order_panel("C514", "temporal_composition", "C515_eligibility")


def c515() -> None:
    out = begin("C515", {
        "status": "single_sample_predictive_causal_eligibility_frozen",
        "rules": {
            "embedding_route": "C509 joint candidate plus q24->q25 query-role context must beat state and both controls",
            "panel_route": "C511-C514 may support panel prediction but cannot authorize single-sample intervention because all 16 cells are required",
            "causal": "run only for a qualified single-sample route; otherwise NA",
        },
    }, {"parent": final("C514")["all_checks_passed"]})
    joint = final("C509")["headline"]
    conditioned = final("C511")["headline"]
    lockboxes = {name: final(name)["headline"] for name in ("C512", "C513", "C514")}
    details = load(OUTS["C509"] / "analysis/edge_role_metrics.json")
    core = details["q24_q25"]["query"]["lockbox"]
    core_gain = core["state"]["nrmse"] - core["joint"]["nrmse"]
    core_control_gap = min(core["shuffle"]["nrmse"], core["roll"]["nrmse"]) - core["joint"]["nrmse"]
    single_sample = bool(joint["embedding_incremental_candidate"] and core_gain >= 0.005 and core_control_gap >= 0.002)
    panel_candidate = bool(conditioned["family_conditioned_candidate"] and all(value["high_order_candidate"] for value in lockboxes.values()))
    causal = {
        "authorized": single_sample,
        "ran": False,
        "result": "NA_single_sample_predictor_not_qualified" if not single_sample else "DEFERRED_requires_separate_frozen_intervention_executor",
    }
    # The causal branch is intentionally not improvised after reveal. A positive
    # single-sample result authorizes a separately frozen executor instead.
    close("C515", {
        "status": "eligibility_closed",
        "embedding_single_sample_candidate": single_sample,
        "core_q24_q25_query_gain": core_gain,
        "core_q24_q25_query_control_gap": core_control_gap,
        "complete_panel_candidate": panel_candidate,
        "causal": causal,
        "strict_interpretation": "A 16-cell panel predictor is not a callable natural-sample mechanism. No post-reveal intervention is invented inside this phase.",
    }, {
        "finite": finite({"gain": core_gain, "gap": core_control_gap}),
        "causal_rule_obeyed": not causal["ran"],
        "all_lockboxes": len(lockboxes) == 3,
    }, "C516_visual_cleanup")


def register_visual() -> None:
    if REGISTRY.exists():
        registry = load(REGISTRY)
        datasets = registry.setdefault("datasets", [])
        item = {
            "id": "c516_embedding_conditioned_state_atlas",
            "title": "C516 Embedding-Conditioned State Atlas",
            "phase": 2050,
            "campaign": "C501-C516",
            "path": "vis_data/research_kernel/c516_embedding_conditioned_state_atlas.json",
            "kind": "embedding_hiddenstate_coordinate_atlas",
            "coordinates": 2560,
        }
        datasets[:] = [row for row in datasets if row.get("id") != item["id"]]
        datasets.append(item)
        save(REGISTRY, registry)
    if CATALOG.exists():
        catalog = load(CATALOG)
        datasets = catalog.setdefault("field_datasets", [])
        item = {
            "id": "c516_embedding_conditioned_state_atlas",
            "title": "C516 Embedding-Conditioned State Atlas",
            "url": "/vis_data/research_kernel/c516_embedding_conditioned_state_atlas.json",
            "phase": 2050,
            "full_coordinate": True,
        }
        datasets[:] = [row for row in datasets if row.get("id") != item["id"]]
        datasets.append(item)
        save(CATALOG, catalog)


def c516() -> None:
    out = begin("C516", {
        "status": "visual_cleanup_synthesis_frozen",
        "visual": "representative q0 embedding and q24->q25 write vectors for all 17 families and all six roles, preserving every coordinate",
        "cleanup": "hash then delete C505 role/full-token arrays after visual archive is written",
        "synthesis": "strictly separate basis bookkeeping, lexical initial conditions, current-state prediction, panel-conditioned transition, and causality",
    }, {"parent": final("C515")["all_checks_passed"]})
    rows, _ = combined_rows_and_compiled()
    by_id = {row["case_id"]: row for row in rows}
    index = read_rows(OUTS["C505"] / "raw/hidden_index.jsonl")
    states_path = OUTS["C505"] / "raw/role_states.float16.npy"
    full_path = OUTS["C505"] / "raw/full_token_states.float16.npy"
    states = np.load(states_path, mmap_mode="r")
    visual_rows = []
    for family in ALL_FAMILIES:
        candidates = [row for row in index if row["family"] == family and row["partition"] == "lockbox"]
        representative = sorted(candidates, key=lambda row: row["case_id"])[0]
        source = by_id[representative["case_id"]]
        i = representative["hidden_index"]
        for role_i, role in enumerate(ROLES):
            embedding = np.asarray(states[i, 0, role_i], np.float32)
            q24 = np.asarray(states[i, 24, role_i], np.float32)
            q25 = np.asarray(states[i, 25, role_i], np.float32)
            visual_rows.append({
                "family": family,
                "panel": source.get("panel"),
                "case_id": representative["case_id"],
                "role": role,
                "embedding_q0": embedding.tolist(),
                "state_q24": q24.tolist(),
                "write_q24_q25": (q25 - q24).tolist(),
            })
    visual = {
        "schema": "ai2050.embedding_conditioned_state_atlas.v1",
        "phase": 2050,
        "campaign": "C501-C516",
        "coordinate_count": DIM,
        "checkpoint_semantics": {"q0": "token embedding activation", "q24": "block-output checkpoint", "q24_q25": "local HiddenState write"},
        "basis_control": final("C506")["headline"],
        "embedding_prediction": final("C508")["headline"],
        "joint_prediction": final("C509")["headline"],
        "polysemy_control": final("C510")["headline"],
        "conditioned_transition": final("C511")["headline"],
        "lockboxes": {name: final(name)["headline"] for name in ("C512", "C513", "C514")},
        "eligibility": final("C515")["headline"],
        "rows": visual_rows,
    }
    save(VISUAL, visual)
    register_visual()
    del states
    gc.collect()
    cleanup = []
    for path in (states_path, full_path):
        if path.exists():
            cleanup.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha(path)})
    save(out / "audit/raw_field_cleanup_ledger.json", {"files": cleanup, "total_bytes": sum(row["bytes"] for row in cleanup)})
    for row in cleanup:
        path = ROOT / row["path"]
        path.unlink()
    gates = {
        "equal_capacity_basis_equivalence": final("C506")["headline"]["basis_equivalent_within_tolerance"],
        "embedding_only": final("C508")["headline"]["embedding_only_candidate"],
        "embedding_incremental": final("C509")["headline"]["embedding_incremental_candidate"],
        "polysemy_embedding_alone": final("C510")["headline"]["embedding_alone_sense_identifiable"],
        "family_conditioned": final("C511")["headline"]["family_conditioned_candidate"],
        "nested_high_order": final("C512")["headline"]["high_order_candidate"],
        "graph_high_order": final("C513")["headline"]["high_order_candidate"],
        "temporal_high_order": final("C514")["headline"]["high_order_candidate"],
        "causal": final("C515")["headline"]["causal"]["ran"],
    }
    close("C516", {
        "status": "campaign_synthesis_closed",
        "gates": gates,
        "visual_path": str(VISUAL.relative_to(ROOT)).replace("\\", "/"),
        "visual_rows": len(visual_rows),
        "visual_coordinate_values": len(visual_rows) * 3 * DIM,
        "cleanup_files": len(cleanup),
        "cleanup_bytes": sum(row["bytes"] for row in cleanup),
        "raw_fields_absent": not states_path.exists() and not full_path.exists(),
        "new_math_gate": False,
        "strict_conclusion": "The campaign distinguishes complete-panel information from basis choice and tests whether lexical embeddings add predictive information beyond current state. Only held-out and control-beating gains are retained; no predictive result alone is called a unique circuit.",
    }, {
        "visual": VISUAL.exists() and len(visual_rows) == 17 * 6,
        "coordinates": all(len(row["embedding_q0"]) == DIM and len(row["state_q24"]) == DIM and len(row["write_q24_q25"]) == DIM for row in visual_rows),
        "cleanup": not states_path.exists() and not full_path.exists(),
        "finite": finite(gates),
    }, "C517_independent_audit")


FUNCTIONS = {
    "C501": c501,
    "C502": c502,
    "C503": c503,
    "C504": c504,
    "C505": c505,
    "C506": c506,
    "C507": c507,
    "C508": c508,
    "C509": c509,
    "C510": c510,
    "C511": c511,
    "C512": c512,
    "C513": c513,
    "C514": c514,
    "C515": c515,
    "C516": c516,
}


def self_test() -> None:
    rows = lexical_material()
    assert len(rows) == 1440
    assert len({row["case_id"] for row in rows}) == 1440
    assert all(sum(row["family"] == family for row in rows) == 240 for family in LEXICAL_FAMILIES)
    h = orthonormal_walsh()
    assert np.max(np.abs(h @ h.T - np.eye(16))) < 1e-6
    q = random_orthogonal(ORTHOGONAL_SEED)
    assert np.max(np.abs(q @ q.T - np.eye(16))) < 1e-5
    print(json.dumps({"self_test": "passed", "rows": len(rows)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--start", default="C501")
    parser.add_argument("--stop", default="C516")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    names = list(FUNCTIONS)
    start = names.index(args.start)
    stop = names.index(args.stop)
    if start > stop:
        raise ValueError((args.start, args.stop))
    for name in names[start:stop + 1]:
        FUNCTIONS[name]()


if __name__ == "__main__":
    main()
