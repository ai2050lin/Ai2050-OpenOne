"""Post-hoc paired failure topology for sealed Phase996 raw results.

Descriptive only: no thresholds are changed and no internal observation is
authorized.  The analysis reconstructs depth patterns, candidate/natural
disagreement, interface transitions, and value-specific accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import phase996_external_semantic_confirmation as frozen

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "tests/glm5/result/phase996_external_semantic_confirmation_protocol"
EXECUTION = ROOT / "tests/glm5/result/phase996_external_semantic_confirmation_execution"
OUT = EXECUTION / "scores/posthoc_failure_topology.json"
MODELS = frozen.MODEL_ORDER
DEPTHS = frozen.DEPTHS
INTERFACES = frozen.INTERFACES
VALUES = frozen.VALUES


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha_json(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def sealed(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    out = deepcopy(dict(value)); out[field] = sha_json(out); return out


def cases(model: str) -> list[dict[str, Any]]:
    tokenizer = frozen.base.engine._load_inspection_bundle(model).tokenizer
    truth = {row["record_id"]: row for row in
             (json.loads(line) for line in (PROTOCOL / "dataset/private_truth.jsonl").read_text(encoding="utf-8").splitlines())}
    raw = frozen.raw_rows(EXECUTION / f"raw/public/{model}.jsonl.gz")
    out: list[dict[str, Any]] = []
    for row in raw:
        gold = str(truth[row["record_id"]]["gold"])
        budgets = [64] + ([128] if row["max_new_tokens"] == 128 else [])
        before = list(row["generated_token_ids_before_eos"])
        logits = {key: float(value) for key, value in row["candidate_logits"].items()}
        candidate = max(VALUES, key=lambda value: logits[value])
        for budget in budgets:
            text = tokenizer.decode(before[:budget], skip_special_tokens=False,
                                    clean_up_tokenization_spaces=False)
            parsed = frozen.parse(text)["value"]
            out.append({"world": row["semantic_world_id"], "ordinal": row["world_ordinal"],
                        "split": row["split"], "transform": row["semantic_transform"],
                        "depth": row["depth"], "interface": row["interface_variant"],
                        "budget": budget, "gold": gold, "natural": parsed == gold,
                        "candidate": candidate == gold, "parsed": parsed is not None})
    return out


def topology(rows: Sequence[Mapping[str, Any]], interface: str, budget: int) -> dict[str, Any] | None:
    selected = [row for row in rows if row["interface"] == interface and row["budget"] == budget]
    grouped: dict[tuple[str, str], dict[str, bool]] = defaultdict(dict)
    for row in selected:
        grouped[(str(row["world"]), str(row["transform"]))][str(row["depth"])] = bool(row["natural"])
    complete = {key: value for key, value in grouped.items() if set(value) == set(DEPTHS)}
    if not complete:
        return None
    patterns = Counter("".join("1" if value[depth] else "0" for depth in DEPTHS)
                       for value in complete.values())
    monotone = sum(value[DEPTHS[0]] >= value[DEPTHS[1]] >= value[DEPTHS[2]]
                   for value in complete.values())
    return {"n_paired_units": len(complete), "pattern_order": list(DEPTHS),
            "patterns": dict(sorted(patterns.items())), "monotone_units": monotone,
            "monotone_rate": monotone / len(complete),
            "one_correct_copy_wrong": sum(value[DEPTHS[1]] and not value[DEPTHS[0]] for value in complete.values()),
            "two_correct_one_wrong": sum(value[DEPTHS[2]] and not value[DEPTHS[1]] for value in complete.values()),
            "two_correct_copy_wrong": sum(value[DEPTHS[2]] and not value[DEPTHS[0]] for value in complete.values())}


def candidate_natural(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for interface in INTERFACES:
        for depth in DEPTHS:
            for budget in (64, 128):
                cell = [row for row in rows if row["interface"] == interface
                        and row["depth"] == depth and row["budget"] == budget]
                if not cell: continue
                counts = Counter(f"candidate_{int(row['candidate'])}|natural_{int(row['natural'])}" for row in cell)
                output[f"{interface}|{depth}|{budget}"] = {"n": len(cell), "matrix": dict(sorted(counts.items()))}
    return output


def interface_transitions(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    subset = [row for row in rows if int(row["ordinal"]) % 4 == 0 and row["budget"] == 64]
    keyed = {(row["world"], row["transform"], row["depth"], row["interface"]): bool(row["natural"])
             for row in subset}
    output: dict[str, Any] = {}
    for target in INTERFACES:
        if target == "raw_plain": continue
        for depth in DEPTHS:
            pairs = []
            for row in subset:
                if row["interface"] != "raw_plain" or row["depth"] != depth: continue
                key = (row["world"], row["transform"], depth, target)
                if key in keyed: pairs.append((bool(row["natural"]), keyed[key]))
            counts = Counter(f"raw_{int(a)}|target_{int(b)}" for a, b in pairs)
            output[f"raw_plain->{target}|{depth}"] = {"n": len(pairs), "matrix": dict(sorted(counts.items()))}
    return output


def value_accuracy(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for interface in INTERFACES:
        for depth in DEPTHS:
            cell = [row for row in rows if row["interface"] == interface and row["depth"] == depth and row["budget"] == 64]
            if not cell: continue
            output[f"{interface}|{depth}|64"] = {
                value: {"n": sum(row["gold"] == value for row in cell),
                        "correct": sum(row["gold"] == value and row["natural"] for row in cell)}
                for value in VALUES}
    return output


def analyze() -> dict[str, Any]:
    score = json.loads((EXECUTION / "scores/public_score.json").read_text(encoding="utf-8"))
    reports: dict[str, Any] = {}
    for model in MODELS:
        rows = cases(model)
        depth = {}
        for interface in frozen.PRIMARY_INTERFACES:
            for budget in (64, 128):
                result = topology(rows, interface, budget)
                if result is not None: depth[f"{interface}|{budget}"] = result
        reports[model] = {"depth_topology": depth, "candidate_natural": candidate_natural(rows),
                          "interface_transitions": interface_transitions(rows),
                          "value_accuracy": value_accuracy(rows)}
    result = sealed({"schema_version": "phase996_posthoc_failure_topology.v1", "phase": 996,
                     "created_at_utc": datetime.now(timezone.utc).isoformat(), "post_hoc": True,
                     "descriptive_only": True, "admission_changed": False,
                     "score_sha256": score["score_sha256"], "models": reports,
                     "internal_observation_authorized": False}, "analysis_sha256")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("xb") as handle: handle.write(canonical(result)); handle.flush()
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--analyze", action="store_true", required=True)
    parser.parse_args(argv); result = analyze()
    print(json.dumps({"passed": True, "analysis_sha256": result["analysis_sha256"],
                      "score_sha256": result["score_sha256"]}, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
