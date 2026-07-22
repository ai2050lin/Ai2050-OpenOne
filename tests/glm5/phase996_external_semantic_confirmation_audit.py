"""Independent raw-token recomputation for Phase996.

This source does not import the Phase996 scorer.  It reconstructs parsed
answers, candidate decisions, integer gates and depth gradients directly from
the sealed raw rows and private truth after the public stage is complete.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import phase983_cross_model_engine as engine

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "tests/glm5/result/phase996_external_semantic_confirmation_protocol"
EXECUTION = ROOT / "tests/glm5/result/phase996_external_semantic_confirmation_execution"
MODELS = ("qwen3", "glm4", "deepseek7b")
DEPTHS = ("copy_control", "one_hop", "two_hop")
TRANSFORMS = ("original", "value_swap", "binding_swap", "query_swap")
PRIMARY = ("raw_plain", "native_full")
INTERFACES = PRIMARY + ("raw_answer_scaffold", "native_role_only", "raw_plus_native_prefill")
VALUES = ("amber", "silver", "violet", "ivory")
PUBLIC_ROWS = 8448
VALUE_RE = re.compile(r"(?<![A-Za-z])(amber|silver|violet|ivory)(?![A-Za-z])", re.I)
SCAFFOLD_RE = re.compile(r"The\s+retrieved\s+marker\s+is\s+(amber|silver|violet|ivory)\s*\.", re.I)
STRICT_RE = re.compile(r"^\s*The\s+retrieved\s+marker\s+is\s+(amber|silver|violet|ivory)\s*\.\s*$", re.I)


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha_json(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def sealed(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    out = deepcopy(dict(value)); require(field not in out, "self hash collision")
    out[field] = sha_json(out); return out


def verify(value: Mapping[str, Any], field: str, label: str) -> None:
    body = {k: v for k, v in value.items() if k != field}
    require(value.get(field) == sha_json(body), f"{label} self hash drift")


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8")); require(isinstance(value, dict), f"not object: {path}")
    return value


def parse(text: str) -> tuple[str | None, bool]:
    full = list(SCAFFOLD_RE.finditer(text))
    if full: value = full[-1].group(1).lower()
    else:
        values = sorted({m.group(1).lower() for m in VALUE_RE.finditer(text)})
        value = values[0] if len(values) == 1 else None
    return value, bool(STRICT_RE.fullmatch(text))


def rows(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def independent_gate(cell: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    split_transform = {}
    for split in ("confirmation_a", "confirmation_b"):
        for transform in TRANSFORMS:
            key = f"{split}|{transform}"
            split_transform[key] = sum(bool(row["correct"]) for row in cell
                                       if row["split"] == split and row["transform"] == transform)
    transform_counts = {transform: sum(bool(row["correct"]) for row in cell if row["transform"] == transform)
                        for transform in TRANSFORMS}
    by_world: dict[str, list[bool]] = defaultdict(list)
    for row in cell: by_world[str(row["world"])].append(bool(row["correct"]))
    all_four = sum(len(values) == 4 and all(values) for values in by_world.values())
    passed = min(split_transform.values()) >= 116 and min(transform_counts.values()) >= 232 and all_four >= 218
    return {"passed": passed, "split_transform_correct": split_transform,
            "transform_correct": transform_counts, "all_four_worlds": all_four,
            "world_denominator": len(by_world)}


def audit() -> dict[str, Any]:
    activation = load(PROTOCOL / "activation.json"); verify(activation, "activation_sha256", "activation")
    stage = load(EXECUTION / "public_stage.json"); verify(stage, "stage_sha256", "public stage")
    score = load(EXECUTION / "scores/public_score.json"); verify(score, "score_sha256", "score")
    truth = {row["record_id"]: row for row in
             (json.loads(line) for line in (PROTOCOL / "dataset/private_truth.jsonl").read_text(encoding="utf-8").splitlines())}
    require(len(truth) == PUBLIC_ROWS, "truth count drift")
    summaries: dict[str, Any] = {}; gates: dict[str, Any] = {}; gradients: dict[str, Any] = {}
    for model in MODELS:
        tokenizer = engine.load_tokenizer_inspection(model).tokenizer
        raw = rows(EXECUTION / f"raw/public/{model}.jsonl.gz")
        require(len(raw) == PUBLIC_ROWS, f"raw count drift: {model}")
        cases: list[dict[str, Any]] = []
        for row in raw:
            gold = str(truth[row["record_id"]]["gold"])
            budgets = [64] + ([128] if row["max_new_tokens"] == 128 else [])
            before = list(row["generated_token_ids_before_eos"])
            for budget in budgets:
                text = tokenizer.decode(before[:budget], skip_special_tokens=False,
                                        clean_up_tokenization_spaces=False)
                parsed, strict = parse(text)
                logits = {key: float(value) for key, value in row["candidate_logits"].items()}
                candidate = max(VALUES, key=lambda value: logits[value])
                margin = logits[gold] - max(logits[value] for value in VALUES if value != gold)
                cases.append({"world": row["semantic_world_id"], "split": row["split"],
                              "transform": row["semantic_transform"], "interface": row["interface_variant"],
                              "depth": row["depth"], "budget": budget, "parsed": parsed,
                              "correct": parsed == gold, "strict": strict,
                              "eos": bool(row["eos_seen"] and row["first_eos_index"] < budget),
                              "candidate_correct": candidate == gold, "candidate_margin": margin})
        model_summary: dict[str, Any] = {}
        for interface in INTERFACES:
            for depth in DEPTHS:
                for budget in (64, 128):
                    cell = [row for row in cases if row["interface"] == interface
                            and row["depth"] == depth and row["budget"] == budget]
                    if not cell: continue
                    key = f"{interface}|{depth}|{budget}"
                    model_summary[key] = {
                        "n": len(cell), "correct": sum(row["correct"] for row in cell),
                        "parsed": sum(row["parsed"] is not None for row in cell),
                        "strict": sum(row["strict"] for row in cell), "eos": sum(row["eos"] for row in cell),
                        "candidate_correct": sum(row["candidate_correct"] for row in cell),
                        "candidate_margin_mean": sum(row["candidate_margin"] for row in cell) / len(cell),
                    }
                    if interface in PRIMARY and len(cell) == 1024:
                        gates[f"{model}|{key}"] = independent_gate(cell)
        summaries[model] = model_summary
        for interface in PRIMARY:
            for budget in (64, 128):
                keys = [f"{interface}|{depth}|{budget}" for depth in DEPTHS]
                if all(key in model_summary for key in keys):
                    counts = [model_summary[key]["correct"] for key in keys]
                    gradients[f"{model}|{interface}|{budget}"] = {
                        "counts": counts, "copy_ge_one_ge_two": counts[0] >= counts[1] >= counts[2],
                    }
        del tokenizer, raw, cases
    two_hop = sorted(key for key, value in gates.items() if "|two_hop|" in key and value["passed"])
    require(summaries == score["summaries"], "independent summaries differ from score")
    require(gates == score["gates"], "independent gates differ from score")
    require(gradients == score["depth_gradients"], "independent gradients differ from score")
    require(two_hop == score["two_hop_passes"], "independent two-hop decision differs")
    result = sealed({"schema_version": "phase996_independent_audit.v1", "phase": 996,
                     "created_at_utc": datetime.now(timezone.utc).isoformat(), "passed": True,
                     "public_stage_sha256": stage["stage_sha256"], "score_sha256": score["score_sha256"],
                     "raw_rows_recomputed": PUBLIC_ROWS * len(MODELS), "summary_match": True,
                     "gate_match": True, "gradient_match": True, "two_hop_passes": two_hop,
                     "internal_observation_authorized": False}, "audit_sha256")
    target = EXECUTION / "scores/public_independent_audit.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as handle: handle.write(canonical(result)); handle.flush()
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--audit", action="store_true", required=True)
    parser.parse_args(argv); print(json.dumps(audit(), sort_keys=True, ensure_ascii=False)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
