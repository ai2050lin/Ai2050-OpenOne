from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase1171_fixed_dimension_formation_trajectory_tomography as p1171  # noqa: E402
import phase1187_typed_evidence_compiler as p1187  # noqa: E402
import phase1189_quotient_formation_operator_calibration as p1189  # noqa: E402
import phase1190_natural_sgd_quotient_transition as p1190  # noqa: E402


PHASE = 1191
SCRIPT = Path(__file__).resolve()
AUDIT_SCRIPT = ROOT / "tests/glm5/phase1191_prefix_future_formation_identity_audit.py"
OUT_ROOT = ROOT / "tests/glm5/result/phase1191_prefix_future_formation_identity"
DEVELOPMENT_ROWS = OUT_ROOT / "development/rows.jsonl"
DEVELOPMENT_SUMMARY = OUT_ROOT / "development/summary.json"
PROTOCOL_PATH = OUT_ROOT / "protocol/preregistration.json"
RAW_ROWS = OUT_ROOT / "analysis/rows.jsonl"
SUMMARY_PATH = OUT_ROOT / "analysis/summary.json"
CLAIMS_PATH = OUT_ROOT / "analysis/typed_claims.json"
FINAL_PATH = OUT_ROOT / "analysis/final.json"
AUDIT_PATH = OUT_ROOT / "audit/independent_audit.json"

DEVELOPMENT_SOURCE = p1171.OUT_ROOT / "runs/training/checkpoints"
FORMAL_SOURCE = p1190.CHECKPOINT_ROOT
STEPS = (25, 150, 1000, 10_000)
POSITIVE_THRESHOLDS = {
    "endpoint_advantage_mean_min": 0.10,
    "endpoint_positive_fraction_min": 0.65,
    "positive_task_count_per_split_min": 3,
    "positive_task_count_development_min": 6,
}
NEGATIVE_THRESHOLDS = {
    "endpoint_advantage_mean_max": 0.05,
    "endpoint_positive_fraction_max": 0.60,
    "positive_task_count_per_split_max": 2,
    "positive_task_count_development_max": 4,
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def endpoints(root: Path) -> list[Path]:
    return sorted(root.glob("*step10000.pt"))


def path_at(endpoint: Path, step: int) -> Path:
    return endpoint.with_name(endpoint.name.replace("step10000", f"step{step:05d}"))


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(left, right) / max(float(np.linalg.norm(left) * np.linalg.norm(right)), 1e-12))


def build_rows(paths: list[Path], corpus: str, device: torch.device) -> list[dict[str, Any]]:
    rows = []
    for index, endpoint in enumerate(paths):
        payload = p1189.load_payload(endpoint)
        panel = p1189.panel_from_payload(payload)
        states: dict[int, dict[str, np.ndarray]] = {}
        for step in STEPS:
            model = p1189.load_model(p1189.load_payload(path_at(endpoint, step)), device)
            states[step] = {
                "calibration": p1189.response_unit_shape(model, panel, panel.train_mask, device),
                "evaluation": p1189.response_unit_shape(model, panel, panel.holdout_mask, device),
            }
            del model
        prefix = states[150]["calibration"] - states[25]["calibration"]
        middle = states[1000]["evaluation"] - states[150]["evaluation"]
        late = states[10000]["evaluation"] - states[1000]["evaluation"]
        endpoint_future = states[10000]["evaluation"] - states[150]["evaluation"]
        rows.append(
            {
                "corpus": corpus,
                "task_name": str(payload["task_name"]),
                "task_index": int(payload["task_index"]),
                "replicate": int(payload["replicate"]),
                "seed": int(payload["seed"]),
                "trajectory_id": p1190.trajectory_id(payload),
                "split": "development" if corpus == "development" else (
                    "discovery" if int(payload["task_index"]) < 4 else "confirmation"
                ),
                "prefix": prefix.tolist(),
                "middle": middle.tolist(),
                "late": late.tolist(),
                "endpoint": endpoint_future.tolist(),
                "prefix_norm": float(np.linalg.norm(prefix)),
                "middle_norm": float(np.linalg.norm(middle)),
                "late_norm": float(np.linalg.norm(late)),
                "endpoint_norm": float(np.linalg.norm(endpoint_future)),
                "middle_true_cosine": cosine(prefix, middle),
                "late_true_cosine": cosine(prefix, late),
                "endpoint_true_cosine": cosine(prefix, endpoint_future),
            }
        )
        print(canonical_json({"corpus": corpus, "completed": index + 1, "total": len(paths)}), flush=True)
        torch.cuda.empty_cache()
    lookup = {(row["task_name"], row["replicate"]): row for row in rows}
    for row in rows:
        null = lookup[(row["task_name"], (row["replicate"] + 1) % p1190.REPLICATES)]
        prefix = np.asarray(row["prefix"], dtype=np.float64)
        for horizon in ("middle", "late", "endpoint"):
            null_cosine = cosine(prefix, np.asarray(null[horizon], dtype=np.float64))
            row[horizon + "_null_cosine"] = null_cosine
            row[horizon + "_advantage"] = row[horizon + "_true_cosine"] - null_cosine
        row["null_trajectory_id"] = null["trajectory_id"]
    return rows


def task_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for task in sorted({row["task_name"] for row in rows}):
        selected = [row for row in rows if row["task_name"] == task]
        advantage = float(np.mean([row["endpoint_advantage"] for row in selected]))
        result.append({"task_name": task, "endpoint_advantage_mean": advantage, "positive": advantage > 0})
    return result


def summarize(rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    selected = rows if split == "development" else [row for row in rows if row["split"] == split]
    tasks = task_summaries(selected)
    expected_systems = 64 if split == "development" else 32
    expected_tasks = 8 if split == "development" else 4
    result: dict[str, Any] = {
        "split": split,
        "system_count": len(selected),
        "task_count": len(tasks),
        "task_summaries": tasks,
        "positive_task_count": sum(task["positive"] for task in tasks),
    }
    for horizon in ("middle", "late", "endpoint"):
        result[horizon + "_true_cosine_mean"] = float(
            np.mean([row[horizon + "_true_cosine"] for row in selected])
        )
        result[horizon + "_null_cosine_mean"] = float(
            np.mean([row[horizon + "_null_cosine"] for row in selected])
        )
        result[horizon + "_advantage_mean"] = float(
            np.mean([row[horizon + "_advantage"] for row in selected])
        )
        result[horizon + "_positive_fraction"] = float(
            np.mean([row[horizon + "_advantage"] > 0 for row in selected])
        )
    positive_tasks_min = (
        POSITIVE_THRESHOLDS["positive_task_count_development_min"]
        if split == "development"
        else POSITIVE_THRESHOLDS["positive_task_count_per_split_min"]
    )
    negative_tasks_max = (
        NEGATIVE_THRESHOLDS["positive_task_count_development_max"]
        if split == "development"
        else NEGATIVE_THRESHOLDS["positive_task_count_per_split_max"]
    )
    result["positive_gate_pass"] = bool(
        len(selected) == expected_systems
        and len(tasks) == expected_tasks
        and result["endpoint_advantage_mean"] >= POSITIVE_THRESHOLDS["endpoint_advantage_mean_min"]
        and result["endpoint_positive_fraction"] >= POSITIVE_THRESHOLDS["endpoint_positive_fraction_min"]
        and result["positive_task_count"] >= positive_tasks_min
    )
    result["negative_boundary_pass"] = bool(
        len(selected) == expected_systems
        and len(tasks) == expected_tasks
        and result["endpoint_advantage_mean"] <= NEGATIVE_THRESHOLDS["endpoint_advantage_mean_max"]
        and result["endpoint_positive_fraction"] <= NEGATIVE_THRESHOLDS["endpoint_positive_fraction_max"]
        and result["positive_task_count"] <= negative_tasks_max
    )
    return result


def source_hashes() -> dict[str, str]:
    paths = [SCRIPT, AUDIT_SCRIPT, Path(p1171.__file__), Path(p1187.__file__), Path(p1189.__file__), Path(p1190.__file__)]
    return {str(path.relative_to(ROOT)): file_sha256(path) for path in paths}


def develop() -> None:
    paths = endpoints(DEVELOPMENT_SOURCE)
    if len(paths) != 64 or not torch.cuda.is_available():
        raise RuntimeError("development materials or CUDA unavailable")
    rows = build_rows(paths, "development", torch.device("cuda"))
    write_jsonl(DEVELOPMENT_ROWS, rows)
    summary = summarize(rows, "development")
    summary.update(
        {
            "phase": PHASE,
            "created_at_utc": utc_now(),
            "rows_sha256": file_sha256(DEVELOPMENT_ROWS),
            "positive_thresholds": POSITIVE_THRESHOLDS,
            "negative_thresholds": NEGATIVE_THRESHOLDS,
            "formal_data_read": False,
        }
    )
    summary["summary_digest"] = digest({key: value for key, value in summary.items() if key != "summary_digest"})
    write_json(DEVELOPMENT_SUMMARY, summary)
    if not summary["negative_boundary_pass"] or summary["positive_gate_pass"]:
        raise RuntimeError("development did not establish the frozen negative-boundary candidate")


def formal_manifest(paths: list[Path]) -> dict[str, str]:
    result = {}
    for endpoint in paths:
        for step in STEPS:
            path = path_at(endpoint, step)
            result[path.name] = file_sha256(path)
    return result


def preregister() -> None:
    development = read_json(DEVELOPMENT_SUMMARY)
    upstream = read_json(p1190.FINAL_PATH)
    if not development["negative_boundary_pass"] or not upstream["main_gate_pass"]:
        raise RuntimeError("upstream authorization failed")
    if RAW_ROWS.exists():
        raise RuntimeError("formal outcomes already exist")
    paths = endpoints(FORMAL_SOURCE)
    if len(paths) != 64:
        raise RuntimeError("formal trajectory count mismatch")
    manifest = formal_manifest(paths)
    protocol = {
        "phase": PHASE,
        "title": "Prefix-to-future formation identity one-shot test",
        "created_at_utc": utc_now(),
        "scientific_question": (
            "Does the early quotient transition from step 25 to 150 identify the same free network's later "
            "transition from step 150 to 10000 better than a same-task replicate control?"
        ),
        "upstream": {
            "phase1190_final_sha256": file_sha256(p1190.FINAL_PATH),
            "phase1190_final_digest": upstream["final_digest"],
            "development_rows_sha256": file_sha256(DEVELOPMENT_ROWS),
            "development_summary_sha256": file_sha256(DEVELOPMENT_SUMMARY),
            "development_summary_digest": development["summary_digest"],
        },
        "steps": STEPS,
        "formal_manifest": manifest,
        "formal_manifest_digest": digest(manifest),
        "positive_thresholds": POSITIVE_THRESHOLDS,
        "negative_thresholds": NEGATIVE_THRESHOLDS,
        "source_hashes": source_hashes(),
        "evidence_contract_sha256": file_sha256(p1187.CONTRACT_PATH),
        "decision": {
            "positive": "both splits pass the positive gate",
            "negative_boundary": "both splits pass the frozen negative boundary",
            "ambiguous": "neither joint condition passes",
        },
        "hard_stops": [
            "No alternative prefix, horizon, weighting, normalization, regression, or null is searched.",
            "Middle and late sub-horizons are descriptive and cannot rescue the endpoint gate.",
            "A negative boundary closes static prefix-to-future identity in this registry.",
            "No Transformer or frozen-language-model transfer is authorized by any outcome.",
        ],
    }
    protocol["protocol_digest"] = digest({key: value for key, value in protocol.items() if key != "protocol_digest"})
    write_json(PROTOCOL_PATH, protocol)


def verify_protocol() -> dict[str, Any]:
    protocol = read_json(PROTOCOL_PATH)
    expected = digest({key: value for key, value in protocol.items() if key != "protocol_digest"})
    if expected != protocol["protocol_digest"]:
        raise RuntimeError("protocol digest mismatch")
    if protocol["source_hashes"] != source_hashes():
        raise RuntimeError("source code changed after preregistration")
    if file_sha256(DEVELOPMENT_ROWS) != protocol["upstream"]["development_rows_sha256"]:
        raise RuntimeError("development rows changed")
    if file_sha256(DEVELOPMENT_SUMMARY) != protocol["upstream"]["development_summary_sha256"]:
        raise RuntimeError("development summary changed")
    if file_sha256(p1190.FINAL_PATH) != protocol["upstream"]["phase1190_final_sha256"]:
        raise RuntimeError("Phase1190 final changed")
    if formal_manifest(endpoints(FORMAL_SOURCE)) != protocol["formal_manifest"]:
        raise RuntimeError("formal checkpoint manifest changed")
    return protocol


def bounded(value: float, threshold: float, comparator: str) -> dict[str, Any]:
    return {
        "claim_type": "bounded_float",
        "gating": True,
        "value": float(value),
        "threshold": float(threshold),
        "comparator": comparator,
        "dtype": "float64",
    }


def compile_claims(summary: dict[str, Any]) -> dict[str, Any]:
    contract = read_json(p1187.CONTRACT_PATH)
    families: dict[str, dict[str, dict[str, Any]]] = {"positive": {}, "negative": {}}
    for split in ("discovery", "confirmation"):
        current = summary[split]
        families["positive"][split + ".advantage"] = bounded(
            current["endpoint_advantage_mean"], POSITIVE_THRESHOLDS["endpoint_advantage_mean_min"], ">="
        )
        families["positive"][split + ".fraction"] = bounded(
            current["endpoint_positive_fraction"], POSITIVE_THRESHOLDS["endpoint_positive_fraction_min"], ">="
        )
        families["positive"][split + ".tasks"] = bounded(
            current["positive_task_count"], POSITIVE_THRESHOLDS["positive_task_count_per_split_min"], ">="
        )
        families["negative"][split + ".advantage"] = bounded(
            current["endpoint_advantage_mean"], NEGATIVE_THRESHOLDS["endpoint_advantage_mean_max"], "<="
        )
        families["negative"][split + ".fraction"] = bounded(
            current["endpoint_positive_fraction"], NEGATIVE_THRESHOLDS["endpoint_positive_fraction_max"], "<="
        )
        families["negative"][split + ".tasks"] = bounded(
            current["positive_task_count"], NEGATIVE_THRESHOLDS["positive_task_count_per_split_max"], "<="
        )
    result: dict[str, Any] = {}
    for family, raw in families.items():
        compiled = {name: p1187.compile_claim(claim, contract) for name, claim in raw.items()}
        conjunction = p1187.compile_claim(
            {
                "claim_type": "conjunction",
                "gating": True,
                "values": [bool(claim["authorizes"]) for claim in compiled.values()],
            },
            contract,
        )
        result[family] = {
            "raw": raw,
            "compiled": compiled,
            "conjunction": conjunction,
            "gate_pass": bool(conjunction["authorizes"]),
        }
    return result


def analyze() -> None:
    protocol = verify_protocol()
    paths = endpoints(FORMAL_SOURCE)
    rows = build_rows(paths, "formal", torch.device("cuda"))
    write_jsonl(RAW_ROWS, rows)
    summary = {
        "phase": PHASE,
        "created_at_utc": utc_now(),
        "protocol_digest": protocol["protocol_digest"],
        "raw_rows_sha256": file_sha256(RAW_ROWS),
        "discovery": summarize(rows, "discovery"),
        "confirmation": summarize(rows, "confirmation"),
    }
    summary["positive_gate_pass"] = bool(
        summary["discovery"]["positive_gate_pass"] and summary["confirmation"]["positive_gate_pass"]
    )
    summary["negative_boundary_pass"] = bool(
        summary["discovery"]["negative_boundary_pass"]
        and summary["confirmation"]["negative_boundary_pass"]
    )
    summary["decision"] = (
        "positive" if summary["positive_gate_pass"] else (
            "negative_boundary" if summary["negative_boundary_pass"] else "ambiguous"
        )
    )
    summary["summary_digest"] = digest({key: value for key, value in summary.items() if key != "summary_digest"})
    write_json(SUMMARY_PATH, summary)
    write_json(CLAIMS_PATH, compile_claims(summary))


def finalize() -> None:
    protocol = verify_protocol()
    summary = read_json(SUMMARY_PATH)
    claims = read_json(CLAIMS_PATH)
    audit_pass = bool(AUDIT_PATH.exists() and read_json(AUDIT_PATH).get("gate_pass"))
    typed_match = bool(claims[summary["decision"] if summary["decision"] != "ambiguous" else "positive"]["gate_pass"])
    completed = bool(summary["decision"] != "ambiguous" and typed_match and audit_pass)
    final = {
        "phase": PHASE,
        "created_at_utc": utc_now(),
        "status": (
            "prefix_future_identity_confirmed" if completed and summary["decision"] == "positive" else (
                "local_event_only_negative_boundary" if completed and summary["decision"] == "negative_boundary" else (
                    "awaiting_independent_audit" if summary["decision"] != "ambiguous" and typed_match and not AUDIT_PATH.exists()
                    else "ambiguous_or_failed"
                )
            )
        ),
        "protocol_digest": protocol["protocol_digest"],
        "summary_digest": summary["summary_digest"],
        "claims_sha256": file_sha256(CLAIMS_PATH),
        "audit_digest": read_json(AUDIT_PATH).get("audit_digest") if AUDIT_PATH.exists() else None,
        "decision": summary["decision"],
        "independent_audit_pass": audit_pass,
        "main_gate_complete": completed,
        "evidence_grade": "E3_KT_free_network" if completed else "no_upgrade",
        "authorized_next": {
            "static_prefix_identity_search": False,
            "formation_causal_branch_preregistration": completed and summary["decision"] == "positive",
            "transformer_or_language_model_transfer": False,
            "theory_closure": False,
        },
        "claim_scope": (
            "The result separates a locally repeatable SGD response event from a persistent trajectory identity. "
            "A negative boundary does not deny local formation events; it denies this frozen early-to-future "
            "identity rule on the tested RoleSquare task family."
        ),
        "final_digest": None,
    }
    final["final_digest"] = digest({key: value for key, value in final.items() if key != "final_digest"})
    write_json(FINAL_PATH, final)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("develop", "preregister", "analyze", "finalize"))
    args = parser.parse_args()
    {"develop": develop, "preregister": preregister, "analyze": analyze, "finalize": finalize}[args.command]()


if __name__ == "__main__":
    main()
