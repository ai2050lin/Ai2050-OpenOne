"""Independent external-model review of the frozen Phase1132 material package.

This produces machine-review evidence only. Outputs are deliberately marked
ineligible for the Phase1132 two-human-review ingest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "ai_rnd_config.json"
SOURCE_PATH = (
    ROOT
    / "tests/glm5/result/phase1132_postrelease_temporal_material"
    / "revision6_temporal_relation_binding_overprovisioned"
    / "material/candidate_package_unreviewed.jsonl"
)
RESULT_ROOT = ROOT / "tests/glm5/result/phase1134_external_api_temporal_annotation"
RAW_ROOT = RESULT_ROOT / "raw"
REVIEW_ROOT = RESULT_ROOT / "reviews"
ANALYSIS_ROOT = RESULT_ROOT / "analysis"
AUDIT_ROOT = RESULT_ROOT / "audit"

PROMPT_VERSION = "phase1134_temporal_material_review.v4"
REQUIRED_JUDGMENTS = (
    "gold_answer_correct",
    "candidate_unique",
    "matched_null_globally_false",
    "matched_null_locally_plausible",
    "natural_language_acceptable",
)
SELECTED_FIELDS = (
    "item_id",
    "context",
    "query",
    "gold_answer",
    "matched_null_candidate",
    "relation_label",
    "subject_label",
    "old_end",
    "new_start",
    "post_query_date",
    "domain",
    "property_id",
    "split",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(value)
    return rows


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical(value))
    temporary.replace(path)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    temporary.replace(path)


def load_reviewers() -> dict[str, dict[str, str]]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    candidates = [config["master_model"], *config.get("analyst_models", [])]
    reviewers: dict[str, dict[str, str]] = {}
    for item in candidates:
        api_type = str(item.get("api_type", "")).lower()
        model_id = str(item.get("model_id", ""))
        if api_type == "deepseek" and "deepseek" not in reviewers:
            reviewers["deepseek"] = {
                "reviewer_id": "external_model_deepseek_chat_v1",
                "provider": api_type,
                "api_base": str(item["api_base"]),
                "api_key": str(item["api_key"]),
                "model_id": model_id,
            }
        if api_type == "nownextai" and "claude" not in reviewers:
            reviewers["claude"] = {
                "reviewer_id": "external_model_claude_opus_4_8_v1",
                "provider": api_type,
                "api_base": str(item["api_base"]),
                "api_key": str(item["api_key"]),
                "model_id": model_id,
            }
    return reviewers


def endpoint(api_base: str) -> str:
    base = api_base.rstrip("/")
    return base if base.endswith("/chat/completions") else f"{base}/chat/completions"


def selected_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{field: row.get(field) for field in SELECTED_FIELDS} for row in rows]


def build_prompt(rows: list[dict[str, Any]]) -> str:
    payload = json.dumps(selected_rows(rows), ensure_ascii=False, separators=(",", ":"))
    return f"""Review this batch of temporal relation-binding dataset records.

Judge only the supplied record, query, dates, candidates, and wording. You are
blind to all outputs from the local target models. Do not infer that a field is
true merely because the dataset generator marked it so. If the supplied text is
ambiguous, contradictory, has a date gap/overlap, or does not establish a
judgment, return false and explain briefly.

Use these frozen temporal semantics for this dataset:
- If the old holder ended on date T and the new holder began on T, the new holder
  is unambiguously active on every supplied query date later than T. Any boundary
  ambiguity at T itself is irrelevant to a later query.
- If there is a gap before the new start date, that gap is irrelevant when the
  supplied query date is on or after the new start date.
- Treat the supplied two-holder record as complete for this two-candidate task;
  do not invent an unmentioned later holder.
- candidate_unique means unique between the supplied active and matched-null
  candidates at the exact query date, not unique across every person in reality.
- Still reject genuine defects: the two candidate strings denote the same
  real-world entity or alias, the text contradicts itself, dates fail to support
  the query, the relation is mismatched, or the wording is not usable.

For every item return these five booleans:
- gold_answer_correct: the dated context unambiguously supports gold_answer for query.
- candidate_unique: at the exact post-query date, the context yields one unambiguous answer among the candidates. A gap or boundary ambiguity at another, unqueried date does not make this false.
- matched_null_globally_false: at the exact post-query date, matched_null_candidate is not correct under the supplied dated record. "Globally" means considering the whole supplied record at that target date, not requiring every date in the timeline to be covered.
- matched_null_locally_plausible: matched_null_candidate is a meaningful hard negative, normally the prior holder for the same relation, rather than nonsense.
- natural_language_acceptable: context and query are grammatical, clear, internally consistent, and natural enough for evaluation.

Return exactly one JSON array and nothing else. Preserve every item_id. Each
element must have item_id, the five booleans, confidence as an integer 0-100,
and notes as a concise string of at most 24 words. Do not use markdown fences.

INPUT:
{payload}
"""


def extract_content(data: dict[str, Any]) -> str:
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message", {})
        content = message.get("content", "")
    elif isinstance(data.get("content"), list):
        # Some OpenAI-routed Anthropic proxies return the native Messages shape.
        content = data["content"]
    else:
        raise ValueError(f"response has no supported content field: {sorted(data)}")
    if isinstance(content, list):
        return "".join(
            str(part.get("text", "")) if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content or "")


def parse_response(text: str, expected_ids: list[str]) -> list[dict[str, Any]]:
    cleaned = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip())
    start, end = cleaned.find("["), cleaned.rfind("]")
    if start < 0 or end < start:
        raise ValueError("response does not contain a JSON array")
    value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, list):
        raise ValueError("response root is not an array")
    by_id: dict[str, dict[str, Any]] = {}
    for row in value:
        if not isinstance(row, dict):
            raise ValueError("review entry is not an object")
        item_id = str(row.get("item_id", ""))
        if not item_id or item_id in by_id:
            raise ValueError(f"missing or duplicate item_id: {item_id!r}")
        for field in REQUIRED_JUDGMENTS:
            if not isinstance(row.get(field), bool):
                raise ValueError(f"{item_id}: {field} is not a JSON boolean")
        confidence = row.get("confidence")
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
            raise ValueError(f"{item_id}: confidence is not numeric")
        if not 0 <= float(confidence) <= 100:
            raise ValueError(f"{item_id}: confidence outside 0-100")
        row["confidence"] = int(round(float(confidence)))
        row["notes"] = str(row.get("notes", "")).strip()
        by_id[item_id] = row
    if set(by_id) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(by_id))
        extra = sorted(set(by_id) - set(expected_ids))
        raise ValueError(f"response ID mismatch: missing={missing}, extra={extra}")
    return [by_id[item_id] for item_id in expected_ids]


def request_batch(
    client: httpx.Client,
    reviewer: dict[str, str],
    rows: list[dict[str, Any]],
    retries: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prompt = build_prompt(rows)
    expected_ids = [str(row["item_id"]) for row in rows]
    provider = reviewer["provider"]
    if provider == "nownextai":
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ]
    else:
        messages = [{"role": "user", "content": prompt}]
    request_body: dict[str, Any] = {
        "model": reviewer["model_id"],
        "messages": messages,
        "max_tokens": 4096,
        "stream": False,
    }
    if provider != "nownextai":
        request_body["temperature"] = 0
    headers = {
        "Authorization": f"Bearer {reviewer['api_key']}",
        "Content-Type": "application/json",
    }
    failures: list[dict[str, Any]] = []
    for attempt in range(1, retries + 2):
        started = time.perf_counter()
        try:
            response = client.post(
                endpoint(reviewer["api_base"]), headers=headers, json=request_body
            )
            elapsed = time.perf_counter() - started
            if response.status_code >= 400:
                raise RuntimeError(
                    f"HTTP {response.status_code}: {response.text[:300]}"
                )
            data = response.json()
            content = extract_content(data)
            parsed = parse_response(content, expected_ids)
            return parsed, {
                "attempt": attempt,
                "elapsed_seconds": round(elapsed, 3),
                "response_text": content,
                "response_sha256": digest_bytes(content.encode("utf-8")),
                "usage": data.get("usage", {}),
                "failures_before_success": failures,
            }
        except Exception as error:
            failures.append(
                {
                    "attempt": attempt,
                    "error_type": type(error).__name__,
                    "message": str(error)[:500],
                }
            )
            if attempt > retries:
                raise RuntimeError(
                    f"batch failed after {attempt} attempts: {failures[-1]}"
                ) from error
            time.sleep(min(2 ** (attempt - 1), 12))
    raise AssertionError("unreachable")


def validate_cached_batch(
    payload: dict[str, Any], expected_ids: list[str], source_sha256: str
) -> list[dict[str, Any]]:
    if payload.get("prompt_version") != PROMPT_VERSION:
        raise ValueError("cached prompt version mismatch")
    if payload.get("source_sha256") != source_sha256:
        raise ValueError("cached source hash mismatch")
    if payload.get("item_ids") != expected_ids:
        raise ValueError("cached item order mismatch")
    parsed = payload.get("parsed_reviews")
    if not isinstance(parsed, list):
        raise ValueError("cached parsed_reviews missing")
    by_id = {str(row.get("item_id", "")): row for row in parsed}
    if list(by_id) != expected_ids:
        raise ValueError("cached parsed review IDs mismatch")
    for row in parsed:
        for field in REQUIRED_JUDGMENTS:
            if not isinstance(row.get(field), bool):
                raise ValueError(f"cached {field} is not boolean")
    return parsed


def annotate_reviewer(
    reviewer_name: str,
    reviewer: dict[str, str],
    rows: list[dict[str, Any]],
    source_sha256: str,
    batch_size: int,
    retries: int,
    max_batches: int | None,
) -> list[dict[str, Any]]:
    raw_dir = RAW_ROOT / PROMPT_VERSION / f"batch_{batch_size:02d}" / reviewer_name
    raw_dir.mkdir(parents=True, exist_ok=True)
    reviews: list[dict[str, Any]] = []
    total_batches = (len(rows) + batch_size - 1) // batch_size
    selected_batches = total_batches if max_batches is None else min(total_batches, max_batches)
    timeout = httpx.Timeout(180.0, connect=30.0)
    with httpx.Client(timeout=timeout, trust_env=False) as client:
        for batch_index in range(selected_batches):
            start = batch_index * batch_size
            batch = rows[start : start + batch_size]
            item_ids = [str(row["item_id"]) for row in batch]
            batch_path = raw_dir / f"batch_{batch_index:04d}.json"
            if batch_path.exists():
                cached = json.loads(batch_path.read_text(encoding="utf-8"))
                parsed = validate_cached_batch(cached, item_ids, source_sha256)
                state = "cached"
            else:
                parsed, response_meta = request_batch(client, reviewer, batch, retries)
                request_fingerprint = {
                    "provider": reviewer["provider"],
                    "api_base": reviewer["api_base"],
                    "model_id": reviewer["model_id"],
                    "prompt_version": PROMPT_VERSION,
                    "item_ids": item_ids,
                }
                atomic_json(
                    batch_path,
                    {
                        "schema_version": "phase1134_external_batch.v1",
                        "created_at_utc": now(),
                        "reviewer_name": reviewer_name,
                        "reviewer_id": reviewer["reviewer_id"],
                        "provider": reviewer["provider"],
                        "model_id": reviewer["model_id"],
                        "prompt_version": PROMPT_VERSION,
                        "source_sha256": source_sha256,
                        "batch_index": batch_index,
                        "item_ids": item_ids,
                        "request_fingerprint_sha256": digest_bytes(
                            canonical(request_fingerprint)
                        ),
                        "parsed_reviews": parsed,
                        "response": response_meta,
                        "api_key_recorded": False,
                    },
                )
                state = "api"
            for row in parsed:
                reviews.append(
                    {
                        "item_id": row["item_id"],
                        "reviewer_id": reviewer["reviewer_id"],
                        "reviewer_type": "external_model",
                        "human_reviewer": False,
                        "eligible_for_phase1132_human_ingest": False,
                        "annotation_blinded_to_model_outputs": False,
                        "blinded_to_local_target_model_outputs": True,
                        "provider": reviewer["provider"],
                        "model_id": reviewer["model_id"],
                        "prompt_version": PROMPT_VERSION,
                        "batch_index": batch_index,
                        **{field: row[field] for field in REQUIRED_JUDGMENTS},
                        "confidence": row["confidence"],
                        "notes": row["notes"],
                    }
                )
            print(
                json.dumps(
                    {
                        "reviewer": reviewer_name,
                        "batch": batch_index + 1,
                        "total_batches": total_batches,
                        "completed_items": len(reviews),
                        "state": state,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    output = REVIEW_ROOT / f"{reviewer_name}_machine_review.jsonl"
    atomic_jsonl(output, reviews)
    return reviews


def summarize(
    source: list[dict[str, Any]], reviews: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    source_ids = [str(row["item_id"]) for row in source]
    source_by_id = {str(row["item_id"]): row for row in source}
    result: dict[str, Any] = {
        "schema_version": "phase1134_external_annotation_summary.v1",
        "created_at_utc": now(),
        "source_count": len(source),
        "reviewers": {},
        "human_review_gate_satisfied": False,
        "claim_scope": "external-model machine review only; not human annotation",
    }
    indexed: dict[str, dict[str, dict[str, Any]]] = {}
    for name, rows in reviews.items():
        by_id = {str(row["item_id"]): row for row in rows}
        indexed[name] = by_id
        acceptance = sum(
            all(row[field] for field in REQUIRED_JUDGMENTS) for row in rows
        )
        field_true = {
            field: sum(bool(row[field]) for row in rows)
            for field in REQUIRED_JUDGMENTS
        }
        split_counts: dict[str, dict[str, int]] = defaultdict(
            lambda: {"reviewed": 0, "accepted": 0}
        )
        for row in rows:
            split = str(source_by_id[str(row["item_id"])]["split"])
            split_counts[split]["reviewed"] += 1
            split_counts[split]["accepted"] += int(
                all(row[field] for field in REQUIRED_JUDGMENTS)
            )
        result["reviewers"][name] = {
            "reviewed_count": len(rows),
            "coverage_complete": set(by_id) == set(source_ids),
            "accepted_count": acceptance,
            "rejected_count": len(rows) - acceptance,
            "field_true_counts": field_true,
            "split_counts": dict(split_counts),
        }
    if len(indexed) >= 2:
        first_name, second_name = list(indexed)[:2]
        common_ids = [
            item_id
            for item_id in source_ids
            if item_id in indexed[first_name] and item_id in indexed[second_name]
        ]
        field_agreement = {
            field: sum(
                indexed[first_name][item_id][field]
                == indexed[second_name][item_id][field]
                for item_id in common_ids
            )
            for field in REQUIRED_JUDGMENTS
        }
        exact_agreement_ids = [
            item_id
            for item_id in common_ids
            if all(
                indexed[first_name][item_id][field]
                == indexed[second_name][item_id][field]
                for field in REQUIRED_JUDGMENTS
            )
        ]
        jointly_accepted = [
            item_id
            for item_id in common_ids
            if all(
                indexed[name][item_id][field]
                for name in (first_name, second_name)
                for field in REQUIRED_JUDGMENTS
            )
        ]
        disagreements: list[dict[str, Any]] = []
        for item_id in common_ids:
            differing = [
                field
                for field in REQUIRED_JUDGMENTS
                if indexed[first_name][item_id][field]
                != indexed[second_name][item_id][field]
            ]
            if differing:
                disagreements.append(
                    {
                        "item_id": item_id,
                        "differing_fields": differing,
                        "source": {
                            field: source_by_id[item_id].get(field)
                            for field in SELECTED_FIELDS
                        },
                        first_name: indexed[first_name][item_id],
                        second_name: indexed[second_name][item_id],
                    }
                )
        result["comparison"] = {
            "reviewer_pair": [first_name, second_name],
            "common_count": len(common_ids),
            "field_agreement_counts": field_agreement,
            "exact_five_field_agreement_count": len(exact_agreement_ids),
            "jointly_accepted_count": len(jointly_accepted),
            "disagreement_count": len(disagreements),
        }
        atomic_jsonl(ANALYSIS_ROOT / "machine_disagreements.jsonl", disagreements)
        atomic_jsonl(
            ANALYSIS_ROOT / "jointly_machine_accepted_item_ids.jsonl",
            [{"item_id": item_id} for item_id in jointly_accepted],
        )
        consensus_rows: list[dict[str, Any]] = []
        for item_id in jointly_accepted:
            row = dict(source_by_id[item_id])
            row["external_machine_review"] = {
                "status": "accepted_by_all_external_reviewers",
                "reviewer_ids": [
                    indexed[first_name][item_id]["reviewer_id"],
                    indexed[second_name][item_id]["reviewer_id"],
                ],
                "prompt_version": PROMPT_VERSION,
                "required_judgments_all_true": True,
                "human_reviewer": False,
                "eligible_for_phase1132_human_ingest": False,
            }
            row["machine_validation_only"] = True
            consensus_rows.append(row)
        atomic_jsonl(
            ANALYSIS_ROOT / "external_machine_consensus_package.jsonl",
            consensus_rows,
        )
        result["comparison"]["consensus_package_count"] = len(consensus_rows)
        result["comparison"]["consensus_package_human_eligible"] = False
    return result


def execution_metrics(
    reviewer_names: list[str], batch_size: int
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "schema_version": "phase1134_external_execution_metrics.v1",
        "created_at_utc": now(),
        "prompt_version": PROMPT_VERSION,
        "batch_size": batch_size,
        "reviewers": {},
    }
    for name in reviewer_names:
        raw_dir = RAW_ROOT / PROMPT_VERSION / f"batch_{batch_size:02d}" / name
        numeric_usage: Counter[str] = Counter()
        failure_types: Counter[str] = Counter()
        elapsed_seconds = 0.0
        files = sorted(raw_dir.glob("batch_*.json"))
        for path in files:
            payload = json.loads(path.read_text(encoding="utf-8"))
            response = payload["response"]
            elapsed_seconds += float(response.get("elapsed_seconds", 0.0))
            for key, value in response.get("usage", {}).items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    numeric_usage[key] += value
            for failure in response.get("failures_before_success", []):
                failure_types[str(failure.get("error_type", "unknown"))] += 1
        metrics["reviewers"][name] = {
            "batch_count": len(files),
            "request_elapsed_seconds": round(elapsed_seconds, 3),
            "numeric_usage": dict(numeric_usage),
            "failures_before_success": dict(failure_types),
        }
    return metrics


def audit_outputs(
    source: list[dict[str, Any]],
    reviews: dict[str, list[dict[str, Any]]],
    summary: dict[str, Any],
) -> dict[str, Any]:
    source_ids = [str(row["item_id"]) for row in source]
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    check("source_has_493_rows", len(source) == 493, len(source))
    check("source_ids_unique", len(source_ids) == len(set(source_ids)), len(set(source_ids)))
    for name, rows in reviews.items():
        ids = [str(row["item_id"]) for row in rows]
        check(f"{name}_coverage_complete", set(ids) == set(source_ids), len(ids))
        check(f"{name}_ids_unique", len(ids) == len(set(ids)), len(set(ids)))
        check(
            f"{name}_booleans_valid",
            all(
                isinstance(row.get(field), bool)
                for row in rows
                for field in REQUIRED_JUDGMENTS
            ),
            len(rows),
        )
        check(
            f"{name}_marked_nonhuman",
            all(
                row.get("human_reviewer") is False
                and row.get("eligible_for_phase1132_human_ingest") is False
                and row.get("annotation_blinded_to_model_outputs") is False
                for row in rows
            ),
            len(rows),
        )
    check(
        "human_gate_remains_false",
        summary.get("human_review_gate_satisfied") is False,
        summary.get("claim_scope"),
    )
    return {
        "schema_version": "phase1134_external_annotation_audit.v1",
        "created_at_utc": now(),
        "passed": all(item["passed"] for item in checks),
        "checks": checks,
        "reviewer_count": len(reviews),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reviewers", nargs="+", choices=("deepseek", "claude"), default=("deepseek", "claude")
    )
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-batches", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size < 1 or args.batch_size > 20:
        raise ValueError("batch-size must be between 1 and 20")
    source = read_jsonl(SOURCE_PATH)
    source_sha256 = digest_file(SOURCE_PATH)
    available = load_reviewers()
    missing = [name for name in args.reviewers if name not in available]
    if missing:
        raise RuntimeError(f"configured external reviewers missing: {missing}")

    protocol = {
        "schema_version": "phase1134_external_annotation_protocol.v1",
        "created_at_utc": now(),
        "prompt_version": PROMPT_VERSION,
        "source_path": str(SOURCE_PATH.relative_to(ROOT)),
        "source_sha256": source_sha256,
        "source_count": len(source),
        "batch_size": args.batch_size,
        "reviewers": {
            name: {
                "reviewer_id": available[name]["reviewer_id"],
                "provider": available[name]["provider"],
                "api_base": available[name]["api_base"],
                "model_id": available[name]["model_id"],
                "api_key_recorded": False,
            }
            for name in args.reviewers
        },
        "evidence_scope": "independent external-model review; not human review",
        "phase1132_human_ingest_eligible": False,
    }
    atomic_json(RESULT_ROOT / "protocol/protocol.json", protocol)

    reviews: dict[str, list[dict[str, Any]]] = {}
    for name in args.reviewers:
        reviews[name] = annotate_reviewer(
            name,
            available[name],
            source,
            source_sha256,
            args.batch_size,
            args.retries,
            args.max_batches,
        )

    summary = summarize(source, reviews)
    atomic_json(ANALYSIS_ROOT / "summary.json", summary)
    atomic_json(
        ANALYSIS_ROOT / "execution_metrics.json",
        execution_metrics(list(args.reviewers), args.batch_size),
    )
    audit = audit_outputs(source, reviews, summary)
    atomic_json(AUDIT_ROOT / "result_audit.json", audit)
    print(json.dumps({"summary": summary, "audit": audit}, ensure_ascii=False), flush=True)
    return 0 if audit["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
