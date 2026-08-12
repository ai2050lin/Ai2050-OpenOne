#!/usr/bin/env python3
"""Phase 1132: post-release temporal material production and hard gate.

This phase freezes a deterministic candidate pool from time-qualified Wikidata
role transitions that begin well after all three local checkpoints were public.
It loads tokenizers only, never model weights. The resulting package is passed
through the unchanged Phase 1131 material contract. Without two independent
blind human reviews, model scoring remains forbidden.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
import urllib.parse
import urllib.request
import urllib.error
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable


PHASE = 1132
REVISION = "revision1_conservative_2026_window"
PRIMITIVE_FAMILY_OVERRIDE: str | None = None
RAW_SNAPSHOT_PROVENANCE = "current_revision_fetch_or_cache"
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = REPO_ROOT / "tests/glm5/result/phase1132_postrelease_temporal_material"
RAW_ROOT = RESULT_ROOT / "raw"
MATERIAL_ROOT = RESULT_ROOT / "material"
PACKAGE_PATH = MATERIAL_ROOT / "candidate_package_unreviewed.jsonl"

PHASE1131_RESULT = (
    REPO_ROOT
    / "tests/glm5/result/phase1131_material_readiness_and_claim_scope"
    / "analysis/readiness_summary.json"
)

sys.path.insert(0, str(REPO_ROOT / "tests/glm5"))
from phase1131_material_readiness_and_claim_scope_audit import audit_package  # noqa: E402


UTC = timezone.utc
FACT_START = datetime(2026, 1, 1, tzinfo=UTC)
FACT_END = datetime(2026, 7, 31, 23, 59, 59, tzinfo=UTC)
FACT_WINDOW_RATIONALE = (
    "Starts more than five months after the latest public metadata change among "
    "the local checkpoints; effective dates alone still do not prove the fact "
    "was unannounced earlier."
)
OLD_SEARCH_START = datetime(2024, 1, 1, tzinfo=UTC)
QUERY_OFFSET_DAYS = 7
MAX_TRANSITION_GAP_DAYS = 31
TARGET_PER_SPLIT = 128
SPLITS = ("discovery", "confirmation", "natural_use")
TARGET_TOTAL = TARGET_PER_SPLIT * len(SPLITS)
ALLOCATE_ALL_QUALIFIED = False

WIKIDATA_SPARQL = "https://qlever.dev/api/wikidata"
USER_AGENT = "Ai2050-OpenOne/Phase1132 temporal-material-audit"
SPARQL_MIN_INTERVAL_SECONDS = 1

PROPERTIES = {
    "P6": {"label": "head of government", "domain": "government"},
    "P35": {"label": "head of state", "domain": "government"},
    "P169": {"label": "chief executive officer", "domain": "corporate"},
    "P286": {"label": "head coach", "domain": "sports"},
    "P488": {"label": "chairperson", "domain": "organization"},
}

MODEL_IDENTITIES = {
    "qwen3": {
        "model_id": "Qwen/Qwen3-4B",
        "path": REPO_ROOT / "models/hf/qwen3-4b",
        "public_release": "2025-04-29",
        "official_source": "https://qwenlm.github.io/blog/qwen3/",
    },
    "glm4": {
        "model_id": "zai-org/glm-4-9b-chat-hf",
        "path": REPO_ROOT / "models/hf/glm4-9b-chat-hf",
        "public_release": "2024-10-23",
        "official_source": "https://huggingface.co/zai-org/glm-4-9b-chat-hf",
    },
    "ds7b": {
        "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "path": REPO_ROOT / "models/hf/deepseek-r1-distill-qwen-7b",
        "public_release": "2025-01-20",
        "official_source": "https://github.com/deepseek-ai/DeepSeek-R1",
    },
}

SURFACE_POLICY = {
    "max_word_count_difference": 1,
    "min_character_length_ratio": 0.75,
    "max_character_length_difference": 6,
    "max_token_count_difference_per_tokenizer": 1,
    "max_candidate_tokens_per_tokenizer": 8,
    "tokenization_prefix": "single_ascii_space",
}

POST_RELEASE_SOURCE_INVENTORY = [
    {
        "name": "LiveSearchBench-2025",
        "url": "https://livesearchbench.github.io/",
        "time_basis": "May 2025 versus August 2025 Wikidata snapshots",
        "fit": "excellent_unique_answer_dynamic_knowledge_design",
        "availability": "project_page_says_data_coming_soon",
        "decision": "method_reference_not_ingested",
    },
    {
        "name": "PTC Benchmark",
        "url": "https://huggingface.co/datasets/EliasHossain/ptc-benchmark",
        "time_basis": "updates through June 2024",
        "fit": "excellent_old_new_role_transition_schema",
        "availability": "public",
        "decision": "calibration_reference_only_not_postrelease_material",
    },
    {
        "name": "BTF-2",
        "url": "https://huggingface.co/datasets/BTF-2/BTF-2",
        "time_basis": "asked October 2025 and resolved December 2025",
        "fit": "postrelease_objective_forecasting_not_role_binding",
        "availability": "public_1417_items",
        "decision": "separate_forecasting_axis_not_mixed_into_this_primitive",
    },
]


def canonical_digest(payload: dict[str, Any], digest_key: str) -> str:
    body = dict(payload)
    body.pop(digest_key, None)
    encoded = json.dumps(
        body, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            )
            handle.write("\n")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def request_json(url: str, params: dict[str, str], retries: int = 5) -> Any:
    encoded = urllib.parse.urlencode(params)
    request = urllib.request.Request(
        f"{url}?{encoded}",
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            last_error = error
            if attempt + 1 < retries:
                retry_after = error.headers.get("Retry-After")
                wait_seconds = int(retry_after) if retry_after and retry_after.isdigit() else 65
                time.sleep(max(wait_seconds, 2**attempt))
        except Exception as error:  # Network errors differ across Python builds.
            last_error = error
            if attempt + 1 < retries:
                time.sleep(2**attempt)
    raise RuntimeError(f"request failed after {retries} attempts: {last_error}")


def iso_date(value: datetime) -> str:
    return value.date().isoformat()


def parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    if value.startswith("http://") or value.startswith("https://"):
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def uri_tail(value: str) -> str:
    return value.rsplit("/", 1)[-1]


def binding_value(binding: dict[str, Any], key: str) -> str | None:
    node = binding.get(key)
    if not node:
        return None
    return str(node.get("value"))


def transition_query(property_id: str) -> str:
    object_type_qid = PROPERTIES[property_id].get("object_type_qid")
    object_constraint = ""
    if object_type_qid:
        object_constraint = (
            f"\n  ?newHolder wdt:P31/wdt:P279* wd:{object_type_qid} ."
            f"\n  ?oldHolder wdt:P31/wdt:P279* wd:{object_type_qid} ."
        )
    return f"""
PREFIX p: <http://www.wikidata.org/prop/>
PREFIX ps: <http://www.wikidata.org/prop/statement/>
PREFIX pq: <http://www.wikidata.org/prop/qualifier/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
SELECT DISTINCT ?item ?itemLabel ?newStatement ?newHolder ?newHolderLabel ?newStart ?newEnd
       ?oldStatement ?oldHolder ?oldHolderLabel ?oldStart ?oldEnd WHERE {{
  ?item p:{property_id} ?newStatement .
  ?newStatement ps:{property_id} ?newHolder ; pq:P580 ?newStart .
  OPTIONAL {{ ?newStatement pq:P582 ?newEnd . }}
  FILTER(
    ?newStart >= \"{FACT_START.isoformat().replace('+00:00', 'Z')}\"^^xsd:dateTime &&
    ?newStart <= \"{FACT_END.isoformat().replace('+00:00', 'Z')}\"^^xsd:dateTime
  )
  ?item p:{property_id} ?oldStatement .
  ?oldStatement ps:{property_id} ?oldHolder ; pq:P582 ?oldEnd .
  OPTIONAL {{ ?oldStatement pq:P580 ?oldStart . }}
  FILTER(
    ?oldEnd <= ?newStart &&
    ?oldEnd >= \"{OLD_SEARCH_START.isoformat().replace('+00:00', 'Z')}\"^^xsd:dateTime
  )
  FILTER(?oldHolder != ?newHolder)
  {object_constraint}
  ?item rdfs:label ?itemLabel .
  ?newHolder rdfs:label ?newHolderLabel .
  ?oldHolder rdfs:label ?oldHolderLabel .
  FILTER(LANG(?itemLabel) = "en")
  FILTER(LANG(?newHolderLabel) = "en")
  FILTER(LANG(?oldHolderLabel) = "en")
}}
ORDER BY ?newStart ?item ?oldEnd
LIMIT 10000
""".strip()


def fetch_transition_bindings(refresh: bool) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    network_request_count = 0
    for property_id in PROPERTIES:
        raw_path = RAW_ROOT / f"wikidata_{property_id}.json"
        if raw_path.exists() and not refresh:
            payload = read_json(raw_path)
        else:
            if network_request_count:
                time.sleep(SPARQL_MIN_INTERVAL_SECONDS)
            payload = request_json(
                WIKIDATA_SPARQL,
                {"action": "json_export", "query": transition_query(property_id)},
            )
            network_request_count += 1
            write_json(raw_path, payload)
        result[property_id] = list(payload.get("results", {}).get("bindings", []))
    return result


def normalize_bindings(
    raw_by_property: dict[str, list[dict[str, Any]]]
) -> tuple[list[dict[str, Any]], Counter[str]]:
    rejections: Counter[str] = Counter()
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)

    for property_id, bindings in raw_by_property.items():
        for binding in bindings:
            item_uri = binding_value(binding, "item")
            item_label = binding_value(binding, "itemLabel")
            new_statement_uri = binding_value(binding, "newStatement")
            old_statement_uri = binding_value(binding, "oldStatement")
            new_holder_uri = binding_value(binding, "newHolder")
            new_holder_label = binding_value(binding, "newHolderLabel")
            old_holder_uri = binding_value(binding, "oldHolder")
            old_holder_label = binding_value(binding, "oldHolderLabel")
            new_start = parse_time(binding_value(binding, "newStart"))
            new_end = parse_time(binding_value(binding, "newEnd"))
            old_start = parse_time(binding_value(binding, "oldStart"))
            old_end = parse_time(binding_value(binding, "oldEnd"))
            required = [
                item_uri,
                new_statement_uri,
                old_statement_uri,
                new_holder_uri,
                new_holder_label,
                old_holder_uri,
                old_holder_label,
                item_label,
                new_start,
                old_end,
            ]
            if not all(required):
                rejections["missing_required_binding"] += 1
                continue
            assert item_uri and new_statement_uri and old_statement_uri
            assert new_holder_uri and old_holder_uri and new_start and old_end
            assert item_label and new_holder_label and old_holder_label
            gap = (new_start - old_end).days
            if gap < 0 or gap > MAX_TRANSITION_GAP_DAYS:
                rejections["transition_gap_outside_window"] += 1
                continue
            post_date = new_start + timedelta(days=QUERY_OFFSET_DAYS)
            pre_date = old_end - timedelta(days=QUERY_OFFSET_DAYS)
            if new_end is not None and new_end < post_date:
                rejections["new_holder_not_valid_at_post_query"] += 1
                continue
            if old_start is not None and old_start > pre_date:
                rejections["old_holder_not_valid_at_pre_query"] += 1
                continue
            record = {
                "property_id": property_id,
                "item_qid": uri_tail(item_uri),
                "item_label": item_label,
                "new_statement_id": uri_tail(new_statement_uri),
                "new_holder_qid": uri_tail(new_holder_uri),
                "new_holder_label": new_holder_label,
                "new_start": new_start,
                "new_end": new_end,
                "old_statement_id": uri_tail(old_statement_uri),
                "old_holder_qid": uri_tail(old_holder_uri),
                "old_holder_label": old_holder_label,
                "old_start": old_start,
                "old_end": old_end,
                "pre_query_date": pre_date,
                "post_query_date": post_date,
            }
            grouped[(record["item_qid"], property_id, record["new_statement_id"])].append(
                record
            )

    selected: list[dict[str, Any]] = []
    for records in grouped.values():
        latest_end = max(record["old_end"] for record in records)
        latest = [record for record in records if record["old_end"] == latest_end]
        latest_holders = {record["old_holder_qid"] for record in latest}
        if len(latest_holders) != 1:
            rejections["ambiguous_previous_holder"] += 1
            continue
        selected.append(sorted(latest, key=lambda row: row["old_statement_id"])[0])

    by_subject_start: dict[tuple[str, str, datetime], list[dict[str, Any]]] = defaultdict(list)
    for record in selected:
        by_subject_start[
            (record["item_qid"], record["property_id"], record["new_start"])
        ].append(record)

    unambiguous: list[dict[str, Any]] = []
    for records in by_subject_start.values():
        if len({record["new_holder_qid"] for record in records}) != 1:
            rejections["multiple_new_holders_same_start"] += 1
            continue
        unambiguous.append(sorted(records, key=lambda row: row["new_statement_id"])[0])

    one_per_subject: dict[tuple[str, str], dict[str, Any]] = {}
    for record in sorted(
        unambiguous,
        key=lambda row: (
            row["item_qid"],
            row["property_id"],
            row["new_start"],
            row["new_statement_id"],
        ),
    ):
        key = (record["item_qid"], record["property_id"])
        if key in one_per_subject:
            rejections["additional_transition_same_subject"] += 1
            continue
        one_per_subject[key] = record
    return list(one_per_subject.values()), rejections


def model_identity_manifest() -> dict[str, Any]:
    manifest: dict[str, Any] = {}
    hex_revision = re.compile(rb"[0-9a-f]{40}")
    for model_name, spec in MODEL_IDENTITIES.items():
        path = Path(spec["path"])
        small_files = ["config.json", "model.safetensors.index.json", "tokenizer_config.json"]
        revisions: list[str] = []
        metadata_path = path / ".msc"
        if metadata_path.exists():
            revisions = sorted(
                {value.decode("ascii") for value in hex_revision.findall(metadata_path.read_bytes())}
            )
        weight_files = sorted(path.glob("*.safetensors"))
        manifest[model_name] = {
            "model_id": spec["model_id"],
            "local_path": str(path.relative_to(REPO_ROOT)),
            "public_release": spec["public_release"],
            "official_source": spec["official_source"],
            "local_metadata_revisions": revisions,
            "local_metadata_created_at": (path / ".mv").read_text(
                encoding="utf-8", errors="replace"
            ).strip()
            if (path / ".mv").exists()
            else None,
            "small_file_sha256": {
                name: file_digest(path / name) for name in small_files if (path / name).exists()
            },
            "weight_files": [
                {"name": file.name, "bytes": file.stat().st_size} for file in weight_files
            ],
        }
    return manifest


def load_tokenizers() -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizers: dict[str, Any] = {}
    for model_name, spec in MODEL_IDENTITIES.items():
        tokenizers[model_name] = AutoTokenizer.from_pretrained(
            spec["path"],
            trust_remote_code=True,
            local_files_only=True,
            use_fast=False,
        )
    return tokenizers


def normalized_length(value: str) -> int:
    return len("".join(character for character in value.strip() if not character.isspace()))


def surface_audit(
    active: str, matched_null: str, tokenizers: dict[str, Any]
) -> tuple[bool, dict[str, Any]]:
    active_words = len(active.split())
    null_words = len(matched_null.split())
    active_chars = normalized_length(active)
    null_chars = normalized_length(matched_null)
    max_chars = max(active_chars, null_chars, 1)
    min_chars = min(active_chars, null_chars)
    counts: dict[str, dict[str, int]] = {}
    token_ok = True
    for model_name, tokenizer in tokenizers.items():
        active_tokens = len(tokenizer.encode(f" {active}", add_special_tokens=False))
        null_tokens = len(tokenizer.encode(f" {matched_null}", add_special_tokens=False))
        counts[model_name] = {"active": active_tokens, "matched_null": null_tokens}
        token_ok = token_ok and (
            abs(active_tokens - null_tokens)
            <= SURFACE_POLICY["max_token_count_difference_per_tokenizer"]
            and max(active_tokens, null_tokens)
            <= SURFACE_POLICY["max_candidate_tokens_per_tokenizer"]
        )
    checks = {
        "different_strings": active.casefold() != matched_null.casefold(),
        "word_count_matched": abs(active_words - null_words)
        <= SURFACE_POLICY["max_word_count_difference"],
        "character_ratio_matched": min_chars / max_chars
        >= SURFACE_POLICY["min_character_length_ratio"],
        "character_difference_matched": abs(active_chars - null_chars)
        <= SURFACE_POLICY["max_character_length_difference"],
        "token_counts_matched": token_ok,
    }
    return all(checks.values()), {
        "checks": checks,
        "word_counts": {"active": active_words, "matched_null": null_words},
        "character_counts": {"active": active_chars, "matched_null": null_chars},
        "token_counts": counts,
    }


def add_labels_and_filter(
    records: list[dict[str, Any]], tokenizers: dict[str, Any]
) -> tuple[list[dict[str, Any]], Counter[str]]:
    accepted: list[dict[str, Any]] = []
    rejections: Counter[str] = Counter()
    for record in records:
        item_label = record.get("item_label")
        new_label = record.get("new_holder_label")
        old_label = record.get("old_holder_label")
        if not item_label or not new_label or not old_label:
            rejections["missing_english_label"] += 1
            continue
        if any(label.startswith("Q") and label[1:].isdigit() for label in (item_label, new_label, old_label)):
            rejections["qid_like_label"] += 1
            continue
        surface_ok, surface = surface_audit(new_label, old_label, tokenizers)
        if not surface_ok:
            for name, passed in surface["checks"].items():
                if not passed:
                    rejections[f"surface_{name}"] += 1
            continue
        enriched = dict(record)
        enriched["surface_audit"] = surface
        accepted.append(enriched)
    return accepted, rejections


def stable_rank(record: dict[str, Any]) -> str:
    key = "|".join(
        [
            record["item_qid"],
            record["property_id"],
            record["new_statement_id"],
            record["old_statement_id"],
        ]
    )
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def allocate_splits(records: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    ranked = sorted(records, key=lambda record: (stable_rank(record), record["item_qid"]))
    allocated: list[tuple[str, dict[str, Any]]] = []
    counts = Counter()
    for index, record in enumerate(ranked):
        split = SPLITS[index % len(SPLITS)]
        if not ALLOCATE_ALL_QUALIFIED and counts[split] >= TARGET_PER_SPLIT:
            continue
        allocated.append((split, record))
        counts[split] += 1
        if not ALLOCATE_ALL_QUALIFIED and all(
            counts[name] >= TARGET_PER_SPLIT for name in SPLITS
        ):
            break
    return allocated


def context_for(split: str, record: dict[str, Any]) -> str:
    relation = PROPERTIES[record["property_id"]]["label"]
    subject = record["item_label"]
    old = record["old_holder_label"]
    new = record["new_holder_label"]
    old_end = iso_date(record["old_end"])
    new_start = iso_date(record["new_start"])
    if PROPERTIES[record["property_id"]].get("template_kind") == "membership":
        if split == "discovery":
            return (
                f"The dated record states that {subject} played for {old} through "
                f"{old_end}, then joined {new} on {new_start}."
            )
        if split == "confirmation":
            return (
                f"After {subject}'s time with {old} ended on {old_end}, the record "
                f"lists a move to {new} beginning {new_start}."
            )
        return (
            f"A current career record for {subject} lists {old} up to {old_end} "
            f"and {new} from {new_start}."
        )
    if split == "discovery":
        return (
            f"The dated record states that {old} served as {relation} of {subject} "
            f"through {old_end}, and {new} began serving in that role on {new_start}."
        )
    if split == "confirmation":
        return (
            f"After {old}'s term as {subject}'s {relation} ended on {old_end}, "
            f"{new} took up the same role on {new_start}."
        )
    return (
        f"A current record for {subject} lists {old} as the former {relation}, "
        f"with the term ending {old_end}; it lists {new} from {new_start}."
    )


def query_for(record: dict[str, Any], query_date: datetime) -> str:
    if PROPERTIES[record["property_id"]].get("template_kind") == "membership":
        member_noun = PROPERTIES[record["property_id"]].get("member_noun", "team")
        return (
            f"Based only on the dated record, which {member_noun} did "
            f"{record['item_label']} play for on {iso_date(query_date)}?"
        )
    relation = PROPERTIES[record["property_id"]]["label"]
    return (
        f"Based only on the dated record, who served as {relation} of "
        f"{record['item_label']} on {iso_date(query_date)}?"
    )


def package_rows(allocated: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split, record in allocated:
        relation = PROPERTIES[record["property_id"]]["label"]
        item_id = f"phase1132-{record['property_id']}-{stable_rank(record)[:16]}"
        row = {
            "item_id": item_id,
            "source_corpus": "Wikidata live RDF statements frozen by Phase1132",
            "source_license": "CC0-1.0",
            "source_document_id": (
                f"{record['old_statement_id']}|{record['new_statement_id']}"
            ),
            "primitive_family": PRIMITIVE_FAMILY_OVERRIDE
            or (
                "temporal_entity_membership_binding"
                if PROPERTIES[record["property_id"]].get("template_kind")
                == "membership"
                else "temporal_entity_role_binding"
            ),
            "primitive_subfamily": (
                "membership"
                if PROPERTIES[record["property_id"]].get("template_kind")
                == "membership"
                else "officeholder"
            ),
            "split": split,
            "context": context_for(split, record),
            "query": query_for(record, record["post_query_date"]),
            "active_candidate": record["new_holder_label"],
            "matched_null_candidate": record["old_holder_label"],
            "gold_answer": record["new_holder_label"],
            "annotator_ids": [],
            "annotation_blinded_to_model_outputs": False,
            "candidate_uniqueness_confirmed": True,
            "matched_null_globally_false_confirmed": True,
            "matched_null_locally_plausible_confirmed": True,
            "null_frozen_before_model_scoring": True,
            "same_part_of_speech": True,
            "surface_length_matched": True,
            "generation_provenance": "deterministic_wikidata_postrelease_template_v1",
            "machine_validation_only": True,
            "context_origin": "deterministic_template_not_independent_natural_prose",
            "property_id": record["property_id"],
            "relation_label": relation,
            "domain": PROPERTIES[record["property_id"]]["domain"],
            "subject_qid": record["item_qid"],
            "subject_label": record["item_label"],
            "old_holder_qid": record["old_holder_qid"],
            "new_holder_qid": record["new_holder_qid"],
            "old_statement_id": record["old_statement_id"],
            "new_statement_id": record["new_statement_id"],
            "old_end": record["old_end"].isoformat(),
            "new_start": record["new_start"].isoformat(),
            "pre_query_date": iso_date(record["pre_query_date"]),
            "post_query_date": iso_date(record["post_query_date"]),
            "paired_pre_query": query_for(record, record["pre_query_date"]),
            "paired_pre_gold_answer": record["old_holder_label"],
            "surface_audit": record["surface_audit"],
        }
        rows.append(row)
    return rows


def build_review_queue(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "item_id": row["item_id"],
            "context": row["context"],
            "query": row["query"],
            "active_candidate": row["active_candidate"],
            "matched_null_candidate": row["matched_null_candidate"],
            "proposed_gold_answer": row["gold_answer"],
            "reviewer_id": None,
            "annotation_blinded_to_model_outputs": None,
            "gold_answer_correct": None,
            "candidate_unique": None,
            "matched_null_globally_false": None,
            "matched_null_locally_plausible": None,
            "natural_language_acceptable": None,
            "notes": None,
        }
        for row in rows
    ]


def serializable_record(record: dict[str, Any]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in record.items():
        value[key] = item.isoformat() if isinstance(item, datetime) else item
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="Refresh frozen web sources.")
    args = parser.parse_args()

    started = datetime.now(UTC)
    previous = read_json(PHASE1131_RESULT)
    protocol: dict[str, Any] = {
        "schema_version": "phase1132_postrelease_temporal_material_protocol.v1",
        "phase": PHASE,
        "revision": REVISION,
        "analysis_type": "postrelease_temporal_material_production_and_hard_gate",
        "fact_window": {
            "start": FACT_START.isoformat(),
            "end": FACT_END.isoformat(),
            "rationale": FACT_WINDOW_RATIONALE,
        },
        "properties": PROPERTIES,
        "target": {
            "minimum_per_split": TARGET_PER_SPLIT,
            "minimum_total": TARGET_TOTAL,
            "allocate_all_machine_qualified": ALLOCATE_ALL_QUALIFIED,
        },
        "surface_policy": SURFACE_POLICY,
        "query_policy": {
            "old_new_nonoverlap": True,
            "maximum_gap_days": MAX_TRANSITION_GAP_DAYS,
            "query_offset_days": QUERY_OFFSET_DAYS,
            "one_transition_per_subject_property": True,
        },
        "material_contract_source": "Phase1131 unchanged audit_package implementation",
        "model_execution": False,
        "tokenizer_execution_only": True,
        "postrelease_source_inventory": POST_RELEASE_SOURCE_INVENTORY,
        "phase1131_final_digest": previous["final_digest"],
    }
    protocol["protocol_digest"] = canonical_digest(protocol, "protocol_digest")
    write_json(RESULT_ROOT / "protocol/protocol.json", protocol)

    identity_manifest = model_identity_manifest()
    write_json(RESULT_ROOT / "protocol/local_model_identity.json", identity_manifest)

    raw_by_property = fetch_transition_bindings(args.refresh)
    normalized, structural_rejections = normalize_bindings(raw_by_property)
    tokenizers = load_tokenizers()
    filtered, surface_rejections = add_labels_and_filter(normalized, tokenizers)
    del tokenizers

    allocated = allocate_splits(filtered)
    rows = package_rows(allocated)
    write_jsonl(PACKAGE_PATH, rows)
    write_jsonl(
        MATERIAL_ROOT / "candidate_pool_machine_filtered.jsonl",
        (serializable_record(record) for record in sorted(filtered, key=stable_rank)),
    )
    write_jsonl(MATERIAL_ROOT / "human_review_queue.jsonl", build_review_queue(rows))
    write_json(
        MATERIAL_ROOT / "human_review_schema.json",
        {
            "required_reviewers": 2,
            "reviewers_must_be_independent": True,
            "reviewers_must_not_see_model_outputs": True,
            "all_boolean_fields_must_be_true_for_acceptance": [
                "gold_answer_correct",
                "candidate_unique",
                "matched_null_globally_false",
                "matched_null_locally_plausible",
                "natural_language_acceptable",
            ],
            "status": "awaiting_external_human_review",
        },
    )

    source_manifest = {
        "manifest_built_at": datetime.now(UTC).isoformat(),
        "endpoint": WIKIDATA_SPARQL,
        "license": "CC0-1.0",
        "raw_snapshot_provenance": RAW_SNAPSHOT_PROVENANCE,
        "raw_rows_by_property": {
            property_id: len(bindings) for property_id, bindings in raw_by_property.items()
        },
        "raw_file_sha256": {
            property_id: file_digest(RAW_ROOT / f"wikidata_{property_id}.json")
            for property_id in PROPERTIES
        },
        "normalized_transition_count": len(normalized),
        "machine_filtered_count": len(filtered),
        "allocated_count": len(rows),
        "split_counts": dict(Counter(row["split"] for row in rows)),
        "property_counts": dict(Counter(row["property_id"] for row in rows)),
        "structural_rejections": dict(structural_rejections),
        "surface_rejections": dict(surface_rejections),
        "candidate_package_sha256": file_digest(PACKAGE_PATH),
        "independence_limit": "All facts come from one structured corpus. Distinct statement IDs are split-disjoint, but corpus-level independence and natural-prose validity are not established.",
    }
    write_json(MATERIAL_ROOT / "source_manifest.json", source_manifest)

    phase1131_audit = audit_package(PACKAGE_PATH)
    blind_review_complete = phase1131_audit.get("checks", {}).get(
        "two_blind_annotators", False
    )
    source_supply_target_met = (
        len(rows) >= TARGET_TOTAL
        and all(
            source_manifest["split_counts"].get(split, 0) >= TARGET_PER_SPLIT
            for split in SPLITS
        )
    )
    model_test_authorized = bool(
        source_supply_target_met and phase1131_audit["material_ready"]
    )

    final: dict[str, Any] = {
        "schema_version": "phase1132_postrelease_temporal_material_final.v1",
        "phase": PHASE,
        "revision": REVISION,
        "protocol_digest": protocol["protocol_digest"],
        "started_at": started.isoformat(),
        "completed_at": datetime.now(UTC).isoformat(),
        "model_execution": False,
        "tokenizer_execution_only": True,
        "source_inventory_judgment": POST_RELEASE_SOURCE_INVENTORY,
        "model_identity_manifest": identity_manifest,
        "source_manifest": source_manifest,
        "phase1131_contract_audit": phase1131_audit,
        "material_state": {
            "postrelease_candidate_supply_target_met": source_supply_target_met,
            "two_independent_blind_reviews_complete": blind_review_complete,
            "independent_natural_prose_validated": False,
            "machine_material_ready": phase1131_audit["material_ready"],
        },
        "claim_corrections": [
            "Data after 2025-01-01 are not uniformly unseen because Qwen3 was released on 2025-04-29 and the local metadata lineage extends into July 2025.",
            "An event effective after checkpoint release may have been announced earlier, so effective date is a strong screen rather than absolute contamination proof.",
            "Correct use of a supplied update demonstrates protocol-bounded context use over a candidate prior, not an independent contextual workspace.",
            "Failure does not falsify relative coding; instruction following, temporal parsing, tokenization, and prior strength remain alternatives.",
            "Entity_A aliases are not a neutral tokenizer fix because prior phases showed temporary symbol mappings can become the task.",
            "Strong ignore-memory instructions must be a separate factor rather than the default prompt.",
            "Behavioral passage cannot authorize immediate ablation; hidden repeat, matched hidden null, specificity, and independent causal confirmation remain required.",
        ],
        "prospective_behavioral_contrasts": {
            "parametric_prior_margin": "H_prior = logit(old | no_context) - logit(new | no_context)",
            "post_context_margin": "M_post = logit(new | dated_context, t_post) - logit(old | dated_context, t_post)",
            "pre_context_margin": "M_pre = logit(old | dated_context, t_pre) - logit(new | dated_context, t_pre)",
            "bidirectional_binding_score": "B_bind = 0.5 * (M_post + M_pre)",
            "override_score": "O = M_post + H_prior",
            "interpretation": "These are behavior measures only and do not identify a hidden mechanism.",
        },
        "restart_decision": {
            "auto_continue": model_test_authorized,
            "model_test_authorized": model_test_authorized,
            "reason": (
                "All frozen material locks passed."
                if model_test_authorized
                else "A large post-release machine-filtered candidate package exists, but Phase1131 still fails independent blind human review and natural-prose validation."
            ),
            "next_legal_input": (
                None
                if model_test_authorized
                else "Two independent blind review manifests covering the frozen human_review_queue.jsonl, followed by a reviewed package audit."
            ),
        },
        "evidence_update": {
            "new_k_item": None,
            "theory_update_number": None,
            "reason": "Material supply is a method asset; no model behavior or internal mechanism was observed.",
        },
    }
    final["final_digest"] = canonical_digest(final, "final_digest")
    write_json(RESULT_ROOT / "analysis/final_summary.json", final)

    checks = {
        "phase1131_linked": previous["final_digest"] == protocol["phase1131_final_digest"],
        "fact_window_after_all_public_releases": all(
            FACT_START.date() > datetime.fromisoformat(spec["public_release"]).date()
            for spec in MODEL_IDENTITIES.values()
        ),
        "all_raw_sources_nonempty": all(raw_by_property.values()),
        "source_supply_target_met": source_supply_target_met,
        "candidate_package_digest_valid": file_digest(PACKAGE_PATH)
        == source_manifest["candidate_package_sha256"],
        "split_counts_meet_minimum": all(
            source_manifest["split_counts"].get(split, 0) >= TARGET_PER_SPLIT
            for split in SPLITS
        ),
        "nulls_prefrozen": all(row["null_frozen_before_model_scoring"] for row in rows),
        "tokenizer_only_no_model_execution": final["tokenizer_execution_only"]
        and not final["model_execution"],
        "blind_review_incomplete_detected": not blind_review_complete,
        "material_not_misclassified_ready": phase1131_audit["material_ready"] is False,
        "model_test_not_authorized": model_test_authorized is False,
        "no_new_k_item": final["evidence_update"]["new_k_item"] is None,
        "no_theory_update": final["evidence_update"]["theory_update_number"] is None,
        "protocol_digest_valid": canonical_digest(protocol, "protocol_digest")
        == protocol["protocol_digest"],
        "final_digest_valid": canonical_digest(final, "final_digest")
        == final["final_digest"],
    }
    audit: dict[str, Any] = {
        "schema_version": "phase1132_postrelease_temporal_material_audit.v1",
        "phase": PHASE,
        "revision": REVISION,
        "checks": checks,
        "passed_count": sum(bool(value) for value in checks.values()),
        "total_count": len(checks),
        "passed": all(checks.values()),
        "protocol_digest": protocol["protocol_digest"],
        "final_digest": final["final_digest"],
    }
    audit["audit_digest"] = canonical_digest(audit, "audit_digest")
    write_json(RESULT_ROOT / "audit/result_audit.json", audit)

    print(
        json.dumps(
            {
                "phase": PHASE,
                "revision": REVISION,
                "audit": f"{audit['passed_count']}/{audit['total_count']}",
                "audit_passed": audit["passed"],
                "raw_rows": sum(len(value) for value in raw_by_property.values()),
                "normalized": len(normalized),
                "machine_filtered": len(filtered),
                "allocated": len(rows),
                "split_counts": source_manifest["split_counts"],
                "phase1131_blockers": phase1131_audit["blockers"],
                "model_test_authorized": model_test_authorized,
                "auto_continue": final["restart_decision"]["auto_continue"],
                "protocol_digest": protocol["protocol_digest"],
                "final_digest": final["final_digest"],
                "audit_digest": audit["audit_digest"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
