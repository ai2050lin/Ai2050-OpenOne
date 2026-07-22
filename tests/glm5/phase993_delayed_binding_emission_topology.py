from __future__ import annotations

import gzip
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PHASE991 = ROOT / "tests/glm5/result/phase991_delayed_binding_gpu_admission"
PHASE992 = ROOT / "tests/glm5/result/phase992_delayed_binding_behavior_execution"
OUT_DIR = ROOT / "tests/glm5/result/phase993_delayed_binding_emission_topology"
OUT_PATH = OUT_DIR / "phase993_emission_topology.json"
MODELS = ("qwen3", "glm4", "deepseek7b")
SPLITS = ("discovery", "confirmation", "adversarial")
VALUES = ("red", "blue", "green", "black")
MARKER_RE = re.compile(r"(?<![A-Za-z])(red|blue|green|black)(?![A-Za-z])", re.I)
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pct(numerator: int, denominator: int) -> float:
    return 100.0 * numerator / denominator if denominator else 0.0


def words(text: str) -> list[str]:
    return [item.casefold() for item in WORD_RE.findall(text)]


def leading_prompt_overlap(prompt: str, generated: str, cap: int = 12) -> int:
    prompt_words = words(prompt)
    generated_words = words(generated)
    for size in range(min(cap, len(generated_words)), 0, -1):
        needle = tuple(generated_words[:size])
        if any(tuple(prompt_words[i : i + size]) == needle for i in range(len(prompt_words) - size + 1)):
            return size
    return 0


def exploratory_lead_class(generated: str) -> str:
    text = " ".join(words(generated)[:12])
    constraint = (
        "do not ",
        "use the ",
        "use a ",
        "instead use ",
        "answer in ",
        "the answer must ",
        "include the ",
    )
    answer = ("the answer is ", "the marker ", "the inner ", "the outer ")
    reasoning = ("okay ", "to solve ", "we need ", "i need ", "let's ", "first ")
    if text.startswith(constraint):
        return "constraint_continuation"
    if text.startswith(answer):
        return "answer_scaffold"
    if text.startswith(reasoning):
        return "reasoning_scaffold"
    return "other"


def load_truth() -> dict[str, dict]:
    truth: dict[str, dict] = {}
    for split in SPLITS:
        path = PHASE991 / f"scoring_truth/private/{split}.jsonl"
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                record_id = row["record_id"]
                if record_id in truth:
                    raise RuntimeError(f"duplicate truth record_id: {record_id}")
                truth[record_id] = row
    if len(truth) != 8192:
        raise RuntimeError(f"expected 8192 public truth rows, got {len(truth)}")
    return truth


def summarize_factor(cells: dict[str, dict[str, int]]) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for label in sorted(cells):
        cell = cells[label]
        denominator = cell["denominator"]
        parsed = cell["parsed"]
        correct = cell["correct"]
        result[label] = {
            "denominator": denominator,
            "parsed": parsed,
            "parsed_percent": pct(parsed, denominator),
            "correct": correct,
            "overall_correct_percent": pct(correct, denominator),
            "correct_given_parsed_percent_posthoc": pct(correct, parsed),
        }
    return result


def analyze_model(model: str, truth: dict[str, dict], published: dict) -> dict:
    raw_path = PHASE992 / f"raw/primary/{model}.jsonl.gz"
    counts = Counter()
    token_lengths = Counter()
    overlap_lengths = Counter()
    lead_classes = Counter()
    leading_six = Counter()
    factors: dict[str, dict[str, dict[str, int]]] = {
        name: defaultdict(lambda: {"denominator": 0, "parsed": 0, "correct": 0})
        for name in ("split", "semantic_transform", "paraphrase_id", "fact_order_id", "horizon_id")
    }
    record_ids: list[str] = []
    generated_hashes: list[str] = []

    with gzip.open(raw_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            record_id = row["record_id"]
            if record_id not in truth:
                raise RuntimeError(f"raw record absent from public truth: {record_id}")
            gold = truth[record_id]
            record_ids.append(record_id)
            generated = row["generated_text"]
            generated_hashes.append(sha256_bytes(generated.encode("utf-8")))
            matches = [match.casefold() for match in MARKER_RE.findall(generated)]
            first_marker = matches[0] if matches else None
            parsed = first_marker is not None
            correct = first_marker == gold["gold_value"]

            candidate_logits = {name: float(row["teacher_forced_candidates"][name]["logit"]) for name in VALUES}
            maximum = max(candidate_logits.values())
            maxima = [name for name in VALUES if candidate_logits[name] == maximum]
            tf_argmax = maxima[0]
            tf_strict_correct = candidate_logits[gold["gold_value"]] > max(
                value for name, value in candidate_logits.items() if name != gold["gold_value"]
            )

            counts["rows"] += 1
            counts["parsed"] += int(parsed)
            counts["correct"] += int(correct)
            counts["ambiguous_distinct"] += int(len(set(matches)) > 1)
            counts["budget_exhausted"] += int(bool(row["budget_exhausted"]))
            counts["eos_seen"] += int(bool(row["eos_seen"]))
            counts["tf_strict_correct"] += int(tf_strict_correct)
            counts["tf_any_argmax_tie"] += int(len(maxima) > 1)
            counts["tf_natural_agreement"] += int(parsed and first_marker == tf_argmax)
            counts["tf_and_natural_correct"] += int(tf_strict_correct and correct)
            counts["natural_parsed_given_tf_correct"] += int(tf_strict_correct and parsed)
            counts["natural_correct_given_tf_incorrect"] += int((not tf_strict_correct) and correct)

            token_lengths[len(row["generated_suffix_token_ids"])] += 1
            overlap_lengths[leading_prompt_overlap(row.get("prompt", "") or "", generated)] += 1
            # Raw rows intentionally omit prompt text; recover it from the public manifest below.
            lead_classes[exploratory_lead_class(generated)] += 1
            leading_six[" ".join(words(generated)[:6])] += 1

            labels = {
                "split": gold["split"],
                "semantic_transform": gold["semantic_transform"],
                "paraphrase_id": gold["paraphrase_id"],
                "fact_order_id": gold["fact_order_id"],
                "horizon_id": gold["horizon_id"],
            }
            for factor_name, label in labels.items():
                cell = factors[factor_name][label]
                cell["denominator"] += 1
                cell["parsed"] += int(parsed)
                cell["correct"] += int(correct)

    if len(record_ids) != len(set(record_ids)) or set(record_ids) != set(truth):
        raise RuntimeError(f"{model}: raw/truth identity mismatch")

    # Join prompts after raw identity is established. This keeps the overlap measure objective.
    prompts: dict[str, str] = {}
    with (PHASE992 / "manifests/primary.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            prompts[row["record_id"]] = row["prompt"]
    overlap_lengths.clear()
    with gzip.open(raw_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            overlap_lengths[leading_prompt_overlap(prompts[row["record_id"]], row["generated_text"])] += 1

    rows = counts["rows"]
    parsed = counts["parsed"]
    tf_correct = counts["tf_strict_correct"]
    if counts["correct"] != int(published["natural_generation"]["overall"]["correct"]):
        raise RuntimeError(f"{model}: recomputed natural score differs from frozen score")
    if tf_correct != int(published["teacher_forced_diagnostic"]["positive"]):
        raise RuntimeError(f"{model}: recomputed teacher-forced count differs from frozen score")

    return {
        "model": model,
        "raw_sha256": sha256_file(raw_path),
        "row_count": rows,
        "record_ids_sha256": sha256_bytes(canonical_bytes(sorted(record_ids))),
        "generated_text_identity_sha256": sha256_bytes(canonical_bytes(generated_hashes)),
        "termination": {
            "budget_exhausted": counts["budget_exhausted"],
            "budget_exhausted_percent": pct(counts["budget_exhausted"], rows),
            "eos_seen": counts["eos_seen"],
            "eos_seen_percent": pct(counts["eos_seen"], rows),
            "generated_token_length_counts": {str(key): value for key, value in sorted(token_lengths.items())},
        },
        "natural_marker": {
            "parsed": parsed,
            "parsed_percent": pct(parsed, rows),
            "correct": counts["correct"],
            "overall_correct_percent": pct(counts["correct"], rows),
            "correct_given_parsed_percent_posthoc": pct(counts["correct"], parsed),
            "multiple_distinct_markers": counts["ambiguous_distinct"],
        },
        "teacher_forced_vs_natural": {
            "teacher_forced_strict_correct": tf_correct,
            "teacher_forced_strict_correct_percent": pct(tf_correct, rows),
            "teacher_forced_any_candidate_argmax_ties": counts["tf_any_argmax_tie"],
            "teacher_forced_any_candidate_argmax_ties_definition": (
                "two or more of the four stored candidate logits share the maximum; this is not the frozen score's "
                "gold-versus-best-foil tie account"
            ),
            "natural_prediction_agrees_with_tf_argmax": counts["tf_natural_agreement"],
            "agreement_percent_all_rows": pct(counts["tf_natural_agreement"], rows),
            "both_strict_tf_and_natural_correct": counts["tf_and_natural_correct"],
            "natural_correct_given_strict_tf_correct_percent_posthoc": pct(counts["tf_and_natural_correct"], tf_correct),
            "natural_parsed_given_strict_tf_correct_percent_posthoc": pct(
                counts["natural_parsed_given_tf_correct"], tf_correct
            ),
            "natural_correct_given_tf_not_strict_correct": counts["natural_correct_given_tf_incorrect"],
            "natural_correct_given_tf_not_strict_correct_percent_posthoc": pct(
                counts["natural_correct_given_tf_incorrect"], rows - tf_correct
            ),
        },
        "prompt_continuation_topology_exploratory": {
            "definition": "posthoc surface description only; not a preregistered behavior gate",
            "leading_prompt_ngram_overlap_word_counts_capped_at_12": {
                str(key): value for key, value in sorted(overlap_lengths.items())
            },
            "leading_prompt_overlap_at_least_3_words": sum(value for key, value in overlap_lengths.items() if key >= 3),
            "leading_prompt_overlap_at_least_3_words_percent": pct(
                sum(value for key, value in overlap_lengths.items() if key >= 3), rows
            ),
            "lead_class_counts": dict(sorted(lead_classes.items())),
            "lead_class_percent": {key: pct(value, rows) for key, value in sorted(lead_classes.items())},
            "top_leading_six_word_prefixes": [
                {"prefix": key, "count": value, "percent": pct(value, rows)}
                for key, value in leading_six.most_common(12)
            ],
        },
        "surface_factor_accounts": {name: summarize_factor(cells) for name, cells in factors.items()},
    }


def main() -> None:
    score_path = PHASE992 / "scores/public_score.json"
    audit_path = PHASE992 / "scores/public_independent_audit.json"
    score = json.loads(score_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if score["passed"] is not False or audit["passed"] is not True:
        raise RuntimeError("Phase992 public score/audit terminal state is not the expected failed-and-verified state")
    truth = load_truth()
    model_reports = {model: analyze_model(model, truth, score["models"][model]) for model in MODELS}

    natural_rank = sorted(
        MODELS,
        key=lambda name: model_reports[name]["natural_marker"]["overall_correct_percent"],
        reverse=True,
    )
    teacher_forced_rank = sorted(
        MODELS,
        key=lambda name: model_reports[name]["teacher_forced_vs_natural"]["teacher_forced_strict_correct_percent"],
        reverse=True,
    )
    report = {
        "schema_version": "phase993_delayed_binding_emission_topology.v1",
        "phase": 993,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "role": "posthoc_external_emission_topology_after_verified_public_failure",
        "source_phase992_run_id": score["run_id"],
        "source_seals": {
            "phase992_score_file_sha256": sha256_file(score_path),
            "phase992_score_self_sha256": score["score_sha256"],
            "phase992_independent_audit_file_sha256": sha256_file(audit_path),
            "phase992_independent_audit_self_sha256": audit["audit_sha256"],
            "analysis_source_sha256": sha256_file(Path(__file__)),
        },
        "scope_guards": {
            "public_truth_was_opened_only_after_all_three_phase992_receipts": True,
            "sealed_holdout_opened": False,
            "sealed_holdout_scored": False,
            "model_execution_performed": False,
            "internal_trace_read": False,
            "causal_intervention_performed": False,
            "posthoc_measures_are_not_preregistered_gates": True,
        },
        "models": model_reports,
        "cross_model_observations": {
            "all_models_exhausted_24_token_budget_on_all_rows": all(
                report["termination"]["generated_token_length_counts"] == {"24": 8192}
                and report["termination"]["budget_exhausted"] == 8192
                for report in model_reports.values()
            ),
            "all_models_emitted_zero_eos": all(report["termination"]["eos_seen"] == 0 for report in model_reports.values()),
            "natural_rank": natural_rank,
            "teacher_forced_rank": teacher_forced_rank,
            "natural_and_teacher_forced_rank_disagree": natural_rank != teacher_forced_rank,
            "scientific_interpretation": [
                "The frozen raw-continuation interface mixes retrieval, answer emission, formatting, and termination.",
                "Repeated external surface behavior is not an internal neural structure.",
                "A native-chat interface comparison must be newly frozen before it can adjudicate interface mismatch.",
                "No internal trace or mechanism formula is authorized by this report.",
            ],
        },
    }
    report["artifact_sha256"] = sha256_bytes(canonical_bytes(report))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_bytes(canonical_bytes(report))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
