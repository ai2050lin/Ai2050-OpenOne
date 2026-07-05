#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase937_semantic_reuse_difference_state_atlas as p937  # noqa: E402
import phase938_semantic_factor_causal_transfer_audit as p938  # noqa: E402


PHASE = 939
MODELS = ["qwen3", "glm4", "deepseek7b"]
RESULT_ROOT = Path("tests/result/phase939_bilingual_specificity_tightening_audit")

ZH_OBJECT = {
    "apple": "苹果",
    "cherry": "樱桃",
    "banana": "香蕉",
    "lemon": "柠檬",
    "orange": "橙子",
    "grape": "葡萄",
    "cardinal": "红雀",
    "canary": "金丝雀",
    "shark": "鲨鱼",
    "whale": "鲸鱼",
    "horse": "马",
    "dog": "狗",
    "car": "汽车",
    "bus": "公交车",
    "taxi": "出租车",
    "bicycle": "自行车",
    "train": "火车",
    "boat": "船",
    "knife": "刀",
    "saw": "锯子",
    "key": "钥匙",
    "hammer": "锤子",
    "cup": "杯子",
    "spoon": "勺子",
    "metal": "金属",
    "glass": "玻璃",
    "wood": "木头",
    "stone": "石头",
    "plastic": "塑料",
    "rubber": "橡胶",
}

PROMPT_TEMPLATES: dict[str, list[dict[str, str]]] = {
    "category": [
        {"template_id": "en_0", "language": "en", "text": "In one word, {article} {object} is a type of"},
        {"template_id": "en_1", "language": "en", "text": "The category of {article} {object} is"},
        {"template_id": "zh_0", "language": "zh", "text": "请用英文一个词回答：{zh_object}属于什么类别？"},
        {"template_id": "zh_1", "language": "zh", "text": "请只回答一个英文词：{zh_object}是一种什么？"},
    ],
    "color": [
        {"template_id": "en_0", "language": "en", "text": "The typical color of {article} {object} is"},
        {"template_id": "en_1", "language": "en", "text": "A common color for {article} {object} is"},
        {"template_id": "zh_0", "language": "zh", "text": "请用英文一个词回答：{zh_object}通常是什么颜色？"},
        {"template_id": "zh_1", "language": "zh", "text": "请只回答一个英文颜色词：{zh_object}的典型颜色是"},
    ],
    "function": [
        {"template_id": "en_0", "language": "en", "text": "{Article} {object} can often"},
        {"template_id": "en_1", "language": "en", "text": "A common action or use for {article} {object} is to"},
        {"template_id": "zh_0", "language": "zh", "text": "请用英文一个词回答：{zh_object}通常可以用来"},
        {"template_id": "zh_1", "language": "zh", "text": "请只回答一个英文动词：{zh_object}常见用途是"},
    ],
}

CONTROL_CONDITIONS = {
    "wrong_label_direction",
    "wrong_mean_direction",
    "random_same_norm",
    "negative_target_direction",
    "template_shift_same_norm",
}
TARGET_CONDITIONS = {
    "target_direction",
    "wrong_mean_subtracted",
    "template_subtracted",
    "specific_direction",
}


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def finite(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def mean(values: list[float | int | None]) -> float | None:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return None if not vals else float(sum(vals) / len(vals))


def parse_csv(text: str) -> list[str]:
    return [part.strip() for part in str(text or "").split(",") if part.strip()]


def selected_cases(args: argparse.Namespace) -> list[dict[str, str]]:
    domains = set(parse_csv(args.domains)) if args.domains else set()
    per_domain: Counter[str] = Counter()
    out = []
    for case in p937.semantic_cases():
        if domains and case["domain"] not in domains:
            continue
        if int(args.max_objects_per_domain) > 0 and per_domain[case["domain"]] >= int(args.max_objects_per_domain):
            continue
        copied = dict(case)
        copied["zh_object"] = ZH_OBJECT.get(copied["object"], copied["object"])
        out.append(copied)
        per_domain[copied["domain"]] += 1
    return out


def selected_templates(relation: str, templates_per_language: int) -> list[dict[str, str]]:
    out = []
    for language in ["en", "zh"]:
        lang_templates = [row for row in PROMPT_TEMPLATES[relation] if row["language"] == language]
        out.extend(lang_templates[: max(1, int(templates_per_language))])
    return out


def build_samples(args: argparse.Namespace) -> list[dict[str, Any]]:
    relations = parse_csv(args.relations)
    samples = []
    for case in selected_cases(args):
        article = "an" if case["object"][0].lower() in {"a", "e", "i", "o", "u"} else "a"
        format_case = {**case, "article": article, "Article": article.capitalize()}
        for relation in relations:
            target = case["domain"] if relation == "category" else case[relation]
            for template in selected_templates(relation, int(args.templates_per_language)):
                sample_id = f"{case['domain']}:{case['object']}:{relation}:{template['template_id']}"
                samples.append(
                    {
                        "phase": PHASE,
                        "sample_id": sample_id,
                        "domain": case["domain"],
                        "object": case["object"],
                        "zh_object": case["zh_object"],
                        "relation": relation,
                        "target_label": target,
                        "prompt_language": template["language"],
                        "prompt_template": f"{relation}_{template['template_id']}",
                        "template_id": template["template_id"],
                        "prompt": template["text"].format(**format_case),
                    }
                )
    return samples


def vector_mean(rows: list[dict[str, Any]], vectors: dict[tuple[str, int], torch.Tensor], hidden_idx: int) -> torch.Tensor | None:
    vals = [vectors[(p938.sample_key(row), hidden_idx)] for row in rows if (p938.sample_key(row), hidden_idx) in vectors]
    if not vals:
        return None
    return torch.stack(vals).mean(dim=0).float().cpu()


def nonzero(vec: torch.Tensor | None, eps: float = 1e-8) -> bool:
    return vec is not None and float(torch.linalg.vector_norm(vec.float()).item()) > eps


def scale_to_norm(vec: torch.Tensor | None, reference: torch.Tensor) -> torch.Tensor | None:
    if not nonzero(vec) or not nonzero(reference):
        return None
    v = vec.float().cpu()
    return v / torch.linalg.vector_norm(v) * torch.linalg.vector_norm(reference.float().cpu())


def orthonormal_basis(vectors: list[torch.Tensor | None], eps: float = 1e-8) -> list[torch.Tensor]:
    basis: list[torch.Tensor] = []
    for vec in vectors:
        if not nonzero(vec, eps):
            continue
        work = vec.float().cpu().clone()
        for q in basis:
            work = work - torch.dot(work, q) * q
        norm = torch.linalg.vector_norm(work)
        if float(norm.item()) > eps:
            basis.append(work / norm)
    return basis


def remove_projection(vec: torch.Tensor, vectors: list[torch.Tensor | None]) -> torch.Tensor | None:
    if not nonzero(vec):
        return None
    out = vec.float().cpu().clone()
    for q in orthonormal_basis(vectors):
        out = out - torch.dot(out, q) * q
    return out if nonzero(out) else None


def build_direction_specs(
    samples: list[dict[str, Any]],
    vectors: dict[tuple[str, int], torch.Tensor],
    hidden_by_relation: dict[str, int],
    min_train_per_label: int,
) -> list[dict[str, Any]]:
    specs = []
    for relation, hidden_idx in sorted(hidden_by_relation.items()):
        rel_samples = [row for row in samples if str(row["relation"]) == relation]
        labels = sorted({str(row["target_label"]) for row in rel_samples})
        templates = sorted({str(row["prompt_template"]) for row in rel_samples})
        for train_template in templates:
            train = [row for row in rel_samples if str(row["prompt_template"]) == train_template]
            train_language = str(train[0]["prompt_language"]) if train else "unknown"
            label_counts = Counter(str(row["target_label"]) for row in train)
            directions: dict[str, torch.Tensor] = {}
            for label in labels:
                if label_counts[label] < int(min_train_per_label):
                    continue
                direction = p938.direction_for_label(train, vectors, int(hidden_idx), label)
                if nonzero(direction):
                    directions[label] = direction.float().cpu()
            if not directions:
                continue
            train_mean = vector_mean(train, vectors, int(hidden_idx))
            for test_template in [template for template in templates if template != train_template]:
                test_all = [row for row in rel_samples if str(row["prompt_template"]) == test_template]
                if not test_all:
                    continue
                test_language = str(test_all[0]["prompt_language"])
                test_mean = vector_mean(test_all, vectors, int(hidden_idx))
                template_shift = None
                if train_mean is not None and test_mean is not None:
                    template_shift = train_mean - test_mean
                for label, target_direction in directions.items():
                    test = [row for row in test_all if str(row["target_label"]) == label]
                    if not test:
                        continue
                    wrong_items = [(item, vec) for item, vec in sorted(directions.items()) if item != label]
                    wrong_label = wrong_items[0][0] if wrong_items else None
                    wrong_direction = wrong_items[0][1] if wrong_items else None
                    wrong_mean = None
                    if wrong_items:
                        wrong_mean = torch.stack([vec for _label, vec in wrong_items]).mean(dim=0).float().cpu()
                    wrong_mean_scaled = scale_to_norm(wrong_mean, target_direction)
                    template_shift_scaled = scale_to_norm(template_shift, target_direction)
                    wrong_mean_subtracted = remove_projection(target_direction, [wrong_mean])
                    template_subtracted = remove_projection(target_direction, [template_shift])
                    specific_direction = remove_projection(target_direction, [wrong_mean, template_shift])
                    random_direction = p938.deterministic_random_same_norm(
                        target_direction, f"{relation}|{hidden_idx}|{label}|{train_template}|{test_template}|phase939"
                    )
                    specs.append(
                        {
                            "relation": relation,
                            "hidden_idx": int(hidden_idx),
                            "target_label": label,
                            "train_template": train_template,
                            "test_template": test_template,
                            "train_language": train_language,
                            "test_language": test_language,
                            "language_pair": f"{train_language}->{test_language}",
                            "test_samples": test,
                            "target_direction": target_direction,
                            "wrong_label": wrong_label,
                            "wrong_direction": wrong_direction,
                            "wrong_mean_direction": wrong_mean_scaled,
                            "template_shift_same_norm": template_shift_scaled,
                            "wrong_mean_subtracted": wrong_mean_subtracted,
                            "template_subtracted": template_subtracted,
                            "specific_direction": specific_direction,
                            "random_direction": random_direction,
                            "direction_norm": float(torch.linalg.vector_norm(target_direction).item()),
                            "specific_direction_norm": None
                            if specific_direction is None
                            else float(torch.linalg.vector_norm(specific_direction).item()),
                            "template_shift_norm": None
                            if template_shift is None
                            else float(torch.linalg.vector_norm(template_shift).item()),
                            "wrong_mean_norm": None
                            if wrong_mean is None
                            else float(torch.linalg.vector_norm(wrong_mean).item()),
                            "train_label_count": int(label_counts[label]),
                            "train_other_count": int(len(train) - label_counts[label]),
                        }
                    )
    return specs


def make_row(
    model_name: str,
    sample: dict[str, Any],
    spec: dict[str, Any],
    condition: str,
    alpha: float | None,
    base_metrics: dict[str, Any],
    patched_metrics: dict[str, Any],
) -> dict[str, Any]:
    base_logit = base_metrics.get("target_label_logit")
    patched_logit = patched_metrics.get("target_label_logit")
    base_margin = base_metrics.get("target_margin_vs_relation_best_other")
    patched_margin = patched_metrics.get("target_margin_vs_relation_best_other")
    base_rank = base_metrics.get("target_label_rank")
    patched_rank = patched_metrics.get("target_label_rank")
    return {
        "phase": PHASE,
        "row_kind": "phase939_bilingual_specificity_row",
        "model": model_name,
        "sample_id": sample.get("sample_id"),
        "domain": sample.get("domain"),
        "object": sample.get("object"),
        "zh_object": sample.get("zh_object"),
        "relation": spec.get("relation"),
        "target_label": spec.get("target_label"),
        "prompt_language": sample.get("prompt_language"),
        "train_language": spec.get("train_language"),
        "test_language": spec.get("test_language"),
        "language_pair": spec.get("language_pair"),
        "prompt_template": sample.get("prompt_template"),
        "train_template": spec.get("train_template"),
        "test_template": spec.get("test_template"),
        "hidden_idx": int(spec.get("hidden_idx")),
        "condition": condition,
        "alpha": alpha,
        "wrong_label": spec.get("wrong_label"),
        "direction_norm": spec.get("direction_norm"),
        "specific_direction_norm": spec.get("specific_direction_norm"),
        "template_shift_norm": spec.get("template_shift_norm"),
        "wrong_mean_norm": spec.get("wrong_mean_norm"),
        "train_label_count": spec.get("train_label_count"),
        "train_other_count": spec.get("train_other_count"),
        "base_target_label_logit": base_logit,
        "patched_target_label_logit": patched_logit,
        "target_logit_delta": None if base_logit is None or patched_logit is None else float(patched_logit - base_logit),
        "base_target_margin": base_margin,
        "patched_target_margin": patched_margin,
        "target_margin_delta": None if base_margin is None or patched_margin is None else float(patched_margin - base_margin),
        "base_target_label_rank": base_rank,
        "patched_target_label_rank": patched_rank,
        "target_rank_delta": None if base_rank is None or patched_rank is None else int(patched_rank) - int(base_rank),
        "rank_improved": bool(base_rank is not None and patched_rank is not None and int(patched_rank) < int(base_rank)),
        "base_relation_winner": base_metrics.get("relation_winner"),
        "patched_relation_winner": patched_metrics.get("relation_winner"),
        "base_target_is_relation_winner": base_metrics.get("target_is_relation_winner"),
        "patched_target_is_relation_winner": patched_metrics.get("target_is_relation_winner"),
        "new_relation_winner_target": bool(
            (not base_metrics.get("target_is_relation_winner")) and patched_metrics.get("target_is_relation_winner")
        ),
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "target_logit_delta_mean": mean([row.get("target_logit_delta") for row in rows]),
        "target_margin_delta_mean": mean([row.get("target_margin_delta") for row in rows]),
        "rank_improved": sum(1 for row in rows if row.get("rank_improved")),
        "new_relation_winner_target": sum(1 for row in rows if row.get("new_relation_winner_target")),
        "patched_relation_winner_target": sum(1 for row in rows if row.get("patched_target_is_relation_winner")),
        "base_relation_winner_target": sum(1 for row in rows if row.get("base_target_is_relation_winner")),
    }


def summarize_by(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row.get(key) for key in keys)].append(row)
    out = []
    for key_tuple, items in buckets.items():
        item = {key: value for key, value in zip(keys, key_tuple)}
        item.update(summarize_rows(items))
        out.append(item)
    out.sort(
        key=lambda row: (
            finite(row.get("target_margin_delta_mean"), -999.0),
            finite(row.get("target_logit_delta_mean"), -999.0),
        ),
        reverse=True,
    )
    return out


def specificity_rows(by_relation_language_condition: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for row in by_relation_language_condition:
        groups[(row.get("relation"), row.get("language_pair"), row.get("alpha"))].append(row)
    out = []
    for (relation, language_pair, alpha), items in groups.items():
        by_cond = {row.get("condition"): row for row in items}
        controls = [by_cond[name] for name in CONTROL_CONDITIONS if name in by_cond]
        control_best = None if not controls else max(finite(row.get("target_margin_delta_mean"), -999.0) for row in controls)
        for condition in sorted(TARGET_CONDITIONS):
            row = by_cond.get(condition)
            if not row:
                continue
            margin = finite(row.get("target_margin_delta_mean"), -999.0)
            out.append(
                {
                    "relation": relation,
                    "language_pair": language_pair,
                    "alpha": alpha,
                    "condition": condition,
                    "rows": row.get("rows"),
                    "target_margin_delta_mean": row.get("target_margin_delta_mean"),
                    "target_logit_delta_mean": row.get("target_logit_delta_mean"),
                    "control_best_margin_delta": control_best,
                    "specificity_gain_vs_best_control": None if control_best is None else float(margin - control_best),
                    "rank_improved": row.get("rank_improved"),
                    "new_relation_winner_target": row.get("new_relation_winner_target"),
                }
            )
    out.sort(
        key=lambda row: (
            finite(row.get("specificity_gain_vs_best_control"), -999.0),
            finite(row.get("target_margin_delta_mean"), -999.0),
        ),
        reverse=True,
    )
    return out


def evidence_label(spec_rows: list[dict[str, Any]]) -> str:
    rel_positive = set()
    bilingual_positive = set()
    raw_only = set()
    for row in spec_rows:
        if row.get("condition") not in {"specific_direction", "target_direction"}:
            continue
        relation = str(row.get("relation"))
        language_pair = str(row.get("language_pair"))
        margin = finite(row.get("target_margin_delta_mean"), 0.0)
        gain = finite(row.get("specificity_gain_vs_best_control"), -999.0)
        if margin > 0 and gain > 0.02:
            if row.get("condition") == "specific_direction":
                rel_positive.add(relation)
                if language_pair in {"en->zh", "zh->en"}:
                    bilingual_positive.add(relation)
            elif row.get("condition") == "target_direction":
                raw_only.add(relation)
    if len(rel_positive) >= 2 and len(bilingual_positive) >= 1:
        return "bilingual_specific_semantic_transfer_retained"
    if len(rel_positive) >= 1:
        return "partial_specific_semantic_transfer_retained"
    if len(raw_only) >= 1:
        return "raw_transfer_but_specificity_weak"
    return "specific_semantic_transfer_weak_or_negative"


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = build_samples(args)
    hidden_by_relation = p938.phase937_best_hidden(args.model, args.phase937_round)
    if not hidden_by_relation:
        hidden_by_relation = {relation: -1 for relation in parse_csv(args.relations)}
    dry_payload = {
        "phase": PHASE,
        "title": "Bilingual Specificity Tightening Audit",
        "model": args.model,
        "sample_count": len(samples),
        "objects": len({row["object"] for row in samples}),
        "relations": sorted({str(row["relation"]) for row in samples}),
        "languages": sorted({str(row["prompt_language"]) for row in samples}),
        "templates_per_language": int(args.templates_per_language),
        "hidden_by_relation": hidden_by_relation,
    }
    if args.dry_run:
        payload = {**dry_payload, "status": "dry_run", "samples": samples[:24]}
        write_json(out_dir / f"phase939_{args.model}_summary.json", payload)
        write_jsonl(out_dir / f"phase939_{args.model}_rows.jsonl", [])
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return payload

    model = None
    tokenizer = None
    attn_impl = None
    rows: list[dict[str, Any]] = []
    direction_specs: list[dict[str, Any]] = []
    try:
        model, tokenizer, device, attn_impl = p938.p862.p844.p828.p796.load_model_bf16_prefer_flash(
            args.model, args.attn_implementations
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        auto_indices = p937.auto_hidden_indices(model)
        hidden_by_relation = {
            rel: (auto_indices[len(auto_indices) // 2] if int(idx) < 0 else int(idx))
            for rel, idx in hidden_by_relation.items()
        }
        hidden_indices = sorted(set(hidden_by_relation.values()))
        vectors = p938.forward_vectors(model, tokenizer, device, samples, hidden_indices, int(args.batch_size))
        baseline_logits = p938.forward_logits(model, tokenizer, device, samples, int(args.batch_size))
        direction_specs = build_direction_specs(samples, vectors, hidden_by_relation, int(args.min_train_per_label))
        alphas = [float(x) for x in parse_csv(args.alphas)]
        labels_by_relation = {relation: p938.relation_labels(samples, relation) for relation in hidden_by_relation}
        token_maps = {relation: p938.label_token_map(tokenizer, labels) for relation, labels in labels_by_relation.items()}
        baseline_metrics: dict[str, dict[str, Any]] = {}
        for sample in samples:
            relation = str(sample["relation"])
            baseline_metrics[p938.sample_key(sample)] = p938.target_margin(
                baseline_logits[p938.sample_key(sample)],
                str(sample["target_label"]),
                token_maps[relation],
            )

        for spec_idx, spec in enumerate(direction_specs, 1):
            relation = str(spec["relation"])
            label_tokens = token_maps[relation]
            for sample in spec["test_samples"]:
                base = baseline_metrics[p938.sample_key(sample)]
                rows.append(make_row(args.model, sample, spec, "baseline", None, base, base))
            conditions = [
                ("target_direction", spec.get("target_direction")),
                ("wrong_mean_subtracted", spec.get("wrong_mean_subtracted")),
                ("template_subtracted", spec.get("template_subtracted")),
                ("specific_direction", spec.get("specific_direction")),
                ("wrong_label_direction", spec.get("wrong_direction")),
                ("wrong_mean_direction", spec.get("wrong_mean_direction")),
                ("random_same_norm", spec.get("random_direction")),
                ("negative_target_direction", -spec["target_direction"]),
                ("template_shift_same_norm", spec.get("template_shift_same_norm")),
            ]
            for condition, direction in conditions:
                if direction is None:
                    continue
                for alpha in alphas:
                    patched = p938.patched_logits_batch(
                        model,
                        tokenizer,
                        device,
                        spec["test_samples"],
                        int(spec["hidden_idx"]),
                        direction,
                        float(alpha),
                        int(args.batch_size),
                    )
                    for sample in spec["test_samples"]:
                        base = baseline_metrics[p938.sample_key(sample)]
                        patched_metrics = p938.target_margin(
                            patched[p938.sample_key(sample)],
                            str(sample["target_label"]),
                            label_tokens,
                        )
                        rows.append(make_row(args.model, sample, spec, condition, float(alpha), base, patched_metrics))
            if spec_idx % max(1, int(args.log_every)) == 0 or spec_idx == len(direction_specs):
                log(f"{args.model}/{args.round_name}: direction_spec={spec_idx}/{len(direction_specs)} rows={len(rows)}")
    finally:
        if model is not None:
            p938.p862.p844.p828.release_model(model)
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    by_condition = summarize_by(rows, ["condition", "alpha"])
    by_relation_language_condition = summarize_by(rows, ["relation", "language_pair", "condition", "alpha"])
    by_language_condition = summarize_by(rows, ["language_pair", "condition", "alpha"])
    spec_rows = specificity_rows(by_relation_language_condition)
    evidence = evidence_label(spec_rows)
    payload = {
        **dry_payload,
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "hidden_by_relation": hidden_by_relation,
        "direction_specs": len(direction_specs),
        "rows": len(rows),
        "overall": summarize_rows(rows),
        "by_condition": by_condition,
        "by_language_condition": by_language_condition,
        "by_relation_language_condition": by_relation_language_condition,
        "specificity_rows": spec_rows,
        "evidence_label": evidence,
        "boundary": "bilingual English-label prompt audit with generic-direction orthogonalization; not natural semantic gate closure",
    }
    write_json(out_dir / f"phase939_{args.model}_summary.json", payload)
    write_jsonl(out_dir / f"phase939_{args.model}_rows.jsonl", rows)
    print(
        json.dumps(
            {
                "phase": PHASE,
                "model": args.model,
                "status": "complete",
                "evidence": evidence,
                "top_specificity_rows": spec_rows[:16],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return payload


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = [read_json(out_dir / f"phase939_{model}_summary.json") for model in MODELS]
    summaries = [summary for summary in summaries if summary]
    evidence_counts = Counter(str(summary.get("evidence_label") or summary.get("status")) for summary in summaries)
    condition_rows = []
    language_rows = []
    specificity = []
    for summary in summaries:
        model = summary.get("model")
        for row in summary.get("by_condition") or []:
            item = dict(row)
            item["model"] = model
            condition_rows.append(item)
        for row in summary.get("by_language_condition") or []:
            item = dict(row)
            item["model"] = model
            language_rows.append(item)
        for row in summary.get("specificity_rows") or []:
            item = dict(row)
            item["model"] = model
            specificity.append(item)
    specificity.sort(
        key=lambda row: (
            finite(row.get("specificity_gain_vs_best_control"), -999.0),
            finite(row.get("target_margin_delta_mean"), -999.0),
        ),
        reverse=True,
    )
    payload = {
        "phase": PHASE,
        "round": round_name,
        "status": "complete" if summaries else "missing",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [summary.get("model") for summary in summaries],
        "evidence_counts": dict(evidence_counts),
        "model_summaries": summaries,
        "condition_rows": condition_rows,
        "language_condition_rows": language_rows,
        "top_specificity_rows": specificity[:180],
    }
    write_json(out_dir / "phase939_cross_model_summary.json", payload)
    write_summary_md(out_dir / "phase939_cross_model_summary.md", payload)
    return payload


def write_summary_md(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# Phase 939 bilingual specificity tightening audit", ""]
    lines += ["## Evidence", ""]
    for key, value in sorted((payload.get("evidence_counts") or {}).items()):
        lines.append(f"- {key}: {value}")
    lines += ["", "## Condition Rows", ""]
    lines.append("| model | condition | alpha | rows | mean logit delta | mean margin delta | rank improved | new winner |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in payload.get("condition_rows") or []:
        lines.append(
            "| {model} | {condition} | {alpha} | {rows} | {target_logit_delta_mean} | {target_margin_delta_mean} | {rank_improved} | {new_relation_winner_target} |".format(
                **row
            )
        )
    lines += ["", "## Language Pair Rows", ""]
    lines.append("| model | pair | condition | alpha | rows | mean margin delta | rank improved | new winner |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for row in payload.get("language_condition_rows") or []:
        lines.append(
            "| {model} | {language_pair} | {condition} | {alpha} | {rows} | {target_margin_delta_mean} | {rank_improved} | {new_relation_winner_target} |".format(
                **row
            )
        )
    lines += ["", "## Top Specificity Rows", ""]
    lines.append("| model | relation | pair | condition | alpha | rows | margin | control best | specificity gain |")
    lines.append("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for row in payload.get("top_specificity_rows") or []:
        lines.append(
            "| {model} | {relation} | {language_pair} | {condition} | {alpha} | {rows} | {target_margin_delta_mean} | {control_best_margin_delta} | {specificity_gain_vs_best_control} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="bilingual_specificity_tightening_audit")
    parser.add_argument("--phase937-round", default="semantic_reuse_difference_state_atlas")
    parser.add_argument("--domains", default="")
    parser.add_argument("--relations", default="category,color,function")
    parser.add_argument("--max-objects-per-domain", type=int, default=6)
    parser.add_argument("--templates-per-language", type=int, default=2)
    parser.add_argument("--min-train-per-label", type=int, default=2)
    parser.add_argument("--alphas", default="1.0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-round", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.summarize_round:
        payload = summarize_round(args.round_name)
        print(json.dumps({"phase": PHASE, "status": payload["status"], "evidence": payload["evidence_counts"]}, ensure_ascii=False, indent=2))
        return
    if not args.model:
        raise SystemExit("--model is required unless --summarize-round is set")
    eval_model(args)


if __name__ == "__main__":
    main()
