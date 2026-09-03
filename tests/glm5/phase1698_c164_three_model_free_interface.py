#!/usr/bin/env python3
"""C164: sequential three-model qualification on natural free-word graph interfaces."""
from __future__ import annotations

import argparse
import gc
import itertools
import json
import re
import sys
import time
import unicodedata
from collections import defaultdict
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase1698_c164_three_model_free_interface"
C163 = RESULT / "phase1697_c163_natural_graph_call_domain"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import MODELS, load_bf16, quantization_audit, release_bf16
from model_utils import get_model_info
import phase1693_c159_natural_isomorphic_dual_graph_atlas as c159

PHASE, CAMPAIGN = 1698, "C164"
INTERFACES = ("plain_target", "natural_sentence")
BATCH = {"qwen3": 8, "glm4": 2, "deepseek7b": 2}
MAX_NEW_TOKENS = 24


def now():
    return datetime.now(timezone.utc).isoformat()


def normalize(value):
    value = unicodedata.normalize("NFKC", value).casefold()
    value = re.sub(r"<think>.*?</think>", " ", value, flags=re.S)
    value = re.sub(r"[^\w\s-]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def partition(unit):
    return ("discovery", "confirmation", "fresh")[unit % 3]


def edge(a, relation, b, surface):
    if surface == 1:
        return f"{a} {relation} {b}"
    return f"The registry records a directed '{relation}' link from {a} to {b}"


def make_case(unit, target_factor, path_factor, surface):
    relation_key, relation, source, b1, b2, t0, t1, other0, other1 = c159.UNITS[unit]
    intended, alternative = (t0, t1) if target_factor == 1 else (t1, t0)
    if path_factor == 1:
        edges = [(source, intended)]
    else:
        edges = [(source, b1), (b1, b2), (b2, intended)]
    distractors = [(other0, alternative), (other1, b1)]
    sentences = [edge(a, relation, b, surface) for a, b in edges]
    sentences += [f"{a} is merely associated with {b}" for a, b in distractors]
    if surface == -1:
        sentences.reverse()
        registry = "The registry contains these entries: " + " | ".join(sentences) + "."
    else:
        registry = "; ".join(sentences) + "."
    question = (
        f"{registry} Start at {source} and follow only directed '{relation}' links. "
        "What is the reachable target?"
    )
    return {
        "case_id": "",
        "unit": unit,
        "partition": partition(unit),
        "relation_family": relation_key,
        "relation_phrase": relation,
        "target_factor": target_factor,
        "path_factor": path_factor,
        "surface_factor": surface,
        "source": source,
        "intermediate_1": b1,
        "intermediate_2": b2,
        "intended": intended,
        "alternative": alternative,
        "semantic_edges": edges,
        "question": question,
    }


def material():
    cases = []
    for unit in range(len(c159.UNITS)):
        for factors in itertools.product((1, -1), repeat=3):
            row = make_case(unit, *factors)
            row["case_id"] = f"c164-{len(cases):03d}"
            cases.append(row)
    return cases


def messages(row, interface):
    system = "Use only the supplied directed registry. Do not use outside knowledge."
    if interface == "plain_target":
        instruction = " Reply with only the exact target phrase, without explanation."
    else:
        instruction = " Reply with one short sentence of the form: The reachable target is <target>."
    return [{"role": "system", "content": system}, {"role": "user", "content": row["question"] + instruction}]


def render_ids(tokenizer, row, interface):
    kwargs = {"tokenize": True, "add_generation_prompt": True}
    try:
        ids = tokenizer.apply_chat_template(messages(row, interface), enable_thinking=False, **kwargs)
    except (TypeError, ValueError):
        try:
            ids = tokenizer.apply_chat_template(messages(row, interface), **kwargs)
        except Exception:
            text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages(row, interface)) + "\nASSISTANT:"
            ids = tokenizer.encode(text, add_special_tokens=True)
    if isinstance(ids, Mapping):
        ids = ids["input_ids"]
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(x) for x in ids]


def semantic_hit(text, row):
    observed = " " + normalize(text) + " "
    intended = " " + normalize(row["intended"]) + " "
    alternative = " " + normalize(row["alternative"]) + " "
    intended_hit = intended in observed
    alternative_hit = alternative in observed
    exact = normalize(text) in {
        normalize(row["intended"]),
        normalize(f"The reachable target is {row['intended']}"),
    }
    return bool(intended_hit and not alternative_hit), bool(exact), bool(intended_hit), bool(alternative_hit)


def contract():
    if OUT.exists():
        raise RuntimeError(OUT)
    parent = core.load(C163 / "audit/independent_final_audit.json")
    cases = material()
    cells = {(r["unit"], r["target_factor"], r["path_factor"], r["surface_factor"]) for r in cases}
    checks = {
        "authorization": parent["all_checks_passed"],
        "cases": len(cases) == 96 and len(cells) == 96,
        "partitions": all(sum(r["partition"] == p for r in cases) == 32 for p in ("discovery", "confirmation", "fresh")),
        "relations": all(sum(r["relation_family"] == rel for r in cases) == 24 for rel in ("is_a", "part_of", "located_in", "precedes")),
        "factors": all(sum(r[k] == v for r in cases) == 48 for k in ("target_factor", "path_factor", "surface_factor") for v in (1, -1)),
        "semantic_unique": all(r["intended"] != r["alternative"] and r["semantic_edges"][-1][1] == r["intended"] for r in cases),
        "unique_prompts": len({r["question"] for r in cases}) == 96,
        "models": tuple(MODELS) == ("qwen3", "glm4", "deepseek7b"),
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    core.write_rows(OUT / "material/cases.jsonl", cases)
    protocol = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": now(),
        "status": "three_model_free_interface_contract_frozen",
        "models": list(MODELS),
        "sequential_loading": True,
        "precision": "BF16 nonquantized; one model at a time; GLM4/DeepSeek CPU-GPU offload permitted",
        "cases": 96,
        "interfaces": list(INTERFACES),
        "generation": {"do_sample": False, "max_new_tokens": MAX_NEW_TOKENS},
        "formal_partition": ["confirmation", "fresh"],
        "qualification": {"global_min": 0.70, "each_relation_min": 0.50, "each_path_min": 0.60, "each_surface_min": 0.60},
        "common_interface_rule": "same interface qualifies on at least two models",
        "parser": "semantic hit iff intended target appears and paired alternative does not appear in generated continuation",
        "naturalness": "hand-authored registry grammar plus deterministic machine uniqueness audit; independent human blind rating absent",
        "claim_boundary": "free natural-word output qualification; no HiddenState mechanism claim",
        "forbidden": ["attention", "MLP", "weights", "PCA", "cross-model token-id equality", "post-unblind threshold changes"],
        "source_hashes": {"C163": core.sha(C163 / "analysis/final.json")},
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "run_qwen3_then_glm4_then_deepseek7b",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": True})
    print(json.dumps({"checks": checks, "protocol": protocol}, indent=2))


def groups(rows, size):
    by_width = defaultdict(list)
    for row in rows:
        by_width[(row["interface"], len(row["input_ids"]) // 32)].append(row)
    for key in sorted(by_width):
        values = by_width[key]
        for start in range(0, len(values), size):
            yield values[start:start + size]


@torch.inference_mode()
def run_model(model_name):
    protocol = core.load(OUT / "protocol/preregistration.json")
    if (OUT / f"analysis/{model_name}.json").exists():
        raise RuntimeError(f"already run: {model_name}")
    cases = core.rows(OUT / "material/cases.jsonl")
    model = tokenizer = None
    started = time.time()
    try:
        model, tokenizer, device, placement = load_bf16(model_name)
        audit = quantization_audit(model)
        if audit["has_quantized_modules"] or not audit["has_bf16_parameters"]:
            raise RuntimeError(audit)
        info = get_model_info(model, model_name)
        tokenizer.padding_side = "left"
        rows = []
        for case in cases:
            for interface in INTERFACES:
                rows.append({**case, "interface": interface, "input_ids": render_ids(tokenizer, case, interface)})
        core.write_rows(OUT / f"compiled/{model_name}.jsonl", rows)
        results = []
        pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
        for batch_index, batch in enumerate(groups(rows, BATCH[model_name]), 1):
            width = max(len(r["input_ids"]) for r in batch)
            input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long, device=device)
            attention_mask = torch.zeros_like(input_ids)
            for i, row in enumerate(batch):
                values = torch.tensor(row["input_ids"], dtype=torch.long, device=device)
                input_ids[i, width - len(values):] = values
                attention_mask[i, width - len(values):] = 1
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                pad_token_id=pad_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            continuation = generated[:, width:].detach().cpu()
            for i, row in enumerate(batch):
                text = tokenizer.decode(continuation[i], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                hit, exact, intended_hit, alternative_hit = semantic_hit(text, row)
                results.append({
                    "case_id": row["case_id"], "unit": row["unit"], "partition": row["partition"],
                    "relation_family": row["relation_family"], "path_factor": row["path_factor"],
                    "surface_factor": row["surface_factor"], "interface": row["interface"],
                    "intended": row["intended"], "alternative": row["alternative"], "generated": text,
                    "semantic_hit": hit, "exact_hit": exact, "intended_hit": intended_hit, "alternative_hit": alternative_hit,
                })
            del input_ids, attention_mask, generated, continuation
            if batch_index % 16 == 0:
                print(f"[C164] {model_name} {len(results)}/{len(rows)} elapsed={time.time()-started:.1f}s", flush=True)
        core.write_rows(OUT / f"raw/{model_name}.jsonl", results)
        report = summarize_model(model_name, results)
        report.update({
            "placement": placement,
            "quantization_audit": audit,
            "model_info": {"layers": info.n_layers, "d_model": info.d_model, "class": info.model_class},
            "elapsed_seconds": time.time() - started,
        })
        core.save(OUT / f"analysis/{model_name}.json", report)
        core.save(OUT / f"audit/internal_{model_name}_audit.json", {
            "checks": {"rows": len(results) == 192, "finite": True, "bf16": audit["has_bf16_parameters"], "unquantized": not audit["has_quantized_modules"]},
            "all_checks_passed": len(results) == 192 and audit["has_bf16_parameters"] and not audit["has_quantized_modules"],
        })
        print(json.dumps(report, indent=2))
    finally:
        if model is not None:
            release_bf16(model)
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def accuracy(rows):
    return float(np.mean([r["semantic_hit"] for r in rows])) if rows else float("nan")


def summarize_model(model_name, rows):
    protocol = core.load(OUT / "protocol/preregistration.json")
    reports = {}
    for interface in INTERFACES:
        values = [r for r in rows if r["interface"] == interface]
        formal = [r for r in values if r["partition"] in protocol["formal_partition"]]
        report = {
            "discovery_accuracy": accuracy([r for r in values if r["partition"] == "discovery"]),
            "formal_accuracy": accuracy(formal),
            "formal_exact_accuracy": float(np.mean([r["exact_hit"] for r in formal])),
            "by_relation": {rel: accuracy([r for r in formal if r["relation_family"] == rel]) for rel in sorted({r["relation_family"] for r in formal})},
            "by_path": {str(path): accuracy([r for r in formal if r["path_factor"] == path]) for path in (1, -1)},
            "by_surface": {str(surface): accuracy([r for r in formal if r["surface_factor"] == surface]) for surface in (1, -1)},
        }
        q = protocol["qualification"]
        report["qualified"] = bool(
            report["formal_accuracy"] >= q["global_min"]
            and min(report["by_relation"].values()) >= q["each_relation_min"]
            and min(report["by_path"].values()) >= q["each_path_min"]
            and min(report["by_surface"].values()) >= q["each_surface_min"]
        )
        reports[interface] = report
    return {"phase": PHASE, "campaign": CAMPAIGN, "model": model_name, "status": "free_interface_adjudicated", "interfaces": reports}


def analyze():
    reports = {model: core.load(OUT / f"analysis/{model}.json") for model in MODELS}
    common = {
        interface: [model for model in MODELS if reports[model]["interfaces"][interface]["qualified"]]
        for interface in INTERFACES
    }
    qualified = [interface for interface, models in common.items() if len(models) >= 2]
    preferred = qualified[0] if qualified else None
    report = {
        "phase": PHASE, "campaign": CAMPAIGN, "status": "three_model_free_interface_adjudicated",
        "models": reports, "common_interface_models": common, "qualified_common_interfaces": qualified,
        "preferred_common_interface": preferred,
        "cross_model_eligibility": preferred is not None,
        "claim_boundary": "free natural-word behavior interface only; no coordinate or mechanism equivalence",
        "next_authorization": "C165 cross-model relative HiddenState topology" if preferred else "C165 typed_not_tested_then_C166",
    }
    core.save(OUT / "analysis/summary.json", report)
    checks = {"three_models": set(reports) == set(MODELS), "interfaces": all(set(r["interfaces"]) == set(INTERFACES) for r in reports.values()), "typed": isinstance(report["cross_model_eligibility"], bool)}
    core.save(OUT / "audit/internal_analysis_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps(report, indent=2))


def close():
    summary = core.load(OUT / "analysis/summary.json")
    checks = {
        "contract": core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"],
        "models": all(core.load(OUT / f"audit/internal_{m}_audit.json")["all_checks_passed"] for m in MODELS),
        "analysis": core.load(OUT / "audit/internal_analysis_audit.json")["all_checks_passed"],
    }
    final = {"phase": PHASE, "campaign": CAMPAIGN, "status": "closed", "checks": checks, "all_checks_passed": all(checks.values()), "headline": {"common": summary["common_interface_models"], "preferred": summary["preferred_common_interface"], "eligible": summary["cross_model_eligibility"]}, "next_authorization": summary["next_authorization"]}
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps(final, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("contract", "run", "analyze", "close"))
    parser.add_argument("--model", choices=MODELS)
    args = parser.parse_args()
    if args.command == "contract": contract()
    elif args.command == "run":
        if not args.model: raise SystemExit("--model required")
        run_model(args.model)
    elif args.command == "analyze": analyze()
    else: close()


if __name__ == "__main__":
    main()
