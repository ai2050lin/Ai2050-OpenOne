"""Phase996: lexical-depth confirmation and interface decomposition.

The stage is deliberately external-only.  It freezes a new lexical domain,
runs qwen3 -> glm4 -> deepseek7b in separate child processes, records natural
rollouts plus a fixed candidate diagnostic, and never requests hidden states.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import uuid
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import phase994_external_failure_localization_runner as base

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
AUDIT_SCRIPT = ROOT / "tests/glm5/phase996_external_semantic_confirmation_audit.py"
PROTOCOL_ROOT = ROOT / "tests/glm5/result/phase996_external_semantic_confirmation_protocol"
EXECUTION_ROOT = ROOT / "tests/glm5/result/phase996_external_semantic_confirmation_execution"
BASE_ACTIVATION = ROOT / "tests/glm5/result/phase994_external_failure_localization_protocol/activation.json"
PHASE = 996
EXPERIMENT = "external_semantic_selection_confirmation_and_interface_decomposition"
MODEL_ORDER = ("qwen3", "glm4", "deepseek7b")
DEPTHS = ("copy_control", "one_hop", "two_hop")
TRANSFORMS = ("original", "value_swap", "binding_swap", "query_swap")
PRIMARY_INTERFACES = ("raw_plain", "native_full")
DIAGNOSTIC_INTERFACES = (
    "raw_answer_scaffold", "native_role_only", "raw_plus_native_prefill",
)
INTERFACES = PRIMARY_INTERFACES + DIAGNOSTIC_INTERFACES
VALUES = ("amber", "silver", "violet", "ivory")
ENTITIES = ("Iris", "Jude", "Kian", "Luma", "Nori", "Oren", "Pia", "Rafi")
OBJECTS = ("prism", "goblet", "flask", "amulet")
RELATION_WORDS = ("CARRY", "BEAR")
SEED = 0x0000_0000_03E4_0080
WORLDS = 256
DECOMPOSITION_WORLDS = 64
BATCH_SIZE = 8
SCAFFOLD = "The retrieved marker is"
PUBLIC_ROWS = WORLDS * 4 * 3 * 2 + DECOMPOSITION_WORLDS * 4 * 3 * 3
ENGINEERING_ROWS = 40
THRESHOLDS = {
    "split_transform_correct_min_of_128": 116,
    "transform_correct_min_of_256": 232,
    "all_four_worlds_min_of_256": 218,
}
HISTORY_FILES = (
    ROOT / "tests/glm5/result/phase990_delayed_binding_protocol/dataset.json",
    ROOT / "tests/glm5/result/phase991_delayed_binding_gpu_admission/extension_dataset.json",
    ROOT / "tests/glm5/result/phase994_external_failure_localization_protocol/dataset/public_manifest.jsonl",
)


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_json(value: object) -> str:
    return sha_bytes(canonical(value))


def sha_file(path: Path) -> str:
    return base.sha256_file(path)


def sealed(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    out = deepcopy(dict(value))
    require(field not in out, f"self hash already exists: {field}")
    out[field] = sha_json(out)
    return out


def verify_sealed(value: Mapping[str, Any], field: str, label: str) -> None:
    expected = value.get(field)
    body = {key: item for key, item in value.items() if key != field}
    require(isinstance(expected, str) and expected == sha_json(body), f"{label} self hash drift")


def file_seal(path: Path, root: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {"path": resolved.relative_to(root.resolve()).as_posix(), "bytes": resolved.stat().st_size,
            "sha256": sha_file(resolved)}


def write_exclusive(path: Path, payload: bytes) -> None:
    base.write_exclusive(path, payload)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    payload = bytearray()
    for row in rows:
        payload.extend(canonical(row) + b"\n")
    write_exclusive(path, bytes(payload))


def prompt_for(entities: Sequence[str], e2o: Mapping[str, str], o2v: Mapping[str, str],
               start: str, commands: Sequence[str], order: Sequence[str]) -> str:
    entity_lines = [f"{entity} CARRY {e2o[entity]}." for entity in order]
    object_order = [e2o[entity] for entity in order]
    value_lines = [f"{obj} BEAR {o2v[obj]}." for obj in object_order]
    return "\n".join([
        "Registry rules: CARRY maps a person to one item; BEAR maps an item to one marker; KEEP leaves a term unchanged.",
        "Registry entries:", *entity_lines, *value_lines,
        f"Start term: {start}.", f"Program: {commands[0]} then {commands[1]}.",
        "Execute the two commands from left to right using only the registry.",
        f'Reply with one short sentence beginning "{SCAFFOLD}".',
    ])


def build_worlds() -> list[dict[str, Any]]:
    rng = random.Random(SEED)
    worlds: list[dict[str, Any]] = []
    identities: set[str] = set()
    attempts = 0
    while len(worlds) < WORLDS:
        attempts += 1
        chosen = rng.sample(list(ENTITIES), 4)
        objects = list(OBJECTS); rng.shuffle(objects)
        values = list(VALUES); rng.shuffle(values)
        e2o = dict(zip(chosen, objects, strict=True))
        o2v = dict(zip(objects, values, strict=True))
        order = list(chosen); rng.shuffle(order)
        identity = sha_json({"e2o": e2o, "o2v": o2v, "order": order})
        if identity in identities:
            continue
        identities.add(identity)
        ordinal = len(worlds)
        desired = [VALUES[(ordinal + offset) % 4] for offset in range(4)]
        value_to_object = {value: obj for obj, value in o2v.items()}
        object_to_entity = {obj: entity for entity, obj in e2o.items()}
        query = object_to_entity[value_to_object[desired[0]]]
        variants: dict[str, dict[str, Any]] = {}
        for index, transform in enumerate(TRANSFORMS):
            ve2o = dict(e2o); vo2v = dict(o2v); vquery = query
            target = desired[index]
            if transform == "value_swap":
                qobj = ve2o[vquery]; other = value_to_object[target]
                vo2v[qobj], vo2v[other] = vo2v[other], vo2v[qobj]
            elif transform == "binding_swap":
                qobj = ve2o[vquery]; other = value_to_object[target]
                other_entity = object_to_entity[other]
                ve2o[vquery], ve2o[other_entity] = ve2o[other_entity], ve2o[vquery]
            elif transform == "query_swap":
                vquery = object_to_entity[value_to_object[target]]
            gold_obj = ve2o[vquery]; gold = vo2v[gold_obj]
            require(gold == target, "transform target construction failed")
            variants[transform] = {"e2o": ve2o, "o2v": vo2v, "query": vquery,
                                   "gold_object": gold_obj, "gold": gold}
        worlds.append({"ordinal": ordinal, "identity": identity, "entities": chosen,
                       "order": order, "variants": variants})
    require(attempts < WORLDS * 4, "world uniqueness search was unexpectedly dense")
    return worlds


def build_dataset() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    public: list[dict[str, Any]] = []
    truth: list[dict[str, Any]] = []
    worlds = build_worlds()
    gold_counts: dict[str, dict[str, int]] = {t: {v: 0 for v in VALUES} for t in TRANSFORMS}
    for world in worlds:
        ordinal = int(world["ordinal"])
        split = "confirmation_a" if ordinal < 128 else "confirmation_b"
        interfaces = list(PRIMARY_INTERFACES)
        if ordinal % 4 == 0:
            interfaces += list(DIAGNOSTIC_INTERFACES)
        for transform in TRANSFORMS:
            variant = world["variants"][transform]
            gold = str(variant["gold"]); gold_counts[transform][gold] += 1
            for depth in DEPTHS:
                if depth == "copy_control": start, commands = gold, ("KEEP", "KEEP")
                elif depth == "one_hop": start, commands = variant["gold_object"], ("KEEP", "BEAR")
                else: start, commands = variant["query"], ("CARRY", "BEAR")
                prompt = prompt_for(world["entities"], variant["e2o"], variant["o2v"],
                                    str(start), commands, world["order"])
                for interface in interfaces:
                    rid = "p996-" + sha_json([world["identity"], transform, depth, interface])[:24]
                    public.append({
                        "schema_version": "phase996_public_manifest.v1", "phase": PHASE,
                        "record_id": rid, "semantic_world_id": "p996w-" + world["identity"][:24],
                        "world_ordinal": ordinal, "split": split, "semantic_transform": transform,
                        "depth": depth, "interface_variant": interface, "prompt": prompt,
                        "prompt_sha256": sha_bytes(prompt.encode("utf-8")),
                        "primary_replication": interface in PRIMARY_INTERFACES,
                    })
                    truth.append({"record_id": rid, "gold": gold,
                                  "gold_object": variant["gold_object"], "query_entity": variant["query"]})
    require(len(public) == PUBLIC_ROWS and len(truth) == PUBLIC_ROWS, "public row count drift")
    require(len({r["record_id"] for r in public}) == PUBLIC_ROWS, "record IDs duplicate")
    require(all(set(counts.values()) == {64} for counts in gold_counts.values()), "gold labels not exactly balanced")
    audit = {"world_count": WORLDS, "public_rows": len(public), "decomposition_world_count": DECOMPOSITION_WORLDS,
             "gold_counts_by_transform": gold_counts, "split_world_counts": {"confirmation_a": 128, "confirmation_b": 128},
             "primary_rows": WORLDS * 4 * 3 * 2, "diagnostic_rows": DECOMPOSITION_WORLDS * 4 * 3 * 3}
    return public, truth, audit


def lexical_history_audit() -> dict[str, Any]:
    terms = tuple(word.lower() for word in (*ENTITIES, *OBJECTS, *VALUES, *RELATION_WORDS))
    hits: dict[str, list[str]] = {term: [] for term in terms}
    file_seals: list[dict[str, Any]] = []
    for path in HISTORY_FILES:
        require(path.is_file(), f"history file missing: {path}")
        file_seals.append(file_seal(path, ROOT))
        lowered = path.read_bytes().lower()
        for term in terms:
            if re.search(rb"(?<![a-z])" + re.escape(term.encode()) + rb"(?![a-z])", lowered):
                hits[term].append(path.relative_to(ROOT).as_posix())
    require(not any(hits.values()), f"new lexical item overlaps historical corpus: {hits}")
    return {"terms": list(terms), "hits": hits, "zero_lexical_overlap": True, "history_file_seals": file_seals}


def native_suffix(tokenizer: Any) -> tuple[str, list[int]]:
    probe = "PHASE996_NATIVE_PREFILL_PROBE"
    messages = [{"role": "user", "content": probe}]
    without = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    with_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    require(isinstance(without, str) and isinstance(with_prompt, str) and with_prompt.startswith(without),
            "native generation prompt is not a textual suffix")
    suffix = with_prompt[len(without):]
    a = list(tokenizer(without, add_special_tokens=False).input_ids)
    b = list(tokenizer(with_prompt, add_special_tokens=False).input_ids)
    require(b[:len(a)] == a and len(b) > len(a), "native generation prompt is not a token suffix")
    return suffix, [int(x) for x in b[len(a):]]


def render(tokenizer: Any, prompt: str, interface: str) -> tuple[str, list[int]]:
    def ids(text: str) -> list[int]:
        return [int(x) for x in tokenizer(text, add_special_tokens=False, return_attention_mask=False).input_ids]
    if interface == "raw_plain": text = prompt
    elif interface == "raw_answer_scaffold": text = prompt.rstrip() + "\n" + SCAFFOLD
    elif interface == "native_role_only":
        text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                             add_generation_prompt=False)
    elif interface == "native_full":
        text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                             add_generation_prompt=True)
    elif interface == "raw_plus_native_prefill":
        suffix, suffix_ids = native_suffix(tokenizer); raw_ids = ids(prompt)
        text = prompt + suffix
        require(ids(text) == raw_ids + suffix_ids, "raw plus native prefill boundary drift")
    else: raise RuntimeError(f"unknown interface: {interface}")
    out = ids(text); require(out, "empty rendered input")
    return text, out


def tokenizer_precheck(public: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    from transformers import AutoTokenizer
    base_activation = base.verify_activation()
    reports: dict[str, Any] = {}
    model_paths = {m: base.p992_runner.verify_model_artifacts(base_activation, m)["resolved_root"] for m in MODEL_ORDER}
    for model in MODEL_ORDER:
        tokenizer = AutoTokenizer.from_pretrained(model_paths[model], local_files_only=True, trust_remote_code=False)
        candidate_ids = {}
        for value in VALUES:
            value_ids = list(tokenizer(" " + value, add_special_tokens=False).input_ids)
            require(len(value_ids) == 1, f"candidate value is not one token for {model}: {value}")
            candidate_ids[value] = int(value_ids[0])
        minimum, maximum = 10**9, 0
        variant_counts: dict[str, int] = defaultdict(int)
        for row in public:
            _, row_ids = render(tokenizer, str(row["prompt"]), str(row["interface_variant"]))
            minimum = min(minimum, len(row_ids)); maximum = max(maximum, len(row_ids))
            variant_counts[str(row["interface_variant"])] += 1
        require(maximum < 1024, f"input too long for {model}")
        reports[model] = {"candidate_token_ids": candidate_ids, "input_token_min": minimum,
                          "input_token_max": maximum, "variant_counts": dict(sorted(variant_counts.items())),
                          "native_prefill": {"text": native_suffix(tokenizer)[0], "token_ids": native_suffix(tokenizer)[1]}}
        del tokenizer
    return reports


def freeze() -> dict[str, Any]:
    require(not PROTOCOL_ROOT.exists() and not EXECUTION_ROOT.exists(), "Phase996 output already exists")
    require(AUDIT_SCRIPT.is_file(), "independent audit source is missing")
    base_activation = base.verify_activation()
    public, truth, dataset_audit = build_dataset()
    lexical_audit = lexical_history_audit()
    precheck = tokenizer_precheck(public)
    PROTOCOL_ROOT.mkdir(parents=True)
    write_jsonl(PROTOCOL_ROOT / "dataset/public_manifest.jsonl", public)
    write_jsonl(PROTOCOL_ROOT / "dataset/private_truth.jsonl", truth)
    engineering: list[dict[str, Any]] = []
    for interface in INTERFACES:
        engineering.extend([row for row in public if row["interface_variant"] == interface][:8])
    require(len(engineering) == ENGINEERING_ROWS, "engineering manifest count drift")
    write_jsonl(PROTOCOL_ROOT / "dataset/engineering_manifest.jsonl", engineering)
    write_exclusive(PROTOCOL_ROOT / "dataset/dataset_audit.json", canonical(dataset_audit))
    write_exclusive(PROTOCOL_ROOT / "dataset/lexical_history_audit.json", canonical(lexical_audit))
    write_exclusive(PROTOCOL_ROOT / "dataset/tokenizer_precheck.json", canonical(precheck))
    dataset_seals = {p.name: file_seal(p, PROTOCOL_ROOT) for p in (PROTOCOL_ROOT / "dataset").iterdir()}
    protocol = sealed({
        "schema_version": "phase996_protocol.v1", "phase": PHASE, "experiment": EXPERIMENT,
        "created_at_utc": now(), "seed_uint64": SEED, "world_count": WORLDS,
        "decomposition_world_count": DECOMPOSITION_WORLDS, "public_row_count": PUBLIC_ROWS,
        "model_order": list(MODEL_ORDER), "depths": list(DEPTHS), "transforms": list(TRANSFORMS),
        "primary_interfaces": list(PRIMARY_INTERFACES), "diagnostic_interfaces": list(DIAGNOSTIC_INTERFACES),
        "values": list(VALUES), "thresholds": THRESHOLDS, "nested_budgets": [64, 128],
        "max_128_policy": "native_full only for qwen3 and deepseek7b; all other cells 64",
        "natural_rollout_is_primary": True, "candidate_diagnostic_is_non_substitutive": True,
        "internal_trace_authorized": False, "phase992_holdout_authorized": False,
        "base_phase994_activation_sha256": base_activation["activation_sha256"], "dataset_seals": dataset_seals,
    }, "protocol_sha256")
    write_exclusive(PROTOCOL_ROOT / "protocol.json", canonical(protocol))
    sources = {
        "phase996_runner": file_seal(SCRIPT, ROOT), "phase996_independent_audit": file_seal(AUDIT_SCRIPT, ROOT),
        "phase994_runner_base": file_seal(Path(base.__file__), ROOT),
        "phase983_engine": file_seal(Path(base.engine.__file__), ROOT),
    }
    activation = sealed({
        "schema_version": "phase996_activation.v1", "phase": PHASE, "experiment": EXPERIMENT,
        "created_at_utc": now(), "gpu_execution_authorized": True, "model_order": list(MODEL_ORDER),
        "formal_python": str(Path(sys.executable).resolve()), "runtime_identity": base.runtime_identity(),
        "batch_size": BATCH_SIZE, "public_row_count": PUBLIC_ROWS, "engineering_row_count": ENGINEERING_ROWS,
        "base_phase994_activation_sha256": base_activation["activation_sha256"],
        "protocol": file_seal(PROTOCOL_ROOT / "protocol.json", PROTOCOL_ROOT),
        "protocol_self_sha256": protocol["protocol_sha256"], "dataset_seals": dataset_seals,
        "source_seals": sources, "thresholds": THRESHOLDS, "internal_trace_authorized": False,
        "phase992_holdout_authorized": False,
    }, "activation_sha256")
    write_exclusive(PROTOCOL_ROOT / "activation.json", canonical(activation))
    return {"passed": True, "activation_sha256": activation["activation_sha256"],
            "protocol_sha256": protocol["protocol_sha256"], "public_rows": PUBLIC_ROWS}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8")); require(isinstance(value, dict), f"not object: {path}")
    return value


def verify_activation() -> dict[str, Any]:
    activation = load_json(PROTOCOL_ROOT / "activation.json")
    verify_sealed(activation, "activation_sha256", "activation")
    require(activation["phase"] == PHASE and activation["gpu_execution_authorized"] is True, "execution unauthorized")
    require(Path(sys.executable).resolve() == Path(activation["formal_python"]).resolve(), "formal Python drift")
    require(base.runtime_identity() == activation["runtime_identity"], "runtime identity drift")
    old = base.verify_activation()
    require(old["activation_sha256"] == activation["base_phase994_activation_sha256"], "base activation drift")
    for role, seal in activation["source_seals"].items():
        path = ROOT / seal["path"]
        require(path.is_file() and path.stat().st_size == seal["bytes"] and sha_file(path) == seal["sha256"],
                f"source drift: {role}")
    for name, seal in activation["dataset_seals"].items():
        path = PROTOCOL_ROOT / seal["path"]
        require(path.is_file() and path.stat().st_size == seal["bytes"] and sha_file(path) == seal["sha256"],
                f"dataset drift: {name}")
    return activation


def read_manifest(path: Path, expected: int) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    require(len(rows) == expected and len({r["record_id"] for r in rows}) == expected, "manifest count/ID drift")
    for row in rows:
        require(row["phase"] == PHASE and row["interface_variant"] in INTERFACES and row["depth"] in DEPTHS,
                "manifest factor drift")
        require(row["prompt_sha256"] == sha_bytes(row["prompt"].encode()), "prompt hash drift")
        require(not ({"gold", "gold_object", "target"} & set(row)), "truth leaked to runner")
    return rows


def left_pad(torch: Any, sequences: Sequence[Sequence[int]], pad: int, device: Any) -> tuple[Any, Any]:
    width = max(map(len, sequences)); ids = torch.full((len(sequences), width), pad, dtype=torch.long, device=device)
    mask = torch.zeros((len(sequences), width), dtype=torch.long, device=device)
    for index, row in enumerate(sequences):
        data = torch.tensor(row, dtype=torch.long, device=device); ids[index, -len(row):] = data; mask[index, -len(row):] = 1
    return ids, mask


def generation_budget(model: str, interface: str) -> int:
    return 128 if interface == "native_full" and model in ("qwen3", "deepseek7b") else 64


def rows_for_model(adapter: Any, torch: Any, rows: Sequence[Mapping[str, Any]], scope: str,
                   run_id: str, activation: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
    tokenizer = adapter.tokenizer; tokenizer.padding_side = "left"
    eos_ids = sorted(int(x) for x in adapter.eos_identity["effective_eos_token_ids"])
    pad = int(adapter.pad_token_id)
    interface_index = {name: i for i, name in enumerate(INTERFACES)}
    depth_index = {name: i for i, name in enumerate(DEPTHS)}
    transform_index = {name: i for i, name in enumerate(TRANSFORMS)}
    ordered = sorted(rows, key=lambda r: (interface_index[r["interface_variant"]], depth_index[r["depth"]],
                                          transform_index[r["semantic_transform"]], r["world_ordinal"]))
    for start in range(0, len(ordered), BATCH_SIZE):
        batch = ordered[start:start+BATCH_SIZE]; require(len(batch) == BATCH_SIZE, "partial formal batch")
        require(len({r["interface_variant"] for r in batch}) == 1, "batch crossed interface")
        interface = str(batch[0]["interface_variant"]); budget = generation_budget(adapter.model_key, interface)
        rendered = [render(tokenizer, str(row["prompt"]), interface) for row in batch]
        raw_ids = [item[1] for item in rendered]
        input_ids, attention = left_pad(torch, raw_ids, pad, adapter.input_device)
        candidate_contexts: list[list[int]] = []
        candidate_ids: dict[str, int] = {}
        for value in VALUES:
            value_ids = list(tokenizer(" " + value, add_special_tokens=False).input_ids)
            require(len(value_ids) == 1, "candidate token identity drift"); candidate_ids[value] = int(value_ids[0])
        for text, _ in rendered:
            context = text if text.rstrip().endswith(SCAFFOLD) else text.rstrip() + "\n" + SCAFFOLD
            context_ids = list(tokenizer(context, add_special_tokens=False).input_ids)
            for value, token_id in candidate_ids.items():
                require(list(tokenizer(context + " " + value, add_special_tokens=False).input_ids) == context_ids + [token_id],
                        f"candidate boundary drift: {adapter.model_key}/{interface}/{value}")
            candidate_contexts.append([int(x) for x in context_ids])
        cand_input, cand_mask = left_pad(torch, candidate_contexts, pad, adapter.input_device)
        with torch.inference_mode():
            cand_out = adapter.model(input_ids=cand_input, attention_mask=cand_mask, use_cache=False)
            last_logits = cand_out.logits[:, -1, :]
            candidate_logits = [{value: float(last_logits[i, token].detach().cpu()) for value, token in candidate_ids.items()}
                                for i in range(len(batch))]
            generated = adapter.model.generate(
                input_ids=input_ids, attention_mask=attention, do_sample=False, num_beams=1,
                num_return_sequences=1, use_cache=True, max_new_tokens=budget, pad_token_id=pad,
                eos_token_id=eos_ids, return_dict_in_generate=True, output_scores=False,
                output_attentions=False, output_hidden_states=False,
            )
        suffix_tensor = generated.sequences[:, input_ids.shape[1]:]
        suffixes = [[int(x) for x in row] for row in suffix_tensor.detach().cpu().tolist()]
        del generated, suffix_tensor, input_ids, attention, cand_input, cand_mask, cand_out, last_logits
        for position, (row, item, suffix, logits) in enumerate(zip(batch, rendered, suffixes, candidate_logits, strict=True)):
            first_eos = next((i for i, token in enumerate(suffix) if token in eos_ids), None)
            before = suffix if first_eos is None else suffix[:first_eos]
            text = tokenizer.decode(before, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            yield {
                "schema_version": "phase996_external_raw.v1", "phase": PHASE, "experiment": EXPERIMENT,
                "scope": scope, "model": adapter.model_key, "run_id": run_id, "record_id": row["record_id"],
                "semantic_world_id": row["semantic_world_id"], "world_ordinal": row["world_ordinal"],
                "split": row["split"], "semantic_transform": row["semantic_transform"], "depth": row["depth"],
                "interface_variant": interface, "primary_replication": row["primary_replication"],
                "prompt_sha256": row["prompt_sha256"], "rendered_prompt_sha256": sha_bytes(item[0].encode()),
                "input_token_ids": item[1], "input_token_ids_sha256": sha_json(item[1]),
                "input_token_count": len(item[1]), "batch_position": position, "max_new_tokens": budget,
                "generated_suffix_token_ids": suffix, "generated_token_ids_before_eos": before,
                "generated_text": text, "effective_eos_token_ids": eos_ids, "first_eos_index": first_eos,
                "eos_seen": first_eos is not None, "budget_exhausted": first_eos is None,
                "candidate_context_token_ids_sha256": sha_json(candidate_contexts[position]),
                "candidate_token_ids": candidate_ids, "candidate_logits": logits,
                "activation_sha256": activation["activation_sha256"],
            }


def gzip_rows(path: Path, rows: Iterable[Mapping[str, Any]]) -> tuple[int, str]:
    return base.gzip_rows_exclusive(path, rows)


def worker(scope: str, model: str, manifest: Path, raw_path: Path, status_path: Path, run_id: str) -> dict[str, Any]:
    activation = verify_activation(); expected = ENGINEERING_ROWS if scope == "engineering" else PUBLIC_ROWS
    rows = read_manifest(manifest, expected)
    old = base.verify_activation(); artifact = base.p992_runner.verify_model_artifacts(old, model)
    import torch
    adapter = None; loaded = None; cleanup = None; canonical_sha = None
    try:
        adapter = base.engine.load_model_adapter(model); base.p992_runner.validate_loaded_identity(adapter.identity, model)
        loaded = deepcopy(adapter.identity)
        if scope == "engineering":
            first = list(rows_for_model(adapter, torch, rows, scope, run_id, activation))
            second = list(rows_for_model(adapter, torch, rows, scope, run_id, activation))
            require(canonical(first) == canonical(second), "engineering repeat mismatch")
            count, canonical_sha = gzip_rows(raw_path, first)
        else:
            count, canonical_sha = gzip_rows(raw_path, rows_for_model(adapter, torch, rows, scope, run_id, activation))
        require(count == expected, "worker output row count drift")
    finally:
        cleanup = base.p992_runner.strict_cuda_release(base.engine, adapter, torch); adapter = None
    require(cleanup and cleanup["cleanup_pass"] is True and loaded is not None, "CUDA cleanup/model load failure")
    raw_seal = file_seal(raw_path, EXECUTION_ROOT)
    status = sealed({"schema_version": "phase996_worker_status.v1", "phase": PHASE, "created_at_utc": now(),
                     "scope": scope, "model": model, "run_id": run_id, "status": "success",
                     "activation_sha256": activation["activation_sha256"], "row_count": expected,
                     "raw_artifact": raw_seal, "raw_canonical_lines_sha256": canonical_sha,
                     "model_artifact_verification": artifact, "loaded_model_identity": loaded,
                     "strict_cuda_release": cleanup, "internal_trace_authorized": False}, "status_sha256")
    write_exclusive(status_path, canonical(status))
    return {"passed": True, "model": model, "scope": scope, "status_sha256": status["status_sha256"]}


def copy_manifest(scope: str) -> Path:
    source = PROTOCOL_ROOT / f"dataset/{scope}_manifest.jsonl"
    target = EXECUTION_ROOT / f"manifests/{scope}_manifest.jsonl"
    write_exclusive(target, source.read_bytes()); return target


def child(activation: Mapping[str, Any], scope: str, model: str, manifest: Path, run_id: str) -> dict[str, Any]:
    raw = EXECUTION_ROOT / f"raw/{scope}/{model}.jsonl.gz"
    status = EXECUTION_ROOT / f"worker_status/{scope}/{model}.json"
    command = [activation["formal_python"], "-B", str(SCRIPT), "--worker", "--scope", scope, "--model", model,
               "--manifest", str(manifest), "--raw-output", str(raw), "--status-output", str(status), "--run-id", run_id]
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "TOKENIZERS_PARALLELISM": "false", "PYTHONDONTWRITEBYTECODE": "1"}
    completed = subprocess.run(command, capture_output=True, text=True, env=env, check=False)
    require(completed.returncode == 0, f"{scope}/{model} failed: {completed.stderr[-4000:]}")
    report = json.loads(completed.stdout); require(report.get("passed") is True, "worker report failed")
    value = load_json(status); verify_sealed(value, "status_sha256", f"{scope}/{model} status")
    require(value["strict_cuda_release"]["allocated_after_release"] == 0 and
            value["strict_cuda_release"]["reserved_after_release"] == 0, "CUDA allocator not empty")
    return {"status_sha256": value["status_sha256"], "raw": value["raw_artifact"], "cleanup_pass": True}


def run_scope(scope: str) -> dict[str, Any]:
    activation = verify_activation(); require(scope in ("engineering", "public"), "bad scope")
    if scope == "engineering":
        require(not EXECUTION_ROOT.exists(), "execution root already exists"); EXECUTION_ROOT.mkdir(parents=True)
    else:
        require((EXECUTION_ROOT / "engineering_stage.json").is_file(), "engineering gate missing")
        require(not (EXECUTION_ROOT / "public_stage.json").exists(), "public stage already exists")
    manifest = copy_manifest(scope); run_id = f"phase996-{scope}-{uuid.uuid4().hex}"
    reports: dict[str, Any] = {}
    for model in MODEL_ORDER:
        reports[model] = child(activation, scope, model, manifest, run_id)
    stage = sealed({"schema_version": "phase996_stage.v1", "phase": PHASE, "created_at_utc": now(),
                    "scope": scope, "run_id": run_id, "passed": True, "model_order": list(MODEL_ORDER),
                    "models": reports, "all_models_serial": True, "truth_opened": False,
                    "internal_trace_authorized": False}, "stage_sha256")
    write_exclusive(EXECUTION_ROOT / f"{scope}_stage.json", canonical(stage)); return stage


VALUE_RE = re.compile(r"(?<![A-Za-z])(amber|silver|violet|ivory)(?![A-Za-z])", re.I)
SCAFFOLD_RE = re.compile(r"The\s+retrieved\s+marker\s+is\s+(amber|silver|violet|ivory)\s*\.", re.I)
STRICT_RE = re.compile(r"^\s*The\s+retrieved\s+marker\s+is\s+(amber|silver|violet|ivory)\s*\.\s*$", re.I)


def parse(text: str) -> dict[str, Any]:
    full = list(SCAFFOLD_RE.finditer(text))
    if full: value = full[-1].group(1).lower(); method = "last_complete_scaffold"
    else:
        values = sorted({m.group(1).lower() for m in VALUE_RE.finditer(text)})
        value = values[0] if len(values) == 1 else None; method = "unique_marker_fallback" if value else "unparsed"
    return {"value": value, "method": method, "strict": bool(STRICT_RE.fullmatch(text))}


def raw_rows(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def gate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    split_transform = {f"{s}|{t}": sum(r["correct"] for r in rows if r["split"] == s and r["transform"] == t)
                       for s in ("confirmation_a", "confirmation_b") for t in TRANSFORMS}
    transform = {t: sum(r["correct"] for r in rows if r["transform"] == t) for t in TRANSFORMS}
    grouped: dict[str, list[bool]] = defaultdict(list)
    for row in rows: grouped[row["world"]].append(bool(row["correct"]))
    all_four = sum(len(v) == 4 and all(v) for v in grouped.values())
    passed = min(split_transform.values()) >= 116 and min(transform.values()) >= 232 and all_four >= 218
    return {"passed": passed, "split_transform_correct": split_transform, "transform_correct": transform,
            "all_four_worlds": all_four, "world_denominator": len(grouped)}


def score() -> dict[str, Any]:
    activation = verify_activation(); stage = load_json(EXECUTION_ROOT / "public_stage.json")
    verify_sealed(stage, "stage_sha256", "public stage"); require(stage["passed"] is True, "public stage failed")
    truth_path = PROTOCOL_ROOT / "dataset/private_truth.jsonl"
    truth = {row["record_id"]: row for row in (json.loads(line) for line in truth_path.read_text(encoding="utf-8").splitlines())}
    access = sealed({"schema_version": "phase996_truth_access.v1", "phase": PHASE, "created_at_utc": now(),
                     "public_stage_sha256": stage["stage_sha256"], "truth_sha256": sha_file(truth_path),
                     "first_access_after_all_raw_sealed": True}, "access_sha256")
    write_exclusive(EXECUTION_ROOT / "scores/truth_access_receipt.json", canonical(access))
    summaries: dict[str, Any] = {}; gates: dict[str, Any] = {}; gradients: dict[str, Any] = {}
    for model in MODEL_ORDER:
        cases: list[dict[str, Any]] = []
        tokenizer = base.engine.load_tokenizer_inspection(model).tokenizer
        rows = raw_rows(EXECUTION_ROOT / f"raw/public/{model}.jsonl.gz")
        require(len(rows) == PUBLIC_ROWS, f"raw row count drift: {model}")
        for row in rows:
            gold = truth[row["record_id"]]["gold"]
            before = list(row["generated_token_ids_before_eos"])
            budgets = [64] + ([128] if row["max_new_tokens"] == 128 else [])
            for budget in budgets:
                tokens = before[:budget]
                text = tokenizer.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                parsed = parse(text); logits = row["candidate_logits"]
                pred_candidate = max(VALUES, key=lambda value: logits[value])
                margin = float(logits[gold] - max(logits[v] for v in VALUES if v != gold))
                cases.append({"record": row["record_id"], "world": row["semantic_world_id"], "split": row["split"],
                              "transform": row["semantic_transform"], "depth": row["depth"],
                              "interface": row["interface_variant"], "budget": budget, "gold": gold,
                              "parsed": parsed["value"], "correct": parsed["value"] == gold,
                              "strict": parsed["strict"], "eos": row["eos_seen"] and row["first_eos_index"] < budget,
                              "candidate_correct": pred_candidate == gold, "candidate_margin": margin})
        model_summary: dict[str, Any] = {}
        for interface in INTERFACES:
            for depth in DEPTHS:
                for budget in (64, 128):
                    cell = [r for r in cases if r["interface"] == interface and r["depth"] == depth and r["budget"] == budget]
                    if not cell: continue
                    key = f"{interface}|{depth}|{budget}"
                    model_summary[key] = {"n": len(cell), "correct": sum(r["correct"] for r in cell),
                                          "parsed": sum(r["parsed"] is not None for r in cell),
                                          "strict": sum(r["strict"] for r in cell), "eos": sum(r["eos"] for r in cell),
                                          "candidate_correct": sum(r["candidate_correct"] for r in cell),
                                          "candidate_margin_mean": sum(r["candidate_margin"] for r in cell) / len(cell)}
                    if interface in PRIMARY_INTERFACES and budget in (64, 128) and len(cell) == 1024:
                        gates[f"{model}|{key}"] = gate(cell)
        summaries[model] = model_summary
        for interface in PRIMARY_INTERFACES:
            for budget in (64, 128):
                counts = []
                for depth in DEPTHS:
                    key = f"{interface}|{depth}|{budget}"
                    if key not in model_summary: break
                    counts.append(model_summary[key]["correct"])
                if len(counts) == 3:
                    gradients[f"{model}|{interface}|{budget}"] = {"counts": counts,
                        "copy_ge_one_ge_two": counts[0] >= counts[1] >= counts[2]}
        del cases, tokenizer
    output = sealed({"schema_version": "phase996_score.v1", "phase": PHASE, "created_at_utc": now(),
                     "activation_sha256": activation["activation_sha256"], "public_stage_sha256": stage["stage_sha256"],
                     "truth_access_sha256": access["access_sha256"], "summaries": summaries, "gates": gates,
                     "depth_gradients": gradients, "two_hop_passes": sorted(k for k, v in gates.items()
                                                                      if "|two_hop|" in k and v["passed"]),
                     "internal_observation_authorized": False}, "score_sha256")
    write_exclusive(EXECUTION_ROOT / "scores/public_score.json", canonical(output)); return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); mode = parser.add_mutually_exclusive_group(required=True)
    for name in ("freeze", "engineering", "public", "score", "worker"):
        mode.add_argument("--" + name, action="store_true")
    parser.add_argument("--scope"); parser.add_argument("--model"); parser.add_argument("--manifest", type=Path)
    parser.add_argument("--raw-output", type=Path); parser.add_argument("--status-output", type=Path); parser.add_argument("--run-id")
    args = parser.parse_args(argv)
    if args.freeze: result = freeze()
    elif args.engineering: result = run_scope("engineering")
    elif args.public: result = run_scope("public")
    elif args.score: result = score()
    else: result = worker(args.scope, args.model, args.manifest, args.raw_output, args.status_output, args.run_id)
    print(json.dumps(result, sort_keys=True, ensure_ascii=False)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
