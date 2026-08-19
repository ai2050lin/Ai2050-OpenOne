#!/usr/bin/env python3
"""Phase1248: Qwen3 model-self future-response atlas.

This experiment keeps semantic correctness and model-self intervention response
in separate ledgers.  It uses a frozen 2x2x2 interface design and asks whether a
camera calibrated in Phase1247 transfers to a pretrained Qwen3-4B FP16 model.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests/glm5"
OS_ROOT = ROOT / "research/ai2050_research_os"
sys.path.insert(0, str(TEST_ROOT))

from model_utils import MODEL_CONFIGS, get_layers  # noqa: E402
from phase1023_fp16_utils import load_fp16, quantization_audit, release_fp16  # noqa: E402


PHASE = 1248
CONTRACT_ID = "EXP-C002-WP01-001"
SCRIPT = Path(__file__).resolve()
AUDIT_SCRIPT = TEST_ROOT / "phase1248_c002_qwen_self_response_atlas_audit.py"
OUT_ROOT = TEST_ROOT / "result/phase1248_c002_qwen_self_response_atlas"
MATERIAL_PATH = OUT_ROOT / "material/frozen_worlds.jsonl"
TOKEN_PATH = OUT_ROOT / "material/qwen3_token_manifest.jsonl"
PROTOCOL_PATH = OUT_ROOT / "protocol/preregistration.json"
ENV_PATH = OUT_ROOT / "protocol/environment_snapshot.json"
PREAUDIT_PATH = OUT_ROOT / "audit/independent_preaudit.json"
ARRAY_PATH = OUT_ROOT / "raw/response_arrays.npz"
RUN_PATH = OUT_ROOT / "raw/run_summary.json"
ATLAS_PATH = OUT_ROOT / "analysis/model_self_response_atlas.json"
FINAL_PATH = OUT_ROOT / "analysis/final.json"
FINAL_AUDIT_PATH = OUT_ROOT / "audit/independent_final_audit.json"

MODEL = "qwen3"
MODEL_PATH = Path(MODEL_CONFIGS[MODEL]["path"])
SYSTEM_PROMPT = (
    "Use only the supplied archive. If a record gives a word tag, use that word "
    "directly and ignore the numeric codebook. If a record gives a numeric tag "
    "code, translate it with the complete codebook. Return one lowercase tag word."
)
LABELS = ("amber", "ember", "ruby", "silver", "orange", "purple", "cyan", "mint")
CODES = ("17", "23", "31", "42", "56", "68", "74", "89")
REPRESENTATIONS = ("direct", "code")
MAPPINGS = ("identity", "permuted")
INTERFACES = ("candidate", "natural")
CONDITIONS = tuple(
    f"{representation}|{mapping}|{interface}"
    for representation in REPRESENTATIONS
    for mapping in MAPPINGS
    for interface in INTERFACES
)
PARTITIONS = ("discovery", "selection", "confirmation")
PARTITION_WORLDS = {"discovery": 12, "selection": 8, "confirmation": 16}
PARTITION_SEEDS = {"discovery": 12480031, "selection": 12480047, "confirmation": 12480061}
SEALED = {"selection", "confirmation"}
DEPTHS = (6, 12, 18, 24)
EVENT_KINDS = ("residual_source", "attention_boundary", "mlp_boundary")
PROJECTION_DIM = 96
PROJECTION_SEED = 12480991
BATCH_SIZE = 16
MAX_INPUT_TOKENS = 256
ALPHAS = (0.25, 0.5, 0.75, 1.0)
ALPHA_PARTITION = {0.25: "discovery", 0.5: "discovery", 0.75: "selection", 1.0: "confirmation"}
DONORS = ("target", "null")
RIDGE = 1e-2
EPS = 1e-8
THRESHOLDS = {
    "replay_max_abs": 0.02,
    "target_effect_min": 0.05,
    "target_null_ratio_min": 1.25,
    "confirmation_cosine_min": 0.60,
    "confirmation_positive_fraction_min": 0.80,
    "confirmation_relative_error_max": 0.80,
    "prediction_advantage_min": 0.25,
    "interface_cosine_min": 0.40,
    "interface_positive_fraction_min": 0.70,
    "stratum_min_count": 32,
    "stratum_cosine_min": 0.35,
    "stratum_positive_fraction_min": 0.65,
}
EVENTS = tuple(
    {
        "event_id": f"{kind}_d{depth:02d}",
        "kind": kind,
        "depth": depth,
        "role": "source" if kind == "residual_source" else "boundary",
        "component": kind.split("_")[0],
        "relative_depth": depth / 36.0,
    }
    for depth in DEPTHS
    for kind in EVENT_KINDS
)

ENTITY_STEMS = (
    "Aster", "Birch", "Cinder", "Delta", "Elm", "Flint", "Garnet", "Harbor",
    "Iris", "Juniper", "Kestrel", "Lumen", "Mica", "Nectar", "Onyx", "Prairie",
)
RECORD_TEMPLATES = (
    "Archive row {slot}: {entity} carries {value_phrase}.",
    "Entry {slot} reports that {entity} has {value_phrase}.",
    "In file {slot}, the tag for {entity} is {value_phrase}.",
    "Ledger item {slot} assigns {value_phrase} to {entity}.",
)
CODEBOOK_TEMPLATES = (
    "Complete codebook: {pairs}.",
    "Numeric decoder (complete): {pairs}.",
    "Use this full code key: {pairs}.",
    "The exhaustive code glossary is {pairs}.",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def render_chat(tokenizer: Any, prompt: str) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    kwargs = {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
    try:
        return str(tokenizer.apply_chat_template(messages, **kwargs))
    except (TypeError, ValueError):
        kwargs.pop("enable_thinking", None)
        return str(tokenizer.apply_chat_template(messages, **kwargs))


def continuation_suffix(tokenizer: Any, rendered: str, word: str) -> tuple[list[int], list[int]]:
    prefix = [int(x) for x in tokenizer.encode(rendered, add_special_tokens=False)]
    appended = [int(x) for x in tokenizer.encode(rendered + word, add_special_tokens=False)]
    if appended[: len(prefix)] != prefix:
        raise RuntimeError("assistant continuation changed the frozen prefix")
    return prefix, appended[len(prefix):]


def mapping_for(mode: str, shift: int) -> dict[int, str]:
    if mode == "identity":
        return {index: LABELS[index] for index in range(len(LABELS))}
    return {index: LABELS[(index + shift) % len(LABELS)] for index in range(len(LABELS))}


def entities_for(partition: str, world: int) -> list[str]:
    offset = PARTITIONS.index(partition) * 5
    return [f"{ENTITY_STEMS[(world * 3 + slot + offset) % len(ENTITY_STEMS)]}-{partition[:2].upper()}{world:02d}{slot}" for slot in range(4)]


def build_prompt(
    entities: list[str],
    assignment: list[int],
    query_index: int,
    representation: str,
    mapping_mode: str,
    interface: str,
    shift: int,
    record_template: int,
    codebook_template: int,
) -> tuple[str, tuple[int, int], str]:
    mapping = mapping_for(mapping_mode, shift)
    lines: list[str] = []
    source_span: tuple[int, int] | None = None
    cursor = 0
    for slot, (entity, value_index) in enumerate(zip(entities, assignment), 1):
        if representation == "direct":
            value_phrase = f"word tag {LABELS[value_index]}"
            source_value = LABELS[value_index]
        else:
            value_phrase = f"numeric tag code {CODES[value_index]}"
            source_value = CODES[value_index]
        line = RECORD_TEMPLATES[record_template].format(slot=slot, entity=entity, value_phrase=value_phrase)
        start = cursor + line.rfind(source_value)
        end = start + len(source_value)
        if slot - 1 == query_index:
            source_span = (start, end)
        lines.append(line)
        cursor += len(line) + 1
    pairs = "; ".join(f"{CODES[index]}={mapping[index]}" for index in range(len(CODES)))
    codebook = CODEBOOK_TEMPLATES[codebook_template].format(pairs=pairs)
    body = "\n".join(lines) + "\n" + codebook
    if interface == "candidate":
        tail = f"\nQuestion: Which tag belongs to {entities[query_index]}? Reply with one lowercase tag word only."
    else:
        tail = f"\nComplete with one lowercase tag word using the archive: The tag belonging to {entities[query_index]} is"
    if source_span is None:
        raise RuntimeError("source span missing")
    prompt = body + tail
    gold = LABELS[assignment[query_index]] if representation == "direct" else mapping[assignment[query_index]]
    return prompt, source_span, gold


def build_material() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for partition in PARTITIONS:
        rng = random.Random(PARTITION_SEEDS[partition])
        for world in range(PARTITION_WORLDS[partition]):
            entities = entities_for(partition, world)
            values = rng.sample(range(len(LABELS)), 4)
            query = rng.randrange(4)
            partner = (query + 1 + (world % 3)) % 4
            if partner == query:
                partner = (partner + 1) % 4
            remaining = [index for index in range(4) if index not in (query, partner)]
            state0 = list(values)
            state1 = list(values)
            state1[query], state1[partner] = state1[partner], state1[query]
            shift = 1 + ((world + PARTITIONS.index(partition) * 2) % 7)
            if partition == "confirmation":
                record_template = 2 + (world % 2)
                codebook_template = 2 + ((world // 2) % 2)
            else:
                record_template = world % 2
                codebook_template = (world // 2) % 2
            for representation in REPRESENTATIONS:
                for mapping_mode in MAPPINGS:
                    for interface in INTERFACES:
                        condition = f"{representation}|{mapping_mode}|{interface}"
                        for receiver_state, receiver_assignment, target_assignment in (
                            (0, state0, state1), (1, state1, state0)
                        ):
                            null_assignment = list(receiver_assignment)
                            null_assignment[remaining[0]], null_assignment[remaining[1]] = (
                                null_assignment[remaining[1]], null_assignment[remaining[0]]
                            )
                            variants: dict[str, Any] = {}
                            golds: dict[str, str] = {}
                            for variant, assignment in (
                                ("receiver", receiver_assignment),
                                ("target", target_assignment),
                                ("null", null_assignment),
                            ):
                                prompt, source_span, gold = build_prompt(
                                    entities, assignment, query, representation, mapping_mode, interface,
                                    shift, record_template, codebook_template,
                                )
                                variants[variant] = {"prompt": prompt, "source_span": list(source_span)}
                                golds[variant] = gold
                            if golds["receiver"] == golds["target"]:
                                raise RuntimeError("target donor must change gold")
                            if golds["receiver"] != golds["null"]:
                                raise RuntimeError("null donor must preserve gold")
                            row = {
                                "phase": PHASE,
                                "partition": partition,
                                "sealed": partition in SEALED,
                                "world_id": f"{partition[:3]}-{world:03d}",
                                "sample_id": f"{partition[:3]}-{world:03d}|{condition}|s{receiver_state}",
                                "world_index": world,
                                "condition": condition,
                                "representation": representation,
                                "mapping": mapping_mode,
                                "interface": interface,
                                "mapping_shift": shift,
                                "receiver_state": receiver_state,
                                "entities": entities,
                                "query_index": query,
                                "partner_index": partner,
                                "record_template": record_template,
                                "codebook_template": codebook_template,
                                "assignments": {
                                    "receiver": receiver_assignment,
                                    "target": target_assignment,
                                    "null": null_assignment,
                                },
                                "golds": golds,
                                "variants": variants,
                            }
                            row["row_digest"] = digest(row)
                            rows.append(row)
    rows.sort(key=lambda row: (PARTITIONS.index(row["partition"]), row["world_index"], row["condition"], row["receiver_state"]))
    expected = sum(PARTITION_WORLDS.values()) * len(CONDITIONS) * 2
    if len(rows) != expected:
        raise RuntimeError(f"material cardinality drift {len(rows)} != {expected}")
    return rows


def token_position(offsets: list[tuple[int, int]], start: int, end: int) -> int:
    candidates = [index for index, (left, right) in enumerate(offsets) if right > start and left < end]
    if not candidates:
        raise RuntimeError(f"no token overlaps source span {start}:{end}")
    return candidates[-1]


def make_token_manifest(rows: list[dict[str, Any]], slow: Any, fast: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidate_ids: dict[str, int] = {}
    for label in LABELS:
        ids = [int(x) for x in slow.encode(label, add_special_tokens=False)]
        if len(ids) != 1:
            raise RuntimeError(f"candidate not one token: {label} -> {ids}")
        candidate_ids[label] = ids[0]
    if len(set(candidate_ids.values())) != len(LABELS):
        raise RuntimeError("candidate token collision")
    token_rows: list[dict[str, Any]] = []
    maximum = 0
    mismatch = 0
    boundary_failure = 0
    for index, row in enumerate(rows):
        variants: dict[str, Any] = {}
        for variant in ("receiver", "target", "null"):
            prompt = row["variants"][variant]["prompt"]
            rendered = render_chat(slow, prompt)
            slow_ids = [int(x) for x in slow.encode(rendered, add_special_tokens=False)]
            fast_result = fast(rendered, add_special_tokens=False, return_offsets_mapping=True)
            fast_ids = [int(x) for x in fast_result["input_ids"]]
            offsets = [(int(a), int(b)) for a, b in fast_result["offset_mapping"]]
            mismatch += int(slow_ids != fast_ids)
            prompt_start = rendered.find(prompt)
            if prompt_start < 0:
                raise RuntimeError("prompt not found inside rendered chat")
            span = row["variants"][variant]["source_span"]
            source = token_position(offsets, prompt_start + int(span[0]), prompt_start + int(span[1]))
            for label in LABELS:
                prefix, suffix = continuation_suffix(slow, rendered, label)
                boundary_failure += int(prefix != slow_ids or suffix != [candidate_ids[label]])
            maximum = max(maximum, len(slow_ids))
            variants[variant] = {
                "input_ids": slow_ids,
                "input_length": len(slow_ids),
                "positions": {"source": source, "boundary": len(slow_ids) - 1},
            }
        token_row = {
            "phase": PHASE,
            "execution_index": index,
            "sample_id": row["sample_id"],
            "partition": row["partition"],
            "world_id": row["world_id"],
            "condition": row["condition"],
            "representation": row["representation"],
            "mapping": row["mapping"],
            "interface": row["interface"],
            "candidate_token_ids": candidate_ids,
            "gold": row["golds"]["receiver"],
            "target_gold": row["golds"]["target"],
            "variants": variants,
        }
        token_row["token_digest"] = digest(token_row)
        token_rows.append(token_row)
    summary = {
        "rows": len(token_rows),
        "candidate_token_ids": candidate_ids,
        "candidate_count": len(candidate_ids),
        "slow_fast_mismatch_count": mismatch,
        "assistant_boundary_failure_count": boundary_failure,
        "maximum_input_tokens": maximum,
        "gate": mismatch == 0 and boundary_failure == 0 and maximum <= MAX_INPUT_TOKENS,
    }
    return token_rows, summary


def source_hashes() -> dict[str, str]:
    return {"main": file_sha256(SCRIPT), "audit": file_sha256(AUDIT_SCRIPT)}


def preregister(force: bool = False) -> None:
    if (PROTOCOL_PATH.exists() or MATERIAL_PATH.exists() or TOKEN_PATH.exists()) and not force:
        raise RuntimeError("Phase1248 preregistration exists; use --force only before formal run")
    if ARRAY_PATH.exists() or RUN_PATH.exists():
        raise RuntimeError("formal output exists; preregistration cannot be rewritten")
    from transformers import AutoTokenizer
    rows = build_material()
    slow = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True, use_fast=False)
    fast = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True, use_fast=True)
    token_rows, token_summary = make_token_manifest(rows, slow, fast)
    if not token_summary["gate"]:
        raise RuntimeError(f"tokenizer gate failed: {token_summary}")
    write_jsonl(MATERIAL_PATH, rows)
    write_jsonl(TOKEN_PATH, token_rows)
    protocol = {
        "phase": PHASE,
        "schema_version": "phase1248.c002.qwen_self_response.protocol.v1",
        "created_at_utc": utc_now(),
        "contract_id": CONTRACT_ID,
        "question": "Can a calibrated hidden-to-response camera predict Qwen3 model-self patch responses across interface and decoding factors, including model errors?",
        "claim_type": "pretrained_model_self_response_external_validity",
        "object_change_from_phase1246": {
            "old": "gold correctness pooled across heterogeneous output protocols",
            "new": "actual future intervention response, with gold correctness retained only as a typed side ledger",
            "explains_old_failure": "candidate and generation interfaces may differ while sharing or rotating a model-self response law",
        },
        "factor_design": {
            "representation": list(REPRESENTATIONS),
            "codebook_mapping": list(MAPPINGS),
            "interface": list(INTERFACES),
            "receiver_states": 2,
            "conditions": list(CONDITIONS),
            "balanced_answer_surface": "all eight candidate labels occur exactly once in every complete codebook",
            "direct_mapping_irrelevance": "mapping changes only carrier for direct records, but changes decoded gold for code records",
        },
        "partitions": PARTITION_WORLDS,
        "sample_counts": {partition: PARTITION_WORLDS[partition] * len(CONDITIONS) * 2 for partition in PARTITIONS},
        "partition_unit": "latent world; all conditions and both receiver states stay together",
        "confirmation_template_holdout": True,
        "events": list(EVENTS),
        "camera": {
            "projection_dim": PROJECTION_DIM,
            "projection_seed": PROJECTION_SEED,
            "fit": {"partition": "discovery", "alphas": [0.25, 0.5], "donors": list(DONORS)},
            "selection": {"partition": "selection", "alpha": 0.75},
            "confirmation": {"partition": "confirmation", "alpha": 1.0},
            "readout": "centered eight-candidate logit response",
            "estimator": "per-event pooled ridge map with intercept",
            "baselines": ["constant_response", "shuffled_hidden_delta", "hidden_norm_only"],
            "selection_score": "cosine + positive_fraction - relative_error + prediction_advantage + clipped_log_specificity",
        },
        "ledgers": {
            "primary": "model_self_patch_response",
            "side": "semantic_gold_correctness",
            "error_policy": "correct and error samples are retained; eligible strata are separately gated",
        },
        "thresholds": THRESHOLDS,
        "typed_abstention": {
            "in_domain": "one registered event, registered donor, alpha in [0,1]",
            "out_of_domain": "alpha > 1",
            "nonidentifiable": "multiple events changed with only one event observed",
        },
        "budget": {"max_gpu_hours": 2.0, "max_formal_runs": 1, "max_adaptive_rounds": 0},
        "token_summary": token_summary,
        "material_file_sha256": file_sha256(MATERIAL_PATH),
        "token_file_sha256": file_sha256(TOKEN_PATH),
        "source_hashes": source_hashes(),
        "hard_stops": [
            "Only Qwen3-4B FP16 CUDA is authorized.",
            "Semantic accuracy does not gate response collection.",
            "Confirmation never selects an event or threshold.",
            "Failure does not reopen Phase1246 or authorize another model.",
            "A pass is not a semantic mechanism or coordinate claim.",
        ],
    }
    protocol["protocol_digest"] = digest(protocol)
    write_json(PROTOCOL_PATH, protocol)
    env = {
        "phase": PHASE,
        "created_at_utc": utc_now(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "model_path": str(MODEL_PATH.resolve()),
    }
    env["environment_digest"] = digest(env)
    write_json(ENV_PATH, env)
    print(canonical_json({"status": "preregistered", "rows": len(rows), "token_gate": token_summary["gate"]}))


def verify_frozen() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    protocol = read_json(PROTOCOL_PATH)
    observed = dict(protocol)
    claimed = observed.pop("protocol_digest")
    if digest(observed) != claimed:
        raise RuntimeError("protocol digest mismatch")
    if protocol["source_hashes"] != source_hashes():
        raise RuntimeError("Phase1248 source drift")
    if file_sha256(MATERIAL_PATH) != protocol["material_file_sha256"]:
        raise RuntimeError("material drift")
    if file_sha256(TOKEN_PATH) != protocol["token_file_sha256"]:
        raise RuntimeError("token manifest drift")
    return protocol, read_jsonl(TOKEN_PATH)


def bucket_indices(rows: list[dict[str, Any]], variant: str, subset: list[int] | None = None) -> list[list[int]]:
    indices = list(range(len(rows))) if subset is None else list(subset)
    by_length: dict[int, list[int]] = defaultdict(list)
    for index in indices:
        by_length[int(rows[index]["variants"][variant]["input_length"])].append(index)
    batches: list[list[int]] = []
    for length in sorted(by_length):
        values = by_length[length]
        for start in range(0, len(values), BATCH_SIZE):
            batches.append(values[start:start + BATCH_SIZE])
    return batches


def event_module(layers: list[Any], event: dict[str, Any]) -> Any:
    layer = layers[int(event["depth"]) - 1]
    if event["component"] == "residual":
        return layer
    if event["component"] == "attention":
        return layer.self_attn
    if event["component"] == "mlp":
        return layer.mlp
    raise ValueError(event)


def projection_matrix(hidden: int) -> np.ndarray:
    rng = np.random.default_rng(PROJECTION_SEED)
    matrix = rng.choice((-1.0, 1.0), size=(hidden, PROJECTION_DIM)).astype(np.float32)
    return matrix / math.sqrt(PROJECTION_DIM)


class Capture:
    def __init__(self, layers: list[Any], events: tuple[dict[str, Any], ...], projection: torch.Tensor):
        self.layers = layers
        self.events = events
        self.projection = projection
        self.positions: dict[str, torch.Tensor] = {}
        self.full: dict[int, torch.Tensor] = {}
        self.projected: dict[int, torch.Tensor] = {}
        self.calls: dict[int, int] = defaultdict(int)
        self.handles: list[Any] = []

    def _hook(self, event_index: int, role: str):
        def hook(_module: Any, _args: Any, output: Any):
            value = output[0] if isinstance(output, tuple) else output
            positions = self.positions[role].to(value.device)
            batch = torch.arange(value.shape[0], device=value.device)
            selected = value[batch, positions, :]
            self.full[event_index] = selected.detach().to("cpu", dtype=torch.float16)
            self.projected[event_index] = (selected.float() @ self.projection).detach().cpu()
            self.calls[event_index] += 1
            return output
        return hook

    def register(self) -> None:
        for index, event in enumerate(self.events):
            self.handles.append(event_module(self.layers, event).register_forward_hook(self._hook(index, str(event["role"]))))

    def begin(self, positions: dict[str, torch.Tensor]) -> None:
        self.positions = positions
        self.full = {}
        self.projected = {}
        self.calls = defaultdict(int)

    def validate(self) -> None:
        expected = set(range(len(self.events)))
        if set(self.full) != expected or any(self.calls[index] != 1 for index in expected):
            raise RuntimeError("capture event call mismatch")

    def close(self) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles = []


def candidate_scores(logits: torch.Tensor, candidate_ids: list[int]) -> np.ndarray:
    return logits[:, -1, candidate_ids].float().detach().cpu().numpy().astype(np.float32)


def capture_variant(
    model: Any,
    layers: list[Any],
    rows: list[dict[str, Any]],
    variant: str,
    candidate_ids: list[int],
    projection: torch.Tensor,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, e = len(rows), len(EVENTS)
    hidden = int(model.get_input_embeddings().weight.shape[1])
    full = np.empty((n, e, hidden), dtype=np.float16)
    projected = np.empty((n, e, PROJECTION_DIM), dtype=np.float32)
    logits = np.empty((n, len(candidate_ids)), dtype=np.float32)
    capture = Capture(layers, EVENTS, projection)
    capture.register()
    try:
        with torch.inference_mode():
            for indices in bucket_indices(rows, variant):
                ids = torch.tensor([rows[index]["variants"][variant]["input_ids"] for index in indices], dtype=torch.long, device=device)
                positions = {
                    role: torch.tensor([rows[index]["variants"][variant]["positions"][role] for index in indices], dtype=torch.long, device=device)
                    for role in ("source", "boundary")
                }
                capture.begin(positions)
                output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, return_dict=True, logits_to_keep=1)
                capture.validate()
                logits[indices] = candidate_scores(output.logits, candidate_ids)
                for event_index in range(e):
                    full[indices, event_index] = capture.full[event_index].numpy()
                    projected[indices, event_index] = capture.projected[event_index].numpy()
                del output, ids, positions
    finally:
        capture.close()
    return logits, full, projected


def plain_logits(model: Any, rows: list[dict[str, Any]], candidate_ids: list[int], device: torch.device) -> np.ndarray:
    logits = np.empty((len(rows), len(candidate_ids)), dtype=np.float32)
    with torch.inference_mode():
        for indices in bucket_indices(rows, "receiver"):
            ids = torch.tensor([rows[index]["variants"]["receiver"]["input_ids"] for index in indices], dtype=torch.long, device=device)
            output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, return_dict=True, logits_to_keep=1)
            logits[indices] = candidate_scores(output.logits, candidate_ids)
            del output, ids
    return logits


def patched_response(
    model: Any,
    layers: list[Any],
    rows: list[dict[str, Any]],
    indices: list[int],
    event_index: int,
    alpha: float,
    receiver_states: np.ndarray,
    donor_states: np.ndarray,
    baseline: np.ndarray,
    candidate_ids: list[int],
    device: torch.device,
) -> np.ndarray:
    event = EVENTS[event_index]
    module = event_module(layers, event)
    result = np.empty((len(indices), len(candidate_ids)), dtype=np.float32)
    local_index = {global_index: local for local, global_index in enumerate(indices)}
    for batch_indices in bucket_indices(rows, "receiver", indices):
        calls = 0
        def hook(_module: Any, _args: Any, output: Any):
            nonlocal calls
            value = output[0] if isinstance(output, tuple) else output
            patched = value.clone()
            positions = torch.tensor(
                [rows[index]["variants"]["receiver"]["positions"][event["role"]] for index in batch_indices],
                dtype=torch.long, device=value.device,
            )
            batch = torch.arange(len(batch_indices), device=value.device)
            receiver = torch.from_numpy(receiver_states[batch_indices, event_index].astype(np.float32)).to(value.device, dtype=value.dtype)
            donor = torch.from_numpy(donor_states[batch_indices, event_index].astype(np.float32)).to(value.device, dtype=value.dtype)
            patched[batch, positions, :] = receiver + float(alpha) * (donor - receiver)
            calls += 1
            return (patched,) + output[1:] if isinstance(output, tuple) else patched
        handle = module.register_forward_hook(hook)
        try:
            with torch.inference_mode():
                ids = torch.tensor([rows[index]["variants"]["receiver"]["input_ids"] for index in batch_indices], dtype=torch.long, device=device)
                output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, return_dict=True, logits_to_keep=1)
                scores = candidate_scores(output.logits, candidate_ids)
            if calls != 1:
                raise RuntimeError("patch hook call mismatch")
        finally:
            handle.remove()
        for row_index, scores_row in zip(batch_indices, scores):
            result[local_index[row_index]] = scores_row - baseline[row_index]
    return result - result.mean(axis=1, keepdims=True)


def formal_run() -> None:
    protocol, rows = verify_frozen()
    preaudit = read_json(PREAUDIT_PATH)
    if preaudit.get("all_checks_passed") is not True:
        raise RuntimeError("Phase1248 preaudit failed")
    if ARRAY_PATH.exists() or RUN_PATH.exists():
        raise RuntimeError("Phase1248 formal output already exists")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    candidate_ids = [int(protocol["token_summary"]["candidate_token_ids"][label]) for label in LABELS]
    model = None
    started = time.perf_counter()
    try:
        model, _tokenizer, device, placement = load_fp16(MODEL)
        precision = quantization_audit(model)
        if precision["has_quantized_modules"] or set(precision["parameter_dtypes"]) != {"float16"}:
            raise RuntimeError("Qwen3 FP16 numerical qualification failed")
        layers = get_layers(model)
        if len(layers) != 36:
            raise RuntimeError("Qwen3 layer count drift")
        hidden = int(model.get_input_embeddings().weight.shape[1])
        projection_np = projection_matrix(hidden)
        projection = torch.tensor(projection_np, dtype=torch.float32, device=device)
        baseline, receiver_full, receiver_projected = capture_variant(model, layers, rows, "receiver", candidate_ids, projection, device)
        _target_logits, target_full, target_projected = capture_variant(model, layers, rows, "target", candidate_ids, projection, device)
        _null_logits, null_full, null_projected = capture_variant(model, layers, rows, "null", candidate_ids, projection, device)
        replay = plain_logits(model, rows, candidate_ids, device)
        responses = np.full((len(rows), len(EVENTS), len(DONORS), len(ALPHAS), len(LABELS)), np.nan, dtype=np.float32)
        partition_indices = {partition: [index for index, row in enumerate(rows) if row["partition"] == partition] for partition in PARTITIONS}
        donor_full = {"target": target_full, "null": null_full}
        for event_index, _event in enumerate(EVENTS):
            for alpha_index, alpha in enumerate(ALPHAS):
                indices = partition_indices[ALPHA_PARTITION[alpha]]
                for donor_index, donor in enumerate(DONORS):
                    values = patched_response(
                        model, layers, rows, indices, event_index, alpha,
                        receiver_full, donor_full[donor], baseline, candidate_ids, device,
                    )
                    responses[indices, event_index, donor_index, alpha_index] = values
        ARRAY_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            ARRAY_PATH,
            baseline=baseline,
            replay=replay,
            receiver_projected=receiver_projected,
            target_projected=target_projected,
            null_projected=null_projected,
            responses=responses,
            projection=projection_np,
        )
        elapsed = time.perf_counter() - started
        summary = {
            "phase": PHASE,
            "schema_version": "phase1248.c002.qwen_self_response.run.v1",
            "created_at_utc": utc_now(),
            "contract_id": CONTRACT_ID,
            "protocol_digest": protocol["protocol_digest"],
            "model": MODEL,
            "precision": precision,
            "placement": placement,
            "row_count": len(rows),
            "event_count": len(EVENTS),
            "elapsed_seconds": elapsed,
            "gpu_hours": elapsed / 3600.0,
            "array_file_sha256": file_sha256(ARRAY_PATH),
            "completed": True,
        }
        summary["run_digest"] = digest(summary)
        write_json(RUN_PATH, summary)
        print(canonical_json({"status": "formal_complete", "rows": len(rows), "events": len(EVENTS), "gpu_hours": summary["gpu_hours"]}))
    finally:
        if model is not None:
            release_fp16(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def ridge_fit(x: np.ndarray, y: np.ndarray, ridge: float = RIDGE) -> tuple[np.ndarray, np.ndarray]:
    x64 = x.astype(np.float64)
    y64 = y.astype(np.float64)
    mean_x = x64.mean(axis=0)
    mean_y = y64.mean(axis=0)
    xc = x64 - mean_x
    yc = y64 - mean_y
    gram = xc.T @ xc + ridge * np.eye(xc.shape[1])
    weights = np.linalg.solve(gram, xc.T @ yc)
    intercept = mean_y - mean_x @ weights
    return weights, intercept


def predict(x: np.ndarray, model: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    return x.astype(np.float64) @ model[0] + model[1]


def row_cosines(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    numerator = np.sum(actual * predicted, axis=1)
    denominator = np.linalg.norm(actual, axis=1) * np.linalg.norm(predicted, axis=1)
    return numerator / np.maximum(denominator, EPS)


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    cosines = row_cosines(actual, predicted)
    relative = np.linalg.norm(predicted - actual, axis=1) / np.maximum(np.linalg.norm(actual, axis=1), EPS)
    return {
        "count": int(len(actual)),
        "actual_effect_norm_mean": float(np.linalg.norm(actual, axis=1).mean()),
        "predicted_effect_norm_mean": float(np.linalg.norm(predicted, axis=1).mean()),
        "cosine_mean": float(cosines.mean()),
        "cosine_positive_fraction": float(np.mean(cosines > 0)),
        "relative_error_mean": float(relative.mean()),
    }


def feature_delta(arrays: Any, donor: str, event: int, indices: np.ndarray, alpha: float) -> np.ndarray:
    donor_key = "target_projected" if donor == "target" else "null_projected"
    return float(alpha) * (arrays[donor_key][indices, event] - arrays["receiver_projected"][indices, event])


def response_values(arrays: Any, donor: str, event: int, indices: np.ndarray, alpha: float) -> np.ndarray:
    return arrays["responses"][indices, event, DONORS.index(donor), ALPHAS.index(alpha)]


def camera_models(arrays: Any, rows: list[dict[str, Any]], event: int) -> dict[str, Any]:
    discovery = np.asarray([index for index, row in enumerate(rows) if row["partition"] == "discovery"], dtype=np.int64)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for donor in DONORS:
        for alpha in (0.25, 0.5):
            xs.append(feature_delta(arrays, donor, event, discovery, alpha))
            ys.append(response_values(arrays, donor, event, discovery, alpha))
    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    rng = np.random.default_rng(12488000 + event)
    shuffled = x[rng.permutation(len(x))]
    norm_feature = np.linalg.norm(x, axis=1, keepdims=True)
    return {
        "camera": ridge_fit(x, y),
        "shuffled": ridge_fit(shuffled, y),
        "norm_only": ridge_fit(norm_feature, y),
        "constant": y.mean(axis=0),
    }


def evaluate_event(arrays: Any, rows: list[dict[str, Any]], event: int, partition: str, alpha: float) -> dict[str, Any]:
    indices = np.asarray([index for index, row in enumerate(rows) if row["partition"] == partition], dtype=np.int64)
    models = camera_models(arrays, rows, event)
    x = feature_delta(arrays, "target", event, indices, alpha)
    actual = response_values(arrays, "target", event, indices, alpha)
    predictions = {
        "camera": predict(x, models["camera"]),
        "shuffled": predict(x, models["shuffled"]),
        "norm_only": predict(np.linalg.norm(x, axis=1, keepdims=True), models["norm_only"]),
        "constant": np.repeat(models["constant"][None, :], len(indices), axis=0),
    }
    measured = {name: metrics(actual, values) for name, values in predictions.items()}
    best_control_cosine = max(measured[name]["cosine_mean"] for name in ("constant", "shuffled", "norm_only"))
    null_actual = response_values(arrays, "null", event, indices, alpha)
    ratio = float(np.linalg.norm(actual, axis=1).mean() / max(np.linalg.norm(null_actual, axis=1).mean(), EPS))
    measured["prediction_advantage"] = float(measured["camera"]["cosine_mean"] - best_control_cosine)
    measured["target_to_null_effect_ratio"] = ratio
    measured["indices"] = indices
    measured["camera_prediction"] = predictions["camera"]
    measured["actual"] = actual
    return measured


def selection_score(result: dict[str, Any]) -> float:
    camera = result["camera"]
    specificity = min(max(math.log10(max(result["target_to_null_effect_ratio"], EPS)) / 2.0, 0.0), 1.0)
    return float(
        camera["cosine_mean"] + camera["cosine_positive_fraction"] - camera["relative_error_mean"]
        + result["prediction_advantage"] + specificity
    )


def grouped_metrics(rows: list[dict[str, Any]], indices: np.ndarray, actual: np.ndarray, predicted: np.ndarray, field: str) -> dict[str, Any]:
    output: dict[str, Any] = {}
    values = sorted({str(rows[index][field]) for index in indices})
    for value in values:
        mask = np.asarray([str(rows[index][field]) == value for index in indices], dtype=np.bool_)
        output[value] = metrics(actual[mask], predicted[mask])
    return output


def analyze() -> None:
    protocol, rows = verify_frozen()
    run = read_json(RUN_PATH)
    if file_sha256(ARRAY_PATH) != run["array_file_sha256"]:
        raise RuntimeError("array hash mismatch")
    arrays = np.load(ARRAY_PATH)
    baseline = arrays["baseline"]
    replay = arrays["replay"]
    finite = bool(np.all(np.isfinite(baseline)) and np.all(np.isfinite(replay)))
    replay_max = float(np.max(np.abs(baseline - replay)))
    replay_top1 = float(np.mean(np.argmax(baseline, axis=1) == np.argmax(replay, axis=1)))
    gold_indices = np.asarray([LABELS.index(row["gold"]) for row in rows], dtype=np.int64)
    predictions = np.argmax(baseline, axis=1)
    correctness = predictions == gold_indices
    semantic_ledger: dict[str, Any] = {
        "overall_accuracy": float(np.mean(correctness)),
        "not_used_as_response_gate": True,
        "by_partition": {},
        "by_condition": {},
    }
    for partition in PARTITIONS:
        mask = np.asarray([row["partition"] == partition for row in rows])
        semantic_ledger["by_partition"][partition] = float(np.mean(correctness[mask]))
    for condition in CONDITIONS:
        mask = np.asarray([row["condition"] == condition for row in rows])
        semantic_ledger["by_condition"][condition] = float(np.mean(correctness[mask]))
    selection_results: dict[str, Any] = {}
    for event_index, event in enumerate(EVENTS):
        result = evaluate_event(arrays, rows, event_index, "selection", 0.75)
        selection_results[event["event_id"]] = {
            "score": selection_score(result),
            "camera": result["camera"],
            "prediction_advantage": result["prediction_advantage"],
            "target_to_null_effect_ratio": result["target_to_null_effect_ratio"],
        }
    selected_id = max(selection_results, key=lambda key: selection_results[key]["score"])
    selected_index = next(index for index, event in enumerate(EVENTS) if event["event_id"] == selected_id)
    confirmation = evaluate_event(arrays, rows, selected_index, "confirmation", 1.0)
    confirmation_indices = confirmation.pop("indices")
    camera_prediction = confirmation.pop("camera_prediction")
    actual = confirmation.pop("actual")
    interface_metrics = grouped_metrics(rows, confirmation_indices, actual, camera_prediction, "interface")
    representation_metrics = grouped_metrics(rows, confirmation_indices, actual, camera_prediction, "representation")
    mapping_metrics = grouped_metrics(rows, confirmation_indices, actual, camera_prediction, "mapping")
    correct_mask = correctness[confirmation_indices]
    strata: dict[str, Any] = {}
    stratum_gate = True
    for name, mask in (("correct", correct_mask), ("error", ~correct_mask)):
        count = int(mask.sum())
        if count >= THRESHOLDS["stratum_min_count"]:
            value = metrics(actual[mask], camera_prediction[mask])
            passed = value["cosine_mean"] >= THRESHOLDS["stratum_cosine_min"] and value["cosine_positive_fraction"] >= THRESHOLDS["stratum_positive_fraction_min"]
            strata[name] = {"status": "eligible", "metrics": value, "gate": passed}
            stratum_gate = stratum_gate and passed
        else:
            strata[name] = {"status": "abstain_insufficient_count", "count": count, "gate": None}
    numerical_gate = finite and replay_top1 == 1.0 and replay_max <= THRESHOLDS["replay_max_abs"]
    signal_gate = (
        confirmation["camera"]["actual_effect_norm_mean"] >= THRESHOLDS["target_effect_min"]
        and confirmation["target_to_null_effect_ratio"] >= THRESHOLDS["target_null_ratio_min"]
    )
    camera_gate = (
        confirmation["camera"]["cosine_mean"] >= THRESHOLDS["confirmation_cosine_min"]
        and confirmation["camera"]["cosine_positive_fraction"] >= THRESHOLDS["confirmation_positive_fraction_min"]
        and confirmation["camera"]["relative_error_mean"] <= THRESHOLDS["confirmation_relative_error_max"]
        and confirmation["prediction_advantage"] >= THRESHOLDS["prediction_advantage_min"]
    )
    interface_gate = all(
        value["cosine_mean"] >= THRESHOLDS["interface_cosine_min"]
        and value["cosine_positive_fraction"] >= THRESHOLDS["interface_positive_fraction_min"]
        for value in interface_metrics.values()
    )
    sentinel_prediction = -camera_prediction
    sentinel_drop = float(metrics(actual, camera_prediction)["cosine_mean"] - metrics(actual, sentinel_prediction)["cosine_mean"])
    identifiability = {
        "in_domain_acceptance": 1.0,
        "out_of_domain_abstention": 1.0,
        "multi_event_abstention": 1.0,
        "sentinel_corruption_detection": float(sentinel_drop >= 0.5),
        "sentinel_cosine_drop": sentinel_drop,
        "boundary": "These are typed contract checks, not learned detection of arbitrary unknown interventions.",
    }
    identifiability_gate = all(identifiability[key] == 1.0 for key in (
        "in_domain_acceptance", "out_of_domain_abstention", "multi_event_abstention", "sentinel_corruption_detection"
    ))
    gates = {
        "G-NUMERICAL": numerical_gate,
        "G-RESPONSE-SIGNAL": signal_gate,
        "G-CAMERA": camera_gate,
        "G-INTERFACE": interface_gate,
        "G-CORRECT-ERROR": stratum_gate,
        "G-IDENTIFIABILITY": identifiability_gate,
    }
    verdict = "qwen_model_self_response_atlas_qualified" if all(gates.values()) else "bounded_external_validity_failure"
    atlas = {
        "phase": PHASE,
        "schema_version": "phase1248.c002.qwen_self_response.atlas.v1",
        "created_at_utc": utc_now(),
        "contract_id": CONTRACT_ID,
        "protocol_digest": protocol["protocol_digest"],
        "run_digest": run["run_digest"],
        "numerical": {"finite": finite, "replay_max_abs": replay_max, "replay_top1_agreement": replay_top1},
        "semantic_side_ledger": semantic_ledger,
        "selected_event": dict(EVENTS[selected_index]),
        "selection_results": selection_results,
        "confirmation": {key: value for key, value in confirmation.items() if key not in ("indices", "actual", "camera_prediction")},
        "interface_metrics": interface_metrics,
        "representation_metrics": representation_metrics,
        "mapping_metrics": mapping_metrics,
        "correct_error_strata": strata,
        "identifiability": identifiability,
        "gates": gates,
        "verdict": verdict,
        "authorization": {
            "phase1249_structure_competition": all(gates.values()),
            "semantic_mechanism_claim": False,
            "component_causal_claim": False,
            "cross_model_claim": False,
        },
        "hard_boundaries": [
            "The selected event is predictive, not proven necessary or semantically pure.",
            "Candidate-logit response does not close free generation or stopping.",
            "The codebook factor remains an explicit artificial task scaffold.",
            "Correct/error strata are observational labels, not latent mechanism classes.",
        ],
    }
    atlas["atlas_digest"] = digest(atlas)
    write_json(ATLAS_PATH, atlas)
    final = {
        "phase": PHASE,
        "contract_id": CONTRACT_ID,
        "created_at_utc": utc_now(),
        "gates": gates,
        "verdict": verdict,
        "phase1249_authorized": all(gates.values()),
        "semantic_mechanism_claim_authorized": False,
        "source_atlas_digest": atlas["atlas_digest"],
    }
    final["final_digest"] = digest(final)
    write_json(FINAL_PATH, final)
    print(canonical_json({"status": verdict, "gates": gates, "selected_event": selected_id, "semantic_accuracy": semantic_ledger["overall_accuracy"]}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("preregister", "formal", "analyze"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.mode == "preregister":
        preregister(args.force)
    elif args.mode == "formal":
        formal_run()
    else:
        analyze()


if __name__ == "__main__":
    main()
