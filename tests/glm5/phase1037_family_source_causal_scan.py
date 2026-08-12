#!/usr/bin/env python3
"""Run the frozen Phase1037 semantic-family source interventions."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import get_layers, get_model_info
from phase1023_fp16_utils import (
    load_fp16,
    quantization_audit,
    release_fp16,
)
import phase1035_native_family_routing_protocol as source
import phase1037_family_source_causal_protocol as protocol


BATCH_SIZE = {"qwen3": 32, "glm4": 8, "deepseek7b": 8}
EPS = 1e-8


def chunks(rows: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(rows), size):
        yield rows[start:start + size]


def output_tensor(output: Any) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


def replace_output(output: Any, hidden: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        return (hidden,) + output[1:]
    return hidden


def make_clean_batch(
    rows: list[dict[str, Any]],
    *,
    pad_token_id: int,
    device: torch.device,
    cache_indices: np.ndarray,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    np.ndarray,
]:
    width = max(len(row["input_ids"]) for row in rows)
    ids = torch.full(
        (len(rows), width), int(pad_token_id), dtype=torch.long
    )
    attention_mask = torch.zeros((len(rows), width), dtype=torch.long)
    positions = torch.empty((len(rows), 2), dtype=torch.long)
    pre_positions = torch.empty(len(rows), dtype=torch.long)
    for row_index, row in enumerate(rows):
        values = torch.tensor(row["input_ids"], dtype=torch.long)
        ids[row_index, :len(values)] = values
        attention_mask[row_index, :len(values)] = 1
        for role_index, role in enumerate(("concept_a", "concept_b")):
            start, end = (
                int(value) for value in row["anchor_spans"][role]
            )
            if start != end:
                raise RuntimeError("Phase1037 requires one-token concepts")
            positions[row_index, role_index] = start
        pre_positions[row_index] = int(
            row["anchor_spans"]["pre_output"][1]
        )
    return (
        ids.to(device),
        attention_mask.to(device),
        positions,
        pre_positions,
        cache_indices,
    )


class SourceCacheCapture:
    def __init__(
        self,
        layers: list[Any],
        depths: list[int],
        cache: np.memmap,
    ):
        self.layers = layers
        self.depths = depths
        self.depth_slots = {
            depth: index for index, depth in enumerate(depths)
        }
        self.cache = cache
        self.positions: torch.Tensor | None = None
        self.cache_indices: np.ndarray | None = None
        self.counts: dict[int, int] = defaultdict(int)
        self.handles = []

    def _hook(self, physical_depth: int):
        def hook(module, args, output):
            if self.positions is None or self.cache_indices is None:
                raise RuntimeError("source cache context missing")
            hidden = output_tensor(output)
            positions = self.positions.to(hidden.device)
            batch = torch.arange(hidden.shape[0], device=hidden.device)
            values = hidden[
                batch[:, None], positions, :
            ]
            self.cache[
                self.cache_indices,
                self.depth_slots[physical_depth],
                :,
                :,
            ] = values.detach().to(
                "cpu", dtype=torch.float16
            ).numpy()
            self.counts[physical_depth] += 1
            return output
        return hook

    def register(self) -> None:
        for physical_depth in self.depths:
            self.handles.append(
                self.layers[physical_depth - 1].register_forward_hook(
                    self._hook(physical_depth)
                )
            )

    def begin(
        self,
        positions: torch.Tensor,
        cache_indices: np.ndarray,
    ) -> None:
        self.positions = positions
        self.cache_indices = cache_indices
        self.counts = defaultdict(int)

    def end(self) -> None:
        expected = {depth: 1 for depth in self.depths}
        if dict(self.counts) != expected:
            raise RuntimeError(
                f"source cache hook count drift: {dict(self.counts)}"
            )
        self.positions = None
        self.cache_indices = None

    def close(self) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles = []


def role_index(role: str) -> int:
    return 0 if role == "concept_a" else 1


def patch_spec(
    target: dict[str, Any],
    condition: str,
) -> tuple[int, str, str]:
    if condition == "self_selected":
        return (
            int(target["target_case_index"]),
            str(target["selected_role"]),
            str(target["selected_role"]),
        )
    if condition == "same_family_selected":
        return (
            int(target["same_family_case_index"]),
            str(target["selected_role"]),
            str(target["selected_role"]),
        )
    if condition == "cross_family_selected":
        return (
            int(target["cross_family_case_index"]),
            str(target["selected_role"]),
            str(target["selected_role"]),
        )
    if condition == "cross_family_unselected":
        return (
            int(target["cross_family_case_index"]),
            str(target["unselected_role"]),
            str(target["unselected_role"]),
        )
    if condition == "cross_family_wrong_target":
        return (
            int(target["cross_family_case_index"]),
            str(target["selected_role"]),
            str(target["unselected_role"]),
        )
    raise ValueError(condition)


def make_patch_batch(
    target_rows: list[dict[str, Any]],
    cases: dict[int, dict[str, Any]],
    cache_lookup: dict[int, int],
    source_cache: np.memmap,
    depth_slot: int,
    condition: str,
    *,
    pad_token_id: int,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    np.ndarray,
]:
    model_rows = [
        cases[int(row["target_case_index"])] for row in target_rows
    ]
    width = max(len(row["input_ids"]) for row in model_rows)
    ids = torch.full(
        (len(model_rows), width), int(pad_token_id), dtype=torch.long
    )
    attention_mask = torch.zeros(
        (len(model_rows), width), dtype=torch.long
    )
    patch_positions = torch.empty(len(model_rows), dtype=torch.long)
    pre_positions = torch.empty(len(model_rows), dtype=torch.long)
    replacements = []
    for row_index, (target, model_row) in enumerate(
        zip(target_rows, model_rows)
    ):
        values = torch.tensor(model_row["input_ids"], dtype=torch.long)
        ids[row_index, :len(values)] = values
        attention_mask[row_index, :len(values)] = 1
        donor_case, donor_role, target_role = patch_spec(
            target, condition
        )
        start, end = (
            int(value)
            for value in model_row["anchor_spans"][target_role]
        )
        if start != end:
            raise RuntimeError("patch target span is not one token")
        patch_positions[row_index] = start
        pre_positions[row_index] = int(
            model_row["anchor_spans"]["pre_output"][1]
        )
        replacements.append(np.asarray(
            source_cache[
                cache_lookup[donor_case],
                depth_slot,
                role_index(donor_role),
                :,
            ],
            dtype=np.float16,
        ))
    return (
        ids.to(device),
        attention_mask.to(device),
        patch_positions,
        pre_positions,
        torch.from_numpy(np.stack(replacements)),
        np.asarray(
            [int(row["target_index"]) for row in target_rows],
            dtype=np.int64,
        ),
    )


class PatchReadoutCapture:
    def __init__(
        self,
        patch_layer: Any,
        readout_layer: Any,
    ):
        self.patch_layer = patch_layer
        self.readout_layer = readout_layer
        self.patch_positions: torch.Tensor | None = None
        self.pre_positions: torch.Tensor | None = None
        self.replacements: torch.Tensor | None = None
        self.readout: torch.Tensor | None = None
        self.counts: dict[str, int] = defaultdict(int)
        self.handles = []

    def _patch_hook(self, module, args, output):
        if self.patch_positions is None or self.replacements is None:
            raise RuntimeError("patch context missing")
        hidden = output_tensor(output)
        patched = hidden.clone()
        batch = torch.arange(hidden.shape[0], device=hidden.device)
        positions = self.patch_positions.to(hidden.device)
        replacement = self.replacements.to(
            hidden.device, dtype=hidden.dtype
        )
        patched[batch, positions, :] = replacement
        self.counts["patch"] += 1
        return replace_output(output, patched)

    def _readout_hook(self, module, args, output):
        if self.pre_positions is None:
            raise RuntimeError("readout positions missing")
        hidden = output_tensor(output)
        batch = torch.arange(hidden.shape[0], device=hidden.device)
        self.readout = hidden[
            batch, self.pre_positions.to(hidden.device), :
        ].detach().to("cpu", dtype=torch.float16)
        self.counts["readout"] += 1
        return output

    def register(self) -> None:
        self.handles.append(
            self.patch_layer.register_forward_hook(self._patch_hook)
        )
        self.handles.append(
            self.readout_layer.register_forward_hook(self._readout_hook)
        )

    def begin(
        self,
        patch_positions: torch.Tensor,
        pre_positions: torch.Tensor,
        replacements: torch.Tensor,
    ) -> None:
        self.patch_positions = patch_positions
        self.pre_positions = pre_positions
        self.replacements = replacements
        self.readout = None
        self.counts = defaultdict(int)

    def end(self) -> torch.Tensor:
        if dict(self.counts) != {"patch": 1, "readout": 1}:
            raise RuntimeError(
                f"patch/readout hook count drift: {dict(self.counts)}"
            )
        if self.readout is None:
            raise RuntimeError("patched readout missing")
        result = self.readout
        self.patch_positions = None
        self.pre_positions = None
        self.replacements = None
        self.readout = None
        return result

    def close(self) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles = []


def normalize(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(norms, EPS)


def discovery_prototypes(
    model: str,
    cases: list[dict[str, Any]],
) -> np.ndarray:
    states = np.load(
        source.OUT_ROOT
        / "atlas"
        / model
        / "boundary_states.fp32.npy",
        mmap_mode="r",
    )
    discovery = [
        int(row["case_index"])
        for row in cases
        if row["split"] == "discovery"
    ]
    expected = np.asarray(
        [int(row["expected_index"]) for row in cases], dtype=np.int64
    )
    values = np.asarray(states[discovery, 0], dtype=np.float32)
    prototypes = []
    for family_index in range(len(source.FAMILIES)):
        mask = expected[discovery] == family_index
        finite = np.all(np.isfinite(values), axis=-1)
        prototypes.append(np.mean(values[mask & finite], axis=0))
    return normalize(np.stack(prototypes))


def scalar_summary(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    clean = values[np.isfinite(values)]
    return {
        "count": int(len(values)),
        "finite_count": int(len(clean)),
        "finite_rate": float(len(clean) / max(1, len(values))),
        "mean": float(np.mean(clean)) if len(clean) else None,
        "median": float(np.median(clean)) if len(clean) else None,
        "positive_rate": (
            float(np.mean(clean > 0)) if len(clean) else None
        ),
    }


def summarize(
    patched_logits: np.ndarray,
    patched_readout: np.ndarray,
    clean_logits: np.ndarray,
    targets: list[dict[str, Any]],
    prototypes: np.ndarray,
    depths: list[int],
) -> dict[str, Any]:
    target_family = np.asarray(
        [int(row["target_family_index"]) for row in targets],
        dtype=np.int64,
    )
    cross_family = np.asarray(
        [int(row["cross_family_index"]) for row in targets],
        dtype=np.int64,
    )
    clean_finite = np.all(np.isfinite(clean_logits), axis=-1)
    clean_prediction = np.argmax(
        np.where(np.isfinite(clean_logits), clean_logits, -np.inf),
        axis=-1,
    )
    clean_correct = clean_finite & (clean_prediction == target_family)
    groups = {
        "all": np.arange(len(targets)),
        "clean_candidate_correct": np.where(clean_correct)[0],
        "clean_candidate_error": np.where(~clean_correct)[0],
        "template_2": np.asarray([
            index
            for index, row in enumerate(targets)
            if int(row["template_index"]) == 2
        ], dtype=np.int64),
        "template_3": np.asarray([
            index
            for index, row in enumerate(targets)
            if int(row["template_index"]) == 3
        ], dtype=np.int64),
    }
    clean_margin = (
        clean_logits[np.arange(len(targets)), cross_family]
        - clean_logits[np.arange(len(targets)), target_family]
    )
    rows = []
    for depth_slot, depth in enumerate(depths):
        for condition_index, condition in enumerate(protocol.CONDITIONS):
            logits = np.asarray(
                patched_logits[:, depth_slot, condition_index],
                dtype=np.float32,
            )
            readout = np.asarray(
                patched_readout[:, depth_slot, condition_index],
                dtype=np.float32,
            )
            finite = np.all(np.isfinite(logits), axis=-1)
            margin = (
                logits[np.arange(len(targets)), cross_family]
                - logits[np.arange(len(targets)), target_family]
            )
            shift = margin - clean_margin
            predictions = np.argmax(
                np.where(np.isfinite(logits), logits, -np.inf), axis=-1
            )
            readout_finite = np.all(np.isfinite(readout), axis=-1)
            readout_predictions = np.full(len(targets), -1, dtype=np.int64)
            if np.any(readout_finite):
                scores = normalize(readout[readout_finite]) @ prototypes.T
                readout_predictions[readout_finite] = np.argmax(
                    scores, axis=-1
                )
            for group, indices in groups.items():
                usable = indices[finite[indices]]
                internal_usable = indices[readout_finite[indices]]
                rows.append({
                    "physical_depth": int(depth),
                    "depth_slot": depth_slot,
                    "condition": condition,
                    "group": group,
                    "row_count": int(len(indices)),
                    "finite_logit_row_rate": float(
                        np.mean(finite[indices])
                    ) if len(indices) else None,
                    "cross_minus_target_margin": scalar_summary(
                        margin[usable]
                    ),
                    "margin_shift_from_clean": scalar_summary(
                        shift[usable]
                    ),
                    "absolute_margin_shift_from_clean": scalar_summary(
                        np.abs(shift[usable])
                    ),
                    "candidate_target_top1_rate": (
                        float(np.mean(
                            predictions[usable] == target_family[usable]
                        ))
                        if len(usable)
                        else None
                    ),
                    "candidate_cross_top1_rate": (
                        float(np.mean(
                            predictions[usable] == cross_family[usable]
                        ))
                        if len(usable)
                        else None
                    ),
                    "internal_readout_finite_rate": (
                        float(np.mean(readout_finite[indices]))
                        if len(indices)
                        else None
                    ),
                    "internal_target_top1_rate": (
                        float(np.mean(
                            readout_predictions[internal_usable]
                            == target_family[internal_usable]
                        ))
                        if len(internal_usable)
                        else None
                    ),
                    "internal_cross_top1_rate": (
                        float(np.mean(
                            readout_predictions[internal_usable]
                            == cross_family[internal_usable]
                        ))
                        if len(internal_usable)
                        else None
                    ),
                })
    return {
        "clean_candidate_correct_count": int(clean_correct.sum()),
        "clean_candidate_error_count": int((~clean_correct).sum()),
        "condition_rows": rows,
    }


def finite_summary(values: np.ndarray) -> dict[str, Any]:
    finite = np.isfinite(values)
    return {
        "all_finite": bool(finite.all()),
        "finite_value_rate": float(np.mean(finite)),
        "nonfinite_value_count": int(np.size(values) - finite.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=protocol.MODELS)
    args = parser.parse_args()
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    cases_list = source.read_jsonl(
        source.OUT_ROOT
        / "protocol"
        / f"cases.{args.model}.jsonl"
    )
    cases = {
        int(row["case_index"]): row for row in cases_list
    }
    targets = protocol.read_jsonl(
        protocol.OUT_ROOT / "protocol" / "targets.jsonl"
    )
    confirmation_rows = [
        row for row in cases_list if row["split"] == "confirmation"
    ]
    cache_lookup = {
        int(row["case_index"]): index
        for index, row in enumerate(confirmation_rows)
    }
    depths = [
        int(value)
        for value in prereg["model_physical_depths"][args.model]
    ]
    atlas_dir = protocol.OUT_ROOT / "atlas" / args.model
    atlas_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    model = tokenizer = None

    try:
        model, tokenizer, device, placement = load_fp16(args.model)
        precision = quantization_audit(model)
        if (
            precision["has_quantized_modules"]
            or precision["has_bf16_parameters"]
            or not precision["has_fp16_parameters"]
        ):
            raise RuntimeError("FP16/no-quantization audit failed")
        layers = get_layers(model)
        info = get_model_info(model, args.model)
        readout_depth = info.n_layers - 1
        if max(depths) >= readout_depth:
            raise RuntimeError("patch depth must precede readout depth")

        source_cache = np.lib.format.open_memmap(
            atlas_dir / "source_states.fp16.npy",
            mode="w+",
            dtype=np.float16,
            shape=(
                len(confirmation_rows),
                len(depths),
                2,
                info.d_model,
            ),
        )
        source_cache[:] = np.nan
        cache_capture = SourceCacheCapture(
            layers, depths, source_cache
        )
        cache_capture.register()
        try:
            for row_batch in chunks(
                confirmation_rows, BATCH_SIZE[args.model]
            ):
                local_indices = np.asarray([
                    cache_lookup[int(row["case_index"])]
                    for row in row_batch
                ], dtype=np.int64)
                (
                    input_ids,
                    attention_mask,
                    positions,
                    _,
                    cache_indices,
                ) = make_clean_batch(
                    row_batch,
                    pad_token_id=(
                        tokenizer.pad_token_id
                        if tokenizer.pad_token_id is not None
                        else tokenizer.eos_token_id
                    ),
                    device=device,
                    cache_indices=local_indices,
                )
                cache_capture.begin(positions, cache_indices)
                with torch.inference_mode():
                    output = model.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                    )
                cache_capture.end()
                del output
        finally:
            cache_capture.close()
        source_cache.flush()

        patched_logits = np.lib.format.open_memmap(
            atlas_dir / "patched_candidate_logits.fp32.npy",
            mode="w+",
            dtype=np.float32,
            shape=(
                len(targets),
                len(depths),
                len(protocol.CONDITIONS),
                len(source.FAMILIES),
            ),
        )
        patched_readout = np.lib.format.open_memmap(
            atlas_dir / "patched_penultimate_readout.fp16.npy",
            mode="w+",
            dtype=np.float16,
            shape=(
                len(targets),
                len(depths),
                len(protocol.CONDITIONS),
                info.d_model,
            ),
        )
        patched_logits[:] = np.nan
        patched_readout[:] = np.nan
        candidate_ids = torch.tensor(
            cases_list[0]["candidate_token_ids"], dtype=torch.long
        )

        for depth_slot, physical_depth in enumerate(depths):
            capture = PatchReadoutCapture(
                layers[physical_depth - 1],
                layers[readout_depth - 1],
            )
            capture.register()
            try:
                for condition_index, condition in enumerate(
                    protocol.CONDITIONS
                ):
                    for batch_number, target_batch in enumerate(
                        chunks(targets, BATCH_SIZE[args.model]), 1
                    ):
                        (
                            input_ids,
                            attention_mask,
                            patch_positions,
                            pre_positions,
                            replacements,
                            target_indices,
                        ) = make_patch_batch(
                            target_batch,
                            cases,
                            cache_lookup,
                            source_cache,
                            depth_slot,
                            condition,
                            pad_token_id=(
                                tokenizer.pad_token_id
                                if tokenizer.pad_token_id is not None
                                else tokenizer.eos_token_id
                            ),
                            device=device,
                        )
                        capture.begin(
                            patch_positions,
                            pre_positions,
                            replacements,
                        )
                        with torch.inference_mode():
                            output = model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                use_cache=False,
                                return_dict=True,
                            )
                        readout = capture.end()
                        logits = output.logits
                        batch = torch.arange(
                            logits.shape[0], device=logits.device
                        )
                        selected = logits[
                            batch, pre_positions.to(logits.device), :
                        ].float()
                        selected_candidates = selected.index_select(
                            -1, candidate_ids.to(selected.device)
                        )
                        patched_logits[
                            target_indices, depth_slot, condition_index, :
                        ] = (
                            selected_candidates.detach().cpu().numpy()
                        )
                        patched_readout[
                            target_indices, depth_slot, condition_index, :
                        ] = readout.numpy()
                        del output, logits, selected, selected_candidates
                    print(
                        f"[phase1037] {args.model} depth={physical_depth} "
                        f"condition={condition}",
                        flush=True,
                    )
            finally:
                capture.close()
        patched_logits.flush()
        patched_readout.flush()

        source_clean_logits = np.load(
            source.OUT_ROOT
            / "atlas"
            / args.model
            / "candidate_logits.fp32.npy",
            mmap_mode="r",
        )
        target_case_indices = np.asarray([
            int(row["target_case_index"]) for row in targets
        ], dtype=np.int64)
        clean_logits = np.asarray(
            source_clean_logits[target_case_indices], dtype=np.float32
        )
        prototypes = discovery_prototypes(args.model, cases_list)
        metrics = {
            "schema_version": "phase1037_model_metrics.v1",
            "phase": protocol.PHASE,
            "model": args.model,
            "causal_summary": summarize(
                patched_logits,
                patched_readout,
                clean_logits,
                targets,
                prototypes,
                depths,
            ),
        }
        summary = {
            "schema_version": "phase1037_model_summary.v1",
            "phase": protocol.PHASE,
            "model": args.model,
            "protocol_digest": prereg["protocol_digest"],
            "source_protocol_digest": prereg["source_protocol_digest"],
            "evidence_protocol_digest": prereg[
                "evidence_protocol_digest"
            ],
            "precision": precision,
            "placement": placement,
            "model_info": {
                "class": info.model_class,
                "n_layers": info.n_layers,
                "d_model": info.d_model,
            },
            "patch_depths": depths,
            "readout_depth": readout_depth,
            "sample_counts": {
                "targets": len(targets),
                "conditions": len(protocol.CONDITIONS),
                "depths": len(depths),
                "patched_forward_rows": (
                    len(targets)
                    * len(protocol.CONDITIONS)
                    * len(depths)
                ),
            },
            "array_finiteness": {
                "source_states": finite_summary(source_cache),
                "patched_candidate_logits": finite_summary(patched_logits),
                "patched_penultimate_readout": finite_summary(
                    patched_readout
                ),
            },
            "elapsed_seconds": time.time() - started,
        }
        protocol.write_json(atlas_dir / "metrics.json", metrics)
        protocol.write_json(atlas_dir / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        if model is not None:
            release_fp16(model)


if __name__ == "__main__":
    main()
