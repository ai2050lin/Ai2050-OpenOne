#!/usr/bin/env python3
"""Run the Phase1028 role-by-depth causal leverage map in FP16."""

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
import phase1028_role_depth_causal_map_protocol as protocol
from phase1023_fp16_utils import (
    MODELS,
    load_fp16,
    quantization_audit,
    release_fp16,
)


BATCH_SIZE = {"qwen3": 32, "glm4": 8, "deepseek7b": 8}
ROLE_INDEX = {role: index for index, role in enumerate(protocol.ROLES)}
EPS = 1e-8


def chunks(rows: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(rows), size):
        yield rows[start:start + size]


def make_batch(
    rows: list[dict[str, Any]],
    *,
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    width = max(len(row["input_ids"]) for row in rows)
    ids = torch.full(
        (len(rows), width),
        int(pad_token_id),
        dtype=torch.long,
    )
    mask = torch.zeros((len(rows), width), dtype=torch.long)
    positions = torch.empty(
        (len(rows), len(protocol.ROLES)),
        dtype=torch.long,
    )
    for index, row in enumerate(rows):
        value = torch.tensor(row["input_ids"], dtype=torch.long)
        ids[index, :len(value)] = value
        mask[index, :len(value)] = 1
        for role_index, role in enumerate(protocol.ROLES):
            positions[index, role_index] = int(
                row["role_positions"][role]
            )
    return ids.to(device), mask.to(device), positions


def replace_output(output, hidden: torch.Tensor):
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    return hidden


class MultiDepthCleanCapture:
    def __init__(
        self,
        layers,
        patch_depths: list[int],
        readout_depth: int,
    ):
        self.layers = layers
        self.patch_depths = patch_depths
        self.readout_depth = readout_depth
        self.positions: torch.Tensor | None = None
        self.values: dict[int, torch.Tensor] = {}
        self.readout: torch.Tensor | None = None
        self.counts: dict[str, int] = defaultdict(int)
        self.handles = []

    def _role_hook(self, depth: int):
        def hook(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if self.positions is None:
                raise RuntimeError("clean positions missing")
            positions = self.positions.to(hidden.device)
            batch = torch.arange(
                hidden.shape[0], device=hidden.device
            )[:, None]
            self.values[depth] = hidden[batch, positions, :].detach()
            self.counts[f"depth/{depth}"] += 1
            return output
        return hook

    def _readout_hook(self, module, args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if self.positions is None:
            raise RuntimeError("readout positions missing")
        positions = self.positions[
            :, ROLE_INDEX["pre_output"]
        ].to(hidden.device)
        batch = torch.arange(hidden.shape[0], device=hidden.device)
        self.readout = hidden[batch, positions, :].detach()
        self.counts["readout"] += 1
        return output

    def register(self) -> None:
        for depth in self.patch_depths:
            self.handles.append(
                self.layers[depth - 1].register_forward_hook(
                    self._role_hook(depth)
                )
            )
        self.handles.append(
            self.layers[
                self.readout_depth - 1
            ].register_forward_hook(self._readout_hook)
        )

    def begin(self, positions: torch.Tensor) -> None:
        self.positions = positions
        self.values = {}
        self.readout = None
        self.counts = defaultdict(int)

    def stacked(self) -> tuple[torch.Tensor, torch.Tensor]:
        expected = {
            f"depth/{depth}": 1 for depth in self.patch_depths
        } | {"readout": 1}
        if dict(self.counts) != expected:
            raise RuntimeError(
                f"clean hook count drift: {dict(self.counts)}"
            )
        if self.readout is None:
            raise RuntimeError("clean readout missing")
        return (
            torch.stack(
                [self.values[depth] for depth in self.patch_depths],
                dim=2,
            ).to("cpu"),
            self.readout.to("cpu"),
        )

    def close(self) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles = []


class SinglePatchCapture:
    def __init__(
        self,
        layers,
        patch_depth: int,
        readout_depth: int,
    ):
        self.layers = layers
        self.patch_depth = patch_depth
        self.readout_depth = readout_depth
        self.patch_positions: torch.Tensor | None = None
        self.pre_positions: torch.Tensor | None = None
        self.replacement: torch.Tensor | None = None
        self.readout: torch.Tensor | None = None
        self.counts = defaultdict(int)
        self.handles = []

    def _patch_hook(self, module, args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if self.patch_positions is None or self.replacement is None:
            raise RuntimeError("patch context missing")
        patched = hidden.clone()
        positions = self.patch_positions.to(hidden.device)
        batch = torch.arange(hidden.shape[0], device=hidden.device)
        patched[batch, positions, :] = self.replacement.to(
            hidden.device, dtype=hidden.dtype
        )
        self.counts["patch"] += 1
        return replace_output(output, patched)

    def _readout_hook(self, module, args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if self.pre_positions is None:
            raise RuntimeError("readout context missing")
        positions = self.pre_positions.to(hidden.device)
        batch = torch.arange(hidden.shape[0], device=hidden.device)
        self.readout = hidden[batch, positions, :].detach()
        self.counts["readout"] += 1
        return output

    def register(self) -> None:
        self.handles = [
            self.layers[
                self.patch_depth - 1
            ].register_forward_hook(self._patch_hook),
            self.layers[
                self.readout_depth - 1
            ].register_forward_hook(self._readout_hook),
        ]

    def begin(
        self,
        *,
        patch_positions: torch.Tensor,
        pre_positions: torch.Tensor,
        replacement: torch.Tensor,
    ) -> None:
        self.patch_positions = patch_positions
        self.pre_positions = pre_positions
        self.replacement = replacement
        self.readout = None
        self.counts = defaultdict(int)

    def value(self) -> torch.Tensor:
        if dict(self.counts) != {"patch": 1, "readout": 1}:
            raise RuntimeError(
                f"intervention hook drift: {dict(self.counts)}"
            )
        if self.readout is None:
            raise RuntimeError("intervention readout missing")
        return self.readout.to("cpu")

    def close(self) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles = []


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(norms, EPS)


def prototype_map(
    clean_readout: np.ndarray,
    cases: list[dict[str, Any]],
) -> dict[tuple[str, int], np.ndarray]:
    result = {}
    for split in protocol.SPLITS:
        for held_surface in range(protocol.SURFACE_COUNT):
            values = np.empty(
                (protocol.TARGET_COUNT, clean_readout.shape[-1]),
                dtype=np.float32,
            )
            for target_index in range(protocol.TARGET_COUNT):
                indices = [
                    int(row["case_index"])
                    for row in cases
                    if row["split"] == split
                    and int(row["target_index"]) == target_index
                    and int(row["surface_index"]) != held_surface
                ]
                values[target_index] = np.asarray(
                    clean_readout[indices], dtype=np.float32
                ).mean(axis=0)
            result[(split, held_surface)] = normalize_rows(values)
    return result


def clean_concept_metrics(
    values: np.ndarray,
    cases: list[dict[str, Any]],
    split: str,
) -> dict[str, Any]:
    rows = [row for row in cases if row["split"] == split]
    state = np.asarray(
        [values[int(row["case_index"])] for row in rows],
        dtype=np.float32,
    )
    state = state - state.mean(axis=0, keepdims=True)
    hits = []
    margins = []
    for row, current in zip(rows, state):
        held_surface = int(row["surface_index"])
        target = int(row["target_index"])
        prototype = []
        for concept in range(protocol.TARGET_COUNT):
            members = [
                state[index]
                for index, candidate in enumerate(rows)
                if int(candidate["target_index"]) == concept
                and int(candidate["surface_index"]) != held_surface
            ]
            prototype.append(np.mean(members, axis=0))
        similarity = (
            normalize_rows(current[None, :])[0]
            @ normalize_rows(np.asarray(prototype)).T
        )
        wrong = np.delete(similarity, target)
        hits.append(int(np.argmax(similarity) == target))
        margins.append(float(similarity[target] - np.max(wrong)))
    return {
        "case_count": len(rows),
        "concept_top1": float(np.mean(hits)),
        "true_vs_wrong_margin": float(np.mean(margins)),
        "chance": 1.0 / protocol.TARGET_COUNT,
    }


def clean_readout_metrics(
    clean_readout: np.ndarray,
    cases: list[dict[str, Any]],
    prototypes: dict[tuple[str, int], np.ndarray],
    split: str,
) -> dict[str, Any]:
    hits = []
    margins = []
    for row in cases:
        if row["split"] != split:
            continue
        index = int(row["case_index"])
        target = int(row["target_index"])
        proto = prototypes[
            (split, int(row["surface_index"]))
        ]
        similarity = normalize_rows(
            clean_readout[index:index + 1]
        )[0] @ proto.T
        wrong = np.delete(similarity, target)
        hits.append(int(np.argmax(similarity) == target))
        margins.append(float(similarity[target] - np.max(wrong)))
    return {
        "case_count": len(hits),
        "target_top1": float(np.mean(hits)),
        "true_vs_wrong_margin": float(np.mean(margins)),
        "chance": 1.0 / protocol.TARGET_COUNT,
    }


def transport_metrics(
    values: np.ndarray,
    clean_readout: np.ndarray,
    pairs: list[dict[str, Any]],
    prototypes: dict[tuple[str, int], np.ndarray],
    split: str,
) -> dict[str, Any]:
    split_pairs = [row for row in pairs if row["split"] == split]
    donor_hits = []
    target_hits = []
    margins = []
    shifts = []
    for local_index, pair in enumerate(split_pairs):
        target_case = int(pair["target_case_index"])
        target = int(pair["target_index"])
        donor = int(pair["donor_index"])
        proto = prototypes[
            (split, int(pair["surface_index"]))
        ]
        clean_similarity = normalize_rows(
            clean_readout[target_case:target_case + 1]
        )[0] @ proto.T
        similarity = normalize_rows(
            values[local_index:local_index + 1]
        )[0] @ proto.T
        donor_hits.append(int(np.argmax(similarity) == donor))
        target_hits.append(int(np.argmax(similarity) == target))
        margin = float(similarity[donor] - similarity[target])
        clean_margin = float(
            clean_similarity[donor] - clean_similarity[target]
        )
        margins.append(margin)
        shifts.append(margin - clean_margin)
    return {
        "pair_count": len(split_pairs),
        "donor_top1": float(np.mean(donor_hits)),
        "target_top1": float(np.mean(target_hits)),
        "donor_vs_target_margin": float(np.mean(margins)),
        "donor_vs_target_margin_shift_from_clean": float(
            np.mean(shifts)
        ),
        "chance": 1.0 / protocol.TARGET_COUNT,
    }


def select_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = {}
    for role in protocol.ROLES:
        candidates = [row for row in rows if row["role"] == role]
        best = max(
            candidates,
            key=lambda row: (
                row["metrics"]["donor_top1"],
                row["metrics"][
                    "donor_vs_target_margin_shift_from_clean"
                ],
                -int(row["depth"]),
            ),
        )
        selected[(best["role"], int(best["depth"]))] = best
    for row in sorted(
        rows,
        key=lambda value: (
            value["metrics"]["donor_top1"],
            value["metrics"][
                "donor_vs_target_margin_shift_from_clean"
            ],
            -int(value["depth"]),
        ),
        reverse=True,
    )[:3]:
        selected[(row["role"], int(row["depth"]))] = row
    return [
        {
            "candidate_index": index,
            "role": role,
            "depth": depth,
            "discovery_metrics": selected[(role, depth)]["metrics"],
            "selection_source": "discovery_only",
        }
        for index, (role, depth) in enumerate(sorted(
            selected,
            key=lambda key: (
                protocol.ROLES.index(key[0]),
                key[1],
            ),
        ))
    ]


def finite_audit(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    result = {}
    for name, values in arrays.items():
        total = int(np.prod(values.shape))
        nonfinite = int(np.count_nonzero(~np.isfinite(values)))
        result[name] = {
            "shape": list(values.shape),
            "value_count": total,
            "nonfinite_count": nonfinite,
            "all_finite": nonfinite == 0,
        }
    return {
        "arrays": result,
        "all_finite": all(row["all_finite"] for row in result.values()),
    }


def run_intervention(
    *,
    base_model,
    layers,
    depth: int,
    readout_depth: int,
    mode: str,
    role: str,
    pair_rows: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    clean_states: np.ndarray,
    depths: list[int],
    tokenizer,
    device: torch.device,
    model_name: str,
    output: np.ndarray,
) -> None:
    depth_index = depths.index(depth)
    role_index = ROLE_INDEX[role]
    patch_role = (
        protocol.WRONG_ROLE[role]
        if mode == "matched_wrong_position"
        else role
    )
    patch_role_index = ROLE_INDEX[patch_role]
    capture = SinglePatchCapture(layers, depth, readout_depth)
    capture.register()
    try:
        offset = 0
        for batch_index, pair_batch in enumerate(
            chunks(pair_rows, BATCH_SIZE[model_name]), 1
        ):
            case_batch = [
                cases[int(pair["target_case_index"])]
                for pair in pair_batch
            ]
            input_ids, attention_mask, positions = make_batch(
                case_batch,
                pad_token_id=tokenizer.pad_token_id,
                device=device,
            )
            donor_field = (
                "scrambled_case_index"
                if mode == "scrambled_concept"
                else "donor_case_index"
            )
            donor_indices = [
                int(pair[donor_field]) for pair in pair_batch
            ]
            replacement = torch.from_numpy(
                np.asarray(
                    clean_states[
                        donor_indices,
                        role_index,
                        depth_index,
                        :,
                    ]
                ).copy()
            )
            capture.begin(
                patch_positions=positions[:, patch_role_index],
                pre_positions=positions[:, ROLE_INDEX["pre_output"]],
                replacement=replacement,
            )
            with torch.inference_mode():
                base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
            value = capture.value().numpy().astype(
                np.float16, copy=False
            )
            output[offset:offset + len(pair_batch)] = value
            offset += len(pair_batch)
            if batch_index % 28 == 0:
                print(
                    f"[phase1028] {model_name} {mode} "
                    f"{role}@{depth} pairs={offset}/{len(pair_rows)}",
                    flush=True,
                )
    finally:
        capture.close()
    output.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODELS)
    args = parser.parse_args()

    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    cases = protocol.read_jsonl(
        protocol.OUT_ROOT / "protocol" / f"cases.{args.model}.jsonl"
    )
    pairs = protocol.read_jsonl(
        protocol.OUT_ROOT / "protocol" / "pairs.jsonl"
    )
    discovery_pairs = [
        row for row in pairs if row["split"] == "discovery"
    ]
    confirmation_pairs = [
        row for row in pairs if row["split"] == "confirmation"
    ]
    depths = [
        int(value) for value in prereg["patch_depths"][args.model]
    ]
    readout_depth = int(prereg["readout_depth"][args.model])
    started = time.time()
    model = tokenizer = None
    atlas_dir = protocol.OUT_ROOT / "atlas" / args.model
    atlas_dir.mkdir(parents=True, exist_ok=True)
    try:
        model, tokenizer, device, placement = load_fp16(args.model)
        precision_audit = quantization_audit(model)
        if (
            precision_audit["has_quantized_modules"]
            or precision_audit["has_bf16_parameters"]
            or not precision_audit["has_fp16_parameters"]
        ):
            raise RuntimeError("FP16/no-quantization audit failed")
        layers = get_layers(model)
        info = get_model_info(model, args.model)
        if max(depths) >= readout_depth:
            raise RuntimeError("patch depth must precede readout depth")
        base_model = model.model

        clean_states = np.lib.format.open_memmap(
            atlas_dir / "clean_role_states.fp16.npy",
            mode="w+",
            dtype=np.float16,
            shape=(
                len(cases),
                len(protocol.ROLES),
                len(depths),
                info.d_model,
            ),
        )
        clean_readout = np.lib.format.open_memmap(
            atlas_dir / "clean_readout.fp16.npy",
            mode="w+",
            dtype=np.float16,
            shape=(len(cases), info.d_model),
        )
        clean_capture = MultiDepthCleanCapture(
            layers, depths, readout_depth
        )
        clean_capture.register()
        try:
            offset = 0
            for batch in chunks(cases, BATCH_SIZE[args.model]):
                input_ids, attention_mask, positions = make_batch(
                    batch,
                    pad_token_id=tokenizer.pad_token_id,
                    device=device,
                )
                clean_capture.begin(positions)
                with torch.inference_mode():
                    base_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                    )
                states, readout = clean_capture.stacked()
                clean_states[offset:offset + len(batch)] = (
                    states.numpy().astype(np.float16, copy=False)
                )
                clean_readout[offset:offset + len(batch)] = (
                    readout.numpy().astype(np.float16, copy=False)
                )
                offset += len(batch)
        finally:
            clean_capture.close()
        clean_states.flush()
        clean_readout.flush()

        prototypes = prototype_map(clean_readout, cases)
        observational = []
        for role_index, role in enumerate(protocol.ROLES):
            for depth_index, depth in enumerate(depths):
                observational.append({
                    "role": role,
                    "depth": depth,
                    "discovery": clean_concept_metrics(
                        np.asarray(
                            clean_states[:, role_index, depth_index, :]
                        ),
                        cases,
                        "discovery",
                    ),
                    "confirmation": clean_concept_metrics(
                        np.asarray(
                            clean_states[:, role_index, depth_index, :]
                        ),
                        cases,
                        "confirmation",
                    ),
                })

        candidate_keys = [
            (role, depth)
            for role in protocol.ROLES
            for depth in depths
        ]
        discovery_output = np.lib.format.open_memmap(
            atlas_dir / "discovery_matched.fp16.npy",
            mode="w+",
            dtype=np.float16,
            shape=(
                len(candidate_keys),
                len(discovery_pairs),
                info.d_model,
            ),
        )
        discovery_rows = []
        for candidate_index, (role, depth) in enumerate(candidate_keys):
            run_intervention(
                base_model=base_model,
                layers=layers,
                depth=depth,
                readout_depth=readout_depth,
                mode="matched",
                role=role,
                pair_rows=discovery_pairs,
                cases=cases,
                clean_states=clean_states,
                depths=depths,
                tokenizer=tokenizer,
                device=device,
                model_name=args.model,
                output=discovery_output[candidate_index],
            )
            discovery_rows.append({
                "role": role,
                "depth": depth,
                "metrics": transport_metrics(
                    np.asarray(discovery_output[candidate_index]),
                    clean_readout,
                    discovery_pairs,
                    prototypes,
                    "discovery",
                ),
            })
        discovery_output.flush()
        selected = select_candidates(discovery_rows)
        protocol.write_json(
            atlas_dir / "selection.json",
            {
                "schema_version": "phase1028_selection.v1",
                "selection_source": "discovery_only",
                "policy": prereg["discovery_selection"],
                "selected": selected,
            },
        )

        confirmation_output = np.lib.format.open_memmap(
            atlas_dir / "confirmation_controls.fp16.npy",
            mode="w+",
            dtype=np.float16,
            shape=(
                len(selected),
                len(protocol.CONFIRMATION_MODES),
                len(confirmation_pairs),
                info.d_model,
            ),
        )
        confirmation_rows = []
        for selected_index, candidate in enumerate(selected):
            mode_metrics = {}
            for mode_index, mode in enumerate(
                protocol.CONFIRMATION_MODES
            ):
                run_intervention(
                    base_model=base_model,
                    layers=layers,
                    depth=int(candidate["depth"]),
                    readout_depth=readout_depth,
                    mode=mode,
                    role=candidate["role"],
                    pair_rows=confirmation_pairs,
                    cases=cases,
                    clean_states=clean_states,
                    depths=depths,
                    tokenizer=tokenizer,
                    device=device,
                    model_name=args.model,
                    output=confirmation_output[
                        selected_index, mode_index
                    ],
                )
                mode_metrics[mode] = transport_metrics(
                    np.asarray(
                        confirmation_output[selected_index, mode_index]
                    ),
                    clean_readout,
                    confirmation_pairs,
                    prototypes,
                    "confirmation",
                )
            confirmation_rows.append({
                "candidate_index": selected_index,
                "role": candidate["role"],
                "depth": candidate["depth"],
                "wrong_role": protocol.WRONG_ROLE[candidate["role"]],
                "discovery_metrics": candidate["discovery_metrics"],
                "confirmation": mode_metrics,
            })
        confirmation_output.flush()

        metrics = {
            "schema_version": "phase1028_metrics.v1",
            "model": args.model,
            "depths": depths,
            "readout_depth": readout_depth,
            "clean_readout": {
                split: clean_readout_metrics(
                    clean_readout, cases, prototypes, split
                )
                for split in protocol.SPLITS
            },
            "observational_role_depth": observational,
            "discovery_causal_map": discovery_rows,
            "confirmation_candidates": confirmation_rows,
        }
        protocol.write_json(atlas_dir / "metrics.json", metrics)
        arrays = {
            "clean_role_states": clean_states,
            "clean_readout": clean_readout,
            "discovery_matched": discovery_output,
            "confirmation_controls": confirmation_output,
        }
        summary = {
            "schema_version": "phase1028_model_summary.v1",
            "phase": protocol.PHASE,
            "protocol_digest": prereg["protocol_digest"],
            "model": args.model,
            "precision": "fp16",
            "quantization": "none",
            "placement": placement,
            "runtime_precision_audit": precision_audit,
            "patch_depths": depths,
            "readout_depth": readout_depth,
            "selection_source": "discovery_only",
            "selected_candidate_count": len(selected),
            "finiteness": finite_audit(arrays),
            "elapsed_seconds": time.time() - started,
            "claim_limit": prereg["claim_limit"],
        }
        protocol.write_json(atlas_dir / "summary.json", summary)
        print(json.dumps({
            "summary": summary,
            "clean_readout": metrics["clean_readout"],
            "selected": selected,
            "confirmation": confirmation_rows,
        }, ensure_ascii=False, indent=2))
    finally:
        if model is not None:
            release_fp16(model)
        del tokenizer


if __name__ == "__main__":
    main()
