#!/usr/bin/env python3
"""Run Phase1027 local internal-state transport in three FP16 models."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import get_layers, get_model_info
import phase1027_binding_transport_protocol as protocol
from phase1023_fp16_utils import (
    MODELS,
    load_fp16,
    quantization_audit,
    release_fp16,
)


BATCH_SIZE = {"qwen3": 32, "glm4": 8, "deepseek7b": 8}
EPS = 1e-8


def chunks(rows: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(rows), size):
        yield rows[start:start + size]


def make_batch(
    rows: list[dict[str, Any]],
    *,
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(row["input_ids"]) for row in rows)
    ids = torch.full(
        (len(rows), width),
        int(pad_token_id),
        dtype=torch.long,
    )
    mask = torch.zeros((len(rows), width), dtype=torch.long)
    for index, row in enumerate(rows):
        value = torch.tensor(row["input_ids"], dtype=torch.long)
        ids[index, :len(value)] = value
        mask[index, :len(value)] = 1
    return ids.to(device), mask.to(device)


def replace_output(output, hidden: torch.Tensor):
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    return hidden


class CleanCapture:
    def __init__(
        self,
        layers,
        patch_depth: int,
        readout_depth: int,
    ):
        self.layers = layers
        self.patch_depth = patch_depth
        self.readout_depth = readout_depth
        self.focus_positions: torch.Tensor | None = None
        self.pre_positions: torch.Tensor | None = None
        self.patch_value: torch.Tensor | None = None
        self.readout_value: torch.Tensor | None = None
        self.handles = []

    def _patch_capture(self, module, args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if self.focus_positions is None:
            raise RuntimeError("missing clean focus positions")
        positions = self.focus_positions.to(hidden.device)
        batch = torch.arange(hidden.shape[0], device=hidden.device)
        self.patch_value = hidden[batch, positions, :].detach()
        return output

    def _readout_capture(self, module, args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if self.pre_positions is None:
            raise RuntimeError("missing clean readout positions")
        positions = self.pre_positions.to(hidden.device)
        batch = torch.arange(hidden.shape[0], device=hidden.device)
        self.readout_value = hidden[batch, positions, :].detach()
        return output

    def register(self) -> None:
        self.handles = [
            self.layers[
                self.patch_depth - 1
            ].register_forward_hook(self._patch_capture),
            self.layers[
                self.readout_depth - 1
            ].register_forward_hook(self._readout_capture),
        ]

    def begin(
        self,
        focus_positions: torch.Tensor,
        pre_positions: torch.Tensor,
    ) -> None:
        self.focus_positions = focus_positions
        self.pre_positions = pre_positions
        self.patch_value = None
        self.readout_value = None

    def values(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.patch_value is None or self.readout_value is None:
            raise RuntimeError("clean capture incomplete")
        return self.patch_value.to("cpu"), self.readout_value.to("cpu")

    def close(self) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles = []


class InterventionCapture:
    def __init__(
        self,
        layers,
        patch_depth: int,
        readout_depth: int,
    ):
        self.layers = layers
        self.patch_depth = patch_depth
        self.readout_depth = readout_depth
        self.mode = ""
        self.focus_positions: torch.Tensor | None = None
        self.pre_positions: torch.Tensor | None = None
        self.replacement: torch.Tensor | None = None
        self.delta: torch.Tensor | None = None
        self.readout_value: torch.Tensor | None = None
        self.handles = []

    def _patch(self, module, args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        patched = hidden.clone()
        batch = torch.arange(hidden.shape[0], device=hidden.device)
        if self.mode in {"matched_focus", "scrambled_focus"}:
            if self.focus_positions is None or self.replacement is None:
                raise RuntimeError("missing focus intervention state")
            positions = self.focus_positions.to(hidden.device)
            patched[batch, positions, :] = self.replacement.to(
                hidden.device, dtype=hidden.dtype
            )
        elif self.mode == "matched_bos_delta":
            if self.delta is None:
                raise RuntimeError("missing BOS intervention delta")
            patched[:, 0, :] = (
                patched[:, 0, :]
                + self.delta.to(hidden.device, dtype=hidden.dtype)
            )
        else:
            raise RuntimeError(f"unknown intervention {self.mode}")
        return replace_output(output, patched)

    def _capture(self, module, args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if self.pre_positions is None:
            raise RuntimeError("missing intervention readout positions")
        positions = self.pre_positions.to(hidden.device)
        batch = torch.arange(hidden.shape[0], device=hidden.device)
        self.readout_value = hidden[batch, positions, :].detach()
        return output

    def register(self) -> None:
        self.handles = [
            self.layers[
                self.patch_depth - 1
            ].register_forward_hook(self._patch),
            self.layers[
                self.readout_depth - 1
            ].register_forward_hook(self._capture),
        ]

    def begin(
        self,
        *,
        mode: str,
        focus_positions: torch.Tensor,
        pre_positions: torch.Tensor,
        target_state: torch.Tensor,
        donor_state: torch.Tensor,
    ) -> None:
        self.mode = mode
        self.focus_positions = focus_positions
        self.pre_positions = pre_positions
        self.readout_value = None
        if mode in {"matched_focus", "scrambled_focus"}:
            self.replacement = donor_state
            self.delta = None
        elif mode == "matched_bos_delta":
            self.replacement = None
            self.delta = donor_state - target_state
        else:
            raise RuntimeError(mode)

    def value(self) -> torch.Tensor:
        if self.readout_value is None:
            raise RuntimeError("intervention capture incomplete")
        return self.readout_value.to("cpu")

    def close(self) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles = []


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(norms, EPS)


def prototypes(
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


def pair_metrics(
    values: np.ndarray,
    clean_readout: np.ndarray,
    pairs: list[dict[str, Any]],
    prototype_map: dict[tuple[str, int], np.ndarray],
    split: str,
) -> dict[str, Any]:
    donor_hits = []
    target_hits = []
    donor_margins = []
    margin_shifts = []
    for pair in pairs:
        if pair["split"] != split:
            continue
        index = int(pair["pair_index"])
        target_case = int(pair["target_case_index"])
        target_index = int(pair["target_index"])
        donor_index = int(pair["donor_index"])
        proto = prototype_map[
            (split, int(pair["surface_index"]))
        ]
        clean_similarity = normalize_rows(
            clean_readout[target_case:target_case + 1]
        )[0] @ proto.T
        similarity = normalize_rows(values[index:index + 1])[0] @ proto.T
        donor_hits.append(int(np.argmax(similarity) == donor_index))
        target_hits.append(int(np.argmax(similarity) == target_index))
        margin = float(
            similarity[donor_index] - similarity[target_index]
        )
        clean_margin = float(
            clean_similarity[donor_index]
            - clean_similarity[target_index]
        )
        donor_margins.append(margin)
        margin_shifts.append(margin - clean_margin)
    return {
        "pair_count": len(donor_hits),
        "donor_top1": float(np.mean(donor_hits)),
        "target_top1": float(np.mean(target_hits)),
        "donor_vs_target_margin": float(np.mean(donor_margins)),
        "donor_vs_target_margin_shift_from_clean": float(
            np.mean(margin_shifts)
        ),
        "chance": 1.0 / protocol.TARGET_COUNT,
    }


def clean_metrics(
    clean_readout: np.ndarray,
    cases: list[dict[str, Any]],
    prototype_map: dict[tuple[str, int], np.ndarray],
    split: str,
) -> dict[str, Any]:
    hits = []
    margins = []
    for row in cases:
        if row["split"] != split:
            continue
        index = int(row["case_index"])
        target = int(row["target_index"])
        proto = prototype_map[
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
    patch_depth = int(
        prereg["patch_depth_frozen_from_phase1026"][args.model]
    )
    readout_depth = int(
        prereg["readout_depth_frozen_from_prior_finite_atlas"][args.model]
    )
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
        base_model = model.model

        clean_patch = np.lib.format.open_memmap(
            atlas_dir / "clean_patch.fp16.npy",
            mode="w+",
            dtype=np.float16,
            shape=(len(cases), info.d_model),
        )
        clean_readout = np.lib.format.open_memmap(
            atlas_dir / "clean_readout.fp16.npy",
            mode="w+",
            dtype=np.float16,
            shape=(len(cases), info.d_model),
        )
        clean_capture = CleanCapture(
            layers,
            patch_depth,
            readout_depth,
        )
        clean_capture.register()
        try:
            offset = 0
            for batch in chunks(cases, BATCH_SIZE[args.model]):
                input_ids, attention_mask = make_batch(
                    batch,
                    pad_token_id=tokenizer.pad_token_id,
                    device=device,
                )
                focus = torch.tensor([
                    row["role_positions"]["focus_end"] for row in batch
                ], dtype=torch.long)
                pre = torch.tensor([
                    row["role_positions"]["pre_output"] for row in batch
                ], dtype=torch.long)
                clean_capture.begin(focus, pre)
                with torch.inference_mode():
                    base_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                    )
                patch_value, readout_value = clean_capture.values()
                clean_patch[offset:offset + len(batch)] = (
                    patch_value.numpy().astype(np.float16, copy=False)
                )
                clean_readout[offset:offset + len(batch)] = (
                    readout_value.numpy().astype(np.float16, copy=False)
                )
                offset += len(batch)
        finally:
            clean_capture.close()
        clean_patch.flush()
        clean_readout.flush()

        intervention_arrays = {}
        capture = InterventionCapture(
            layers,
            patch_depth,
            readout_depth,
        )
        capture.register()
        try:
            for mode in protocol.INTERVENTIONS:
                output = np.lib.format.open_memmap(
                    atlas_dir / f"{mode}.fp16.npy",
                    mode="w+",
                    dtype=np.float16,
                    shape=(len(pairs), info.d_model),
                )
                offset = 0
                for batch_index, pair_batch in enumerate(
                    chunks(pairs, BATCH_SIZE[args.model]), 1
                ):
                    case_batch = [
                        cases[int(pair["target_case_index"])]
                        for pair in pair_batch
                    ]
                    input_ids, attention_mask = make_batch(
                        case_batch,
                        pad_token_id=tokenizer.pad_token_id,
                        device=device,
                    )
                    focus = torch.tensor([
                        row["role_positions"]["focus_end"]
                        for row in case_batch
                    ], dtype=torch.long)
                    pre = torch.tensor([
                        row["role_positions"]["pre_output"]
                        for row in case_batch
                    ], dtype=torch.long)
                    target_indices = [
                        int(pair["target_case_index"])
                        for pair in pair_batch
                    ]
                    donor_field = (
                        "scrambled_case_index"
                        if mode == "scrambled_focus"
                        else "donor_case_index"
                    )
                    donor_indices = [
                        int(pair[donor_field]) for pair in pair_batch
                    ]
                    target_state = torch.from_numpy(
                        np.asarray(clean_patch[target_indices]).copy()
                    )
                    donor_state = torch.from_numpy(
                        np.asarray(clean_patch[donor_indices]).copy()
                    )
                    capture.begin(
                        mode=mode,
                        focus_positions=focus,
                        pre_positions=pre,
                        target_state=target_state,
                        donor_state=donor_state,
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
                            f"[phase1027] {args.model} {mode} "
                            f"pairs={offset}/{len(pairs)}",
                            flush=True,
                        )
                output.flush()
                intervention_arrays[mode] = output
        finally:
            capture.close()

        proto = prototypes(clean_readout, cases)
        metrics = {
            "schema_version": "phase1027_metrics.v1",
            "model": args.model,
            "patch_depth": patch_depth,
            "readout_depth": readout_depth,
            "clean": {
                split: clean_metrics(
                    clean_readout,
                    cases,
                    proto,
                    split,
                )
                for split in protocol.SPLITS
            },
            "interventions": {
                mode: {
                    split: pair_metrics(
                        values,
                        clean_readout,
                        pairs,
                        proto,
                        split,
                    )
                    for split in protocol.SPLITS
                }
                for mode, values in intervention_arrays.items()
            },
        }
        protocol.write_json(atlas_dir / "metrics.json", metrics)
        arrays = {
            "clean_patch": clean_patch,
            "clean_readout": clean_readout,
            **intervention_arrays,
        }
        summary = {
            "schema_version": "phase1027_model_summary.v1",
            "phase": protocol.PHASE,
            "protocol_digest": prereg["protocol_digest"],
            "model": args.model,
            "precision": "fp16",
            "quantization": "none",
            "placement": placement,
            "runtime_precision_audit": precision_audit,
            "patch_depth": patch_depth,
            "readout_depth": readout_depth,
            "selection_source": "phase1026_and_prior_finite_atlas",
            "finiteness": finite_audit(arrays),
            "clean_case_count": len(cases),
            "pair_count": len(pairs),
            "intervention_count": len(protocol.INTERVENTIONS),
            "elapsed_seconds": time.time() - started,
            "claim_limit": prereg["claim_limit"],
        }
        protocol.write_json(atlas_dir / "summary.json", summary)
        print(json.dumps({
            "summary": summary,
            "metrics": metrics,
        }, ensure_ascii=False, indent=2))
    finally:
        if model is not None:
            release_fp16(model)
        del tokenizer


if __name__ == "__main__":
    main()
