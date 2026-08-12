#!/usr/bin/env python3
"""Run the Phase1042 role-by-depth actual-write atlas."""

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
import phase1037_family_source_causal_scan as metric_tools
import phase1042_role_depth_write_atlas_protocol as protocol


TARGET_BATCH_SIZE = {"qwen3": 8, "glm4": 1, "deepseek7b": 2}
EPS = 1e-8


def chunks(rows: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(rows), size):
        yield rows[start:start + size]


def output_tensor(output: Any) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


def scalar_summary(values: np.ndarray) -> dict[str, Any]:
    return metric_tools.scalar_summary(np.asarray(values, dtype=np.float32))


def finite_summary(values: np.ndarray) -> dict[str, Any]:
    current = np.asarray(values)
    finite = np.isfinite(current)
    return {
        "all_finite": bool(np.all(finite)),
        "finite_value_rate": float(np.mean(finite)),
        "nonfinite_value_count": int(np.sum(~finite)),
    }


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32).reshape(len(a), -1)
    b = np.asarray(b, dtype=np.float32).reshape(len(b), -1)
    numerator = np.sum(a * b, axis=-1)
    denominator = (
        np.linalg.norm(a, axis=-1)
        * np.linalg.norm(b, axis=-1)
    )
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=np.float32),
        where=denominator > EPS,
    )


def vector_norms(values: np.ndarray) -> np.ndarray:
    current = np.asarray(values, dtype=np.float32)
    return np.linalg.norm(current.reshape(len(current), -1), axis=-1)


def make_batch(
    target_rows: list[dict[str, Any]],
    cases: dict[int, dict[str, Any]],
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
    flat_rows = [
        cases[int(target["world_case_indices"][world])]
        for target in target_rows
        for world in protocol.WORLD_ORDER
    ]
    width = max(len(row["input_ids"]) for row in flat_rows)
    ids = torch.full(
        (len(flat_rows), width), int(pad_token_id), dtype=torch.long
    )
    attention_mask = torch.zeros(
        (len(flat_rows), width), dtype=torch.long
    )
    positions = torch.zeros(
        (
            len(target_rows),
            len(protocol.WORLD_ORDER),
            len(protocol.SEMANTIC_SITES),
            protocol.MAX_SPAN,
        ),
        dtype=torch.long,
    )
    masks = torch.zeros_like(positions, dtype=torch.bool)
    pre_positions = torch.empty(len(flat_rows), dtype=torch.long)
    for target_slot, target in enumerate(target_rows):
        for world_slot, world in enumerate(protocol.WORLD_ORDER):
            row = cases[int(target["world_case_indices"][world])]
            flat_index = (
                target_slot * len(protocol.WORLD_ORDER) + world_slot
            )
            values = torch.tensor(row["input_ids"], dtype=torch.long)
            ids[flat_index, :len(values)] = values
            attention_mask[flat_index, :len(values)] = 1
            for site_slot, site in enumerate(
                protocol.SEMANTIC_SITES
            ):
                current_role = protocol.semantic_role(site, target)
                start, end = (
                    int(value)
                    for value in row["anchor_spans"][current_role]
                )
                span = list(range(start, end + 1))
                positions[
                    target_slot,
                    world_slot,
                    site_slot,
                    :len(span),
                ] = torch.tensor(span, dtype=torch.long)
                masks[
                    target_slot,
                    world_slot,
                    site_slot,
                    :len(span),
                ] = True
            pre_positions[flat_index] = int(
                row["anchor_spans"]["pre_output"][1]
            )
    return (
        ids.to(device),
        attention_mask.to(device),
        positions,
        masks,
        pre_positions,
        np.asarray(
            [int(row["atlas_index"]) for row in target_rows],
            dtype=np.int64,
        ),
    )


class AtlasCapture:
    def __init__(
        self,
        layers: list[Any],
        selected_depths: list[int],
        contrasts: np.memmap,
        lexical_norms: np.memmap,
        closure: np.memmap,
    ):
        self.layers = layers
        self.selected_depths = selected_depths
        self.depth_slots = {
            depth: slot
            for slot, depth in enumerate(selected_depths)
        }
        self.contrasts = contrasts
        self.lexical_norms = lexical_norms
        self.closure = closure
        self.positions: torch.Tensor | None = None
        self.masks: torch.Tensor | None = None
        self.atlas_indices: np.ndarray | None = None
        self.current: dict[int, dict[str, torch.Tensor]] = {}
        self.counts: dict[str, int] = defaultdict(int)
        self.handles = []

    def begin(
        self,
        positions: torch.Tensor,
        masks: torch.Tensor,
        atlas_indices: np.ndarray,
    ) -> None:
        self.positions = positions
        self.masks = masks
        self.atlas_indices = atlas_indices
        self.current = {}
        self.counts = defaultdict(int)

    def _states(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.positions is None or self.masks is None:
            raise RuntimeError("atlas positions missing")
        target_count = self.positions.shape[0]
        positions = self.positions.reshape(
            -1,
            len(protocol.SEMANTIC_SITES),
            protocol.MAX_SPAN,
        ).to(hidden.device)
        masks = self.masks.reshape_as(positions).to(hidden.device)
        batch = torch.arange(hidden.shape[0], device=hidden.device)
        batch = batch[:, None, None].expand_as(positions)
        values = hidden[batch, positions, :].clone()
        values = values.masked_fill(~masks[..., None], 0)
        return values.reshape(
            target_count,
            len(protocol.WORLD_ORDER),
            len(protocol.SEMANTIC_SITES),
            protocol.MAX_SPAN,
            hidden.shape[-1],
        ).detach()

    def _pre_hook(self, depth: int):
        def hook(module, args):
            self.current[depth] = {
                "upstream_residual": self._states(args[0])
            }
            self.counts[f"{depth}/pre"] += 1
        return hook

    def _component_hook(self, depth: int, name: str):
        def hook(module, args, output):
            self.current[depth][name] = self._states(
                output_tensor(output)
            )
            self.counts[f"{depth}/{name}"] += 1
            return output
        return hook

    def _layer_hook(self, depth: int):
        def hook(module, args, output):
            if self.atlas_indices is None or self.masks is None:
                raise RuntimeError("atlas context missing")
            current = self.current[depth]
            current["layer_output"] = self._states(
                output_tensor(output)
            )
            depth_slot = self.depth_slots[depth]
            signs = torch.tensor(
                [
                    protocol.SITE_SIGNS[site]
                    for site in protocol.SEMANTIC_SITES
                ],
                dtype=torch.float32,
                device=output_tensor(output).device,
            )[None, :, None, None]
            for channel_slot, channel in enumerate(
                protocol.CHANNELS
            ):
                states = current[channel]
                d0 = (states[:, 1] - states[:, 0]) * signs
                d1 = (states[:, 3] - states[:, 2]) * signs
                values = torch.stack((d0, d1), dim=2)
                self.contrasts[
                    self.atlas_indices,
                    depth_slot,
                    channel_slot,
                    :,
                    :,
                    :,
                    :,
                ] = values.to(
                    "cpu", dtype=torch.float16
                ).numpy()
                lexical = states[:, 2] - states[:, 0]
                self.lexical_norms[
                    self.atlas_indices,
                    depth_slot,
                    channel_slot,
                    :,
                ] = torch.linalg.vector_norm(
                    lexical.float().flatten(start_dim=-2), dim=-1
                ).cpu().numpy()

            output_states = current["layer_output"]
            accounted = (
                current["upstream_residual"]
                + current["attention_write"]
                + current["mlp_write"]
            )
            error = torch.linalg.vector_norm(
                (output_states - accounted).float(), dim=-1
            )
            transition = torch.linalg.vector_norm(
                (
                    output_states - current["upstream_residual"]
                ).float(),
                dim=-1,
            )
            relative = error / torch.clamp(transition, min=EPS)
            valid = self.masks[:, 0].to(relative.device)
            relative = relative[:, 0].masked_fill(~valid, torch.nan)
            self.closure[
                self.atlas_indices, depth_slot, :
            ] = torch.nanmean(relative, dim=-1).cpu().numpy()
            self.counts[f"{depth}/layer"] += 1
            return output
        return hook

    def register(self) -> None:
        for depth in self.selected_depths:
            layer = self.layers[depth - 1]
            self.handles.append(
                layer.register_forward_pre_hook(self._pre_hook(depth))
            )
            self.handles.append(
                layer.self_attn.register_forward_hook(
                    self._component_hook(depth, "attention_write")
                )
            )
            self.handles.append(
                layer.mlp.register_forward_hook(
                    self._component_hook(depth, "mlp_write")
                )
            )
            self.handles.append(
                layer.register_forward_hook(self._layer_hook(depth))
            )

    def end(self) -> None:
        expected = {
            f"{depth}/{stage}": 1
            for depth in self.selected_depths
            for stage in (
                "pre",
                "attention_write",
                "mlp_write",
                "layer",
            )
        }
        if dict(self.counts) != expected:
            raise RuntimeError(
                f"atlas hook count drift: {dict(self.counts)}"
            )
        self.positions = None
        self.masks = None
        self.atlas_indices = None
        self.current = {}

    def close(self) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles = []


def groups(targets: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    result = {"all": np.arange(len(targets), dtype=np.int64)}
    for template in (0, 1):
        for stratum in protocol.material.SURFACE_STRATA:
            result[f"template_{template}/{stratum}"] = np.asarray([
                index for index, row in enumerate(targets)
                if int(row["template_index"]) == template
                and row["surface_stratum"] == stratum
            ], dtype=np.int64)
    return result


def atlas_metrics(
    contrasts: np.ndarray,
    lexical_norms: np.ndarray,
    targets: list[dict[str, Any]],
    selected_depths: list[int],
) -> list[dict[str, Any]]:
    shuffled = np.asarray(
        [int(row["shuffled_atlas_index"]) for row in targets],
        dtype=np.int64,
    )
    group_indices = groups(targets)
    rows = []
    for depth_slot, physical_depth in enumerate(selected_depths):
        for channel_slot, channel in enumerate(protocol.CHANNELS):
            for site_slot, site in enumerate(
                protocol.SEMANTIC_SITES
            ):
                d0 = np.asarray(
                    contrasts[
                        :, depth_slot, channel_slot, site_slot, 0
                    ],
                    dtype=np.float32,
                )
                d1 = np.asarray(
                    contrasts[
                        :, depth_slot, channel_slot, site_slot, 1
                    ],
                    dtype=np.float32,
                )
                matched = cosine_rows(d0, d1)
                shuffled_cosine = cosine_rows(d0, d1[shuffled])
                advantage = matched - shuffled_cosine
                family_norm = (
                    vector_norms(d0) + vector_norms(d1)
                ) / 2.0
                lexical = np.asarray(
                    lexical_norms[
                        :, depth_slot, channel_slot, site_slot
                    ],
                    dtype=np.float32,
                )
                ratio = np.divide(
                    family_norm,
                    lexical + EPS,
                    out=np.full_like(family_norm, np.nan),
                    where=np.isfinite(lexical),
                )
                current_groups = {}
                for group, indices in group_indices.items():
                    current_groups[group] = {
                        "same_pair_cosine": scalar_summary(
                            matched[indices]
                        ),
                        "shuffled_pair_cosine": scalar_summary(
                            shuffled_cosine[indices]
                        ),
                        "matched_minus_shuffled": scalar_summary(
                            advantage[indices]
                        ),
                        "family_contrast_norm": scalar_summary(
                            family_norm[indices]
                        ),
                        "same_family_lexical_norm": scalar_summary(
                            lexical[indices]
                        ),
                        "family_to_lexical_norm_ratio": scalar_summary(
                            ratio[indices]
                        ),
                    }
                rows.append({
                    "normalized_depth_slot": depth_slot + 1,
                    "physical_depth": physical_depth,
                    "channel": channel,
                    "site": site,
                    "groups": current_groups,
                })
    return rows


def behavior_metrics(
    logits: np.ndarray,
    targets: list[dict[str, Any]],
    cases: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    expected = np.asarray([
        [
            int(cases[int(row["world_case_indices"][world])][
                "expected_index"
            ])
            for world in protocol.WORLD_ORDER
        ]
        for row in targets
    ], dtype=np.int64)
    finite = np.all(np.isfinite(logits), axis=-1)
    prediction = np.argmax(
        np.where(np.isfinite(logits), logits, -np.inf), axis=-1
    )
    return {
        "row_count": int(logits.shape[0] * logits.shape[1]),
        "finite_rate": float(np.mean(finite)),
        "candidate_accuracy": float(np.mean(prediction == expected)),
    }


def run(model_name: str) -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    audit = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "audit.json"
    )
    if not all(audit["checks"].values()):
        raise RuntimeError("Phase1042 protocol audit failed")
    targets = protocol.read_jsonl(
        protocol.OUT_ROOT / "protocol" / "targets.jsonl"
    )
    cases_list = protocol.read_jsonl(
        protocol.OUT_ROOT
        / "protocol"
        / f"cases.{model_name}.jsonl"
    )
    cases = {
        int(row["case_index"]): row for row in cases_list
    }
    selected_depths = [
        int(value)
        for value in prereg["model_physical_depths"][model_name]
    ]
    atlas_dir = protocol.OUT_ROOT / "atlas" / model_name
    atlas_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    model = tokenizer = None

    try:
        model, tokenizer, device, placement = load_fp16(model_name)
        precision = quantization_audit(model)
        if (
            precision["has_quantized_modules"]
            or precision["has_bf16_parameters"]
            or not precision["has_fp16_parameters"]
        ):
            raise RuntimeError("FP16/no-quantization audit failed")
        layers = get_layers(model)
        info = get_model_info(model, model_name)
        pad_token_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )
        candidate_ids = torch.tensor(
            cases_list[0]["candidate_token_ids"], dtype=torch.long
        )
        contrasts = np.lib.format.open_memmap(
            atlas_dir / "family_contrasts.fp16.npy",
            mode="w+",
            dtype=np.float16,
            shape=(
                len(targets),
                len(selected_depths),
                len(protocol.CHANNELS),
                len(protocol.SEMANTIC_SITES),
                2,
                protocol.MAX_SPAN,
                info.d_model,
            ),
        )
        lexical_norms = np.lib.format.open_memmap(
            atlas_dir / "lexical_norms.fp32.npy",
            mode="w+",
            dtype=np.float32,
            shape=(
                len(targets),
                len(selected_depths),
                len(protocol.CHANNELS),
                len(protocol.SEMANTIC_SITES),
            ),
        )
        closure = np.lib.format.open_memmap(
            atlas_dir / "channel_closure.fp32.npy",
            mode="w+",
            dtype=np.float32,
            shape=(
                len(targets),
                len(selected_depths),
                len(protocol.SEMANTIC_SITES),
            ),
        )
        logits_out = np.lib.format.open_memmap(
            atlas_dir / "candidate_logits.fp32.npy",
            mode="w+",
            dtype=np.float32,
            shape=(
                len(targets),
                len(protocol.WORLD_ORDER),
                len(protocol.material.FAMILIES),
            ),
        )
        contrasts[:] = np.nan
        lexical_norms[:] = np.nan
        closure[:] = np.nan
        logits_out[:] = np.nan
        capture = AtlasCapture(
            layers,
            selected_depths,
            contrasts,
            lexical_norms,
            closure,
        )
        capture.register()
        try:
            for target_batch in chunks(
                targets, TARGET_BATCH_SIZE[model_name]
            ):
                (
                    input_ids,
                    attention_mask,
                    positions,
                    masks,
                    pre_positions,
                    atlas_indices,
                ) = make_batch(
                    target_batch,
                    cases,
                    pad_token_id=pad_token_id,
                    device=device,
                )
                capture.begin(positions, masks, atlas_indices)
                with torch.inference_mode():
                    output = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        return_dict=True,
                    )
                capture.end()
                logits = output.logits
                batch = torch.arange(
                    logits.shape[0], device=logits.device
                )
                selected = logits[
                    batch, pre_positions.to(logits.device), :
                ].float().index_select(
                    -1, candidate_ids.to(logits.device)
                )
                logits_out[atlas_indices] = (
                    selected.reshape(
                        len(target_batch),
                        len(protocol.WORLD_ORDER),
                        -1,
                    ).detach().cpu().numpy()
                )
                del output, logits, selected
        finally:
            capture.close()
        contrasts.flush()
        lexical_norms.flush()
        closure.flush()
        logits_out.flush()

        metrics = {
            "schema_version": "phase1042_model_metrics.v1",
            "phase": protocol.PHASE,
            "model": model_name,
            "behavior": behavior_metrics(
                logits_out, targets, cases
            ),
            "role_depth_rows": atlas_metrics(
                contrasts,
                lexical_norms,
                targets,
                selected_depths,
            ),
        }
        summary = {
            "schema_version": "phase1042_model_summary.v1",
            "phase": protocol.PHASE,
            "model": model_name,
            "protocol_digest": prereg["protocol_digest"],
            "precision": precision,
            "placement": placement,
            "model_info": {
                "class": model.__class__.__name__,
                "n_layers": int(info.n_layers),
                "d_model": int(info.d_model),
            },
            "selected_depths": selected_depths,
            "sample_counts": {
                "targets": len(targets),
                "cases": len(cases_list),
                "depths": len(selected_depths),
                "channels": len(protocol.CHANNELS),
                "semantic_sites": len(protocol.SEMANTIC_SITES),
            },
            "array_finiteness": {
                "family_contrasts": finite_summary(contrasts),
                "lexical_norms": finite_summary(lexical_norms),
                "channel_closure": finite_summary(closure),
                "candidate_logits": finite_summary(logits_out),
            },
            "instrumentation_closure": scalar_summary(closure),
            "elapsed_seconds": time.time() - started,
        }
        protocol.write_json(atlas_dir / "metrics.json", metrics)
        protocol.write_json(atlas_dir / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        if model is not None:
            release_fp16(model)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=protocol.MODELS)
    args = parser.parse_args()
    run(args.model)


if __name__ == "__main__":
    main()
