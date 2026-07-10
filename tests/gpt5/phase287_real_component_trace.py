#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch
from safetensors import safe_open


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

from hf_probe_env import get_layers, load_probe_model, release_loaded, vram_gb  # noqa: E402


PHASE = 287
RESULT_ROOT = ROOT / "tests" / "result" / "phase287_real_component_trace"
PUBLIC_ROOT = ROOT / "frontend" / "public" / "vis_data" / "real_component_trace"
COLOR_LABELS = ["black", "blue", "brown", "gray", "green", "orange", "purple", "red", "silver", "white", "yellow"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def first_token_id(tokenizer: Any, label: str) -> int:
    candidates: list[int] = []
    for text in (label, f" {label}"):
        ids = tokenizer(text, add_special_tokens=False).get("input_ids") or []
        if ids:
            candidates.append(int(ids[0]))
    if not candidates:
        raise ValueError(f"Could not tokenize label {label!r}")
    return candidates[-1]


def to_last_token(value: Any) -> torch.Tensor | None:
    if isinstance(value, tuple):
        value = value[0]
    if not torch.is_tensor(value):
        return None
    if value.ndim >= 3:
        value = value[0, -1]
    elif value.ndim == 2:
        value = value[-1]
    return value.detach().float().cpu().contiguous()


def norm(value: torch.Tensor | None) -> float | None:
    return None if value is None else float(torch.linalg.vector_norm(value).item())


def top_values(value: torch.Tensor | None, top_k: int, unit_kind: str, **address: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    flat = value.reshape(-1)
    count = min(max(1, int(top_k)), int(flat.numel()))
    magnitudes, indices = torch.topk(flat.abs(), k=count)
    rows: list[dict[str, Any]] = []
    for rank, (magnitude, index) in enumerate(zip(magnitudes.tolist(), indices.tolist()), start=1):
        rows.append(
            {
                "rank": rank,
                "unit_kind": unit_kind,
                "unit_index": int(index),
                "value": float(flat[index].item()),
                "magnitude": float(magnitude),
                **address,
            }
        )
    return rows


def attention_top_values(
    value: torch.Tensor | None,
    top_k: int,
    head_count: int,
    projection: str,
) -> list[dict[str, Any]]:
    if value is None or head_count <= 0 or value.numel() % head_count != 0:
        return top_values(value, top_k, "residual_dimension", projection=projection)
    head_dim = int(value.numel() // head_count)
    flat_rows = top_values(value, top_k, "attention_head_channel", projection=projection)
    for row in flat_rows:
        flat_index = int(row.pop("unit_index"))
        row["head_index"] = int(flat_index // head_dim)
        row["unit_index"] = int(flat_index % head_dim)
        row["flat_index"] = flat_index
    return flat_rows


class CheckpointReadout:
    def __init__(self, model: Any, tokenizer: Any, model_dir: Path) -> None:
        self.token_ids = {label: first_token_id(tokenizer, label) for label in COLOR_LABELS}
        self.eps = float(getattr(model.config, "rms_norm_eps", 1e-6))
        self.weight_map: dict[str, str] = {}
        for index_path in sorted(model_dir.glob("*.safetensors.index.json")):
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            self.weight_map.update({str(key): str(value) for key, value in (payload.get("weight_map") or {}).items()})
        self.direct_files = sorted(model_dir.glob("*.safetensors"))
        self.model_dir = model_dir
        self.norm_weight = self._tensor(["model.norm.weight", "transformer.encoder.final_layernorm.weight"])
        self.readout_rows = self._rows(
            ["lm_head.weight", "model.embed_tokens.weight", "transformer.embedding.word_embeddings.weight"],
            sorted(set(self.token_ids.values())),
        )

    def _source_for(self, names: list[str]) -> tuple[Path, str] | None:
        for name in names:
            shard = self.weight_map.get(name)
            if shard:
                return self.model_dir / shard, name
        for path in self.direct_files:
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                keys = set(handle.keys())
                for name in names:
                    if name in keys:
                        return path, name
        return None

    def _tensor(self, names: list[str]) -> torch.Tensor | None:
        source = self._source_for(names)
        if source is None:
            return None
        path, name = source
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            return handle.get_tensor(name).detach().float().cpu()

    def _rows(self, names: list[str], row_ids: list[int]) -> dict[int, torch.Tensor]:
        source = self._source_for(names)
        if source is None:
            raise ValueError(f"Could not locate readout weights in {self.model_dir}")
        path, name = source
        rows: dict[int, torch.Tensor] = {}
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            weight_slice = handle.get_slice(name)
            for row_id in row_ids:
                rows[row_id] = weight_slice[row_id].detach().float().cpu()
        return rows

    def normalize(self, hidden: torch.Tensor) -> torch.Tensor:
        vector = hidden.detach().float().cpu()
        if self.norm_weight is None:
            return vector
        rms = torch.rsqrt(vector.pow(2).mean() + self.eps)
        return vector * rms * self.norm_weight

    def candidate_field(self, hidden: torch.Tensor, target_label: str) -> dict[str, Any]:
        vector = self.normalize(hidden)
        token_ids = self.token_ids
        scores: list[dict[str, Any]] = []
        for label, token_id in token_ids.items():
            score = torch.dot(vector, self.readout_rows[token_id]).item()
            scores.append({"label": label, "token_id": token_id, "score": float(score)})
        scores.sort(key=lambda row: row["score"], reverse=True)
        target = next(row for row in scores if row["label"] == target_label)
        competitor = next(row for row in scores if row["label"] != target_label)
        shifted = torch.tensor([row["score"] for row in scores], dtype=torch.float32)
        probs = torch.softmax(shifted, dim=0).tolist()
        for row, probability in zip(scores, probs):
            row["probability"] = float(probability)
        return {
            "target_label": target_label,
            "target_score": target["score"],
            "target_rank": 1 + next(index for index, row in enumerate(scores) if row["label"] == target_label),
            "competitor_label": competitor["label"],
            "competitor_score": competitor["score"],
            "margin": float(target["score"] - competitor["score"]),
            "scores": scores,
        }


def color_candidate_field(
    projector: CheckpointReadout,
    hidden: torch.Tensor,
    target_label: str,
) -> dict[str, Any]:
    return projector.candidate_field(hidden, target_label)


def module_if_present(parent: Any, name: str) -> Any | None:
    value = getattr(parent, name, None)
    return value if value is not None and hasattr(value, "register_forward_hook") else None


def register_capture(
    handles: list[Any],
    captured: dict[str, torch.Tensor],
    key: str,
    module: Any | None,
    pre: bool = False,
) -> None:
    if module is None:
        return

    if pre:
        def hook(_module: Any, inputs: tuple[Any, ...]) -> None:
            value = to_last_token(inputs[0] if inputs else None)
            if value is not None:
                captured[key] = value

        handles.append(module.register_forward_pre_hook(hook))
    else:
        def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
            value = to_last_token(output)
            if value is not None:
                captured[key] = value

        handles.append(module.register_forward_hook(hook))


def model_snapshot(loaded: Any) -> dict[str, Any]:
    config = loaded.model.config
    config_path = loaded.spec.local_dir / "config.json"
    hidden_size = int(config.hidden_size)
    heads = int(config.num_attention_heads)
    return {
        "schema_version": "model_snapshot.v1",
        "model": loaded.key,
        "model_revision": f"config-sha256:{sha256(config_path)}",
        "architecture": type(loaded.model).__name__,
        "model_dir": str(loaded.spec.local_dir.resolve()),
        "num_hidden_layers": int(config.num_hidden_layers),
        "hidden_size": hidden_size,
        "intermediate_size": int(config.intermediate_size),
        "num_attention_heads": heads,
        "num_key_value_heads": int(getattr(config, "num_key_value_heads", heads)),
        "head_dim": int(getattr(config, "head_dim", hidden_size // heads)),
        "vocab_size": int(config.vocab_size),
        "captured_at": utc_now(),
    }


def build_event(
    events: list[dict[str, Any]],
    vectors: dict[str, torch.Tensor],
    run_id: str,
    snapshot: dict[str, Any],
    layer: int,
    event_type: str,
    component: str,
    vector: torch.Tensor | None,
    top_units: list[dict[str, Any]],
    token_position: int,
    candidate_field: dict[str, Any] | None = None,
) -> None:
    vector_key = f"L{layer}:{event_type}"
    if vector is not None:
        vectors[vector_key] = vector.to(dtype=torch.float32).contiguous()
    events.append(
        {
            "schema_version": "real_trace_event.v1",
            "run_id": run_id,
            "event_index": len(events),
            "event_type": event_type,
            "model": snapshot["model"],
            "model_revision": snapshot["model_revision"],
            "layer": layer,
            "component": component,
            "token_position": token_position,
            "vector_ref": f"full_vectors.pt#{vector_key}" if vector is not None else None,
            "vector_shape": list(vector.shape) if vector is not None else None,
            "norm": norm(vector),
            "top_units": top_units,
            "candidate_field": candidate_field,
            "source_artifact": f"tests/result/phase287_real_component_trace/{run_id}/trace.json#event={len(events)}",
        }
    )


def run_trace(args: argparse.Namespace) -> dict[str, Any]:
    run_id = args.round_name or f"phase287_{args.model}_{args.target_label}_component_trace"
    run_dir = RESULT_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        payload = {
            "phase": PHASE,
            "run_id": run_id,
            "model": args.model,
            "prompt": args.prompt,
            "target_label": args.target_label,
            "status": "dry_run",
        }
        write_json(run_dir / "trace.json", payload)
        return payload

    loaded = None
    handles: list[Any] = []
    try:
        print(f"[Phase287] loading model={args.model}", flush=True)
        loaded = load_probe_model(args.model)
        model = loaded.model
        tokenizer = loaded.tokenizer
        snapshot = model_snapshot(loaded)
        projector = CheckpointReadout(model, tokenizer, loaded.spec.local_dir)
        layers = get_layers(model)
        captured: dict[str, torch.Tensor] = {}
        vectors: dict[str, torch.Tensor] = {}
        events: list[dict[str, Any]] = []

        for layer_index, layer in enumerate(layers):
            register_capture(handles, captured, f"L{layer_index}:layer_input", layer, pre=True)
            register_capture(handles, captured, f"L{layer_index}:norm1", module_if_present(layer, "input_layernorm"))
            register_capture(handles, captured, f"L{layer_index}:residual1", module_if_present(layer, "post_attention_layernorm"), pre=True)
            register_capture(handles, captured, f"L{layer_index}:norm2", module_if_present(layer, "post_attention_layernorm"))

            attention = getattr(layer, "self_attn", None)
            if attention is not None:
                for projection in ("q_proj", "k_proj", "v_proj", "qkv_proj", "o_proj"):
                    register_capture(handles, captured, f"L{layer_index}:{projection}", module_if_present(attention, projection))
                register_capture(handles, captured, f"L{layer_index}:attention_output", attention)

            mlp = getattr(layer, "mlp", None)
            if mlp is not None:
                for projection in ("gate_proj", "up_proj", "gate_up_proj"):
                    register_capture(handles, captured, f"L{layer_index}:{projection}", module_if_present(mlp, projection))
                down_proj = module_if_present(mlp, "down_proj")
                register_capture(handles, captured, f"L{layer_index}:mlp_product", down_proj, pre=True)
                register_capture(handles, captured, f"L{layer_index}:down_proj", down_proj)

        print(f"[Phase287] hooks={len(handles)} layers={len(layers)} running forward", flush=True)
        encoded = tokenizer(args.prompt, return_tensors="pt", add_special_tokens=False)
        inputs = {key: value.to(loaded.input_device) for key, value in encoded.items()}
        token_position = int(inputs["input_ids"].shape[1] - 1)
        with torch.inference_mode():
            output = model(**inputs, use_cache=False, output_hidden_states=True, return_dict=True)
        print(f"[Phase287] forward complete captured={len(captured)}", flush=True)
        for handle in handles:
            handle.remove()
        handles.clear()

        hidden_states = [state[0, -1].detach().float().cpu() for state in output.hidden_states]
        build_event(
            events,
            vectors,
            run_id,
            snapshot,
            -1,
            "embedding",
            "embedding",
            hidden_states[0],
            top_values(hidden_states[0], args.top_k, "residual_dimension"),
            token_position,
            color_candidate_field(projector, hidden_states[0], args.target_label),
        )

        q_heads = int(snapshot["num_attention_heads"])
        kv_heads = int(snapshot["num_key_value_heads"])
        for layer_index in range(len(layers)):
            layer_hidden = hidden_states[layer_index + 1]
            ordered = [
                ("layer_input", "residual", "residual_input"),
                ("norm1", "norm", "norm1"),
                ("q_proj", "attention", "q_projection"),
                ("k_proj", "attention", "k_projection"),
                ("v_proj", "attention", "v_projection"),
                ("qkv_proj", "attention", "qkv_projection"),
                ("attention_output", "attention", "attention_output"),
                ("residual1", "residual", "residual1"),
                ("norm2", "norm", "norm2"),
                ("gate_proj", "mlp", "mlp_gate"),
                ("up_proj", "mlp", "mlp_up"),
                ("gate_up_proj", "mlp", "mlp_gate_up_merged"),
                ("mlp_product", "mlp", "mlp_product"),
                ("down_proj", "mlp", "mlp_down"),
            ]
            for capture_name, component, event_type in ordered:
                vector = captured.get(f"L{layer_index}:{capture_name}")
                if vector is None:
                    continue
                if capture_name == "q_proj":
                    units = attention_top_values(vector, args.top_k, q_heads, "q")
                elif capture_name in {"k_proj", "v_proj"}:
                    units = attention_top_values(vector, args.top_k, kv_heads, capture_name[0])
                elif capture_name == "mlp_product":
                    units = top_values(vector, args.top_k, "mlp_product_neuron")
                elif capture_name == "gate_proj":
                    units = top_values(vector, args.top_k, "mlp_gate_neuron")
                elif capture_name == "up_proj":
                    units = top_values(vector, args.top_k, "mlp_up_neuron")
                else:
                    units = top_values(vector, args.top_k, "residual_dimension")
                build_event(
                    events,
                    vectors,
                    run_id,
                    snapshot,
                    layer_index,
                    event_type,
                    component,
                    vector,
                    units,
                    token_position,
                )
            build_event(
                events,
                vectors,
                run_id,
                snapshot,
                layer_index,
                "residual2",
                "residual",
                layer_hidden,
                top_values(layer_hidden, args.top_k, "residual_dimension"),
                token_position,
                color_candidate_field(projector, layer_hidden, args.target_label),
            )
            if layer_index == 0 or (layer_index + 1) % 10 == 0 or layer_index == len(layers) - 1:
                print(f"[Phase287] compacted layers={layer_index + 1}/{len(layers)} events={len(events)}", flush=True)

        final_logits = output.logits[0, -1].detach().float().cpu()
        top_logits, top_token_ids = torch.topk(final_logits, k=min(args.top_k, int(final_logits.numel())))
        next_tokens = [
            {
                "rank": rank,
                "token_id": int(token_id),
                "token": tokenizer.decode([int(token_id)]),
                "logit": float(logit),
            }
            for rank, (logit, token_id) in enumerate(zip(top_logits.tolist(), top_token_ids.tolist()), start=1)
        ]
        final_field = color_candidate_field(projector, hidden_states[-1], args.target_label)
        build_event(
            events,
            vectors,
            run_id,
            snapshot,
            len(layers) - 1,
            "unembedding_readout",
            "unembedding",
            None,
            [
                {
                    "rank": row["rank"],
                    "unit_kind": "unembedding_token",
                    "unit_index": row["token_id"],
                    "value": row["logit"],
                    "token": row["token"],
                }
                for row in next_tokens
            ],
            token_position,
            final_field,
        )

        tensor_path = run_dir / "full_vectors.pt"
        torch.save(
            {
                "schema_version": "real_component_vectors.v1",
                "run_id": run_id,
                "model": args.model,
                "model_revision": snapshot["model_revision"],
                "vectors": vectors,
            },
            tensor_path,
        )
        print(f"[Phase287] vector archive saved vectors={len(vectors)}", flush=True)
        trace = {
            "schema_version": "real_component_trace.v1",
            "phase": PHASE,
            "run_id": run_id,
            "created_at": utc_now(),
            "status": "complete",
            "model": args.model,
            "model_snapshot": snapshot,
            "prompt": args.prompt,
            "target_label": args.target_label,
            "token_position": token_position,
            "tokens": [tokenizer.decode([int(token_id)]) for token_id in inputs["input_ids"][0].tolist()],
            "events": events,
            "next_tokens": next_tokens,
            "summary": {
                "event_count": len(events),
                "vector_count": len(vectors),
                "layer_count": len(layers),
                "target_color_rank": final_field.get("target_rank"),
                "target_color_margin": final_field.get("margin"),
                "next_token": next_tokens[0] if next_tokens else None,
                "full_vector_archive": "full_vectors.pt",
                "full_vector_archive_sha256": sha256(tensor_path),
                "vram_gb": vram_gb(),
            },
            "evidence_boundary": "Exact forward activations and physical unit addresses; no causal claim is made by this trace.",
        }
        trace_path = run_dir / "trace.json"
        write_json(trace_path, trace)
        write_json(run_dir / "model_snapshot.json", snapshot)
        PUBLIC_ROOT.mkdir(parents=True, exist_ok=True)
        public_path = PUBLIC_ROOT / f"{run_id}.json"
        write_json(public_path, trace)
        return trace
    finally:
        for handle in handles:
            handle.remove()
        release_loaded(loaded)
        gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture a full real-component forward trace")
    parser.add_argument("--model", choices=["qwen3", "glm4", "deepseek7b"], required=True)
    parser.add_argument("--prompt", default="A red cube is placed on the table. The color of the cube is")
    parser.add_argument("--target-label", default="red", choices=COLOR_LABELS)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--round-name", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    trace = run_trace(args)
    print(
        json.dumps(
            {
                "phase": PHASE,
                "run_id": trace["run_id"],
                "model": trace["model"],
                "status": trace["status"],
                "summary": trace.get("summary"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
