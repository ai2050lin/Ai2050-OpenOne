from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from hf_probe_env import (
    encode,
    first_token_id,
    get_layers,
    load_probe_model,
    local_model_status,
    release_loaded,
    vram_gb,
)
from model_registry import REPO_ROOT, all_model_keys


FRUIT_PROMPTS = [
    "An apple is a kind of",
    "A banana is a kind of",
    "An orange is a kind of",
    "A pear is a kind of",
    "A grape is a kind of",
]

CONTROL_PROMPTS = [
    "A hammer is a kind of",
    "A chair is a kind of",
    "A river is a kind of",
    "A stone is a kind of",
    "A car is a kind of",
]

EVAL_PROMPTS = [
    "A mango is a kind of",
    "A screwdriver is a kind of",
    "A peach is a kind of",
    "A train is a kind of",
]


def select_layers(n_layers: int) -> list[int]:
    raw = [0, n_layers // 4, n_layers // 2, (3 * n_layers) // 4, n_layers - 1]
    return sorted(set(max(0, min(n_layers - 1, x)) for x in raw))


@torch.no_grad()
def collect_layer_vectors(loaded, prompts: list[str], layer_ids: list[int]) -> dict[int, torch.Tensor]:
    buckets: dict[int, list[torch.Tensor]] = {i: [] for i in layer_ids}
    for prompt in prompts:
        batch = encode(loaded, prompt)
        out = loaded.model(**batch, output_hidden_states=True, use_cache=False)
        for layer_id in layer_ids:
            hidden = out.hidden_states[layer_id + 1][0, -1].detach().float().cpu()
            buckets[layer_id].append(hidden)
    return {layer_id: torch.stack(vecs) for layer_id, vecs in buckets.items()}


def centroid_probe(fruit: torch.Tensor, control: torch.Tensor, eval_vecs: torch.Tensor) -> dict:
    fruit_center = fruit.mean(dim=0)
    control_center = control.mean(dim=0)
    direction = fruit_center - control_center
    direction = direction / direction.norm().clamp_min(1e-6)

    fruit_scores = (fruit @ direction).tolist()
    control_scores = (control @ direction).tolist()
    eval_scores = (eval_vecs @ direction).tolist()
    margin = float(torch.tensor(fruit_scores).mean() - torch.tensor(control_scores).mean())
    return {
        "fruit_mean": float(torch.tensor(fruit_scores).mean()),
        "control_mean": float(torch.tensor(control_scores).mean()),
        "fruit_control_margin": margin,
        "eval_scores": eval_scores,
        "direction_norm": float(direction.norm()),
    }


@torch.no_grad()
def target_margin(loaded, prompt: str) -> float:
    fruit_id = first_token_id(loaded.tokenizer, " fruit")
    tool_id = first_token_id(loaded.tokenizer, " tool")
    batch = encode(loaded, prompt)
    out = loaded.model(**batch, use_cache=False)
    logits = out.logits[0, -1].float()
    return float(logits[fruit_id] - logits[tool_id])


@torch.no_grad()
def ablated_target_margin(loaded, prompt: str, layer_id: int, direction: torch.Tensor) -> float:
    layers = get_layers(loaded.model)
    device_direction = direction.to(loaded.input_device)

    def hook(_module, _inputs, output):
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = None
        d = device_direction.to(hidden.device, dtype=hidden.dtype)
        d = d / d.float().norm().to(hidden.dtype).clamp_min(torch.tensor(1e-6, device=hidden.device))
        last = hidden[:, -1:, :]
        coeff = (last.float() @ d.float()).to(hidden.dtype)
        edited = hidden.clone()
        edited[:, -1:, :] = last - coeff.unsqueeze(-1) * d
        if rest is None:
            return edited
        return (edited, *rest)

    handle = layers[layer_id].register_forward_hook(hook)
    try:
        return target_margin(loaded, prompt)
    finally:
        handle.remove()


def run_model(model_key: str, output_dir: Path) -> dict:
    loaded = None
    try:
        loaded = load_probe_model(model_key)
        layers = get_layers(loaded.model)
        layer_ids = select_layers(len(layers))
        fruit = collect_layer_vectors(loaded, FRUIT_PROMPTS, layer_ids)
        control = collect_layer_vectors(loaded, CONTROL_PROMPTS, layer_ids)
        eval_vecs = collect_layer_vectors(loaded, EVAL_PROMPTS, layer_ids)

        probe_by_layer = {}
        best_layer = layer_ids[0]
        best_margin = -1e9
        best_direction = None
        for layer_id in layer_ids:
            probe = centroid_probe(fruit[layer_id], control[layer_id], eval_vecs[layer_id])
            probe_by_layer[str(layer_id)] = probe
            if probe["fruit_control_margin"] > best_margin:
                best_margin = probe["fruit_control_margin"]
                best_layer = layer_id
                direction = fruit[layer_id].mean(dim=0) - control[layer_id].mean(dim=0)
                best_direction = direction / direction.norm().clamp_min(1e-6)

        ablation_prompt = "A mango is a kind of"
        clean_margin = target_margin(loaded, ablation_prompt)
        ablated_margin = ablated_target_margin(
            loaded,
            ablation_prompt,
            best_layer,
            best_direction,
        )
        allocated, reserved = vram_gb()
        return {
            "model": model_key,
            "class": type(loaded.model).__name__,
            "n_layers": len(layers),
            "tested_layers": layer_ids,
            "best_probe_layer": best_layer,
            "probe_by_layer": probe_by_layer,
            "ablation": {
                "prompt": ablation_prompt,
                "clean_fruit_minus_tool_logit": clean_margin,
                "ablated_fruit_minus_tool_logit": ablated_margin,
                "delta_after_removing_fruit_direction": ablated_margin - clean_margin,
            },
            "gpu_gb": {"allocated": allocated, "reserved": reserved},
        }
    finally:
        release_loaded(loaded)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="*", default=["qwen3"])
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "results" / "gpt5_probe_ablation_smoke"))
    args = parser.parse_args()

    if args.models == ["all"]:
        models = all_model_keys()
    else:
        models = args.models

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"local_model_status": local_model_status(), "results": []}
    by_model = {}
    for existing in output_dir.glob("*_probe_ablation_smoke.json"):
        try:
            data = json.loads(existing.read_text(encoding="utf-8"))
            by_model[data["model"]] = data
        except Exception:
            pass
    for model_key in models:
        print(f"[smoke] {model_key}")
        result = run_model(model_key, output_dir)
        by_model[model_key] = result
        model_file = output_dir / f"{model_key}_probe_ablation_smoke.json"
        model_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result["ablation"], ensure_ascii=False, indent=2))

    summary["results"] = [by_model[key] for key in sorted(by_model)]
    summary_file = output_dir / "summary.json"
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[smoke] summary: {summary_file}")


if __name__ == "__main__":
    main()
