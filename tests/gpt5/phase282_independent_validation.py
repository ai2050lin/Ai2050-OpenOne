from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from hf_probe_env import encode, get_layers, load_probe_model, release_loaded
from model_registry import REPO_ROOT, all_model_keys


PAIRS = [
    ("animal", "the dog chases the cat", "the cat chases the dog"),
    ("animal", "the wolf hunts the sheep", "the sheep hunts the wolf"),
    ("animal", "the lion chases the deer", "the deer chases the lion"),
    ("animal", "the cat catches the mouse", "the mouse catches the cat"),
    ("human", "the man loves the woman", "the woman loves the man"),
    ("human", "the boy calls the girl", "the girl calls the boy"),
    ("human", "the teacher teaches the student", "the student teaches the teacher"),
    ("human", "the doctor helps the patient", "the patient helps the doctor"),
    ("object", "the chef uses the knife", "the knife uses the chef"),
    ("object", "the painter holds the brush", "the brush holds the painter"),
    ("object", "the writer uses the pen", "the pen uses the writer"),
    ("place", "the king rules the city", "the city rules the king"),
    ("place", "the explorer discovers the island", "the island discovers the explorer"),
    ("abstract", "justice defeats corruption", "corruption defeats justice"),
    ("abstract", "wisdom guides folly", "folly guides wisdom"),
    ("abstract", "truth exposes the lie", "the lie exposes the truth"),
]


def selected_layers(n_layers: int) -> list[int]:
    return sorted({0, n_layers // 4, n_layers // 2, (3 * n_layers) // 4, n_layers - 1})


def _shape_v(v: torch.Tensor, n_kv: int, d_head: int, n_heads: int) -> torch.Tensor:
    # [1, seq, n_kv*d_head] -> [heads, seq, d_head]
    seq = v.shape[1]
    v = v.reshape(1, seq, n_kv, d_head).permute(0, 2, 1, 3)[0].float().cpu()
    if n_kv != n_heads:
        group = n_heads // n_kv
        v = v.repeat_interleave(group, dim=0)
    return v


def _attn_apply(attn: torch.Tensor, v: torch.Tensor, o_weight: torch.Tensor) -> torch.Tensor:
    # attn [heads, seq, seq], v [heads, seq, d_head]
    z = torch.matmul(attn.float().cpu(), v.float())
    seq = z.shape[1]
    flat = z.permute(1, 0, 2).reshape(seq, -1)
    return F.linear(flat, o_weight.float().cpu())


@torch.no_grad()
def run_capture(loaded: Any, sentence: str, layer_ids: list[int]) -> dict:
    layers = get_layers(loaded.model)
    captured: dict[int, torch.Tensor] = {}
    hooks = []

    def make_hook(layer_id: int):
        def hook(_module, inputs, kwargs, _outputs):
            if inputs:
                hidden = inputs[0]
            else:
                hidden = kwargs["hidden_states"]
            captured[layer_id] = hidden.detach()
        return hook

    for layer_id in layer_ids:
        hooks.append(
            layers[layer_id].self_attn.register_forward_hook(make_hook(layer_id), with_kwargs=True)
        )
    try:
        batch = encode(loaded, sentence)
        out = loaded.model(**batch, output_attentions=True, use_cache=False)
    finally:
        for hook in hooks:
            hook.remove()

    return {
        "input_ids": batch["input_ids"].detach().cpu(),
        "attentions": [a.detach().float().cpu() if a is not None else None for a in out.attentions],
        "attn_inputs": captured,
    }


def layer_effects(loaded: Any, cap_a: dict, cap_b: dict, layer_id: int) -> dict | None:
    attn_a = cap_a["attentions"][layer_id]
    attn_b = cap_b["attentions"][layer_id]
    if attn_a is None or attn_b is None:
        return None
    attn_a = attn_a[0]
    attn_b = attn_b[0]
    if attn_a.shape != attn_b.shape:
        return None

    layer = get_layers(loaded.model)[layer_id]
    sa = layer.self_attn
    n_heads = int(getattr(sa, "num_heads", getattr(loaded.model.config, "num_attention_heads")))
    n_kv = int(getattr(sa, "num_key_value_heads", getattr(loaded.model.config, "num_key_value_heads", n_heads)))
    d_head = int(sa.q_proj.weight.shape[0] // n_heads)

    v_a_raw = sa.v_proj(cap_a["attn_inputs"][layer_id]).detach()
    v_b_raw = sa.v_proj(cap_b["attn_inputs"][layer_id]).detach()
    v_a = _shape_v(v_a_raw, n_kv, d_head, n_heads)
    v_b = _shape_v(v_b_raw, n_kv, d_head, n_heads)

    o_weight = sa.o_proj.weight.detach()
    pure_a = _attn_apply(attn_a, v_a, o_weight)
    pure_b = _attn_apply(attn_b, v_b, o_weight)
    route_swap = _attn_apply(attn_a, v_b, o_weight)  # A routing, B value
    content_swap = _attn_apply(attn_b, v_a, o_weight)  # B routing, A value

    gap = torch.linalg.norm(pure_a - pure_b).item()
    if gap < 1e-9:
        return None

    routing_effect = torch.linalg.norm(route_swap - pure_b).item() / gap
    content_effect = torch.linalg.norm(content_swap - pure_b).item() / gap
    last_gap = torch.linalg.norm(pure_a[-1] - pure_b[-1]).item()
    if last_gap < 1e-9:
        last_routing = 0.0
        last_content = 0.0
    else:
        last_routing = torch.linalg.norm(route_swap[-1] - pure_b[-1]).item() / last_gap
        last_content = torch.linalg.norm(content_swap[-1] - pure_b[-1]).item() / last_gap

    return {
        "routing_effect_correct": routing_effect,
        "content_effect_correct": content_effect,
        "last_routing_effect": last_routing,
        "last_content_effect": last_content,
        "routing_dominates": routing_effect > content_effect,
        "total_gap": gap,
        "attn_shape": list(attn_a.shape),
    }


def aggregate(rows: list[dict], layer_ids: list[int]) -> dict:
    per_layer = {}
    for layer_id in layer_ids:
        vals = [r["layers"].get(str(layer_id)) for r in rows]
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        rout = torch.tensor([v["routing_effect_correct"] for v in vals])
        cont = torch.tensor([v["content_effect_correct"] for v in vals])
        per_layer[str(layer_id)] = {
            "n": len(vals),
            "routing_effect_mean": float(rout.mean()),
            "content_effect_mean": float(cont.mean()),
            "routing_dominance_rate": float((rout > cont).float().mean()),
            "winner": "ROUTING" if float(rout.mean()) > float(cont.mean()) else "CONTENT",
        }
    return per_layer


def validate_model(model_key: str) -> dict:
    loaded = None
    try:
        loaded = load_probe_model(model_key)
        layers = get_layers(loaded.model)
        layer_ids = selected_layers(len(layers))
        rows = []
        skipped = []
        for idx, (category, sent_a, sent_b) in enumerate(PAIRS):
            tok_a = loaded.tokenizer(sent_a, return_tensors="pt").input_ids
            tok_b = loaded.tokenizer(sent_b, return_tensors="pt").input_ids
            if tok_a.shape != tok_b.shape:
                skipped.append({"index": idx, "A": sent_a, "B": sent_b, "reason": "token_length_mismatch"})
                continue
            cap_a = run_capture(loaded, sent_a, layer_ids)
            cap_b = run_capture(loaded, sent_b, layer_ids)
            layer_rows = {}
            for layer_id in layer_ids:
                eff = layer_effects(loaded, cap_a, cap_b, layer_id)
                if eff is not None:
                    layer_rows[str(layer_id)] = eff
            rows.append({"category": category, "A": sent_a, "B": sent_b, "layers": layer_rows})
        return {
            "model": model_key,
            "class": type(loaded.model).__name__,
            "n_layers": len(layers),
            "tested_layers": layer_ids,
            "n_pairs_total": len(PAIRS),
            "n_pairs_used": len(rows),
            "skipped": skipped,
            "per_layer": aggregate(rows, layer_ids),
            "pairs": rows,
        }
    finally:
        release_loaded(loaded)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="*", default=["qwen3"])
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "results" / "gpt5_phase282_validation"))
    args = parser.parse_args()

    model_keys = all_model_keys() if args.models == ["all"] else args.models
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"results": []}
    for model_key in model_keys:
        print(f"[phase282-validation] {model_key}", flush=True)
        result = validate_model(model_key)
        summary["results"].append(result)
        (output_dir / f"{model_key}_true_attention_validation.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(result["per_layer"], ensure_ascii=False, indent=2), flush=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[phase282-validation] summary: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
