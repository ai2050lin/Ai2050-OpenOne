#!/usr/bin/env python3
"""Phase 975 natural-rollout test of frozen and online protocol transfer.

Two distinct claims are kept separate:

* ``frozen_mean_to_plain`` adds one content-independent direction learned only
  from discovery+development P states at every generation step.
* ``paired_chat_to_plain`` recomputes the donor chat residual from exactly the
  same already-generated continuation at every step.  It uses no answer or EOS
  oracle, but it is a two-forward online state transfer, not a fixed mechanism.

The reverse ``paired_plain_to_chat`` condition tests necessity.  GLM4 and Qwen3
no-think are eligible.  DS7B is excluded because its native receiver is in an
open-thinking stage and is audited separately by phase975_ds_two_stage.py.
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import get_layers, load_model, release_model
from phase951_protocol_atlas import ensure_dir
from phase966_natural_stop import log
from phase973_conditional_trajectory import build_dataset, get_eos_ids
from phase975_protocol_causal_transfer import (
    chat_template_text, run_snapshot, split_items, token_ids,
)


OUT = Path("tests/glm5/result/phase975_online_residual_transfer")
CAUSAL_OUT = Path("tests/glm5/result/phase975_protocol_causal_transfer")


def prefixes(tok, model_name, item):
    plain = token_ids(tok, item["prompt"], add_special_tokens=True)
    chat_text = chat_template_text(tok, model_name, item["prompt"], True, False)
    chat = token_ids(tok, chat_text, add_special_tokens=False)
    return plain, chat


def row_metrics(tok, eos_ids, item, generated):
    eos_positions = [i for i, x in enumerate(generated) if int(x) in eos_ids]
    before_eos = generated[:eos_positions[0]] if eos_positions else generated
    decoded = tok.decode(generated, skip_special_tokens=False)
    plain = tok.decode(before_eos, skip_special_tokens=True)
    expected = item["answer"].lower() in plain.lower()
    return {"generated_ids": [int(x) for x in generated], "generated": decoded,
            "plain_before_eos": plain, "has_expected": expected,
            "has_eos": bool(eos_positions), "valid_eos": bool(eos_positions) and expected,
            "early_eos": bool(eos_positions) and not expected,
            "first_eos_step": eos_positions[0] if eos_positions else None,
            "n_tokens": len(generated)}


def paired_generate(model_name, model, tok, layers, device, eos_ids, item, layer, direction,
                    max_new=32):
    plain_prefix, chat_prefix = prefixes(tok, model_name, item)
    if direction == "chat_to_plain":
        donor_prefix, receiver_prefix = chat_prefix, plain_prefix
    elif direction == "plain_to_chat":
        donor_prefix, receiver_prefix = plain_prefix, chat_prefix
    else:
        raise ValueError(direction)
    continuation = []
    for _ in range(max_new):
        donor_ids = donor_prefix + continuation
        receiver_ids = receiver_prefix + continuation
        donor = run_snapshot(model, layers, device, eos_ids, donor_ids,
                             capture_layers=[layer + 1])
        receiver = run_snapshot(model, layers, device, eos_ids, receiver_ids,
                                patch_layer=layer,
                                patch_vector=donor["vectors"][layer + 1])
        nxt = int(receiver["greedy_id"])
        continuation.append(nxt)
        if nxt in eos_ids:
            break
    return row_metrics(tok, eos_ids, item, continuation)


def fixed_generate(model_name, model, tok, layers, device, eos_ids, item, layer, vector,
                   max_new=32):
    plain_prefix, _ = prefixes(tok, model_name, item)
    x = torch.tensor([plain_prefix], dtype=torch.long, device=device)
    mask = torch.ones_like(x)
    calls = [0]

    def hook(module, args, output):
        is_tuple = isinstance(output, tuple)
        y = output[0] if is_tuple else output
        z = y.clone()
        z[:, -1, :] += vector.to(device=z.device, dtype=z.dtype)
        calls[0] += 1
        return (z,) + output[1:] if is_tuple else z

    h = layers[layer].register_forward_hook(hook)
    try:
        with torch.no_grad():
            out = model.generate(input_ids=x, attention_mask=mask, max_new_tokens=max_new,
                                 do_sample=False, pad_token_id=tok.pad_token_id,
                                 eos_token_id=eos_ids, return_dict_in_generate=True)
    finally:
        h.remove()
    ids = out.sequences[0, len(plain_prefix):].tolist()
    row = row_metrics(tok, eos_ids, item, ids)
    row["layer_hook_calls"] = calls[0]
    return row


def summarize(rows):
    return {"n": len(rows),
            "expected_rate": float(np.mean([r["has_expected"] for r in rows])),
            "eos_rate": float(np.mean([r["has_eos"] for r in rows])),
            "valid_eos_rate": float(np.mean([r["valid_eos"] for r in rows])),
            "early_eos_rate": float(np.mean([r["early_eos"] for r in rows])),
            "mean_tokens": float(np.mean([r["n_tokens"] for r in rows]))}


def run_condition(name, fn, items, model_name):
    rows = []
    for idx, item in enumerate(items):
        rows.append({"id": item["id"], "task": item["task"],
                     "prompt_template": item["prompt_template"], "condition": name,
                     **fn(item)})
        if (idx + 1) % 16 == 0:
            log(f"  {model_name} online {name} {idx+1}/{len(items)}")
    return {"summary": summarize(rows), "rows": rows}


def baseline_lookup(model_name, ids):
    """Read prior natural baselines; no GPU rerun and no intervention tuning."""
    wanted = set(ids)
    p_plain = Path(f"tests/glm5/result/phase973_conditional_trajectory/{model_name}_natural_trace.json")
    p_chat = Path(f"tests/glm5/result/phase974_protocol_conditioning/{model_name}_result.json")
    plain = json.loads(p_plain.read_text(encoding="utf-8"))["rows"]
    chat = json.loads(p_chat.read_text(encoding="utf-8"))["natural_rows"]

    def adapt(rows, protocol):
        out = []
        for r in rows:
            if r["id"] not in wanted:
                continue
            expected = bool(r["has_expected"])
            eos = bool(r["has_eos"])
            out.append({"id": r["id"], "has_expected": expected, "has_eos": eos,
                        "valid_eos": expected and eos, "early_eos": eos and not expected,
                        "n_tokens": r["n_tokens"], "protocol": protocol})
        return {"budget_warning": "plain prior budget=24, chat prior budget=32",
                "summary": summarize(out), "rows": out}
    return {"plain": adapt(plain, "plain"), "chat": adapt(chat, "native_chat")}


def run(model_name):
    if model_name not in ("glm4", "qwen3"):
        raise ValueError("online unified transfer is only for glm4/qwen3")
    ensure_dir(OUT)
    t0 = time.time()
    causal_path = CAUSAL_OUT / f"{model_name}_result.json"
    causal = json.loads(causal_path.read_text(encoding="utf-8"))
    layer = int(causal["selected_layers"]["plain_chat_transfer"])
    saved = torch.load(CAUSAL_OUT / f"{model_name}_frozen_direction.pt",
                       map_location="cpu", weights_only=True)
    direction = saved["direction"].to(torch.float32)
    items = build_dataset()
    _, development, holdout = split_items(items)
    model, tok, device = load_model(model_name)
    layers = get_layers(model)
    eos_ids = get_eos_ids(model, tok)
    result = {"phase": 975, "model": model_name, "layer": layer,
              "frozen_direction_train_n": int(saved["n_train"]),
              "warning": "paired online transfer is content-conditioned two-forward diagnosis; frozen mean is the universal-vector test",
              "development_baselines": baseline_lookup(model_name, [x["id"] for x in development])}

    result["development"] = {}
    result["development"]["frozen_mean_to_plain"] = run_condition(
        "frozen_mean_to_plain",
        lambda x: fixed_generate(model_name, model, tok, layers, device, eos_ids,
                                 x, layer, direction), development, model_name)
    result["development"]["paired_chat_to_plain"] = run_condition(
        "paired_chat_to_plain",
        lambda x: paired_generate(model_name, model, tok, layers, device, eos_ids,
                                  x, layer, "chat_to_plain"), development, model_name)
    result["development"]["paired_plain_to_chat"] = run_condition(
        "paired_plain_to_chat",
        lambda x: paired_generate(model_name, model, tok, layers, device, eos_ids,
                                  x, layer, "plain_to_chat"), development, model_name)

    # Frozen, pre-registered expansion gate.  The paired transfer must restore
    # >=20pp valid EOS without >5pp expected loss versus prior plain; the fixed
    # vector is independently expanded only if it meets the same condition.
    plain_base = result["development_baselines"]["plain"]["summary"]
    gates = {}
    for name in ["frozen_mean_to_plain", "paired_chat_to_plain"]:
        s = result["development"][name]["summary"]
        gates[name] = bool(s["valid_eos_rate"] - plain_base["valid_eos_rate"] >= 0.20
                           and s["expected_rate"] >= plain_base["expected_rate"] - 0.05
                           and s["early_eos_rate"] <= 0.05)
    result["expansion_gates"] = gates
    path = OUT / f"{model_name}_result.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    result["holdout_baselines"] = baseline_lookup(model_name, [x["id"] for x in holdout])
    result["holdout"] = {}
    if gates["frozen_mean_to_plain"]:
        result["holdout"]["frozen_mean_to_plain"] = run_condition(
            "frozen_mean_to_plain",
            lambda x: fixed_generate(model_name, model, tok, layers, device, eos_ids,
                                     x, layer, direction), holdout, model_name)
    else:
        result["holdout"]["frozen_mean_to_plain"] = {"not_run": True,
                                                       "reason": "development gate failed"}
    if gates["paired_chat_to_plain"]:
        result["holdout"]["paired_chat_to_plain"] = run_condition(
            "paired_chat_to_plain",
            lambda x: paired_generate(model_name, model, tok, layers, device, eos_ids,
                                      x, layer, "chat_to_plain"), holdout, model_name)
        result["holdout"]["paired_plain_to_chat"] = run_condition(
            "paired_plain_to_chat",
            lambda x: paired_generate(model_name, model, tok, layers, device, eos_ids,
                                      x, layer, "plain_to_chat"), holdout, model_name)
    else:
        result["holdout"]["paired_chat_to_plain"] = {"not_run": True,
                                                       "reason": "development gate failed"}
        result["holdout"]["paired_plain_to_chat"] = {"not_run": True,
                                                       "reason": "forward development gate failed"}
    result["elapsed_seconds"] = time.time() - t0
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    release_model(model)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log(f"Saved {path}; elapsed={result['elapsed_seconds']/60:.1f} min")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "glm4")
