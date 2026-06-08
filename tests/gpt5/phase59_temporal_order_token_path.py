from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np
import torch


sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tests" / "gpt5"))

from model_registry import get_model_spec  # noqa: E402


MODEL_LAYERS = {
    "qwen3": [21, 23, 25, 27, 29],
    "glm4": [30, 33, 36, 38],
    "deepseek7b": [19, 21, 23, 24],
}


def log(msg: str = "") -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def safe_mean(xs: list[float]) -> float:
    return float(mean(xs)) if xs else 0.0


def safe_std(xs: list[float]) -> float:
    return float(pstdev(xs)) if len(xs) > 1 else 0.0


def load_model(model_name: str, attn_impls: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = get_model_spec(model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        spec.local_dir, trust_remote_code=True, local_files_only=True, use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = None
    for impl in [x.strip() for x in attn_impls.split(",") if x.strip()]:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                spec.local_dir,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                local_files_only=True,
                attn_implementation=impl,
            )
            log(f"Loaded {model_name} with attn_impl={impl}")
            break
        except Exception as exc:
            log(f"Failed with {impl}: {exc}")
    if model is None:
        raise RuntimeError(f"failed to load {model_name}")
    model.eval()
    return model, tokenizer, next(model.parameters()).device


def get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise ValueError("cannot find model layers")


def get_lm_head(model) -> torch.Tensor:
    if not hasattr(model, "lm_head"):
        raise ValueError("model has no lm_head")
    return model.lm_head.weight.detach().float().cpu()


def encode_one(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def find_subseq(haystack: list[int], needle: list[int]) -> int | None:
    if not needle:
        return None
    for i in range(0, len(haystack) - len(needle) + 1):
        if haystack[i : i + len(needle)] == needle:
            return i + len(needle) - 1
    return None


def build_cases(max_cases: int) -> list[dict[str, Any]]:
    event_pairs = [
        ("dax smiled", "wug left"),
        ("mip slept", "tev arrived"),
        ("zog jumped", "nif waited"),
        ("pav cooked", "lom ate"),
        ("sarn opened", "fep closed"),
        ("kiv started", "ral finished"),
        ("norb called", "tess answered"),
        ("vok entered", "pem exited"),
        ("laz won", "gup lost"),
        ("dorn broke", "mav repaired"),
        ("siv wrote", "tul read"),
        ("bex rose", "karn fell"),
    ]
    cases: list[dict[str, Any]] = []
    templates = [
        ("EVENT_A happened before EVENT_B.", "A"),
        ("EVENT_A happened after EVENT_B.", "B"),
        ("EVENT_B happened before EVENT_A.", "B"),
        ("EVENT_B happened after EVENT_A.", "A"),
    ]
    prefixes = ["", "In this record, ", "For the timeline, "]
    for prefix in prefixes:
        for event_a, event_b in event_pairs:
            for rel, answer in templates:
                statement = rel.replace("EVENT_A", event_a).replace("EVENT_B", event_b)
                prompt = (
                    f"{prefix}A = {event_a}. B = {event_b}. "
                    f"Relation: {statement} Answer with A or B. FIRST_EVENT:"
                )
                cases.append(
                    {
                        "prompt": prompt,
                        "answer": answer,
                        "event_a": event_a,
                        "event_b": event_b,
                        "operator": "before" if "before" in rel else "after",
                    }
                )
                if len(cases) >= max_cases:
                    return cases
    return cases


def sequence_logprob(model, tokenizer, device, prompt: str, completion: str) -> float:
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    comp_ids = tokenizer(completion, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    input_ids = torch.cat([prompt_ids, comp_ids], dim=1)
    with torch.no_grad():
        logits = model(input_ids).logits
        log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    start = prompt_ids.shape[1] - 1
    total = 0.0
    for i, tok in enumerate(comp_ids[0].tolist()):
        total += float(log_probs[0, start + i, tok].detach().cpu())
    return total


def capture_token_states(model, tokenizer, device, prompt: str, layers_to_probe: list[int], positions: dict[str, int]) -> dict[str, np.ndarray]:
    layers = get_layers(model)
    captured: dict[str, np.ndarray] = {}

    valid_positions = {k: v for k, v in positions.items() if v is not None and v >= 0}

    def grab_tensor(key_prefix: str, tensor: torch.Tensor) -> None:
        if tensor.dim() == 3:
            arr = tensor[0].detach().float().cpu()
        else:
            return
        for pos_name, pos in valid_positions.items():
            if pos < arr.shape[0]:
                captured[f"{key_prefix}:{pos_name}"] = arr[pos].numpy()

    hooks = []
    for li in layers_to_probe:
        layer = layers[li]

        def pre_hook(_module, inputs, idx=li):
            if inputs and isinstance(inputs[0], torch.Tensor):
                grab_tensor(f"L{idx}:resid_in", inputs[0])

        def layer_hook(_module, _inputs, output, idx=li):
            val = output[0] if isinstance(output, tuple) else output
            if isinstance(val, torch.Tensor):
                grab_tensor(f"L{idx}:resid_out", val)

        hooks.append(layer.register_forward_pre_hook(pre_hook))
        hooks.append(layer.register_forward_hook(layer_hook))

        if hasattr(layer, "self_attn"):
            def attn_hook(_module, _inputs, output, idx=li):
                val = output[0] if isinstance(output, tuple) else output
                if isinstance(val, torch.Tensor):
                    grab_tensor(f"L{idx}:attn_out", val)

            hooks.append(layer.self_attn.register_forward_hook(attn_hook))

        if hasattr(layer, "mlp"):
            def mlp_hook(_module, _inputs, output, idx=li):
                val = output[0] if isinstance(output, tuple) else output
                if isinstance(val, torch.Tensor):
                    grab_tensor(f"L{idx}:mlp_out", val)

            hooks.append(layer.mlp.register_forward_hook(mlp_hook))

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=192).to(device)
    with torch.no_grad():
        model(**inputs)
    for hook in hooks:
        hook.remove()
    return captured


def summarize_scores(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        key = f"{row['module']}:{row['position']}"
        grouped[key].append(float(row["signed_projection"]))
    ranked = [
        {
            "path": key,
            "mean_signed_projection": safe_mean(vals),
            "std": safe_std(vals),
            "n": len(vals),
        }
        for key, vals in grouped.items()
    ]
    ranked.sort(key=lambda x: x["mean_signed_projection"], reverse=True)
    return {"ranked_paths": ranked}


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    model, tokenizer, device = load_model(args.model, args.attn_implementations)
    W_U = get_lm_head(model)
    candidate_a = encode_one(tokenizer, " A")
    candidate_b = encode_one(tokenizer, " B")
    if not candidate_a or not candidate_b:
        raise RuntimeError("could not tokenize A/B candidates")
    dir_ab = (W_U[candidate_a[0]] - W_U[candidate_b[0]]).numpy()
    dir_ab = dir_ab / max(float(np.linalg.norm(dir_ab)), 1e-10)
    layers_to_probe = MODEL_LAYERS[args.model]
    cases = build_cases(args.max_cases)
    log(f"Phase59 Temporal Order Token Path — {args.model}, cases={len(cases)}, layers={layers_to_probe}")
    t0 = time.time()

    baseline_rows: list[dict[str, Any]] = []
    token_rows: list[dict[str, Any]] = []
    for i, case in enumerate(cases, 1):
        prompt = case["prompt"]
        ids = tokenizer(prompt, add_special_tokens=False).input_ids
        positions = {
            "A_label": find_subseq(ids, encode_one(tokenizer, "A")),
            "B_label": find_subseq(ids, encode_one(tokenizer, "B")),
            "before": find_subseq(ids, encode_one(tokenizer, "before")),
            "after": find_subseq(ids, encode_one(tokenizer, "after")),
            "last": len(ids) - 1,
        }
        lp_a = sequence_logprob(model, tokenizer, device, prompt, " A")
        lp_b = sequence_logprob(model, tokenizer, device, prompt, " B")
        margin = lp_a - lp_b
        answer = case["answer"]
        correct = (margin > 0 and answer == "A") or (margin < 0 and answer == "B")
        baseline_rows.append(
            {
                "prompt": prompt,
                "answer": answer,
                "lp_A": lp_a,
                "lp_B": lp_b,
                "margin_A_minus_B": margin,
                "correct": bool(correct),
                "operator": case["operator"],
            }
        )
        states = capture_token_states(model, tokenizer, device, prompt, layers_to_probe, positions)
        sign = 1.0 if answer == "A" else -1.0
        for key, vec in states.items():
            layer_mod, pos_name = key.split(":", 1)
            token_rows.append(
                {
                    "case_index": i - 1,
                    "module": layer_mod,
                    "position": pos_name,
                    "answer": answer,
                    "operator": case["operator"],
                    "signed_projection": sign * float(vec @ dir_ab),
                }
            )
        if i % args.progress_every == 0 or i == len(cases):
            acc = safe_mean([1.0 if r["correct"] else 0.0 for r in baseline_rows])
            log(f"  {i}/{len(cases)} acc={acc:.3f} elapsed={time.time()-t0:.0f}s")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    accuracy = safe_mean([1.0 if r["correct"] else 0.0 for r in baseline_rows])
    margin_abs = safe_mean([abs(float(r["margin_A_minus_B"])) for r in baseline_rows])
    result = {
        "model": args.model,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementations": args.attn_implementations,
        "layers": layers_to_probe,
        "n_cases": len(cases),
        "baseline": {
            "accuracy": accuracy,
            "mean_abs_margin": margin_abs,
            "rows": baseline_rows,
        },
        "token_path_summary": summarize_scores(token_rows),
        "token_rows": token_rows,
        "elapsed_sec": time.time() - t0,
    }
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.model}_phase59_temporal_order_token_path.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    log(f"Saved {out_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--output-dir", default=os.environ.get("PHASE59_OUTPUT_DIR", "results/gpt5_phase59_temporal_order_token_path_full"))
    parser.add_argument("--max-cases", type=int, default=int(os.environ.get("PHASE59_MAX_CASES", "96")))
    parser.add_argument("--attn-implementations", default=os.environ.get("PHASE59_ATTN_IMPLEMENTATIONS", "flash_attention_2,sdpa,eager"))
    parser.add_argument("--progress-every", type=int, default=16)
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()
    try:
        run_model(args)
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
