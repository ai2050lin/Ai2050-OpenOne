"""
Phase 320b: Layer-Specific Causal Verification (Confirmation Round)
===================================================================

Phase 320 found that attribute/function causal efficacy is highly model-dependent:
- GLM4: strong (70-87% positive)
- Qwen3: weak (35-75%)
- DS7B: none/negative

Possible explanation: injection at the DEEPEST layer may not be optimal.
Different models may encode causal directions at different layers.

This test: inject at ALL target layers and find the optimal injection layer.

Usage:
  python tests/glm5/phase320b_layer_causal.py qwen3
  python tests/glm5/phase320b_layer_causal.py glm4
  python tests/glm5/phase320b_layer_causal.py deepseek7b
"""
import sys, os, gc, time, json
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from model_utils import MODEL_CONFIGS, get_model_info, get_layers, release_model, get_W_U

RESULT_DIR = Path("results/phase320b_layer")
RESULT_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR = Path("tmp"); TMP_DIR.mkdir(parents=True, exist_ok=True)
_log_file = None

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if _log_file:
        try:
            with open(_log_file, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except:
            pass


# Simplified stimuli — focused subset
ATTRIBUTE_SOURCES = [
    ("apple", "color", "red"), ("sky", "color", "blue"), ("fire", "temperature", "hot"),
    ("ice", "temperature", "cold"), ("lemon", "taste", "sour"), ("honey", "taste", "sweet"),
    ("silk", "texture", "smooth"), ("sandpaper", "texture", "rough"),
]

ATTRIBUTE_TARGETS = [
    ("strawberry", "color", "red"), ("ocean", "color", "blue"), ("stove", "temperature", "hot"),
    ("snow", "temperature", "cold"), ("vinegar", "taste", "sour"), ("sugar", "taste", "sweet"),
    ("satin", "texture", "smooth"), ("concrete", "texture", "rough"),
]

FUNCTION_SOURCES = [
    ("knife", "cut"), ("pen", "write"), ("car", "drive"),
    ("phone", "call"), ("key", "unlock"), ("lamp", "illuminate"),
    ("camera", "capture"), ("brush", "paint"),
]

SAME_FUNCTION_GROUPS = [
    ("knife", "scissors", "cut"), ("pen", "pencil", "write"),
    ("car", "bus", "drive"), ("lamp", "flashlight", "illuminate"),
]

NEGATION_ADJECTIVES = [
    "happy", "good", "clean", "safe", "warm",
    "fast", "strong", "bright", "easy", "soft",
]

NEGATION_TYPES = {
    "not": lambda adj: (f"very {adj}", f"not {adj}"),
    "never": lambda adj: (f"very {adj}", f"never {adj}"),
    "barely": lambda adj: (f"very {adj}", f"barely {adj}"),
    "morphological": lambda adj: (f"very {adj}", f"un{adj}") if adj[0] in "bhcmnprstw" else None,
    "double_neg": lambda adj: (f"not {adj}", f"not un{adj}") if adj[0] in "bhcmnprstw" else None,
    "scope_neg": lambda adj: (f"tried to be {adj}", f"did not try to be {adj}"),
}


def load_model_bf16(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    attn_impl = "flash_attention_2"
    log(f"Loading {model_name} (bf16 + device_map=auto + {attn_impl})...")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for impl in [attn_impl, "sdpa", "eager"]:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
                trust_remote_code=True, local_files_only=True,
                attn_implementation=impl,
            )
            log(f"  Loaded with {impl}")
            break
        except Exception as e:
            log(f"  {impl} failed, trying next...")
            continue

    model.eval()
    device = next(model.parameters()).device
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    log(f"  Model: {type(model).__name__}, device={device}, GPU={gpu_mem:.2f}GB")
    return model, tokenizer, device


def get_all_target_layers(n_layers):
    """Return ALL layers for injection testing."""
    if n_layers >= 36:
        return list(range(0, n_layers, 4)) + [n_layers - 2]
    elif n_layers >= 28:
        return list(range(0, n_layers, 3)) + [n_layers - 2]
    else:
        return list(range(0, n_layers, 2)) + [n_layers - 2]


def extract_rep_at_all_layers(model, tokenizer, device, sentences, n_layers, label=""):
    """Extract representations at ALL layers for a list of sentences."""
    layers_list = get_layers(model)
    cache = {}
    captured = {}

    def make_hook(li):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                captured[li] = output[0].detach().float().cpu()
            else:
                captured[li] = output.detach().float().cpu()
        return hook_fn

    hooks = [layers_list[li].register_forward_hook(make_hook(li)) for li in range(n_layers)]

    try:
        for idx, sent in enumerate(sentences):
            inp = tokenizer(sent, return_tensors="pt", truncation=True, max_length=128).to(device)
            captured.clear()
            with torch.no_grad():
                model(**inp)

            for li in range(n_layers):
                if li in captured:
                    cache[(sent, li)] = captured[li][0, -1].numpy()

            if (idx + 1) % 30 == 0 or idx == len(sentences) - 1:
                log(f"    {label} Extracted {idx+1}/{len(sentences)}, GPU={torch.cuda.memory_allocated()/1e9:.2f}GB")

            if (idx + 1) % 60 == 0:
                torch.cuda.empty_cache()
    finally:
        for h in hooks:
            h.remove()

    return cache


def inject_and_get_logits(model, tokenizer, device, prompt, direction, layer_idx, alpha, top_k=20):
    """Inject direction at a specific layer and get logits."""
    layers_list = get_layers(model)
    inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
    
    injected_logits = None
    
    def hook_fn(module, input, output):
        nonlocal injected_logits
        hidden = output[0] if isinstance(output, tuple) else output
        d_tensor = torch.tensor(direction, dtype=hidden.dtype, device=hidden.device)
        hidden_modified = hidden.clone()
        hidden_modified[0, -1, :] += (alpha * d_tensor).to(hidden.dtype)
        if isinstance(output, tuple):
            return (hidden_modified,) + output[1:]
        return hidden_modified
    
    hook = layers_list[layer_idx].register_forward_hook(hook_fn)
    
    try:
        with torch.no_grad():
            out = model(**inp)
        injected_logits = out.logits[0, -1].float().cpu().numpy()
    finally:
        hook.remove()
    
    top_k_ids = np.argsort(injected_logits)[-top_k:][::-1]
    top_k_tokens = [(tokenizer.decode([i]).strip().lower(), float(injected_logits[i])) for i in top_k_ids]
    return injected_logits, top_k_tokens


def get_baseline_logits(model, tokenizer, device, prompt, top_k=20):
    inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        out = model(**inp)
    logits = out.logits[0, -1].float().cpu().numpy()
    top_k_ids = np.argsort(logits)[-top_k:][::-1]
    top_k_tokens = [(tokenizer.decode([i]).strip().lower(), float(logits[i])) for i in top_k_ids]
    return logits, top_k_tokens


def run_model(model_name):
    global _log_file
    _log_file = str(TMP_DIR / f"phase320b_{model_name}.log")
    
    log(f"=== Phase 320b: Layer-Specific Causal for {model_name} ===")
    
    t0 = time.time()
    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    log(f"  n_layers={info.n_layers}, d_model={info.d_model}, class={info.model_class}")
    t_load = time.time() - t0
    
    # Get target layers for injection
    inject_layers = get_all_target_layers(info.n_layers)
    log(f"  Injection layers: {inject_layers}")
    
    # Collect all sentences needed
    all_sentences = set()
    for noun, attr_type, value in ATTRIBUTE_SOURCES:
        all_sentences.add(f"the {noun} is {value}")
        all_sentences.add(f"the {noun} is just an object")
    for noun, attr_type, value in ATTRIBUTE_TARGETS:
        all_sentences.add(f"the {noun} is {value}")
        all_sentences.add(f"the {noun} is just an object")
    for tool, action in FUNCTION_SOURCES:
        all_sentences.add(f"people use the {tool} to {action}")
        all_sentences.add(f"people use the {tool}")
    for adj in NEGATION_ADJECTIVES:
        for nt, fn in NEGATION_TYPES.items():
            pair = fn(adj)
            if pair:
                all_sentences.add(pair[0])
                all_sentences.add(pair[1])
    
    all_sentences = sorted(all_sentences)
    log(f"  Total sentences: {len(all_sentences)}")
    
    # Extract at ALL layers
    log("Extracting representations at all layers...")
    t0 = time.time()
    cache = extract_rep_at_all_layers(model, tokenizer, device, all_sentences, info.n_layers, label="All")
    t_extract = time.time() - t0
    log(f"  Extraction time: {t_extract:.1f}s")
    
    results = {}
    
    # ===== Attribute: layer-by-layer injection =====
    log("\n--- Attribute: Layer-by-layer injection ---")
    attr_layer_results = []
    alpha = 2.0
    
    for src_idx, (src_noun, src_attr, src_val) in enumerate(ATTRIBUTE_SOURCES[:4]):
        # Use target at same index
        if src_idx >= len(ATTRIBUTE_TARGETS):
            break
        tgt_noun, tgt_attr, tgt_val = ATTRIBUTE_TARGETS[src_idx]
        
        target_prompt = f"The {tgt_noun} is"
        baseline_logits, _ = get_baseline_logits(model, tokenizer, device, target_prompt, top_k=30)
        
        tgt_val_ids = tokenizer.encode(tgt_val, add_special_tokens=False)
        if not tgt_val_ids:
            continue
        baseline_val_logit = float(baseline_logits[tgt_val_ids[0]])
        
        for li in inject_layers:
            sent_pos = f"the {src_noun} is {src_val}"
            sent_neg = f"the {src_noun} is just an object"
            k_pos = (sent_pos, li)
            k_neg = (sent_neg, li)
            
            if k_pos not in cache or k_neg not in cache:
                continue
            
            d_attr = cache[k_pos] - cache[k_neg]
            d_norm = np.linalg.norm(d_attr)
            if d_norm < 1e-10:
                continue
            d_attr_unit = d_attr / d_norm
            
            inj_logits, _ = inject_and_get_logits(
                model, tokenizer, device, target_prompt,
                d_attr_unit, li, alpha, top_k=30
            )
            
            delta = float(inj_logits[tgt_val_ids[0]] - baseline_val_logit)
            
            attr_layer_results.append({
                "source": f"{src_noun}→{src_val}",
                "target": f"{tgt_noun}→{tgt_val}",
                "layer": li,
                "alpha": alpha,
                "delta": round(delta, 4),
                "positive": delta > 0,
            })
            
            if src_idx == 0:  # Only log first source in detail
                log(f"  L{li}: delta={delta:.4f}")
    
    results["attribute_layers"] = attr_layer_results
    
    # Summary by layer
    log("\n  Attribute injection by layer (alpha=2.0):")
    layer_deltas = {}
    for r in attr_layer_results:
        li = r["layer"]
        if li not in layer_deltas:
            layer_deltas[li] = []
        layer_deltas[li].append(r["delta"])
    
    for li in sorted(layer_deltas.keys()):
        deltas = layer_deltas[li]
        mean_d = np.mean(deltas)
        pos_frac = np.mean([1 if d > 0 else 0 for d in deltas])
        log(f"  L{li}: mean_delta={mean_d:.4f}, positive={pos_frac:.0%}")
    
    # ===== Function: layer-by-layer injection =====
    log("\n--- Function: Layer-by-layer injection ---")
    func_layer_results = []
    alpha = 2.0
    
    for src_idx, (tool, action) in enumerate(FUNCTION_SOURCES[:4]):
        target_prompt = f"People use the {tool} to"
        baseline_logits, _ = get_baseline_logits(model, tokenizer, device, target_prompt, top_k=30)
        
        action_ids = tokenizer.encode(action, add_special_tokens=False)
        if not action_ids:
            continue
        baseline_action_logit = float(baseline_logits[action_ids[0]])
        
        for li in inject_layers:
            sent_pos = f"people use the {tool} to {action}"
            sent_neg = f"people use the {tool}"
            k_pos = (sent_pos, li)
            k_neg = (sent_neg, li)
            
            if k_pos not in cache or k_neg not in cache:
                continue
            
            d_func = cache[k_pos] - cache[k_neg]
            d_norm = np.linalg.norm(d_func)
            if d_norm < 1e-10:
                continue
            d_func_unit = d_func / d_norm
            
            inj_logits, _ = inject_and_get_logits(
                model, tokenizer, device, target_prompt,
                d_func_unit, li, alpha, top_k=30
            )
            
            delta = float(inj_logits[action_ids[0]] - baseline_action_logit)
            
            func_layer_results.append({
                "source": f"{tool}→{action}",
                "layer": li,
                "alpha": alpha,
                "delta": round(delta, 4),
                "positive": delta > 0,
            })
        
        if src_idx == 0:
            for r in func_layer_results:
                if r["source"] == f"{FUNCTION_SOURCES[0][0]}→{FUNCTION_SOURCES[0][1]}":
                    log(f"  L{r['layer']}: delta={r['delta']:.4f}")
    
    results["function_layers"] = func_layer_results
    
    # Summary by layer
    log("\n  Function injection by layer (alpha=2.0):")
    layer_deltas = {}
    for r in func_layer_results:
        li = r["layer"]
        if li not in layer_deltas:
            layer_deltas[li] = []
        layer_deltas[li].append(r["delta"])
    
    for li in sorted(layer_deltas.keys()):
        deltas = layer_deltas[li]
        mean_d = np.mean(deltas)
        pos_frac = np.mean([1 if d > 0 else 0 for d in deltas])
        log(f"  L{li}: mean_delta={mean_d:.4f}, positive={pos_frac:.0%}")
    
    # ===== Negation: layer-by-layer injection =====
    log("\n--- Negation: Layer-by-layer injection ---")
    neg_layer_results = []
    alpha = 2.0
    
    for adj in NEGATION_ADJECTIVES[:3]:
        # Extract "not" direction from each layer
        sent_pos = f"very {adj}"
        sent_neg = f"not {adj}"
        
        prompt = f"very {adj}"
        baseline_logits, _ = get_baseline_logits(model, tokenizer, device, prompt, top_k=30)
        
        neg_words = ["not", "never", "no"]
        
        for li in inject_layers:
            k_pos = (sent_pos, li)
            k_neg = (sent_neg, li)
            
            if k_pos not in cache or k_neg not in cache:
                continue
            
            d_neg = cache[k_neg] - cache[k_pos]
            d_norm = np.linalg.norm(d_neg)
            if d_norm < 1e-10:
                continue
            d_neg_unit = d_neg / d_norm
            
            inj_logits, _ = inject_and_get_logits(
                model, tokenizer, device, prompt,
                d_neg_unit, li, alpha, top_k=30
            )
            
            max_neg_delta = max(float(inj_logits[tokenizer.encode(nw, add_special_tokens=False)[0]] - baseline_logits[tokenizer.encode(nw, add_special_tokens=False)[0]]) for nw in neg_words if tokenizer.encode(nw, add_special_tokens=False))
            
            adj_ids = tokenizer.encode(adj, add_special_tokens=False)
            adj_delta = float(inj_logits[adj_ids[0]] - baseline_logits[adj_ids[0]]) if adj_ids else 0
            
            neg_layer_results.append({
                "adjective": adj,
                "layer": li,
                "alpha": alpha,
                "max_neg_delta": round(max_neg_delta, 4),
                "adj_delta": round(adj_delta, 4),
            })
    
    results["negation_layers"] = neg_layer_results
    
    # Summary by layer
    log("\n  Negation injection by layer (alpha=2.0):")
    layer_data = {}
    for r in neg_layer_results:
        li = r["layer"]
        if li not in layer_data:
            layer_data[li] = {"neg_deltas": [], "adj_deltas": []}
        layer_data[li]["neg_deltas"].append(r["max_neg_delta"])
        layer_data[li]["adj_deltas"].append(r["adj_delta"])
    
    for li in sorted(layer_data.keys()):
        d = layer_data[li]
        mean_neg = np.mean(d["neg_deltas"])
        mean_adj = np.mean(d["adj_deltas"])
        log(f"  L{li}: neg_delta={mean_neg:.4f}, adj_delta={mean_adj:.4f}")
    
    # Save
    output = {
        "model": model_name,
        "n_layers": info.n_layers,
        "d_model": info.d_model,
        "inject_layers": inject_layers,
        "results": results,
    }
    
    out_path = RESULT_DIR / f"{model_name}_phase320b.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    log(f"Results saved to {out_path}")
    
    # Print overall summary
    log("\n" + "="*60)
    log(f"PHASE 320b SUMMARY - {model_name}")
    log("="*60)
    
    # Find best layer for each type
    log("\nBest injection layer (max mean_delta):")
    
    for rtype, key in [("attribute", "attribute_layers"), ("function", "function_layers")]:
        data = results.get(key, [])
        if not data:
            continue
        layer_deltas = {}
        for r in data:
            li = r["layer"]
            if li not in layer_deltas:
                layer_deltas[li] = []
            layer_deltas[li].append(r["delta"])
        best_li = max(layer_deltas.keys(), key=lambda li: np.mean(layer_deltas[li]))
        log(f"  {rtype}: best=L{best_li}, mean_delta={np.mean(layer_deltas[best_li]):.4f}")
    
    # Negation
    neg_data = results.get("negation_layers", [])
    if neg_data:
        layer_neg = {}
        for r in neg_data:
            li = r["layer"]
            if li not in layer_neg:
                layer_neg[li] = []
            layer_neg[li].append(r["max_neg_delta"])
        best_li = max(layer_neg.keys(), key=lambda li: np.mean(layer_neg[li]))
        log(f"  negation: best=L{best_li}, mean_neg_delta={np.mean(layer_neg[best_li]):.4f}")
    
    # Cleanup
    del cache
    release_model(model)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    log(f"Model {model_name} released.")
    
    return output


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    
    if model_name == "all":
        for mn in ["qwen3", "glm4", "deepseek7b"]:
            try:
                run_model(mn)
            except Exception as e:
                log(f"ERROR running {mn}: {e}")
                import traceback; traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(10)
    else:
        run_model(model_name)
