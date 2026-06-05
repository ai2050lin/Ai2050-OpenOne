"""
Phase 379b: 确认测试——扩大数据量验证PROPER vs PSEUDO差异
==================================================

Phase 379发现DS7B L4的PROPER与PSEUDO post-RMSNorm结果完全不同。
这是纠正Phase 378方法论错误的关键发现。

确认测试目标：
1. 增加数据量（从132对扩展到200+对）
2. 只测试DS7B L4（最关键的发现点）和GLM4 L4（对照）
3. 重点验证：PROPER gap是否稳定为正，PC1与norm_ratio相关性是否稳定

新增数据：更多binding pairs，确保每类别至少25对
"""

import sys, os, time, json, gc, traceback
import torch
import numpy as np
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, 'tests/glm5')

from model_utils import get_layers, release_model, get_W_U, MODEL_CONFIGS


def log(msg="", end="\n"):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", end=end, flush=True)


# ===== Expanded binding pairs =====
COLOR_PAIRS = [
    ("apple", "red", "blue"), ("banana", "yellow", "purple"), ("snow", "white", "black"),
    ("sky", "blue", "green"), ("cherry", "red", "blue"), ("leaf", "green", "red"),
    ("rose", "red", "blue"), ("gold", "yellow", "purple"), ("coal", "black", "white"),
    ("silver", "white", "black"), ("milk", "white", "black"), ("honey", "yellow", "blue"),
    ("ruby", "red", "green"), ("emerald", "green", "red"), ("sapphire", "blue", "red"),
    ("moon", "white", "black"), ("flame", "orange", "blue"), ("forest", "green", "white"),
    ("ocean", "blue", "yellow"), ("sun", "yellow", "purple"),
    ("tomato", "red", "blue"), ("lemon", "yellow", "purple"), ("ivory", "white", "black"),
    ("grass", "green", "red"), ("blood", "red", "blue"), ("cloud", "white", "black"),
    ("amber", "yellow", "green"), ("moss", "green", "blue"), ("raven", "black", "white"),
    ("coral", "orange", "blue"), ("teal", "blue", "red"), ("plum", "purple", "yellow"),
    ("bronze", "brown", "blue"), ("crimson", "red", "green"), ("azure", "blue", "red"),
    ("scarlet", "red", "blue"), ("turquoise", "blue", "green"),
    ("maroon", "brown", "white"), ("lime", "green", "purple"), ("peach", "orange", "blue"),
    # New color pairs
    ("strawberry", "red", "blue"), ("carrot", "orange", "blue"), ("lavender", "purple", "yellow"),
    ("sand", "yellow", "blue"), ("chalk", "white", "black"), ("soot", "black", "white"),
    ("mint", "green", "red"), ("indigo", "blue", "red"), ("crimson", "red", "green"),
    ("jade", "green", "blue"), ("rust", "brown", "blue"), ("pearl", "white", "black"),
]
TEMP_PAIRS = [
    ("fire", "hot", "cold"), ("desert", "hot", "cold"), ("lava", "hot", "cold"),
    ("ice", "cold", "hot"), ("snow", "cold", "hot"), ("volcano", "hot", "cold"),
    ("furnace", "hot", "cold"), ("glacier", "cold", "hot"),
    ("oven", "hot", "cold"), ("frost", "cold", "hot"), ("magma", "hot", "cold"),
    ("winter", "cold", "hot"), ("summer", "hot", "cold"), ("arctic", "cold", "hot"),
    ("stove", "hot", "cold"), ("blizzard", "cold", "hot"), ("tundra", "cold", "hot"),
    ("inferno", "hot", "cold"), ("iceberg", "cold", "hot"),
    # New temp pairs
    ("sauna", "hot", "cold"), ("freezer", "cold", "hot"), ("torch", "hot", "cold"),
    ("igloo", "cold", "hot"), ("heater", "hot", "cold"), ("permafrost", "cold", "hot"),
    ("boiler", "hot", "cold"), ("sleet", "cold", "hot"), ("campfire", "hot", "cold"),
    ("snowflake", "cold", "hot"), ("ember", "hot", "cold"), ("frostbite", "cold", "hot"),
    ("thermos", "hot", "cold"), ("icicle", "cold", "hot"), ("sunburn", "hot", "cold"),
    ("refrigerator", "cold", "hot"), ("candle", "hot", "cold"), ("slush", "cold", "hot"),
]
SIZE_PAIRS = [
    ("elephant", "big", "small"), ("mountain", "big", "small"), ("ant", "small", "big"),
    ("planet", "big", "small"), ("grain", "small", "big"), ("whale", "big", "small"),
    ("galaxy", "big", "small"), ("atom", "small", "big"), ("continent", "big", "small"),
    ("bacteria", "small", "big"), ("tower", "big", "small"), ("speck", "small", "big"),
    ("universe", "big", "small"), ("pixel", "small", "big"), ("castle", "big", "small"),
    ("dust_mote", "small", "big"),
    # New size pairs
    ("ocean", "big", "small"), ("needle", "small", "big"), ("skyscraper", "big", "small"),
    ("molecule", "small", "big"), ("dinosaur", "big", "small"), ("pebble", "small", "big"),
    ("pyramid", "big", "small"), ("flea", "small", "big"), ("moon_size", "big", "small"),
    ("droplet", "small", "big"), ("cathedral", "big", "small"), ("crumb", "small", "big"),
]
WEIGHT_PAIRS = [
    ("boulder", "heavy", "light"), ("feather", "light", "heavy"), ("lead", "heavy", "light"),
    ("balloon", "light", "heavy"), ("steel", "heavy", "light"), ("cotton", "light", "heavy"),
    ("anchor", "heavy", "light"), ("bubble", "light", "heavy"), ("concrete", "heavy", "light"),
    ("air", "light", "heavy"), ("truck", "heavy", "light"), ("petal", "light", "heavy"),
    ("elephant", "heavy", "light"), ("cloud", "light", "heavy"),
    # New weight pairs
    ("anvil", "heavy", "light"), ("dandelion_seed", "light", "heavy"), ("iron", "heavy", "light"),
    ("silk", "light", "heavy"), ("ship", "heavy", "light"), ("confetti", "light", "heavy"),
    ("boulder_stone", "heavy", "light"), ("smoke", "light", "heavy"), ("tank", "heavy", "light"),
    ("snowflake_w", "light", "heavy"), ("dumbbell", "heavy", "light"), ("leaf_w", "light", "heavy"),
]
SPEED_PAIRS = [
    ("cheetah", "fast", "slow"), ("turtle", "slow", "fast"), ("rocket", "fast", "slow"),
    ("snail", "slow", "fast"), ("lightning", "fast", "slow"), ("sloth", "slow", "fast"),
    ("falcon", "fast", "slow"), ("worm", "slow", "fast"), ("bullet", "fast", "slow"),
    ("glacier_motion", "slow", "fast"), ("jet", "fast", "slow"),
    ("racecar", "fast", "slow"), ("caterpillar", "slow", "fast"),
    # New speed pairs
    ("meteor", "fast", "slow"), ("molasses", "slow", "fast"), ("ferrari", "fast", "slow"),
    ("tortoise", "slow", "fast"), ("arrow", "fast", "slow"), ("sludge", "slow", "fast"),
    ("missile", "fast", "slow"), ("lava_flow", "slow", "fast"), ("sprinter", "fast", "slow"),
    ("moss_growth", "slow", "fast"), ("porsche", "fast", "slow"), ("stalactite", "slow", "fast"),
]
BRIGHT_PAIRS = [
    ("star", "bright", "dark"), ("cave", "dark", "bright"), ("sun", "bright", "dark"),
    ("shadow", "dark", "bright"), ("lamp", "bright", "dark"), ("night", "dark", "bright"),
    ("flashlight", "bright", "dark"), ("abyss", "dark", "bright"), ("diamond", "bright", "dark"),
    ("tunnel", "dark", "bright"), ("beacon", "bright", "dark"), ("eclipse", "dark", "bright"),
    ("lighthouse", "bright", "dark"), ("dungeon", "dark", "bright"),
    # New bright pairs
    ("neon", "bright", "dark"), ("void", "dark", "bright"), ("candle_b", "bright", "dark"),
    ("cellar", "dark", "bright"), ("spotlight", "bright", "dark"), ("midnight", "dark", "bright"),
    ("lantern", "bright", "dark"), ("blackout", "dark", "bright"), ("sparkler", "bright", "dark"),
    ("crypt", "dark", "bright"), ("phosphor", "bright", "dark"), ("bunker", "dark", "bright"),
]
MOISTURE_PAIRS = [
    ("rain", "wet", "dry"), ("ocean", "wet", "dry"), ("river", "wet", "dry"),
    ("sand", "dry", "wet"), ("dust", "dry", "wet"), ("bone", "dry", "wet"),
    ("swamp", "wet", "dry"), ("desert", "dry", "wet"),
    ("lake", "wet", "dry"), ("sponge", "wet", "dry"), ("cracker", "dry", "wet"),
    ("fog", "wet", "dry"), ("prairie", "dry", "wet"), ("puddle", "wet", "dry"),
    ("cactus", "dry", "wet"), ("waterfall", "wet", "dry"),
    # New moisture pairs
    ("dew", "wet", "dry"), ("sahara", "dry", "wet"), ("mist", "wet", "dry"),
    ("biscuit", "dry", "wet"), ("flood", "wet", "dry"), ("ash", "dry", "wet"),
    ("tide", "wet", "dry"), ("parchment", "dry", "wet"), ("drizzle", "wet", "dry"),
    ("gravel", "dry", "wet"), ("tsunami", "wet", "dry"), ("tinder", "dry", "wet"),
]

ALL_PAIRS = COLOR_PAIRS + TEMP_PAIRS + MOISTURE_PAIRS + SIZE_PAIRS + WEIGHT_PAIRS + SPEED_PAIRS + BRIGHT_PAIRS

PAIR_CATEGORIES = (
    ["color"] * len(COLOR_PAIRS) +
    ["temperature"] * len(TEMP_PAIRS) +
    ["moisture"] * len(MOISTURE_PAIRS) +
    ["size"] * len(SIZE_PAIRS) +
    ["weight"] * len(WEIGHT_PAIRS) +
    ["speed"] * len(SPEED_PAIRS) +
    ["brightness"] * len(BRIGHT_PAIRS)
)

# Remove duplicates
seen = set()
unique_pairs = []
unique_cats = []
for pair, cat in zip(ALL_PAIRS, PAIR_CATEGORIES):
    key = (pair[0], pair[1])
    if key not in seen:
        seen.add(key)
        unique_pairs.append(pair)
        unique_cats.append(cat)

ALL_PAIRS = unique_pairs
PAIR_CATEGORIES = unique_cats
ALL_CATEGORIES = sorted(set(PAIR_CATEGORIES))

CORRUPTED_BASELINE = "The item"
TEMPLATE = "The {obj} is {attr}."


# ===== Math utilities (same as Phase 379) =====
def _silu(x):
    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -50, 50))))

def _gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))

def compute_pca_full(X):
    M = X - X.mean(axis=0, keepdims=True)
    try:
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
    except:
        return None, None, None, None, None
    total_var = np.sum(S**2)
    if total_var < 1e-10:
        return None, None, None, None, None
    explained = (S**2) / total_var
    eff_rank = int(np.searchsorted(np.cumsum(explained), 0.95) + 1)
    scores = U * S
    return explained, eff_rank, Vt, scores, S

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

def rms_norm_single(x, weight=None, eps=1e-6):
    d = x.shape[-1]
    rms = np.sqrt(np.mean(x**2) + eps)
    result = x / rms * np.sqrt(d)
    if weight is not None:
        result = result * weight
    return result


# ===== Model loading =====
def load_model_bf16(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = None
    for impl in ["flash_attention_2", "sdpa", "eager"]:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
                trust_remote_code=True, local_files_only=True, attn_implementation=impl)
            log(f"  Loaded with attn_impl={impl}")
            break
        except Exception as e:
            log(f"  Failed with {impl}: {str(e)[:80]}")
            continue
    if model is None:
        raise RuntimeError(f"Failed to load {model_name}")
    model.eval()
    return model, tokenizer


def load_mlp_weights(model, model_name, layer_idx):
    import glob
    from safetensors import safe_open
    layers = get_layers(model)
    mlp = layers[layer_idx].mlp
    gate_proj = getattr(mlp, "gate_proj", None)
    up_proj = getattr(mlp, "up_proj", None)
    down_proj = getattr(mlp, "down_proj", None)
    W_gate = W_up = W_down = None
    if gate_proj is not None and up_proj is not None and down_proj is not None:
        try:
            W_gate = gate_proj.weight.detach().cpu().float().numpy()
            W_up = up_proj.weight.detach().cpu().float().numpy()
            W_down = down_proj.weight.detach().cpu().float().numpy()
        except (NotImplementedError, RuntimeError):
            pass
    if W_gate is None:
        model_path = MODEL_CONFIGS[model_name]["path"]
        for sf_file in glob.glob(os.path.join(model_path, '*.safetensors')):
            try:
                with safe_open(sf_file, framework='pt', device='cpu') as sf:
                    for key in sf.keys():
                        if f"layers.{layer_idx}.mlp.gate_proj.weight" in key:
                            W_gate = sf.get_tensor(key).float().numpy()
                        elif f"layers.{layer_idx}.mlp.up_proj.weight" in key:
                            W_up = sf.get_tensor(key).float().numpy()
                        elif f"layers.{layer_idx}.mlp.down_proj.weight" in key:
                            W_down = sf.get_tensor(key).float().numpy()
                        elif f"layers.{layer_idx}.mlp.gate_up_proj.weight" in key:
                            full_w = sf.get_tensor(key).float().numpy()
                            half = full_w.shape[0] // 2
                            W_gate = full_w[:half]
                            W_up = full_w[half:]
                        elif f"layers.{layer_idx}.mlp.dense_h_to_4h.weight" in key:
                            full_w = sf.get_tensor(key).float().numpy()
                            half = full_w.shape[0] // 2
                            W_gate = full_w[:half]
                            W_up = full_w[half:]
                        elif f"layers.{layer_idx}.mlp.dense_4h_to_h.weight" in key:
                            W_down = sf.get_tensor(key).float().numpy()
                if W_gate is not None and W_down is not None:
                    break
            except:
                continue
    return W_gate, W_up, W_down


def _load_ln_weight(model, model_name, layer_idx):
    import glob
    from safetensors import safe_open
    layers = get_layers(model)
    for attr_name in ["post_attention_layernorm", "ln2", "input_layernorm"]:
        ln = getattr(layers[layer_idx], attr_name, None)
        if ln is not None:
            try:
                w = ln.weight.detach().cpu().float().numpy()
                if w is not None and len(w) > 0:
                    return w
            except (NotImplementedError, RuntimeError):
                pass
    model_path = MODEL_CONFIGS[model_name]["path"]
    for sf_file in glob.glob(os.path.join(model_path, '*.safetensors')):
        try:
            with safe_open(sf_file, framework='pt', device='cpu') as sf:
                for key in sf.keys():
                    for ln_name in ["post_attention_layernorm", "ln2", "input_layernorm"]:
                        if f"layers.{layer_idx}.{ln_name}.weight" in key:
                            return sf.get_tensor(key).float().numpy()
        except:
            continue
    log(f"    WARNING: Could not load LN weight for layer {layer_idx}")
    return None


def run_confirmation(model_name, target_layer=4):
    """Run confirmation test with expanded dataset."""
    log(f"\n{'='*70}")
    log(f"Phase 379b Confirmation: {model_name} Layer {target_layer}")
    log(f"Expanded dataset: {len(ALL_PAIRS)} pairs ({len(ALL_CATEGORIES)} categories)")
    log(f"{'='*70}")
    
    model, tokenizer = load_model_bf16(model_name)
    act_fn = "gelu" if model_name == "glm4" else "silu"
    layers = get_layers(model)
    input_device = next(model.parameters()).device
    n_pairs = len(ALL_PAIRS)
    
    W_gate, W_up, W_down = load_mlp_weights(model, model_name, target_layer)
    ln_weight = _load_ln_weight(model, model_name, target_layer)
    
    log(f"Collecting {n_pairs} pairs...")
    
    h_post_clean_list = []
    h_post_corrupt_list = []
    
    for pidx, (obj, target, competitor) in enumerate(ALL_PAIRS):
        if pidx % 40 == 0:
            log(f"  Pair {pidx+1}/{n_pairs}")
        
        clean_prompt = TEMPLATE.format(obj=obj, attr=target)
        corrupt_prompt = TEMPLATE.format(obj=CORRUPTED_BASELINE, attr=target)
        
        with torch.no_grad():
            clean_out = model(
                input_ids=tokenizer(clean_prompt, return_tensors="pt",
                                   truncation=True, max_length=64)["input_ids"].to(input_device),
                attention_mask=tokenizer(clean_prompt, return_tensors="pt",
                                        truncation=True, max_length=64)["attention_mask"].to(input_device),
                output_hidden_states=True)
        last_pos = tokenizer(clean_prompt, return_tensors="pt")["input_ids"].shape[1] - 1
        h_post_clean_list.append(clean_out.hidden_states[target_layer+1][0, last_pos].detach().cpu().float().numpy())
        del clean_out
        
        with torch.no_grad():
            corrupt_out = model(
                input_ids=tokenizer(corrupt_prompt, return_tensors="pt",
                                   truncation=True, max_length=64)["input_ids"].to(input_device),
                attention_mask=tokenizer(corrupt_prompt, return_tensors="pt",
                                        truncation=True, max_length=64)["attention_mask"].to(input_device),
                output_hidden_states=True)
        last_pos_r = tokenizer(corrupt_prompt, return_tensors="pt")["input_ids"].shape[1] - 1
        h_post_corrupt_list.append(corrupt_out.hidden_states[target_layer+1][0, last_pos_r].detach().cpu().float().numpy())
        del corrupt_out
        
        if pidx % 5 == 0:
            torch.cuda.empty_cache()
    
    h_post_clean = np.array(h_post_clean_list)
    h_post_corrupt = np.array(h_post_corrupt_list)
    dh = h_post_clean - h_post_corrupt
    
    # === PROPER post-RMSNorm ===
    h_clean_norm = np.zeros_like(h_post_clean)
    h_corrupt_norm = np.zeros_like(h_post_corrupt)
    for i in range(n_pairs):
        h_clean_norm[i] = rms_norm_single(h_post_clean[i], ln_weight)
        h_corrupt_norm[i] = rms_norm_single(h_post_corrupt[i], ln_weight)
    dh_proper = h_clean_norm - h_corrupt_norm
    
    # === PSEUDO post-RMSNorm ===
    dh_pseudo = np.zeros_like(dh)
    for i in range(n_pairs):
        dh_pseudo[i] = rms_norm_single(dh[i], ln_weight)
    
    # === Compute metrics ===
    def compute_metrics(dh_vec, label):
        pca, rank, Vt, scores, _ = compute_pca_full(dh_vec)
        same_cos, cross_cos = [], []
        for i in range(n_pairs):
            for j in range(i+1, n_pairs):
                cos = cosine_sim(dh_vec[i], dh_vec[j])
                if PAIR_CATEGORIES[i] == PAIR_CATEGORIES[j]:
                    same_cos.append(cos)
                else:
                    cross_cos.append(cos)
        
        # Category R² for PC1
        cat_onehot = np.zeros((n_pairs, len(ALL_CATEGORIES)))
        for i, cat in enumerate(PAIR_CATEGORIES):
            cat_onehot[i, ALL_CATEGORIES.index(cat)] = 1.0
        
        cat_r2_pc1 = None
        if scores is not None:
            try:
                beta = np.linalg.lstsq(cat_onehot, scores[:, 0], rcond=None)[0]
                pred = cat_onehot @ beta
                cat_r2_pc1 = float(1.0 - np.var(scores[:, 0] - pred) / np.var(scores[:, 0]))
            except:
                pass
        
        # norm_ratio correlation with PC1
        norm_ratio = np.sum(h_post_clean**2, axis=1) / (np.sum(h_post_corrupt**2, axis=1) + 1e-10)
        norm_ratio_corr = None
        if scores is not None:
            norm_ratio_corr = float(np.corrcoef(scores[:, 0], norm_ratio)[0, 1])
        
        log(f"  {label}: PC1={pca[0]:.4f}, rank={rank}, "
            f"same={np.mean(same_cos):.4f}, cross={np.mean(cross_cos):.4f}, "
            f"gap={np.mean(same_cos)-np.mean(cross_cos):.4f}, "
            f"cat_R²={cat_r2_pc1:.4f}, norm_ratio_corr={norm_ratio_corr:.4f}")
        
        return {
            "pc1": float(pca[0]) if pca is not None else None,
            "rank": rank,
            "same_cat": float(np.mean(same_cos)),
            "cross_cat": float(np.mean(cross_cos)),
            "gap": float(np.mean(same_cos) - np.mean(cross_cos)),
            "cat_r2_pc1": cat_r2_pc1,
            "norm_ratio_corr": norm_ratio_corr,
        }
    
    results = {
        "model": model_name,
        "phase": "379b",
        "n_pairs": n_pairs,
        "layer": target_layer,
        "raw": compute_metrics(dh, "raw"),
        "proper": compute_metrics(dh_proper, "PROPER"),
        "pseudo": compute_metrics(dh_pseudo, "PSEUDO"),
    }
    
    os.makedirs("results/phase379_rmsnorm_remapping", exist_ok=True)
    out_path = f"results/phase379_rmsnorm_remapping/{model_name}_phase379b.json"
    
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            if isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)
    
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, cls=NpEncoder, ensure_ascii=False)
    
    log(f"\nResults saved to {out_path}")
    
    release_model(model)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    return results


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "deepseek7b"
    run_confirmation(model_name)
