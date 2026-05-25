"""
Phase 282: Component Causal Patching with RoPE
===============================================
Key improvements over Phase 281:
1. Full RoPE implementation for manual attention (P0 critical fix)
2. 4-way causal patching: attn_weights, value_vectors, attn_out, mlp_out
3. 50+ SVO sentence pairs covering diverse syntactic categories
4. Component contribution matrix per layer

Usage:
  python tests/glm5/phase282_causal_patching_rope.py qwen3
  python tests/glm5/phase282_causal_patching_rope.py glm4
  python tests/glm5/phase282_causal_patching_rope.py deepseek7b
"""
import sys, os, json, gc, time, warnings
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from model_utils import MODEL_CONFIGS, get_model_info, get_layers, release_model

RESULT_DIR = Path("results/phase282_causal_patching")
RESULT_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR = Path("tmp")
TMP_DIR.mkdir(parents=True, exist_ok=True)

_log_file = None
def log_time(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if _log_file:
        with open(_log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")

# =============================================================================
# RoPE Implementation (GPT-NeoX style: half-half pairing)
# =============================================================================

def compute_rope_cos_sin(seq_len, head_dim, rotary_dim, base=10000.0):
    """
    Compute RoPE cos/sin matrices.
    
    GPT-NeoX style (rotary_adjacent_pairs=False):
    - Split rotary_dim into two halves
    - freq = base^(-2*dim/rotary_dim) for dim in [0, rotary_dim/2)
    - half1 gets cos, half2 gets sin (via rotate_every_two)
    
    If rotary_dim < head_dim (partial RoPE): remaining dims are identity.
    
    Returns: cos [seq_len, head_dim], sin [seq_len, head_dim]
    """
    # Frequencies for rotary_dim/2 dims
    d = np.arange(rotary_dim // 2)
    freq = base ** (-2.0 * d / rotary_dim)  # [rotary_dim/2]
    
    # Positions
    pos = np.arange(seq_len)
    angles = np.outer(pos, freq)  # [seq_len, rotary_dim/2]
    
    # NeoX: duplicate to fill rotary_dim
    # [f0, f1, f2, ...] → [f0, f1, f2, ..., f0, f1, f2, ...]
    angles_full = np.concatenate([angles, angles], axis=-1)  # [seq_len, rotary_dim]
    
    cos = np.cos(angles_full).astype(np.float32)
    sin = np.sin(angles_full).astype(np.float32)
    
    # Pad to head_dim if partial RoPE
    if rotary_dim < head_dim:
        pad_dim = head_dim - rotary_dim
        cos_pad = np.ones((seq_len, pad_dim), dtype=np.float32)
        sin_pad = np.zeros((seq_len, pad_dim), dtype=np.float32)
        cos = np.concatenate([cos, cos_pad], axis=-1)
        sin = np.concatenate([sin, sin_pad], axis=-1)
    
    return cos, sin


def rotate_every_two(x):
    """
    GPT-NeoX style rotation: swap halves and negate the second half.
    x: [..., head_dim] where head_dim is even.
    Returns: [-x2, x1] for x = [x1, x2]
    """
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return np.concatenate([-x2, x1], axis=-1)


def apply_rope(x, cos, sin, rotary_dim):
    """
    Apply RoPE to query/key tensor.
    x: [..., head_dim]
    cos, sin: [seq_len, head_dim] (padded to head_dim for partial RoPE)
    Only first rotary_dim dimensions are rotated.
    """
    x_rot = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]
    
    cos_rot = cos[..., :rotary_dim]
    sin_rot = sin[..., :rotary_dim]
    
    x_flip = rotate_every_two(x_rot)
    x_rotated = x_rot * cos_rot + x_flip * sin_rot
    
    if rotary_dim < x.shape[-1]:
        return np.concatenate([x_rotated, x_pass], axis=-1)
    return x_rotated


# =============================================================================
# Model-specific RoPE parameters
# =============================================================================
MODEL_ROPE_CONFIGS = {
    "qwen3": {
        "rope_theta": 1000000.0,
        "rotary_dim": 128,         # full RoPE (head_dim=128)
        "partial_rotary_factor": 1.0,
        "has_qk_norm": True,
    },
    "glm4": {
        "rope_theta": 10000.0,
        "rotary_dim": 64,          # partial RoPE (head_dim=128, factor=0.5)
        "partial_rotary_factor": 0.5,
        "has_qk_norm": False,
    },
    "deepseek7b": {
        "rope_theta": 10000.0,
        "rotary_dim": 128,         # full RoPE (head_dim=128)
        "partial_rotary_factor": 1.0,
        "has_qk_norm": False,
    },
}


# =============================================================================
# 50+ Diverse SVO Pairs (8 categories)
# =============================================================================

def build_svo_pairs():
    """
    Build 50+ diverse sentence pairs covering:
    - Animal swap, Human swap, Object swap, Place swap
    - Passive voice, Negation, Quantifier, Abstract/Clause
    Returns list of {"name", "A", "B", "category"}
    """
    pairs = []
    
    # Category 1: Animal-animal swap (9 pairs)
    animal_templates = [
        ("dog_cat_chase", "the dog chases the cat", "the cat chases the dog"),
        ("wolf_sheep_hunt", "the wolf hunts the sheep", "the sheep hunts the wolf"),
        ("lion_deer_chase", "the lion chases the deer", "the deer chases the lion"),
        ("bird_snake_eat", "the bird eats the snake", "the snake eats the bird"),
        ("cat_mouse_catch", "the cat catches the mouse", "the mouse catches the cat"),
        ("fox_rabbit_hunt", "the fox hunts the rabbit", "the rabbit hunts the fox"),
        ("eagle_fish_catch", "the eagle catches the fish", "the fish catches the eagle"),
        ("bear_salmon_eat", "the bear eats the salmon", "the salmon eats the bear"),
        ("shark_seal_hunt", "the shark hunts the seal", "the seal hunts the shark"),
    ]
    for name, a, b in animal_templates:
        pairs.append({"name": name, "A": a, "B": b, "category": "animal"})
    
    # Category 2: Human-human swap (8 pairs)
    human_templates = [
        ("man_woman_love", "the man loves the woman", "the woman loves the man"),
        ("boy_girl_call", "the boy calls the girl", "the girl calls the boy"),
        ("teacher_student_teach", "the teacher teaches the student", "the student teaches the teacher"),
        ("mother_child_hold", "the mother holds the child", "the child holds the mother"),
        ("father_son_guide", "the father guides the son", "the son guides the father"),
        ("doctor_patient_help", "the doctor helps the patient", "the patient helps the doctor"),
        ("king_queen_trust", "the king trusts the queen", "the queen trusts the king"),
        ("brother_sister_call", "the brother calls the sister", "the sister calls the brother"),
    ]
    for name, a, b in human_templates:
        pairs.append({"name": name, "A": a, "B": b, "category": "human"})
    
    # Category 3: Human-object swap (6 pairs)
    obj_templates = [
        ("child_apple_eat", "the child eats the apple", "the apple eats the child"),
        ("chef_knife_use", "the chef uses the knife", "the knife uses the chef"),
        ("painter_brush_hold", "the painter holds the brush", "the brush holds the painter"),
        ("driver_car_control", "the driver controls the car", "the car controls the driver"),
        ("writer_pen_use", "the writer uses the pen", "the pen uses the writer"),
        ("guard_key_hold", "the guard holds the key", "the key holds the guard"),
    ]
    for name, a, b in obj_templates:
        pairs.append({"name": name, "A": a, "B": b, "category": "human_object"})
    
    # Category 4: Human-place swap (6 pairs)
    place_templates = [
        ("king_city_rule", "the king rules the city", "the city rules the king"),
        ("explorer_island_discover", "the explorer discovers the island", "the island discovers the explorer"),
        ("tourist_museum_visit", "the tourist visits the museum", "the museum visits the tourist"),
        ("guard_prison_watch", "the guard watches the prison", "the prison watches the guard"),
        ("soldier_bridge_defend", "the soldier defends the bridge", "the bridge defends the soldier"),
        ("mayor_town_govern", "the mayor governs the town", "the town governs the mayor"),
    ]
    for name, a, b in place_templates:
        pairs.append({"name": name, "A": a, "B": b, "category": "place"})
    
    # Category 5: Passive voice (5 pairs)
    passive_templates = [
        ("dog_chase_passive", "the dog chases the cat", "the cat is chased by the dog"),
        ("teacher_teach_passive", "the teacher teaches the student", "the student is taught by the teacher"),
        ("wolf_hunt_passive", "the wolf hunts the sheep", "the sheep is hunted by the wolf"),
        ("king_rule_passive", "the king rules the city", "the city is ruled by the king"),
        ("mother_hold_passive", "the mother holds the child", "the child is held by the mother"),
    ]
    for name, a, b in passive_templates:
        pairs.append({"name": name, "A": a, "B": b, "category": "passive"})
    
    # Category 6: Negation (7 pairs)
    negation_templates = [
        ("happy_not", "the man is happy", "the man is not happy"),
        ("reason_no", "there is a reason", "there is no reason"),
        ("rain_not", "it will rain", "it will not rain"),
        ("agree_no", "they agree", "they do not agree"),
        ("safe_not", "the place is safe", "the place is not safe"),
        ("found_nothing", "she found something", "she found nothing"),
        ("possible_not", "it is possible", "it is not possible"),
    ]
    for name, a, b in negation_templates:
        pairs.append({"name": name, "A": a, "B": b, "category": "negation"})
    
    # Category 7: Quantifier/Intensity swap (6 pairs)
    quantifier_templates = [
        ("few_many", "a few people came", "many people came"),
        ("some_all", "some students passed", "all students passed"),
        ("slow_fast", "the car is slow", "the car is fast"),
        ("small_large", "the house is small", "the house is large"),
        ("quiet_loud", "the music is quiet", "the music is loud"),
        ("cold_hot", "the water is cold", "the water is hot"),
    ]
    for name, a, b in quantifier_templates:
        pairs.append({"name": name, "A": a, "B": b, "category": "quantifier"})
    
    # Category 8: Abstract / Metaphorical (5 pairs)
    abstract_templates = [
        ("justice_corruption", "justice defeats corruption", "corruption defeats justice"),
        ("wisdom_folly", "wisdom guides folly", "folly guides wisdom"),
        ("courage_fear", "courage conquers fear", "fear conquers courage"),
        ("love_hate", "love overcomes hate", "hate overcomes love"),
        ("truth_lie", "truth exposes the lie", "the lie exposes the truth"),
    ]
    for name, a, b in abstract_templates:
        pairs.append({"name": name, "A": a, "B": b, "category": "abstract"})
    
    return pairs


# =============================================================================
# Model Loading
# =============================================================================

def load_model_bf16_flash(model_name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    log_time(f"Loading {model_name} (bf16 + flash_attention_2)...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    attn_impl = "flash_attention_2"
    try:
        import flash_attn
        log_time("  flash_attn available, using flash_attention_2")
    except ImportError:
        attn_impl = "eager"
        log_time("  flash_attn not found, using eager")
    
    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation=attn_impl,
    )
    
    device = next(model.parameters()).device
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    log_time(f"  {model_name} loaded: device={device}, GPU={gpu_mem:.2f}GB, attn={attn_impl}")
    return model, tokenizer, device


# =============================================================================
# Hook utilities
# =============================================================================

def run_with_layer_hooks(model, tokenizer, device, sentence, layers_to_hook):
    """Capture hidden states, attn_out, mlp_out, attn_in for specified layers."""
    inputs = tokenizer(sentence, return_tensors="pt").to(device)
    
    captured_hidden = {}
    captured_attn_in = {}
    captured_attn_out = {}
    captured_mlp_out = {}
    
    layers = get_layers(model)
    
    def make_hidden_hook(key):
        def hook(module, input_t, output_t):
            if isinstance(input_t, tuple) and len(input_t) > 0:
                captured_hidden[key] = input_t[0].detach().float().cpu()
        return hook
    
    def make_attn_hook(key):
        def hook(module, input_t, output_t):
            if isinstance(input_t, tuple) and len(input_t) > 0:
                captured_attn_in[key] = input_t[0].detach().float().cpu()
            if isinstance(output_t, tuple):
                captured_attn_out[key] = output_t[0].detach().float().cpu()
            else:
                captured_attn_out[key] = output_t.detach().float().cpu()
        return hook
    
    def make_mlp_hook(key):
        def hook(module, input_t, output_t):
            if isinstance(output_t, tuple):
                captured_mlp_out[key] = output_t[0].detach().float().cpu()
            else:
                captured_mlp_out[key] = output_t.detach().float().cpu()
        return hook
    
    all_hooks = []
    for li in layers_to_hook:
        if li < len(layers):
            layer = layers[li]
            all_hooks.append(layer.register_forward_hook(make_hidden_hook(f"L{li}")))
            if hasattr(layer, 'self_attn'):
                all_hooks.append(layer.self_attn.register_forward_hook(make_attn_hook(f"L{li}")))
            if hasattr(layer, 'mlp'):
                all_hooks.append(layer.mlp.register_forward_hook(make_mlp_hook(f"L{li}")))
    
    with torch.no_grad():
        try:
            _ = model(**inputs)
        except Exception as e:
            log_time(f"  Forward failed: {e}")
    
    for h in all_hooks:
        h.remove()
    
    return {
        "hidden": captured_hidden,
        "attn_in": captured_attn_in,
        "attn_out": captured_attn_out,
        "mlp_out": captured_mlp_out,
        "input_ids": inputs.input_ids.cpu(),
    }


def run_patched_forward(model, tokenizer, device, sentence, patch_info):
    """
    Run forward with component patching.
    patch_info = {"type": "attn_out"|"mlp_out", "layer": int, "replacement": tensor}
    Returns logits [1, seq, vocab] as float32 CPU numpy
    
    NOTE: self_attn hook must return (output, None) tuple because layer forward
    unpacks: hidden_states, self_attn_weights = self.self_attn(...)
    """
    inputs = tokenizer(sentence, return_tensors="pt").to(device)
    layers = get_layers(model)
    li = patch_info["layer"]
    layer = layers[li]
    
    if patch_info["type"] == "attn_out":
        replacement = patch_info["replacement"].to(device).to(model.dtype)
        # self_attn returns (attn_output, attn_weights_or_None) → hook must return tuple
        def patch_hook(module, input_t, output_t):
            # Preserve the second element (attn_weights or None) from original output
            if isinstance(output_t, tuple) and len(output_t) > 1:
                return (replacement, output_t[1])
            return (replacement,)
        hook = layer.self_attn.register_forward_hook(patch_hook)
    elif patch_info["type"] == "mlp_out":
        replacement = patch_info["replacement"].to(device).to(model.dtype)
        def patch_hook(module, input_t, output_t):
            return replacement
        hook = layer.mlp.register_forward_hook(patch_hook)
    else:
        raise ValueError(f"Unknown patch type: {patch_info['type']}")
    
    with torch.no_grad():
        try:
            out = model(**inputs)
        except Exception as e:
            log_time(f"  Patched forward L{li} {patch_info['type']} failed: {e}")
            hook.remove()
            return None
    
    hook.remove()
    return out.logits.float().cpu().numpy()


# =============================================================================
# LN Application (for manual attention)
# =============================================================================

def apply_input_ln(hidden_np, layer, eps=1e-6):
    """Apply input layernorm/RMSNorm."""
    ln = None
    for ln_name in ["input_layernorm", "ln_1", "layernorm"]:
        if hasattr(layer, ln_name):
            ln = getattr(layer, ln_name)
            break
    if ln is None:
        return hidden_np
    
    try:
        w = ln.weight.detach().cpu().float().numpy()
    except (NotImplementedError, RuntimeError):
        return hidden_np
    
    has_bias = hasattr(ln, 'bias') and ln.bias is not None
    if has_bias:
        b = ln.bias.detach().cpu().float().numpy()
        mean = np.mean(hidden_np, axis=-1, keepdims=True)
        var = np.var(hidden_np, axis=-1, keepdims=True)
        return ((hidden_np - mean) / np.sqrt(var + eps)) * w + b
    else:
        rms = np.sqrt(np.mean(hidden_np ** 2, axis=-1, keepdims=True) + eps)
        return hidden_np * w / rms


def softmax_np(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


# =============================================================================
# Block B1+B2: Manual Attention with RoPE (Weight & Value Patching)
# =============================================================================

def block_b12_manual_rope_patching(model, tokenizer, device, model_info, model_name, all_pairs):
    """
    B1: attn_weight_patch — fix V, swap attention weights (routing effect)
    B2: value_patch — fix attention weights, swap V (content effect)
    
    Uses manual Q/K/V/O with RoPE-corrected attention computation.
    """
    n_layers = model_info.n_layers
    rope_cfg = MODEL_ROPE_CONFIGS[model_name]
    rope_base = rope_cfg["rope_theta"]
    rotary_dim = rope_cfg["rotary_dim"]
    has_qk_norm = rope_cfg["has_qk_norm"]
    
    # Sample layers
    sample_layers = sorted(set([0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]))
    
    log_time(f"\n{'='*50}")
    log_time(f"Block B1+B2: Manual RoPE Patching (Weight/Value)")
    log_time(f"  Layers: {sample_layers}, Pairs: {len(all_pairs)}")
    layers = get_layers(model)
    d_head_rope = rotary_dim
    log_time(f"  RoPE: base={rope_base}, rotary_dim={d_head_rope}")
    
    # Warmup forward
    log_time("  Warmup forward...")
    dummy = tokenizer("hello world", return_tensors="pt").to(device)
    with torch.no_grad():
        try: _ = model(**dummy)
        except: pass
    
    # Extract layer weights
    layer_weights = {}
    for li in sample_layers:
        layer = layers[li]
        sa = layer.self_attn
        try:
            W_q = sa.q_proj.weight.detach().cpu().float().numpy()
            W_k = sa.k_proj.weight.detach().cpu().float().numpy()
            W_v = sa.v_proj.weight.detach().cpu().float().numpy()
            W_o = sa.o_proj.weight.detach().cpu().float().numpy()
        except (NotImplementedError, RuntimeError) as e:
            log_time(f"  L{li}: weight failed ({e}), skip")
            continue
        
        n_heads_q = getattr(sa, 'num_heads', getattr(sa, 'num_attention_heads', 
                          getattr(model.config, 'num_attention_heads', 32)))
        n_heads_kv = getattr(sa, 'num_key_value_heads', 
                            getattr(model.config, 'num_key_value_heads', n_heads_q))
        d_head = W_q.shape[0] // n_heads_q
        gqa_group = n_heads_q // n_heads_kv if n_heads_kv > 0 else 1
        
        layer_weights[li] = {
            "W_q": W_q, "W_k": W_k, "W_v": W_v, "W_o": W_o,
            "n_heads_q": n_heads_q, "n_heads_kv": n_heads_kv,
            "d_head": d_head, "gqa_group": gqa_group,
        }
        log_time(f"  L{li}: Q={n_heads_q}, KV={n_heads_kv}, d_head={d_head}, gqa={gqa_group}")
    
    # Pre-compute QK norm weights if needed
    qk_norms = {}
    if has_qk_norm:
        for li in sample_layers:
            layer = layers[li]
            sa = layer.self_attn
            if hasattr(sa, 'q_norm'):
                try:
                    q_norm_w = sa.q_norm.weight.detach().cpu().float().numpy()
                    k_norm_w = sa.k_norm.weight.detach().cpu().float().numpy()
                    qk_norms[li] = {"q_norm": q_norm_w, "k_norm": k_norm_w}
                except (NotImplementedError, RuntimeError):
                    pass
    
    results = {}
    
    for pair in all_pairs:
        pname = pair["name"]
        sent_a, sent_b = pair["A"], pair["B"]
        category = pair.get("category", "unknown")
        
        log_time(f"  [{category}] {pname} ...")
        
        # Run both sentences with hooks
        rA = run_with_layer_hooks(model, tokenizer, device, sent_a, sample_layers)
        rB = run_with_layer_hooks(model, tokenizer, device, sent_b, sample_layers)
        
        pair_results = {"name": pname, "category": category,
                        "sent_A": sent_a, "sent_B": sent_b,
                        "layers": {}}
        
        for li in sample_layers:
            key = f"L{li}"
            if li not in layer_weights:
                continue
            if key not in rA["hidden"] or key not in rB["hidden"]:
                continue
            
            w = layer_weights[li]
            # Use hidden (pre-LN layer input) and apply LN manually
            # Skip attn_in check — we use hidden+LN instead (more reliable)
            hA_pre = rA["hidden"][key][0].numpy()
            hB_pre = rB["hidden"][key][0].numpy()
            
            # Apply LN
            layer = layers[li]
            hA = apply_input_ln(hA_pre, layer)
            hB = apply_input_ln(hB_pre, layer)
            seqA, seqB = hA.shape[0], hB.shape[0]
            min_seq = min(seqA, seqB)
            
            # Get weights
            W_q = w["W_q"]
            W_k = w["W_k"]
            W_v = w["W_v"]
            W_o = w["W_o"]
            n_q = w["n_heads_q"]
            n_kv = w["n_heads_kv"]
            d_head = w["d_head"]
            gqa = w["gqa_group"]
            
            # Compute Q, K, V
            QA = hA[:min_seq] @ W_q.T  # [min_seq, n_q*d_head]
            KA = hA[:min_seq] @ W_k.T  # [min_seq, n_kv*d_head]
            VA = hA[:min_seq] @ W_v.T
            
            QB = hB[:min_seq] @ W_q.T
            KB = hB[:min_seq] @ W_k.T
            VB = hB[:min_seq] @ W_v.T
            
            # Reshape to multi-head
            def to_mha(X, n_h, d_h):
                s = X.shape[0]
                return X.reshape(s, n_h, d_h).transpose(1, 0, 2)
            
            QA_mha = to_mha(QA, n_q, d_head)
            KA_mha_raw = to_mha(KA, n_kv, d_head)
            VA_mha_raw = to_mha(VA, n_kv, d_head)
            
            QB_mha = to_mha(QB, n_q, d_head)
            KB_mha_raw = to_mha(KB, n_kv, d_head)
            VB_mha_raw = to_mha(VB, n_kv, d_head)
            
            # Apply QK Norm (Qwen3)
            if has_qk_norm and li in qk_norms:
                qn = qk_norms[li]["q_norm"]
                kn = qk_norms[li]["k_norm"]
                eps = 1e-6
                for h in range(n_q):
                    q_rms = np.sqrt(np.mean(QA_mha[h]**2) + eps)
                    QA_mha[h] = QA_mha[h] * qn / q_rms
                    q_rms = np.sqrt(np.mean(QB_mha[h]**2) + eps)
                    QB_mha[h] = QB_mha[h] * qn / q_rms
                for h in range(n_kv):
                    k_rms = np.sqrt(np.mean(KA_mha_raw[h]**2) + eps)
                    KA_mha_raw[h] = KA_mha_raw[h] * kn / k_rms
                    k_rms = np.sqrt(np.mean(KB_mha_raw[h]**2) + eps)
                    KB_mha_raw[h] = KB_mha_raw[h] * kn / k_rms
            
            # GQA expand K/V
            if gqa > 1:
                KA_mha = np.repeat(KA_mha_raw, gqa, axis=0)
                VA_mha = np.repeat(VA_mha_raw, gqa, axis=0)
                KB_mha = np.repeat(KB_mha_raw, gqa, axis=0)
                VB_mha = np.repeat(VB_mha_raw, gqa, axis=0)
            else:
                KA_mha = KA_mha_raw
                VA_mha = VA_mha_raw
                KB_mha = KB_mha_raw
                VB_mha = VB_mha_raw
            
            # Apply RoPE to Q and K
            rope_cos, rope_sin = compute_rope_cos_sin(min_seq, d_head, rotary_dim, rope_base)
            
            for h in range(n_q):
                QA_mha[h] = apply_rope(QA_mha[h], rope_cos, rope_sin, rotary_dim)
                QB_mha[h] = apply_rope(QB_mha[h], rope_cos, rope_sin, rotary_dim)
                KA_mha[h] = apply_rope(KA_mha[h], rope_cos, rope_sin, rotary_dim)
                KB_mha[h] = apply_rope(KB_mha[h], rope_cos, rope_sin, rotary_dim)
            
            # Attention weights
            scale = np.sqrt(d_head)
            AA = np.zeros((n_q, min_seq, min_seq))
            AB = np.zeros((n_q, min_seq, min_seq))
            for h in range(n_q):
                AA[h] = softmax_np(QA_mha[h] @ KA_mha[h].T / scale)
                AB[h] = softmax_np(QB_mha[h] @ KB_mha[h].T / scale)
            
            # === B1: Weight Patching (attn_weight swap) ===
            # (AA ⊗ VB): routing=A, content=B  → measures how much content carries the diff
            # (AB ⊗ VA): routing=B, content=A  → measures how much routing carries the diff
            mixed_AW_BV = np.zeros((n_q, min_seq, d_head))  # AA @ VB
            mixed_BW_AV = np.zeros((n_q, min_seq, d_head))  # AB @ VA
            pure_AA_VA = np.zeros((n_q, min_seq, d_head))
            pure_AB_VB = np.zeros((n_q, min_seq, d_head))
            
            for h in range(n_q):
                mixed_AW_BV[h] = AA[h] @ VB_mha[h]    # A's routing + B's content
                mixed_BW_AV[h] = AB[h] @ VA_mha[h]    # B's routing + A's content
                pure_AA_VA[h] = AA[h] @ VA_mha[h]
                pure_AB_VB[h] = AB[h] @ VB_mha[h]
            
            # Project through W_o
            def flatten_mha(X, n_h, s, d_h):
                return X.transpose(1, 0, 2).reshape(s, n_h * d_h)
            
            mixed_AW_BV_o = flatten_mha(mixed_AW_BV, n_q, min_seq, d_head) @ W_o.T
            mixed_BW_AV_o = flatten_mha(mixed_BW_AV, n_q, min_seq, d_head) @ W_o.T
            pure_AA_VA_o = flatten_mha(pure_AA_VA, n_q, min_seq, d_head) @ W_o.T
            pure_AB_VB_o = flatten_mha(pure_AB_VB, n_q, min_seq, d_head) @ W_o.T
            
            # Compute effects
            total_gap = float(np.linalg.norm(pure_AA_VA_o - pure_AB_VB_o))
            weight_effect_raw = float(np.linalg.norm(mixed_BW_AV_o - pure_AB_VB_o))
            value_effect_raw = float(np.linalg.norm(mixed_AW_BV_o - pure_AB_VB_o))
            weight_value_mix_raw = float(np.linalg.norm(mixed_AW_BV_o - pure_AA_VA_o))
            
            if total_gap > 1e-10:
                weight_effect = weight_effect_raw / total_gap  # higher = weight swap matters more
                value_effect = value_effect_raw / total_gap    # higher = value swap matters more
                wv_mix = weight_value_mix_raw / total_gap
            else:
                weight_effect = value_effect = wv_mix = 0.0
            
            # Last token specific
            lt = min_seq - 1
            lt_gap = float(np.linalg.norm(pure_AA_VA_o[lt] - pure_AB_VB_o[lt]))
            if lt_gap > 1e-10:
                lt_weight_eff = float(np.linalg.norm(mixed_BW_AV_o[lt] - pure_AB_VB_o[lt])) / lt_gap
                lt_value_eff = float(np.linalg.norm(mixed_AW_BV_o[lt] - pure_AB_VB_o[lt])) / lt_gap
            else:
                lt_weight_eff = lt_value_eff = 0.0
            
            pair_results["layers"][str(li)] = {
                "total_gap": total_gap,
                "weight_effect": weight_effect,
                "value_effect": value_effect,
                "weight_value_ratio": weight_effect / max(value_effect, 1e-10),
                "last_token": {
                    "weight_effect": lt_weight_eff,
                    "value_effect": lt_value_eff,
                    "gap": lt_gap,
                },
                "content_dominates": value_effect > weight_effect,
            }
        
        results[pname] = pair_results
        log_time(f"    {pname}: {len(pair_results['layers'])} layers")
    
    # Aggregate by category
    agg = {"per_layer": {}, "per_category": defaultdict(lambda: defaultdict(list))}
    for li in sample_layers:
        key = str(li)
        w_effs = []; v_effs = []; lt_w = []; lt_v = []; doms = []
        for pname, pr in results.items():
            if key in pr.get("layers", {}):
                lr = pr["layers"][key]
                w_effs.append(lr["weight_effect"])
                v_effs.append(lr["value_effect"])
                lt_w.append(lr["last_token"]["weight_effect"])
                lt_v.append(lr["last_token"]["value_effect"])
                doms.append(1.0 if lr["content_dominates"] else 0.0)
                # Per category
                cat = pr.get("category", "unknown")
                agg["per_category"][cat]["weight"].append(lr["weight_effect"])
                agg["per_category"][cat]["value"].append(lr["value_effect"])
        
        if w_effs:
            agg["per_layer"][key] = {
                "weight_effect_mean": float(np.mean(w_effs)),
                "value_effect_mean": float(np.mean(v_effs)),
                "last_token_weight": float(np.mean(lt_w)),
                "last_token_value": float(np.mean(lt_v)),
                "content_dominance_rate": float(np.mean(doms)),
            }
    
    # Log
    log_time(f"\n  Manual RoPE Patching Summary:")
    log_time(f"  {'Layer':>5} {'wt_eff':>8} {'val_eff':>8} {'lt_wt':>8} {'lt_val':>8} {'dom%':>6} {'winner':>20}")
    for li in sample_layers:
        key = str(li)
        if key in agg["per_layer"]:
            a = agg["per_layer"][key]
            winner = "VALUE>>weight" if a["content_dominance_rate"] > 0.5 else "WEIGHT>>value"
            log_time(f"  L{li:>4} {a['weight_effect_mean']:8.3f} {a['value_effect_mean']:8.3f} "
                     f"{a['last_token_weight']:8.3f} {a['last_token_value']:8.3f} "
                     f"{a['content_dominance_rate']:6.2f} {winner:>20}")
    
    # Category breakdown
    log_time(f"\n  Per-category weight/value effects:")
    for cat in sorted(agg["per_category"].keys()):
        wm = np.mean(agg["per_category"][cat]["weight"])
        vm = np.mean(agg["per_category"][cat]["value"])
        dom = "VALUE" if vm > wm else "WEIGHT"
        log_time(f"    {cat:>15}: weight={wm:.3f}, value={vm:.3f} → {dom} dominates")
    
    return {"pairs": results, "aggregate": agg}


# =============================================================================
# Block B3+B4: Hook-based Output Patching (attn_out / mlp_out)
# =============================================================================

def block_b34_output_patching(model, tokenizer, device, model_info, all_pairs):
    """
    B3: attn_out_patch — replace attention output at layer L with B's attn_out
    B4: mlp_out_patch — replace MLP output at layer L with B's mlp_out
    
    Measures logit distribution shift after patching.
    effect = ||logits_patched - logits_B|| / ||logits_A - logits_B||
    Lower effect → patching this component brings A closer to B
    """
    n_layers = model_info.n_layers
    # More layers for output patching (it's faster than manual attention)
    sample_layers = list(range(0, n_layers, max(1, n_layers // 10)))
    sample_layers = sorted(set(sample_layers + [n_layers - 1]))
    
    log_time(f"\n{'='*50}")
    log_time(f"Block B3+B4: Output Patching (attn_out / mlp_out)")
    log_time(f"  Layers: {sample_layers}, Pairs: {len(all_pairs)}")
    
    results = {}
    
    for pair_idx, pair in enumerate(all_pairs):
        pname = pair["name"]
        sent_a, sent_b = pair["A"], pair["B"]
        category = pair.get("category", "unknown")
        
        log_time(f"  [{category}] {pname} ({pair_idx+1}/{len(all_pairs)}) ...")
        
        # Capture all the outputs we need from A and B
        rA = run_with_layer_hooks(model, tokenizer, device, sent_a, sample_layers)
        rB = run_with_layer_hooks(model, tokenizer, device, sent_b, sample_layers)
        
        # Baseline logits
        inputs_A = tokenizer(sent_a, return_tensors="pt").to(device)
        inputs_B = tokenizer(sent_b, return_tensors="pt").to(device)
        with torch.no_grad():
            logits_A = model(**inputs_A).logits.float().cpu().numpy()
            logits_B = model(**inputs_B).logits.float().cpu().numpy()
        
        min_seq_logits = min(logits_A.shape[1], logits_B.shape[1])
        logits_A_trunc = logits_A[0, :min_seq_logits, :]
        logits_B_trunc = logits_B[0, :min_seq_logits, :]
        total_logit_gap = float(np.linalg.norm(logits_A_trunc - logits_B_trunc))
        
        pair_results = {"name": pname, "category": category,
                        "sent_A": sent_a, "sent_B": sent_b,
                        "total_logit_gap": total_logit_gap,
                        "layers": {}}
        
        for li in sample_layers:
            key = f"L{li}"
            if key not in rA["attn_out"] or key not in rB["attn_out"]:
                continue
            if key not in rA["mlp_out"] or key not in rB["mlp_out"]:
                continue
            
            attn_out_B = rB["attn_out"][key]
            mlp_out_B = rB["mlp_out"][key]
            
            # Skip if sequence lengths don't match
            # attn_out_B: [1, seq_B, d_model], sent_a: [1, seq_A, d_model]
            seq_B = attn_out_B.shape[1]
            inputs_A_check = tokenizer(sent_a, return_tensors="pt")
            seq_A = inputs_A_check.input_ids.shape[1]
            if seq_A != seq_B:
                continue  # Can't patch with mismatched sequence lengths
            
            layer_results = {}
            
            # B3: Patch attn_out
            patch_info = {"type": "attn_out", "layer": li, "replacement": attn_out_B}
            logits_attn = run_patched_forward(model, tokenizer, device, sent_a, patch_info)
            
            if logits_attn is not None:
                logits_attn_trunc = logits_attn[0, :min_seq_logits, :]
                attn_shift = float(np.linalg.norm(logits_attn_trunc - logits_B_trunc))
                if total_logit_gap > 1e-10:
                    attn_effect = attn_shift / total_logit_gap
                else:
                    attn_effect = 1.0
                layer_results["attn_out_patch_effect"] = attn_effect
                layer_results["attn_out_patch_raw"] = attn_shift
            
            # B4: Patch mlp_out
            patch_info = {"type": "mlp_out", "layer": li, "replacement": mlp_out_B}
            logits_mlp = run_patched_forward(model, tokenizer, device, sent_a, patch_info)
            
            if logits_mlp is not None:
                logits_mlp_trunc = logits_mlp[0, :min_seq_logits, :]
                mlp_shift = float(np.linalg.norm(logits_mlp_trunc - logits_B_trunc))
                if total_logit_gap > 1e-10:
                    mlp_effect = mlp_shift / total_logit_gap
                else:
                    mlp_effect = 1.0
                layer_results["mlp_out_patch_effect"] = mlp_effect
                layer_results["mlp_out_patch_raw"] = mlp_shift
            
            if layer_results:
                pair_results["layers"][str(li)] = layer_results
        
        results[pname] = pair_results
        
        # Log every 10 pairs
        if (pair_idx + 1) % 10 == 0:
            log_time(f"    {pair_idx+1}/{len(all_pairs)} done, GPU={torch.cuda.memory_allocated()/1e9:.1f}GB")
    
    # Aggregate
    agg = {"per_layer": {}, "per_category": defaultdict(lambda: defaultdict(list))}
    for li in sample_layers:
        key = str(li)
        attn_effects = []; mlp_effects = []
        for pname, pr in results.items():
            if key in pr.get("layers", {}):
                lr = pr["layers"][key]
                if "attn_out_patch_effect" in lr:
                    attn_effects.append(lr["attn_out_patch_effect"])
                if "mlp_out_patch_effect" in lr:
                    mlp_effects.append(lr["mlp_out_patch_effect"])
                cat = pr.get("category", "unknown")
                if "attn_out_patch_effect" in lr:
                    agg["per_category"][cat]["attn_out"].append(lr["attn_out_patch_effect"])
                if "mlp_out_patch_effect" in lr:
                    agg["per_category"][cat]["mlp_out"].append(lr["mlp_out_patch_effect"])
        
        if attn_effects and mlp_effects:
            agg["per_layer"][key] = {
                "attn_out_effect_mean": float(np.mean(attn_effects)),
                "mlp_out_effect_mean": float(np.mean(mlp_effects)),
                "attn_out_effect_std": float(np.std(attn_effects)),
                "mlp_out_effect_std": float(np.std(mlp_effects)),
                "n_pairs": len(attn_effects),
            }
    
    # Log summary (interpretation: lower effect → more important for role)
    log_time(f"\n  Output Patching Summary (lower effect = more role-critical):")
    log_time(f"  {'Layer':>5} {'attn_eff':>10} {'mlp_eff':>10} {'winner':>20} {'n':>5}")
    for li in sample_layers:
        key = str(li)
        if key in agg["per_layer"]:
            a = agg["per_layer"][key]
            winner = "ATTN_OUT" if a["attn_out_effect_mean"] < a["mlp_out_effect_mean"] else "MLP_OUT"
            log_time(f"  L{li:>4} {a['attn_out_effect_mean']:10.3f} {a['mlp_out_effect_mean']:10.3f} "
                     f"{winner:>20} {a['n_pairs']:>5}")
    
    # Category
    log_time(f"\n  Per-category attn/mlp effects:")
    for cat in sorted(agg["per_category"].keys()):
        ae = np.mean(agg["per_category"][cat]["attn_out"])
        me = np.mean(agg["per_category"][cat]["mlp_out"])
        winner = "ATTN" if ae < me else "MLP"
        log_time(f"    {cat:>15}: attn={ae:.3f}, mlp={me:.3f} → {winner} more critical")
    
    return {"pairs": results, "aggregate": agg}


# =============================================================================
# Block C: Component Contribution Matrix
# =============================================================================

def block_c_contribution_matrix(block_b12_agg, block_b34_agg, model_info, model_name):
    """
    Build per-layer component contribution matrix:
    
    Layer | weight_eff | value_eff | attn_out_eff | mlp_out_eff | primary_component
    
    Interpretation:
    - weight_eff: how much swapping attention weights changes output (ROUTING)
    - value_eff: how much swapping value vectors changes output (CONTENT)
    - attn_out_eff: how much patching attn_out with B's brings A toward B
    - mlp_out_eff: how much patching mlp_out with B's brings A toward B
    
    Lower attn_out_eff/mlp_out_eff = more critical for role encoding.
    Higher weight_eff/value_eff = that component carries more of the difference.
    """
    n_layers = model_info.n_layers
    
    # Interpolate to all layers from sampled layers
    matrix = {}
    
    # Get B12 sampled layers
    b12_layers = sorted([int(k) for k in block_b12_agg["per_layer"].keys()]) if block_b12_agg else []
    b34_layers = sorted([int(k) for k in block_b34_agg["per_layer"].keys()]) if block_b34_agg else []
    
    # Merge: use B34 layers as base (more layers), add B12 where available
    all_layers = sorted(set(b12_layers + b34_layers))
    
    for li in all_layers:
        entry = {"layer": li}
        
        # B12: manual RoPE
        key = str(li)
        if key in block_b12_agg["per_layer"]:
            a = block_b12_agg["per_layer"][key]
            entry["weight_effect"] = a["weight_effect_mean"]
            entry["value_effect"] = a["value_effect_mean"]
            entry["content_dominance"] = a["content_dominance_rate"]
        
        # B34: output patching (invert: 1-effect to get "importance")
        if key in block_b34_agg["per_layer"]:
            a = block_b34_agg["per_layer"][key]
            entry["attn_out_effect"] = a["attn_out_effect_mean"]
            entry["mlp_out_effect"] = a["mlp_out_effect_mean"]
            # Importance = 1 - effect (closer to 1 = more important)
            entry["attn_out_importance"] = 1.0 - min(a["attn_out_effect_mean"], 1.0)
            entry["mlp_out_importance"] = 1.0 - min(a["mlp_out_effect_mean"], 1.0)
        
        # Determine primary component
        if "value_effect" in entry and "weight_effect" in entry:
            if entry["value_effect"] > entry["weight_effect"]:
                entry["primary_manual"] = "VALUE"
            else:
                entry["primary_manual"] = "WEIGHT"
        
        if "attn_out_effect" in entry and "mlp_out_effect" in entry:
            if entry["attn_out_effect"] < entry["mlp_out_effect"]:
                entry["primary_output"] = "ATTN_OUT"
            else:
                entry["primary_output"] = "MLP_OUT"
        
        matrix[str(li)] = entry
    
    # Log the matrix
    log_time(f"\n{'='*50}")
    log_time(f"Block C: Component Contribution Matrix")
    log_time(f"  Interpretation:")
    log_time(f"    weight_effect: routing swap impact (higher = routing matters)")
    log_time(f"    value_effect:  content swap impact (higher = content matters)")
    log_time(f"    attn_out_importance: 1-patcher (higher = attn_out critical)")
    log_time(f"    mlp_out_importance:  1-patcher (higher = mlp_out critical)")
    log_time(f"\n  {'L':>5} {'wt_eff':>8} {'val_eff':>8} {'prim_man':>10} {'attn_imp':>10} {'mlp_imp':>10} {'prim_out':>10}")
    
    for li in sorted(matrix.keys(), key=int):
        e = matrix[li]
        pm = e.get("primary_manual", "N/A")
        po = e.get("primary_output", "N/A")
        log_time(f"  L{li:>4} {e.get('weight_effect','N/A'):>8} {e.get('value_effect','N/A'):>8} "
                 f"{pm:>10} {e.get('attn_out_importance','N/A'):>10} {e.get('mlp_out_importance','N/A'):>10} {po:>10}")
    
    return matrix


# =============================================================================
# Main
# =============================================================================

def run_phase282(model_name):
    global _log_file
    _log_file = str(TMP_DIR / f"phase282_{model_name}.txt")
    
    log_time(f"{'='*60}")
    log_time(f"Phase 282: Component Causal Patching with RoPE — {model_name}")
    log_time(f"{'='*60}")
    
    # Build 50+ pairs
    all_pairs = build_svo_pairs()
    log_time(f"Dataset: {len(all_pairs)} pairs across {len(set(p['category'] for p in all_pairs))} categories")
    
    # Category breakdown
    cat_counts = defaultdict(int)
    for p in all_pairs:
        cat_counts[p["category"]] += 1
    for cat, cnt in sorted(cat_counts.items()):
        log_time(f"  {cat}: {cnt} pairs")
    
    # Load model
    model, tokenizer, device = load_model_bf16_flash(model_name)
    model_info = get_model_info(model, model_name)
    log_time(f"Model: {model_info.model_class}, L={model_info.n_layers}, d={model_info.d_model}")
    
    # Do a warmup forward to materialize all layers
    log_time("Global warmup forward...")
    warmup_text = "The quick brown fox jumps over the lazy dog"
    warmup_inputs = tokenizer(warmup_text, return_tensors="pt").to(device)
    with torch.no_grad():
        try:
            model(**warmup_inputs)
        except:
            pass
    gpu_mem = torch.cuda.memory_allocated() / 1e9
    log_time(f"Warmup done, GPU={gpu_mem:.1f}GB")
    
    try:
        # Block B1+B2: Manual RoPE Patching
        t0 = time.time()
        results_b12 = block_b12_manual_rope_patching(model, tokenizer, device, model_info, model_name, all_pairs)
        t_b12 = time.time() - t0
        log_time(f"Block B1+B2 done in {t_b12:.1f}s ({t_b12/60:.1f}min)")
        
        # Save B12
        save_json = {}
        for pname, pr in results_b12["pairs"].items():
            save_json[pname] = pr
        save_json["aggregate"] = results_b12["aggregate"]
        # Convert defaultdict
        save_json["aggregate"]["per_category"] = dict(save_json["aggregate"]["per_category"])
        with open(RESULT_DIR / f"{model_name}_block_b12_rope.json", "w") as f:
            json.dump(save_json, f, indent=2)
        
        # Block B3+B4: Output Patching
        t0 = time.time()
        results_b34 = block_b34_output_patching(model, tokenizer, device, model_info, all_pairs)
        t_b34 = time.time() - t0
        log_time(f"Block B3+B4 done in {t_b34:.1f}s ({t_b34/60:.1f}min)")
        
        # Save B34
        save_json = {}
        for pname, pr in results_b34["pairs"].items():
            save_json[pname] = pr
        save_json["aggregate"] = results_b34["aggregate"]
        save_json["aggregate"]["per_category"] = dict(save_json["aggregate"]["per_category"])
        with open(RESULT_DIR / f"{model_name}_block_b34_output.json", "w") as f:
            json.dump(save_json, f, indent=2)
        
        # Block C: Contribution Matrix
        b12_agg = results_b12["aggregate"] if "aggregate" in results_b12 else {}
        b34_agg = results_b34["aggregate"] if "aggregate" in results_b34 else {}
        matrix = block_c_contribution_matrix(b12_agg, b34_agg, model_info, model_name)
        
        with open(RESULT_DIR / f"{model_name}_contribution_matrix.json", "w") as f:
            json.dump(matrix, f, indent=2)
        
        log_time(f"\nPhase 282 complete: B12={t_b12:.0f}s, B34={t_b34:.0f}s, total={t_b12+t_b34:.0f}s")
        
        return {
            "model": model_name,
            "time_B12": round(t_b12, 1),
            "time_B34": round(t_b34, 1),
            "pairs": len(all_pairs),
        }
        
    finally:
        release_model(model)
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    
    if model_name == "all":
        for name in ["qwen3", "glm4", "deepseek7b"]:
            try:
                r = run_phase282(name)
                log_time(f"{name} done: {r}")
            except Exception as e:
                log_time(f"!!! {name} FAILED: {e}")
                import traceback
                traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(3)
    else:
        run_phase282(model_name)
