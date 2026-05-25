"""
Phase 281: Routing-Content Separation (RCS)
=============================================
Phase 280 found: attention graph is stable under role-swap, but hidden states change.
This phase directly tests WHERE role/semantic information is carried:
  - Attention weights (routing layer)?
  - Value vectors (content layer)?
  - MLP output?
  - Residual mixing?

Core experiments:
  Block A: Component Delta Decomposition
    - Decompose hidden state differences into attn_out vs mlp_out contributions
    
  Block B: Value vs Attention Causal Swap (THE KEY EXPERIMENT)
    - Fix attention weights, swap value vectors → measure output shift
    - Fix value vectors, swap attention weights → measure output shift
    - Uses manual Q,K,V computation from weights + hidden states (no extra forward passes)

  Block C: Head-Level Specialization Map
    - Per-head: position sensitivity, content sensitivity, value direction variance

Usage:
  python tests/glm5/phase281_routing_content_separation.py qwen3
  python tests/glm5/phase281_routing_content_separation.py glm4
  python tests/glm5/phase281_routing_content_separation.py deepseek7b
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

RESULT_DIR = Path("results/phase281_routing_content")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

_log_file = None

def log_time(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if _log_file:
        with open(_log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")


# ===== Stimulus: Role-Swap Pairs (minimal contrast, same syntax) =====
ROLE_SWAP_PAIRS = [
    # SVO bidirectional swaps
    {
        "name": "dog_cat_chase",
        "A": "the dog chases the cat",
        "B": "the cat chases the dog",
    },
    {
        "name": "man_woman_love",
        "A": "the man loves the woman",
        "B": "the woman loves the man",
    },
    {
        "name": "king_city_rule",
        "A": "the king rules the city",
        "B": "the city rules the king",
    },
    {
        "name": "child_apple_eat",
        "A": "the child eats the apple",
        "B": "the apple eats the child",
    },
    # More pairs for robust statistics
    {
        "name": "boy_girl_call",
        "A": "the boy calls the girl",
        "B": "the girl calls the boy",
    },
    {
        "name": "teacher_student_teach",
        "A": "the teacher teaches the student",
        "B": "the student teaches the teacher",
    },
    {
        "name": "wolf_sheep_hunt",
        "A": "the wolf hunts the sheep",
        "B": "the sheep hunts the wolf",
    },
    {
        "name": "mother_child_hold",
        "A": "the mother holds the child",
        "B": "the child holds the mother",
    },
]

# Also test operator pairs (same carrier, different operator)
OPERATOR_SWAP_PAIRS = [
    {"name": "not_happy", "A": "he is happy", "B": "he is not happy"},
    {"name": "no_reason", "A": "there is a reason", "B": "there is no reason"},
    {"name": "must_go", "A": "you go now", "B": "you must go now"},
    {"name": "if_rain", "A": "it will rain", "B": "if it will rain"},
]


# ===== Model Loading (bf16 + flash_attention_2) =====
def load_model_bf16_flash(model_name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    log_time(f"Loading {model_name} (bf16 + flash_attention_2)...")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Try flash_attention_2, fall back to eager
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


# ===== Utility: Capture hidden states at each layer =====
def run_with_layer_hooks(model, tokenizer, device, sentence, layers_to_hook):
    """
    Run model forward, capture hidden states at input to each specified layer.
    Returns: {layer_idx: hidden_state_tensor_cpu [1, seq, d_model]}
    """
    inputs = tokenizer(sentence, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    seq_len = input_ids.shape[1]

    captured = {}

    def make_hook(key):
        def hook(module, input_t, output_t):
            # input_t is (hidden_states,) tuple — capture the input hidden state
            if isinstance(input_t, tuple) and len(input_t) > 0:
                captured[key] = input_t[0].detach().float().cpu()
            elif isinstance(input_t, torch.Tensor):
                captured[key] = input_t.detach().float().cpu()
        return hook

    layers = get_layers(model)
    hooks = []
    for li in layers_to_hook:
        if li < len(layers):
            hooks.append(layers[li].register_forward_hook(make_hook(f"L{li}")))

    # Also capture MLP and attn outputs AND attn inputs (post-LN)
    attn_captured = {}
    attn_input_captured = {}  # <-- NEW: post-LN hidden states
    mlp_captured = {}

    def make_attn_hook(key):
        def hook(module, input_t, output_t):
            # input_t[0] = post-LN hidden state (the REAL attention input)
            if isinstance(input_t, tuple) and len(input_t) > 0:
                attn_input_captured[key] = input_t[0].detach().float().cpu()
            if isinstance(output_t, tuple):
                attn_captured[key] = output_t[0].detach().float().cpu()
            else:
                attn_captured[key] = output_t.detach().float().cpu()
        return hook

    def make_mlp_hook(key):
        def hook(module, input_t, output_t):
            if isinstance(output_t, tuple):
                mlp_captured[key] = output_t[0].detach().float().cpu()
            else:
                mlp_captured[key] = output_t.detach().float().cpu()
        return hook

    attn_hooks = []
    mlp_hooks = []
    for li in layers_to_hook:
        if li < len(layers):
            layer = layers[li]
            if hasattr(layer, 'self_attn'):
                attn_hooks.append(layer.self_attn.register_forward_hook(make_attn_hook(f"L{li}")))
            if hasattr(layer, 'mlp'):
                mlp_hooks.append(layer.mlp.register_forward_hook(make_mlp_hook(f"L{li}")))

    with torch.no_grad():
        try:
            _ = model(**inputs)
        except Exception as e:
            log_time(f"  Forward failed: {e}")

    for h in hooks + attn_hooks + mlp_hooks:
        h.remove()

    return {
        "hidden": captured,
        "attn_in": attn_input_captured,  # post-LN hidden (correct for Q,K,V)
        "attn_out": attn_captured,
        "mlp_out": mlp_captured,
        "seq_len": seq_len,
        "input_ids": input_ids.cpu(),
    }


# ===== Block A: Component Delta Decomposition =====
def block_a_component_delta(model, tokenizer, device, model_info):
    """
    Decompose hidden state differences into attention vs MLP contributions.
    
    For each pair at each layer:
      Δ_hidden = hidden_next_B - hidden_next_A
      Δ_attn_out = attn_out_B - attn_out_A
      Δ_mlp_out = mlp_out_B - mlp_out_A
    
    Compute: cos(Δ_hidden, Δ_attn_out), cos(Δ_hidden, Δ_mlp_out)
             norm ratio: ||Δ_attn_out|| / ||Δ_hidden||
    
    This reveals which component drives the semantic/role difference.
    """
    n_layers = model_info.n_layers
    sample_layers = list(range(0, n_layers, max(1, n_layers // 8)))
    sample_layers = sorted(set(sample_layers + [n_layers - 1]))
    d_model = model_info.d_model
    
    log_time(f"\n{'='*50}")
    log_time(f"Block A: Component Delta Decomposition")
    log_time(f"  Layers: {sample_layers}, Pairs: {len(ROLE_SWAP_PAIRS)}")
    
    results = {}
    
    for pair in ROLE_SWAP_PAIRS:
        pname = pair["name"]
        log_time(f"  Processing: {pname} ...")
        
        # Run both sentences
        sent_a = pair["A"]
        sent_b = pair["B"]
        
        rA = run_with_layer_hooks(model, tokenizer, device, sent_a, sample_layers)
        rB = run_with_layer_hooks(model, tokenizer, device, sent_b, sample_layers)
        
        # Also run with output_hidden_states to get the initial embedding
        inputs_A = tokenizer(sent_a, return_tensors="pt").to(device)
        inputs_B = tokenizer(sent_b, return_tensors="pt").to(device)
        
        with torch.no_grad():
            out_A = model(**inputs_A, output_hidden_states=True)
            out_B = model(**inputs_B, output_hidden_states=True)
        
        embed_A = out_A.hidden_states[0].float().cpu()  # [1, seq_A, d_model]
        embed_B = out_B.hidden_states[0].float().cpu()
        
        pair_results = {"name": pname, "sent_A": sent_a, "sent_B": sent_b,
                        "seq_len_A": rA["seq_len"], "seq_len_B": rB["seq_len"],
                        "layers": {}}
        
        for li in sample_layers:
            key = f"L{li}"
            if key not in rA["hidden"] or key not in rB["hidden"]:
                continue
            if key not in rA["attn_out"] or key not in rB["attn_out"]:
                continue
            if key not in rA["mlp_out"] or key not in rB["mlp_out"]:
                continue
            
            # Get last token hidden states (position -1)
            hA = rA["hidden"][key][0, -1, :].numpy()  # [d_model]
            hB = rB["hidden"][key][0, -1, :].numpy()
            
            attnA = rA["attn_out"][key][0, -1, :].numpy()
            attnB = rB["attn_out"][key][0, -1, :].numpy()
            
            mlpA = rA["mlp_out"][key][0, -1, :].numpy()
            mlpB = rB["mlp_out"][key][0, -1, :].numpy()
            
            # Deltas
            dh = hB - hA
            d_attn = attnB - attnA
            d_mlp = mlpB - mlpA
            
            dh_norm = np.linalg.norm(dh)
            d_attn_norm = np.linalg.norm(d_attn)
            d_mlp_norm = np.linalg.norm(d_mlp)
            
            if dh_norm < 1e-10:
                continue
            
            cos_attn = float(np.dot(dh, d_attn) / (dh_norm * max(d_attn_norm, 1e-10)))
            cos_mlp = float(np.dot(dh, d_mlp) / (dh_norm * max(d_mlp_norm, 1e-10)))
            
            # Norm ratios: how much of total change comes from attn vs mlp
            attn_ratio = d_attn_norm / dh_norm
            mlp_ratio = d_mlp_norm / dh_norm
            
            # Cosine between attn delta and mlp delta (are they aligned?)
            cos_attn_mlp = float(np.dot(d_attn, d_mlp) / max(d_attn_norm * d_mlp_norm, 1e-10))
            
            pair_results["layers"][str(li)] = {
                "dh_norm": float(dh_norm),
                "d_attn_norm": float(d_attn_norm),
                "d_mlp_norm": float(d_mlp_norm),
                "cos_dh_attn": cos_attn,
                "cos_dh_mlp": cos_mlp,
                "attn_ratio": float(attn_ratio),
                "mlp_ratio": float(mlp_ratio),
                "cos_attn_mlp": cos_attn_mlp,
            }
        
        results[pname] = pair_results
        log_time(f"    {pname}: {len(pair_results['layers'])} layers analyzed")
    
    # Aggregate across pairs
    agg = {"per_layer": {}}
    for li in sample_layers:
        key = str(li)
        cos_attn_vals = []
        cos_mlp_vals = []
        attn_ratio_vals = []
        mlp_ratio_vals = []
        cos_am_vals = []
        
        for pname, pr in results.items():
            if key in pr.get("layers", {}):
                lr = pr["layers"][key]
                cos_attn_vals.append(lr["cos_dh_attn"])
                cos_mlp_vals.append(lr["cos_dh_mlp"])
                attn_ratio_vals.append(lr["attn_ratio"])
                mlp_ratio_vals.append(lr["mlp_ratio"])
                cos_am_vals.append(lr["cos_attn_mlp"])
        
        if cos_attn_vals:
            agg["per_layer"][key] = {
                "cos_dh_attn_mean": float(np.mean(cos_attn_vals)),
                "cos_dh_attn_std": float(np.std(cos_attn_vals)),
                "cos_dh_mlp_mean": float(np.mean(cos_mlp_vals)),
                "cos_dh_mlp_std": float(np.std(cos_mlp_vals)),
                "attn_ratio_mean": float(np.mean(attn_ratio_vals)),
                "mlp_ratio_mean": float(np.mean(mlp_ratio_vals)),
                "cos_attn_mlp_mean": float(np.mean(cos_am_vals)),
            }
    
    # Log summary
    log_time(f"\n  Summary (agg across {len(ROLE_SWAP_PAIRS)} pairs):")
    for li in sample_layers:
        key = str(li)
        if key in agg["per_layer"]:
            a = agg["per_layer"][key]
            log_time(f"  L{li}: cos(dh,attn)={a['cos_dh_attn_mean']:.3f}, "
                     f"cos(dh,mlp)={a['cos_dh_mlp_mean']:.3f}, "
                     f"attn_ratio={a['attn_ratio_mean']:.2f}, "
                     f"mlp_ratio={a['mlp_ratio_mean']:.2f}, "
                     f"cos(attn,mlp)={a['cos_attn_mlp_mean']:.3f}")
    
    return {"pairs": results, "aggregate": agg}


# ===== Block B: Value vs Attention Causal Swap =====
def block_b_value_attention_swap(model, tokenizer, device, model_info):
    """
    THE KEY EXPERIMENT: Separate attention weights (routing) from value vectors (content).
    
    For each layer with weights W_q, W_k, W_v, W_o:
      1. Get hidden states hA, hB at layer input
      2. Compute QA, KA, VA, QB, KB, VB manually
      3. Compute attention weights: AA = softmax(QA@KA^T), AB = softmax(QB@KB^T)
      4. Mixed outputs:
         - A_weights + V_B: the routing stays but content changes → measure shift
         - B_weights + V_A: different routing, same content → measure shift
      5. Compare: which mixed output is closer to B's actual output?
    
    If value swap (A_weights+B_values) produces output close to B's output,
    then VALUE carries role information, not attention routing.
    """
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    
    # Sample fewer layers for this expensive experiment
    sample_layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
    sample_layers = sorted(set(sample_layers))
    
    log_time(f"\n{'='*50}")
    log_time(f"Block B: Value vs Attention Causal Swap")
    log_time(f"  Layers: {sample_layers}, Pairs: {len(ROLE_SWAP_PAIRS)}")
    
    layers = get_layers(model)
    
    # Pre-extract weights for sampled layers
    # For device_map="auto" models, run a dummy forward pass first to materialize meta tensors
    log_time("  Running warm-up forward pass to materialize weights...")
    dummy_text = "hello world"
    dummy_inputs = tokenizer(dummy_text, return_tensors="pt").to(device)
    with torch.no_grad():
        try:
            _ = model(**dummy_inputs)
        except Exception:
            pass
    
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
            log_time(f"  L{li}: weight access failed ({e}), skipping")
            continue
        
        # Get number of heads: Q heads vs KV heads (GQA support)
        if hasattr(sa, 'num_heads'):
            n_heads_q = sa.num_heads
        elif hasattr(sa, 'num_attention_heads'):
            n_heads_q = sa.num_attention_heads
        else:
            n_heads_q = getattr(model.config, 'num_attention_heads', 32)
        
        # KV heads (may differ from Q heads in GQA)
        if hasattr(sa, 'num_key_value_heads'):
            n_heads_kv = sa.num_key_value_heads
        elif hasattr(model, 'config'):
            n_heads_kv = getattr(model.config, 'num_key_value_heads', n_heads_q)
        else:
            n_heads_kv = n_heads_q
        
        # head_dim from Q weight shape
        d_head = W_q.shape[0] // n_heads_q
        
        # GQA group size
        gqa_group = n_heads_q // n_heads_kv if n_heads_kv > 0 else 1
        
        layer_weights[li] = {
            "W_q": W_q, "W_k": W_k, "W_v": W_v, "W_o": W_o,
            "n_heads_q": n_heads_q, "n_heads_kv": n_heads_kv, "d_head": d_head,
            "gqa_group": gqa_group,
        }
        log_time(f"  L{li}: Q_heads={n_heads_q}, KV_heads={n_heads_kv}, "
                 f"d_head={d_head}, gqa_group={gqa_group}, "
                 f"Q_dim={W_q.shape[0]}, K_dim={W_k.shape[0]}")
    
    results = {}
    
    for pair in ROLE_SWAP_PAIRS:
        pname = pair["name"]
        log_time(f"  Processing: {pname} ...")
        
        sent_a = pair["A"]
        sent_b = pair["B"]
        
        # Run both with hooks
        rA = run_with_layer_hooks(model, tokenizer, device, sent_a, sample_layers)
        rB = run_with_layer_hooks(model, tokenizer, device, sent_b, sample_layers)
        
        pair_results = {"name": pname, "sent_A": sent_a, "sent_B": sent_b, "layers": {}}
        
        for li in sample_layers:
            key = f"L{li}"
            if li not in layer_weights:
                continue
            if key not in rA["hidden"] or key not in rB["hidden"]:
                continue
            if key not in rA["attn_out"] or key not in rB["attn_out"]:
                continue
            
            # Use pre-LN hidden state, then apply LN manually
            # The self_attn module receives post-LN input
            hA_pre = rA["hidden"][key][0].numpy()
            hB_pre = rB["hidden"][key][0].numpy()
            
            # Apply LayerNorm/RMSNorm to get correct attention input
            layer = layers[li]
            hA_ln = apply_input_ln(hA_pre, layer)
            hB_ln = apply_input_ln(hB_pre, layer)
            seqA = hA_ln.shape[0]
            seqB = hB_ln.shape[0]
            
            # Actual attention outputs
            attnA_actual = rA["attn_out"][key][0].numpy()  # [seq_A, d_model]
            attnB_actual = rB["attn_out"][key][0].numpy()  # [seq_B, d_model]
            
            w = layer_weights[li]
            W_q = w["W_q"]  # [n_heads_q * d_head, d_model]
            W_k = w["W_k"]  # [n_heads_kv * d_head, d_model]
            W_v = w["W_v"]
            W_o = w["W_o"]
            n_heads_q = w["n_heads_q"]
            n_heads_kv = w["n_heads_kv"]
            d_head = w["d_head"]
            gqa_group = w["gqa_group"]
            
            # Compute Q, K, V from post-LN hidden states (correct attention input)
            QA = hA_ln @ W_q.T
            KA = hA_ln @ W_k.T
            VA = hA_ln @ W_v.T
            
            QB = hB_ln @ W_q.T
            KB = hB_ln @ W_k.T
            VB = hB_ln @ W_v.T
            
            # Reshape to multi-head: Q → [n_heads_q, seq, d_head], K,V → [n_heads_kv, seq, d_head]
            def reshape_mha(X, n_heads, d_head):
                seq = X.shape[0]
                return X.reshape(seq, n_heads, d_head).transpose(1, 0, 2)
            
            QA_mha = reshape_mha(QA, n_heads_q, d_head)    # [n_heads_q, seqA, d_head]
            KA_mha = reshape_mha(KA, n_heads_kv, d_head)   # [n_heads_kv, seqA, d_head]
            VA_mha = reshape_mha(VA, n_heads_kv, d_head)
            
            QB_mha = reshape_mha(QB, n_heads_q, d_head)
            KB_mha = reshape_mha(KB, n_heads_kv, d_head)
            VB_mha = reshape_mha(VB, n_heads_kv, d_head)
            
            # GQA: expand K/V to match Q heads by repeating
            if gqa_group > 1:
                KA_mha = np.repeat(KA_mha, gqa_group, axis=0)  # [n_heads_q, seqA, d_head]
                VA_mha = np.repeat(VA_mha, gqa_group, axis=0)
                KB_mha = np.repeat(KB_mha, gqa_group, axis=0)
                VB_mha = np.repeat(VB_mha, gqa_group, axis=0)
            
            # Attention weights
            scale = np.sqrt(d_head)
            AA = np.zeros((n_heads_q, seqA, seqA))
            for h in range(n_heads_q):
                scores = QA_mha[h] @ KA_mha[h].T / scale
                AA[h] = softmax_np(scores)
            
            AB = np.zeros((n_heads_q, seqB, seqB))
            for h in range(n_heads_q):
                scores = QB_mha[h] @ KB_mha[h].T / scale
                AB[h] = softmax_np(scores)
            
            # Compute mixed outputs:
            # 1. A_weights + V_B (same routing, different content)
            # 2. B_weights + V_A (different routing, same content)
            
            # For fair comparison, we need same sequence lengths
            # Pad/truncate to match
            min_seq = min(seqA, seqB)
            if seqA != seqB:
                log_time(f"    {pname} L{li}: seq mismatch ({seqA} vs {seqB}), using min={min_seq}")
                AA_trunc = AA[:, :min_seq, :min_seq]
                AB_trunc = AB[:, :min_seq, :min_seq]
                VA_trunc = VA_mha[:, :min_seq, :]
                VB_trunc = VB_mha[:, :min_seq, :]
            else:
                AA_trunc = AA
                AB_trunc = AB
                VA_trunc = VA_mha
                VB_trunc = VB_mha
                min_seq = seqA
            
            # Mixed outputs
            # A_weights + V_B
            mixed_VB = np.zeros((n_heads_q, min_seq, d_head))
            for h in range(n_heads_q):
                mixed_VB[h] = AA_trunc[h] @ VB_trunc[h]
            
            # B_weights + V_A
            mixed_VA = np.zeros((n_heads_q, min_seq, d_head))
            for h in range(n_heads_q):
                mixed_VA[h] = AB_trunc[h] @ VA_trunc[h]
            
            # Reshape back and project through W_o
            def unreshape_mha(X, n_heads, seq_len, d_head):
                return X.transpose(1, 0, 2).reshape(seq_len, n_heads * d_head)
            
            mixed_VB_flat = unreshape_mha(mixed_VB, n_heads_q, min_seq, d_head) @ W_o.T
            mixed_VA_flat = unreshape_mha(mixed_VA, n_heads_q, min_seq, d_head) @ W_o.T
            
            # Pure outputs
            pure_AA_VA = np.zeros((n_heads_q, min_seq, d_head))
            pure_AB_VB = np.zeros((n_heads_q, min_seq, d_head))
            for h in range(n_heads_q):
                pure_AA_VA[h] = AA_trunc[h] @ VA_trunc[h]
                pure_AB_VB[h] = AB_trunc[h] @ VB_trunc[h]
            
            pure_AA_VA_flat = unreshape_mha(pure_AA_VA, n_heads_q, min_seq, d_head) @ W_o.T
            pure_AB_VB_flat = unreshape_mha(pure_AB_VB, n_heads_q, min_seq, d_head) @ W_o.T
            
            # Actual outputs (truncated)
            attnA_actual_trunc = attnA_actual[:min_seq]
            attnB_actual_trunc = attnB_actual[:min_seq]
            
            # Key metrics (clarified):
            #   routing_effect = ||(AA⊗VB) - (AB⊗VB)|| / total_gap
            #     → keeping content VB fixed, how much does changing routing (AA→AB) matter?
            #   content_effect = ||(AB⊗VA) - (AB⊗VB)|| / total_gap
            #     → keeping routing AB fixed, how much does changing content (VA→VB) matter?
            # Interpretation:
            #   If content_effect > routing_effect: content/values carry more information
            #   If content_effect < routing_effect: routing/attention carries more information
            
            total_gap = float(np.linalg.norm(pure_AA_VA_flat - pure_AB_VB_flat))
            
            # routing_effect: ||(AA⊗VB) - (AB⊗VB)|| (= dist_mix_VB_to_B)
            routing_effect_raw = float(np.linalg.norm(mixed_VB_flat - pure_AB_VB_flat))
            # content_effect: ||(AB⊗VA) - (AB⊗VB)|| (= dist_mix_VA_to_B)
            content_effect_raw = float(np.linalg.norm(mixed_VA_flat - pure_AB_VB_flat))
            
            if total_gap > 1e-10:
                routing_effect = routing_effect_raw / total_gap
                content_effect = content_effect_raw / total_gap
            else:
                routing_effect = 0.0
                content_effect = 0.0
            
            # Last-token specific
            lt = min_seq - 1
            lt_gap = float(np.linalg.norm(pure_AA_VA_flat[lt] - pure_AB_VB_flat[lt]))
            if lt_gap > 1e-10:
                lt_routing_effect = float(np.linalg.norm(mixed_VB_flat[lt] - pure_AB_VB_flat[lt])) / lt_gap
                lt_content_effect = float(np.linalg.norm(mixed_VA_flat[lt] - pure_AB_VB_flat[lt])) / lt_gap
            else:
                lt_routing_effect = 0.0
                lt_content_effect = 0.0
            
            # Who dominates? content > routing → values carry more
            content_dominates = content_effect > routing_effect
            
            pair_results["layers"][str(li)] = {
                "total_gap": float(total_gap),
                "routing_effect": float(routing_effect),
                "content_effect": float(content_effect),
                "routing_vs_content_ratio": float(routing_effect / max(content_effect, 1e-10)),
                "last_token": {
                    "lt_gap": float(lt_gap),
                    "routing_effect": float(lt_routing_effect),
                    "content_effect": float(lt_content_effect),
                },
                "content_dominates": content_dominates,
            }
        
        results[pname] = pair_results
        log_time(f"    {pname}: done")
    
    # Aggregate
    agg = {"per_layer": {}}
    for li in sample_layers:
        key = str(li)
        routing_effects = []
        content_effects = []
        lt_routing_effects = []
        lt_content_effects = []
        content_dominance = []
        
        for pname, pr in results.items():
            if key in pr.get("layers", {}):
                lr = pr["layers"][key]
                routing_effects.append(lr["routing_effect"])
                content_effects.append(lr["content_effect"])
                lt_routing_effects.append(lr["last_token"]["routing_effect"])
                lt_content_effects.append(lr["last_token"]["content_effect"])
                content_dominance.append(1.0 if lr["content_dominates"] else 0.0)
        
        if routing_effects:
            agg["per_layer"][key] = {
                "routing_effect_mean": float(np.mean(routing_effects)),
                "content_effect_mean": float(np.mean(content_effects)),
                "last_token_routing_effect": float(np.mean(lt_routing_effects)),
                "last_token_content_effect": float(np.mean(lt_content_effects)),
                "content_dominance_rate": float(np.mean(content_dominance)),
            }
    
    # Log summary
    log_time(f"\n  Summary (higher effect = component matters more, {len(ROLE_SWAP_PAIRS)} pairs):")
    log_time(f"  routing_effect: how much changing attention weights changes output")
    log_time(f"  content_effect: how much changing value vectors changes output")
    for li in sample_layers:
        key = str(li)
        if key in agg["per_layer"]:
            a = agg["per_layer"][key]
            dom = "CONTENT>>routing" if a["content_effect_mean"] > a["routing_effect_mean"] else "ROUTING>>content"
            log_time(f"  L{li}: routing_eff={a['last_token_routing_effect']:.3f} | "
                     f"content_eff={a['last_token_content_effect']:.3f} | "
                     f"lt_winner={dom} | "
                     f"content_dom%={a['content_dominance_rate']:.2f}")
    
    return {"pairs": results, "aggregate": agg}


# ===== Block C: Head-Level Specialization =====
def block_c_head_specialization(model, tokenizer, device, model_info):
    """
    Per-head analysis of attention pattern specialization.
    
    For each head at each sampled layer, compute:
    - Position sensitivity: correlation of attention pattern with distance matrix
    - Content sensitivity: how much does the pattern change across sentence pairs?
    - Value direction variance: across sentences, how stable are value vector directions?
    """
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    sample_layers = list(range(0, n_layers, max(1, n_layers // 8)))
    sample_layers = sorted(set(sample_layers + [n_layers - 1]))
    
    log_time(f"\n{'='*50}")
    log_time(f"Block C: Head-Level Specialization Map")
    log_time(f"  Layers: {sample_layers}")
    
    layers = get_layers(model)
    
    # Get all sentences from role swap pairs
    all_sentences = []
    for pair in ROLE_SWAP_PAIRS:
        all_sentences.append(pair["A"])
        all_sentences.append(pair["B"])
    
    log_time(f"  Testing {len(all_sentences)} sentences across {len(sample_layers)} layers")
    
    # Extract head info for each sampled layer
    head_info = {}
    
    for li in sample_layers:
        layer = layers[li]
        sa = layer.self_attn
        
        if hasattr(sa, 'num_heads'):
            n_heads_q = sa.num_heads
        elif hasattr(sa, 'num_attention_heads'):
            n_heads_q = sa.num_attention_heads
        else:
            n_heads_q = getattr(model.config, 'num_attention_heads', 32)
        
        if hasattr(sa, 'num_key_value_heads'):
            n_heads_kv = sa.num_key_value_heads
        elif hasattr(model, 'config'):
            n_heads_kv = getattr(model.config, 'num_key_value_heads', n_heads_q)
        else:
            n_heads_kv = n_heads_q
        try:
            W_q = sa.q_proj.weight.detach().cpu().float().numpy()
            W_k_base = sa.k_proj.weight.detach().cpu().float().numpy()
            W_v = sa.v_proj.weight.detach().cpu().float().numpy()
            W_o = sa.o_proj.weight.detach().cpu().float().numpy()
        except (NotImplementedError, RuntimeError) as e:
            log_time(f"  L{li}: weight access failed ({e}), skipping")
            continue
        
        # head_dim from Q weight shape
        d_head = W_q.shape[0] // n_heads_q
        gqa_group = n_heads_q // n_heads_kv if n_heads_kv > 0 else 1
        
        head_info[li] = {"n_heads_q": n_heads_q, "n_heads_kv": n_heads_kv,
                         "d_head": d_head, "gqa_group": gqa_group,
                         "W_q": W_q, "W_k": W_k_base, "W_v": W_v, "W_o": W_o}
        
        log_time(f"  L{li}: Q={n_heads_q} KV={n_heads_kv} d_head={d_head} gqa={gqa_group}")
    
    # For each sentence, capture hidden state and compute attention + value
    # Use flash forward pass for hidden states, then offline compute attention
    
    results = {"per_layer": {}}
    
    for li in sample_layers:
        if li not in head_info:
            continue
        hi = head_info[li]
        n_heads_q = hi["n_heads_q"]
        n_heads_kv = hi["n_heads_kv"]
        d_head = hi["d_head"]
        gqa_group = hi["gqa_group"]
        W_q = hi["W_q"]
        W_v = hi["W_v"]
        W_k_fixed = hi["W_k"]  # pre-extracted
        
        # Collect per-head attention patterns and value vectors across all sentences
        head_attn_patterns = defaultdict(list)
        head_value_dirs = defaultdict(list)
        
        log_time(f"  L{li}: computing head specialization...")
        
        for sent in all_sentences:
            r = run_with_layer_hooks(model, tokenizer, device, sent, [li])
            key = f"L{li}"
            if key not in r["hidden"]:
                continue
            
            # Apply LN to get post-LN hidden (correct attention input)
            h_pre = r["hidden"][key][0].numpy()
            h = apply_input_ln(h_pre, layers[li])
            seq = h.shape[0]
            
            # Compute Q, K, V
            Q = h @ W_q.T  # [seq, n_heads_q * d_head]
            K = h @ W_k_fixed.T  # [seq, n_heads_kv * d_head]
            V = h @ W_v.T
            
            # Reshape to multi-head
            def to_mha(X, n_h, d_h):
                s = X.shape[0]
                return X.reshape(s, n_h, d_h).transpose(1, 0, 2)
            
            Q_mha = to_mha(Q, n_heads_q, d_head)
            K_mha = to_mha(K, n_heads_kv, d_head)
            V_mha_raw = to_mha(V, n_heads_kv, d_head)
            
            # GQA: expand K/V to match Q
            if gqa_group > 1:
                K_mha = np.repeat(K_mha, gqa_group, axis=0)
                V_mha = np.repeat(V_mha_raw, gqa_group, axis=0)
            else:
                V_mha = V_mha_raw
            
            scale = np.sqrt(d_head)
            
            for h_idx in range(n_heads_q):
                scores = Q_mha[h_idx] @ K_mha[h_idx].T / scale
                attn_pat = softmax_np(scores)
                head_attn_patterns[h_idx].append(attn_pat)
                
                lt_value = V_mha[h_idx][-1, :]
                lt_value_norm = np.linalg.norm(lt_value)
                if lt_value_norm > 1e-10:
                    head_value_dirs[h_idx].append(lt_value / lt_value_norm)
        
        # Compute metrics per head
        head_metrics = {}
        
        # Distance matrix template (for position sensitivity)
        # For each sentence length, compute the ideal distance-based attention
        # Position sensitivity = correlation of attention with [1/distance] matrix
        
        for h_idx in range(n_heads_q):
            patterns = head_attn_patterns[h_idx]  # list of [seq, seq]
            values = head_value_dirs[h_idx]       # list of [d_head]
            
            if len(patterns) < 2:
                continue
            
            # 1. Position sensitivity: average correlation with distance matrix
            position_corrs = []
            for p in patterns:
                s = p.shape[0]
                # Distance matrix: d[i,j] = |i-j|
                dist = np.abs(np.arange(s)[:, None] - np.arange(s))
                # Expected: closer tokens have higher attention
                dist_norm = 1.0 / (1.0 + dist)
                # Correlation (flattened)
                p_flat = p.flatten()
                dn_flat = dist_norm.flatten()
                corr = np.corrcoef(p_flat, dn_flat)[0, 1]
                position_corrs.append(corr)
            
            pos_mean = float(np.mean(position_corrs))
            pos_std = float(np.std(position_corrs))
            
            # 2. Content sensitivity: variance of attention pattern across sentences
            # Stack patterns and compute pairwise Frobenius distances
            pattern_vars = []
            for i in range(len(patterns)):
                for j in range(i + 1, len(patterns)):
                    si, sj = patterns[i].shape[0], patterns[j].shape[0]
                    min_s = min(si, sj)
                    frob = float(np.linalg.norm(patterns[i][:min_s, :min_s] - patterns[j][:min_s, :min_s]))
                    pattern_vars.append(frob)
            
            content_sensitivity = float(np.mean(pattern_vars)) if pattern_vars else 0.0
            
            # 3. Value direction variance: std of pairwise cosine distances
            value_cos_dists = []
            for i in range(len(values)):
                for j in range(i + 1, len(values)):
                    cos = float(np.dot(values[i], values[j]))
                    value_cos_dists.append(1.0 - cos)  # cos distance
            value_dir_variance = float(np.mean(value_cos_dists)) if value_cos_dists else 0.0
            
            # 4. Head type classification
            if content_sensitivity < 0.05 and pos_mean > 0.3:
                head_type = "positional"
            elif content_sensitivity > 0.15:
                head_type = "content_sensitive"
            elif value_dir_variance > 0.1:
                head_type = "value_dynamic"
            else:
                head_type = "mixed"
            
            head_metrics[str(h_idx)] = {
                "position_sensitivity": pos_mean,
                "position_sensitivity_std": pos_std,
                "content_sensitivity": content_sensitivity,
                "value_dir_variance": value_dir_variance,
                "head_type": head_type,
            }
        
        results["per_layer"][str(li)] = head_metrics
        
        # Count types
        type_counts = defaultdict(int)
        for hm in head_metrics.values():
            type_counts[hm["head_type"]] += 1
        
        log_time(f"    L{li}: heads={n_heads_q}, types={dict(type_counts)}")
    
    # Aggregate
    agg = {"per_layer": {}}
    for li in sample_layers:
        key = str(li)
        if key not in results["per_layer"]:
            continue
        
        hm = results["per_layer"][key]
        
        pos_sens = [v["position_sensitivity"] for v in hm.values()]
        content_sens = [v["content_sensitivity"] for v in hm.values()]
        value_vars = [v["value_dir_variance"] for v in hm.values()]
        
        type_counts = defaultdict(int)
        for v in hm.values():
            type_counts[v["head_type"]] += 1
        
        agg["per_layer"][key] = {
            "pos_sensitivity_mean": float(np.mean(pos_sens)),
            "content_sensitivity_mean": float(np.mean(content_sens)),
            "value_variance_mean": float(np.mean(value_vars)),
            "type_distribution": dict(type_counts),
        }
        
        log_time(f"  L{li}: pos_sens={np.mean(pos_sens):.3f}, "
                 f"content_sens={np.mean(content_sens):.3f}, "
                 f"value_var={np.mean(value_vars):.3f}, "
                 f"types={dict(type_counts)}")
    
    return {"heads": results, "aggregate": agg}


# ===== Utility: Apply LayerNorm to hidden state =====
def apply_input_ln(hidden_np, layer, eps=1e-6):
    """
    Apply the layer's input_layernorm to hidden states.
    Supports RMSNorm (Qwen3) and standard LayerNorm.
    
    Args:
        hidden_np: [seq, d_model] numpy pre-LN hidden state
        layer: transformer layer object
    Returns:
        [seq, d_model] post-LN hidden state
    """
    # Try to get input_layernorm
    ln = None
    for ln_name in ["input_layernorm", "ln_1", "layernorm"]:
        if hasattr(layer, ln_name):
            ln = getattr(layer, ln_name)
            break
    
    if ln is None:
        return hidden_np  # No LN found, return as-is
    
    try:
        w = ln.weight.detach().cpu().float().numpy()  # [d_model]
    except (NotImplementedError, RuntimeError):
        return hidden_np  # Meta tensor, return as-is
    
    # Check if RMSNorm (no bias, no mean subtraction)
    has_bias = hasattr(ln, 'bias') and ln.bias is not None
    
    if has_bias:
        # Standard LayerNorm
        b = ln.bias.detach().cpu().float().numpy()
        mean = np.mean(hidden_np, axis=-1, keepdims=True)
        var = np.var(hidden_np, axis=-1, keepdims=True)
        out = (hidden_np - mean) / np.sqrt(var + eps)
        return out * w + b
    else:
        # RMSNorm (Qwen, GLM use this)
        rms = np.sqrt(np.mean(hidden_np ** 2, axis=-1, keepdims=True) + eps)
        return hidden_np * w / rms


# ===== Utility: Softmax =====
def softmax_np(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


# ===== Main =====
def run_phase281(model_name):
    global _log_file
    _log_file = str(RESULT_DIR / f"{model_name}_log.txt")
    
    log_time(f"{'='*60}")
    log_time(f"Phase 281: Routing-Content Separation — {model_name}")
    log_time(f"{'='*60}")
    
    # Load model
    model, tokenizer, device = load_model_bf16_flash(model_name)
    model_info = get_model_info(model, model_name)
    log_time(f"Model: {model_info.model_class}, L={model_info.n_layers}, "
             f"d={model_info.d_model}, mlp={model_info.mlp_type}")
    
    try:
        # Block A: Component Delta Decomposition
        t0 = time.time()
        results_a = block_a_component_delta(model, tokenizer, device, model_info)
        t_a = time.time() - t0
        log_time(f"Block A done in {t_a:.1f}s")
        
        # Save
        save_json = {}
        for pname, pr in results_a["pairs"].items():
            save_json[pname] = pr
        save_json["aggregate"] = results_a["aggregate"]
        with open(RESULT_DIR / f"{model_name}_block_a_delta.json", "w") as f:
            json.dump(save_json, f, indent=2)
        
        # Block B: Value vs Attention Swap
        t0 = time.time()
        results_b = block_b_value_attention_swap(model, tokenizer, device, model_info)
        t_b = time.time() - t0
        log_time(f"Block B done in {t_b:.1f}s")
        
        save_json = {}
        for pname, pr in results_b["pairs"].items():
            save_json[pname] = pr
        save_json["aggregate"] = results_b["aggregate"]
        with open(RESULT_DIR / f"{model_name}_block_b_swap.json", "w") as f:
            json.dump(save_json, f, indent=2)
        
        # Block C: Head Specialization
        t0 = time.time()
        results_c = block_c_head_specialization(model, tokenizer, device, model_info)
        t_c = time.time() - t0
        log_time(f"Block C done in {t_c:.1f}s")
        
        save_json = {}
        for li_key, hm in results_c["heads"]["per_layer"].items():
            save_json[li_key] = hm
        save_json["aggregate"] = results_c["aggregate"]
        with open(RESULT_DIR / f"{model_name}_block_c_heads.json", "w") as f:
            json.dump(save_json, f, indent=2)
        
        log_time(f"\nPhase 281 complete for {model_name}: A={t_a:.0f}s, B={t_b:.0f}s, C={t_c:.0f}s, total={t_a+t_b+t_c:.0f}s")
        
    finally:
        release_model(model)
        gc.collect()
        torch.cuda.empty_cache()
    
    return {
        "model": model_name,
        "time_A": round(t_a, 1),
        "time_B": round(t_b, 1),
        "time_C": round(t_c, 1),
    }


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    
    if model_name == "all":
        for name in ["qwen3", "glm4", "deepseek7b"]:
            try:
                r = run_phase281(name)
                log_time(f"{name} done: {r}")
            except Exception as e:
                log_time(f"!!! {name} FAILED: {e}")
                import traceback
                traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(3)
    else:
        run_phase281(model_name)
