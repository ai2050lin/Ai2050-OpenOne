"""
Phase 211: Circuit-Level Causal Analysis — FROM ROUTING TO WRITING
====================================================================

THEORETICAL CONTEXT:
  Phase 210 identified routing heads and showed subject-swap causal effect.
  But: attention weights ≠ information contribution (read ≠ write).
  The critical missing piece: what does each head WRITE to the verb position?

KEY CRITIQUE (from Phase 210 review):
  1. "attention weight ≠ 信息贡献" — softmax(QK^T) tells WHERE to look,
     but OV circuit tells WHAT is written. This is the critical gap.
  2. "可解码 ≠ 被使用" — probe artifact is still a risk.
  3. "约束场/能量景观 are metaphors, not theories" — avoid over-theorizing.
  4. Need: head-by-head patching, path patching, causal scrubbing, OV write analysis.

EXPERIMENTS:
  EXP1: OV Write Analysis ★★★ (THE CRITICAL EXPERIMENT)
    - For each (layer, head), compute OV_h = W_O[h] @ W_V[h]
    - Apply to subject hidden state, project through W_U
    - Answer: does this head push toward sg or pl verb?
    - This is what matters — NOT just where the head looks

  EXP2: Residual Stream Activation Patching ★★
    - Clean: sg subject → model prefers sg verb
    - Corrupted: pl subject → model prefers pl verb
    - At each layer, patch verb_pred_pos hidden state from clean → corrupted
    - Identify the "critical layer" for number information

  EXP3: Causal Scrubbing ★★
    - Replace subject with same-number different token → agreement should hold
    - Replace with different-number token → agreement should break
    - This controls for: does model use NUMBER or just token identity?

  EXP4: MLP Conditional Analysis ★
    - Compute MLP output at verb_pred_pos
    - Project through W_U to see if MLP writes sg/pl verb info
    - Is MLP the "if sg→V-3sg" conditional?

DATA: 75+ sentence pairs (with longer-distance variants)
MODELS: Qwen3, GLM4, DS7B (bf16 + device_map="auto")
"""

import sys, os
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent.parent / "tests"))

import gc, time, json, math, warnings, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from scipy.stats import ttest_rel, mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from model_utils import (get_model_info, release_model, get_layers, get_W_U,
                          MODEL_CONFIGS, get_layer_weights)

warnings.filterwarnings('ignore')
LITE = os.environ.get('LITE', '0') == '1'

# ========================================================================
# Sentence Data (EXPANDED with longer-distance variants)
# ========================================================================
IRREGULAR_PLURALS = {
    "child": "children", "man": "men", "woman": "women",
    "mouse": "mice", "goose": "geese", "foot": "feet",
    "tooth": "teeth", "person": "people", "fish": "fish",
    "deer": "deer", "sheep": "sheep",
}

# Simple: The [subj] [verb] the [obj]
SENTENCE_TRIPLES = [
    ("cat", "chases", "chase", "dog"),
    ("dog", "follows", "follow", "cat"),
    ("bear", "watches", "watch", "wolf"),
    ("wolf", "avoids", "avoid", "bear"),
    ("lion", "finds", "find", "tiger"),
    ("tiger", "fears", "fear", "lion"),
    ("horse", "meets", "meet", "deer"),
    ("fox", "tracks", "track", "rabbit"),
    ("rabbit", "hears", "hear", "fox"),
    ("whale", "joins", "join", "shark"),
    ("shark", "follows", "follow", "whale"),
    ("monkey", "greets", "greet", "elephant"),
    ("elephant", "trusts", "trust", "monkey"),
    ("bird", "watches", "watch", "eagle"),
    ("eagle", "finds", "find", "bird"),
    ("snake", "avoids", "avoid", "frog"),
    ("frog", "hears", "hear", "snake"),
    ("fish", "follows", "follow", "seal"),
    ("seal", "catches", "catch", "fish"),
    ("cow", "meets", "meet", "horse"),
    ("sheep", "follows", "follow", "goat"),
    ("duck", "watches", "watch", "goose"),
    ("mouse", "fears", "fear", "rat"),
    ("rat", "chases", "chase", "mouse"),
    ("hawk", "finds", "find", "owl"),
    ("owl", "watches", "watch", "hawk"),
    ("ant", "follows", "follow", "bee"),
    ("bee", "joins", "join", "ant"),
    ("crab", "meets", "meet", "fish"),
    ("teacher", "helps", "help", "student"),
    ("student", "thanks", "thank", "teacher"),
    ("doctor", "treats", "treat", "patient"),
    ("patient", "trusts", "trust", "doctor"),
    ("king", "protects", "protect", "queen"),
    ("queen", "guides", "guide", "king"),
    ("captain", "leads", "lead", "soldier"),
    ("soldier", "follows", "follow", "captain"),
    ("chef", "helps", "help", "baker"),
    ("baker", "thanks", "thank", "chef"),
    ("writer", "meets", "meet", "reader"),
    ("reader", "thanks", "thank", "writer"),
    ("singer", "joins", "join", "dancer"),
    ("dancer", "follows", "follow", "singer"),
    ("judge", "watches", "watch", "lawyer"),
    ("lawyer", "finds", "find", "judge"),
    ("driver", "meets", "meet", "rider"),
    ("rider", "thanks", "thank", "driver"),
    ("builder", "helps", "help", "painter"),
    ("painter", "guides", "guide", "builder"),
    ("farmer", "feeds", "feed", "worker"),
    ("worker", "helps", "help", "farmer"),
    ("priest", "guides", "guide", "monk"),
    ("monk", "follows", "follow", "priest"),
    ("nurse", "helps", "help", "clerk"),
    ("clerk", "thanks", "thank", "nurse"),
    ("master", "teaches", "teach", "pupil"),
    ("pupil", "thanks", "thank", "master"),
    ("hero", "protects", "protect", "child"),
    ("child", "trusts", "trust", "hero"),
    ("girl", "watches", "watch", "boy"),
    ("boy", "follows", "follow", "girl"),
    ("mother", "helps", "help", "father"),
    ("father", "guides", "guide", "mother"),
    ("sister", "meets", "meet", "brother"),
    ("brother", "finds", "find", "sister"),
    ("cat", "catches", "catch", "mouse"),
    ("dog", "bites", "bite", "cat"),
    ("rose", "attracts", "attract", "bee"),
    ("sun", "warms", "warm", "earth"),
    ("river", "feeds", "feed", "ocean"),
    ("wind", "moves", "move", "cloud"),
    ("fire", "burns", "burn", "wood"),
    ("rain", "helps", "help", "flower"),
    ("snow", "covers", "cover", "mountain"),
    ("star", "guides", "guide", "ship"),
]

# Longer distance: The [adj] [subj] [verb] the [obj]
ADJ_SENTENCE_TRIPLES = [
    ("big", "cat", "chases", "chase", "dog"),
    ("small", "dog", "follows", "follow", "cat"),
    ("old", "bear", "watches", "watch", "wolf"),
    ("young", "wolf", "avoids", "avoid", "bear"),
    ("fast", "lion", "finds", "find", "tiger"),
    ("slow", "tiger", "fears", "fear", "lion"),
    ("white", "horse", "meets", "meet", "deer"),
    ("red", "fox", "tracks", "track", "rabbit"),
    ("gray", "rabbit", "hears", "hear", "fox"),
    ("blue", "whale", "joins", "join", "shark"),
    ("wise", "teacher", "helps", "help", "student"),
    ("kind", "doctor", "treats", "treat", "patient"),
    ("brave", "king", "protects", "protect", "queen"),
    ("good", "captain", "leads", "lead", "soldier"),
    ("tall", "builder", "helps", "help", "painter"),
    ("warm", "sun", "warms", "warm", "earth"),
    ("cold", "wind", "moves", "move", "cloud"),
    ("dark", "night", "covers", "cover", "city"),
    ("bright", "star", "guides", "guide", "ship"),
    ("green", "river", "feeds", "feed", "ocean"),
]


def make_plural(noun):
    if noun in IRREGULAR_PLURALS:
        return IRREGULAR_PLURALS[noun]
    if noun.endswith(("s", "sh", "ch", "x", "z")):
        return noun + "es"
    if noun.endswith("y") and noun[-2] not in "aeiou":
        return noun[:-1] + "ies"
    return noun + "s"


def generate_test_sentences():
    """Generate simple and adjective-modified sentence pairs."""
    test_data = []

    # Simple: "The [subj] [verb] the [obj]"
    for subj, v3sg, vbase, obj in SENTENCE_TRIPLES:
        sg_sent = f"The {subj} {v3sg} the {obj}"
        pl_subj = make_plural(subj)
        pl_sent = f"The {pl_subj} {vbase} the {obj}"
        test_data.append({
            'sent_sg': sg_sent, 'sent_pl': pl_sent,
            'verb_sg': v3sg, 'verb_pl': vbase,
            'subj': subj, 'obj': obj,
            'distance': 'short',  # subj and verb are adjacent
        })

    # With adjective: "The [adj] [subj] [verb] the [obj]"
    for adj, subj, v3sg, vbase, obj in ADJ_SENTENCE_TRIPLES:
        sg_sent = f"The {adj} {subj} {v3sg} the {obj}"
        pl_subj = make_plural(subj)
        pl_sent = f"The {adj} {pl_subj} {vbase} the {obj}"
        test_data.append({
            'sent_sg': sg_sent, 'sent_pl': pl_sent,
            'verb_sg': v3sg, 'verb_pl': vbase,
            'subj': subj, 'obj': obj,
            'distance': 'medium',  # 1 token between subj and verb
        })

    print(f"  Generated {len(test_data)} sentence pairs "
          f"({sum(1 for d in test_data if d['distance']=='short')} short, "
          f"{sum(1 for d in test_data if d['distance']=='medium')} medium)")
    return test_data


# ========================================================================
# Model Loading
# ========================================================================
def load_model_bf16(model_name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    print(f"[load] Loading {model_name} (bfloat16 + device_map=auto)...")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = None
    for attn_impl in ["eager", "sdpa"]:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
                trust_remote_code=True, local_files_only=True,
                attn_implementation=attn_impl,
            )
            print(f"[load] Using attn_implementation={attn_impl}")
            break
        except Exception as e:
            print(f"[load] attn_implementation={attn_impl} failed: {e}")
            model = None
    if model is None:
        raise RuntimeError(f"Failed to load {model_name}")
    model.eval()
    device = next(model.parameters()).device
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    print(f"[load] {model_name}: device={device}, GPU={gpu_mem:.2f}GB")
    return model, tokenizer, device


# ========================================================================
# Tokenization & Position Finding
# ========================================================================
def tokenize_and_annotate(sentence, tokenizer):
    """Tokenize and find subject/verb positions in the sentence."""
    toks = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=64)
    input_ids = toks["input_ids"]
    ids_list = input_ids[0].tolist()
    decoded_tokens = [tokenizer.decode([tid]) for tid in ids_list]

    # Pattern 1: "The [adj] [subj] [verb] the [obj]" — medium distance
    for i in range(len(decoded_tokens) - 5):
        t0 = decoded_tokens[i].strip().lower()
        t4 = decoded_tokens[i+4].strip().lower()
        if t0 in ("the", "a", "an") and t4 in ("the", "a", "an"):
            return {
                'det1_pos': i, 'adj_pos': i+1, 'subj_pos': i+2, 'verb_pos': i+3,
                'det2_pos': i+4, 'obj_pos': i+5,
                'input_ids': input_ids, 'attention_mask': toks["attention_mask"],
                'decoded_tokens': decoded_tokens,
                'verb_pred_pos': i+2,  # at subj_pos, predict next token = verb
                'verb_token_id': ids_list[i + 3],
                'distance': 'medium',
            }

    # Pattern 2: "The [subj] [verb] the [obj]" — short distance
    for i in range(len(decoded_tokens) - 4):
        t0 = decoded_tokens[i].strip().lower()
        t3 = decoded_tokens[i+3].strip().lower()
        if t0 in ("the", "a", "an") and t3 in ("the", "a", "an"):
            return {
                'det1_pos': i, 'subj_pos': i + 1, 'verb_pos': i + 2,
                'det2_pos': i + 3, 'obj_pos': i + 4,
                'input_ids': input_ids, 'attention_mask': toks["attention_mask"],
                'decoded_tokens': decoded_tokens,
                'verb_pred_pos': i + 1,  # at subj_pos, predict next token = verb
                'verb_token_id': ids_list[i + 2],
                'distance': 'short',
            }

    return None


def find_verb_token_ids(tokenizer, verb_sg, verb_pl):
    """Find single-token IDs for verb forms."""
    candidates_sg = [
        tokenizer.encode(" " + verb_sg, add_special_tokens=False),
        tokenizer.encode(verb_sg, add_special_tokens=False),
    ]
    candidates_pl = [
        tokenizer.encode(" " + verb_pl, add_special_tokens=False),
        tokenizer.encode(verb_pl, add_special_tokens=False),
    ]
    sg_id = pl_id = None
    for ids in candidates_sg:
        if len(ids) == 1: sg_id = ids[0]; break
    for ids in candidates_pl:
        if len(ids) == 1: pl_id = ids[0]; break
    return sg_id, pl_id


def measure_agreement_from_logits(logits, verb_pred_pos, correct_id, wrong_id):
    """Measure agreement: sigmoid(logit_correct - logit_wrong)."""
    verb_logits = logits[0, verb_pred_pos]
    logit_correct = verb_logits[correct_id].float().item()
    logit_wrong = verb_logits[wrong_id].float().item()
    logit_diff = logit_correct - logit_wrong
    agreement = torch.sigmoid(torch.tensor(logit_diff)).item()
    return agreement, logit_diff


# ========================================================================
# Helper: Get head-level weight slices
# ========================================================================
def get_ov_circuits(layer, d_model, n_q_heads, head_dim, n_kv_heads, model=None, layer_idx=0):
    """
    Compute per-head OV circuits: OV_h = W_O[:, h*hd:(h+1)*hd] @ W_V[kv_h*hd:(kv_h+1)*hd, :]
    
    Handles GQA: multiple Q-heads may share the same KV-head.
    Handles meta tensors (device_map="auto" offloading).
    
    Returns:
        ov_circuits: np.ndarray [n_q_heads, d_model, d_model] or None if weights unavailable
    """
    sa = layer.self_attn
    
    def weight_to_numpy(param_name):
        """Get weight as numpy, handling meta tensors from device_map="auto"."""
        try:
            # Try direct access first (works for GPU/CPU tensors)
            param = getattr(sa, param_name).weight
            if not param.is_meta:
                return param.detach().cpu().float().numpy()
        except (NotImplementedError, RuntimeError):
            pass
        
        # Meta tensor: try to load from state_dict or safetensors
        # Build the full parameter name
        layers = get_layers(model) if model is not None else None
        if layers is not None and layer_idx < len(layers):
            # Find the actual parameter key in model's state dict
            for name, param in model.named_parameters():
                if f"layers.{layer_idx}.self_attn.{param_name}.weight" in name:
                    if not param.is_meta:
                        return param.detach().cpu().float().numpy()
        
        # Last resort: try to use accelerate to dispatch
        try:
            import accelerate
            param = getattr(sa, param_name).weight
            # Force materialization via HF dispatch
            if hasattr(param, '_hf_hook'):
                param = param._hf_hook.post_forward(param)
            return param.detach().cpu().float().numpy()
        except Exception:
            pass
        
        print(f"  [OV] Warning: Cannot access {param_name}.weight at L{layer_idx} (meta tensor)")
        return None

    W_V = weight_to_numpy('v_proj')
    W_O = weight_to_numpy('o_proj')
    
    if W_V is None or W_O is None:
        return None

    # GQA mapping: which KV-head does Q-head h use?
    kv_ratio = n_q_heads // n_kv_heads

    ov_circuits = []
    for h in range(n_q_heads):
        kv_h = h // kv_ratio
        v_slice = W_V[kv_h * head_dim:(kv_h + 1) * head_dim, :]
        o_slice = W_O[:, h * head_dim:(h + 1) * head_dim]
        ov_h = o_slice @ v_slice
        ov_circuits.append(ov_h)

    return np.array(ov_circuits)


def apply_layernorm(layer, hidden_state_tensor):
    """Apply input layernorm to a hidden state tensor."""
    ln = None
    for ln_name in ["input_layernorm", "ln_1", "layernorm"]:
        if hasattr(layer, ln_name):
            ln = getattr(layer, ln_name)
            break
    if ln is None:
        return hidden_state_tensor
    with torch.no_grad():
        return ln(hidden_state_tensor)


# ========================================================================
# EXP1: OV Write Analysis ★★★
# ========================================================================
def run_exp1_ov_write(model, tokenizer, device, test_sentences, n_layers, d_model, n_heads, head_dim, n_kv_heads):
    """
    THE CRITICAL EXPERIMENT: What does each head WRITE to the verb position?
    
    Method:
    1. Run clean forward, capture hidden_states and attention_weights
    2. For each (layer, head):
       a. Get subject hidden state at that layer (after layernorm)
       b. Compute OV contribution: attn_weight * OV_h @ h_subj
       c. Project through W_U: effect on sg/pl verb logits
       d. Classify head as "sg-pusher" or "pl-pusher"
    
    This answers: "what does each head WRITE" not just "where does it look"
    """
    print("\n" + "=" * 70)
    print("EXP1: OV Write Analysis ★★★ — What does each head WRITE?")
    print("=" * 70)

    layers = get_layers(model)
    W_U = get_W_U(model, None)
    if isinstance(W_U, torch.Tensor):
        W_U = W_U.float().numpy()

    # Sample layers
    if n_layers <= 12:
        sample_layers = list(range(n_layers))
    else:
        step = max(1, n_layers // 10)
        sample_layers = list(range(0, n_layers, step))
        if n_layers - 1 not in sample_layers:
            sample_layers.append(n_layers - 1)

    # Collect OV analysis data across sentences
    # For each (layer, head): collect [effect_sg, effect_pl, attn_to_subj, total_write_norm]
    ov_data = {l: {h: {'sg_effect': [], 'pl_effect': [], 'attn_subj': [], 'write_norm': []}
                    for h in range(n_heads)}
               for l in sample_layers}

    n_processed = 0
    n_valid = 0
    t_exp_start = time.time()

    for td in test_sentences:
        if n_processed >= 50:
            break

        sg_ann = tokenize_and_annotate(td['sent_sg'], tokenizer)
        if sg_ann is None:
            continue

        sg_verb_id, pl_verb_id = find_verb_token_ids(tokenizer, td['verb_sg'], td['verb_pl'])
        if sg_verb_id is None or pl_verb_id is None:
            continue

        # Verify model gets agreement right
        with torch.no_grad():
            out = model(input_ids=sg_ann['input_ids'].to(device),
                       attention_mask=sg_ann['attention_mask'].to(device),
                       output_hidden_states=True, output_attentions=True)

        baseline_agree, _ = measure_agreement_from_logits(
            out.logits, sg_ann['verb_pred_pos'], sg_verb_id, pl_verb_id)

        if baseline_agree < 0.5:
            n_processed += 1
            continue

        subj_pos = sg_ann['subj_pos']
        verb_pred_pos = sg_ann['verb_pred_pos']

        for l_idx in sample_layers:
            if l_idx >= len(out.hidden_states) or l_idx >= len(out.attentions):
                continue

            # Get hidden states at this layer
            hidden_l = out.hidden_states[l_idx][0].float().cpu()  # [seq, d_model]
            h_subj = hidden_l[subj_pos].numpy()  # [d_model] — subject position

            # Apply layernorm to get the actual input to attention
            h_subj_tensor = hidden_l[subj_pos:subj_pos+1].unsqueeze(0).to(device)
            h_subj_ln = apply_layernorm(layers[l_idx], h_subj_tensor)
            h_subj_ln_np = h_subj_ln[0, 0].float().cpu().numpy()  # [d_model]

            # Get attention weights
            attn = out.attentions[l_idx]  # [1, n_heads, seq, seq]
            n_heads_actual = attn.shape[1]

            # Get OV circuits for this layer (handles GQA + meta tensors)
            ov_circuits = get_ov_circuits(layers[l_idx], d_model, n_heads, head_dim, n_kv_heads,
                                          model=model, layer_idx=l_idx)
            if ov_circuits is None:
                continue  # Skip layers with meta tensor weights

            for h in range(min(n_heads_actual, n_heads)):
                # Attention weight from verb_pred_pos to subj_pos
                attn_to_subj = attn[0, h, verb_pred_pos, subj_pos].float().item()

                # OV write contribution: attn_weight * OV_h @ h_subj_ln
                ov_h = ov_circuits[h]  # [d_model, d_model]
                write_h = attn_to_subj * (ov_h @ h_subj_ln_np)  # [d_model]

                # Project through W_U to see effect on verb logits
                effect_sg = float(write_h @ W_U[sg_verb_id])
                effect_pl = float(write_h @ W_U[pl_verb_id])
                write_norm = float(np.linalg.norm(write_h))

                ov_data[l_idx][h]['sg_effect'].append(effect_sg)
                ov_data[l_idx][h]['pl_effect'].append(effect_pl)
                ov_data[l_idx][h]['attn_subj'].append(attn_to_subj)
                ov_data[l_idx][h]['write_norm'].append(write_norm)

        n_valid += 1
        n_processed += 1

        if n_processed % 10 == 0:
            elapsed = time.time() - t_exp_start
            print(f"  [EXP1] Processed {n_processed} sentences, {n_valid} valid, "
                  f"elapsed={elapsed:.0f}s")

    print(f"\n  Processed {n_valid} valid sentences")

    # Analyze OV write results
    print(f"\n  {'Layer':>6} {'Head':>5} {'Attn→Subj':>10} {'SG_Effect':>10} {'PL_Effect':>10} "
          f"{'SG-PL':>10} {'Write_Norm':>11} {'Role':>10}")
    print("  " + "-" * 75)

    head_roles = {}  # {layer: {head: role}}
    key_findings = []

    for l_idx in sorted(ov_data.keys()):
        head_roles[l_idx] = {}
        for h in sorted(ov_data[l_idx].keys()):
            d = ov_data[l_idx][h]
            if len(d['sg_effect']) < 5:
                continue

            mean_attn = np.mean(d['attn_subj'])
            mean_sg = np.mean(d['sg_effect'])
            mean_pl = np.mean(d['pl_effect'])
            mean_diff = mean_sg - mean_pl
            mean_norm = np.mean(d['write_norm'])

            # Classify head role
            if mean_attn > 0.1 and abs(mean_diff) > 0.01:
                if mean_diff > 0:
                    role = "SG-PUSHER"
                else:
                    role = "PL-PUSHER"
            elif mean_attn > 0.1:
                role = "ROUTER"
            elif abs(mean_diff) > 0.01:
                role = "WRITER"
            else:
                role = ""

            head_roles[l_idx][h] = {
                'attn_subj': float(mean_attn),
                'sg_effect': float(mean_sg),
                'pl_effect': float(mean_pl),
                'sg_pl_diff': float(mean_diff),
                'write_norm': float(mean_norm),
                'role': role,
            }

            # Print notable heads
            if mean_attn > 0.1 or abs(mean_diff) > 0.02:
                print(f"  L{l_idx:>4} H{h:>3} {mean_attn:>10.4f} {mean_sg:>10.4f} {mean_pl:>10.4f} "
                      f"{mean_diff:>10.4f} {mean_norm:>11.4f} {role:>10}")

                if role in ("SG-PUSHER", "PL-PUSHER"):
                    key_findings.append(f"L{l_idx}H{h}: {role} (diff={mean_diff:.4f}, attn={mean_attn:.4f})")

    # Summary
    print(f"\n  ★★★ Key OV Write Findings ★★★")
    sg_pushers = sum(1 for l in head_roles.values() for h in l.values() if h['role'] == 'SG-PUSHER')
    pl_pushers = sum(1 for l in head_roles.values() for h in l.values() if h['role'] == 'PL-PUSHER')
    routers = sum(1 for l in head_roles.values() for h in l.values() if h['role'] == 'ROUTER')
    writers = sum(1 for l in head_roles.values() for h in l.values() if h['role'] == 'WRITER')

    print(f"  SG-PUSHER heads: {sg_pushers}")
    print(f"  PL-PUSHER heads: {pl_pushers}")
    print(f"  ROUTER heads (high attn, no push): {routers}")
    print(f"  WRITER heads (low attn, writes): {writers}")

    if key_findings:
        print(f"\n  Notable heads:")
        for kf in key_findings[:15]:
            print(f"    {kf}")

    return {
        'head_roles': {str(k): {str(hh): vv for hh, vv in v.items()}
                       for k, v in head_roles.items()},
        'summary': {
            'sg_pushers': sg_pushers,
            'pl_pushers': pl_pushers,
            'routers': routers,
            'writers': writers,
            'key_findings': key_findings[:20],
        }
    }


# ========================================================================
# EXP2: Residual Stream Activation Patching ★★
# ========================================================================
def run_exp2_activation_patching(model, tokenizer, device, test_sentences, n_layers, d_model):
    """
    Causal intervention: at each layer, patch the verb_pred_pos hidden state
    from the clean run into the corrupted run.
    
    Clean: "The cat chases the dog" → model prefers sg verb
    Corrupted: "The cats chase the dog" → model prefers pl verb
    Patch: at layer L, inject clean hidden state at verb_pred_pos
    → if agreement recovers, layer L carries critical number info
    """
    print("\n" + "=" * 70)
    print("EXP2: Residual Stream Activation Patching ★★")
    print("=" * 70)

    layers = get_layers(model)

    if n_layers <= 12:
        sample_layers = list(range(n_layers))
    else:
        step = max(1, n_layers // 8)
        sample_layers = list(range(0, n_layers, step))
        if n_layers - 1 not in sample_layers:
            sample_layers.append(n_layers - 1)

    # Collect patching results: {layer: [agreement_recovery values]}
    patch_recovery = {l: [] for l in sample_layers}
    n_tested = 0
    t_exp_start = time.time()

    for td in test_sentences:
        if n_tested >= 30:
            break

        sg_ann = tokenize_and_annotate(td['sent_sg'], tokenizer)
        pl_ann = tokenize_and_annotate(td['sent_pl'], tokenizer)
        if sg_ann is None or pl_ann is None:
            continue

        sg_verb_id, pl_verb_id = find_verb_token_ids(tokenizer, td['verb_sg'], td['verb_pl'])
        if sg_verb_id is None or pl_verb_id is None:
            continue

        # Verify positions match (both sentences should have same structure)
        # The verb_pred_pos might differ due to tokenization
        # Use the sg sentence's positions for both
        verb_pred_pos = sg_ann['verb_pred_pos']

        # Run clean forward, capture hidden states
        clean_hidden = {}
        def make_capture_hook(layer_idx):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    clean_hidden[layer_idx] = output[0].detach().clone()
                else:
                    clean_hidden[layer_idx] = output.detach().clone()
            return hook

        hooks = [layers[i].register_forward_hook(make_capture_hook(i)) for i in sample_layers]
        with torch.no_grad():
            clean_out = model(input_ids=sg_ann['input_ids'].to(device),
                            attention_mask=sg_ann['attention_mask'].to(device))
        for h in hooks:
            h.remove()

        # Measure clean and corrupted agreement
        clean_agree, _ = measure_agreement_from_logits(
            clean_out.logits, sg_ann['verb_pred_pos'], sg_verb_id, pl_verb_id)

        if clean_agree < 0.6:
            continue

        with torch.no_grad():
            corrupt_out = model(input_ids=pl_ann['input_ids'].to(device),
                               attention_mask=pl_ann['attention_mask'].to(device))

        # For corrupted, we measure whether the model prefers pl verb (correct for pl subject)
        # So we measure agreement as: pl_verb logit > sg_verb logit
        corrupt_pl_agree, _ = measure_agreement_from_logits(
            corrupt_out.logits, pl_ann['verb_pred_pos'], pl_verb_id, sg_verb_id)

        if corrupt_pl_agree < 0.5:
            continue

        # Now do patching at each layer
        for l_idx in sample_layers:
            if l_idx not in clean_hidden:
                continue

            # Patch: replace verb_pred_pos hidden state with clean version
            target_pos = verb_pred_pos

            def make_patching_hook(clean_h, pos):
                def hook(module, input, output):
                    patched = output[0].clone()
                    # Only patch the verb_pred_pos
                    if pos < patched.shape[1] and pos < clean_h.shape[1]:
                        patched[0, pos, :] = clean_h[0, pos, :]
                    return (patched,) + output[1:] if isinstance(output, tuple) else patched
                return hook

            hook = layers[l_idx].register_forward_hook(
                make_patching_hook(clean_hidden[l_idx], target_pos))

            with torch.no_grad():
                patched_out = model(input_ids=pl_ann['input_ids'].to(device),
                                  attention_mask=pl_ann['attention_mask'].to(device))

            hook.remove()

            # After patching, measure sg verb agreement
            # (we patched clean sg info, so model should now prefer sg verb)
            patched_agree, _ = measure_agreement_from_logits(
                patched_out.logits, pl_ann['verb_pred_pos'], sg_verb_id, pl_verb_id)

            # Recovery = how much the sg agreement improves
            # Baseline corrupted sg agree (should be low, ~0.2)
            baseline_corrupt_sg_agree, _ = measure_agreement_from_logits(
                corrupt_out.logits, pl_ann['verb_pred_pos'], sg_verb_id, pl_verb_id)

            recovery = patched_agree - baseline_corrupt_sg_agree
            patch_recovery[l_idx].append(recovery)

            del patched_out
            torch.cuda.empty_cache()

        n_tested += 1

        if n_tested % 5 == 0:
            elapsed = time.time() - t_exp_start
            print(f"  [EXP2] Tested {n_tested} sentences, elapsed={elapsed:.0f}s")

    # Analyze patching results
    print(f"\n  Tested {n_tested} sentences")
    print(f"\n  {'Layer':>6} {'Mean_Recovery':>14} {'N':>4} {'Importance':>12}")
    print("  " + "-" * 45)

    patch_results = {}
    critical_layers = []

    for l_idx in sorted(patch_recovery.keys()):
        vals = patch_recovery[l_idx]
        if len(vals) < 3:
            continue
        mean_rec = np.mean(vals)
        patch_results[f"L{l_idx}"] = {
            'mean_recovery': float(mean_rec),
            'n_test': len(vals),
            'std_recovery': float(np.std(vals)),
        }

        label = "★★★ CRITICAL" if mean_rec > 0.3 else "★★ IMPORTANT" if mean_rec > 0.1 else "★ MODERATE" if mean_rec > 0.03 else "✗ MINIMAL"
        print(f"  L{l_idx:>4} {mean_rec:>14.4f} {len(vals):>4} {label:>12}")

        if mean_rec > 0.1:
            critical_layers.append((l_idx, mean_rec))

    if critical_layers:
        print(f"\n  ★ Critical layers (recovery > 0.1):")
        for l, rec in sorted(critical_layers, key=lambda x: -x[1]):
            print(f"    L{l}: recovery={rec:.4f}")
    else:
        print(f"\n  ✗ No critical layers found (all recovery < 0.1)")

    return patch_results


# ========================================================================
# EXP3: Causal Scrubbing ★★
# ========================================================================
def run_exp3_causal_scrubbing(model, tokenizer, device, test_sentences, n_layers, d_model):
    """
    Causal Scrubbing: Does the model use NUMBER or just TOKEN IDENTITY?
    
    Test A (Same-number swap): Replace "cat" with "dog" (both sg)
      → If agreement holds, model uses structural number info, not just "cat" token
    
    Test B (Cross-number swap): Replace "cat" with "cats"
      → Agreement should break (different number)
    
    Test C (Random word with same number): Replace "cat" with "book" (both sg)
      → If agreement holds, model uses grammatical number, not semantics
    """
    print("\n" + "=" * 70)
    print("EXP3: Causal Scrubbing ★★ — Number vs Token Identity")
    print("=" * 70)

    # Same-number different-token subjects
    SG_SWAP_SUBJECTS = ["dog", "bear", "horse", "teacher", "doctor", "king", "bird", "fish"]
    PL_SWAP_SUBJECTS = ["dogs", "bears", "horses", "teachers", "doctors", "kings", "birds", "fish"]

    results = {
        'same_number_swap': [],     # Test A: replace subj with same-number different token
        'cross_number_swap': [],    # Test B: replace subj with different-number token
        'agreement_retention': [],  # For same-number swaps, does agreement hold?
    }

    n_tested = 0
    for td in test_sentences:
        if n_tested >= 40:
            break

        sg_ann = tokenize_and_annotate(td['sent_sg'], tokenizer)
        if sg_ann is None:
            continue

        sg_verb_id, pl_verb_id = find_verb_token_ids(tokenizer, td['verb_sg'], td['verb_pl'])
        if sg_verb_id is None or pl_verb_id is None:
            continue

        # Baseline: original sentence agreement
        with torch.no_grad():
            base_out = model(input_ids=sg_ann['input_ids'].to(device),
                           attention_mask=sg_ann['attention_mask'].to(device))
        base_agree, _ = measure_agreement_from_logits(
            base_out.logits, sg_ann['verb_pred_pos'], sg_verb_id, pl_verb_id)

        if base_agree < 0.6:
            continue

        # Test A: Same-number swap (e.g., "The cat chases" → "The dog chases")
        for swap_subj in SG_SWAP_SUBJECTS:
            if swap_subj == td['subj']:
                continue
            swap_sent = f"The {swap_subj} {td['verb_sg']} the {td['obj']}"
            swap_ann = tokenize_and_annotate(swap_sent, tokenizer)
            if swap_ann is None:
                continue

            with torch.no_grad():
                swap_out = model(input_ids=swap_ann['input_ids'].to(device),
                               attention_mask=swap_ann['attention_mask'].to(device))
            swap_agree, _ = measure_agreement_from_logits(
                swap_out.logits, swap_ann['verb_pred_pos'], sg_verb_id, pl_verb_id)

            results['same_number_swap'].append({
                'original': td['subj'], 'swapped': swap_subj,
                'base_agree': base_agree, 'swap_agree': swap_agree,
                'agreement_held': swap_agree > 0.5,
            })
            del swap_out
            break  # One swap per sentence

        # Test B: Cross-number swap (e.g., "The cat chases" → "The cats chases")
        pl_subj = make_plural(td['subj'])
        cross_sent = f"The {pl_subj} {td['verb_sg']} the {td['obj']}"
        cross_ann = tokenize_and_annotate(cross_sent, tokenizer)
        if cross_ann is not None:
            with torch.no_grad():
                cross_out = model(input_ids=cross_ann['input_ids'].to(device),
                                 attention_mask=cross_ann['attention_mask'].to(device))
            cross_agree, _ = measure_agreement_from_logits(
                cross_out.logits, cross_ann['verb_pred_pos'], sg_verb_id, pl_verb_id)

            results['cross_number_swap'].append({
                'original': td['subj'], 'swapped': pl_subj,
                'base_agree': base_agree, 'cross_agree': cross_agree,
                'agreement_broken': cross_agree < 0.5,
            })
            del cross_out

        n_tested += 1

    # Analyze
    same_num = results['same_number_swap']
    cross_num = results['cross_number_swap']

    print(f"\n  Tested {n_tested} sentences")

    if same_num:
        n_held = sum(1 for r in same_num if r['agreement_held'])
        mean_swap_agree = np.mean([r['swap_agree'] for r in same_num])
        mean_base_agree = np.mean([r['base_agree'] for r in same_num])
        print(f"\n  Test A: Same-number swap (replace subj with same-number different token)")
        print(f"    Agreement held: {n_held}/{len(same_num)} ({100*n_held/len(same_num):.1f}%)")
        print(f"    Mean agreement: base={mean_base_agree:.3f}, swap={mean_swap_agree:.3f}")
        print(f"    Agreement drop: {mean_base_agree - mean_swap_agree:.3f}")

        if n_held > 0.8 * len(same_num):
            print(f"    ★★★ Model uses STRUCTURAL NUMBER, not token identity")
        elif n_held > 0.5 * len(same_num):
            print(f"    ★★ Model partially uses structural number")
        else:
            print(f"    ✗ Model may rely on token identity, not number")

    if cross_num:
        n_broken = sum(1 for r in cross_num if r['agreement_broken'])
        mean_cross_agree = np.mean([r['cross_agree'] for r in cross_num])
        mean_base2 = np.mean([r['base_agree'] for r in cross_num])
        print(f"\n  Test B: Cross-number swap (replace sg subj with pl)")
        print(f"    Agreement broken: {n_broken}/{len(cross_num)} ({100*n_broken/len(cross_num):.1f}%)")
        print(f"    Mean agreement: base={mean_base2:.3f}, cross={mean_cross_agree:.3f}")

        if n_broken > 0.7 * len(cross_num):
            print(f"    ★★★ Model is sensitive to NUMBER CHANGE")
        else:
            print(f"    ★ Model partially sensitive to number change")

    return {
        'same_number_retention_rate': float(n_held / len(same_num)) if same_num else 0,
        'cross_number_break_rate': float(n_broken / len(cross_num)) if cross_num else 0,
        'mean_same_number_agree': float(np.mean([r['swap_agree'] for r in same_num])) if same_num else 0,
        'mean_cross_number_agree': float(np.mean([r['cross_agree'] for r in cross_num])) if cross_num else 0,
    }


# ========================================================================
# EXP4: MLP Conditional Analysis ★
# ========================================================================
def run_exp4_mlp_analysis(model, tokenizer, device, test_sentences, n_layers, d_model, n_heads, head_dim):
    """
    Does the MLP write sg/pl verb information?
    
    Method:
    1. Run clean forward, capture pre-MLP and post-MLP hidden states
    2. Compute MLP output = post_attn_h - pre_mlp_h (approximately)
    3. Project through W_U to see if MLP pushes toward sg or pl verb
    """
    print("\n" + "=" * 70)
    print("EXP4: MLP Conditional Analysis ★ — Does MLP write verb form info?")
    print("=" * 70)

    layers = get_layers(model)
    W_U = get_W_U(model, None)
    if isinstance(W_U, torch.Tensor):
        W_U = W_U.float().numpy()

    if n_layers <= 12:
        sample_layers = list(range(n_layers))
    else:
        step = max(1, n_layers // 10)
        sample_layers = list(range(0, n_layers, step))
        if n_layers - 1 not in sample_layers:
            sample_layers.append(n_layers - 1)

    # Collect MLP output data
    # {layer: {'sg_mlp_effect': [...], 'pl_mlp_effect': [...]}}
    mlp_data = {l: {'sg_effect': [], 'pl_effect': [], 'attn_effect': []}
                for l in sample_layers}

    n_tested = 0
    for td in test_sentences:
        if n_tested >= 30:
            break

        sg_ann = tokenize_and_annotate(td['sent_sg'], tokenizer)
        pl_ann = tokenize_and_annotate(td['sent_pl'], tokenizer)
        if sg_ann is None or pl_ann is None:
            continue

        sg_verb_id, pl_verb_id = find_verb_token_ids(tokenizer, td['verb_sg'], td['verb_pl'])
        if sg_verb_id is None or pl_verb_id is None:
            continue

        # Run both with hidden states
        with torch.no_grad():
            sg_out = model(input_ids=sg_ann['input_ids'].to(device),
                          attention_mask=sg_ann['attention_mask'].to(device),
                          output_hidden_states=True)
            pl_out = model(input_ids=pl_ann['input_ids'].to(device),
                          attention_mask=pl_ann['attention_mask'].to(device),
                          output_hidden_states=True)

        # At verb_pred_pos, compare MLP effect for sg vs pl
        for l in sample_layers:
            if l >= len(sg_out.hidden_states) or l >= len(pl_out.hidden_states):
                continue

            # hidden_states[l] = output of layer l-1 (before layer l)
            # hidden_states[l+1] = output of layer l (after layer l)
            if l + 1 >= len(sg_out.hidden_states):
                continue

            # MLP output ≈ hidden[l+1] - hidden[l] - attn_output
            # Simplified: just use the full layer output difference
            # hidden[l+1] = hidden[l] + attn_out + mlp_out
            # So: layer_output = hidden[l+1] - hidden[l] = attn_out + mlp_out
            
            sg_verb_pos = sg_ann['verb_pred_pos']
            pl_verb_pos = pl_ann['verb_pred_pos']

            # Full layer output at verb_pred_pos
            sg_pre = sg_out.hidden_states[l][0, sg_verb_pos].float().cpu().numpy()
            sg_post = sg_out.hidden_states[l+1][0, sg_verb_pos].float().cpu().numpy()
            sg_layer_out = sg_post - sg_pre  # attn + mlp

            pl_pre = pl_out.hidden_states[l][0, pl_verb_pos].float().cpu().numpy()
            pl_post = pl_out.hidden_states[l+1][0, pl_verb_pos].float().cpu().numpy()
            pl_layer_out = pl_post - pl_pre  # attn + mlp

            # Project through W_U
            sg_effect_on_sg = float(sg_layer_out @ W_U[sg_verb_id])
            sg_effect_on_pl = float(sg_layer_out @ W_U[pl_verb_id])
            pl_effect_on_sg = float(pl_layer_out @ W_U[sg_verb_id])
            pl_effect_on_pl = float(pl_layer_out @ W_U[pl_verb_id])

            mlp_data[l]['sg_effect'].append(sg_effect_on_sg - sg_effect_on_pl)
            mlp_data[l]['pl_effect'].append(pl_effect_on_sg - pl_effect_on_pl)
            mlp_data[l]['attn_effect'].append(
                float((sg_layer_out - pl_layer_out) @ (W_U[sg_verb_id] - W_U[pl_verb_id])))

        n_tested += 1

    print(f"  Tested {n_tested} sentence pairs")

    print(f"\n  {'Layer':>6} {'SG→SG-PL':>10} {'PL→SG-PL':>10} {'Cross':>10} {'Role':>15}")
    print("  " + "-" * 55)

    mlp_results = {}
    for l in sorted(mlp_data.keys()):
        d = mlp_data[l]
        if len(d['sg_effect']) < 5:
            continue

        mean_sg = np.mean(d['sg_effect'])
        mean_pl = np.mean(d['pl_effect'])
        mean_cross = np.mean(d['attn_effect'])

        # Role classification
        if mean_sg > 0.1 and mean_pl < -0.1:
            role = "AGREEMENT-MLP"
        elif mean_cross > 0.1:
            role = "NUMBER-READER"
        elif abs(mean_sg) > 0.05 or abs(mean_pl) > 0.05:
            role = "CONDITIONAL"
        else:
            role = ""

        mlp_results[f"L{l}"] = {
            'sg_effect': float(mean_sg),
            'pl_effect': float(mean_pl),
            'cross_effect': float(mean_cross),
        }

        if abs(mean_sg) > 0.02 or abs(mean_pl) > 0.02:
            print(f"  L{l:>4} {mean_sg:>10.4f} {mean_pl:>10.4f} {mean_cross:>10.4f} {role:>15}")

    return mlp_results


# ========================================================================
# Main
# ========================================================================
def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    t_start = time.time()
    print(f"\n{'='*70}")
    print(f"Phase 211: Circuit-Level Causal Analysis — {model_name}")
    print(f"FROM ROUTING TO WRITING: What does each head WRITE?")
    print(f"{'='*70}")

    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    n_heads = getattr(model.config, 'num_attention_heads', d_model // 64)
    n_kv_heads = getattr(model.config, 'num_key_value_heads', n_heads)  # GQA support
    head_dim = d_model // n_heads
    print(f"  n_layers={n_layers}, d_model={d_model}, n_heads={n_heads}, "
          f"n_kv_heads={n_kv_heads}, head_dim={head_dim}")

    test_sentences = generate_test_sentences()
    all_results = {'model': model_name, 'n_layers': n_layers, 'd_model': d_model,
                   'n_heads': n_heads, 'head_dim': head_dim}

    # EXP1: OV Write Analysis ★★★
    try:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting EXP1: OV Write Analysis...")
        exp1_results = run_exp1_ov_write(
            model, tokenizer, device, test_sentences, n_layers, d_model, n_heads, head_dim, n_kv_heads)
        all_results['exp1_ov_write'] = exp1_results
    except Exception as e:
        print(f"  EXP1 FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results['exp1_ov_write'] = {"error": str(e)}
    torch.cuda.empty_cache()

    # EXP2: Activation Patching ★★
    try:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting EXP2: Activation Patching...")
        exp2_results = run_exp2_activation_patching(
            model, tokenizer, device, test_sentences, n_layers, d_model)
        all_results['exp2_activation_patching'] = exp2_results
    except Exception as e:
        print(f"  EXP2 FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results['exp2_activation_patching'] = {"error": str(e)}
    torch.cuda.empty_cache()

    # EXP3: Causal Scrubbing ★★
    try:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting EXP3: Causal Scrubbing...")
        exp3_results = run_exp3_causal_scrubbing(
            model, tokenizer, device, test_sentences, n_layers, d_model)
        all_results['exp3_causal_scrubbing'] = exp3_results
    except Exception as e:
        print(f"  EXP3 FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results['exp3_causal_scrubbing'] = {"error": str(e)}
    torch.cuda.empty_cache()

    # EXP4: MLP Analysis ★
    try:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting EXP4: MLP Analysis...")
        exp4_results = run_exp4_mlp_analysis(
            model, tokenizer, device, test_sentences, n_layers, d_model, n_heads, head_dim)
        all_results['exp4_mlp_analysis'] = exp4_results
    except Exception as e:
        print(f"  EXP4 FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results['exp4_mlp_analysis'] = {"error": str(e)}
    torch.cuda.empty_cache()

    # ========================================================================
    # Summary
    # ========================================================================
    print(f"\n{'='*70}")
    print(f"PHASE 211 SUMMARY — {model_name}")
    print(f"{'='*70}")

    # Exp1 Summary
    exp1 = all_results.get('exp1_ov_write', {})
    if isinstance(exp1, dict) and 'summary' in exp1:
        s = exp1['summary']
        print(f"\n--- Exp1: OV Write Analysis ---")
        print(f"  SG-PUSHER heads: {s.get('sg_pushers', 0)}")
        print(f"  PL-PUSHER heads: {s.get('pl_pushers', 0)}")
        print(f"  ROUTER heads: {s.get('routers', 0)}")
        print(f"  WRITER heads: {s.get('writers', 0)}")
        for kf in s.get('key_findings', [])[:5]:
            print(f"    {kf}")

    # Exp2 Summary
    exp2 = all_results.get('exp2_activation_patching', {})
    if isinstance(exp2, dict):
        print(f"\n--- Exp2: Activation Patching ---")
        critical = [(k, v['mean_recovery']) for k, v in exp2.items()
                    if isinstance(v, dict) and v.get('mean_recovery', 0) > 0.1]
        if critical:
            for lk, rec in sorted(critical, key=lambda x: -x[1]):
                print(f"  {lk}: recovery={rec:.4f}")
        else:
            print(f"  No critical layers (all recovery ≤ 0.1)")

    # Exp3 Summary
    exp3 = all_results.get('exp3_causal_scrubbing', {})
    if isinstance(exp3, dict):
        print(f"\n--- Exp3: Causal Scrubbing ---")
        print(f"  Same-number retention: {exp3.get('same_number_retention_rate', 0):.3f}")
        print(f"  Cross-number break: {exp3.get('cross_number_break_rate', 0):.3f}")

    # Exp4 Summary
    exp4 = all_results.get('exp4_mlp_analysis', {})
    if isinstance(exp4, dict):
        print(f"\n--- Exp4: MLP Analysis ---")
        for lk in sorted(exp4.keys()):
            r = exp4[lk]
            if isinstance(r, dict):
                print(f"  {lk}: sg_effect={r.get('sg_effect',0):.4f}, "
                      f"pl_effect={r.get('pl_effect',0):.4f}")

    # Save results
    results_path = f"tests/glm5_temp/phase211_{model_name}_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nSaved to {results_path}")

    total_time = time.time() - t_start
    print(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f}min)")

    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
