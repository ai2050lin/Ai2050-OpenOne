"""
Phase 172: Attention Transport & Constraint Propagation
=========================================================

★★★ PARADIGM SHIFT: From "Jacobian geometry" → "Attention transport + constraint dynamics"! ★★★

User's key theoretical corrections (ALL CORRECT):

1. "Attention is the REAL transport operator" — h_i ← Σ_j α_{ij} V_j
   The Jacobian is a DERIVATIVE of this process, not the process itself.

2. "Transformer is non-autonomous" — Each layer has DIFFERENT dynamics.
   A_l = A(h, token, context, attention_pattern), not A(h, t).

3. "Residual flow: h_{l+1} = h_l + F_l(h_l)" — This is Euler flow.
   The "force" F_l = F_l^{attn} + F_l^{mlp} is the actual dynamical object.
   h_{l+1} - h_l = attention_residual + mlp_residual

4. "Late-layer contraction = probabilistic projection curvature" — NOT semantic curvature.

5. "Language = constraint propagation" — Syntactic/logical/semantic constraints
   propagate through the network via attention. The key object is WHICH
   constraints are carried by WHICH attention patterns.

6. "Mode switching" — The model switches between different computational modes:
   - syntax mode (parsing grammatical structure)
   - retrieval mode (accessing stored knowledge)
   - reasoning mode (applying logical constraints)
   These correspond to DIFFERENT attention pattern regimes.

7. "Sparse activation topology" — Language structure may be in the sparse graph
   of which heads carry which constraints, not in continuous geometry.

★★★ FOUR CRITICAL EXPERIMENTS ★★★

Exp 1: ★★★ Attention Transport Structure — 注意力输运结构
  - Extract attention patterns α_{ij}^{(l)} for diverse sentences
  - Measure: attention entropy, inter-token flow, head specialization
  - KEY QUESTION: What is the "transport graph" of attention?

Exp 2: ★★★ Residual Flow Analysis — 残差流分析
  - F_l = h_{l+1} - h_l (the "force" or "increment")
  - ||F_l|| / ||h_l|| = relative force magnitude
  - cos(F_l, h_l) = alignment of force with state
  - KEY QUESTION: Is the system "near-identity + small force" or "large force"?

Exp 3: ★★★ Constraint Propagation — 约束传播
  - Subject-verb agreement: "The cat sleeps" vs "The cats sleep"
  - Long-distance agreement: "The cat that the dogs chased sleeps"
  - Gender agreement: "The woman said she" vs "The man said he"
  - Measure: attention weight from constraint-source to constraint-target
  - KEY QUESTION: Do constraints propagate through attention?
  - Which heads carry the constraint signal?

Exp 4: ★★ Mode Switching — 模式切换
  - Factual: "Paris is the capital of"
  - Syntactic: "The cat that the dog chased ran"
  - Reasoning: "If all men are mortal and Socrates is a man then"
  - Translation: "The French word for cat is"
  - Compare attention patterns across modes
  - KEY QUESTION: Do different modes activate different head clusters?

Usage: python tests/glm5/phase172_attention_transport.py <model_name>
  model_name: qwen3, glm4, deepseek7b
"""

import sys
import os
import time
import json
import gc
import numpy as np
from datetime import datetime
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'glm5'))

from model_utils import get_layers, get_model_info, release_model, MODEL_CONFIGS


# ===== SENTENCE SETS =====

# Simple subject-verb agreement (15 pairs × 2 = 30 sentences)
AGREEMENT_PAIRS = [
    ("cat", "cats", "sleeps", "sleep"),
    ("dog", "dogs", "runs", "run"),
    ("bird", "birds", "flies", "fly"),
    ("horse", "horses", "walks", "walk"),
    ("child", "children", "plays", "play"),
    ("woman", "women", "reads", "read"),
    ("man", "men", "works", "work"),
    ("tree", "trees", "grows", "grow"),
    ("car", "cars", "moves", "move"),
    ("river", "rivers", "flows", "flow"),
    ("star", "stars", "shines", "shine"),
    ("book", "books", "opens", "open"),
    ("door", "doors", "closes", "close"),
    ("light", "lights", "flickers", "flicker"),
    ("flower", "flowers", "blooms", "bloom"),
]

# Long-distance agreement (8 pairs × 2 = 16 sentences)
LONG_DISTANCE_PAIRS = [
    ("cat", "cats", "dog", "dogs", "chased", "sleeps", "sleep"),
    ("bird", "birds", "cat", "cats", "watched", "flies", "fly"),
    ("key", "keys", "door", "doors", "opened", "was", "were"),
    ("man", "men", "woman", "women", "saw", "runs", "run"),
    ("tree", "trees", "boy", "boys", "climbed", "grows", "grow"),
    ("king", "kings", "queen", "queens", "served", "rules", "rule"),
    ("doctor", "doctors", "nurse", "nurses", "helped", "arrives", "arrive"),
    ("student", "students", "teacher", "teachers", "praised", "studies", "study"),
]

# Gender agreement (8 pairs × 2 = 16 sentences)
GENDER_PAIRS = [
    ("woman", "man", "she", "he"),
    ("girl", "boy", "her", "his"),
    ("queen", "king", "her", "his"),
    ("mother", "father", "her", "his"),
    ("sister", "brother", "her", "his"),
    ("aunt", "uncle", "her", "his"),
    ("lady", "lord", "her", "his"),
    ("wife", "husband", "her", "his"),
]

# Mode switching sentences (10 per mode = 40 sentences)
MODE_SENTENCES = {
    "factual": [
        "Paris is the capital of France",
        "Water freezes at zero degrees",
        "The Earth orbits around the sun",
        "Two plus two equals four",
        "The largest ocean is the Pacific",
        "Iron is a type of metal",
        "The speed of light is very fast",
        "Beethoven was a famous composer",
        "The moon causes the tides",
        "Oxygen is necessary for breathing",
    ],
    "syntactic": [
        "The cat that the dog chased ran",
        "The more the merrier the better",
        "What the man who the woman saw did",
        "It was the cat that the dog bit",
        "The boy whose father the teacher praised",
        "Not only did she sing but she danced",
        "Hardly had he arrived when she left",
        "The book that I told you about is",
        "She is more intelligent than him",
        "Had I known that I would have gone",
    ],
    "reasoning": [
        "If all men are mortal and Socrates is a man then he must",
        "Since A is greater than B and B is greater than C therefore A is",
        "Given that every cat is an animal and every animal is alive then cats are",
        "If it rains then the ground is wet and it is raining so the ground is",
        "All birds can fly and penguins are birds therefore penguins can",
        "If the switch is on then the light is on and the switch is on so the light",
        "Since today is Monday and yesterday was Sunday then tomorrow is",
        "If x equals five and y equals three then x plus y equals",
        "Because all mammals have hair and whales are mammals therefore whales have",
        "If no students failed and everyone took the test then everyone",
    ],
    "translation": [
        "The French word for cat is chat",
        "In Spanish the word for dog is perro",
        "The German word for book is Buch",
        "The Italian word for love is amore",
        "The Japanese word for hello is konnichiwa",
        "The Chinese word for mountain is shan",
        "The Portuguese word for water is agua",
        "The Russian word for goodbye is do svidaniya",
        "The Korean word for thank you is gamsahamnida",
        "The Arabic word for peace is salaam",
    ],
}


# ===== MODEL LOADING =====

def load_model_auto_bf16(model_name):
    """Load model with bfloat16 + device_map='auto' (no 8-bit)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = MODEL_CONFIGS[model_name]
    print(f"[load] Loading {model_name} with bfloat16 + device_map='auto'...")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[load] GPU memory: {gpu_mem:.1f}GB")

        if model_name == "qwen3":
            model = AutoModelForCausalLM.from_pretrained(
                cfg["path"],
                torch_dtype=torch.bfloat16,
                device_map="cpu",
                trust_remote_code=True,
                local_files_only=True,
                low_cpu_mem_usage=True,
                attn_implementation="eager",
            )
            model = model.to("cuda")
        else:
            model = AutoModelForCausalLM.from_pretrained(
                cfg["path"],
                torch_dtype=torch.bfloat16,
                device_map="auto",
                max_memory={0: "10GiB", "cpu": "30GiB"},
                trust_remote_code=True,
                local_files_only=True,
                low_cpu_mem_usage=True,
                attn_implementation="eager",
            )

        model.eval()
        device = next(model.parameters()).device
        gpu_alloc = torch.cuda.memory_allocated() / 1e9
        print(f"[load] {model_name} loaded in bfloat16, device={device}, "
              f"class={type(model).__name__}, GPU={gpu_alloc:.2f}GB")
        use_8bit = False

    except Exception as e:
        print(f"[load] bfloat16+auto failed: {e}")
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            cfg["path"],
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
            attn_implementation="eager",
        )
        model.eval()
        device = next(model.parameters()).device
        gpu_alloc = torch.cuda.memory_allocated() / 1e9
        print(f"[load] {model_name} loaded in 8-bit, device={device}, "
              f"class={type(model).__name__}, GPU={gpu_alloc:.2f}GB")
        use_8bit = True

    return model, tokenizer, device, use_8bit


# ===== UTILITY FUNCTIONS =====

def get_device_for_input(model):
    import torch
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cosine_sim(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-10 or n2 < 1e-10:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def forward_extract(model, tokenizer, prompt, max_length=64):
    """Run forward pass and extract attention + hidden states."""
    import torch

    input_device = get_device_for_input(model)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = inputs["input_ids"].to(input_device)
    attention_mask = inputs["attention_mask"].to(input_device)

    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            output_hidden_states=True,
        )

    tokens = [tokenizer.decode([t.item()]) for t in input_ids[0]]

    return {
        "attentions": out.attentions,  # tuple of [1, n_heads, seq, seq]
        "hidden_states": out.hidden_states,  # tuple of [1, seq, d_model]
        "tokens": tokens,
        "seq_len": len(tokens),
    }


# ===== EXP 1: ATTENTION TRANSPORT STRUCTURE ★★★ =====

def exp1_attention_transport(model, tokenizer, device, model_name, use_8bit):
    """
    ★★★ Attention Transport Structure ★★★

    KEY QUESTION: What is the "transport graph" of attention?

    Metrics per layer:
    1. Attention entropy: H = -Σ α log(α) for each head
       Low entropy = focused transport (specific token→token)
       High entropy = diffuse transport (broadcast)

    2. Head specialization index: variance of head entropies
       High variance = diverse transport strategies (specialized heads)
       Low variance = uniform transport

    3. Local vs global attention: how much attention goes to
       nearby (±1) vs distant tokens?

    4. Self-attention fraction: how much does each position attend to itself?
    """
    import torch

    print("\n" + "="*60)
    print("EXP 1: Attention Transport Structure ★★★")
    print("="*60)

    info = get_model_info(model, model_name)
    n_layers = info.n_layers

    # Use mode sentences for diversity
    all_sentences = []
    for mode, sents in MODE_SENTENCES.items():
        for s in sents:
            all_sentences.append((mode, s))

    print(f"  Testing {len(all_sentences)} sentences across 4 modes")

    # Per-layer aggregate statistics
    layer_stats = defaultdict(lambda: {
        "entropies": [], "local_fracs": [], "global_fracs": [],
        "self_attns": [], "head_variances": [],
    })

    for sidx, (mode, sentence) in enumerate(all_sentences):
        result = forward_extract(model, tokenizer, sentence)
        attentions = result["attentions"]
        seq_len = result["seq_len"]

        for l_idx, attn in enumerate(attentions):
            if attn is None:
                continue
            attn_np = attn[0].float().cpu().numpy()  # [n_heads, seq, seq]
            n_heads = attn_np.shape[0]

            head_entropies = []
            head_local = []
            head_self = []

            for h in range(n_heads):
                for i in range(seq_len):
                    # Entropy of attention from position i
                    probs = np.maximum(attn_np[h, i, :], 1e-10)
                    entropy = -np.sum(probs * np.log(probs))
                    head_entropies.append(float(entropy))

                    # Local fraction: attention to ±1 positions
                    local_mass = 0.0
                    total_mass = 0.0
                    for j in range(seq_len):
                        total_mass += attn_np[h, i, j]
                        if abs(i - j) <= 1:
                            local_mass += attn_np[h, i, j]
                    head_local.append(local_mass / max(total_mass, 1e-10))

                    # Self-attention
                    head_self.append(float(attn_np[h, i, i]))

            layer_stats[l_idx]["entropies"].extend(head_entropies)
            layer_stats[l_idx]["local_fracs"].extend(head_local)
            layer_stats[l_idx]["self_attns"].extend(head_self)

            # Head variance: compute per-head average entropy, then variance across heads
            per_head_avg = []
            for h in range(n_heads):
                h_ent = [head_entropies[h * seq_len + i] for i in range(seq_len)]
                per_head_avg.append(np.mean(h_ent))
            layer_stats[l_idx]["head_variances"].append(float(np.var(per_head_avg)))

        if (sidx + 1) % 10 == 0:
            print(f"    Processed {sidx+1}/{len(all_sentences)} sentences")

    # Aggregate
    layer_summary = {}
    for l in range(n_layers):
        if l not in layer_stats or not layer_stats[l]["entropies"]:
            continue
        ls = layer_stats[l]
        layer_summary[f"L{l}"] = {
            "avg_entropy": round(float(np.mean(ls["entropies"])), 4),
            "entropy_std": round(float(np.std(ls["entropies"])), 4),
            "avg_local_frac": round(float(np.mean(ls["local_fracs"])), 4),
            "avg_self_attn": round(float(np.mean(ls["self_attns"])), 4),
            "avg_head_variance": round(float(np.mean(ls["head_variances"])), 6),
        }

    # Print key layers
    key_layers = [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]
    print(f"\n  === Attention Transport Summary ===")
    for l in key_layers:
        key = f"L{l}"
        if key in layer_summary:
            ls = layer_summary[key]
            print(f"    {key}: entropy={ls['avg_entropy']:.3f}±{ls['entropy_std']:.3f}, "
                  f"local={ls['avg_local_frac']:.3f}, self={ls['avg_self_attn']:.4f}, "
                  f"head_var={ls['avg_head_variance']:.5f}")

    # Identify transition points
    entropies = [layer_summary[f"L{l}"]["avg_entropy"]
                 for l in range(n_layers) if f"L{l}" in layer_summary]
    local_fracs = [layer_summary[f"L{l}"]["avg_local_frac"]
                   for l in range(n_layers) if f"L{l}" in layer_summary]

    results = {
        "layer_summary": layer_summary,
        "overall": {
            "entropy_trend": round(float(entropies[-1] - entropies[0]), 4) if len(entropies) > 1 else 0,
            "local_attn_trend": round(float(local_fracs[-1] - local_fracs[0]), 4) if len(local_fracs) > 1 else 0,
            "entropy_early": round(float(np.mean(entropies[:len(entropies)//3])), 4) if entropies else 0,
            "entropy_mid": round(float(np.mean(entropies[len(entropies)//3:2*len(entropies)//3])), 4) if entropies else 0,
            "entropy_late": round(float(np.mean(entropies[2*len(entropies)//3:])), 4) if entropies else 0,
        },
        "n_sentences": len(all_sentences),
    }

    return results


# ===== EXP 2: RESIDUAL FLOW ANALYSIS ★★★ =====

def exp2_residual_flow(model, tokenizer, device, model_name, use_8bit):
    """
    ★★★ Residual Flow Analysis — F_l = h_{l+1} - h_l ★★★

    KEY QUESTION: Is the system "near-identity + small force" or "large force"?

    User's insight: h_{l+1} = h_l + F_l(h_l) is an Euler flow.
    The "force" F_l = F_l^{attn} + F_l^{mlp} is the actual dynamical object.

    Metrics per layer:
    1. ||F_l|| / ||h_l|| = relative force magnitude
       If small (< 0.1) → near-identity → residual dominates
       If large (> 0.3) → strong force → attention/MLP dominates

    2. cos(F_l, h_l) = alignment of force with state
       If positive → force amplifies current state
       If negative → force opposes current state
       If ≈0 → force is orthogonal (pure rotation/transport)

    3. ||F_l||^2 decomposition: how much of the force is
       "parallel to state" vs "orthogonal to state"?
    """
    import torch

    print("\n" + "="*60)
    print("EXP 2: Residual Flow Analysis ★★★")
    print("="*60)

    info = get_model_info(model, model_name)
    n_layers = info.n_layers

    # Use all mode sentences
    all_sentences = []
    for mode, sents in MODE_SENTENCES.items():
        for s in sents:
            all_sentences.append(s)

    # Also add agreement sentences
    for sg, pl, sv, pv in AGREEMENT_PAIRS:
        all_sentences.append(f"The {sg} {sv}")
        all_sentences.append(f"The {pl} {pv}")

    print(f"  Testing {len(all_sentences)} sentences")

    layer_stats = defaultdict(lambda: {
        "force_ratios": [], "force_state_cosines": [],
        "force_norms": [], "state_norms": [],
        "ortho_fracs": [], "parallel_fracs": [],
    })

    for sidx, sentence in enumerate(all_sentences):
        result = forward_extract(model, tokenizer, sentence)
        hs = result["hidden_states"]

        for l in range(min(len(hs) - 1, n_layers)):
            h_l = hs[l][0, -1, :].float().cpu().numpy()
            h_next = hs[l + 1][0, -1, :].float().cpu().numpy()

            h_l = np.nan_to_num(h_l, nan=0.0, posinf=1e4, neginf=-1e4)
            h_next = np.nan_to_num(h_next, nan=0.0, posinf=1e4, neginf=-1e4)

            F_l = h_next - h_l  # The "force"
            norm_h = np.linalg.norm(h_l)
            norm_F = np.linalg.norm(F_l)

            # Force ratio
            force_ratio = norm_F / max(norm_h, 1e-10)

            # Force-state cosine
            cos_F_h = cosine_sim(F_l, h_l)

            # Decompose force into parallel and orthogonal
            if norm_h > 1e-10 and norm_F > 1e-10:
                h_unit = h_l / norm_h
                F_parallel = np.dot(F_l, h_unit) * h_unit
                F_orthogonal = F_l - F_parallel
                parallel_frac = np.linalg.norm(F_parallel)**2 / max(norm_F**2, 1e-20)
                ortho_frac = np.linalg.norm(F_orthogonal)**2 / max(norm_F**2, 1e-20)
            else:
                parallel_frac = 0
                ortho_frac = 0

            layer_stats[l]["force_ratios"].append(force_ratio)
            layer_stats[l]["force_state_cosines"].append(cos_F_h)
            layer_stats[l]["force_norms"].append(norm_F)
            layer_stats[l]["state_norms"].append(norm_h)
            layer_stats[l]["ortho_fracs"].append(ortho_frac)
            layer_stats[l]["parallel_fracs"].append(parallel_frac)

        if (sidx + 1) % 10 == 0:
            print(f"    Processed {sidx+1}/{len(all_sentences)} sentences")

    # Aggregate
    layer_summary = {}
    for l in range(n_layers):
        if l not in layer_stats or not layer_stats[l]["force_ratios"]:
            continue
        ls = layer_stats[l]
        layer_summary[f"L{l}"] = {
            "avg_force_ratio": round(float(np.mean(ls["force_ratios"])), 4),
            "avg_force_state_cos": round(float(np.mean(ls["force_state_cosines"])), 4),
            "avg_ortho_frac": round(float(np.mean(ls["ortho_fracs"])), 4),
            "avg_parallel_frac": round(float(np.mean(ls["parallel_fracs"])), 4),
            "avg_force_norm": round(float(np.mean(ls["force_norms"])), 2),
            "avg_state_norm": round(float(np.mean(ls["state_norms"])), 2),
        }

    # Print key layers
    key_layers = [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]
    print(f"\n  === Residual Flow Summary ===")
    for l in key_layers:
        key = f"L{l}"
        if key in layer_summary:
            ls = layer_summary[key]
            print(f"    {key}: ||F||/||h||={ls['avg_force_ratio']:.4f}, "
                  f"cos(F,h)={ls['avg_force_state_cos']:.4f}, "
                  f"ortho={ls['avg_ortho_frac']:.3f}, parallel={ls['avg_parallel_frac']:.3f}")

    # Key insight: is the system "near-identity + small force"?
    force_ratios = [layer_summary[f"L{l}"]["avg_force_ratio"]
                    for l in range(n_layers) if f"L{l}" in layer_summary]
    ortho_fracs = [layer_summary[f"L{l}"]["avg_ortho_frac"]
                   for l in range(n_layers) if f"L{l}" in layer_summary]

    results = {
        "layer_summary": layer_summary,
        "overall": {
            "avg_force_ratio": round(float(np.mean(force_ratios)), 4) if force_ratios else 0,
            "max_force_ratio": round(float(np.max(force_ratios)), 4) if force_ratios else 0,
            "force_ratio_early": round(float(np.mean(force_ratios[:len(force_ratios)//3])), 4) if force_ratios else 0,
            "force_ratio_mid": round(float(np.mean(force_ratios[len(force_ratios)//3:2*len(force_ratios)//3])), 4) if force_ratios else 0,
            "force_ratio_late": round(float(np.mean(force_ratios[2*len(force_ratios)//3:])), 4) if force_ratios else 0,
            "avg_ortho_frac": round(float(np.mean(ortho_fracs)), 4) if ortho_fracs else 0,
            "near_identity_system": float(np.mean(force_ratios)) < 0.15 if force_ratios else False,
        },
        "n_sentences": len(all_sentences),
    }

    return results


# ===== EXP 3: CONSTRAINT PROPAGATION ★★★ =====

def exp3_constraint_propagation(model, tokenizer, device, model_name, use_8bit):
    """
    ★★★ Constraint Propagation — 约束传播 ★★★

    KEY QUESTION: Do syntactic/logical constraints propagate through attention?

    Test 1: Simple subject-verb agreement
      "The cat sleeps" (singular) vs "The cats sleep" (plural)
      - At the VERB position, measure attention from verb→subject
      - The NUMBER constraint should propagate from subject to verb via attention

    Test 2: Long-distance agreement
      "The cat that the dogs chased sleeps" (singular subject, plural intervenor)
      - Does attention still connect the distant subject to the verb?

    Test 3: Gender agreement
      "The woman said she" vs "The man said he"
      - At the PRONOUN position, measure attention to the gendered noun

    KEY METRICS:
    1. Attention weight from target position to source position
       (verb→subject for agreement, pronoun→noun for gender)
    2. Hidden state difference at target position between conditions
    3. Which heads carry the constraint signal?
       (heads with high attention from target→source that also have
        different values between conditions)
    """
    import torch

    print("\n" + "="*60)
    print("EXP 3: Constraint Propagation ★★★")
    print("="*60)

    info = get_model_info(model, model_name)
    n_layers = info.n_layers

    # ---- Test 1: Simple agreement ----
    print("\n  --- Test 1: Simple Subject-Verb Agreement ---")
    agreement_results = {"singular": [], "plural": []}

    for sg_noun, pl_noun, sg_verb, pl_verb in AGREEMENT_PAIRS:
        for condition, noun, verb in [("singular", sg_noun, sg_verb),
                                       ("plural", pl_noun, pl_verb)]:
            sentence = f"The {noun} {verb}"
            result = forward_extract(model, tokenizer, sentence)
            tokens = result["tokens"]
            attentions = result["attentions"]
            seq_len = result["seq_len"]

            # Find subject position (index of noun token) and verb position
            # For "The cat sleeps": tokens ≈ ["The", " cat", " sleeps"]
            # subject_pos = 1, verb_pos = 2
            subject_pos = None
            verb_pos = None
            for i, tok in enumerate(tokens):
                tok_clean = tok.strip().lower()
                if tok_clean == noun.lower() and subject_pos is None:
                    subject_pos = i
                if tok_clean.startswith(verb.lower()[:3]) and verb_pos is None and i > 0:
                    verb_pos = i

            if subject_pos is None or verb_pos is None:
                # Fallback: assume position 1 and 2
                if seq_len >= 3:
                    subject_pos = 1
                    verb_pos = 2

            if subject_pos is None or verb_pos is None:
                continue

            # Extract attention from verb position to subject position
            verb_to_subject_attn = []  # per layer, per head
            for l_idx, attn in enumerate(attentions):
                if attn is None:
                    continue
                attn_np = attn[0].float().cpu().numpy()  # [n_heads, seq, seq]
                n_heads = attn_np.shape[0]
                for h in range(n_heads):
                    w = float(attn_np[h, verb_pos, subject_pos])
                    verb_to_subject_attn.append({
                        "layer": l_idx, "head": h, "weight": w
                    })

            # Hidden state at verb position
            hs = result["hidden_states"]
            verb_hidden = {}
            for l in range(min(len(hs), n_layers + 1)):
                h_vec = hs[l][0, verb_pos, :].float().cpu().numpy()
                h_vec = np.nan_to_num(h_vec, nan=0.0, posinf=1e4, neginf=-1e4)
                verb_hidden[l] = h_vec

            agreement_results[condition].append({
                "sentence": sentence,
                "tokens": tokens,
                "subject_pos": subject_pos,
                "verb_pos": verb_pos,
                "verb_to_subject_attn": verb_to_subject_attn,
                "verb_hidden_norm": {l: round(float(np.linalg.norm(v)), 2) for l, v in verb_hidden.items()},
            })

    # Compare singular vs plural
    print(f"  Agreement: {len(agreement_results['singular'])} singular, "
          f"{len(agreement_results['plural'])} plural")

    # Aggregate attention weights per layer
    agreement_attn_by_layer = defaultdict(lambda: {"singular": [], "plural": []})
    for condition in ["singular", "plural"]:
        for item in agreement_results[condition]:
            for entry in item["verb_to_subject_attn"]:
                agreement_attn_by_layer[entry["layer"]][condition].append(entry["weight"])

    agreement_layer_summary = {}
    for l in range(n_layers):
        if l not in agreement_attn_by_layer:
            continue
        sg_attns = agreement_attn_by_layer[l]["singular"]
        pl_attns = agreement_attn_by_layer[l]["plural"]
        agreement_layer_summary[f"L{l}"] = {
            "singular_attn": round(float(np.mean(sg_attns)), 4) if sg_attns else 0,
            "plural_attn": round(float(np.mean(pl_attns)), 4) if pl_attns else 0,
            "attn_diff": round(float(np.mean(pl_attns) - np.mean(sg_attns)), 4) if sg_attns and pl_attns else 0,
        }

    # ---- Test 2: Long-distance agreement ----
    print("\n  --- Test 2: Long-Distance Agreement ---")
    longdist_results = {"singular_across_plural": [], "plural_across_singular": []}

    for sg_subj, pl_subj, sg_interv, pl_interv, past_v, sg_verb, pl_verb in LONG_DISTANCE_PAIRS:
        for condition, subj, interv, verb in [
            ("singular_across_plural", sg_subj, pl_interv, sg_verb),
            ("plural_across_singular", pl_subj, sg_interv, pl_verb)]:
            sentence = f"The {subj} that the {interv} {past_v} {verb}"
            result = forward_extract(model, tokenizer, sentence)
            tokens = result["tokens"]
            attentions = result["attentions"]
            seq_len = result["seq_len"]

            # Find positions
            subj_pos = None
            verb_pos = None
            interv_pos = None
            for i, tok in enumerate(tokens):
                tok_clean = tok.strip().lower()
                if tok_clean == subj.lower() and subj_pos is None:
                    subj_pos = i
                if tok_clean == interv.lower() and interv_pos is None and i > subj_pos if subj_pos else True:
                    interv_pos = i
                if tok_clean.startswith(verb.lower()[:3]) and verb_pos is None and i > 2:
                    verb_pos = i

            if subj_pos is None or verb_pos is None:
                continue

            # Attention from verb to subject (distant) and to intervening noun (near)
            verb_to_subj_attn = []
            verb_to_interv_attn = []
            for l_idx, attn in enumerate(attentions):
                if attn is None:
                    continue
                attn_np = attn[0].float().cpu().numpy()
                n_heads = attn_np.shape[0]
                for h in range(n_heads):
                    w_subj = float(attn_np[h, verb_pos, subj_pos])
                    w_interv = float(attn_np[h, verb_pos, interv_pos]) if interv_pos is not None else 0
                    verb_to_subj_attn.append({"layer": l_idx, "head": h, "weight": w_subj})
                    verb_to_interv_attn.append({"layer": l_idx, "head": h, "weight": w_interv})

            longdist_results[condition].append({
                "sentence": sentence,
                "subj_pos": subj_pos, "verb_pos": verb_pos, "interv_pos": interv_pos,
                "verb_to_subj_attn": verb_to_subj_attn,
                "verb_to_interv_attn": verb_to_interv_attn,
            })

    # ---- Test 3: Gender agreement ----
    print("\n  --- Test 3: Gender Agreement ---")
    gender_results = {"feminine": [], "masculine": []}

    for fem_noun, masc_noun, fem_pron, masc_pron in GENDER_PAIRS:
        for condition, noun, pron in [("feminine", fem_noun, fem_pron),
                                      ("masculine", masc_noun, masc_pron)]:
            sentence = f"The {noun} said {pron}"
            result = forward_extract(model, tokenizer, sentence)
            tokens = result["tokens"]
            attentions = result["attentions"]

            # Find positions
            noun_pos = None
            pron_pos = None
            for i, tok in enumerate(tokens):
                tok_clean = tok.strip().lower()
                if tok_clean == noun.lower() and noun_pos is None:
                    noun_pos = i
                if tok_clean == pron.lower() and pron_pos is None:
                    pron_pos = i

            if noun_pos is None or pron_pos is None:
                continue

            pron_to_noun_attn = []
            for l_idx, attn in enumerate(attentions):
                if attn is None:
                    continue
                attn_np = attn[0].float().cpu().numpy()
                n_heads = attn_np.shape[0]
                for h in range(n_heads):
                    w = float(attn_np[h, pron_pos, noun_pos])
                    pron_to_noun_attn.append({"layer": l_idx, "head": h, "weight": w})

            # Hidden state at pronoun position
            hs = result["hidden_states"]
            pron_hidden = {}
            for l in range(min(len(hs), n_layers + 1)):
                h_vec = hs[l][0, pron_pos, :].float().cpu().numpy()
                h_vec = np.nan_to_num(h_vec, nan=0.0, posinf=1e4, neginf=-1e4)
                pron_hidden[l] = h_vec

            gender_results[condition].append({
                "sentence": sentence,
                "noun_pos": noun_pos, "pron_pos": pron_pos,
                "pron_to_noun_attn": pron_to_noun_attn,
                "pron_hidden_norm": {l: round(float(np.linalg.norm(v)), 2) for l, v in pron_hidden.items()},
            })

    # ---- Identify constraint-carrying heads ----
    print("\n  --- Identifying Constraint-Carrying Heads ---")

    # For agreement: heads where attention from verb to subject
    # differs significantly between singular and plural
    constraint_heads_agreement = {}
    for l in range(n_layers):
        sg_attns = agreement_attn_by_layer[l]["singular"] if l in agreement_attn_by_layer else []
        pl_attns = agreement_attn_by_layer[l]["plural"] if l in agreement_attn_by_layer else []
        if sg_attns and pl_attns:
            # Heads where verb→subject attention is above median
            all_attns = sg_attns + pl_attns
            median_attn = np.median(all_attns)
            high_attn_count = sum(1 for a in all_attns if a > median_attn)
            constraint_heads_agreement[f"L{l}"] = {
                "median_attn": round(float(median_attn), 4),
                "high_attn_fraction": round(high_attn_count / max(len(all_attns), 1), 4),
            }

    # Summary
    print(f"\n  === Constraint Propagation Summary ===")

    # Agreement: average verb→subject attention by layer section
    sg_attns_all = []
    pl_attns_all = []
    for l in range(n_layers):
        key = f"L{l}"
        if key in agreement_layer_summary:
            sg_attns_all.append(agreement_layer_summary[key]["singular_attn"])
            pl_attns_all.append(agreement_layer_summary[key]["plural_attn"])

    print(f"  Agreement: avg verb→subject attn: singular={np.mean(sg_attns_all):.4f}, "
          f"plural={np.mean(pl_attns_all):.4f}, diff={np.mean(pl_attns_all)-np.mean(sg_attns_all):.4f}")

    # Long-distance: verb→subject vs verb→intervening
    ld_subj_attns = []
    ld_interv_attns = []
    for condition in longdist_results:
        for item in longdist_results[condition]:
            for entry in item["verb_to_subj_attn"]:
                ld_subj_attns.append(entry["weight"])
            for entry in item["verb_to_interv_attn"]:
                ld_interv_attns.append(entry["weight"])

    if ld_subj_attns and ld_interv_attns:
        print(f"  Long-distance: verb→subject avg={np.mean(ld_subj_attns):.4f}, "
              f"verb→intervening avg={np.mean(ld_interv_attns):.4f}, "
              f"ratio={np.mean(ld_subj_attns)/max(np.mean(ld_interv_attns),1e-10):.3f}")

    # Gender: pronoun→noun attention
    fem_attns = []
    masc_attns = []
    for item in gender_results["feminine"]:
        for entry in item["pron_to_noun_attn"]:
            fem_attns.append(entry["weight"])
    for item in gender_results["masculine"]:
        for entry in item["pron_to_noun_attn"]:
            masc_attns.append(entry["weight"])

    if fem_attns and masc_attns:
        print(f"  Gender: pronoun→noun attn: feminine={np.mean(fem_attns):.4f}, "
              f"masculine={np.mean(masc_attns):.4f}")

    results = {
        "agreement": {
            "n_singular": len(agreement_results["singular"]),
            "n_plural": len(agreement_results["plural"]),
            "layer_summary": agreement_layer_summary,
            "constraint_heads": constraint_heads_agreement,
        },
        "long_distance": {
            "n_sg_across_pl": len(longdist_results["singular_across_plural"]),
            "n_pl_across_sg": len(longdist_results["plural_across_singular"]),
            "avg_verb_to_subj_attn": round(float(np.mean(ld_subj_attns)), 4) if ld_subj_attns else 0,
            "avg_verb_to_interv_attn": round(float(np.mean(ld_interv_attns)), 4) if ld_interv_attns else 0,
            "subj_vs_interv_ratio": round(float(np.mean(ld_subj_attns) / max(np.mean(ld_interv_attns), 1e-10)), 4)
                if ld_subj_attns and ld_interv_attns else 0,
        },
        "gender": {
            "n_feminine": len(gender_results["feminine"]),
            "n_masculine": len(gender_results["masculine"]),
            "avg_fem_pron_to_noun_attn": round(float(np.mean(fem_attns)), 4) if fem_attns else 0,
            "avg_masc_pron_to_noun_attn": round(float(np.mean(masc_attns)), 4) if masc_attns else 0,
        },
    }

    return results


# ===== EXP 4: MODE SWITCHING ★★ =====

def exp4_mode_switching(model, tokenizer, device, model_name, use_8bit):
    """
    ★★ Mode Switching — 模式切换 ★★

    KEY QUESTION: Do different sentence types activate different head clusters?

    Compare attention patterns across 4 modes:
    - factual: retrieval of stored knowledge
    - syntactic: complex grammatical processing
    - reasoning: logical constraint satisfaction
    - translation: cross-domain mapping

    Metrics:
    1. Average attention entropy per mode (higher = more diffuse = less focused)
    2. Local vs global attention fraction per mode
    3. Head specialization per mode (which heads are most active?)
    4. Cross-mode similarity of attention patterns
    """
    import torch

    print("\n" + "="*60)
    print("EXP 4: Mode Switching ★★")
    print("="*60)

    info = get_model_info(model, model_name)
    n_layers = info.n_layers

    mode_attention_data = defaultdict(lambda: {
        "entropies": [], "local_fracs": [], "global_fracs": [],
        "self_attns": [],
    })

    for mode, sentences in MODE_SENTENCES.items():
        print(f"  Processing mode: {mode} ({len(sentences)} sentences)")

        for sentence in sentences:
            result = forward_extract(model, tokenizer, sentence)
            attentions = result["attentions"]
            seq_len = result["seq_len"]

            for l_idx, attn in enumerate(attentions):
                if attn is None:
                    continue
                attn_np = attn[0].float().cpu().numpy()  # [n_heads, seq, seq]
                n_heads = attn_np.shape[0]

                for h in range(n_heads):
                    head_ent = []
                    head_local = []
                    head_self = []
                    for i in range(seq_len):
                        probs = np.maximum(attn_np[h, i, :], 1e-10)
                        entropy = -np.sum(probs * np.log(probs))
                        head_ent.append(entropy)

                        local_mass = sum(attn_np[h, i, j] for j in range(seq_len) if abs(i-j) <= 1)
                        total_mass = sum(attn_np[h, i, j] for j in range(seq_len))
                        head_local.append(local_mass / max(total_mass, 1e-10))
                        head_self.append(float(attn_np[h, i, i]))

                    mode_attention_data[mode]["entropies"].append(np.mean(head_ent))
                    mode_attention_data[mode]["local_fracs"].append(np.mean(head_local))
                    mode_attention_data[mode]["self_attns"].append(np.mean(head_self))

    # Compute mode-specific statistics
    mode_summary = {}
    for mode in MODE_SENTENCES:
        md = mode_attention_data[mode]
        mode_summary[mode] = {
            "avg_entropy": round(float(np.mean(md["entropies"])), 4) if md["entropies"] else 0,
            "avg_local_frac": round(float(np.mean(md["local_fracs"])), 4) if md["local_fracs"] else 0,
            "avg_self_attn": round(float(np.mean(md["self_attns"])), 4) if md["self_attns"] else 0,
            "n_patterns": len(md["entropies"]),
        }

    # Cross-mode similarity: compare head pattern statistics (not raw patterns, which have different lengths)
    # Use scalar statistics instead of pattern vectors
    cross_mode_similarity = {}
    mode_names = list(MODE_SENTENCES.keys())
    for i in range(len(mode_names)):
        for j in range(i+1, len(mode_names)):
            m1, m2 = mode_names[i], mode_names[j]
            md1, md2 = mode_attention_data[m1], mode_attention_data[m2]
            # Similarity based on entropy and local fraction correlation
            if md1["entropies"] and md2["entropies"]:
                e1, e2 = np.mean(md1["entropies"]), np.mean(md2["entropies"])
                l1, l2 = np.mean(md1["local_fracs"]), np.mean(md2["local_fracs"])
                s1, s2 = np.mean(md1["self_attns"]), np.mean(md2["self_attns"])
                # Cosine similarity of the feature vector
                v1 = np.array([e1, l1, s1])
                v2 = np.array([e2, l2, s2])
                sim = cosine_sim(v1, v2)
                cross_mode_similarity[f"{m1}_vs_{m2}"] = round(sim, 4)

    print(f"\n  === Mode Switching Summary ===")
    for mode, ms in mode_summary.items():
        print(f"    {mode}: entropy={ms['avg_entropy']:.3f}, local={ms['avg_local_frac']:.3f}, "
              f"self_attn={ms['avg_self_attn']:.4f}")

    print(f"\n  Cross-mode similarities:")
    for key, sim in sorted(cross_mode_similarity.items(), key=lambda x: x[1]):
        print(f"    {key}: {sim:.4f}")

    # Detect mode-specific heads
    # For each mode, find heads that have significantly different entropy from the average
    all_entropies_by_mode = {}
    for mode in MODE_SENTENCES:
        all_entropies_by_mode[mode] = mode_attention_data[mode]["entropies"]

    overall_mean_ent = np.mean([e for ents in all_entropies_by_mode.values() for e in ents]) if all_entropies_by_mode else 0

    mode_specificity = {}
    for mode, ents in all_entropies_by_mode.items():
        if ents:
            mode_mean = np.mean(ents)
            # Positive = more diffuse than average, negative = more focused
            mode_specificity[mode] = round(float(mode_mean - overall_mean_ent), 4)
        else:
            mode_specificity[mode] = 0

    results = {
        "mode_summary": mode_summary,
        "cross_mode_similarity": cross_mode_similarity,
        "mode_specificity": mode_specificity,
        "n_sentences_per_mode": {mode: len(sents) for mode, sents in MODE_SENTENCES.items()},
    }

    return results


# ===== MAIN =====

def main():
    import torch

    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    print(f"\n{'='*60}")
    print(f"Phase 172: Attention Transport & Constraint Propagation")
    print(f"Model: {model_name}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}")

    # Load model
    model, tokenizer, device, use_8bit = load_model_auto_bf16(model_name)
    info = get_model_info(model, model_name)
    print(f"[0] Model: {info.model_class}, L={info.n_layers}, d={info.d_model}, "
          f"V={info.vocab_size}, 8bit={use_8bit}")

    all_results = {
        "model": model_name,
        "model_class": info.model_class,
        "n_layers": info.n_layers,
        "d_model": info.d_model,
        "vocab_size": info.vocab_size,
        "use_8bit": use_8bit,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

    # ===== Exp 1: Attention Transport =====
    print(f"\n[1] Exp 1: Attention Transport Structure")
    t0 = time.time()
    all_results["exp1_attention_transport"] = exp1_attention_transport(
        model, tokenizer, device, model_name, use_8bit)
    print(f"  Time: {time.time()-t0:.1f}s")

    # ===== Exp 2: Residual Flow =====
    print(f"\n[2] Exp 2: Residual Flow Analysis")
    t0 = time.time()
    all_results["exp2_residual_flow"] = exp2_residual_flow(
        model, tokenizer, device, model_name, use_8bit)
    print(f"  Time: {time.time()-t0:.1f}s")

    # ===== Exp 3: Constraint Propagation =====
    print(f"\n[3] Exp 3: Constraint Propagation")
    t0 = time.time()
    all_results["exp3_constraint_propagation"] = exp3_constraint_propagation(
        model, tokenizer, device, model_name, use_8bit)
    print(f"  Time: {time.time()-t0:.1f}s")

    # ===== Exp 4: Mode Switching =====
    print(f"\n[4] Exp 4: Mode Switching")
    t0 = time.time()
    all_results["exp4_mode_switching"] = exp4_mode_switching(
        model, tokenizer, device, model_name, use_8bit)
    print(f"  Time: {time.time()-t0:.1f}s")

    # ===== Save Results =====
    os.makedirs("tests/glm5_temp", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = f"tests/glm5_temp/phase172_{model_name}_{timestamp}.json"

    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, tuple):
            return list(obj)
        return obj

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=convert, ensure_ascii=False)

    print(f"\n[5] Results saved to: {out_path}")

    # Release model
    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\nPhase 172 complete for {model_name}!")


if __name__ == "__main__":
    main()
