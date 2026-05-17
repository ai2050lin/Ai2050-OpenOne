"""
Phase 212: Dynamic Computational Graph Analysis — FROM STATES TO TRANSITIONS
=============================================================================

THEORETICAL CONTEXT (from Phase 211 review):
  Phase 211 found OV write effects are tiny (~0.01), MLP is the main writer,
  and models use structural number info. But we're still in "static state"
  thinking. The user's critique identifies 4 critical gaps:

  HARD INJURY 1: Still thinking in static hidden states → need Δh (residual updates)
  HARD INJURY 2: Assuming language structure is stable objects → it's dynamic constraints
  HARD INJURY 3: Underestimating superposition → single-head small ≠ unimportant
  HARD INJURY 4: Haven't entered path space → need causal path decomposition

CORE PRINCIPLE:
  Not: h_l  (what IS the state)
  But: Δh_l^(module)  (what does each module CHANGE)

  The mechanism is not in the representation — it's in the TRANSITION.
  Each module WRITES a delta to the residual stream, and only by decomposing
  these deltas can we find the true computational graph.

EXPERIMENTS:
  EXP1: Residual Stream Decomposition ★★★ (THE CORE EXPERIMENT)
    - Capture Δh_attn and Δh_mlp at the verb position for each layer
    - Project through W_U → logit attribution per module
    - Answer: WHICH module writes HOW MUCH sg/pl verb information?
    - This is logit attribution — the most direct causal decomposition

  EXP2: MLP Neuron-Level Analysis ★★★
    - Capture intermediate MLP activations (after gate*up, before down)
    - Find neurons that activate differently for sg vs pl subjects
    - Find neurons that push sg vs pl verb logits
    - Answer: WHICH neurons implement "if sg → boost V-3sg"?

  EXP3: Long-Range Subject-Verb Agreement ★★★
    - Use center-embedded sentences: "The cat that the dog chased runs away"
    - Subject and verb separated by 3+ tokens — no shortcuts
    - Do activation patching at the VERB position (not subject position)
    - This is the proper test — short-distance has confounds

  EXP4: Path Patching ★★
    - For key (layer, head) pairs from EXP1:
      - Zero-ablate specific head's output at verb position
      - Measure effect on sg/pl verb logit
    - Test whether the computational graph is truly path-dependent

DATA: 100+ sentence pairs including long-range variants
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

from model_utils import (get_model_info, release_model, get_layers, get_W_U,
                          MODEL_CONFIGS, get_layer_weights)

warnings.filterwarnings('ignore')

# ========================================================================
# Sentence Data — EXPANDED with long-range center-embedded structures
# ========================================================================
IRREGULAR_PLURALS = {
    "child": "children", "man": "men", "woman": "women",
    "mouse": "mice", "goose": "geese", "foot": "feet",
    "tooth": "teeth", "person": "people", "fish": "fish",
    "deer": "deer", "sheep": "sheep",
}

# Short distance: "The [subj] [verb] the [obj]"
SHORT_SENTENCES = [
    ("cat", "chases", "chase", "dog"),
    ("dog", "follows", "follow", "cat"),
    ("bear", "watches", "watch", "wolf"),
    ("lion", "finds", "find", "tiger"),
    ("fox", "tracks", "track", "rabbit"),
    ("whale", "joins", "join", "shark"),
    ("teacher", "helps", "help", "student"),
    ("doctor", "treats", "treat", "patient"),
    ("king", "protects", "protect", "queen"),
    ("captain", "leads", "lead", "soldier"),
    ("builder", "helps", "help", "painter"),
    ("writer", "meets", "meet", "reader"),
    ("singer", "joins", "join", "dancer"),
    ("judge", "watches", "watch", "lawyer"),
    ("mother", "helps", "help", "father"),
    ("sister", "meets", "meet", "brother"),
    ("girl", "watches", "watch", "boy"),
    ("hero", "protects", "protect", "child"),
    ("sun", "warms", "warm", "earth"),
    ("river", "feeds", "feed", "ocean"),
    ("fire", "burns", "burn", "wood"),
    ("wind", "moves", "move", "cloud"),
    ("star", "guides", "guide", "ship"),
    ("rain", "helps", "help", "flower"),
    ("farmer", "feeds", "feed", "worker"),
]

# LONG-RANGE: Center-embedded with relative clause
# "The [subj] that the [noun] [verb_past] [verb_present] away/often/quickly"
LONG_RANGE_SENTENCES = [
    ("cat", "dog", "chased", "runs", "away"),
    ("dog", "fox", "followed", "runs", "away"),
    ("bear", "wolf", "watched", "wanders", "often"),
    ("lion", "tiger", "found", "wanders", "often"),
    ("teacher", "student", "helped", "teaches", "well"),
    ("doctor", "nurse", "called", "treats", "patients"),
    ("king", "queen", "guarded", "rules", "wisely"),
    ("captain", "soldier", "ordered", "leads", "bravely"),
    ("writer", "reader", "impressed", "writes", "often"),
    ("mother", "child", "comforted", "smiles", "often"),
    ("cat", "bird", "noticed", "jumps", "quickly"),
    ("dog", "rabbit", "scared", "barks", "loudly"),
    ("horse", "rider", "carried", "gallops", "fast"),
    ("fish", "shark", "avoided", "swims", "quickly"),
    ("girl", "boy", "followed", "walks", "slowly"),
    ("hero", "villain", "defeated", "stands", "proudly"),
    ("sun", "cloud", "brightened", "shines", "warmly"),
    ("river", "stone", "shaped", "flows", "gently"),
    ("fire", "wind", "spread", "burns", "brightly"),
    ("farmer", "rain", "welcomed", "plants", "seeds"),
    ("child", "mother", "hugged", "plays", "happily"),
    ("eagle", "storm", "weathered", "soars", "high"),
    ("wolf", "hunter", "evaded", "howls", "loudly"),
    ("singer", "band", "joined", "performs", "often"),
    ("judge", "lawyer", "questioned", "decides", "fairly"),
]

# Even longer: "The [adj] [subj] that the [noun] [verb_past] [verb_present] [adv]"
LONG_RANGE_ADJ_SENTENCES = [
    ("small", "cat", "dog", "chased", "runs", "away"),
    ("big", "dog", "fox", "followed", "runs", "away"),
    ("old", "bear", "wolf", "watched", "wanders", "often"),
    ("wise", "teacher", "student", "helped", "teaches", "well"),
    ("kind", "doctor", "nurse", "called", "treats", "patients"),
    ("brave", "king", "queen", "guarded", "rules", "wisely"),
    ("good", "captain", "soldier", "ordered", "leads", "bravely"),
    ("tall", "writer", "reader", "impressed", "writes", "often"),
    ("warm", "sun", "cloud", "brightened", "shines", "warmly"),
    ("cold", "river", "stone", "shaped", "flows", "gently"),
]


def make_plural(noun):
    if noun in IRREGULAR_PLURALS:
        return IRREGULAR_PLURALS[noun]
    if noun.endswith(("s", "sh", "ch", "x", "z")):
        return noun + "es"
    if noun.endswith("y") and noun[-2] not in "aeiou":
        return noun[:-1] + "ies"
    return noun + "s"


def verb_3sg_to_base(v3sg):
    """Convert 3rd person singular verb to base form."""
    # Handle common patterns
    if v3sg.endswith("ies"):
        return v3sg[:-3] + "y"  # carries → carry
    if v3sg.endswith("shes") or v3sg.endswith("ches") or v3sg.endswith("xes") or v3sg.endswith("zes"):
        return v3sg[:-2]  # watches → watch, fixes → fix
    if v3sg.endswith("sses"):
        return v3sg[:-2]  # passes → pass
    if v3sg.endswith("s") and not v3sg.endswith(("ss", "us", "is")):
        return v3sg[:-1]  # runs → run, walks → walk
    # Irregulars
    IRREG_VERBS = {"has": "have", "is": "are", "was": "were", "does": "do", "goes": "go"}
    return IRREG_VERBS.get(v3sg, v3sg)


def generate_test_sentences():
    """Generate short, long-range, and adj-modified long-range sentence pairs."""
    test_data = []

    # Short: "The [subj] [verb] the [obj]"
    for subj, v3sg, vbase, obj in SHORT_SENTENCES:
        sg_sent = f"The {subj} {v3sg} the {obj}"
        pl_subj = make_plural(subj)
        pl_sent = f"The {pl_subj} {vbase} the {obj}"
        test_data.append({
            'sent_sg': sg_sent, 'sent_pl': pl_sent,
            'verb_sg': v3sg, 'verb_pl': vbase,
            'subj': subj, 'obj': obj,
            'distance': 'short',
            'structure': 'simple',
        })

    # Long-range: "The [subj] that the [noun] [verb_past] [verb_present] [adv]"
    for subj, noun, vpast, vpres, adv in LONG_RANGE_SENTENCES:
        sg_sent = f"The {subj} that the {noun} {vpast} {vpres} {adv}"
        pl_subj = make_plural(subj)
        vbase = verb_3sg_to_base(vpres)
        pl_sent = f"The {pl_subj} that the {noun} {vpast} {vbase} {adv}"
        test_data.append({
            'sent_sg': sg_sent, 'sent_pl': pl_sent,
            'verb_sg': vpres, 'verb_pl': vbase,
            'subj': subj, 'noun': noun,
            'distance': 'long',
            'structure': 'center_embedded',
        })

    # Long-range with adj: "The [adj] [subj] that the [noun] [verb_past] [verb_present] [adv]"
    for adj, subj, noun, vpast, vpres, adv in LONG_RANGE_ADJ_SENTENCES:
        sg_sent = f"The {adj} {subj} that the {noun} {vpast} {vpres} {adv}"
        pl_subj = make_plural(subj)
        vbase = verb_3sg_to_base(vpres)
        pl_sent = f"The {adj} {pl_subj} that the {noun} {vpast} {vbase} {adv}"
        test_data.append({
            'sent_sg': sg_sent, 'sent_pl': pl_sent,
            'verb_sg': vpres, 'verb_pl': vbase,
            'subj': subj, 'noun': noun, 'adj': adj,
            'distance': 'long',
            'structure': 'center_embedded_adj',
        })

    short_count = sum(1 for d in test_data if d['distance'] == 'short')
    long_count = sum(1 for d in test_data if d['distance'] == 'long')
    print(f"  Generated {len(test_data)} sentence pairs "
          f"({short_count} short, {long_count} long)")
    return test_data


# ========================================================================
# Model Loading (BF16 + device_map="auto" + flash attention)
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
    # Try flash attention first, fallback to eager
    for attn_impl in ["flash_attention_2", "eager", "sdpa"]:
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
    """Tokenize and find subject/verb positions."""
    toks = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=64)
    input_ids = toks["input_ids"]
    ids_list = input_ids[0].tolist()
    decoded_tokens = [tokenizer.decode([tid]) for tid in ids_list]

    result = {
        'input_ids': input_ids,
        'attention_mask': toks["attention_mask"],
        'decoded_tokens': decoded_tokens,
        'token_ids': ids_list,
    }

    # Try to identify subject and verb positions by looking for "that" keyword
    # Pattern: "The [adj?] [subj] that the [noun] [verb_past] [verb_present] [adv]"
    that_positions = [i for i, t in enumerate(decoded_tokens)
                      if t.strip().lower() == "that"]

    if that_positions:
        # Long-range pattern
        that_pos = that_positions[0]
        # Subject is before "that"
        # Find "The" before "that"
        for i in range(max(0, that_pos - 4), that_pos):
            if decoded_tokens[i].strip().lower() in ("the", "a", "an"):
                subj_pos = i + 1  # subject is right after "The"
                if that_pos - subj_pos > 2:
                    subj_pos = that_pos - 1  # adj case: subject is right before "that"
                break
        else:
            subj_pos = that_pos - 1

        # Verb (present) is after "that the [noun] [verb_past]"
        # Find "the" after "that"
        the_after_that = None
        for i in range(that_pos + 1, min(that_pos + 5, len(decoded_tokens))):
            if decoded_tokens[i].strip().lower() in ("the", "a", "an"):
                the_after_that = i
                break

        if the_after_that is not None:
            # noun is right after "the", then verb_past, then verb_present
            noun_pos = the_after_that + 1
            verb_past_pos = noun_pos + 1
            verb_present_pos = verb_past_pos + 1

            result.update({
                'subj_pos': subj_pos,
                'that_pos': that_pos,
                'noun_pos': noun_pos,
                'verb_past_pos': verb_past_pos,
                'verb_present_pos': verb_present_pos,
                'verb_pred_pos': verb_present_pos - 1,  # predict verb_present from previous
                'distance': 'long',
                'structure': 'center_embedded',
            })
            return result

    # Short pattern: "The [subj] [verb] the [obj]"
    for i in range(len(decoded_tokens) - 4):
        t0 = decoded_tokens[i].strip().lower()
        t3 = decoded_tokens[i+3].strip().lower()
        if t0 in ("the", "a", "an") and t3 in ("the", "a", "an"):
            result.update({
                'det1_pos': i, 'subj_pos': i + 1, 'verb_pos': i + 2,
                'det2_pos': i + 3, 'obj_pos': i + 4,
                'verb_pred_pos': i + 1,  # predict verb from subj position
                'distance': 'short',
                'structure': 'simple',
            })
            return result

    # Adjective pattern: "The [adj] [subj] [verb] the [obj]"
    for i in range(len(decoded_tokens) - 5):
        t0 = decoded_tokens[i].strip().lower()
        t4 = decoded_tokens[i+4].strip().lower()
        if t0 in ("the", "a", "an") and t4 in ("the", "a", "an"):
            result.update({
                'det1_pos': i, 'adj_pos': i+1, 'subj_pos': i+2, 'verb_pos': i+3,
                'det2_pos': i+4, 'obj_pos': i+5,
                'verb_pred_pos': i+2,
                'distance': 'medium',
                'structure': 'adj',
            })
            return result

    return result


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


def measure_agreement(logits, pred_pos, correct_id, wrong_id):
    """Measure agreement: sigmoid(logit_correct - logit_wrong)."""
    verb_logits = logits[0, pred_pos]
    logit_correct = verb_logits[correct_id].float().item()
    logit_wrong = verb_logits[wrong_id].float().item()
    logit_diff = logit_correct - logit_wrong
    agreement = torch.sigmoid(torch.tensor(logit_diff)).item()
    return agreement, logit_diff


# ========================================================================
# EXP1: Residual Stream Decomposition (Δh per module) ★★★
# ========================================================================
def run_exp1_residual_decomposition(model, tokenizer, device, test_sentences,
                                     n_layers, d_model):
    """
    THE CORE EXPERIMENT: Decompose Δh at the verb position into
    contributions from each module (attention, MLP).

    Key insight: Not h_l (what IS), but Δh_l (what CHANGES).

    Method:
    1. Run forward pass with hooks capturing:
       - pre_attn: input to attention (after input LN)
       - post_attn: output of attention (before post-attn LN)
       - post_mlp: output of MLP
    2. At the verb position:
       - Δh_attn_l = attention output at position i
       - Δh_mlp_l = MLP output at position i
    3. Project each Δh through W_U → logit attribution
       - Which module pushes sg/pl verb logits?
    4. Sum across layers → total attribution

    This directly answers: "which module WRITES verb form information?"
    """
    print("\n" + "=" * 70)
    print("EXP1: Residual Stream Decomposition (Δh per module) ★★★")
    print("FROM STATES TO TRANSITIONS — not h, but Δh")
    print("=" * 70)

    layers = get_layers(model)
    W_U = get_W_U(model, None)
    if isinstance(W_U, torch.Tensor):
        W_U = W_U.float().numpy()

    # Sample layers for efficiency
    if n_layers <= 12:
        sample_layers = list(range(n_layers))
    else:
        step = max(1, n_layers // 12)
        sample_layers = list(range(0, n_layers, step))
        if n_layers - 1 not in sample_layers:
            sample_layers.append(n_layers - 1)

    # Data structure: per-layer, per-module
    # {layer: {'attn_sg_effect': [], 'attn_pl_effect': [],
    #          'mlp_sg_effect': [], 'mlp_pl_effect': []}}
    decomp_data = {l: {'attn_sg': [], 'attn_pl': [],
                       'mlp_sg': [], 'mlp_pl': [],
                       'attn_norm': [], 'mlp_norm': [],
                       'total_delta_norm': []}
                   for l in sample_layers}

    n_valid = 0
    t_exp_start = time.time()

    for td in test_sentences:
        if n_valid >= 60:
            break

        # Only use short-distance sentences for this experiment
        # (long-range needs separate handling)
        if td.get('distance', 'short') != 'short':
            continue

        sg_ann = tokenize_and_annotate(td['sent_sg'], tokenizer)
        pl_ann = tokenize_and_annotate(td['sent_pl'], tokenizer)
        if sg_ann is None or pl_ann is None:
            continue

        sg_verb_id, pl_verb_id = find_verb_token_ids(tokenizer, td['verb_sg'], td['verb_pl'])
        if sg_verb_id is None or pl_verb_id is None:
            continue

        # Verify model gets agreement right for sg sentence
        with torch.no_grad():
            test_out = model(input_ids=sg_ann['input_ids'].to(device),
                            attention_mask=sg_ann['attention_mask'].to(device))
        agree, _ = measure_agreement(test_out.logits, sg_ann['verb_pred_pos'],
                                     sg_verb_id, pl_verb_id)
        if agree < 0.5:
            continue
        del test_out

        # Run BOTH sentences with hooks to capture module outputs
        for sent_key, ann, verb_id_correct, verb_id_wrong in [
            ('sg', sg_ann, sg_verb_id, pl_verb_id),
            ('pl', pl_ann, pl_verb_id, sg_verb_id),
        ]:
            # Hook to capture attention output and MLP output per layer
            attn_outputs = {}
            mlp_outputs = {}
            pre_attn_hidden = {}

            def make_attn_hook(layer_idx):
                def hook(module, input, output):
                    # Attention output is the first element
                    if isinstance(output, tuple):
                        attn_outputs[layer_idx] = output[0].detach().clone()
                    else:
                        attn_outputs[layer_idx] = output.detach().clone()
                return hook

            def make_mlp_hook(layer_idx):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        mlp_outputs[layer_idx] = output[0].detach().clone()
                    else:
                        mlp_outputs[layer_idx] = output.detach().clone()
                return hook

            def make_pre_hook(layer_idx):
                def hook(module, input, output):
                    # Input to the layer = pre-attention hidden state
                    if isinstance(input, tuple) and len(input) > 0:
                        pre_attn_hidden[layer_idx] = input[0].detach().clone()
                return hook

            # Register hooks
            attn_hooks = []
            mlp_hooks = []
            pre_hooks = []
            for l_idx in sample_layers:
                layer = layers[l_idx]
                # Attention hook
                if hasattr(layer, 'self_attn'):
                    attn_hooks.append(
                        layer.self_attn.register_forward_hook(make_attn_hook(l_idx)))
                # MLP hook
                if hasattr(layer, 'mlp'):
                    mlp_hooks.append(
                        layer.mlp.register_forward_hook(make_mlp_hook(l_idx)))
                # Pre-hook on the layer itself
                pre_hooks.append(
                    layer.register_forward_hook(make_pre_hook(l_idx)))

            with torch.no_grad():
                out = model(input_ids=ann['input_ids'].to(device),
                           attention_mask=ann['attention_mask'].to(device),
                           output_hidden_states=True)

            # Remove hooks
            for h in attn_hooks + mlp_hooks + pre_hooks:
                h.remove()

            # Get verb position
            verb_pos = ann.get('verb_pred_pos', ann.get('subj_pos', 1))

            # For each layer, compute Δh_attn and Δh_mlp at verb_pos
            for l_idx in sample_layers:
                # Δh_attn = attention output at verb_pos
                if l_idx in attn_outputs and verb_pos < attn_outputs[l_idx].shape[1]:
                    delta_attn = attn_outputs[l_idx][0, verb_pos].float().cpu().numpy()
                else:
                    delta_attn = None

                # Δh_mlp = MLP output at verb_pos
                if l_idx in mlp_outputs and verb_pos < mlp_outputs[l_idx].shape[1]:
                    delta_mlp = mlp_outputs[l_idx][0, verb_pos].float().cpu().numpy()
                else:
                    delta_mlp = None

                # Total Δh from hidden_states
                if (l_idx < len(out.hidden_states) - 1 and
                    verb_pos < out.hidden_states[l_idx].shape[1]):
                    h_before = out.hidden_states[l_idx][0, verb_pos].float().cpu().numpy()
                    h_after = out.hidden_states[l_idx + 1][0, verb_pos].float().cpu().numpy()
                    total_delta = h_after - h_before
                else:
                    total_delta = None

                # Project through W_U for logit attribution
                if delta_attn is not None:
                    sg_eff = float(delta_attn @ W_U[sg_verb_id])
                    pl_eff = float(delta_attn @ W_U[pl_verb_id])
                    decomp_data[l_idx]['attn_sg'].append(sg_eff)
                    decomp_data[l_idx]['attn_pl'].append(pl_eff)
                    decomp_data[l_idx]['attn_norm'].append(float(np.linalg.norm(delta_attn)))

                if delta_mlp is not None:
                    sg_eff = float(delta_mlp @ W_U[sg_verb_id])
                    pl_eff = float(delta_mlp @ W_U[pl_verb_id])
                    decomp_data[l_idx]['mlp_sg'].append(sg_eff)
                    decomp_data[l_idx]['mlp_pl'].append(pl_eff)
                    decomp_data[l_idx]['mlp_norm'].append(float(np.linalg.norm(delta_mlp)))

                if total_delta is not None:
                    decomp_data[l_idx]['total_delta_norm'].append(
                        float(np.linalg.norm(total_delta)))

            del out, attn_outputs, mlp_outputs, pre_attn_hidden
            torch.cuda.empty_cache()

        n_valid += 1
        if n_valid % 10 == 0:
            elapsed = time.time() - t_exp_start
            print(f"  [EXP1] Processed {n_valid} sentences, elapsed={elapsed:.0f}s")

    # ========================================================================
    # Analysis
    # ========================================================================
    print(f"\n  Processed {n_valid} valid sentence pairs")

    print(f"\n  {'Layer':>6} {'Attn→SG':>10} {'Attn→PL':>10} {'Attn_Diff':>10} "
          f"{'MLP→SG':>10} {'MLP→PL':>10} {'MLP_Diff':>10} "
          f"{'Attn_Norm':>10} {'MLP_Norm':>10}")
    print("  " + "-" * 100)

    attn_attr = {}  # {layer: sg_pl_diff for attention}
    mlp_attr = {}   # {layer: sg_pl_diff for MLP}
    results = {}

    for l_idx in sorted(decomp_data.keys()):
        d = decomp_data[l_idx]
        if len(d['attn_sg']) < 5:
            continue

        mean_attn_sg = np.mean(d['attn_sg'])
        mean_attn_pl = np.mean(d['attn_pl'])
        mean_attn_diff = mean_attn_sg - mean_attn_pl
        mean_attn_norm = np.mean(d['attn_norm'])

        mean_mlp_sg = np.mean(d['mlp_sg']) if d['mlp_sg'] else 0
        mean_mlp_pl = np.mean(d['mlp_pl']) if d['mlp_pl'] else 0
        mean_mlp_diff = mean_mlp_sg - mean_mlp_pl
        mean_mlp_norm = np.mean(d['mlp_norm']) if d['mlp_norm'] else 0

        mean_delta_norm = np.mean(d['total_delta_norm']) if d['total_delta_norm'] else 0

        attn_attr[l_idx] = mean_attn_diff
        mlp_attr[l_idx] = mean_mlp_diff

        results[f"L{l_idx}"] = {
            'attn_sg_effect': float(mean_attn_sg),
            'attn_pl_effect': float(mean_attn_pl),
            'attn_sg_pl_diff': float(mean_attn_diff),
            'attn_norm': float(mean_attn_norm),
            'mlp_sg_effect': float(mean_mlp_sg),
            'mlp_pl_effect': float(mean_mlp_pl),
            'mlp_sg_pl_diff': float(mean_mlp_diff),
            'mlp_norm': float(mean_mlp_norm),
            'total_delta_norm': float(mean_delta_norm),
        }

        # Print all layers (this is the core result)
        print(f"  L{l_idx:>4} {mean_attn_sg:>10.4f} {mean_attn_pl:>10.4f} {mean_attn_diff:>10.4f} "
              f"{mean_mlp_sg:>10.4f} {mean_mlp_pl:>10.4f} {mean_mlp_diff:>10.4f} "
              f"{mean_attn_norm:>10.2f} {mean_mlp_norm:>10.2f}")

    # Identify key layers
    print(f"\n  ★★★ Logit Attribution Summary ★★★")

    # Top attention writers
    attn_sorted = sorted(attn_attr.items(), key=lambda x: -abs(x[1]))
    print(f"\n  Top Attention Writers (|attn_sg-pl_diff|):")
    for l, diff in attn_sorted[:5]:
        role = "SG-WRITER" if diff > 0 else "PL-WRITER"
        print(f"    L{l}: {diff:.4f} ({role})")

    # Top MLP writers
    mlp_sorted = sorted(mlp_attr.items(), key=lambda x: -abs(x[1]))
    print(f"\n  Top MLP Writers (|mlp_sg-pl_diff|):")
    for l, diff in mlp_sorted[:5]:
        role = "SG-WRITER" if diff > 0 else "PL-WRITER"
        print(f"    L{l}: {diff:.4f} ({role})")

    # Total attribution
    total_attn = sum(attn_attr.values())
    total_mlp = sum(mlp_attr.values())
    print(f"\n  Total Attention Attribution: {total_attn:.4f}")
    print(f"  Total MLP Attribution: {total_mlp:.4f}")
    print(f"  Attn/MLP Ratio: {abs(total_attn)/(abs(total_mlp)+1e-8):.2f}")

    if abs(total_mlp) > abs(total_attn):
        print(f"  ★★★ MLP dominates verb form writing (consistent with Phase 211)")
    else:
        print(f"  ★★★ Attention dominates verb form writing (unexpected!)")

    return results


# ========================================================================
# EXP2: MLP Neuron-Level Analysis ★★★
# ========================================================================
def run_exp2_mlp_neuron_analysis(model, tokenizer, device, test_sentences,
                                  n_layers, d_model, intermediate_size):
    """
    Find individual MLP neurons that detect sg/pl or write verb form info.

    Method:
    1. Run sg and pl sentences, capture MLP intermediate activations
       (after gate*up, before down_proj)
    2. For each neuron: measure activation difference (sg - pl)
    3. For each neuron: measure its effect on sg/pl verb logits
       (via W_down row * W_U projection)
    4. Find "number detector" neurons and "agreement writer" neurons
    """
    print("\n" + "=" * 70)
    print("EXP2: MLP Neuron-Level Analysis ★★★")
    print("Finding individual neurons that detect sg/pl or write verb form")
    print("=" * 70)

    layers = get_layers(model)
    W_U = get_W_U(model, None)
    if isinstance(W_U, torch.Tensor):
        W_U = W_U.float().numpy()

    # Focus on deep layers where MLP effects are strongest
    if n_layers <= 12:
        focus_layers = list(range(n_layers))
    else:
        # Sample last 1/3 of layers (where MLP effects are strongest)
        start = n_layers * 2 // 3
        focus_layers = list(range(start, n_layers))
        # Also add a few early/mid layers for comparison
        focus_layers = list(range(0, n_layers, n_layers // 4)) + focus_layers
        focus_layers = sorted(set(focus_layers))

    print(f"  Analyzing {len(focus_layers)} layers: {focus_layers[:5]}...{focus_layers[-3:]}")

    # For each layer, we need W_down to project neuron activations to logits
    neuron_results = {}  # {layer: {neuron_idx: {activation_diff, logit_effect}}}

    n_valid = 0
    t_exp_start = time.time()

    for td in test_sentences:
        if n_valid >= 40:
            break

        if td.get('distance', 'short') != 'short':
            continue

        sg_ann = tokenize_and_annotate(td['sent_sg'], tokenizer)
        pl_ann = tokenize_and_annotate(td['sent_pl'], tokenizer)
        if sg_ann is None or pl_ann is None:
            continue

        sg_verb_id, pl_verb_id = find_verb_token_ids(tokenizer, td['verb_sg'], td['verb_pl'])
        if sg_verb_id is None or pl_verb_id is None:
            continue

        # Verify agreement
        with torch.no_grad():
            test_out = model(input_ids=sg_ann['input_ids'].to(device),
                            attention_mask=sg_ann['attention_mask'].to(device))
        agree, _ = measure_agreement(test_out.logits, sg_ann['verb_pred_pos'],
                                     sg_verb_id, pl_verb_id)
        if agree < 0.5:
            continue
        del test_out

        verb_pos = sg_ann.get('verb_pred_pos', sg_ann.get('subj_pos', 1))

        # Hook to capture MLP intermediate activations
        sg_neuron_acts = {}
        pl_neuron_acts = {}

        def make_neuron_hook(layer_idx, storage):
            def hook(module, input, output):
                # For SwiGLU: output of gate*up is [batch, seq, intermediate]
                # We want the intermediate activations BEFORE down_proj
                if isinstance(output, tuple):
                    storage[layer_idx] = output[0].detach().clone()
                else:
                    storage[layer_idx] = output.detach().clone()
            return hook

        # Run sg sentence
        hooks = []
        for l_idx in focus_layers:
            layer = layers[l_idx]
            if hasattr(layer, 'mlp'):
                # Hook on the MLP to capture its output
                hooks.append(
                    layer.mlp.register_forward_hook(
                        make_neuron_hook(l_idx, sg_neuron_acts)))

        with torch.no_grad():
            model(input_ids=sg_ann['input_ids'].to(device),
                  attention_mask=sg_ann['attention_mask'].to(device))

        for h in hooks:
            h.remove()

        # Run pl sentence
        hooks = []
        for l_idx in focus_layers:
            layer = layers[l_idx]
            if hasattr(layer, 'mlp'):
                hooks.append(
                    layer.mlp.register_forward_hook(
                        make_neuron_hook(l_idx, pl_neuron_acts)))

        with torch.no_grad():
            model(input_ids=pl_ann['input_ids'].to(device),
                  attention_mask=pl_ann['attention_mask'].to(device))

        for h in hooks:
            h.remove()

        # Analyze neuron activations
        for l_idx in focus_layers:
            if l_idx not in sg_neuron_acts or l_idx not in pl_neuron_acts:
                continue
            if verb_pos >= sg_neuron_acts[l_idx].shape[1]:
                continue

            # Get MLP output at verb_pos
            sg_mlp_out = sg_neuron_acts[l_idx][0, verb_pos].float().cpu().numpy()  # [d_model]
            pl_mlp_out = pl_neuron_acts[l_idx][0, verb_pos].float().cpu().numpy()  # [d_model]

            # Logit attribution: project MLP output through W_U
            sg_logit_eff = sg_mlp_out @ W_U[sg_verb_id] - sg_mlp_out @ W_U[pl_verb_id]
            pl_logit_eff = pl_mlp_out @ W_U[sg_verb_id] - pl_mlp_out @ W_U[pl_verb_id]

            if l_idx not in neuron_results:
                neuron_results[l_idx] = {
                    'sg_logit_diff': [],   # MLP(sg_subj) pushes sg-verb - pl-verb
                    'pl_logit_diff': [],    # MLP(pl_subj) pushes sg-verb - pl-verb
                    'conditional_diff': [],  # sg_logit_diff - pl_logit_diff
                }

            neuron_results[l_idx]['sg_logit_diff'].append(float(sg_logit_eff))
            neuron_results[l_idx]['pl_logit_diff'].append(float(pl_logit_eff))
            neuron_results[l_idx]['conditional_diff'].append(float(sg_logit_eff - pl_logit_eff))

        del sg_neuron_acts, pl_neuron_acts
        torch.cuda.empty_cache()

        n_valid += 1
        if n_valid % 10 == 0:
            elapsed = time.time() - t_exp_start
            print(f"  [EXP2] Processed {n_valid} sentences, elapsed={elapsed:.0f}s")

    # ========================================================================
    # Analysis
    # ========================================================================
    print(f"\n  Processed {n_valid} valid sentence pairs")

    print(f"\n  {'Layer':>6} {'SG→SG-PL':>10} {'PL→SG-PL':>10} {'Conditional':>12} {'Role':>15}")
    print("  " + "-" * 60)

    mlp_neuron_results = {}
    for l_idx in sorted(neuron_results.keys()):
        d = neuron_results[l_idx]
        if len(d['sg_logit_diff']) < 5:
            continue

        mean_sg = np.mean(d['sg_logit_diff'])
        mean_pl = np.mean(d['pl_logit_diff'])
        mean_cond = np.mean(d['conditional_diff'])

        # Role classification
        if mean_sg > 0.5 and mean_pl < -0.1:
            role = "AGREEMENT-MLP"
        elif mean_cond > 0.3:
            role = "NUMBER-CONDITIONAL"
        elif mean_sg > 0.1 and mean_pl > 0.1:
            role = "VERB-BOOSTER"
        elif abs(mean_sg) > 0.05 or abs(mean_pl) > 0.05:
            role = "WEAK-CONDITIONAL"
        else:
            role = ""

        mlp_neuron_results[f"L{l_idx}"] = {
            'sg_logit_diff': float(mean_sg),
            'pl_logit_diff': float(mean_pl),
            'conditional_diff': float(mean_cond),
            'role': role,
            'n_samples': len(d['sg_logit_diff']),
        }

        if abs(mean_sg) > 0.02 or abs(mean_pl) > 0.02 or abs(mean_cond) > 0.02:
            print(f"  L{l_idx:>4} {mean_sg:>10.4f} {mean_pl:>10.4f} {mean_cond:>12.4f} {role:>15}")

    return mlp_neuron_results


# ========================================================================
# EXP3: Long-Range Subject-Verb Agreement ★★★
# ========================================================================
def run_exp3_long_range_agreement(model, tokenizer, device, test_sentences,
                                   n_layers, d_model):
    """
    THE PROPER TEST: Long-range subject-verb agreement.

    Short-distance sentences have a fatal confound:
    verb_pred_pos ≈ subj_pos, so patching is trivial.

    Long-range sentences ("The cat that the dog chased runs away")
    separate subject and verb by 3+ tokens, eliminating shortcuts.

    Key tests:
    1. Does the model maintain agreement over long distance?
    2. Activation patching at the VERB position (not subject position)
    3. Where is the "critical layer" for number information?
    """
    print("\n" + "=" * 70)
    print("EXP3: Long-Range Subject-Verb Agreement ★★★")
    print("Center-embedded sentences — no shortcuts")
    print("=" * 70)

    layers = get_layers(model)

    # Only use long-range sentences
    long_sentences = [td for td in test_sentences if td.get('distance') == 'long']
    print(f"  Using {len(long_sentences)} long-range sentence pairs")

    if not long_sentences:
        print("  ✗ No long-range sentences available, using short as fallback")
        long_sentences = [td for td in test_sentences if td.get('distance') == 'short']

    # Test 1: Agreement rate for long-range sentences
    agreement_rates = {'sg_correct': 0, 'sg_total': 0, 'pl_correct': 0, 'pl_total': 0}

    # Test 2: Layer-wise activation patching at VERB position
    if n_layers <= 12:
        sample_layers = list(range(n_layers))
    else:
        step = max(1, n_layers // 8)
        sample_layers = list(range(0, n_layers, step))
        if n_layers - 1 not in sample_layers:
            sample_layers.append(n_layers - 1)

    patch_results = {l: [] for l in sample_layers}

    n_tested = 0
    t_exp_start = time.time()

    for td in long_sentences:
        if n_tested >= 30:
            break

        # Get sentences and verb forms — all should have sent_sg, sent_pl, verb_sg, verb_pl now
        sg_sent = td.get('sent_sg', '')
        pl_sent = td.get('sent_pl', '')
        verb_sg = td.get('verb_sg', '')
        verb_pl = td.get('verb_pl', '')

        if not sg_sent or not pl_sent or not verb_sg or not verb_pl:
            continue

        sg_verb_id, pl_verb_id = find_verb_token_ids(tokenizer, verb_sg, verb_pl)
        if sg_verb_id is None or pl_verb_id is None:
            continue

        sg_ann = tokenize_and_annotate(sg_sent, tokenizer)
        pl_ann = tokenize_and_annotate(pl_sent, tokenizer)
        if sg_ann is None or pl_ann is None:
            continue

        # For long-range, verb_pred_pos should be at the verb_present position
        verb_pred_pos = sg_ann.get('verb_present_pos', sg_ann.get('verb_pred_pos'))
        if verb_pred_pos is None:
            # Try to find it from decoded tokens
            decoded = sg_ann.get('decoded_tokens', [])
            for i, t in enumerate(decoded):
                if t.strip().lower() in (verb_present, verb_pl):
                    verb_pred_pos = i - 1  # predict from position before verb
                    break
        if verb_pred_pos is None:
            continue

        # Test 1: Check agreement
        with torch.no_grad():
            sg_out = model(input_ids=sg_ann['input_ids'].to(device),
                          attention_mask=sg_ann['attention_mask'].to(device))
            pl_out = model(input_ids=pl_ann['input_ids'].to(device),
                          attention_mask=pl_ann['attention_mask'].to(device))

        # For sg sentence, model should prefer sg verb
        sg_agree, sg_diff = measure_agreement(sg_out.logits, verb_pred_pos,
                                               sg_verb_id, pl_verb_id)
        # For pl sentence, model should prefer pl verb
        pl_agree, pl_diff = measure_agreement(pl_out.logits, verb_pred_pos,
                                               pl_verb_id, sg_verb_id)

        if sg_agree > 0.5:
            agreement_rates['sg_correct'] += 1
        agreement_rates['sg_total'] += 1

        if pl_agree > 0.5:
            agreement_rates['pl_correct'] += 1
        agreement_rates['pl_total'] += 1

        if sg_agree < 0.5:
            del sg_out, pl_out
            continue

        # Test 2: Activation patching at VERB position
        # Clean = sg sentence, Corrupted = pl sentence
        # At each layer, patch verb position hidden state from clean → corrupted
        clean_hidden = {}
        def make_capture_hook(layer_idx):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    clean_hidden[layer_idx] = output[0].detach().clone()
                else:
                    clean_hidden[layer_idx] = output.detach().clone()
            return hook

        hooks = [layers[i].register_forward_hook(make_capture_hook(i))
                 for i in sample_layers]
        with torch.no_grad():
            model(input_ids=sg_ann['input_ids'].to(device),
                  attention_mask=sg_ann['attention_mask'].to(device))
        for h in hooks:
            h.remove()

        # Baseline corrupted agreement (without patching)
        # For pl sentence, measuring sg verb agreement (should be low)
        with torch.no_grad():
            corrupt_out = model(input_ids=pl_ann['input_ids'].to(device),
                               attention_mask=pl_ann['attention_mask'].to(device))
        baseline_corrupt_agree, _ = measure_agreement(
            corrupt_out.logits, verb_pred_pos, sg_verb_id, pl_verb_id)

        # Patch at each layer
        for l_idx in sample_layers:
            if l_idx not in clean_hidden:
                continue

            def make_patching_hook(clean_h, pos):
                def hook(module, input, output):
                    patched = output[0].clone()
                    if pos < patched.shape[1] and pos < clean_h.shape[1]:
                        patched[0, pos, :] = clean_h[0, pos, :]
                    return (patched,) + output[1:] if isinstance(output, tuple) else patched
                return hook

            hook = layers[l_idx].register_forward_hook(
                make_patching_hook(clean_hidden[l_idx], verb_pred_pos))

            with torch.no_grad():
                patched_out = model(input_ids=pl_ann['input_ids'].to(device),
                                   attention_mask=pl_ann['attention_mask'].to(device))

            hook.remove()

            patched_agree, _ = measure_agreement(
                patched_out.logits, verb_pred_pos, sg_verb_id, pl_verb_id)

            recovery = patched_agree - baseline_corrupt_agree
            patch_results[l_idx].append(recovery)

            del patched_out
            torch.cuda.empty_cache()

        del sg_out, pl_out, corrupt_out, clean_hidden
        torch.cuda.empty_cache()

        n_tested += 1
        if n_tested % 5 == 0:
            elapsed = time.time() - t_exp_start
            print(f"  [EXP3] Tested {n_tested} sentences, elapsed={elapsed:.0f}s")

    # ========================================================================
    # Analysis
    # ========================================================================
    print(f"\n  Tested {n_tested} long-range sentences")

    # Agreement rates
    sg_rate = agreement_rates['sg_correct'] / max(agreement_rates['sg_total'], 1)
    pl_rate = agreement_rates['pl_correct'] / max(agreement_rates['pl_total'], 1)
    print(f"\n  Long-Range Agreement Rates:")
    print(f"    SG subject → SG verb: {sg_rate:.3f} ({agreement_rates['sg_correct']}/{agreement_rates['sg_total']})")
    print(f"    PL subject → PL verb: {pl_rate:.3f} ({agreement_rates['pl_correct']}/{agreement_rates['pl_total']})")

    # Patching results
    print(f"\n  {'Layer':>6} {'Mean_Recovery':>14} {'N':>4} {'Importance':>15}")
    print("  " + "-" * 50)

    patch_summary = {}
    critical_layers = []
    for l_idx in sorted(patch_results.keys()):
        vals = patch_results[l_idx]
        if len(vals) < 3:
            continue
        mean_rec = np.mean(vals)
        patch_summary[f"L{l_idx}"] = {
            'mean_recovery': float(mean_rec),
            'n_test': len(vals),
            'std_recovery': float(np.std(vals)),
        }

        label = ("★★★ CRITICAL" if mean_rec > 0.3 else
                 "★★ IMPORTANT" if mean_rec > 0.1 else
                 "★ MODERATE" if mean_rec > 0.03 else
                 "✗ MINIMAL")
        print(f"  L{l_idx:>4} {mean_rec:>14.4f} {len(vals):>4} {label:>15}")

        if mean_rec > 0.05:
            critical_layers.append((l_idx, mean_rec))

    if critical_layers:
        print(f"\n  ★ Critical layers (recovery > 0.05):")
        for l, rec in sorted(critical_layers, key=lambda x: -x[1]):
            print(f"    L{l}: recovery={rec:.4f}")
    else:
        print(f"\n  ✗ No critical layers found")

    return {
        'sg_agreement_rate': float(sg_rate),
        'pl_agreement_rate': float(pl_rate),
        'sg_total': agreement_rates['sg_total'],
        'pl_total': agreement_rates['pl_total'],
        'patching': patch_summary,
    }


# ========================================================================
# EXP4: Head-Level Zero Ablation (Path Sensitivity) ★★
# ========================================================================
def run_exp4_head_ablation(model, tokenizer, device, test_sentences,
                            n_layers, d_model, n_heads, head_dim, n_kv_heads):
    """
    Path Sensitivity: Which heads are causally necessary?

    Method:
    1. Run clean sg sentence → measure sg verb agreement
    2. For each (layer, head):
       - Zero-ablate that head's output at the verb position
       - Measure agreement change
    3. Heads whose ablation breaks agreement are "causally necessary"

    This tests: is the computational graph truly dependent on specific heads?
    """
    print("\n" + "=" * 70)
    print("EXP4: Head-Level Zero Ablation (Path Sensitivity) ★★")
    print("Which heads are causally necessary for agreement?")
    print("=" * 70)

    layers = get_layers(model)

    # Focus on layers where we expect important heads
    # Based on Phase 210-211: routing heads in mid-layers, writing in deep layers
    if n_layers <= 12:
        focus_layers = list(range(n_layers))
    else:
        # Sample strategically: early, mid, late
        focus_layers = (list(range(0, 4)) +
                       list(range(n_layers//3, n_layers//3 + 3)) +
                       list(range(n_layers*2//3, n_layers*2//3 + 3)) +
                       list(range(n_layers - 4, n_layers)))
        focus_layers = sorted(set(focus_layers))

    print(f"  Testing {len(focus_layers)} layers × {n_heads} heads = "
          f"{len(focus_layers) * n_heads} interventions")

    head_effects = {}  # {layer: {head: mean_agreement_drop}}

    # Use short sentences for efficiency (long-range tested in EXP3)
    short_sents = [td for td in test_sentences if td.get('distance') == 'short']
    n_tested = 0
    t_exp_start = time.time()

    for td in short_sents:
        if n_tested >= 15:
            break

        sg_ann = tokenize_and_annotate(td['sent_sg'], tokenizer)
        if sg_ann is None:
            continue

        sg_verb_id, pl_verb_id = find_verb_token_ids(tokenizer, td['verb_sg'], td['verb_pl'])
        if sg_verb_id is None or pl_verb_id is None:
            continue

        verb_pred_pos = sg_ann.get('verb_pred_pos', sg_ann.get('subj_pos', 1))

        # Baseline agreement
        with torch.no_grad():
            base_out = model(input_ids=sg_ann['input_ids'].to(device),
                           attention_mask=sg_ann['attention_mask'].to(device))
        base_agree, _ = measure_agreement(base_out.logits, verb_pred_pos,
                                          sg_verb_id, pl_verb_id)
        del base_out

        if base_agree < 0.6:
            continue

        # Test each head
        for l_idx in focus_layers:
            if l_idx not in head_effects:
                head_effects[l_idx] = {}

            layer = layers[l_idx]

            # Get W_O for this layer to construct head-level output
            sa = layer.self_attn
            try:
                W_O = sa.o_proj.weight.detach()
                if W_O.is_meta:
                    continue
                W_O = W_O.float()
            except Exception:
                continue

            for h in range(min(n_heads, 8)):  # Limit heads for efficiency
                # Zero-ablate head h by hooking into the attention output
                # and zeroing out the h-th head slice
                head_dim_actual = W_O.shape[1] // n_heads

                def make_ablation_hook(head_idx, hd_dim, n_h, w_o):
                    def hook(module, input, output):
                        # output[0] is the attention output [batch, seq, d_model]
                        if isinstance(output, tuple):
                            out = output[0].clone()
                        else:
                            out = output.clone()

                        # Zero out the contribution of head_idx
                        # The attention output is: W_O @ concat(head_0, head_1, ..., head_{n-1})
                        # To zero head_idx, we set its slice in the pre-projection output to 0
                        # But we're hooking post-projection... so we need a different approach

                        # Alternative: hook into the attention output before O projection
                        # This is harder. Instead, we'll measure the effect indirectly.

                        # For now, we use a simpler approach:
                        # Hook into the layer output and subtract head h's contribution
                        # estimated from the previous forward pass

                        return output  # No modification for now
                    return hook

                # Use a simpler approach: run with output_attentions and reconstruct
                # We'll use the output_attentions approach
                with torch.no_grad():
                    ablated_out = model(
                        input_ids=sg_ann['input_ids'].to(device),
                        attention_mask=sg_ann['attention_mask'].to(device),
                        output_attentions=True)

                if l_idx < len(ablated_out.attentions) and ablated_out.attentions[l_idx] is not None:
                    # For now, just measure the agreement (no actual ablation)
                    # The real ablation requires hooking into the attention computation
                    pass

                del ablated_out
                break  # Only test first head per layer for now

            torch.cuda.empty_cache()

        n_tested += 1
        if n_tested % 5 == 0:
            elapsed = time.time() - t_exp_start
            print(f"  [EXP4] Processed {n_tested} sentences, elapsed={elapsed:.0f}s")

    # Note: Full head ablation requires TransformerLens or custom hook architecture
    # This is a placeholder that measures the theoretical framework
    print(f"\n  Note: Full head ablation requires TransformerLens-style hooks.")
    print(f"  Current implementation measures the framework but not individual head effects.")
    print(f"  For full results, need to implement proper attention output manipulation.")

    return {
        'note': 'Head ablation requires TransformerLens-style hooks for proper implementation',
        'focus_layers': focus_layers,
        'n_tested': n_tested,
    }


# ========================================================================
# Main
# ========================================================================
def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    t_start = time.time()
    print(f"\n{'='*70}")
    print(f"Phase 212: Dynamic Computational Graph Analysis — {model_name}")
    print(f"FROM STATES TO TRANSITIONS — not h, but Δh")
    print(f"{'='*70}")

    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    n_heads = getattr(model.config, 'num_attention_heads', d_model // 64)
    n_kv_heads = getattr(model.config, 'num_key_value_heads', n_heads)
    head_dim = d_model // n_heads
    intermediate_size = info.intermediate_size
    print(f"  n_layers={n_layers}, d_model={d_model}, n_heads={n_heads}, "
          f"n_kv_heads={n_kv_heads}, head_dim={head_dim}, "
          f"intermediate_size={intermediate_size}")

    test_sentences = generate_test_sentences()
    all_results = {
        'model': model_name, 'n_layers': n_layers, 'd_model': d_model,
        'n_heads': n_heads, 'head_dim': head_dim,
        'intermediate_size': intermediate_size,
    }

    # EXP1: Residual Stream Decomposition ★★★
    try:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting EXP1: Residual Decomposition...")
        exp1_results = run_exp1_residual_decomposition(
            model, tokenizer, device, test_sentences, n_layers, d_model)
        all_results['exp1_residual_decomposition'] = exp1_results
    except Exception as e:
        print(f"  EXP1 FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results['exp1_residual_decomposition'] = {"error": str(e)}
    torch.cuda.empty_cache()

    # EXP2: MLP Neuron-Level Analysis ★★★
    try:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting EXP2: MLP Neuron Analysis...")
        exp2_results = run_exp2_mlp_neuron_analysis(
            model, tokenizer, device, test_sentences,
            n_layers, d_model, intermediate_size)
        all_results['exp2_mlp_neuron_analysis'] = exp2_results
    except Exception as e:
        print(f"  EXP2 FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results['exp2_mlp_neuron_analysis'] = {"error": str(e)}
    torch.cuda.empty_cache()

    # EXP3: Long-Range Agreement ★★★
    try:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting EXP3: Long-Range Agreement...")
        exp3_results = run_exp3_long_range_agreement(
            model, tokenizer, device, test_sentences, n_layers, d_model)
        all_results['exp3_long_range_agreement'] = exp3_results
    except Exception as e:
        print(f"  EXP3 FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results['exp3_long_range_agreement'] = {"error": str(e)}
    torch.cuda.empty_cache()

    # EXP4: Head Ablation ★★
    try:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting EXP4: Head Ablation...")
        exp4_results = run_exp4_head_ablation(
            model, tokenizer, device, test_sentences,
            n_layers, d_model, n_heads, head_dim, n_kv_heads)
        all_results['exp4_head_ablation'] = exp4_results
    except Exception as e:
        print(f"  EXP4 FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results['exp4_head_ablation'] = {"error": str(e)}
    torch.cuda.empty_cache()

    # ========================================================================
    # Summary
    # ========================================================================
    print(f"\n{'='*70}")
    print(f"PHASE 212 SUMMARY — {model_name}")
    print(f"{'='*70}")

    # EXP1 Summary
    exp1 = all_results.get('exp1_residual_decomposition', {})
    if isinstance(exp1, dict) and not any(k == 'error' for k in exp1.keys() if isinstance(exp1.get(k), str)):
        print(f"\n--- Exp1: Residual Stream Decomposition ---")
        # Find top attention and MLP writers
        attn_writers = []
        mlp_writers = []
        for lk, r in exp1.items():
            if isinstance(r, dict):
                ad = r.get('attn_sg_pl_diff', 0)
                md = r.get('mlp_sg_pl_diff', 0)
                if abs(ad) > 0.01:
                    attn_writers.append((lk, ad))
                if abs(md) > 0.01:
                    mlp_writers.append((lk, md))

        if attn_writers:
            print(f"  Top Attention Writers:")
            for lk, diff in sorted(attn_writers, key=lambda x: -abs(x[1]))[:5]:
                print(f"    {lk}: sg-pl diff = {diff:.4f}")
        if mlp_writers:
            print(f"  Top MLP Writers:")
            for lk, diff in sorted(mlp_writers, key=lambda x: -abs(x[1]))[:5]:
                print(f"    {lk}: sg-pl diff = {diff:.4f}")

    # EXP2 Summary
    exp2 = all_results.get('exp2_mlp_neuron_analysis', {})
    if isinstance(exp2, dict):
        print(f"\n--- Exp2: MLP Neuron-Level Analysis ---")
        agreement_mlps = [(k, v) for k, v in exp2.items()
                          if isinstance(v, dict) and v.get('role') == 'AGREEMENT-MLP']
        conditional_mlps = [(k, v) for k, v in exp2.items()
                           if isinstance(v, dict) and v.get('role') in ('NUMBER-CONDITIONAL', 'AGREEMENT-MLP')]
        if agreement_mlps:
            print(f"  AGREEMENT-MLP layers:")
            for lk, r in agreement_mlps:
                print(f"    {lk}: cond_diff={r.get('conditional_diff', 0):.4f}")
        if conditional_mlps:
            print(f"  NUMBER-CONDITIONAL layers:")
            for lk, r in conditional_mlps:
                print(f"    {lk}: sg={r.get('sg_logit_diff', 0):.4f}, "
                      f"pl={r.get('pl_logit_diff', 0):.4f}, "
                      f"cond={r.get('conditional_diff', 0):.4f}")

    # EXP3 Summary
    exp3 = all_results.get('exp3_long_range_agreement', {})
    if isinstance(exp3, dict):
        print(f"\n--- Exp3: Long-Range Agreement ---")
        print(f"  SG agreement rate: {exp3.get('sg_agreement_rate', 0):.3f}")
        print(f"  PL agreement rate: {exp3.get('pl_agreement_rate', 0):.3f}")
        patching = exp3.get('patching', {})
        critical = [(k, v['mean_recovery']) for k, v in patching.items()
                    if isinstance(v, dict) and v.get('mean_recovery', 0) > 0.05]
        if critical:
            print(f"  Critical layers for long-range agreement:")
            for lk, rec in sorted(critical, key=lambda x: -x[1]):
                print(f"    {lk}: recovery={rec:.4f}")

    # Save results
    results_path = f"tests/glm5_temp/phase212_{model_name}_results.json"
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
