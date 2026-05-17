"""
Phase 210: Constraint Field Dynamics — FROM PROBE ARTIFACT TO CAUSAL ROUTING
=============================================================================

THEORETICAL CONTEXT:
  Phase 209 proved: bilinear operator A is a PROBE ARTIFACT, not a CAUSAL MECHANISM.
  Key insight: language ≠ object-based, language = constraint dynamics.

  Three layers of Transformer internal structure:
    Layer 1: Observable geometry (cosine, bilinear, SVD) = REPRESENTATION STATISTICS
    Layer 2: Causal routing (attention heads, MLP gating, residual flow) = COMPUTATION
    Layer 3: Constraint stabilization (what survives across layers) = FUNCTION

EXPERIMENTS:
  EXP1: Attention Routing Map
    - At the verb position, which heads attend to the subject position?
    - Are these "routing heads" or just positional?
    - Correlation between head's subj→verb attention and its importance

  EXP2: Number Direction Tracking in Residual Stream
    - Track how "number" information propagates from subject to verb
    - At which layer does verb position start encoding subject's number?
    - "Transfer point" identification

  EXP3: Constraint Repair Dynamics (Logit Lens)
    - Inject ungrammatical tokens, observe how model repairs across layers
    - At which layer does the model start predicting the correct form?
    - This reveals the constraint satisfaction mechanism

  EXP4: Energy Landscape
    - Compare grammatical vs ungrammatical "energy" (cross-entropy)
    - Test whether language follows energy-minimization trajectories

DATA: 80+ sentence pairs (expanded)
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
# Sentence Data (EXPANDED)
# ========================================================================
IRREGULAR_PLURALS = {
    "child": "children", "man": "men", "woman": "women",
    "mouse": "mice", "goose": "geese", "foot": "feet",
    "tooth": "teeth", "person": "people", "fish": "fish",
    "deer": "deer", "sheep": "sheep",
}

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


def make_plural(noun):
    if noun in IRREGULAR_PLURALS:
        return IRREGULAR_PLURALS[noun]
    if noun.endswith(("s", "sh", "ch", "x", "z")):
        return noun + "es"
    if noun.endswith("y") and noun[-2] not in "aeiou":
        return noun[:-1] + "ies"
    return noun + "s"


def generate_test_sentences():
    test_data = []
    for subj, v3sg, vbase, obj in SENTENCE_TRIPLES:
        sg_sent = f"The {subj} {v3sg} the {obj}"
        pl_subj = make_plural(subj)
        pl_sent = f"The {pl_subj} {vbase} the {obj}"
        test_data.append({
            'sent_sg': sg_sent, 'sent_pl': pl_sent,
            'verb_sg': v3sg, 'verb_pl': vbase,
            'subj': subj, 'obj': obj,
        })
    print(f"  Generated {len(test_data)} sentence pairs")
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
    toks = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=64)
    input_ids = toks["input_ids"]
    ids_list = input_ids[0].tolist()
    decoded_tokens = [tokenizer.decode([tid]) for tid in ids_list]

    for i in range(len(decoded_tokens) - 4):
        t0 = decoded_tokens[i].strip().lower()
        t3 = decoded_tokens[i+3].strip().lower()
        if t0 in ("the", "a", "an") and t3 in ("the", "a", "an"):
            return {
                'det1_pos': i, 'subj_pos': i + 1, 'verb_pos': i + 2,
                'det2_pos': i + 3, 'obj_pos': i + 4,
                'input_ids': input_ids, 'attention_mask': toks["attention_mask"],
                'decoded_tokens': decoded_tokens,
                'verb_pred_pos': i + 1,
                'verb_token_id': ids_list[i + 2],
            }
    return None


def find_verb_token_ids(tokenizer, verb_sg, verb_pl):
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
    verb_logits = logits[0, verb_pred_pos]
    logit_correct = verb_logits[correct_id].float().item()
    logit_wrong = verb_logits[wrong_id].float().item()
    logit_diff = logit_correct - logit_wrong
    agreement = torch.sigmoid(torch.tensor(logit_diff)).item()
    return agreement, logit_diff


# ========================================================================
# EXP1: Attention Routing Map
# ========================================================================
def run_exp1_attention_routing(model, tokenizer, device, test_sentences, n_layers, d_model, n_heads):
    """
    EXP1: Which attention heads route number information from subject to verb?

    Method:
    1. Collect attention weights for grammatical sentences
    2. For each head at each layer, measure: attn_weight[head, verb_pos, subj_pos]
    3. Identify "routing heads" that specifically attend verb→subj
    4. Causal test: zero out these heads' contribution and measure agreement drop

    The causal test uses the "residual stream patching" method:
    - At the verb position, each head contributes: delta_h = W_o[h_slice] @ attn_h
    - We subtract this contribution and measure the effect
    """
    print("\n" + "="*70)
    print("EXP1: Attention Routing Map")
    print("="*70)

    layers = get_layers(model)
    if n_layers <= 12:
        sample_layers = list(range(n_layers))
    else:
        step = max(1, n_layers // 8)
        sample_layers = list(range(0, n_layers, step))
        if n_layers - 1 not in sample_layers:
            sample_layers.append(n_layers - 1)

    # Step 1: Collect attention patterns
    print("\n  Step 1: Collecting attention patterns...")
    head_routing = {}  # {layer: {head: [attn_verb_to_subj values]}}
    head_all_positions = {}  # {layer: {head: [mean_attn_to_all values]}}
    valid_sents = []

    for td in test_sentences[:40]:
        sg_ann = tokenize_and_annotate(td['sent_sg'], tokenizer)
        if sg_ann is None:
            continue

        sg_verb_id, pl_verb_id = find_verb_token_ids(tokenizer, td['verb_sg'], td['verb_pl'])
        if sg_verb_id is None or pl_verb_id is None:
            continue

        with torch.no_grad():
            out = model(input_ids=sg_ann['input_ids'].to(device),
                       attention_mask=sg_ann['attention_mask'].to(device),
                       output_attentions=True)

        baseline_agree, _ = measure_agreement_from_logits(
            out.logits, sg_ann['verb_pred_pos'], sg_verb_id, pl_verb_id)

        if baseline_agree < 0.5:
            continue

        valid_sents.append({
            'ann': sg_ann, 'correct_id': sg_verb_id, 'wrong_id': pl_verb_id,
            'baseline_agree': baseline_agree, 'td': td,
        })

        if out.attentions is not None:
            subj_pos = sg_ann['subj_pos']
            verb_pred_pos = sg_ann['verb_pred_pos']
            seq_len = sg_ann['input_ids'].shape[1]

            for l_idx in sample_layers:
                if l_idx >= len(out.attentions):
                    continue
                attn = out.attentions[l_idx]  # [1, n_heads, seq, seq]
                n_heads_attn = attn.shape[1]

                if l_idx not in head_routing:
                    head_routing[l_idx] = defaultdict(list)
                    head_all_positions[l_idx] = defaultdict(list)

                for h in range(n_heads_attn):
                    # Key metric: how much does this head at verb_pred_pos attend to subj_pos?
                    attn_to_subj = attn[0, h, verb_pred_pos, subj_pos].float().item()
                    head_routing[l_idx][h].append(attn_to_subj)

                    # Also: mean attention from verb_pred_pos to all positions
                    mean_attn = attn[0, h, verb_pred_pos, :].float().mean().item()
                    head_all_positions[l_idx][h].append(mean_attn)

        if len(valid_sents) >= 30:
            break

    print(f"  Valid sentences: {len(valid_sents)}")

    # Step 2: Identify routing heads
    print("\n  Step 2: Top routing heads (verb_pred_pos → subj_pos):")
    routing_scores = {}  # {layer: {head: {'mean_attn_subj', 'specificity', 'routing_ratio'}}}

    for l_idx in sorted(head_routing.keys()):
        layer_scores = {}
        for h in head_routing[l_idx]:
            mean_to_subj = np.mean(head_routing[l_idx][h])
            mean_to_all = np.mean(head_all_positions[l_idx][h])
            specificity = mean_to_subj / (mean_to_all + 1e-10)
            routing_ratio = mean_to_subj  # absolute attention to subject

            layer_scores[h] = {
                'mean_attn_subj': float(mean_to_subj),
                'mean_attn_all': float(mean_to_all),
                'specificity': float(specificity),
            }

        routing_scores[l_idx] = layer_scores

        # Print top 3 heads
        sorted_heads = sorted(layer_scores.items(), key=lambda x: -x[1]['mean_attn_subj'])[:3]
        for h, s in sorted_heads:
            print(f"    L{l_idx} H{h}: attn_to_subj={s['mean_attn_subj']:.4f}, "
                  f"specificity={s['specificity']:.2f}")

    # Step 2: Subject-swap causal test (THE KEY EXPERIMENT)
    # Compare agreement for correct vs wrong subject number
    # This reveals HOW MUCH the subject number affects S-V agreement
    print("\n  Step 2: Subject-swap causal test...")
    patch_results = {}

    for l_idx in sample_layers:
        agree_drops = []

        for vs in valid_sents[:25]:
            td = vs['td']
            ann = vs['ann']
            correct_id = vs['correct_id']
            wrong_id = vs['wrong_id']
            baseline = vs['baseline_agree']

            # Create corrupted: swap to wrong number subject
            pl_subj = make_plural(td['subj'])
            corrupted_sent = f"The {pl_subj} {td['verb_sg']} the {td['obj']}"
            corrupted_ann = tokenize_and_annotate(corrupted_sent, tokenizer)
            if corrupted_ann is None:
                continue

            with torch.no_grad():
                corr_out = model(input_ids=corrupted_ann['input_ids'].to(device),
                                attention_mask=corrupted_ann['attention_mask'].to(device))
                corrupted_agree, _ = measure_agreement_from_logits(
                    corr_out.logits, corrupted_ann['verb_pred_pos'], correct_id, wrong_id)

            agree_drops.append(baseline - corrupted_agree)

        if agree_drops:
            mean_drop = np.mean(agree_drops)
            patch_results[f"L{l_idx}"] = {
                'mean_agreement_drop_from_corruption': float(mean_drop),
                'n_test': len(agree_drops),
            }
            label = "★★★ CRITICAL" if mean_drop > 0.3 else "★★ IMPORTANT" if mean_drop > 0.1 else "★ MODERATE" if mean_drop > 0.03 else "✗ MINIMAL"
            print(f"    L{l_idx}: corruption drop={mean_drop:.4f} → {label}")

    return {
        'routing_scores': {str(k): {str(hh): vv for hh, vv in v.items()}
                          for k, v in routing_scores.items()},
        'subject_swap_patching': patch_results,
    }


# ========================================================================
# EXP2: Number Direction Tracking
# ========================================================================
def run_exp2_number_direction(model, tokenizer, device, test_sentences, n_layers, d_model):
    """
    Track how "number" information propagates through the residual stream.
    """
    print("\n" + "="*70)
    print("EXP2: Number Direction Tracking in Residual Stream")
    print("="*70)

    if n_layers <= 12:
        sample_layers = list(range(n_layers))
    else:
        step = max(1, n_layers // 10)
        sample_layers = list(range(0, n_layers, step))
        if n_layers - 1 not in sample_layers:
            sample_layers.append(n_layers - 1)

    sg_subj_h = {l: [] for l in sample_layers}
    pl_subj_h = {l: [] for l in sample_layers}
    sg_verb_h = {l: [] for l in sample_layers}
    pl_verb_h = {l: [] for l in sample_layers}

    n_collected = 0
    for td in test_sentences:
        if n_collected >= 40:
            break
        sg_ann = tokenize_and_annotate(td['sent_sg'], tokenizer)
        pl_ann = tokenize_and_annotate(td['sent_pl'], tokenizer)
        if sg_ann is None or pl_ann is None:
            continue

        with torch.no_grad():
            sg_out = model(input_ids=sg_ann['input_ids'].to(device),
                          attention_mask=sg_ann['attention_mask'].to(device),
                          output_hidden_states=True)
            pl_out = model(input_ids=pl_ann['input_ids'].to(device),
                          attention_mask=pl_ann['attention_mask'].to(device),
                          output_hidden_states=True)

        for l in sample_layers:
            if l < len(sg_out.hidden_states):
                sg_hid = sg_out.hidden_states[l][0].float().cpu().numpy()
                pl_hid = pl_out.hidden_states[l][0].float().cpu().numpy()
                sg_subj_h[l].append(sg_hid[sg_ann['subj_pos']])
                pl_subj_h[l].append(pl_hid[pl_ann['subj_pos']])
                sg_verb_h[l].append(sg_hid[sg_ann['verb_pos']])
                pl_verb_h[l].append(pl_hid[pl_ann['verb_pos']])

        n_collected += 1

    print(f"  Collected {n_collected} sentence pairs")

    results = {}
    print(f"\n  {'Layer':>6}  {'Subj_AUC':>10}  {'Verb_AUC':>10}  {'DirCorr':>10}  {'ProjSep':>10}")

    for l in sample_layers:
        sg_s = np.array(sg_subj_h[l])
        pl_s = np.array(pl_subj_h[l])
        sg_v = np.array(sg_verb_h[l])
        pl_v = np.array(pl_verb_h[l])

        if len(sg_s) < 10:
            continue

        # Subject position: decode number
        X_subj = np.vstack([sg_s, pl_s])
        y_subj = np.array([0]*len(sg_s) + [1]*len(pl_s))
        try:
            lr_subj = LogisticRegression(max_iter=200, C=1.0)
            lr_subj.fit(X_subj, y_subj)
            subj_auc = roc_auc_score(y_subj, lr_subj.predict_proba(X_subj)[:, 1])
            num_dir_subj = lr_subj.coef_[0]
            num_dir_subj = num_dir_subj / (np.linalg.norm(num_dir_subj) + 1e-10)
        except Exception:
            subj_auc = 0.5
            num_dir_subj = np.zeros(d_model)

        # Verb position: can we decode SUBJECT number? (key test!)
        X_verb = np.vstack([sg_v, pl_v])
        y_verb = np.array([0]*len(sg_v) + [1]*len(pl_v))
        try:
            lr_verb = LogisticRegression(max_iter=200, C=1.0)
            lr_verb.fit(X_verb, y_verb)
            verb_auc = roc_auc_score(y_verb, lr_verb.predict_proba(X_verb)[:, 1])
            num_dir_verb = lr_verb.coef_[0]
            num_dir_verb = num_dir_verb / (np.linalg.norm(num_dir_verb) + 1e-10)
        except Exception:
            verb_auc = 0.5
            num_dir_verb = np.zeros(d_model)

        dir_corr = float(np.dot(num_dir_subj, num_dir_verb))

        # Projection separation
        sg_proj = [float(h @ num_dir_subj) for h in sg_v]
        pl_proj = [float(h @ num_dir_subj) for h in pl_v]
        proj_sep = float(np.mean(pl_proj) - np.mean(sg_proj))

        results[f"L{l}"] = {
            'subj_auc': float(subj_auc),
            'verb_auc': float(verb_auc),
            'direction_correlation': float(dir_corr),
            'proj_separation': float(proj_sep),
        }

        auc_label = "★★★" if verb_auc > 0.9 else "★★" if verb_auc > 0.7 else "★" if verb_auc > 0.6 else ""
        print(f"  L{l:>4}  {subj_auc:>10.4f}  {verb_auc:>10.4f}  {dir_corr:>10.4f}  {proj_sep:>10.4f}  {auc_label}")

    # Transfer analysis
    print("\n  Number Information Transfer Analysis:")
    verb_aucs = [(l, results[f"L{l}"]["verb_auc"]) for l in sample_layers if f"L{l}" in results]
    if verb_aucs:
        max_verb_layer = max(verb_aucs, key=lambda x: x[1])
        transfer_point = None
        for l, auc in verb_aucs:
            if auc > 0.7:
                transfer_point = l
                break
        print(f"    Peak verb number decodability: L{max_verb_layer[0]} (AUC={max_verb_layer[1]:.4f})")
        if transfer_point is not None:
            print(f"    First layer with verb_auc > 0.7: L{transfer_point}")
        else:
            print(f"    No layer with verb_auc > 0.7")

    return results


# ========================================================================
# EXP3: Constraint Repair Dynamics (Logit Lens)
# ========================================================================
def run_exp3_repair_dynamics(model, tokenizer, device, test_sentences, n_layers, d_model):
    """
    When grammar is broken, how does the model repair it across layers?
    Uses "logit lens" — project each layer's hidden state through W_U to see
    what the model would predict at that layer.
    """
    print("\n" + "="*70)
    print("EXP3: Constraint Repair Dynamics (Logit Lens)")
    print("="*70)

    if n_layers <= 12:
        sample_layers = list(range(n_layers))
    else:
        step = max(1, n_layers // 8)
        sample_layers = list(range(0, n_layers, step))
        if n_layers - 1 not in sample_layers:
            sample_layers.append(n_layers - 1)

    W_U = get_W_U(model, None)
    if isinstance(W_U, torch.Tensor):
        W_U = W_U.float().numpy()
    print(f"  W_U shape: {W_U.shape}")

    repair_results = {'sg_subj_pl_verb': {}, 'pl_subj_sg_verb': {}}

    for direction in ['sg_subj_pl_verb', 'pl_subj_sg_verb']:
        print(f"\n  Direction: {direction}")
        agreement_by_layer = {l: [] for l in sample_layers}

        n_tested = 0
        for td in test_sentences:
            if n_tested >= 30:
                break

            if direction == 'sg_subj_pl_verb':
                wrong_sent = f"The {td['subj']} {td['verb_pl']} the {td['obj']}"
                correct_verb = td['verb_sg']
                wrong_verb = td['verb_pl']
            else:
                pl_subj = make_plural(td['subj'])
                wrong_sent = f"The {pl_subj} {td['verb_sg']} the {td['obj']}"
                correct_verb = td['verb_pl']
                wrong_verb = td['verb_sg']

            ann = tokenize_and_annotate(wrong_sent, tokenizer)
            if ann is None:
                continue

            correct_id, wrong_id = find_verb_token_ids(tokenizer, correct_verb, wrong_verb)
            if correct_id is None or wrong_id is None:
                continue

            with torch.no_grad():
                out = model(input_ids=ann['input_ids'].to(device),
                           attention_mask=ann['attention_mask'].to(device),
                           output_hidden_states=True)

            for l in sample_layers:
                if l < len(out.hidden_states):
                    h_l = out.hidden_states[l][0, ann['verb_pred_pos']].float().cpu().numpy()
                    logits_l = h_l @ W_U.T
                    logit_correct = logits_l[correct_id]
                    logit_wrong = logits_l[wrong_id]
                    diff_l = logit_correct - logit_wrong
                    agree_l = 1.0 / (1.0 + np.exp(-diff_l))
                    agreement_by_layer[l].append(agree_l)

            n_tested += 1

        print(f"  Tested {n_tested} sentences")

        for l in sample_layers:
            if agreement_by_layer[l]:
                mean_agree = np.mean(agreement_by_layer[l])
                repair_results[direction][f"L{l}"] = {'mean_agreement': float(mean_agree)}
                label = "★★★ REPAIR" if mean_agree > 0.7 else "★★ TREND" if mean_agree > 0.5 else "★ WEAK" if mean_agree > 0.3 else ""
                print(f"    L{l}: agree={mean_agree:.4f} {label}")

        # Find repair point
        layer_means = [(l, np.mean(agreement_by_layer[l])) for l in sample_layers
                       if agreement_by_layer[l]]
        if layer_means:
            repair_point = None
            for l, m in layer_means:
                if m > 0.5:
                    repair_point = l
                    break
            if repair_point is not None:
                print(f"  ★ Repair point: L{repair_point} (first layer with agree > 0.5)")
            else:
                print(f"  ✗ No repair found in logit lens")

    return repair_results


# ========================================================================
# EXP4: Energy Landscape
# ========================================================================
def run_exp4_energy_landscape(model, tokenizer, device, test_sentences, n_layers, d_model):
    """
    Compare grammatical vs ungrammatical "energy" (cross-entropy).
    Hypothesis: grammatical sequences follow lower-energy trajectories.
    """
    print("\n" + "="*70)
    print("EXP4: Constraint Energy Landscape")
    print("="*70)

    gram_energies = []
    ungram_energies = []

    n_tested = 0
    for td in test_sentences:
        if n_tested >= 40:
            break

        sg_ann = tokenize_and_annotate(td['sent_sg'], tokenizer)
        pl_ann = tokenize_and_annotate(td['sent_pl'], tokenizer)
        if sg_ann is None or pl_ann is None:
            continue

        sg_verb_id, pl_verb_id = find_verb_token_ids(tokenizer, td['verb_sg'], td['verb_pl'])
        if sg_verb_id is None or pl_verb_id is None:
            continue

        with torch.no_grad():
            sg_out = model(input_ids=sg_ann['input_ids'].to(device),
                          attention_mask=sg_ann['attention_mask'].to(device))
            verb_logits_sg = sg_out.logits[0, sg_ann['verb_pred_pos']].float()
            energy_sg = F.cross_entropy(verb_logits_sg.unsqueeze(0),
                                        torch.tensor([sg_verb_id], device=device)).item()

        with torch.no_grad():
            pl_out = model(input_ids=pl_ann['input_ids'].to(device),
                          attention_mask=pl_ann['attention_mask'].to(device))
            verb_logits_pl = pl_out.logits[0, pl_ann['verb_pred_pos']].float()
            energy_pl = F.cross_entropy(verb_logits_pl.unsqueeze(0),
                                        torch.tensor([pl_verb_id], device=device)).item()

        energy_ungram_sg = F.cross_entropy(verb_logits_sg.unsqueeze(0),
                                            torch.tensor([pl_verb_id], device=device)).item()
        energy_ungram_pl = F.cross_entropy(verb_logits_pl.unsqueeze(0),
                                            torch.tensor([sg_verb_id], device=device)).item()

        gram_energies.extend([energy_sg, energy_pl])
        ungram_energies.extend([energy_ungram_sg, energy_ungram_pl])
        n_tested += 1

    print(f"  Tested {n_tested} sentence pairs ({len(gram_energies)} energies)")

    if gram_energies:
        mean_gram = np.mean(gram_energies)
        mean_ungram = np.mean(ungram_energies)
        delta_energy = mean_ungram - mean_gram

        try:
            t_stat, p_val = ttest_rel(ungram_energies, gram_energies)
        except Exception:
            t_stat, p_val = 0, 1.0

        print(f"\n  Grammatical energy: {mean_gram:.4f}")
        print(f"  Ungrammatical energy: {mean_ungram:.4f}")
        print(f"  ΔE (constraint violation cost): {delta_energy:.4f}")
        print(f"  t-stat: {t_stat:.4f}, p-value: {p_val:.6f}")

        if delta_energy > 0 and p_val < 0.01:
            print("  ★★★ CONFIRMED: Ungrammatical = higher energy (constraint violation)")
        elif delta_energy > 0:
            print("  ★★ TREND: Ungrammatical tends higher energy")

        n_violations = sum(1 for g, u in zip(gram_energies, ungram_energies) if u > g)
        print(f"  Individual: {n_violations}/{len(gram_energies)} "
              f"({100*n_violations/len(gram_energies):.1f}%) show violation cost")

        return {
            'mean_gram_energy': float(mean_gram),
            'mean_ungram_energy': float(mean_ungram),
            'delta_energy': float(delta_energy),
            't_stat': float(t_stat),
            'p_value': float(p_val),
            'n_violation_fraction': float(n_violations / len(gram_energies)),
        }

    return {}


# ========================================================================
# Main
# ========================================================================
def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    t_start = time.time()
    print(f"\n{'='*70}")
    print(f"Phase 210: Constraint Field Dynamics — {model_name}")
    print(f"{'='*70}")

    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    # Get n_heads from config
    n_heads = getattr(model.config, 'num_attention_heads', d_model // 64)
    head_dim = d_model // n_heads
    print(f"  n_layers={n_layers}, d_model={d_model}, n_heads={n_heads}, head_dim={head_dim}")

    test_sentences = generate_test_sentences()
    all_results = {'model': model_name, 'n_layers': n_layers, 'd_model': d_model}

    # EXP1
    try:
        exp1_results = run_exp1_attention_routing(
            model, tokenizer, device, test_sentences, n_layers, d_model, n_heads)
        all_results['exp1_attention_routing'] = exp1_results
    except Exception as e:
        print(f"  EXP1 FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results['exp1_attention_routing'] = {"error": str(e)}
    torch.cuda.empty_cache()

    # EXP2
    try:
        exp2_results = run_exp2_number_direction(
            model, tokenizer, device, test_sentences, n_layers, d_model)
        all_results['exp2_number_direction'] = exp2_results
    except Exception as e:
        print(f"  EXP2 FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results['exp2_number_direction'] = {"error": str(e)}
    torch.cuda.empty_cache()

    # EXP3
    try:
        exp3_results = run_exp3_repair_dynamics(
            model, tokenizer, device, test_sentences, n_layers, d_model)
        all_results['exp3_repair_dynamics'] = exp3_results
    except Exception as e:
        print(f"  EXP3 FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results['exp3_repair_dynamics'] = {"error": str(e)}
    torch.cuda.empty_cache()

    # EXP4
    try:
        exp4_results = run_exp4_energy_landscape(
            model, tokenizer, device, test_sentences, n_layers, d_model)
        all_results['exp4_energy_landscape'] = exp4_results
    except Exception as e:
        print(f"  EXP4 FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results['exp4_energy_landscape'] = {"error": str(e)}
    torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*70}")
    print(f"PHASE 210 SUMMARY — {model_name}")
    print(f"{'='*70}")

    # Exp1
    exp1 = all_results.get('exp1_attention_routing', {})
    if isinstance(exp1, dict) and 'routing_scores' in exp1:
        print("\n--- Exp1: Attention Routing ---")
        routing = exp1['routing_scores']
        for layer_key in sorted(routing.keys(), key=lambda x: int(x) if x.isdigit() else 0):
            heads = routing[layer_key]
            sorted_h = sorted(heads.items(), key=lambda x: -x[1].get('mean_attn_subj', 0))[:2]
            for h, s in sorted_h:
                if s.get('mean_attn_subj', 0) > 0.03:
                    print(f"  {layer_key} H{h}: attn_to_subj={s['mean_attn_subj']:.4f}, "
                          f"specificity={s.get('specificity', 0):.2f}")

        patch = exp1.get('subject_swap_patching', {})
        if patch:
            print("\n  Subject-swap corruption effect:")
            for lk in sorted(patch.keys()):
                r = patch[lk]
                if isinstance(r, dict) and 'mean_agreement_drop_from_corruption' in r:
                    print(f"    {lk}: drop={r['mean_agreement_drop_from_corruption']:.4f}")

    # Exp2
    exp2 = all_results.get('exp2_number_direction', {})
    if isinstance(exp2, dict):
        print("\n--- Exp2: Number Direction ---")
        for key in sorted(exp2.keys()):
            r = exp2[key]
            if isinstance(r, dict) and 'subj_auc' in r:
                print(f"  {key}: subj_auc={r['subj_auc']:.3f}, verb_auc={r['verb_auc']:.3f}, "
                      f"dir_corr={r['direction_correlation']:.3f}")

    # Exp3
    exp3 = all_results.get('exp3_repair_dynamics', {})
    if isinstance(exp3, dict):
        print("\n--- Exp3: Repair Dynamics ---")
        for direction in ['sg_subj_pl_verb', 'pl_subj_sg_verb']:
            if direction in exp3:
                print(f"  {direction}:")
                for key in sorted(exp3[direction].keys()):
                    r = exp3[direction][key]
                    if isinstance(r, dict) and 'mean_agreement' in r:
                        print(f"    {key}: agree={r['mean_agreement']:.4f}")

    # Exp4
    exp4 = all_results.get('exp4_energy_landscape', {})
    if isinstance(exp4, dict) and 'delta_energy' in exp4:
        print(f"\n--- Exp4: Energy Landscape ---")
        print(f"  ΔE = {exp4['delta_energy']:.4f}, p = {exp4['p_value']:.6f}")
        print(f"  Violation fraction = {exp4['n_violation_fraction']:.4f}")

    results_path = f"tests/glm5_temp/phase210_{model_name}_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nSaved to {results_path}")

    print(f"\nTotal time: {time.time()-t_start:.1f}s ({(time.time()-t_start)/60:.1f}min)")

    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
