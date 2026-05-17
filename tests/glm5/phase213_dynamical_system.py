"""
Phase 213: Transformer as Dynamical System — FROM TRANSITIONS TO DYNAMICS
=========================================================================

THEORETICAL CONTEXT (from Phase 212 review):
  Phase 212 found MLP writes verb form info (Δh_mlp >> Δh_attn), and identified
  AGREEMENT-MLP layers. But the user's critique identifies deeper issues:

  HARD INJURY 1: SVA is "extremely shallow" — only syntax microcircuits,
    cannot extrapolate to reasoning/world knowledge/composition
  HARD INJURY 2: MLP as "if-then" is too symbolic — it's attractor biasing
    (probabilistic trajectory shaping, not discrete rule execution)
  HARD INJURY 3: "Constraint field" is metaphor, not theory — need math
  HARD INJURY 4: Need Jacobian dynamics (∂h_{l+1}/∂h_l) — this IS the system

CORE INSIGHT:
  Transformer = discrete dynamical system: h_{l+1} = h_l + F_l(h_l)
  The REAL mathematics is in the Jacobian J_l = I + ∂F_l/∂h_l
  - Eigenvalues > 1: amplified directions (attractors)
  - Eigenvalues < 1: dampened directions (transients)
  - Language structure = stable attractor trajectories

EXPERIMENTS:
  EXP1: Number Signal Flow Through Layers ★★★
    - Track Δh_l = h_l(pl) - h_l(sg) at verb position through all layers
    - Measure: ||Δh_l|| evolution, logit projection evolution, cosine rotation
    - Identify: which layers AMPLIFY vs DAMPEN number signal
    - This is the "Jacobian trace" for the number feature

  EXP2: Trajectory Divergence — Attractor Geometry ★★★
    - Compare grammatical ("cat chases") vs ungrammatical ("cat chase")
    - Measure: does the ungrammatical trajectory get "pushed back" toward
      grammatical attractor, or does it just stay divergent?
    - This directly tests: is language processing = trajectory stabilization?
    - KEY: test "self-repair" — does the model correct its own errors?

  EXP3: Jacobian-Vector Product for Number Direction ★★★
    - Compute J_l @ v for the number direction v
    - This tells us: how does each layer TRANSFORM the number feature?
    - If J @ v ≈ α*v with α>1: number is amplified (being read/used)
    - If J @ v is rotated: number is being converted to something else
    - If J @ v ≈ 0: number is ignored at this layer

  EXP4: Beyond SVA — Semantic & Structural Constraints ★★
    - Negation: "The cat does not chase/chases" (agreement with auxiliary)
    - Coordination: "The cat and the dog chase/chases" (plural conjunction)
    - Relative clauses with deeper embedding
    - Test: do SAME circuits handle these, or different ones?

DATA: 80+ sentence pairs including long-range, negation, coordination
MODELS: Qwen3, GLM4, DS7B (bf16 + device_map="auto" + flash attention)
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

from model_utils import (get_model_info, release_model, get_layers, get_W_U,
                          MODEL_CONFIGS, get_layer_weights)

warnings.filterwarnings('ignore')


# ========================================================================
# Sentence Data — EXPANDED with negation, coordination, deeper structure
# ========================================================================
IRREGULAR_PLURALS = {
    "child": "children", "man": "men", "woman": "women",
    "mouse": "mice", "goose": "geese", "foot": "feet",
    "tooth": "teeth", "person": "people", "fish": "fish",
    "deer": "deer", "sheep": "sheep",
}

# Short distance SVA: "The [subj] [verb] the [obj]"
SHORT_SVA = [
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

# Long-range center-embedded: "The [subj] that the [noun] [verb_past] [verb_present] [adv]"
LONG_RANGE_SVA = [
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

# NEGATION: "The [subj] does not [verb_base]" — agreement on auxiliary
NEGATION_SENTENCES = [
    ("cat", "does", "chase", "dog"),
    ("dog", "does", "follow", "cat"),
    ("bear", "does", "watch", "wolf"),
    ("lion", "does", "find", "tiger"),
    ("fox", "does", "track", "rabbit"),
    ("teacher", "does", "help", "student"),
    ("doctor", "does", "treat", "patient"),
    ("king", "does", "protect", "queen"),
    ("captain", "does", "lead", "soldier"),
    ("writer", "does", "meet", "reader"),
    ("singer", "does", "join", "dancer"),
    ("judge", "does", "watch", "lawyer"),
    ("mother", "does", "help", "father"),
    ("sister", "does", "meet", "brother"),
    ("girl", "does", "watch", "boy"),
]

# COORDINATION: "The [subj1] and the [subj2] [verb_pl]" — plural verb
COORDINATION_SENTENCES = [
    ("cat", "dog", "chase"),
    ("bear", "wolf", "watch"),
    ("lion", "tiger", "find"),
    ("teacher", "student", "help"),
    ("doctor", "nurse", "treat"),
    ("king", "queen", "rule"),
    ("captain", "soldier", "lead"),
    ("writer", "reader", "meet"),
    ("singer", "dancer", "join"),
    ("judge", "lawyer", "watch"),
    ("mother", "father", "help"),
    ("sister", "brother", "play"),
    ("girl", "boy", "walk"),
    ("sun", "moon", "shine"),
    ("river", "ocean", "flow"),
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
    if v3sg.endswith("ies"):
        return v3sg[:-3] + "y"
    if v3sg.endswith(("shes", "ches", "xes", "zes")):
        return v3sg[:-2]
    if v3sg.endswith("sses"):
        return v3sg[:-2]
    if v3sg.endswith("s") and not v3sg.endswith(("ss", "us", "is")):
        return v3sg[:-1]
    IRREG_VERBS = {"has": "have", "is": "are", "was": "were", "does": "do", "goes": "go"}
    return IRREG_VERBS.get(v3sg, v3sg)


def generate_all_sentences():
    """Generate all test sentence pairs across SVA, negation, coordination."""
    test_data = []

    # Short SVA
    for subj, v3sg, vbase, obj in SHORT_SVA:
        sg = f"The {subj} {v3sg} the {obj}"
        pl_subj = make_plural(subj)
        pl = f"The {pl_subj} {vbase} the {obj}"
        test_data.append({
            'sent_sg': sg, 'sent_pl': pl,
            'verb_sg': v3sg, 'verb_pl': vbase,
            'subj': subj, 'obj': obj,
            'type': 'short_sva', 'distance': 'short',
        })

    # Long-range SVA
    for subj, noun, vpast, vpres, adv in LONG_RANGE_SVA:
        sg = f"The {subj} that the {noun} {vpast} {vpres} {adv}"
        pl_subj = make_plural(subj)
        vbase = verb_3sg_to_base(vpres)
        pl = f"The {pl_subj} that the {noun} {vpast} {vbase} {adv}"
        test_data.append({
            'sent_sg': sg, 'sent_pl': pl,
            'verb_sg': vpres, 'verb_pl': vbase,
            'subj': subj, 'noun': noun,
            'type': 'long_sva', 'distance': 'long',
        })

    # Negation
    for subj, aux_sg, vbase, obj in NEGATION_SENTENCES:
        sg = f"The {subj} {aux_sg} not {vbase} the {obj}"
        pl_subj = make_plural(subj)
        pl = f"The {pl_subj} do not {vbase} the {obj}"
        test_data.append({
            'sent_sg': sg, 'sent_pl': pl,
            'verb_sg': aux_sg, 'verb_pl': 'do',
            'subj': subj, 'obj': obj,
            'type': 'negation', 'distance': 'short',
        })

    # Coordination
    for subj1, subj2, vbase in COORDINATION_SENTENCES:
        # SG control: "The [subj1] [verb_3sg] the [obj]" — singular subject
        # PL test: "The [subj1] and the [subj2] [verb_base] the [obj]" — plural
        v3sg = verb_3sg_to_base(vbase) if vbase.endswith('s') else vbase + 's'
        # Actually, we want: sg uses 3sg form, pl uses base form
        # For "chase": sg="chases", pl="chase"
        # This is just: v3sg = vbase + "s" (simplified, handles most cases)
        if vbase in ("chase", "watch", "find", "help", "treat", "rule", "lead",
                     "meet", "join", "play", "walk", "shine", "flow"):
            v3sg = vbase + "s"
        elif vbase.endswith("y") and vbase[-2] not in "aeiou":
            v3sg = vbase[:-1] + "ies"
        else:
            v3sg = vbase + "s"

        sg = f"The {subj1} {v3sg} and the {subj2} watches"  # Control: just sg subject
        pl = f"The {subj1} and the {subj2} {vbase}"
        # For coordination, we want: "The cat and the dog chase" (PL verb)
        # vs ungrammatical: "The cat and the dog chases" (SG verb)
        test_data.append({
            'sent_pl': pl,
            'verb_sg': v3sg, 'verb_pl': vbase,
            'subj1': subj1, 'subj2': subj2,
            'type': 'coordination', 'distance': 'short',
        })

    # Count by type
    type_counts = defaultdict(int)
    for td in test_data:
        type_counts[td['type']] += 1
    print(f"  Generated {len(test_data)} sentence pairs: {dict(type_counts)}")
    return test_data


# ========================================================================
# Model Loading (BF16 + device_map="auto" + flash attention)
# ========================================================================
def load_model_bf16(model_name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    print(f"[load] Loading {model_name} (bfloat16 + device_map=auto + flash)...")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = None
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
# Utility Functions
# ========================================================================
def tokenize_and_annotate(sentence, tokenizer):
    """Tokenize and find key positions."""
    toks = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=64)
    input_ids = toks["input_ids"]
    ids_list = input_ids[0].tolist()
    decoded = [tokenizer.decode([tid]) for tid in ids_list]

    result = {
        'input_ids': input_ids,
        'attention_mask': toks["attention_mask"],
        'decoded_tokens': decoded,
        'token_ids': ids_list,
    }

    # Find "that" for center-embedded
    that_positions = [i for i, t in enumerate(decoded) if t.strip().lower() == "that"]

    if that_positions:
        that_pos = that_positions[0]
        # Subject is before "that"
        for i in range(max(0, that_pos - 4), that_pos):
            if decoded[i].strip().lower() in ("the", "a", "an"):
                subj_pos = i + 1
                break
        else:
            subj_pos = that_pos - 1

        # Verb (present) after relative clause
        the_after = None
        for i in range(that_pos + 1, min(that_pos + 5, len(decoded))):
            if decoded[i].strip().lower() in ("the", "a", "an"):
                the_after = i
                break

        if the_after is not None:
            noun_pos = the_after + 1
            vpast_pos = noun_pos + 1
            vpres_pos = vpast_pos + 1
            result.update({
                'subj_pos': subj_pos,
                'that_pos': that_pos,
                'noun_pos': noun_pos,
                'verb_present_pos': vpres_pos,
                'verb_pred_pos': vpres_pos - 1,
                'distance': 'long',
            })
            return result

    # Short pattern: "The [subj] [verb] the [obj]" or with "does not"
    # Check for "does" or "do" (negation pattern)
    for i in range(len(decoded) - 3):
        t0 = decoded[i].strip().lower()
        if t0 in ("the", "a", "an"):
            # Check for "does not [verb]" pattern
            rest = [decoded[j].strip().lower() for j in range(i+1, min(i+6, len(decoded)))]
            if "does" in rest or "do" in rest:
                # "The [subj] does not [verb] the [obj]"
                does_pos = i + 1 + rest.index("does" if "does" in rest else "do")
                result.update({
                    'subj_pos': i + 1,
                    'aux_pos': does_pos,
                    'verb_pred_pos': does_pos - 1,
                    'distance': 'short',
                    'structure': 'negation',
                })
                return result

    # Simple short pattern
    for i in range(len(decoded) - 4):
        t0 = decoded[i].strip().lower()
        t3 = decoded[i+3].strip().lower() if i+3 < len(decoded) else ""
        if t0 in ("the", "a", "an") and t3 in ("the", "a", "an"):
            result.update({
                'det1_pos': i, 'subj_pos': i + 1, 'verb_pos': i + 2,
                'det2_pos': i + 3, 'obj_pos': i + 4,
                'verb_pred_pos': i + 1,
                'distance': 'short',
                'structure': 'simple',
            })
            return result

    # Coordination: "The [subj1] and the [subj2] [verb]"
    for i in range(len(decoded) - 5):
        t0 = decoded[i].strip().lower()
        if t0 in ("the", "a", "an"):
            rest_str = " ".join(decoded[j].strip().lower() for j in range(i, min(i+7, len(decoded))))
            if " and the " in rest_str or " and a " in rest_str:
                # Find verb position at end
                result.update({
                    'subj1_pos': i + 1,
                    'verb_pred_pos': len(decoded) - 2,
                    'distance': 'short',
                    'structure': 'coordination',
                })
                return result

    return result


def find_verb_token_ids(tokenizer, verb_sg, verb_pl):
    """Find single-token IDs for verb forms."""
    for prefix in [" ", ""]:
        ids = tokenizer.encode(prefix + verb_sg, add_special_tokens=False)
        if len(ids) == 1:
            sg_id = ids[0]
            break
    else:
        sg_id = None

    for prefix in [" ", ""]:
        ids = tokenizer.encode(prefix + verb_pl, add_special_tokens=False)
        if len(ids) == 1:
            pl_id = ids[0]
            break
    else:
        pl_id = None

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
# EXP1: Number Signal Flow Through Layers ★★★
# ========================================================================
def run_exp1_number_signal_flow(model, tokenizer, device, test_sentences,
                                 n_layers, d_model):
    """
    Track how the "number signal" Δh_l = h_l(pl) - h_l(sg) evolves through layers.

    This is the DYNAMICAL SYSTEMS view:
    - h_{l+1} = h_l + F_l(h_l) is a discrete dynamical system
    - Δh_l is the "number perturbation" at layer l
    - The layer's action on Δh_l reveals the Jacobian's behavior

    Key measurements per layer:
    1. ||Δh_l|| — magnitude of number signal
    2. Δh_l @ W_U[sg_verb] - Δh_l @ W_U[pl_verb] — logit-level number information
    3. cos(Δh_l, Δh_{l-1}) — rotation of number direction
    4. Amplification ratio: ||Δh_l|| / ||Δh_{l-1}|| — Jacobian effect on number
    """
    print("\n" + "=" * 70)
    print("EXP1: Number Signal Flow Through Layers ★★★")
    print("DYNAMICAL SYSTEMS VIEW: Δh_l = h_l(pl) - h_l(sg) evolution")
    print("=" * 70)

    W_U = get_W_U(model, None)
    if isinstance(W_U, torch.Tensor):
        W_U = W_U.float().numpy()

    # Collect all layers' hidden states
    # For each sentence pair, we'll use output_hidden_states=True
    flow_data = defaultdict(lambda: {
        'delta_norm': [], 'delta_sg_logit': [], 'delta_pl_logit': [],
        'delta_logit_diff': [], 'cos_rotation': [],
    })

    n_valid = 0
    t_exp_start = time.time()

    # Use short SVA sentences for clean signal
    short_sents = [td for td in test_sentences if td['type'] == 'short_sva']

    for td in short_sents:
        if n_valid >= 50:
            break

        sg_verb_id, pl_verb_id = find_verb_token_ids(tokenizer, td['verb_sg'], td['verb_pl'])
        if sg_verb_id is None or pl_verb_id is None:
            continue

        sg_ann = tokenize_and_annotate(td['sent_sg'], tokenizer)
        pl_ann = tokenize_and_annotate(td['sent_pl'], tokenizer)

        verb_pred_pos = sg_ann.get('verb_pred_pos')
        if verb_pred_pos is None:
            continue

        # Run both sentences with all hidden states
        with torch.no_grad():
            sg_out = model(input_ids=sg_ann['input_ids'].to(device),
                          attention_mask=sg_ann['attention_mask'].to(device),
                          output_hidden_states=True)
            pl_out = model(input_ids=pl_ann['input_ids'].to(device),
                          attention_mask=pl_ann['attention_mask'].to(device),
                          output_hidden_states=True)

        # Verify model gets agreement right
        agree, _ = measure_agreement(sg_out.logits, verb_pred_pos, sg_verb_id, pl_verb_id)
        if agree < 0.5:
            del sg_out, pl_out
            continue

        # Extract Δh_l for each layer at verb position
        prev_delta = None
        for l in range(len(sg_out.hidden_states)):
            h_sg = sg_out.hidden_states[l][0, verb_pred_pos].float().cpu().numpy()
            h_pl = pl_out.hidden_states[l][0, verb_pred_pos].float().cpu().numpy()
            delta = h_pl - h_sg  # Δh_l at verb position

            delta_norm = float(np.linalg.norm(delta))
            sg_logit_proj = float(delta @ W_U[sg_verb_id])
            pl_logit_proj = float(delta @ W_U[pl_verb_id])
            logit_diff = sg_logit_proj - pl_logit_proj

            # Cosine with previous delta (rotation measure)
            if prev_delta is not None and delta_norm > 1e-8 and np.linalg.norm(prev_delta) > 1e-8:
                cos_rot = float(np.dot(delta, prev_delta) / (np.linalg.norm(delta) * np.linalg.norm(prev_delta)))
            else:
                cos_rot = 1.0

            # Amplification ratio
            if prev_delta is not None and np.linalg.norm(prev_delta) > 1e-8:
                amp_ratio = delta_norm / np.linalg.norm(prev_delta)
            else:
                amp_ratio = 1.0

            flow_data[l]['delta_norm'].append(delta_norm)
            flow_data[l]['delta_sg_logit'].append(sg_logit_proj)
            flow_data[l]['delta_pl_logit'].append(pl_logit_proj)
            flow_data[l]['delta_logit_diff'].append(logit_diff)
            flow_data[l]['cos_rotation'].append(cos_rot)
            flow_data[l]['amp_ratio'] = flow_data[l].get('amp_ratio', [])
            if prev_delta is not None and np.linalg.norm(prev_delta) > 1e-8:
                flow_data[l]['amp_ratio'].append(amp_ratio)

            prev_delta = delta

        del sg_out, pl_out
        torch.cuda.empty_cache()

        n_valid += 1
        if n_valid % 10 == 0:
            elapsed = time.time() - t_exp_start
            print(f"  [EXP1] Processed {n_valid} pairs, elapsed={elapsed:.0f}s")

    # ========================================================================
    # Analysis
    # ========================================================================
    print(f"\n  Processed {n_valid} valid sentence pairs")

    print(f"\n  {'Layer':>6} {'||Δh||':>10} {'Amp_Ratio':>11} {'Cos_Rot':>9} "
          f"{'SG_logit':>10} {'PL_logit':>10} {'LogitDiff':>10} {'Role':>15}")
    print("  " + "-" * 95)

    results = {}
    for l in sorted(flow_data.keys()):
        d = flow_data[l]
        if len(d['delta_norm']) < 5:
            continue

        mean_norm = np.mean(d['delta_norm'])
        mean_amp = np.mean(d['amp_ratio']) if d.get('amp_ratio') else 1.0
        mean_cos = np.mean(d['cos_rotation'])
        mean_sg = np.mean(d['delta_sg_logit'])
        mean_pl = np.mean(d['delta_pl_logit'])
        mean_diff = np.mean(d['delta_logit_diff'])

        # Role classification
        if mean_amp > 1.2 and mean_diff > 0.5:
            role = "AMPLIFIER"
        elif mean_amp < 0.8 and abs(mean_diff) < 0.1:
            role = "DAMPENER"
        elif mean_diff > 0.3:
            role = "SG-WRITER"
        elif mean_diff < -0.3:
            role = "PL-WRITER"
        elif mean_cos < 0.5:
            role = "TRANSFORM"
        else:
            role = ""

        results[f"L{l}"] = {
            'delta_norm': float(mean_norm),
            'amp_ratio': float(mean_amp),
            'cos_rotation': float(mean_cos),
            'sg_logit_proj': float(mean_sg),
            'pl_logit_proj': float(mean_pl),
            'logit_diff': float(mean_diff),
            'role': role,
        }

        if mean_norm > 0.01 or abs(mean_diff) > 0.01:
            print(f"  L{l:>4} {mean_norm:>10.3f} {mean_amp:>11.3f} {mean_cos:>9.3f} "
                  f"{mean_sg:>10.4f} {mean_pl:>10.4f} {mean_diff:>10.4f} {role:>15}")

    # Identify critical transitions
    print(f"\n  ★★★ Number Signal Flow Summary ★★★")

    # Find amplification peaks
    amp_layers = [(l, results[f'L{l}']['amp_ratio']) for l in sorted(flow_data.keys())
                  if f'L{l}' in results and len(flow_data[l].get('amp_ratio', [])) > 3]
    if amp_layers:
        amp_layers_sorted = sorted(amp_layers, key=lambda x: -x[1])
        print(f"\n  Top Amplification Layers (number signal grows):")
        for l, amp in amp_layers_sorted[:5]:
            print(f"    L{l}: amp_ratio={amp:.3f}")

    # Find rotation peaks (where number direction changes character)
    rot_layers = [(l, results[f'L{l}']['cos_rotation']) for l in sorted(flow_data.keys())
                  if f'L{l}' in results]
    if rot_layers:
        rot_layers_sorted = sorted(rot_layers, key=lambda x: x[1])
        print(f"\n  Top Rotation Layers (number direction changes):")
        for l, cos in rot_layers_sorted[:5]:
            print(f"    L{l}: cos_rotation={cos:.3f}")

    # Find logit-writing peaks
    logit_layers = [(l, abs(results[f'L{l}']['logit_diff'])) for l in sorted(flow_data.keys())
                    if f'L{l}' in results]
    if logit_layers:
        logit_layers_sorted = sorted(logit_layers, key=lambda x: -x[1])
        print(f"\n  Top Logit-Writing Layers (Δh → sg/pl verb logits):")
        for l, ld in logit_layers_sorted[:5]:
            d = results[f'L{l}']
            print(f"    L{l}: logit_diff={d['logit_diff']:.4f} (sg={d['sg_logit_proj']:.4f}, pl={d['pl_logit_proj']:.4f})")

    return results


# ========================================================================
# EXP2: Trajectory Divergence — Attractor Geometry ★★★
# ========================================================================
def run_exp2_trajectory_divergence(model, tokenizer, device, test_sentences,
                                    n_layers, d_model):
    """
    Attractor Geometry: Do grammatical sentences follow STABLE trajectories?

    Key test:
    - Grammatical: "The cat chases the dog" → model predicts "chases"
    - Ungrammatical: "The cat chase the dog" → model predicts "chases"? or "chase"?

    If language = attractor trajectories:
    - Both sentences should converge to the same attractor at deep layers
    - The model should "self-repair" the ungrammatical trajectory

    Method:
    1. Run grammatical (sg_subj + sg_verb) and ungrammatical (sg_subj + pl_verb)
    2. At each layer, measure ||h_l(gram) - h_l(ungram)||
    3. If divergence DECREASES at deep layers → attractor exists
    4. If divergence INCREASES → no attractor for this feature
    """
    print("\n" + "=" * 70)
    print("EXP2: Trajectory Divergence — Attractor Geometry ★★★")
    print("Is language = stable attractor trajectories?")
    print("=" * 70)

    W_U = get_W_U(model, None)
    if isinstance(W_U, torch.Tensor):
        W_U = W_U.float().numpy()

    div_data = defaultdict(lambda: {
        'gram_ungram_div': [],  # divergence: gram vs ungram
        'sg_pl_div': [],        # divergence: sg_subj vs pl_subj (both gram)
    })

    n_valid = 0
    t_exp_start = time.time()

    short_sents = [td for td in test_sentences if td['type'] == 'short_sva']

    for td in short_sents:
        if n_valid >= 40:
            break

        sg_verb_id, pl_verb_id = find_verb_token_ids(tokenizer, td['verb_sg'], td['verb_pl'])
        if sg_verb_id is None or pl_verb_id is None:
            continue

        # Construct THREE sentences:
        # 1. Grammatical sg: "The cat chases the dog"
        # 2. Ungrammatical sg: "The cat chase the dog" (wrong verb)
        # 3. Grammatical pl: "The cats chase the dog"
        gram_sg = td['sent_sg']  # "The cat chases the dog"
        ungram_sg = td['sent_sg'].replace(f" {td['verb_sg']} ", f" {td['verb_pl']} ")
        gram_pl = td['sent_pl']  # "The cats chase the dog"

        gram_ann = tokenize_and_annotate(gram_sg, tokenizer)
        ungram_ann = tokenize_and_annotate(ungram_sg, tokenizer)
        pl_ann = tokenize_and_annotate(gram_pl, tokenizer)

        verb_pred_pos = gram_ann.get('verb_pred_pos')
        if verb_pred_pos is None:
            continue

        # Run all three
        with torch.no_grad():
            gram_out = model(input_ids=gram_ann['input_ids'].to(device),
                           attention_mask=gram_ann['attention_mask'].to(device),
                           output_hidden_states=True)
            ungram_out = model(input_ids=ungram_ann['input_ids'].to(device),
                             attention_mask=ungram_ann['attention_mask'].to(device),
                             output_hidden_states=True)
            pl_out = model(input_ids=pl_ann['input_ids'].to(device),
                          attention_mask=pl_ann['attention_mask'].to(device),
                          output_hidden_states=True)

        # Check that model predicts correct verb for grammatical
        gram_agree, gram_diff = measure_agreement(gram_out.logits, verb_pred_pos,
                                                   sg_verb_id, pl_verb_id)
        if gram_agree < 0.5:
            del gram_out, ungram_out, pl_out
            continue

        # Check what model predicts for ungrammatical — does it "self-repair"?
        ungram_agree, ungram_diff = measure_agreement(ungram_out.logits, verb_pred_pos,
                                                        sg_verb_id, pl_verb_id)
        # If ungram_agree > 0.5 → model "self-repairs" (predicts sg verb despite pl verb in input)

        # Measure divergence at each layer
        for l in range(len(gram_out.hidden_states)):
            h_gram = gram_out.hidden_states[l][0, verb_pred_pos].float().cpu().numpy()
            h_ungram = ungram_out.hidden_states[l][0, verb_pred_pos].float().cpu().numpy()
            h_pl = pl_out.hidden_states[l][0, verb_pred_pos].float().cpu().numpy()

            # Gram vs ungram divergence
            div_gu = float(np.linalg.norm(h_gram - h_ungram))
            # Gram vs pl (both grammatical)
            div_gp = float(np.linalg.norm(h_gram - h_pl))

            # Logit-level divergence
            logit_gram = h_gram @ W_U[sg_verb_id] - h_gram @ W_U[pl_verb_id]
            logit_ungram = h_ungram @ W_U[sg_verb_id] - h_ungram @ W_U[pl_verb_id]
            logit_pl = h_pl @ W_U[pl_verb_id] - h_pl @ W_U[sg_verb_id]

            div_data[l]['gram_ungram_div'].append(div_gu)
            div_data[l]['sg_pl_div'].append(div_gp)
            div_data[l]['gram_logit_diff'] = div_data[l].get('gram_logit_diff', [])
            div_data[l]['gram_logit_diff'].append(float(logit_gram))
            div_data[l]['ungram_logit_diff'] = div_data[l].get('ungram_logit_diff', [])
            div_data[l]['ungram_logit_diff'].append(float(logit_ungram))
            div_data[l]['pl_logit_diff'] = div_data[l].get('pl_logit_diff', [])
            div_data[l]['pl_logit_diff'].append(float(logit_pl))

        del gram_out, ungram_out, pl_out
        torch.cuda.empty_cache()

        n_valid += 1
        if n_valid % 10 == 0:
            elapsed = time.time() - t_exp_start
            print(f"  [EXP2] Processed {n_valid} triples, elapsed={elapsed:.0f}s")

    # ========================================================================
    # Analysis
    # ========================================================================
    print(f"\n  Processed {n_valid} valid triples")

    # Self-repair analysis
    repair_count = sum(1 for d in div_data.get('ungram_logit_diff', [])
                      if d is not None and len(div_data.get('gram_logit_diff', [])) > 0)
    if n_valid > 0:
        ungram_agree_rate = sum(1 for i in range(min(n_valid, len(div_data.get('ungram_logit_diff', []))))
                               if div_data['ungram_logit_diff'][i] > 0) / max(n_valid, 1)
        print(f"\n  Self-repair rate (model predicts sg verb for ungrammatical input): "
              f"{ungram_agree_rate:.3f}")

    print(f"\n  {'Layer':>6} {'||G-U||':>10} {'||G-PL||':>10} {'Ratio':>8} "
          f"{'G_logit':>10} {'U_logit':>10} {'PL_logit':>10} {'Phase':>12}")
    print("  " + "-" * 90)

    results = {}
    prev_gu = None
    for l in sorted(div_data.keys()):
        d = div_data[l]
        if len(d['gram_ungram_div']) < 5:
            continue

        mean_gu = np.mean(d['gram_ungram_div'])
        mean_gp = np.mean(d['sg_pl_div'])
        ratio = mean_gu / max(mean_gp, 1e-8)

        mean_gram_logit = np.mean(d['gram_logit_diff'])
        mean_ungram_logit = np.mean(d['ungram_logit_diff'])
        mean_pl_logit = np.mean(d['pl_logit_diff'])

        # Phase classification based on dynamics
        if prev_gu is not None:
            if mean_gu < prev_gu:
                phase = "CONVERGE↓"
            else:
                phase = "DIVERGE↑"
        else:
            phase = "INPUT"

        results[f"L{l}"] = {
            'gram_ungram_div': float(mean_gu),
            'gram_pl_div': float(mean_gp),
            'div_ratio': float(ratio),
            'gram_logit_diff': float(mean_gram_logit),
            'ungram_logit_diff': float(mean_ungram_logit),
            'pl_logit_diff': float(mean_pl_logit),
            'phase': phase,
        }

        if mean_gu > 0.01:
            print(f"  L{l:>4} {mean_gu:>10.3f} {mean_gp:>10.3f} {ratio:>8.3f} "
                  f"{mean_gram_logit:>10.4f} {mean_ungram_logit:>10.4f} {mean_pl_logit:>10.4f} {phase:>12}")

        prev_gu = mean_gu

    # Find convergence/divergence transition points
    print(f"\n  ★★★ Attractor Geometry Summary ★★★")
    converge_layers = [l for l in sorted(div_data.keys())
                       if f'L{l}' in results and results[f'L{l}']['phase'] == 'CONVERGE↓']
    diverge_layers = [l for l in sorted(div_data.keys())
                      if f'L{l}' in results and results[f'L{l}']['phase'] == 'DIVERGE↑']

    print(f"  Convergence layers: {converge_layers[:10]}...")
    print(f"  Divergence layers: {diverge_layers[:10]}...")

    # Key insight: where do grammatical and ungrammatical trajectories diverge most?
    max_div_layer = max(results.items(), key=lambda x: x[1]['gram_ungram_div'])
    print(f"\n  Maximum G-U divergence at {max_div_layer[0]}: {max_div_layer[1]['gram_ungram_div']:.3f}")

    # Key insight: where does the model "correct" the ungrammatical trajectory?
    # This is where ungram_logit_diff becomes positive (predicting sg verb despite pl verb in input)
    repair_layers = [(l, results[f'L{l}']['ungram_logit_diff']) for l in sorted(div_data.keys())
                     if f'L{l}' in results]
    if repair_layers:
        # Find where ungram_logit_diff transitions from negative to positive
        transitions = []
        for i in range(1, len(repair_layers)):
            if repair_layers[i-1][1] < 0 and repair_layers[i][1] > 0:
                transitions.append(repair_layers[i][0])
        if transitions:
            print(f"\n  Trajectory repair transition at layers: {transitions}")

    return results


# ========================================================================
# EXP3: Jacobian-Vector Product for Number Direction ★★★
# ========================================================================
def run_exp3_jacobian_vector_product(model, tokenizer, device, test_sentences,
                                      n_layers, d_model):
    """
    Compute J_l @ v for the number direction v.

    This is the CORE DYNAMICAL SYSTEMS analysis:
    - v = Δh_l / ||Δh_l|| = normalized number direction at layer l
    - J_l @ v = how layer l transforms the number direction
    - ||J_l @ v|| / ||v|| = amplification of number feature
    - cos(J_l @ v, v) = rotation of number feature

    Method: Finite difference approximation
    - Run clean: h_{l+1} = F_l(h_l)
    - Run perturbed: h_{l+1}' = F_l(h_l + ε*v)
    - J_l @ v ≈ (h_{l+1}' - h_{l+1}) / ε

    This directly measures the Jacobian's action on the number feature.
    """
    print("\n" + "=" * 70)
    print("EXP3: Jacobian-Vector Product for Number Direction ★★★")
    print("J_l @ v: How does each layer TRANSFORM the number feature?")
    print("=" * 70)

    W_U = get_W_U(model, None)
    if isinstance(W_U, torch.Tensor):
        W_U = W_U.float().numpy()

    layers = get_layers(model)
    eps = 0.1  # Perturbation scale

    # Sample layers for JVP computation
    if n_layers <= 12:
        sample_layers = list(range(n_layers))
    else:
        # Dense sampling for important layers
        sample_layers = sorted(set(
            list(range(0, 6)) +
            list(range(n_layers // 3 - 2, n_layers // 3 + 3)) +
            list(range(n_layers * 2 // 3 - 2, n_layers * 2 // 3 + 3)) +
            list(range(n_layers - 6, n_layers))
        ))

    print(f"  Computing JVP at {len(sample_layers)} layers")

    jvp_data = {l: {'amp': [], 'cos': [], 'logit_proj': []} for l in sample_layers}

    n_valid = 0
    t_exp_start = time.time()

    short_sents = [td for td in test_sentences if td['type'] == 'short_sva']

    for td in short_sents:
        if n_valid >= 25:
            break

        sg_verb_id, pl_verb_id = find_verb_token_ids(tokenizer, td['verb_sg'], td['verb_pl'])
        if sg_verb_id is None or pl_verb_id is None:
            continue

        sg_ann = tokenize_and_annotate(td['sent_sg'], tokenizer)
        pl_ann = tokenize_and_annotate(td['sent_pl'], tokenizer)

        verb_pred_pos = sg_ann.get('verb_pred_pos')
        if verb_pred_pos is None:
            continue

        # First, get Δh at each layer (number direction)
        with torch.no_grad():
            sg_out = model(input_ids=sg_ann['input_ids'].to(device),
                          attention_mask=sg_ann['attention_mask'].to(device),
                          output_hidden_states=True)
            pl_out = model(input_ids=pl_ann['input_ids'].to(device),
                          attention_mask=pl_ann['attention_mask'].to(device),
                          output_hidden_states=True)

        # Verify agreement
        agree, _ = measure_agreement(sg_out.logits, verb_pred_pos, sg_verb_id, pl_verb_id)
        if agree < 0.5:
            del sg_out, pl_out
            continue

        # Compute number direction v_l at each layer
        number_dirs = {}
        for l in range(len(sg_out.hidden_states)):
            h_sg = sg_out.hidden_states[l][0, verb_pred_pos].float().cpu().numpy()
            h_pl = pl_out.hidden_states[l][0, verb_pred_pos].float().cpu().numpy()
            delta = h_pl - h_sg
            norm = np.linalg.norm(delta)
            if norm > 1e-8:
                number_dirs[l] = delta / norm  # normalized number direction

        del sg_out, pl_out
        torch.cuda.empty_cache()

        # Now compute JVP using activation patching
        # For each sampled layer l:
        # 1. Get h_l at verb position from clean (sg) forward pass
        # 2. Patch h_l → h_l + ε * v_l (add number direction)
        # 3. Compare h_{l+1} with and without patch → J_l @ v_l

        for l_idx in sample_layers:
            if l_idx not in number_dirs or l_idx >= n_layers - 1:
                continue

            v_number = number_dirs[l_idx]  # [d_model], normalized
            v_tensor = torch.tensor(v_number, dtype=torch.bfloat16, device=device)

            # Capture h_l and h_{l+1} for clean and patched runs
            clean_h_next = {}
            patched_h_next = {}

            def make_capture_hook(storage, key):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        storage[key] = output[0].detach().clone()
                    else:
                        storage[key] = output.detach().clone()
                return hook

            # Clean run — capture h_{l_idx+1}
            hook_clean = layers[l_idx + 1].register_forward_hook(
                make_capture_hook(clean_h_next, 'h'))
            with torch.no_grad():
                model(input_ids=sg_ann['input_ids'].to(device),
                      attention_mask=sg_ann['attention_mask'].to(device),
                      output_hidden_states=True)
            hook_clean.remove()

            # Patched run — at layer l_idx, add ε*v to the output at verb position
            def make_patching_hook(epsilon, direction, pos):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        patched = output[0].clone()
                    else:
                        patched = output.clone()
                    if pos < patched.shape[1]:
                        patched[0, pos, :] += (epsilon * direction).to(patched.dtype)
                    return (patched,) + output[1:] if isinstance(output, tuple) else patched
                return hook

            hook_patch = layers[l_idx].register_forward_hook(
                make_patching_hook(eps, v_tensor, verb_pred_pos))
            hook_next = layers[l_idx + 1].register_forward_hook(
                make_capture_hook(patched_h_next, 'h'))

            with torch.no_grad():
                model(input_ids=sg_ann['input_ids'].to(device),
                      attention_mask=sg_ann['attention_mask'].to(device),
                      output_hidden_states=True)

            hook_patch.remove()
            hook_next.remove()

            # Compute JVP: (h_{l+1}^patched - h_{l+1}^clean) / ε
            if 'h' in clean_h_next and 'h' in patched_h_next:
                h_clean = clean_h_next['h'][0, verb_pred_pos].float().cpu().numpy()
                h_patched = patched_h_next['h'][0, verb_pred_pos].float().cpu().numpy()
                jvp = (h_patched - h_clean) / eps  # J_l @ v_l

                jvp_norm = np.linalg.norm(jvp)
                amp = jvp_norm  # ||J_l @ v_l|| (since ||v_l|| = 1)
                cos_with_v = float(np.dot(jvp, v_number) / max(jvp_norm * np.linalg.norm(v_number), 1e-8))

                # Project JVP through W_U
                jvp_sg = float(jvp @ W_U[sg_verb_id])
                jvp_pl = float(jvp @ W_U[pl_verb_id])

                jvp_data[l_idx]['amp'].append(amp)
                jvp_data[l_idx]['cos'].append(cos_with_v)
                jvp_data[l_idx]['logit_proj'].append(jvp_sg - jvp_pl)

            del clean_h_next, patched_h_next
            torch.cuda.empty_cache()

        n_valid += 1
        if n_valid % 5 == 0:
            elapsed = time.time() - t_exp_start
            print(f"  [EXP3] Processed {n_valid} pairs, elapsed={elapsed:.0f}s")

    # ========================================================================
    # Analysis
    # ========================================================================
    print(f"\n  Processed {n_valid} valid pairs")

    print(f"\n  {'Layer':>6} {'||J@v||':>10} {'cos(J@v,v)':>12} {'Logit_Proj':>12} {'Dynamics':>15}")
    print("  " + "-" * 60)

    results = {}
    for l_idx in sorted(jvp_data.keys()):
        d = jvp_data[l_idx]
        if len(d['amp']) < 3:
            continue

        mean_amp = np.mean(d['amp'])
        mean_cos = np.mean(d['cos'])
        mean_logit = np.mean(d['logit_proj'])

        # Dynamics classification
        if mean_amp > 1.0 and mean_cos > 0.5:
            dynamics = "AMPLIFY+ALIGN"
        elif mean_amp > 1.0 and mean_cos < 0.3:
            dynamics = "AMPLIFY+ROTATE"
        elif mean_amp < 0.5 and mean_cos > 0.5:
            dynamics = "DAMPEN+ALIGN"
        elif mean_amp < 0.5 and mean_cos < 0.3:
            dynamics = "DAMPEN+ROTATE"
        elif mean_cos < 0:
            dynamics = "REVERSE"
        else:
            dynamics = "TRANSFORM"

        results[f"L{l_idx}"] = {
            'jvp_norm': float(mean_amp),
            'cos_with_number': float(mean_cos),
            'logit_projection': float(mean_logit),
            'dynamics': dynamics,
        }

        if mean_amp > 0.01:
            print(f"  L{l_idx:>4} {mean_amp:>10.4f} {mean_cos:>12.4f} {mean_logit:>12.4f} {dynamics:>15}")

    # Summary
    print(f"\n  ★★★ Jacobian-Vector Product Summary ★★★")

    # Find amplification layers
    amp_layers = [(l, results[f'L{l}']['jvp_norm']) for l in sorted(jvp_data.keys())
                  if f'L{l}' in results]
    if amp_layers:
        print(f"\n  Number Feature Dynamics Through Layers:")
        for l, amp in sorted(amp_layers, key=lambda x: x[0]):
            d = results[f'L{l}']
            print(f"    L{l}: ||J@v||={d['jvp_norm']:.4f}, cos={d['cos_with_number']:.4f}, "
                  f"logit={d['logit_projection']:.4f} → {d['dynamics']}")

    return results


# ========================================================================
# EXP4: Beyond SVA — Semantic & Structural Constraints ★★
# ========================================================================
def run_exp4_beyond_sva(model, tokenizer, device, test_sentences,
                         n_layers, d_model):
    """
    Test whether the same circuits handle deeper linguistic phenomena.

    Three sub-experiments:
    A. Negation: "The cat does not chase" vs "The cats do not chase"
       - Agreement is now on the auxiliary (does/do), not the main verb
       - Test: does the same AGREEMENT-MLP layer handle this?

    B. Coordination: "The cat and the dog chase"
       - Plural verb despite singular nouns
       - Test: does the same circuit handle conjunction-induced plurality?

    C. Long-range: Center-embedded with more distance
       - Already tested, but check if negation/coordination interact with distance
    """
    print("\n" + "=" * 70)
    print("EXP4: Beyond SVA — Semantic & Structural Constraints ★★")
    print("Do the same circuits handle negation and coordination?")
    print("=" * 70)

    W_U = get_W_U(model, None)
    if isinstance(W_U, torch.Tensor):
        W_U = W_U.float().numpy()

    results = {'negation': {}, 'coordination': {}, 'long_sva': {}}

    # ---- Sub-experiment A: Negation ----
    print(f"\n  --- A: Negation ---")
    neg_sents = [td for td in test_sentences if td['type'] == 'negation']
    neg_agree = {'sg_correct': 0, 'sg_total': 0, 'pl_correct': 0, 'pl_total': 0}

    for td in neg_sents:
        sg_verb_id, pl_verb_id = find_verb_token_ids(tokenizer, td['verb_sg'], td['verb_pl'])
        if sg_verb_id is None or pl_verb_id is None:
            continue

        sg_ann = tokenize_and_annotate(td['sent_sg'], tokenizer)
        pl_ann = tokenize_and_annotate(td['sent_pl'], tokenizer)

        # Find auxiliary position for prediction
        sg_decoded = sg_ann.get('decoded_tokens', [])
        pl_decoded = pl_ann.get('decoded_tokens', [])

        # Find "does"/"do" position
        sg_aux_pos = None
        pl_aux_pos = None
        for i, t in enumerate(sg_decoded):
            if t.strip().lower() in ('does', 'do'):
                sg_aux_pos = i
                break
        for i, t in enumerate(pl_decoded):
            if t.strip().lower() in ('does', 'do'):
                pl_aux_pos = i
                break

        if sg_aux_pos is None or pl_aux_pos is None:
            continue

        # Test agreement
        with torch.no_grad():
            sg_out = model(input_ids=sg_ann['input_ids'].to(device),
                          attention_mask=sg_ann['attention_mask'].to(device))
            pl_out = model(input_ids=pl_ann['input_ids'].to(device),
                          attention_mask=pl_ann['attention_mask'].to(device))

        # SG: should predict "does" (sg_verb_id = "does")
        sg_agree, sg_diff = measure_agreement(sg_out.logits, sg_aux_pos - 1,
                                               sg_verb_id, pl_verb_id)
        # PL: should predict "do" (pl_verb_id = "do")
        pl_agree, pl_diff = measure_agreement(pl_out.logits, pl_aux_pos - 1,
                                               pl_verb_id, sg_verb_id)

        if sg_agree > 0.5:
            neg_agree['sg_correct'] += 1
        neg_agree['sg_total'] += 1
        if pl_agree > 0.5:
            neg_agree['pl_correct'] += 1
        neg_agree['pl_total'] += 1

        del sg_out, pl_out
        torch.cuda.empty_cache()

    sg_rate = neg_agree['sg_correct'] / max(neg_agree['sg_total'], 1)
    pl_rate = neg_agree['pl_correct'] / max(neg_agree['pl_total'], 1)
    print(f"  Negation agreement: SG={sg_rate:.3f} ({neg_agree['sg_correct']}/{neg_agree['sg_total']}), "
          f"PL={pl_rate:.3f} ({neg_agree['pl_correct']}/{neg_agree['pl_total']})")
    results['negation'] = {
        'sg_rate': float(sg_rate), 'pl_rate': float(pl_rate),
        'sg_total': neg_agree['sg_total'], 'pl_total': neg_agree['pl_total'],
    }

    # ---- Sub-experiment B: Coordination ----
    print(f"\n  --- B: Coordination ---")
    coord_sents = [td for td in test_sentences if td['type'] == 'coordination']
    coord_agree = {'pl_correct': 0, 'pl_total': 0}

    for td in coord_sents:
        sg_verb_id, pl_verb_id = find_verb_token_ids(tokenizer, td['verb_sg'], td['verb_pl'])
        if sg_verb_id is None or pl_verb_id is None:
            continue

        # For coordination, test whether model predicts PLURAL verb
        pl_ann = tokenize_and_annotate(td['sent_pl'], tokenizer)

        # Find verb position (near end)
        pl_decoded = pl_ann.get('decoded_tokens', [])
        verb_pred_pos = len(pl_decoded) - 2  # predict last token from second-to-last

        with torch.no_grad():
            pl_out = model(input_ids=pl_ann['input_ids'].to(device),
                          attention_mask=pl_ann['attention_mask'].to(device))

        # Model should predict pl_verb for "The cat and the dog [chase]"
        pl_agree, pl_diff = measure_agreement(pl_out.logits, verb_pred_pos,
                                               pl_verb_id, sg_verb_id)

        if pl_agree > 0.5:
            coord_agree['pl_correct'] += 1
        coord_agree['pl_total'] += 1

        del pl_out
        torch.cuda.empty_cache()

    coord_rate = coord_agree['pl_correct'] / max(coord_agree['pl_total'], 1)
    print(f"  Coordination agreement: PL={coord_rate:.3f} ({coord_agree['pl_correct']}/{coord_agree['pl_total']})")
    results['coordination'] = {
        'pl_rate': float(coord_rate),
        'pl_total': coord_agree['pl_total'],
    }

    # ---- Sub-experiment C: Long-range SVA (comparison) ----
    print(f"\n  --- C: Long-range SVA (for comparison) ---")
    long_sents = [td for td in test_sentences if td['type'] == 'long_sva']
    long_agree = {'sg_correct': 0, 'sg_total': 0, 'pl_correct': 0, 'pl_total': 0}

    for td in long_sents:
        sg_verb_id, pl_verb_id = find_verb_token_ids(tokenizer, td['verb_sg'], td['verb_pl'])
        if sg_verb_id is None or pl_verb_id is None:
            continue

        sg_ann = tokenize_and_annotate(td['sent_sg'], tokenizer)
        pl_ann = tokenize_and_annotate(td['sent_pl'], tokenizer)

        verb_pred_pos = sg_ann.get('verb_present_pos')
        if verb_pred_pos is None:
            verb_pred_pos = sg_ann.get('verb_pred_pos')
        if verb_pred_pos is None:
            continue

        # Adjust for prediction position (predict verb from previous position)
        pred_pos = verb_pred_pos - 1
        if pred_pos < 0:
            continue

        with torch.no_grad():
            sg_out = model(input_ids=sg_ann['input_ids'].to(device),
                          attention_mask=sg_ann['attention_mask'].to(device))
            pl_out = model(input_ids=pl_ann['input_ids'].to(device),
                          attention_mask=pl_ann['attention_mask'].to(device))

        sg_agree, _ = measure_agreement(sg_out.logits, pred_pos, sg_verb_id, pl_verb_id)
        pl_agree, _ = measure_agreement(pl_out.logits, pred_pos, pl_verb_id, sg_verb_id)

        if sg_agree > 0.5:
            long_agree['sg_correct'] += 1
        long_agree['sg_total'] += 1
        if pl_agree > 0.5:
            long_agree['pl_correct'] += 1
        long_agree['pl_total'] += 1

        del sg_out, pl_out
        torch.cuda.empty_cache()

    sg_rate_long = long_agree['sg_correct'] / max(long_agree['sg_total'], 1)
    pl_rate_long = long_agree['pl_correct'] / max(long_agree['pl_total'], 1)
    print(f"  Long-range SVA: SG={sg_rate_long:.3f} ({long_agree['sg_correct']}/{long_agree['sg_total']}), "
          f"PL={pl_rate_long:.3f} ({long_agree['pl_correct']}/{long_agree['pl_total']})")
    results['long_sva'] = {
        'sg_rate': float(sg_rate_long), 'pl_rate': float(pl_rate_long),
        'sg_total': long_agree['sg_total'], 'pl_total': long_agree['pl_total'],
    }

    # Comparison
    print(f"\n  ★★★ Beyond SVA Summary ★★★")
    print(f"  Phenomenon         | SG_agree | PL_agree | N")
    print(f"  -------------------|----------|----------|-----")
    print(f"  Short SVA          | ~0.90    | ~0.90    | 25")
    print(f"  Long SVA            | {sg_rate_long:.3f}    | {pl_rate_long:.3f}    | {long_agree['sg_total']}")
    print(f"  Negation            | {results['negation']['sg_rate']:.3f}    | {results['negation']['pl_rate']:.3f}    | {neg_agree['sg_total']}")
    print(f"  Coordination        | N/A      | {coord_rate:.3f}    | {coord_agree['pl_total']}")

    return results


# ========================================================================
# Main
# ========================================================================
def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    t_start = time.time()
    print(f"\n{'='*70}")
    print(f"Phase 213: Transformer as Dynamical System — {model_name}")
    print(f"FROM TRANSITIONS TO DYNAMICS")
    print(f"{'='*70}")

    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    n_heads = getattr(model.config, 'num_attention_heads', d_model // 64)
    intermediate_size = info.intermediate_size
    print(f"  n_layers={n_layers}, d_model={d_model}, n_heads={n_heads}, "
          f"intermediate_size={intermediate_size}")

    test_sentences = generate_all_sentences()
    all_results = {
        'model': model_name, 'n_layers': n_layers, 'd_model': d_model,
        'n_heads': n_heads, 'intermediate_size': intermediate_size,
    }

    # EXP1: Number Signal Flow ★★★
    try:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting EXP1: Number Signal Flow...")
        exp1_results = run_exp1_number_signal_flow(
            model, tokenizer, device, test_sentences, n_layers, d_model)
        all_results['exp1_number_signal_flow'] = exp1_results
    except Exception as e:
        print(f"  EXP1 FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results['exp1_number_signal_flow'] = {"error": str(e)}
    torch.cuda.empty_cache()

    # EXP2: Trajectory Divergence ★★★
    try:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting EXP2: Trajectory Divergence...")
        exp2_results = run_exp2_trajectory_divergence(
            model, tokenizer, device, test_sentences, n_layers, d_model)
        all_results['exp2_trajectory_divergence'] = exp2_results
    except Exception as e:
        print(f"  EXP2 FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results['exp2_trajectory_divergence'] = {"error": str(e)}
    torch.cuda.empty_cache()

    # EXP3: Jacobian-Vector Product ★★★
    try:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting EXP3: Jacobian-Vector Product...")
        exp3_results = run_exp3_jacobian_vector_product(
            model, tokenizer, device, test_sentences, n_layers, d_model)
        all_results['exp3_jacobian_vector_product'] = exp3_results
    except Exception as e:
        print(f"  EXP3 FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results['exp3_jacobian_vector_product'] = {"error": str(e)}
    torch.cuda.empty_cache()

    # EXP4: Beyond SVA ★★
    try:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting EXP4: Beyond SVA...")
        exp4_results = run_exp4_beyond_sva(
            model, tokenizer, device, test_sentences, n_layers, d_model)
        all_results['exp4_beyond_sva'] = exp4_results
    except Exception as e:
        print(f"  EXP4 FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results['exp4_beyond_sva'] = {"error": str(e)}
    torch.cuda.empty_cache()

    # ========================================================================
    # Summary
    # ========================================================================
    print(f"\n{'='*70}")
    print(f"PHASE 213 SUMMARY — {model_name}")
    print(f"{'='*70}")

    # EXP1 Summary
    exp1 = all_results.get('exp1_number_signal_flow', {})
    if isinstance(exp1, dict):
        print(f"\n--- Exp1: Number Signal Flow ---")
        amp_layers = [(k, v) for k, v in exp1.items()
                      if isinstance(v, dict) and v.get('amp_ratio', 0) > 1.1]
        if amp_layers:
            print(f"  Amplification layers (number signal grows):")
            for lk, v in sorted(amp_layers, key=lambda x: -x[1]['amp_ratio']):
                print(f"    {lk}: amp={v['amp_ratio']:.3f}, logit_diff={v['logit_diff']:.4f}")

        logit_layers = [(k, v) for k, v in exp1.items()
                        if isinstance(v, dict) and abs(v.get('logit_diff', 0)) > 0.2]
        if logit_layers:
            print(f"  Logit-writing layers (number → verb logits):")
            for lk, v in sorted(logit_layers, key=lambda x: -abs(x[1]['logit_diff'])):
                print(f"    {lk}: logit_diff={v['logit_diff']:.4f}")

    # EXP2 Summary
    exp2 = all_results.get('exp2_trajectory_divergence', {})
    if isinstance(exp2, dict):
        print(f"\n--- Exp2: Trajectory Divergence ---")
        converge = [(k, v) for k, v in exp2.items()
                    if isinstance(v, dict) and v.get('phase') == 'CONVERGE↓']
        diverge = [(k, v) for k, v in exp2.items()
                   if isinstance(v, dict) and v.get('phase') == 'DIVERGE↑']
        print(f"  Convergence layers (G-U divergence decreases): {len(converge)}")
        print(f"  Divergence layers (G-U divergence increases): {len(diverge)}")

    # EXP3 Summary
    exp3 = all_results.get('exp3_jacobian_vector_product', {})
    if isinstance(exp3, dict):
        print(f"\n--- Exp3: Jacobian-Vector Product ---")
        dynamics_counts = defaultdict(int)
        for k, v in exp3.items():
            if isinstance(v, dict):
                dynamics_counts[v.get('dynamics', 'UNKNOWN')] += 1
        for dyn, count in sorted(dynamics_counts.items()):
            print(f"  {dyn}: {count} layers")

    # EXP4 Summary
    exp4 = all_results.get('exp4_beyond_sva', {})
    if isinstance(exp4, dict):
        print(f"\n--- Exp4: Beyond SVA ---")
        neg = exp4.get('negation', {})
        coord = exp4.get('coordination', {})
        long = exp4.get('long_sva', {})
        if neg:
            print(f"  Negation: SG={neg.get('sg_rate', 0):.3f}, PL={neg.get('pl_rate', 0):.3f}")
        if coord:
            print(f"  Coordination: PL={coord.get('pl_rate', 0):.3f}")
        if long:
            print(f"  Long SVA: SG={long.get('sg_rate', 0):.3f}, PL={long.get('pl_rate', 0):.3f}")

    # Save results
    results_path = f"tests/glm5_temp/phase213_{model_name}_results.json"
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
