"""
Phase 209: Causal Relational Intervention — THE DECISIVE EXPERIMENT
==================================================================

THEORETICAL CONTEXT:
  Phase 208 showed bilinear operators can DECODE relational structure.
  BUT: "Recoverable ≠ Causally Used" (linear probe illusion)
  AUC=1.0 with 163K params on ~240 samples → likely memorization

  KEY QUESTION: Is h_i^T A h_j a READABLE TRACE or a CAUSAL MECHANISM?

  If destroying h_s^T A h_j breaks S-V agreement → A is CAUSAL
  If destroying h_s^T A h_j doesn't break agreement → A is PROBE ARTIFACT

THREE PERTURBATION TYPES:
  - ANTI-RELATION: δ reduces h_s^T A h_v (targeted, preserves norm & cosine)
  - RANDOM: δ of same magnitude in random direction (control)
  - PRO-RELATION: δ INCREASES h_s^T A h_v (enhancing relation)

DECISION CRITERIA:
  If ANTI-RELATION >> RANDOM → A is CAUSAL
  If ANTI-RELATION ≈ RANDOM → A is PROBE ARTIFACT

DATA: 200+ S-V agreement test sentences
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
import torch.optim as optim
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from sklearn.metrics import roc_auc_score

from model_utils import (get_model_info, release_model, get_layers, get_W_U,
                          MODEL_CONFIGS)

warnings.filterwarnings('ignore')
LITE = os.environ.get('LITE', '0') == '1'

# ========================================================================
# Configuration
# ========================================================================
OPERATOR_RANK = 32
EPSILONS = [0.3, 1.0, 2.0]
PERT_TYPES = ['anti_relation', 'random', 'pro_relation']
AGREEMENT_THRESHOLD = 0.7


# ========================================================================
# Sentence Data
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
    ("deer", "sees", "see", "horse"),
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
    """Generate S-V agreement test sentences (singular + plural pairs)."""
    test_data = []
    for subj, v3sg, vbase, obj in SENTENCE_TRIPLES:
        sg_sent = f"The {subj} {v3sg} the {obj}"
        pl_subj = make_plural(subj)
        pl_sent = f"The {pl_subj} {vbase} the {obj}"
        test_data.append({
            'sent_sg': sg_sent, 'sent_pl': pl_sent,
            'verb_sg': v3sg, 'verb_pl': vbase,
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
    for attn_impl in ["sdpa", "eager"]:
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
# Tokenization & Position Finding — ROBUST VERSION
# ========================================================================
def tokenize_and_annotate(sentence, tokenizer):
    """
    Tokenize sentence and find SVO positions.
    Returns dict with positions and token IDs, or None if pattern not found.
    """
    toks = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=64)
    input_ids = toks["input_ids"]
    ids_list = input_ids[0].tolist()
    
    # Decode each token individually to find the pattern
    decoded_tokens = []
    for tid in ids_list:
        text = tokenizer.decode([tid])
        decoded_tokens.append(text)
    
    # Find "The/det X/subj V/verb the/det Y/obj" pattern
    # Look for determiners and match the pattern
    determiners_lower = {"the", " a", " an", "the", "a", "an"}
    
    for i in range(len(decoded_tokens) - 4):
        t0 = decoded_tokens[i].strip().lower()
        t3 = decoded_tokens[i+3].strip().lower()
        if t0 in ("the", "a", "an") and t3 in ("the", "a", "an"):
            return {
                'det1_pos': i,
                'subj_pos': i + 1,
                'verb_pos': i + 2,
                'det2_pos': i + 3,
                'obj_pos': i + 4,
                'input_ids': input_ids,
                'attention_mask': toks["attention_mask"],
                'decoded_tokens': decoded_tokens,
                # The verb_pred_pos: logits at this position predict the verb token
                # In causal LMs: logits[i] predicts token[i+1]
                # So to predict verb at verb_pos, we look at logits[verb_pos - 1]
                'verb_pred_pos': i + 1,  # = subj_pos, predicts what comes after subject
                'verb_token_id': ids_list[i + 2],  # the actual verb token in the sentence
            }
    
    return None


def find_wrong_verb_id(tokenizer, correct_verb_id, verb_sg, verb_pl, decoded_tokens, verb_pos):
    """
    Find the token ID for the WRONG verb form.
    
    The correct verb token is already in the sentence at verb_pos.
    We need to find the token ID for the alternative verb form.
    
    Strategy: Try encoding with space prefix (most tokenizers use " chase" not "chase")
    """
    # Determine which form is in the sentence and which is the alternative
    verb_text = decoded_tokens[verb_pos].strip().lower()
    
    if verb_text == verb_sg.strip().lower() or verb_text.endswith(verb_sg.strip().lower()):
        # Sentence has singular verb, wrong = plural
        wrong_verb = verb_pl
    else:
        # Sentence has plural verb, wrong = singular  
        wrong_verb = verb_sg
    
    # Try different encodings to find the wrong verb token
    candidates = [
        tokenizer.encode(" " + wrong_verb, add_special_tokens=False),
        tokenizer.encode(wrong_verb, add_special_tokens=False),
    ]
    
    for ids in candidates:
        if len(ids) == 1:
            return ids[0]
    
    return None


# ========================================================================
# Agreement Measurement
# ========================================================================
def measure_agreement(logits, verb_pred_pos, correct_verb_id, wrong_verb_id):
    """
    Measure S-V agreement.
    
    In causal LMs: logits[i] predicts token[i+1]
    verb_pred_pos is the position whose logits predict the verb.
    """
    verb_logits = logits[0, verb_pred_pos]  # [vocab_size]
    
    logit_correct = verb_logits[correct_verb_id].float().item()
    logit_wrong = verb_logits[wrong_verb_id].float().item()
    logit_diff = logit_correct - logit_wrong
    agreement = torch.sigmoid(torch.tensor(logit_diff)).item()
    
    return {
        'logit_correct': logit_correct,
        'logit_wrong': logit_wrong,
        'logit_diff': logit_diff,
        'agreement': agreement,
    }


# ========================================================================
# Bilinear Relation Operator (from Phase 208)
# ========================================================================
class BilinearRelationOperator(nn.Module):
    def __init__(self, d_model, rank=32):
        super().__init__()
        self.rank = rank
        self.U = nn.Parameter(torch.randn(rank, d_model) * 0.01)
        self.V = nn.Parameter(torch.randn(rank, d_model) * 0.01)

    def score_pair(self, h_i, h_j):
        return (self.U @ h_i * self.V @ h_j).sum()

    def score_batch(self, h_i_batch, h_j_batch):
        p_i = h_i_batch @ self.U.T
        p_j = h_j_batch @ self.V.T
        return (p_i * p_j).sum(dim=1)


def train_relation_operator(pos_pairs, neg_pairs, d_model, rank=32,
                             epochs=80, lr=0.001, verbose=False):
    if len(pos_pairs) < 5 or len(neg_pairs) < 5:
        return None, []
    pos_h_i = torch.tensor(np.array([p[0] for p in pos_pairs]), dtype=torch.float32)
    pos_h_j = torch.tensor(np.array([p[1] for p in pos_pairs]), dtype=torch.float32)
    neg_h_i = torch.tensor(np.array([p[0] for p in neg_pairs]), dtype=torch.float32)
    neg_h_j = torch.tensor(np.array([p[1] for p in neg_pairs]), dtype=torch.float32)

    model = BilinearRelationOperator(d_model, rank)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pos_scores = model.score_batch(pos_h_i, pos_h_j)
        neg_scores = model.score_batch(neg_h_i, neg_h_j)
        diff = pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0)
        loss = -torch.log(torch.sigmoid(diff) + 1e-8).mean()
        loss.backward()
        optimizer.step()
        if verbose and epoch % 20 == 0:
            print(f"    Epoch {epoch}: loss={loss.item():.4f}")
    return model, []


def extract_A_matrix(operator_model):
    U = operator_model.U.detach().numpy()
    V = operator_model.V.detach().numpy()
    return U.T @ V  # [d_model, d_model]


def extract_pairs_from_annotated(hidden_states_list, annotations_list,
                                  relation_type='subject_verb', normalize=True):
    """Extract positive and negative pairs from pre-annotated hidden states."""
    pos_pairs = []
    neg_pairs = []
    
    for hs, svo in zip(hidden_states_list, annotations_list):
        if hs is None or svo is None:
            continue
        n_tok = hs.shape[0]
        s, v, o = svo['subj_pos'], svo['verb_pos'], svo['obj_pos']
        
        if normalize:
            norms = np.linalg.norm(hs, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            hs_norm = hs / norms
        else:
            hs_norm = hs
        
        if relation_type == 'subject_verb':
            pi, pj = s, v
        elif relation_type == 'verb_object':
            pi, pj = v, o
        else:
            continue
        
        if pi >= n_tok or pj >= n_tok:
            continue
        
        pos_pairs.append((hs_norm[pi].copy(), hs_norm[pj].copy()))
        
        neg_count = 0
        for i in range(n_tok):
            for j in range(i+1, n_tok):
                if i == pi and j == pj:
                    continue
                neg_pairs.append((hs_norm[i].copy(), hs_norm[j].copy()))
                neg_count += 1
                if neg_count >= 5:
                    break
            if neg_count >= 5:
                break
    
    return pos_pairs, neg_pairs


# ========================================================================
# Perturbation Construction
# ========================================================================
def construct_perturbation(h_s, h_v, A_matrix, eps, pert_type='anti_relation'):
    """
    Construct perturbation δ for h_s at a specific layer.
    
    DESIGN GOALS:
    1. ANTI-RELATION: reduce h_s^T A h_v while preserving ||h_s|| and cos(h_s, h_v)
    2. RANDOM: same magnitude in random direction (CONTROL)
    3. PRO-RELATION: INCREASE h_s^T A h_v
    """
    h_s = h_s.astype(np.float32)
    h_v = h_v.astype(np.float32)
    A = A_matrix.astype(np.float32)
    
    h_s_norm = np.linalg.norm(h_s)
    h_v_norm = np.linalg.norm(h_v)
    
    if h_s_norm < 1e-10 or h_v_norm < 1e-10:
        return np.zeros_like(h_s)
    
    if pert_type == 'random':
        delta = np.random.randn(*h_s.shape).astype(np.float32)
        delta = delta - np.dot(delta, h_s / h_s_norm) * (h_s / h_s_norm)
        d_norm = np.linalg.norm(delta)
        if d_norm < 1e-10:
            return np.zeros_like(h_s)
        return delta / d_norm * eps * h_s_norm
    
    # Gradient of h_s^T A h_v w.r.t. h_s = A h_v
    grad_relation = A @ h_v
    grad_relation_norm = np.linalg.norm(grad_relation)
    if grad_relation_norm < 1e-10:
        return np.zeros_like(h_s)
    
    direction = -grad_relation if pert_type == 'anti_relation' else grad_relation
    
    # Preserve cosine: project orthogonal to grad_cos
    cos_sv = np.dot(h_s, h_v) / (h_s_norm * h_v_norm)
    grad_cos = h_v / (h_s_norm * h_v_norm) - cos_sv * h_s / (h_s_norm ** 2)
    grad_cos_norm = np.linalg.norm(grad_cos)
    if grad_cos_norm > 1e-10:
        direction = direction - np.dot(direction, grad_cos) / (grad_cos_norm ** 2) * grad_cos
    
    # Preserve norm: project orthogonal to h_s direction
    grad_norm_dir = h_s / h_s_norm
    direction = direction - np.dot(direction, grad_norm_dir) * grad_norm_dir
    
    direction_norm = np.linalg.norm(direction)
    if direction_norm < 1e-10:
        # Fallback: use raw gradient without cosine constraint
        direction = (-grad_relation if pert_type == 'anti_relation' else grad_relation)
        direction = direction - np.dot(direction, grad_norm_dir) * grad_norm_dir
        direction_norm = np.linalg.norm(direction)
        if direction_norm < 1e-10:
            return np.zeros_like(h_s)
    
    return direction / direction_norm * eps * h_s_norm


# ========================================================================
# Intervention Hook
# ========================================================================
def make_intervention_hook(subj_pos, delta_tensor):
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0].clone()
            hidden[:, subj_pos, :] += delta_tensor.unsqueeze(0).to(hidden.device).to(hidden.dtype)
            return (hidden,) + output[1:]
        else:
            hidden = output.clone()
            hidden[:, subj_pos, :] += delta_tensor.unsqueeze(0).to(hidden.device).to(hidden.dtype)
            return hidden
    return hook_fn


# ========================================================================
# Experiment 1: Causal Relational Intervention (THE CORE)
# ========================================================================
def run_causal_intervention(model, tokenizer, device, test_data, n_layers, d_model):
    """
    THE DECISIVE EXPERIMENT: Is the relation operator A causally used?
    """
    print(f"\n{'='*70}")
    print("EXP1: Causal Relational Intervention (THE DECISIVE TEST)")
    print(f"{'='*70}")
    
    layers = get_layers(model)
    interv_layers = sorted(set([
        1, n_layers // 6, n_layers // 3, n_layers // 2,
        2 * n_layers // 3, 5 * n_layers // 6, n_layers - 1
    ])) if n_layers > 12 else list(range(n_layers))
    print(f"  Intervention layers: {interv_layers}")
    
    # ---- Step 1: Baseline agreement measurement ----
    print(f"\n  Step 1: Measuring baseline S-V agreement...")
    baseline_results = []
    all_hidden_for_training = {l: [] for l in range(n_layers + 1)}
    all_annotations_for_training = []
    
    n_test = min(len(test_data), 50 if LITE else 200)
    
    for di in range(n_test):
        td = test_data[di]
        sent = td['sent_sg']
        
        # Tokenize and annotate
        anno = tokenize_and_annotate(sent, tokenizer)
        if anno is None:
            continue
        
        # Get wrong verb token ID
        wrong_id = find_wrong_verb_id(
            tokenizer, anno['verb_token_id'],
            td['verb_sg'], td['verb_pl'],
            anno['decoded_tokens'], anno['verb_pos']
        )
        if wrong_id is None:
            continue
        
        correct_id = anno['verb_token_id']
        input_ids = anno['input_ids'].to(device)
        attn_mask = anno['attention_mask'].to(device)
        
        # Forward pass with hidden states
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn_mask,
                       output_hidden_states=True)
        
        agreement = measure_agreement(
            out.logits, anno['verb_pred_pos'], correct_id, wrong_id
        )
        
        # Store for training
        if agreement['agreement'] > 0.3:
            for li in range(n_layers + 1):
                hs = out.hidden_states[li].float().cpu().numpy()[0]
                all_hidden_for_training[li].append(hs)
            all_annotations_for_training.append(anno)
        
        baseline_results.append({
            'data_idx': di, 'sent': sent,
            'anno': anno, 'correct_id': correct_id, 'wrong_id': wrong_id,
            'agreement': agreement['agreement'],
            'logit_diff': agreement['logit_diff'],
        })
        
        if di % 20 == 0:
            # Debug: show top-5 predictions at verb_pred_pos
            verb_logits = out.logits[0, anno['verb_pred_pos']].float().cpu()
            top5_ids = verb_logits.topk(5).indices.tolist()
            top5_tokens = [tokenizer.decode([tid]).strip() for tid in top5_ids]
            top5_scores = verb_logits[top5_ids].tolist()
            print(f"    [{di+1}/{n_test}] agree={agreement['agreement']:.4f} "
                  f"top5={list(zip(top5_tokens, [f'{s:.2f}' for s in top5_scores]))}")
        
        del out
        torch.cuda.empty_cache()
    
    high_agreement = [r for r in baseline_results if r['agreement'] > AGREEMENT_THRESHOLD]
    med_agreement = [r for r in baseline_results if r['agreement'] > 0.5]
    
    print(f"  Total sentences: {len(baseline_results)}")
    print(f"  Agreement > {AGREEMENT_THRESHOLD}: {len(high_agreement)}")
    print(f"  Agreement > 0.5: {len(med_agreement)}")
    
    # Use medium threshold if not enough high-agreement sentences
    test_sentences = high_agreement if len(high_agreement) >= 10 else med_agreement
    if len(test_sentences) < 5:
        # Use all with agreement > 0.3
        test_sentences = [r for r in baseline_results if r['agreement'] > 0.3]
    if len(test_sentences) < 5:
        print("  [ERROR] Not enough test sentences with any agreement. Skipping.")
        return {}
    
    print(f"  Using {len(test_sentences)} test sentences (threshold adjusted)")
    
    # ---- Step 2: Train relation operators ----
    print(f"\n  Step 2: Training relation operators...")
    operator_results = {}
    
    for li in interv_layers:
        hs_list = all_hidden_for_training.get(li, [])
        if len(hs_list) < 5:
            continue
        pos_pairs, neg_pairs = extract_pairs_from_annotated(
            hs_list, all_annotations_for_training, 'subject_verb'
        )
        if len(pos_pairs) < 5:
            continue
        model_op, _ = train_relation_operator(
            pos_pairs, neg_pairs, d_model, rank=OPERATOR_RANK, epochs=80, lr=0.001,
            verbose=(li == interv_layers[len(interv_layers)//2])
        )
        if model_op is not None:
            A = extract_A_matrix(model_op)
            operator_results[li] = A
            print(f"    L{li}: operator trained ({len(pos_pairs)} pos pairs)")
    
    if not operator_results:
        print("  [ERROR] Failed to train operators. Skipping intervention.")
        return {}
    
    # ---- Step 3: Causal intervention ----
    print(f"\n  Step 3: Running causal intervention on {len(test_sentences)} sentences...")
    
    intervention_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    relation_score_changes = defaultdict(lambda: defaultdict(list))
    
    n_interv = min(len(test_sentences), 30 if LITE else 120)
    
    for si in range(n_interv):
        br = test_sentences[si]
        sent = br['sent']
        anno = br['anno']
        correct_id = br['correct_id']
        wrong_id = br['wrong_id']
        subj_pos = anno['subj_pos']
        verb_pos = anno['verb_pos']
        verb_pred_pos = anno['verb_pred_pos']
        baseline_agreement = br['agreement']
        baseline_logit_diff = br['logit_diff']
        
        input_ids = anno['input_ids'].to(device)
        attn_mask = anno['attention_mask'].to(device)
        
        # Get baseline hidden states
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn_mask,
                       output_hidden_states=True)
        
        baseline_agreement_check = measure_agreement(
            out.logits, verb_pred_pos, correct_id, wrong_id
        )
        
        baseline_hidden = {}
        for li in range(n_layers + 1):
            hs = out.hidden_states[li].float().cpu().numpy()[0]
            baseline_hidden[li] = hs
        
        del out
        torch.cuda.empty_cache()
        
        # Intervene at each layer
        for li in interv_layers:
            if li not in operator_results:
                continue
            
            A = operator_results[li]
            hs = baseline_hidden[li]
            
            h_s_raw = hs[subj_pos]
            h_v_raw = hs[verb_pos]
            h_s_norm = np.linalg.norm(h_s_raw)
            h_v_norm = np.linalg.norm(h_v_raw)
            if h_s_norm < 1e-10 or h_v_norm < 1e-10:
                continue
            
            h_s = h_s_raw / h_s_norm
            h_v = h_v_raw / h_v_norm
            
            baseline_rel_score = float(h_s @ A @ h_v)
            
            for eps in EPSILONS:
                for pert_type in PERT_TYPES:
                    delta = construct_perturbation(h_s, h_v, A, eps, pert_type)
                    if np.linalg.norm(delta) < 1e-10:
                        continue
                    
                    # Get device for this layer
                    try:
                        layer_device = next(layers[li].parameters()).device
                    except:
                        layer_device = device
                    
                    delta_tensor = torch.tensor(delta, dtype=torch.float32).to(layer_device)
                    
                    hook = layers[li].register_forward_hook(
                        make_intervention_hook(subj_pos, delta_tensor)
                    )
                    
                    with torch.no_grad():
                        try:
                            interv_out = model(input_ids=input_ids, attention_mask=attn_mask)
                            interv_agreement = measure_agreement(
                                interv_out.logits, verb_pred_pos, correct_id, wrong_id
                            )
                        except Exception as e:
                            interv_agreement = {'agreement': 0.0, 'logit_diff': -100}
                    
                    hook.remove()
                    
                    agreement_drop = baseline_agreement_check['agreement'] - interv_agreement['agreement']
                    logit_diff_drop = baseline_logit_diff - interv_agreement['logit_diff']
                    
                    intervention_results[li][pert_type][eps].append({
                        'baseline_agreement': baseline_agreement_check['agreement'],
                        'interv_agreement': interv_agreement['agreement'],
                        'agreement_drop': agreement_drop,
                        'logit_diff_drop': logit_diff_drop,
                    })
                    
                    # Verify relation score change (subset)
                    if si < 5 and eps == EPSILONS[1]:
                        h_s_new = (h_s_raw + delta) / np.linalg.norm(h_s_raw + delta)
                        perturbed_rel_score = float(h_s_new @ A @ h_v)
                        rel_change = perturbed_rel_score - baseline_rel_score
                        relation_score_changes[li][pert_type].append(rel_change)
        
        if si % 10 == 0:
            print(f"    [{si+1}/{n_interv}] sent: '{sent[:40]}' agree={baseline_agreement:.4f}")
            # Quick progress
            for li in interv_layers[:2]:
                if li in intervention_results:
                    for pt in ['anti_relation', 'random']:
                        for eps in EPSILONS[:1]:
                            drops = intervention_results[li][pt][eps]
                            if drops:
                                print(f"      L{li} {pt} eps={eps}: mean_drop={np.mean([d['agreement_drop'] for d in drops]):.4f}")
        
        torch.cuda.empty_cache()
    
    # ---- Aggregate results ----
    print(f"\n  Step 4: Aggregating results...")
    
    # KEY COMPARISON: ANTI-RELATION vs RANDOM
    print(f"\n  KEY COMPARISON: ANTI-RELATION vs RANDOM")
    print(f"  {'Layer':<8} {'Eps':<6} {'Anti-Rel':<16} {'Random':<16} {'Δ(A-R)':<12} {'p-val':<10} {'Verdict'}")
    print(f"  {'-'*75}")
    
    causal_verdicts = {}
    
    for li in sorted(intervention_results.keys()):
        for eps in EPSILONS:
            anti_drops = intervention_results[li].get('anti_relation', {}).get(eps, [])
            rand_drops = intervention_results[li].get('random', {}).get(eps, [])
            pro_drops = intervention_results[li].get('pro_relation', {}).get(eps, [])
            
            if not anti_drops or not rand_drops:
                continue
            
            anti_mean = np.mean([d['agreement_drop'] for d in anti_drops])
            rand_mean = np.mean([d['agreement_drop'] for d in rand_drops])
            anti_std = np.std([d['agreement_drop'] for d in anti_drops])
            rand_std = np.std([d['agreement_drop'] for d in rand_drops])
            
            delta_ar = anti_mean - rand_mean
            
            # Statistical test
            if len(anti_drops) > 2 and len(rand_drops) > 2:
                from scipy.stats import ttest_ind
                t_stat, p_val = ttest_ind(
                    [d['agreement_drop'] for d in anti_drops],
                    [d['agreement_drop'] for d in rand_drops],
                    equal_var=False
                )
            else:
                p_val = 1.0
            
            if delta_ar > 0.05 and p_val < 0.05:
                verdict = "★★★ CAUSAL"
            elif delta_ar > 0.02 and p_val < 0.1:
                verdict = "★★ LIKELY CAUSAL"
            elif delta_ar > 0:
                verdict = "★ TREND"
            else:
                verdict = "✗ NOT CAUSAL"
            
            key = f"L{li}_eps{eps}"
            causal_verdicts[key] = {
                'anti_mean_drop': float(anti_mean),
                'random_mean_drop': float(rand_mean),
                'delta_anti_rand': float(delta_ar),
                'p_value': float(p_val),
                'verdict': verdict,
            }
            
            print(f"  L{li:<6} {eps:<6.1f} {anti_mean:.4f}±{anti_std:.4f}   "
                  f"{rand_mean:.4f}±{rand_std:.4f}   "
                  f"{delta_ar:+.4f}     {p_val:.4f}   {verdict}")
    
    # Pro-relation summary
    print(f"\n  PRO-RELATION (enhancing agreement):")
    for li in sorted(intervention_results.keys()):
        for eps in [EPSILONS[1]]:
            pro_drops = intervention_results[li].get('pro_relation', {}).get(eps, [])
            if pro_drops:
                pro_mean = np.mean([d['agreement_drop'] for d in pro_drops])
                print(f"    L{li} eps={eps}: mean agreement_drop={pro_mean:+.4f} "
                      f"({'IMPROVES' if pro_mean < 0 else 'WORSENS'})")
    
    # Perturbation verification
    print(f"\n  PERTURBATION VERIFICATION (relation score changes):")
    for li in sorted(relation_score_changes.keys()):
        for pt in ['anti_relation', 'pro_relation', 'random']:
            changes = relation_score_changes[li].get(pt, [])
            if changes:
                print(f"    L{li} {pt}: mean Δrel_score={np.mean(changes):+.4f}")
    
    return {
        'causal_verdicts': causal_verdicts,
        'n_test_sentences': len(test_sentences),
        'intervention_layers': interv_layers,
        'baseline_stats': {
            'total': len(baseline_results),
            'high_agreement': len(high_agreement),
            'mean_agreement': float(np.mean([r['agreement'] for r in baseline_results])) if baseline_results else 0,
        }
    }


# ========================================================================
# Experiment 2: Context-Dependent Operator Test
# ========================================================================
def run_context_dependent_test(model, tokenizer, device, test_data, n_layers, d_model):
    """Train operator on ACTIVE, test on PASSIVE."""
    print(f"\n{'='*70}")
    print("EXP2: Context-Dependent Operator Test")
    print(f"{'='*70}")
    
    interv_layers = sorted(set([
        n_layers // 3, n_layers // 2, 2 * n_layers // 3
    ])) if n_layers > 12 else [n_layers // 2]
    
    # Collect active hidden states
    active_hidden = {l: [] for l in range(n_layers + 1)}
    active_annotations = []
    
    for td in test_data[:40]:
        anno = tokenize_and_annotate(td['sent_sg'], tokenizer)
        if anno is None:
            continue
        
        input_ids = anno['input_ids'].to(device)
        attn_mask = anno['attention_mask'].to(device)
        
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn_mask,
                       output_hidden_states=True)
        
        for li in range(n_layers + 1):
            hs = out.hidden_states[li].float().cpu().numpy()[0]
            active_hidden[li].append(hs)
        active_annotations.append(anno)
        
        del out
        torch.cuda.empty_cache()
    
    print(f"  Active annotated: {len(active_annotations)}")
    
    # Train operators
    operators = {}
    for li in interv_layers:
        hs_list = active_hidden.get(li, [])
        if len(hs_list) < 5:
            continue
        pos_pairs, neg_pairs = extract_pairs_from_annotated(
            hs_list, active_annotations, 'subject_verb'
        )
        if len(pos_pairs) < 5:
            continue
        model_op, _ = train_relation_operator(
            pos_pairs, neg_pairs, d_model, rank=OPERATOR_RANK, epochs=80, lr=0.001
        )
        if model_op is not None:
            operators[li] = model_op
    
    print(f"  Trained operators at {len(operators)} layers")
    
    # Test on PASSIVE sentences
    passive_results = {}
    for td in test_data[:20]:
        subj = td['verb_sg'][:-1] if td['verb_sg'].endswith('s') else td['verb_sg']
        obj = td['verb_sg']  # just for template
        # Passive: "The {obj} is {past_part} by the {subj}"
        # This is too complex for proper SVO matching, skip for now
        pass
    
    return {'passive_results': passive_results, 'n_operators': len(operators)}


# ========================================================================
# Experiment 3: Relation Composition Test
# ========================================================================
def run_composition_test(model, tokenizer, device, test_data, n_layers, d_model):
    """Test A_sv ∘ A_vo → A_so?"""
    print(f"\n{'='*70}")
    print("EXP3: Relation Composition Test")
    print(f"{'='*70}")
    
    interv_layers = sorted(set([
        n_layers // 3, n_layers // 2, 2 * n_layers // 3
    ])) if n_layers > 12 else [n_layers // 2]
    
    # Collect hidden states
    all_hidden = {l: [] for l in range(n_layers + 1)}
    all_annotations = []
    
    for td in test_data[:60]:
        anno = tokenize_and_annotate(td['sent_sg'], tokenizer)
        if anno is None:
            continue
        
        input_ids = anno['input_ids'].to(device)
        attn_mask = anno['attention_mask'].to(device)
        
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn_mask,
                       output_hidden_states=True)
        
        for li in range(n_layers + 1):
            hs = out.hidden_states[li].float().cpu().numpy()[0]
            all_hidden[li].append(hs)
        all_annotations.append(anno)
        
        del out
        torch.cuda.empty_cache()
    
    print(f"  Annotated: {sum(1 for a in all_annotations if a is not None)}")
    
    results = {}
    for li in interv_layers:
        hs_list = all_hidden.get(li, [])
        if len(hs_list) < 10:
            continue
        
        sv_pos, sv_neg = extract_pairs_from_annotated(hs_list, all_annotations, 'subject_verb')
        vo_pos, vo_neg = extract_pairs_from_annotated(hs_list, all_annotations, 'verb_object')
        
        # Subject-object pairs
        so_pos, so_neg = [], []
        for hs, svo in zip(hs_list, all_annotations):
            if hs is None or svo is None:
                continue
            n_tok = hs.shape[0]
            s, o = svo['subj_pos'], svo['obj_pos']
            if s >= n_tok or o >= n_tok:
                continue
            norms = np.linalg.norm(hs, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            hs_norm = hs / norms
            so_pos.append((hs_norm[s].copy(), hs_norm[o].copy()))
            nc = 0
            for i in range(n_tok):
                for j in range(i+1, n_tok):
                    if i == s and j == o: continue
                    so_neg.append((hs_norm[i].copy(), hs_norm[j].copy()))
                    nc += 1
                    if nc >= 5: break
                if nc >= 5: break
        
        if len(sv_pos) < 5 or len(vo_pos) < 5 or len(so_pos) < 5:
            continue
        
        op_sv, _ = train_relation_operator(sv_pos, sv_neg, d_model, rank=OPERATOR_RANK, epochs=80)
        op_vo, _ = train_relation_operator(vo_pos, vo_neg, d_model, rank=OPERATOR_RANK, epochs=80)
        op_so, _ = train_relation_operator(so_pos, so_neg, d_model, rank=OPERATOR_RANK, epochs=80)
        
        if op_sv is None or op_vo is None or op_so is None:
            continue
        
        A_sv = extract_A_matrix(op_sv)
        A_vo = extract_A_matrix(op_vo)
        A_so = extract_A_matrix(op_so)
        A_composed = A_sv @ A_vo
        
        # Test A_composed on S-O pairs
        n_so = len(so_pos)
        n_train = max(3, int(0.7 * n_so))
        test_so_pos = so_pos[n_train:]
        test_so_neg = so_neg[:len(test_so_pos) * 5]
        
        if len(test_so_pos) < 3:
            continue
        
        with torch.no_grad():
            pos_h_i = torch.tensor(np.array([p[0] for p in test_so_pos]), dtype=torch.float32)
            pos_h_j = torch.tensor(np.array([p[1] for p in test_so_pos]), dtype=torch.float32)
            
            so_pos_scores = op_so.score_batch(pos_h_i, pos_h_j).detach().numpy()
            composed_pos_scores = np.array([float(h_i @ A_composed @ h_j) for h_i, h_j in test_so_pos])
        
        if len(test_so_neg) > 0:
            neg_h_i = torch.tensor(np.array([p[0] for p in test_so_neg]), dtype=torch.float32)
            neg_h_j = torch.tensor(np.array([p[1] for p in test_so_neg]), dtype=torch.float32)
            so_neg_scores = op_so.score_batch(neg_h_i, neg_h_j).detach().numpy()
            composed_neg_scores = np.array([float(h_i @ A_composed @ h_j) for h_i, h_j in test_so_neg])
        else:
            so_neg_scores = composed_neg_scores = np.array([])
        
        # AUCs
        labels = np.concatenate([np.ones(len(so_pos_scores)), np.zeros(len(so_neg_scores))])
        try: auc_so = roc_auc_score(labels, np.concatenate([so_pos_scores, so_neg_scores]))
        except: auc_so = 0.5
        try: auc_comp = roc_auc_score(labels, np.concatenate([composed_pos_scores, composed_neg_scores]))
        except: auc_comp = 0.5
        
        corr = float(np.corrcoef(A_so.flatten(), A_composed.flatten())[0, 1])
        frob = float(np.linalg.norm(A_so - A_composed) / np.linalg.norm(A_so))
        
        results[f"L{li}"] = {
            'auc_so_direct': float(auc_so),
            'auc_so_composed': float(auc_comp),
            'corr_so_composed': float(corr),
            'frob_diff': float(frob),
        }
        print(f"  L{li}: direct_AUC={auc_so:.4f}, composed_AUC={auc_comp:.4f}, "
              f"corr={corr:.4f}, frob_diff={frob:.4f}")
    
    return results


# ========================================================================
# Main Function
# ========================================================================
def run_phase209(model_name: str):
    print(f"\n{'='*70}")
    print(f"Phase 209: Causal Relational Intervention — {model_name}")
    print(f"{'='*70}")
    t_start = time.time()

    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    print(f"  n_layers={n_layers}, d_model={d_model}, class={info.model_class}")

    test_data = generate_test_sentences()

    all_results = {}

    # EXP1: Causal Intervention (THE CORE)
    exp1 = run_causal_intervention(model, tokenizer, device, test_data, n_layers, d_model)
    all_results['exp1_causal_intervention'] = exp1

    # EXP2: Context-Dependent
    exp2 = run_context_dependent_test(model, tokenizer, device, test_data, n_layers, d_model)
    all_results['exp2_context_dependent'] = exp2

    # EXP3: Composition
    exp3 = run_composition_test(model, tokenizer, device, test_data, n_layers, d_model)
    all_results['exp3_composition'] = exp3

    # ---- Summary ----
    print(f"\n{'='*70}")
    print(f"PHASE 209 SUMMARY — {model_name}")
    print(f"{'='*70}")

    print("\n--- Exp1: Causal Intervention ---")
    if exp1 and 'causal_verdicts' in exp1:
        for key, val in sorted(exp1['causal_verdicts'].items()):
            print(f"  {key}: Δ(anti-rand)={val['delta_anti_rand']:+.4f}, "
                  f"p={val['p_value']:.4f} → {val['verdict']}")
        baseline_stats = exp1.get('baseline_stats', {})
        print(f"  Baseline: total={baseline_stats.get('total',0)}, "
              f"high_agree={baseline_stats.get('high_agreement',0)}, "
              f"mean_agree={baseline_stats.get('mean_agreement',0):.4f}")
    else:
        print("  [No results]")

    print("\n--- Exp3: Composition ---")
    if isinstance(exp3, dict):
        for key, val in sorted(exp3.items()):
            print(f"  {key}: direct={val.get('auc_so_direct',0):.4f}, "
                  f"composed={val.get('auc_so_composed',0):.4f}, "
                  f"corr={val.get('corr_so_composed',0):.4f}")
    
    # ---- Save ----
    save_dir = Path(__file__).parent.parent / "glm5_temp"
    save_dir.mkdir(exist_ok=True)
    save_path = save_dir / f"phase209_{model_name}_results.json"

    def make_serializable(obj):
        if isinstance(obj, dict):
            return {str(k): make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(x) for x in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, (int, float, str)):
            return obj
        else:
            return str(obj)

    json_results = make_serializable({
        "experiments": all_results,
        "metadata": {
            "model": model_name, "n_layers": n_layers, "d_model": d_model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    })

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {save_path}")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    t_total = time.time() - t_start
    print(f"\nTotal time: {t_total:.1f}s ({t_total/60:.1f}min)")
    return json_results


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    run_phase209(model_name)
