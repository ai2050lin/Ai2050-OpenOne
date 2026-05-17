"""
Phase 207: Relational Invariants — Semantic vs. Architectural Correlations
===========================================================================

CORE QUESTION (项目最关键实验):
  Are deep-layer token-pair relations REAL SEMANTIC RELATIONS,
  or just TOKEN COLLAPSE (representation homogenization)?

KEY METRIC:
  RELATION SPECIFICITY = cos(specific pair) - cos(random pair)

  If specificity ≈ 0  → token collapse (architectural correlation)
  If specificity >> 0  → real semantic relations
  If specificity << 0  → anti-relations (tokens actively distinguished)

THEORETICAL CONTEXT:
  Phase 206 found cos(h_i, h_j) → 0.6-0.9 in deep layers.
  BUT: this could be because ALL tokens become similar (LayerNorm + residual),
  NOT because specific tokens are bound by semantic relations.

  The ONLY way to distinguish: compare specific pairs against RANDOM baselines.

EXPERIMENTS:
  Exp1: Random Baseline Control (MOST CRITICAL — 决定项目走向)
    - Compare: adjacent, subject_verb, det_noun, verb_object, long_distance
    - Against: random_within (same sentence, same distance, no syntactic relation)
    - Against: random_cross (different sentences, same layer)
    - Key output: relation_specificity at each layer

  Exp2: Residual Relation Geometry (去除全局塌缩分量)
    - L2-normalize tokens → remove norm effect
    - Remove the GLOBAL DIRECTION (mean of normalized tokens) → remove collapse
    - Recompute specific vs random pairs
    - If specificity INCREASES after removing global → collapse was hiding semantics
    - If specificity DECREASES → relations were driven by collapse

  Exp3: Cross-Sentence Relation (关系是句子内构建还是词级别泛化?)
    - Within-sentence: cos(h_A[subject], h_A[verb]) at layer l
    - Cross-sentence same-role: cos(h_A[subject], h_B[subject]) at layer l
    - Cross-sentence diff-role: cos(h_A[subject], h_B[verb]) at layer l
    - If within >> cross → relations are sentence-constructed (supporting constraint-field)
    - If same-role >> within → word-level representations dominate (supporting vector coding)

  Exp4: Relation Specificity Curve (特异性随深度的演化)
    - Plot specificity(syntactic) - specificity(random) across layers
    - Identify "semantic construction" phase
    - Compare across models and modes

  Exp5: Low-Rank Relation Field (关系矩阵是否低秩?)
    - Compute full cosine similarity matrix C for each sentence at each layer
    - SVD decompose C → study rank structure
    - If rank(C) is low → supports constraint-field hypothesis R = H^T A H
    - Track rank evolution across layers

DATA: 80 sentences × 3 modes (normal, cot, translation)
MODELS: Qwen3, GLM4, DS7B (bf16 + device_map="auto", NO 8-bit)
LOAD: Reference tests/model_demo_bf16.py for loading pattern
"""

import sys, os
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent.parent / "tests"))

import gc, time, json, math, warnings
import numpy as np
import torch
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from model_utils import (get_model_info, release_model, get_layers, get_W_U,
                          MODEL_CONFIGS)

warnings.filterwarnings('ignore')

LITE = os.environ.get('LITE', '0') == '1'  # Default: FULL data for Phase 207

# ========================================================================
# Sentence Data — 80 SVO sentences with clear syntactic structure
# Pattern: "The [subject] [verb] the [object]" (6 tokens)
# ========================================================================
BASE_SENTENCES = [
    "The cat chases the dog",
    "The teacher helps the student",
    "The leader guides the team",
    "The doctor treats the patient",
    "The chef cooks the meal",
    "The writer drafts the letter",
    "The farmer plants the seed",
    "The artist paints the portrait",
    "The scientist discovers the element",
    "The engineer designs the bridge",
    "The judge delivers the verdict",
    "The soldier defends the fortress",
    "The musician composes the symphony",
    "The pilot flies the airplane",
    "The author writes the novel",
    "The builder constructs the house",
    "The driver operates the vehicle",
    "The hunter tracks the animal",
    "The swimmer crosses the river",
    "The climber reaches the summit",
    "The baker prepares the bread",
    "The tailor makes the garment",
    "The gardener grows the flowers",
    "The fisherman catches the fish",
    "The librarian organizes the books",
    "The mechanic repairs the engine",
    "The programmer writes the code",
    "The analyst studies the data",
    "The manager oversees the project",
    "The director produces the film",
    "The philosopher questions the assumption",
    "The historian examines the evidence",
    "The linguist analyzes the grammar",
    "The mathematician proves the theorem",
    "The physicist tests the hypothesis",
    "The chemist synthesizes the compound",
    "The biologist observes the organism",
    "The geologist studies the rock",
    "The astronomer observes the star",
    "The meteorologist predicts the weather",
    "The economist models the market",
    "The psychologist studies the mind",
    "The sociologist examines the culture",
    "The anthropologist studies the tradition",
    "The architect designs the building",
    "The surveyor measures the land",
    "The technician calibrates the instrument",
    "The inspector checks the quality",
    "The auditor reviews the accounts",
    "The consultant advises the client",
    "The mediator resolves the conflict",
    "The negotiator reaches the agreement",
    "The coordinator manages the schedule",
    "The supervisor monitors the progress",
    "The trainer teaches the skill",
    "The mentor guides the protege",
    "The volunteer helps the community",
    "The researcher explores the frontier",
    "The pioneer discovers the territory",
    # 20 additional sentences for Phase 207 (increased data)
    "The captain commands the vessel",
    "The governor rules the province",
    "The priest blesses the congregation",
    "The knight protects the kingdom",
    "The merchant trades the goods",
    "The inventor creates the device",
    "The critic reviews the performance",
    "The curator arranges the exhibition",
    "The referee enforces the rules",
    "The detective solves the mystery",
    "The surgeon performs the operation",
    "The therapist treats the condition",
    "The professor explains the concept",
    "The student learns the material",
    "The worker builds the structure",
    "The athlete wins the competition",
    "The politician debates the policy",
    "The journalist reports the news",
    "The editor revises the manuscript",
    "The poet composes the verse",
]

COT_PROMPTS = [
    f"Let's think step by step. {s}" for s in BASE_SENTENCES
]

TRANSLATION_SENTENCES = [
    "Le chat chase le chien",
    "Le professeur aide l'étudiant",
    "Le leader guide l'équipe",
    "Le médecin traite le patient",
    "Le chef prépare le repas",
    "L'écrivain rédige la lettre",
    "Le fermier plante la graine",
    "L'artiste peint le portrait",
    "Le scientifique découvre l'élément",
    "L'ingénieur conçoit le pont",
    "Le juge prononce le verdict",
    "Le soldat défend la forteresse",
    "Le musicien compose la symphonie",
    "Le pilote vole l'avion",
    "L'auteur écrit le roman",
    "Le constructeur bâtit la maison",
    "Le conducteur opère le véhicule",
    "Le chasseur traque l'animal",
    "Le nageur traverse la rivière",
    "L'alpiniste atteint le sommet",
    "Le boulanger prépare le pain",
    "Le tailleur fait le vêtement",
    "Le jardinier cultive les fleurs",
    "Le pêcheur attrape le poisson",
    "Le bibliothécaire organise les livres",
    "Le mécanicien répare le moteur",
    "Le programmeur écrit le code",
    "L'analyste étudie les données",
    "Le gestionnaire supervise le projet",
    "Le directeur produit le film",
    "Le philosophe questionne l'hypothèse",
    "L'historien examine la preuve",
    "Le linguiste analyse la grammaire",
    "Le mathématicien prouve le théorème",
    "Le physicien teste l'hypothèse",
    "Le chimiste synthétise le composé",
    "Le biologiste observe l'organisme",
    "Le géologue étudie la roche",
    "L'astronome observe l'étoile",
    "Le météorologue prédit le temps",
    "L'économiste modélise le marché",
    "Le psychologue étudie l'esprit",
    "Le sociologue examine la culture",
    "L'anthropologue étudie la tradition",
    "L'architecte conçoit le bâtiment",
    "Le géomètre mesure le terrain",
    "Le technicien calibre l'instrument",
    "L'inspecteur vérifie la qualité",
    "L'auditeur examine les comptes",
    "Le conseiller advise le client",
    "Le médiateur résout le conflit",
    "Le négociateur atteint l'accord",
    "Le coordonnateur gère le planning",
    "Le superviseur suit les progrès",
    "Le formateur enseigne la compétence",
    "Le mentor guide le protégé",
    "Le bénévole aide la communauté",
    "Le chercheur explore la frontière",
    "Le pionnier découvre le territoire",
    "Le capitaine commande le vaisseau",
    "Le gouverneur dirige la province",
    "Le prêtre bénit la congrégation",
    "Le chevalier protège le royaume",
    "Le marchand échange les marchandises",
    "L'inventeur crée le dispositif",
    "Le critique examine la performance",
    "Le conservateur arrange l'exposition",
    "L'arbitre applique les règles",
    "Le détective résout le mystère",
    "Le chirurgien effectue l'opération",
    "Le thérapeute traite la condition",
    "Le professeur explique le concept",
    "L'étudiant apprend le matériel",
    "L'ouvrier construit la structure",
    "L'athlète gagne la compétition",
    "Le politicien débat la politique",
    "Le journaliste rapporte les nouvelles",
    "L'éditeur révise le manuscrit",
    "Le poète compose le vers",
]


# ========================================================================
# Model Loading (bf16 + device_map="auto", following model_demo_bf16.py)
# ========================================================================
def load_model_bf16(model_name: str):
    """BF16加载模型 — 所有模型均用bfloat16 + device_map="auto" (NO 8-bit)"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = MODEL_CONFIGS[model_name]
    print(f"[load] Loading {model_name} (bfloat16 + device_map=auto)...")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Try sdpa (flash-like) first for memory efficiency, then eager as fallback
    model = None
    for attn_impl in ["sdpa", "eager"]:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                cfg["path"],
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                local_files_only=True,
                attn_implementation=attn_impl,
            )
            print(f"[load] Using attn_implementation={attn_impl}")
            break
        except Exception as e:
            print(f"[load] attn_implementation={attn_impl} failed: {e}")
            model = None
            continue

    if model is None:
        raise RuntimeError(f"Failed to load {model_name} with any attention implementation")

    model.eval()
    device = next(model.parameters()).device
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0

    if hasattr(model, 'hf_device_map'):
        dmap = model.hf_device_map
        gpu_count = sum(1 for v in dmap.values() if 'cuda' in str(v))
        cpu_count = sum(1 for v in dmap.values() if 'cpu' in str(v))
        print(f"[load] {model_name} loaded: GPU={gpu_count} components, CPU={cpu_count} components, "
              f"GPU mem={gpu_mem:.2f}GB")
    else:
        print(f"[load] {model_name} loaded: device={device}, GPU={gpu_mem:.2f}GB")

    return model, tokenizer, device


# ========================================================================
# Syntactic Annotation — Find SVO pattern in tokenized sentences
# ========================================================================
def find_svo_positions(token_ids, tokenizer):
    """
    Find the SVO pattern "The/a/an X V the/a/an Y" in a tokenized sentence.
    
    Returns dict with {det1, subject, verb, det2, object} positions,
    or None if pattern not found.
    
    Works for both normal and CoT sentences (pattern appears after prefix).
    """
    # Decode each token to lowercase string
    tokens = []
    for tid in token_ids:
        decoded = tokenizer.decode([tid]).strip().lower()
        tokens.append(decoded)
    
    # Search for pattern: "the/a/an/le/la/l'" [word] [word] "the/a/an/le/la/l'" [word]
    # English: "The X verbs the Y"
    # French:  "Le/La X verbs le/la Y"
    determiners = {"the", "a", "an", "le", "la", "l'", "un", "une"}
    
    for i in range(len(tokens) - 4):
        if tokens[i] in determiners:
            # Look for second determiner at position i+3
            if tokens[i+3] in determiners:
                return {
                    'det1': i,
                    'subject': i + 1,
                    'verb': i + 2,
                    'det2': i + 3,
                    'object': i + 4,
                }
    
    return None


# ========================================================================
# Hidden State Collection with Annotation
# ========================================================================
def collect_hidden_states_annotated(model, tokenizer, device, sentences, n_layers, max_len=64):
    """
    Collect hidden states at all layers for all sentences.
    Also annotate syntactic positions for each sentence.
    
    Returns:
        all_hidden: dict {layer_idx: list of [seq_len, d_model] arrays}
        annotations: list of dicts (one per sentence) with syntactic positions
        seq_lengths: list of int
    """
    all_hidden = {l: [] for l in range(n_layers + 1)}
    annotations = []
    seq_lengths = []

    for si, sent in enumerate(sentences):
        if LITE and si >= 20:
            break
        if si % 10 == 0:
            print(f"    Sentence {si+1}/{len(sentences)}: '{sent[:50]}...'")

        toks = tokenizer(sent, return_tensors="pt", truncation=True, max_length=max_len)
        input_ids = toks["input_ids"].to(device)
        attention_mask = toks["attention_mask"].to(device)

        with torch.no_grad():
            try:
                out = model(input_ids=input_ids, attention_mask=attention_mask,
                           output_hidden_states=True)
                for li in range(n_layers + 1):
                    hs = out.hidden_states[li].float().cpu().numpy()  # [1, seq_len, d]
                    all_hidden[li].append(hs[0])  # [seq_len, d]
                seq_lengths.append(hs.shape[1])
            except Exception as e:
                print(f"    [WARN] Forward failed for sentence {si}: {e}")
                # Add None placeholders
                for li in range(n_layers + 1):
                    all_hidden[li].append(None)
                seq_lengths.append(0)
                annotations.append(None)
                continue

        # Annotate syntactic positions
        input_ids_np = input_ids[0].cpu().numpy()
        svo = find_svo_positions(input_ids_np, tokenizer)
        annotations.append(svo)

        if svo is None and si < 3:
            # Debug: show tokenization for first few failed sentences
            tok_strs = [tokenizer.decode([tid]) for tid in input_ids_np]
            print(f"    [DEBUG] Sentence {si} no SVO found: {tok_strs}")

        # Periodic memory cleanup
        if si % 20 == 0:
            torch.cuda.empty_cache()

    return all_hidden, annotations, seq_lengths


# ========================================================================
# Core Computation: Pairwise Relations
# ========================================================================
def compute_all_pair_relations(hs, svo=None):
    """
    Compute ALL pairwise cosine similarities for a single sentence at one layer.
    
    Args:
        hs: [seq_len, d_model] hidden states (raw, not normalized)
        svo: dict with {det1, subject, verb, det2, object} positions, or None
    
    Returns:
        pair_data: dict with keys 'all_pairs', 'by_type', 'by_distance'
    """
    n_tok = hs.shape[0]
    if n_tok < 2:
        return None

    # L2 normalize
    norms = np.linalg.norm(hs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    hs_norm = hs / norms

    # Full cosine similarity matrix
    cos_matrix = hs_norm @ hs_norm.T  # [n_tok, n_tok]

    # Collect all pairs (upper triangle)
    all_pairs = []  # list of (i, j, cos_sim, distance, pair_type)
    by_type = defaultdict(list)
    by_distance = defaultdict(list)

    for i in range(n_tok):
        for j in range(i + 1, n_tok):
            dist = j - i
            cos_val = float(cos_matrix[i, j])
            all_pairs.append((i, j, cos_val, dist))
            by_distance[dist].append(cos_val)

    # Classify pairs by syntactic type if SVO annotation available
    if svo is not None:
        for i, j, cos_val, dist in all_pairs:
            pair_type = classify_pair(i, j, svo, n_tok)
            by_type[pair_type].append(cos_val)

    # Also add "random_within" = all pairs that are NOT syntactically annotated
    if svo is not None:
        syntactic_pairs = set()
        for pair_type in ['subject_verb', 'verb_object', 'det_noun', 'adjacent', 'long_distance']:
            for val in by_type.get(pair_type, []):
                pass  # We identify by pair_type, not by exact indices
        
        # Simpler approach: random = all pairs minus specific syntactic types
        # We'll compute random_baseline as: all pairs NOT in subject_verb, verb_object, det_noun
        specific_indices = set()
        s, v, o = svo['subject'], svo['verb'], svo['object']
        d1, d2 = svo['det1'], svo['det2']
        
        # Mark specific syntactic pairs
        specific_indices.add((min(s,v), max(s,v)))    # subject_verb
        specific_indices.add((min(v,o), max(v,o)))    # verb_object
        specific_indices.add((min(d1,s), max(d1,s)))  # det_noun (first)
        specific_indices.add((min(d2,o), max(d2,o)))  # det_noun (second)
        
        # Random = all pairs NOT in specific set
        random_cos = []
        for i, j, cos_val, dist in all_pairs:
            if (i, j) not in specific_indices:
                random_cos.append(cos_val)
        by_type['random_within'] = random_cos

    return {
        'all_pairs': all_pairs,
        'by_type': dict(by_type),
        'by_distance': dict(by_distance),
        'cos_matrix': cos_matrix,
    }


def classify_pair(i, j, svo, n_tok):
    """Classify a token pair into syntactic type."""
    s, v, o = svo['subject'], svo['verb'], svo['object']
    d1, d2 = svo['det1'], svo['det2']
    
    pair = (min(i, j), max(i, j))
    
    # Subject-verb
    if pair == (min(s, v), max(s, v)):
        return 'subject_verb'
    # Verb-object
    if pair == (min(v, o), max(v, o)):
        return 'verb_object'
    # Determiner-noun
    if pair == (min(d1, s), max(d1, s)) or pair == (min(d2, o), max(d2, o)):
        return 'det_noun'
    # Adjacent (consecutive positions)
    if abs(i - j) == 1:
        return 'adjacent'
    # Long distance (first vs last token)
    if (i == 0 and j == n_tok - 1) or (j == 0 and i == n_tok - 1):
        return 'long_distance'
    
    return 'other'


# ========================================================================
# Exp1: Random Baseline Control (MOST CRITICAL)
# ========================================================================
def run_exp1_random_baseline(all_hidden, annotations, n_layers, sample_layers):
    """
    Exp1: Compare specific syntactic pairs against random baselines.
    
    Key output: relation_specificity = cos(specific) - cos(random)
    """
    print(f"\n{'='*70}")
    print("Exp1: Random Baseline Control (MOST CRITICAL)")
    print(f"{'='*70}")

    results = {}
    pair_types = ['subject_verb', 'verb_object', 'det_noun', 'adjacent', 'long_distance', 'random_within']

    for li in sample_layers:
        hs_list = all_hidden.get(li, [])
        if not hs_list:
            continue

        # Collect pair cosines across all sentences
        type_cosines = defaultdict(list)

        for si, hs in enumerate(hs_list):
            if hs is None:
                continue
            svo = annotations[si] if si < len(annotations) else None

            pair_data = compute_all_pair_relations(hs, svo)
            if pair_data is None:
                continue

            for pt in pair_types:
                if pt in pair_data['by_type'] and pair_data['by_type'][pt]:
                    type_cosines[pt].extend(pair_data['by_type'][pt])

        # Compute averages and specificities
        layer_result = {}
        avg_by_type = {}
        for pt in pair_types:
            if pt in type_cosines and type_cosines[pt]:
                avg_by_type[pt] = float(np.mean(type_cosines[pt]))
                layer_result[pt] = {
                    'mean': float(np.mean(type_cosines[pt])),
                    'std': float(np.std(type_cosines[pt])),
                    'count': len(type_cosines[pt]),
                }
            else:
                avg_by_type[pt] = 0.0
                layer_result[pt] = {'mean': 0.0, 'std': 0.0, 'count': 0}

        # Compute relation specificity = cos(specific) - cos(random_within)
        random_baseline = avg_by_type.get('random_within', 0.0)
        for pt in ['subject_verb', 'verb_object', 'det_noun', 'adjacent', 'long_distance']:
            specificity = avg_by_type.get(pt, 0.0) - random_baseline
            layer_result[f'{pt}_specificity'] = float(specificity)

        results[li] = layer_result

        # Print summary for key layers
        if li in [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]:
            print(f"\n  Layer {li}:")
            for pt in pair_types:
                if pt in layer_result and layer_result[pt]['count'] > 0:
                    print(f"    {pt:>15}: {layer_result[pt]['mean']:.4f} "
                          f"(n={layer_result[pt]['count']})")
            for pt in ['subject_verb', 'verb_object', 'det_noun']:
                spec_key = f'{pt}_specificity'
                if spec_key in layer_result:
                    spec = layer_result[spec_key]
                    marker = "★★★" if spec > 0.05 else ("★★" if spec > 0.02 else ("★" if spec > 0 else "✗"))
                    print(f"    {pt:>15} SPECIFICITY: {spec:+.4f} {marker}")

    return results


# ========================================================================
# Exp2: Residual Relation Geometry (去除全局塌缩分量)
# ========================================================================
def run_exp2_residual_relations(all_hidden, annotations, n_layers, sample_layers):
    """
    Exp2: Remove the global direction (mean of L2-normalized tokens) and
    recompute pair relations. This tests whether relations survive after
    removing the token collapse component.
    
    Procedure:
    1. L2-normalize all tokens: h̃_i = h_i / ||h_i||
    2. Compute global direction: μ = mean(h̃_i)
    3. Remove global direction: r_i = h̃_i - (h̃_i · μ̂) μ̂  (where μ̂ = μ/||μ||)
    4. Re-normalize: r̃_i = r_i / ||r_i||
    5. Compute cos(r̃_i, r̃_j) for specific vs random pairs
    """
    print(f"\n{'='*70}")
    print("Exp2: Residual Relation Geometry (Remove Global Collapse)")
    print(f"{'='*70}")

    results = {}
    pair_types = ['subject_verb', 'verb_object', 'det_noun', 'adjacent', 'random_within']

    for li in sample_layers:
        hs_list = all_hidden.get(li, [])
        if not hs_list:
            continue

        # Collect residual pair cosines
        type_cosines_raw = defaultdict(list)
        type_cosines_residual = defaultdict(list)

        for si, hs in enumerate(hs_list):
            if hs is None:
                continue
            svo = annotations[si] if si < len(annotations) else None

            n_tok = hs.shape[0]
            if n_tok < 2:
                continue

            # === RAW (L2-normalized only) ===
            norms = np.linalg.norm(hs, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            hs_norm = hs / norms

            # Raw cosine matrix
            cos_raw = hs_norm @ hs_norm.T

            # === RESIDUAL (remove global direction) ===
            # Compute global direction (mean of normalized tokens)
            mu = hs_norm.mean(axis=0)  # [d]
            mu_norm = np.linalg.norm(mu)
            if mu_norm > 1e-10:
                mu_hat = mu / mu_norm
            else:
                mu_hat = np.zeros_like(mu)

            # Project out global direction
            proj_coeffs = hs_norm @ mu_hat  # [n_tok]
            hs_residual = hs_norm - np.outer(proj_coeffs, mu_hat)  # [n_tok, d]

            # Re-normalize residual
            res_norms = np.linalg.norm(hs_residual, axis=1, keepdims=True)
            res_norms = np.maximum(res_norms, 1e-10)
            hs_residual_norm = hs_residual / res_norms

            # Residual cosine matrix
            cos_resid = hs_residual_norm @ hs_residual_norm.T

            # Classify pairs
            if svo is not None:
                s, v, o = svo['subject'], svo['verb'], svo['object']
                d1, d2 = svo['det1'], svo['det2']
                
                specific_pairs = {
                    'subject_verb': (min(s,v), max(s,v)),
                    'verb_object': (min(v,o), max(v,o)),
                    'det_noun_1': (min(d1,s), max(d1,s)),
                    'det_noun_2': (min(d2,o), max(d2,o)),
                }

                # Collect specific pairs
                for pt, (pi, pj) in specific_pairs.items():
                    if pi < n_tok and pj < n_tok:
                        type_cosines_raw[pt.replace('_1','').replace('_2','')].append(float(cos_raw[pi, pj]))
                        type_cosines_residual[pt.replace('_1','').replace('_2','')].append(float(cos_resid[pi, pj]))

                # Collect random pairs (all non-specific)
                specific_set = set(specific_pairs.values())
                for i in range(n_tok):
                    for j in range(i+1, n_tok):
                        if (i, j) not in specific_set:
                            type_cosines_raw['random_within'].append(float(cos_raw[i, j]))
                            type_cosines_residual['random_within'].append(float(cos_resid[i, j]))

                # Also collect adjacent pairs
                for i in range(n_tok - 1):
                    if (i, i+1) not in specific_set:
                        type_cosines_raw['adjacent'].append(float(cos_raw[i, i+1]))
                        type_cosines_residual['adjacent'].append(float(cos_resid[i, i+1]))

        # Compute averages
        layer_result = {}
        for pt in pair_types:
            raw_mean = float(np.mean(type_cosines_raw[pt])) if type_cosines_raw[pt] else 0.0
            resid_mean = float(np.mean(type_cosines_residual[pt])) if type_cosines_residual[pt] else 0.0
            layer_result[pt] = {
                'raw_mean': raw_mean,
                'residual_mean': resid_mean,
                'change': resid_mean - raw_mean,
                'count': len(type_cosines_raw[pt]),
            }

        # Compute specificity change
        random_raw = layer_result.get('random_within', {}).get('raw_mean', 0.0)
        random_resid = layer_result.get('random_within', {}).get('residual_mean', 0.0)

        for pt in ['subject_verb', 'verb_object', 'det_noun']:
            raw_spec = layer_result.get(pt, {}).get('raw_mean', 0.0) - random_raw
            resid_spec = layer_result.get(pt, {}).get('residual_mean', 0.0) - random_resid
            layer_result[f'{pt}_specificity_raw'] = float(raw_spec)
            layer_result[f'{pt}_specificity_residual'] = float(resid_spec)
            layer_result[f'{pt}_specificity_change'] = float(resid_spec - raw_spec)

        results[li] = layer_result

        if li in [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]:
            print(f"\n  Layer {li}:")
            for pt in pair_types:
                r = layer_result.get(pt, {})
                print(f"    {pt:>15}: raw={r.get('raw_mean',0):.4f} → residual={r.get('residual_mean',0):.4f} "
                      f"(Δ={r.get('change',0):+.4f})")
            for pt in ['subject_verb', 'verb_object', 'det_noun']:
                raw_s = layer_result.get(f'{pt}_specificity_raw', 0)
                res_s = layer_result.get(f'{pt}_specificity_residual', 0)
                delta_s = layer_result.get(f'{pt}_specificity_change', 0)
                verdict = "SPECIFICITY ↑" if delta_s > 0.01 else ("COLLAPSE-DRIVEN" if delta_s < -0.01 else "NEUTRAL")
                print(f"    {pt:>15} specificity: raw={raw_s:+.4f} → residual={res_s:+.4f} (Δ={delta_s:+.4f}) [{verdict}]")

    return results


# ========================================================================
# Exp3: Cross-Sentence Relations
# ========================================================================
def run_exp3_cross_sentence(all_hidden, annotations, n_layers, sample_layers):
    """
    Exp3: Compare within-sentence relations vs cross-sentence relations.
    
    Key question: Are relations sentence-constructed or word-level generic?
    
    Comparisons:
    - within_sv: cos(h_A[subject], h_A[verb]) within same sentence
    - cross_same_role: cos(h_A[subject], h_B[subject]) across sentences
    - cross_diff_role: cos(h_A[subject], h_B[verb]) across sentences
    """
    print(f"\n{'='*70}")
    print("Exp3: Cross-Sentence Relation Transfer")
    print(f"{'='*70}")

    results = {}
    target_layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]

    for li in target_layers:
        if li not in sample_layers:
            continue
        hs_list = all_hidden.get(li, [])
        if not hs_list:
            continue

        # Collect specific token representations
        subject_vecs = []  # L2-normalized subject vectors
        verb_vecs = []
        object_vecs = []

        within_sv = []  # within-sentence subject-verb cosine
        within_vo = []  # within-sentence verb-object cosine

        for si, hs in enumerate(hs_list):
            if hs is None:
                continue
            svo = annotations[si] if si < len(annotations) else None
            if svo is None:
                continue

            n_tok = hs.shape[0]
            s, v, o = svo['subject'], svo['verb'], svo['object']

            if s >= n_tok or v >= n_tok or o >= n_tok:
                continue

            # L2 normalize
            norms = np.linalg.norm(hs, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            hs_norm = hs / norms

            subject_vecs.append(hs_norm[s])
            verb_vecs.append(hs_norm[v])
            object_vecs.append(hs_norm[o])

            # Within-sentence cosines
            within_sv.append(float(hs_norm[s] @ hs_norm[v]))
            within_vo.append(float(hs_norm[v] @ hs_norm[o]))

        if len(subject_vecs) < 5:
            print(f"  Layer {li}: Not enough annotated sentences ({len(subject_vecs)})")
            continue

        # Cross-sentence comparisons
        n_sents = len(subject_vecs)
        cross_same_role_ss = []  # subject-subject across sentences
        cross_diff_role_sv = []  # subject(verb role) across sentences

        # Sample cross-sentence pairs (limit to avoid O(n^2))
        n_cross = min(500, n_sents * (n_sents - 1) // 2)
        np.random.seed(42)
        indices_used = set()
        for _ in range(n_cross):
            i = np.random.randint(0, n_sents)
            j = np.random.randint(0, n_sents)
            if i == j:
                continue
            if (i, j) in indices_used:
                continue
            indices_used.add((i, j))

            # Same role: subject_A with subject_B
            cross_same_role_ss.append(float(subject_vecs[i] @ subject_vecs[j]))

            # Different role: subject_A with verb_B
            cross_diff_role_sv.append(float(subject_vecs[i] @ verb_vecs[j]))

        # Also: verb-verb and object-object across sentences
        cross_same_role_vv = []
        cross_same_role_oo = []
        for _ in range(n_cross):
            i = np.random.randint(0, n_sents)
            j = np.random.randint(0, n_sents)
            if i == j:
                continue
            cross_same_role_vv.append(float(verb_vecs[i] @ verb_vecs[j]))
            cross_same_role_oo.append(float(object_vecs[i] @ object_vecs[j]))

        layer_result = {
            'within_subject_verb': {
                'mean': float(np.mean(within_sv)) if within_sv else 0,
                'std': float(np.std(within_sv)) if within_sv else 0,
                'count': len(within_sv),
            },
            'within_verb_object': {
                'mean': float(np.mean(within_vo)) if within_vo else 0,
                'std': float(np.std(within_vo)) if within_vo else 0,
                'count': len(within_vo),
            },
            'cross_subject_subject': {
                'mean': float(np.mean(cross_same_role_ss)) if cross_same_role_ss else 0,
                'std': float(np.std(cross_same_role_ss)) if cross_same_role_ss else 0,
                'count': len(cross_same_role_ss),
            },
            'cross_verb_verb': {
                'mean': float(np.mean(cross_same_role_vv)) if cross_same_role_vv else 0,
                'std': float(np.std(cross_same_role_vv)) if cross_same_role_vv else 0,
                'count': len(cross_same_role_vv),
            },
            'cross_object_object': {
                'mean': float(np.mean(cross_same_role_oo)) if cross_same_role_oo else 0,
                'std': float(np.std(cross_same_role_oo)) if cross_same_role_oo else 0,
                'count': len(cross_same_role_oo),
            },
            'cross_subject_verb': {
                'mean': float(np.mean(cross_diff_role_sv)) if cross_diff_role_sv else 0,
                'std': float(np.std(cross_diff_role_sv)) if cross_diff_role_sv else 0,
                'count': len(cross_diff_role_sv),
            },
        }

        # Key diagnostic: within vs cross comparison
        within_sv_mean = layer_result['within_subject_verb']['mean']
        cross_ss_mean = layer_result['cross_subject_subject']['mean']
        cross_sv_mean = layer_result['cross_subject_verb']['mean']

        # Specificity: within_sentence - cross_different_role
        relation_specificity = within_sv_mean - cross_sv_mean
        # Word specificity: cross_same_role - cross_diff_role
        word_specificity = cross_ss_mean - cross_sv_mean

        layer_result['relation_specificity'] = float(relation_specificity)
        layer_result['word_specificity'] = float(word_specificity)

        # Verdict
        if relation_specificity > 0.05 and relation_specificity > word_specificity:
            verdict = "SENTENCE-CONSTRUCTED RELATIONS (supports constraint-field)"
        elif word_specificity > 0.05 and word_specificity > relation_specificity:
            verdict = "WORD-LEVEL GENERIC (supports vector coding)"
        elif relation_specificity < 0.02 and word_specificity < 0.02:
            verdict = "TOKEN COLLAPSE (no real specificity)"
        else:
            verdict = "MIXED (both word-level and sentence-level)"

        layer_result['verdict'] = verdict

        results[li] = layer_result

        print(f"\n  Layer {li}:")
        print(f"    within S-V:  {within_sv_mean:.4f}")
        print(f"    cross S-S:   {cross_ss_mean:.4f}")
        print(f"    cross S-V:   {cross_sv_mean:.4f}")
        print(f"    relation_specificity (within_SV - cross_SV): {relation_specificity:+.4f}")
        print(f"    word_specificity (cross_SS - cross_SV):      {word_specificity:+.4f}")
        print(f"    VERDICT: {verdict}")

    return results


# ========================================================================
# Exp4: Relation Specificity Curve
# ========================================================================
def run_exp4_specificity_curve(exp1_results, n_layers, sample_layers):
    """
    Exp4: Track how relation specificity evolves across layers.
    This is a summary of Exp1 data, plotted as a curve.
    """
    print(f"\n{'='*70}")
    print("Exp4: Relation Specificity Curve (Across Layers)")
    print(f"{'='*70}")

    results = {}
    specific_types = ['subject_verb', 'verb_object', 'det_noun']

    for pt in specific_types:
        spec_key = f'{pt}_specificity'
        curve = {}
        for li in sorted(sample_layers):
            if li in exp1_results and spec_key in exp1_results[li]:
                curve[li] = exp1_results[li][spec_key]
        results[pt] = curve

        if curve:
            # Find peak specificity
            peak_layer = max(curve, key=curve.get)
            peak_val = curve[peak_layer]
            min_layer = min(curve, key=curve.get)
            min_val = curve[min_layer]
            print(f"  {pt}:")
            print(f"    Peak: L{peak_layer} = {peak_val:+.4f}")
            print(f"    Min:  L{min_layer} = {min_val:+.4f}")
            # Is there a "semantic construction" phase?
            positive_layers = sum(1 for v in curve.values() if v > 0.02)
            print(f"    Positive specificity layers: {positive_layers}/{len(curve)}")

    return results


# ========================================================================
# Exp5: Low-Rank Relation Field
# ========================================================================
def run_exp5_low_rank_relation_field(all_hidden, annotations, n_layers, sample_layers):
    """
    Exp5: Study the rank structure of the full cosine similarity matrix.
    
    For each sentence at each layer:
    1. Compute C = H_norm @ H_norm^T (cosine similarity matrix)
    2. SVD decompose C = U S V^T
    3. Compute effective rank: rank_eff = (Σ σ_i)^2 / Σ(σ_i^2)
    
    If rank_eff is low → supports constraint-field hypothesis R = H^T A H
    (relations are determined by a few latent factors, not independent)
    """
    print(f"\n{'='*70}")
    print("Exp5: Low-Rank Relation Field")
    print(f"{'='*70}")

    results = {}
    target_layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]

    for li in target_layers:
        if li not in sample_layers:
            continue
        hs_list = all_hidden.get(li, [])
        if not hs_list:
            continue

        ranks = []
        singular_value_profiles = []

        for si, hs in enumerate(hs_list):
            if hs is None or hs.shape[0] < 3:
                continue

            # L2 normalize
            norms = np.linalg.norm(hs, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            hs_norm = hs / norms

            # Cosine similarity matrix
            C = hs_norm @ hs_norm.T  # [n_tok, n_tok]

            # SVD
            try:
                svd_vals = np.linalg.svd(C, compute_uv=False)
                svd_vals = np.maximum(svd_vals, 0)

                # Effective rank (participation ratio)
                total = np.sum(svd_vals)
                if total > 1e-10:
                    s_norm = svd_vals / total
                    rank_eff = (np.sum(s_norm))**2 / (np.sum(s_norm**2) + 1e-10)
                else:
                    rank_eff = 0

                ranks.append(rank_eff)
                singular_value_profiles.append(svd_vals.tolist())
            except:
                continue

        if ranks:
            layer_result = {
                'rank_eff_mean': float(np.mean(ranks)),
                'rank_eff_std': float(np.std(ranks)),
                'rank_eff_min': float(np.min(ranks)),
                'rank_eff_max': float(np.max(ranks)),
                'n_sentences': len(ranks),
            }
            results[li] = layer_result
            print(f"  Layer {li}: rank_eff = {np.mean(ranks):.2f} ± {np.std(ranks):.2f} "
                  f"(range: {np.min(ranks):.2f} - {np.max(ranks):.2f}, n={len(ranks)})")

    # Track rank evolution
    if len(results) > 1:
        sorted_layers = sorted(results.keys())
        rank_values = [results[l]['rank_eff_mean'] for l in sorted_layers]
        rank_trend = "INCREASING" if rank_values[-1] > rank_values[0] + 0.5 else \
                     "DECREASING" if rank_values[-1] < rank_values[0] - 0.5 else "STABLE"
        print(f"\n  Rank trend: {rank_trend} (L{sorted_layers[0]}={rank_values[0]:.2f} → "
              f"L{sorted_layers[-1]}={rank_values[-1]:.2f})")

    return results


# ========================================================================
# Main Function
# ========================================================================
def run_phase207(model_name: str):
    print(f"\n{'='*70}")
    print(f"Phase 207: Relational Invariants — {model_name}")
    print(f"{'='*70}")
    t_start = time.time()

    # ---- Load model ----
    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    print(f"  n_layers={n_layers}, d_model={d_model}, class={info.model_class}")

    # ---- Sample layers ----
    if n_layers <= 12:
        sample_layers = list(range(n_layers + 1))
    else:
        step = max(1, n_layers // 12)
        sample_layers = sorted(set(list(range(0, n_layers + 1, step)) + [n_layers]))
    print(f"  Sample layers: {sample_layers}")

    # ---- Collect hidden states for 3 modes ----
    modes = ["normal", "cot", "translation"]
    n_sents = 20 if LITE else 80
    mode_sentences = {
        "normal": BASE_SENTENCES[:n_sents],
        "cot": COT_PROMPTS[:n_sents],
        "translation": TRANSLATION_SENTENCES[:n_sents],
    }

    all_hidden = {}
    all_annotations = {}

    for mode in modes:
        print(f"\n--- Collecting {mode} hidden states ({len(mode_sentences[mode])} sentences) ---")
        hidden, annotations, seq_lens = collect_hidden_states_annotated(
            model, tokenizer, device, mode_sentences[mode], n_layers
        )
        all_hidden[mode] = hidden
        all_annotations[mode] = annotations

        # Count successfully annotated sentences
        n_annotated = sum(1 for a in annotations if a is not None)
        print(f"  Collected {len(seq_lens)} sentences, {n_annotated} with SVO annotation")

    # ---- Release model ----
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("\nModel released.")

    # ====================================================================
    # Run all experiments
    # ====================================================================
    all_results = {}

    for mode in modes:
        print(f"\n{'#'*70}")
        print(f"# Mode: {mode}")
        print(f"{'#'*70}")

        mode_results = {}

        # Exp1: Random Baseline Control
        exp1 = run_exp1_random_baseline(
            all_hidden[mode], all_annotations[mode], n_layers, sample_layers
        )
        mode_results['exp1_random_baseline'] = exp1

        # Exp2: Residual Relation Geometry
        exp2 = run_exp2_residual_relations(
            all_hidden[mode], all_annotations[mode], n_layers, sample_layers
        )
        mode_results['exp2_residual_relations'] = exp2

        # Exp3: Cross-Sentence Relations (normal mode only — needs SVO annotation)
        if mode == "normal":
            exp3 = run_exp3_cross_sentence(
                all_hidden[mode], all_annotations[mode], n_layers, sample_layers
            )
            mode_results['exp3_cross_sentence'] = exp3

        # Exp4: Relation Specificity Curve
        exp4 = run_exp4_specificity_curve(exp1, n_layers, sample_layers)
        mode_results['exp4_specificity_curve'] = exp4

        # Exp5: Low-Rank Relation Field
        exp5 = run_exp5_low_rank_relation_field(
            all_hidden[mode], all_annotations[mode], n_layers, sample_layers
        )
        mode_results['exp5_low_rank_field'] = exp5

        all_results[mode] = mode_results

    # ====================================================================
    # Cross-mode comparison summary
    # ====================================================================
    print(f"\n{'='*70}")
    print("PHASE 207 SUMMARY — Cross-Mode Comparison")
    print(f"{'='*70}")

    # Key metric: relation specificity at deep layers
    deep_layer = n_layers - 1
    mid_layer = n_layers // 2

    print(f"\n--- Relation Specificity at L{deep_layer} (deepest) ---")
    for mode in modes:
        exp1 = all_results[mode].get('exp1_random_baseline', {})
        if deep_layer in exp1:
            for pt in ['subject_verb', 'verb_object', 'det_noun']:
                spec_key = f'{pt}_specificity'
                if spec_key in exp1[deep_layer]:
                    print(f"  {mode} {pt}: specificity={exp1[deep_layer][spec_key]:+.4f}")

    print(f"\n--- Residual Specificity Change at L{deep_layer} ---")
    for mode in modes:
        exp2 = all_results[mode].get('exp2_residual_relations', {})
        if deep_layer in exp2:
            for pt in ['subject_verb', 'verb_object', 'det_noun']:
                change_key = f'{pt}_specificity_change'
                if change_key in exp2[deep_layer]:
                    print(f"  {mode} {pt}: Δ_specificity={exp2[deep_layer][change_key]:+.4f}")

    # Cross-sentence verdict (normal mode only)
    if 'normal' in all_results and 'exp3_cross_sentence' in all_results['normal']:
        print(f"\n--- Cross-Sentence Verdict ---")
        for li in sorted(all_results['normal']['exp3_cross_sentence'].keys()):
            r = all_results['normal']['exp3_cross_sentence'][li]
            if 'verdict' in r:
                print(f"  L{li}: {r['verdict']}")

    # Low-rank field summary
    print(f"\n--- Low-Rank Relation Field ---")
    for mode in modes:
        exp5 = all_results[mode].get('exp5_low_rank_field', {})
        ranks = [exp5[li]['rank_eff_mean'] for li in sorted(exp5.keys())]
        if ranks:
            print(f"  {mode}: rank_eff range = {min(ranks):.2f} - {max(ranks):.2f}")

    # ====================================================================
    # Save results
    # ====================================================================
    save_dir = Path(__file__).parent.parent / "glm5_temp"
    save_dir.mkdir(exist_ok=True)
    save_path = save_dir / f"phase207_{model_name}_results.json"

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
            "model": model_name,
            "n_layers": n_layers,
            "d_model": d_model,
            "lite": LITE,
            "n_sentences": n_sents,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    })

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {save_path}")

    t_total = time.time() - t_start
    print(f"\nTotal time: {t_total:.1f}s ({t_total/60:.1f}min)")
    return json_results


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    run_phase207(model_name)
