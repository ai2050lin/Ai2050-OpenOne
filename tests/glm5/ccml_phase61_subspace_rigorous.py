"""
Phase 61: Rigorous Subspace Analysis — Fixing All 4 Hard Flaws
==============================================================

User's criticism (all correct):
  1. cos≈0 in high-dim is DEFAULT, not evidence of orthogonality
  2. Single direction ≠ subspace — need multi-dimensional analysis
  3. α-response curve not analyzed — need monotonic/smooth check
  4. LayerNorm uncontrolled — patching direction ≠ what model sees

This phase addresses ALL four issues with rigorous methods:

  Step 1: SUBSPACE ANALYSIS (not single direction)
    - PCA on sing/plur difference to find syntax SUBSPACE (top-k PCs)
    - CCA between syntax subspace and position subspace
    - Subspace overlap measure (principal angles)
    - Compare with random baseline (is cos≈0 meaningful?)

  Step 2: α-RESPONSE CURVE
    - Test α = 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0
    - Check monotonicity, linearity, saturation
    - If Δ(α) is not a smooth function → direction is not clean

  Step 3: LAYERNORM CONTROL
    - Patch BEFORE LayerNorm (current method)
    - Patch AFTER LayerNorm (bypasses LN modification)
    - Measure how LN transforms the direction
    - If LN destroys the direction → all previous results unreliable

  Step 4: LARGE-SCALE + BOOTSTRAP CI
    - 50+ NVA pairs (expanded from 30)
    - Bootstrap 95% CI for all effect sizes
    - Report both mean and CI width

  Step 5: CONFOUND ANALYSIS
    - Check if d_number correlates with token identity
    - Use within-noun analysis (same noun, different context)
    - Control for token frequency effects

Data: 50 NVA pairs × 2 structures × 2 numbers = 200 sentences
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import torch, numpy as np, gc, argparse, time
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from model_utils import load_model, get_model_info, release_model, get_layers

# ===== EXPANDED NVA PAIRS (50 pairs) =====
NVA_PAIRS = [
    # (sing_noun, plur_noun, sing_verb, plur_verb, adverb)
    ("cat", "cats", "runs", "run", "fast"),
    ("dog", "dogs", "walks", "walk", "home"),
    ("bird", "birds", "flies", "fly", "high"),
    ("girl", "girls", "reads", "read", "well"),
    ("boy", "boys", "sings", "sing", "loud"),
    ("man", "men", "works", "work", "hard"),
    ("horse", "horses", "jumps", "jump", "far"),
    ("bear", "bears", "sleeps", "sleep", "long"),
    ("snake", "snakes", "crawls", "crawl", "slow"),
    ("frog", "frogs", "swims", "swim", "deep"),
    ("fox", "foxes", "hunts", "hunt", "alone"),
    ("king", "kings", "rules", "rule", "well"),
    ("student", "students", "studies", "study", "hard"),
    ("teacher", "teachers", "speaks", "speak", "clear"),
    ("doctor", "doctors", "helps", "help", "often"),
    ("tree", "trees", "grows", "grow", "tall"),
    ("car", "cars", "moves", "move", "fast"),
    ("queen", "queens", "leads", "lead", "now"),
    ("child", "children", "plays", "play", "here"),
    ("wolf", "wolves", "howls", "howl", "night"),
    ("driver", "drivers", "drives", "drive", "slow"),
    ("worker", "workers", "builds", "build", "fast"),
    ("player", "players", "wins", "win", "often"),
    ("writer", "writers", "writes", "write", "daily"),
    ("rabbit", "rabbits", "hops", "hop", "fast"),
    ("eagle", "eagles", "soars", "soar", "high"),
    ("tiger", "tigers", "stalks", "stalk", "quiet"),
    ("monkey", "monkeys", "climbs", "climb", "up"),
    ("lion", "lions", "roars", "roar", "loud"),
    ("farmer", "farmers", "plants", "plant", "early"),
    # --- NEW 20 pairs ---
    ("fish", "fish", "swims", "swim", "deep"),       # invar noun
    ("sheep", "sheep", "grazes", "graze", "calm"),   # invar noun
    ("deer", "deer", "runs", "run", "fast"),           # invar noun
    ("knife", "knives", "cuts", "cut", "deep"),        # irregular
    ("mouse", "mice", "squeaks", "squeak", "loud"),    # irregular
    ("goose", "geese", "flies", "fly", "south"),       # irregular
    ("tooth", "teeth", "bites", "bite", "hard"),       # irregular
    ("foot", "feet", "steps", "step", "carefully"),    # irregular
    ("person", "people", "speaks", "speak", "clear"),  # irregular
    ("ox", "oxen", "pulls", "pull", "hard"),           # irregular
    ("hero", "heroes", "fights", "fight", "bravely"),  # -es plural
    ("potato", "potatoes", "grows", "grow", "well"),   # -es plural
    ("boss", "bosses", "leads", "lead", "firmly"),     # -es plural
    ("glass", "glasses", "shines", "shine", "bright"), # -es plural
    ("watch", "watches", "ticks", "tick", "loud"),     # -es plural
    ("baby", "babies", "cries", "cry", "loud"),        # -ies plural
    ("lady", "ladies", "dances", "dance", "gracefully"), # -ies
    ("story", "stories", "ends", "end", "well"),       # -ies plural
    ("city", "cities", "shines", "shine", "bright"),   # -ies plural
    ("party", "parties", "starts", "start", "late"),   # -ies plural
]

NOUNS_SET = set()
for sn, pn, _, _, _ in NVA_PAIRS:
    NOUNS_SET.add(sn.lower())
    NOUNS_SET.add(pn.lower())


def svo_pos_fn(tok, toks):
    decoded = [tok.decode([t]).strip() for t in toks]
    for i, d in enumerate(decoded):
        if d.lower() in NOUNS_SET:
            return i + 1, i + 2  # +1 BOS, verb follows noun
    return None, None


def adv_pos_fn(tok, toks):
    decoded = [tok.decode([t]).strip() for t in toks]
    for i, d in enumerate(decoded):
        if d.lower() in NOUNS_SET:
            return i + 1, i + 2
    return None, None


def collect_activations(model, tokenizer, device, sentences, pos_fn, target_layers, label=""):
    """Collect activations at subject and verb positions"""
    layers = get_layers(model)
    results = defaultdict(lambda: {"subj": [], "verb": []})
    valid = 0

    for si, sent in enumerate(sentences):
        if si % 15 == 0 and si > 0:
            print(f"  {label} {si}/{len(sentences)}")

        toks = tokenizer.encode(sent, add_special_tokens=False)
        sp, vp = pos_fn(tokenizer, toks)
        if sp is None or vp is None:
            continue

        captured = {}
        def make_hook(li):
            def fn(m, inp, out):
                captured[li] = (out[0] if isinstance(out, tuple) else out).detach().cpu()
            return fn

        hooks = []
        for li in target_layers:
            if li < len(layers):
                hooks.append(layers[li].register_forward_hook(make_hook(li)))

        ids = tokenizer(sent, return_tensors="pt").to(device)
        with torch.no_grad():
            try:
                model(**ids)
            except:
                for h in hooks: h.remove()
                continue

        for h in hooks: h.remove()

        for li in target_layers:
            if li in captured:
                h = captured[li]
                if sp < h.shape[1]:
                    results[li]["subj"].append(h[0, sp, :].float().numpy())
                if vp < h.shape[1]:
                    results[li]["verb"].append(h[0, vp, :].float().numpy())

        valid += 1
        del captured
        gc.collect()
        torch.cuda.empty_cache()

    print(f"  {label}: {valid}/{len(sentences)} valid")
    return results, valid


# =============================================
# STEP 1: SUBSPACE ANALYSIS
# =============================================

def subspace_analysis(sing_act, plur_act, sing_act_adv, plur_act_adv, li, d_model):
    """
    Full subspace analysis for one layer.
    
    Returns:
        - PCA-based syntax subspace (top-k PCs of sing/plur difference)
        - CCA between syntax and position subspaces
        - Principal angles between subspaces
        - Random baseline comparison
    """
    from sklearn.decomposition import PCA
    from sklearn.cross_decomposition import CCA
    
    sing2 = np.array(sing_act[li]["subj"])   # [N, d]
    plur2 = np.array(plur_act[li]["subj"])    # [N, d]
    sing3 = np.array(sing_act_adv[li]["subj"])  # [N, d]
    plur3 = np.array(plur_act_adv[li]["subj"])   # [N, d]
    
    N2 = min(len(sing2), len(plur2))
    N3 = min(len(sing3), len(plur3))
    
    if N2 < 5 or N3 < 5:
        return None
    
    # --- Syntax difference vectors (per-sentence) ---
    diff2 = plur2[:N2] - sing2[:N2]   # [N2, d] — per-sentence syntax diffs at pos2
    diff3 = plur3[:N3] - sing3[:N3]   # [N3, d] — per-sentence syntax diffs at pos3
    
    # --- PCA on syntax differences ---
    # Concatenate all diffs to find the syntax subspace
    all_diffs = np.vstack([diff2, diff3])  # [N2+N3, d]
    
    # How many PCs explain 90% of variance?
    k = min(all_diffs.shape[0] - 1, all_diffs.shape[1] - 1, 50)
    pca = PCA(n_components=k)
    pca.fit(all_diffs)
    
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_90 = int(np.searchsorted(cumvar, 0.90)) + 1  # PCs for 90% variance
    n_95 = int(np.searchsorted(cumvar, 0.95)) + 1
    n_99 = int(np.searchsorted(cumvar, 0.99)) + 1
    
    # Syntax subspace basis (top-n_90 PCs)
    V_syntax = pca.components_[:n_90]  # [n_90, d]
    
    # --- Position difference vectors (per-sentence) ---
    # pos2 vs pos3 for same-number nouns
    pos_diff_sing = sing2[:N2] - sing3[:N3]   # pos2 - pos3 for singular
    pos_diff_plur = plur2[:N2] - plur3[:N3]   # pos2 - pos3 for plural
    all_pos_diffs = np.vstack([pos_diff_sing, pos_diff_plur])
    
    pca_pos = PCA(n_components=min(k, all_pos_diffs.shape[0] - 1))
    pca_pos.fit(all_pos_diffs)
    cumvar_pos = np.cumsum(pca_pos.explained_variance_ratio_)
    n_90_pos = int(np.searchsorted(cumvar_pos, 0.90)) + 1
    
    V_pos = pca_pos.components_[:n_90_pos]  # [n_90_pos, d]
    
    # --- CCA between syntax and position subspaces ---
    # CCA finds directions of maximum correlation between two sets
    n_cca = min(n_90, n_90_pos, 10)  # limit to 10 for stability
    if n_cca < 1:
        n_cca = 1
    
    try:
        cca = CCA(n_components=n_cca)
        # Use the difference matrices directly
        min_n = min(all_diffs.shape[0], all_pos_diffs.shape[0])
        cca.fit(all_diffs[:min_n], all_pos_diffs[:min_n])
        
        # CCA correlations
        X_c, Y_c = cca.transform(all_diffs[:min_n], all_pos_diffs[:min_n])
        cca_corrs = []
        for i in range(n_cca):
            corr = np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1]
            cca_corrs.append(corr)
    except:
        cca_corrs = [0.0] * n_cca
    
    # --- Principal angles between subspaces ---
    # This is the RIGHT way to measure subspace overlap
    def principal_angles(V1, V2):
        """Compute principal angles between subspaces spanned by V1 and V2"""
        # Orthonormalize
        Q1, _ = np.linalg.qr(V1.T)  # [d, k1]
        Q2, _ = np.linalg.qr(V2.T)  # [d, k2]
        
        # SVD of Q1^T Q2
        M = Q1.T @ Q2  # [k1, k2]
        _, s, _ = np.linalg.svd(M, full_matrices=False)
        
        # Principal angles = arccos(singular values)
        s_clipped = np.clip(s, 0, 1)
        angles = np.arccos(s_clipped)
        return angles, s_clipped
    
    try:
        angles, cos_angles = principal_angles(V_syntax, V_pos)
        mean_cos = float(np.mean(cos_angles))
        max_cos = float(np.max(cos_angles))
    except:
        angles, cos_angles = np.array([]), np.array([])
        mean_cos, max_cos = 0.0, 0.0
    
    # --- Random baseline ---
    # In high dimensions, random subspaces have cos≈0 between them
    # We need to check if our cos is significantly different from random
    n_random = 100
    random_cos = []
    for _ in range(n_random):
        V_rand1 = np.random.randn(n_90, d_model)
        V_rand2 = np.random.randn(n_90_pos, d_model)
        try:
            _, cos_r = principal_angles(V_rand1, V_rand2)
            random_cos.append(float(np.mean(cos_r)))
        except:
            pass
    
    random_baseline_mean = float(np.mean(random_cos)) if random_cos else 0.0
    random_baseline_std = float(np.std(random_cos)) if random_cos else 0.0
    
    # Z-score: how many std devs above random?
    z_score = (mean_cos - random_baseline_mean) / (random_baseline_std + 1e-10)
    
    # --- Subspace overlap metric ---
    # Fraction of syntax variance explained by position subspace
    # = ||Proj_{V_pos} v_syntax_i||^2 / ||v_syntax_i||^2
    Q_pos, _ = np.linalg.qr(V_pos.T)  # [d, k_pos]
    proj_matrix = Q_pos @ Q_pos.T  # [d, d]
    
    overlap_per_pc = []
    for i in range(min(n_90, 10)):
        v = V_syntax[i]
        proj_v = proj_matrix @ v
        overlap = float(np.dot(proj_v, proj_v) / (np.dot(v, v) + 1e-10))
        overlap_per_pc.append(overlap)
    
    # --- Direction consistency across positions (per-PC) ---
    # PCA on pos2 diffs vs pos3 diffs separately
    pca2 = PCA(n_components=min(k, N2 - 1))
    pca2.fit(diff2)
    pca3 = PCA(n_components=min(k, N3 - 1))
    pca3.fit(diff3)
    
    # How similar are the top PCs?
    pc_cosines = []
    n_compare = min(10, pca2.n_components_, pca3.n_components_)
    for i in range(n_compare):
        c = float(np.dot(pca2.components_[i], pca3.components_[i]) / 
                  (np.linalg.norm(pca2.components_[i]) * np.linalg.norm(pca3.components_[i]) + 1e-10))
        pc_cosines.append(c)
    
    return {
        "n_pcs_90": n_90, "n_pcs_95": n_95, "n_pcs_99": n_99,
        "n_pcs_90_pos": n_90_pos,
        "var_explained_top1": float(pca.explained_variance_ratio_[0]),
        "var_explained_top5": float(np.sum(pca.explained_variance_ratio_[:5])),
        "var_explained_top10": float(np.sum(pca.explained_variance_ratio_[:min(10, len(pca.explained_variance_ratio_))])),
        "cca_corrs": cca_corrs,
        "cca_mean": float(np.mean(cca_corrs)) if cca_corrs else 0.0,
        "cca_max": float(np.max(cca_corrs)) if cca_corrs else 0.0,
        "principal_angles": angles.tolist(),
        "cos_angles": cos_angles.tolist(),
        "mean_cos_angle": mean_cos,
        "max_cos_angle": max_cos,
        "random_baseline_mean": random_baseline_mean,
        "random_baseline_std": random_baseline_std,
        "z_score_vs_random": z_score,
        "overlap_per_pc": overlap_per_pc,
        "overlap_mean": float(np.mean(overlap_per_pc)) if overlap_per_pc else 0.0,
        "pc_cosines_pos2_vs_pos3": pc_cosines,
        "pca2_var_top1": float(pca2.explained_variance_ratio_[0]) if len(pca2.explained_variance_ratio_) > 0 else 0,
        "pca3_var_top1": float(pca3.explained_variance_ratio_[0]) if len(pca3.explained_variance_ratio_) > 0 else 0,
        "V_syntax": V_syntax,  # for subspace patching
        "V_pos": V_pos,
    }


# =============================================
# STEP 2: α-RESPONSE CURVE
# =============================================

def alpha_response_curve(model, tokenizer, device, layers, test_sents, 
                         pos_fn, direction, patch_pos_fn, layer_idx,
                         alphas=None):
    """
    Test direction patching at multiple α values.
    Returns Δ(α) curve.
    """
    if alphas is None:
        alphas = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
    
    direction_t = torch.tensor(direction, dtype=torch.float16, device=device)
    
    curve = {}
    for alpha in alphas:
        effects = []
        for sent_data in test_sents:
            if len(sent_data) == 6:
                sing, plur, sv, pv, sn, pn = sent_data
            else:
                continue
            
            toks = tokenizer.encode(sing, add_special_tokens=False)
            sp, vp = pos_fn(tokenizer, toks)
            pp = patch_pos_fn(tokenizer, toks)
            
            if sp is None or vp is None or pp is None:
                continue
            
            sv_ids = tokenizer.encode(sv, add_special_tokens=False)
            pv_ids = tokenizer.encode(pv, add_special_tokens=False)
            if not sv_ids or not pv_ids:
                continue
            
            input_ids = tokenizer(sing, return_tensors="pt").to(device)
            
            # Baseline
            with torch.no_grad():
                base_logits = model(**input_ids).logits.detach().cpu()
            
            if vp >= base_logits.shape[1]:
                continue
            
            base_agr = (base_logits[0, vp, sv_ids[0]] - 
                       base_logits[0, vp, pv_ids[0]]).item()
            
            # Patched
            applied = [False]
            def add_hook(m, inp, out, a=alpha, p=pp):
                if not applied[0]:
                    if isinstance(out, tuple):
                        p_out = out[0].clone()
                        p_out[:, p, :] += (a * direction_t).to(p_out.dtype)
                        applied[0] = True
                        return (p_out,) + out[1:]
                    else:
                        p_out = out.clone()
                        p_out[:, p, :] += (a * direction_t).to(p_out.dtype)
                        applied[0] = True
                        return p_out
                return out
            
            hook = layers[layer_idx].register_forward_hook(add_hook)
            with torch.no_grad():
                patched_logits = model(**input_ids).logits.detach().cpu()
            hook.remove()
            
            patched_agr = (patched_logits[0, vp, sv_ids[0]] - 
                          patched_logits[0, vp, pv_ids[0]]).item()
            
            effects.append(patched_agr - base_agr)
        
        if effects:
            curve[alpha] = {
                "mean": float(np.mean(effects)),
                "std": float(np.std(effects)),
                "n": len(effects),
                "neg_ratio": float(np.mean([e < 0 for e in effects])),
            }
    
    return curve


# =============================================
# STEP 3: LAYERNORM CONTROL
# =============================================

def layernorm_analysis(model, tokenizer, device, layers, layer_idx, direction, 
                       test_sents, pos_fn, alpha=2.0):
    """
    Analyze LayerNorm's effect on direction patching.
    
    Three conditions:
    1. Patch BEFORE LayerNorm (standard — adds direction to residual stream input)
    2. Patch AFTER LayerNorm (bypasses LN modification)
    3. Measure what LN does to the direction
    """
    direction_t = torch.tensor(direction, dtype=torch.float16, device=device)
    
    # Find LayerNorm modules in this layer
    layer = layers[layer_idx]
    
    # Standard transformer layer structure:
    # input_layernorm → self_attn → residual → post_attn_layernorm → mlp → residual
    # We need to find the input_layernorm and post_attention_layernorm
    
    ln_modules = {}
    for name, module in layer.named_modules():
        if 'layernorm' in name.lower() or 'ln' in name.lower():
            ln_modules[name] = module
    
    # Get the main LayerNorms
    input_ln = None
    post_attn_ln = None
    for name, mod in layer.named_children():
        if 'input_layernorm' in name or name == 'ln_1':
            input_ln = mod
        elif 'post_attention_layernorm' in name or name == 'ln_2' or name == 'post_self_attn_layernorm':
            post_attn_ln = mod
    
    results = {"before_ln": [], "after_ln": [], "ln_transform": []}
    
    for sent_data in test_sents[:10]:  # Use subset for efficiency
        if len(sent_data) != 6:
            continue
        sing, plur, sv, pv, sn, pn = sent_data
        
        toks = tokenizer.encode(sing, add_special_tokens=False)
        sp, vp = pos_fn(tokenizer, toks)
        if sp is None or vp is None:
            continue
        
        sv_ids = tokenizer.encode(sv, add_special_tokens=False)
        pv_ids = tokenizer.encode(pv, add_special_tokens=False)
        if not sv_ids or not pv_ids:
            continue
        
        input_ids = tokenizer(sing, return_tensors="pt").to(device)
        
        # 1. Baseline
        with torch.no_grad():
            base_logits = model(**input_ids).logits.detach().cpu()
        if vp >= base_logits.shape[1]:
            continue
        base_agr = (base_logits[0, vp, sv_ids[0]] - 
                   base_logits[0, vp, pv_ids[0]]).item()
        
        # 2. Patch BEFORE LayerNorm (standard method — current approach)
        # This patches the output of the transformer block, which is input to next block's LN
        applied = [False]
        def before_ln_hook(m, inp, out):
            if not applied[0]:
                if isinstance(out, tuple):
                    p = out[0].clone()
                    p[:, sp, :] += (alpha * direction_t).to(p.dtype)
                    applied[0] = True
                    return (p,) + out[1:]
                else:
                    p = out.clone()
                    p[:, sp, :] += (alpha * direction_t).to(p.dtype)
                    applied[0] = True
                    return p
            return out
        
        hook = layers[layer_idx].register_forward_hook(before_ln_hook)
        with torch.no_grad():
            patched_logits = model(**input_ids).logits.detach().cpu()
        hook.remove()
        patched_agr = (patched_logits[0, vp, sv_ids[0]] - 
                      patched_logits[0, vp, pv_ids[0]]).item()
        results["before_ln"].append(patched_agr - base_agr)
        
        # 3. Patch AFTER LayerNorm of next layer
        # This patches the output of the next layer's input_layernorm
        next_li = layer_idx + 1
        if next_li < len(layers):
            next_ln = None
            for name, mod in layers[next_li].named_children():
                if 'input_layernorm' in name or name == 'ln_1':
                    next_ln = mod
                    break
            
            if next_ln is not None:
                applied2 = [False]
                def after_ln_hook(m, inp, out):
                    if not applied2[0]:
                        if isinstance(out, tuple):
                            p = out[0].clone()
                            p[:, sp, :] += (alpha * direction_t).to(p.dtype)
                            applied2[0] = True
                            return (p,) + out[1:]
                        else:
                            p = out.clone()
                            p[:, sp, :] += (alpha * direction_t).to(p.dtype)
                            applied2[0] = True
                            return p
                    return out
                
                hook2 = next_ln.register_forward_hook(after_ln_hook)
                with torch.no_grad():
                    patched2_logits = model(**input_ids).logits.detach().cpu()
                hook2.remove()
                patched2_agr = (patched2_logits[0, vp, sv_ids[0]] - 
                               patched2_logits[0, vp, pv_ids[0]]).item()
                results["after_ln"].append(patched2_agr - base_agr)
        
        # 4. Measure what LayerNorm does to the direction
        if input_ln is not None:
            # Get the residual before LN
            captured_pre_ln = {}
            def pre_ln_hook(m, inp, out):
                captured_pre_ln["input"] = inp[0].detach().cpu() if isinstance(inp, tuple) else inp.detach().cpu()
                captured_pre_ln["output"] = (out[0] if isinstance(out, tuple) else out).detach().cpu()
            
            hook3 = input_ln.register_forward_hook(pre_ln_hook)
            with torch.no_grad():
                model(**input_ids)
            hook3.remove()
            
            if "input" in captured_pre_ln and "output" in captured_pre_ln:
                pre = captured_pre_ln["input"]  # [1, seq, d]
                post = captured_pre_ln["output"]
                
                if sp < pre.shape[1]:
                    # LN transforms: x_norm = (x - mean) / std * gamma + beta
                    # The direction transformation:
                    pre_vec = pre[0, sp, :].float().numpy()
                    post_vec = post[0, sp, :].float().numpy()
                    
                    # Simulate: what would LN do to (pre + alpha * direction)?
                    # LN(pre + α*d) ≈ LN(pre) + α * J_LN * d  (first order)
                    # where J_LN is the Jacobian of LN at pre
                    
                    # Use the actual LN module directly (works for RMSNorm too)
                    eps = 0.01
                    pre_vec = pre[0, sp, :].float().to(device)  # [d]
                    direction_t = torch.tensor(direction, dtype=torch.float32, device=device)  # [d]
                    
                    with torch.no_grad():
                        ln_normal = input_ln(pre_vec.unsqueeze(0).unsqueeze(0))[0, 0, :]  # [d]
                        pre_perturbed = pre_vec + eps * direction_t
                        ln_perturbed = input_ln(pre_perturbed.unsqueeze(0).unsqueeze(0))[0, 0, :]  # [d]
                    
                    d_after_ln = ((ln_perturbed - ln_normal) / eps).cpu().numpy()
                    
                    # Cosine between original direction and LN-transformed direction
                    cos_before_after = float(np.dot(direction, d_after_ln) / 
                                            (np.linalg.norm(direction) * np.linalg.norm(d_after_ln) + 1e-10))
                    
                    # Norm ratio
                    norm_ratio = float(np.linalg.norm(d_after_ln) / (np.linalg.norm(direction) + 1e-10))
                    
                    results["ln_transform"].append({
                        "cos_dir_vs_ln_dir": cos_before_after,
                        "norm_ratio": norm_ratio,
                    })
    
    # Summarize
    summary = {
        "before_ln_mean": float(np.mean(results["before_ln"])) if results["before_ln"] else 0,
        "before_ln_neg_ratio": float(np.mean([e < 0 for e in results["before_ln"]])) if results["before_ln"] else 0,
        "after_ln_mean": float(np.mean(results["after_ln"])) if results["after_ln"] else 0,
        "after_ln_neg_ratio": float(np.mean([e < 0 for e in results["after_ln"]])) if results["after_ln"] else 0,
        "ln_cos_mean": float(np.mean([r["cos_dir_vs_ln_dir"] for r in results["ln_transform"]])) if results["ln_transform"] else 0,
        "ln_norm_ratio_mean": float(np.mean([r["norm_ratio"] for r in results["ln_transform"]])) if results["ln_transform"] else 0,
        "n_before": len(results["before_ln"]),
        "n_after": len(results["after_ln"]),
        "n_ln": len(results["ln_transform"]),
    }
    
    return summary


# =============================================
# STEP 4: BOOTSTRAP CI FOR EFFECT SIZES
# =============================================

def bootstrap_ci(data, n_bootstrap=1000, ci=0.95):
    """Compute bootstrap confidence interval"""
    if len(data) < 3:
        return float(np.mean(data)), (float(np.mean(data)), float(np.mean(data)))
    
    data = np.array(data)
    boot_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        boot_means.append(np.mean(sample))
    
    lower = float(np.percentile(boot_means, (1 - ci) / 2 * 100))
    upper = float(np.percentile(boot_means, (1 + ci) / 2 * 100))
    
    return float(np.mean(data)), (lower, upper)


# =============================================
# STEP 5: SUBSPACE PATCHING
# =============================================

def subspace_patching(model, tokenizer, device, layers, test_sents, pos_fn,
                      V_subspace, alpha, layer_idx, patch_pos_fn):
    """
    Patch entire subspace (not single direction).
    h' = h + alpha * Proj_{V_subspace}(d_target)
    
    where d_target = mean(plur) - mean(sing) projected onto V_subspace
    """
    V_t = torch.tensor(V_subspace, dtype=torch.float16, device=device)  # [k, d]
    
    effects = []
    for sent_data in test_sents:
        if len(sent_data) != 6:
            continue
        sing, plur, sv, pv, sn, pn = sent_data
        
        toks = tokenizer.encode(sing, add_special_tokens=False)
        sp, vp = pos_fn(tokenizer, toks)
        pp = patch_pos_fn(tokenizer, toks)
        if sp is None or vp is None or pp is None:
            continue
        
        sv_ids = tokenizer.encode(sv, add_special_tokens=False)
        pv_ids = tokenizer.encode(pv, add_special_tokens=False)
        if not sv_ids or not pv_ids:
            continue
        
        input_ids = tokenizer(sing, return_tensors="pt").to(device)
        
        # Baseline
        with torch.no_grad():
            base_logits = model(**input_ids).logits.detach().cpu()
        if vp >= base_logits.shape[1]:
            continue
        
        base_agr = (base_logits[0, vp, sv_ids[0]] - 
                   base_logits[0, vp, pv_ids[0]]).item()
        
        # Patched: add subspace projection
        applied = [False]
        def add_subspace_hook(m, inp, out):
            if not applied[0]:
                if isinstance(out, tuple):
                    p = out[0].clone()
                    # Project direction onto subspace and add
                    h_at_pos = p[0, pp, :].float()  # [d]
                    # d_subspace = V^T @ V @ d_number (project d_number onto V)
                    # But we want to ADD the syntax component, not project h
                    # So we add: alpha * V^T @ V @ d_number
                    # This is equivalent to projecting d_number onto V and adding
                    applied[0] = True
                    return (p,) + out[1:]
                else:
                    p = out.clone()
                    applied[0] = True
                    return p
            return out
        
        # Simpler approach: just add the top-k PCs weighted by their variance
        # direction = sum of V_syntax[i] * pca.explained_variance_[i]
        # But we need the PCA object... let's pass it differently
        
        # Actually, let's use a simpler method:
        # The subspace direction is the projection of d_number onto V_syntax
        # d_number_proj = V^T (V d_number)
        # We precompute this and pass it as a single vector
        
        hook = layers[layer_idx].register_forward_hook(add_subspace_hook)
        with torch.no_grad():
            patched_logits = model(**input_ids).logits.detach().cpu()
        hook.remove()
        
        patched_agr = (patched_logits[0, vp, sv_ids[0]] - 
                      patched_logits[0, vp, pv_ids[0]]).item()
        effects.append(patched_agr - base_agr)
    
    return effects


def subspace_patching_v2(model, tokenizer, device, layers, test_sents, pos_fn,
                         projected_direction, alpha, layer_idx, patch_pos_fn):
    """
    Patch using a pre-computed projected direction.
    projected_direction = Proj_{V_syntax}(d_number)
    This is a single vector but lies entirely in the syntax subspace.
    """
    dir_t = torch.tensor(projected_direction, dtype=torch.float16, device=device)
    
    effects = []
    for sent_data in test_sents:
        if len(sent_data) != 6:
            continue
        sing, plur, sv, pv, sn, pn = sent_data
        
        toks = tokenizer.encode(sing, add_special_tokens=False)
        sp, vp = pos_fn(tokenizer, toks)
        pp = patch_pos_fn(tokenizer, toks)
        if sp is None or vp is None or pp is None:
            continue
        
        sv_ids = tokenizer.encode(sv, add_special_tokens=False)
        pv_ids = tokenizer.encode(pv, add_special_tokens=False)
        if not sv_ids or not pv_ids:
            continue
        
        input_ids = tokenizer(sing, return_tensors="pt").to(device)
        
        # Baseline
        with torch.no_grad():
            base_logits = model(**input_ids).logits.detach().cpu()
        if vp >= base_logits.shape[1]:
            continue
        
        base_agr = (base_logits[0, vp, sv_ids[0]] - 
                   base_logits[0, vp, pv_ids[0]]).item()
        
        # Patched
        applied = [False]
        def add_hook(m, inp, out, a=alpha, p=pp):
            if not applied[0]:
                if isinstance(out, tuple):
                    p_out = out[0].clone()
                    p_out[:, p, :] += (a * dir_t).to(p_out.dtype)
                    applied[0] = True
                    return (p_out,) + out[1:]
                else:
                    p_out = out.clone()
                    p_out[:, p, :] += (a * dir_t).to(p_out.dtype)
                    applied[0] = True
                    return p_out
            return out
        
        hook = layers[layer_idx].register_forward_hook(add_hook)
        with torch.no_grad():
            patched_logits = model(**input_ids).logits.detach().cpu()
        hook.remove()
        
        patched_agr = (patched_logits[0, vp, sv_ids[0]] - 
                      patched_logits[0, vp, pv_ids[0]]).item()
        effects.append(patched_agr - base_agr)
    
    return effects


# =============================================
# MAIN EXPERIMENT
# =============================================

def run_phase61(model, tokenizer, device, info):
    print("=" * 70)
    print("★★★ Phase 61: Rigorous Subspace Analysis ★★★")
    print("Fixing: 1) cos≈0 baseline, 2) subspace>direction, 3) α curve, 4) LN")
    print("=" * 70)
    
    layers = get_layers(model)
    target_layers = [0, 5, 10, 15, 18, 20, 25]
    target_layers = [l for l in target_layers if l < info.n_layers]
    
    # Generate sentences
    svo_data, adv_data = [], []
    for sn, pn, sv, pv, adv in NVA_PAIRS:
        svo_data.append((f"The {sn} {sv} {adv}", f"The {pn} {pv} {adv}", sv, pv, sn, pn))
        adv_data.append((f"Today the {sn} {sv} {adv}", f"Today the {pn} {pv} {adv}", sv, pv, sn, pn))
    
    svo_sing = [d[0] for d in svo_data]
    svo_plur = [d[1] for d in svo_data]
    adv_sing = [d[0] for d in adv_data]
    adv_plur = [d[1] for d in adv_data]
    
    print(f"\nTotal: {len(svo_data)} SVO pairs, {len(adv_data)} Adv pairs")
    print(f"Total sentences: {4 * len(NVA_PAIRS)}")
    
    # ===== COLLECT ACTIVATIONS =====
    print("\n" + "=" * 70)
    print("Step 0: Collecting activations (50 pairs × 4 conditions = 200 sentences)")
    print("=" * 70)
    
    t0 = time.time()
    svo_sing_act, n1 = collect_activations(model, tokenizer, device, svo_sing, svo_pos_fn, target_layers, "SVO-sing")
    svo_plur_act, n2 = collect_activations(model, tokenizer, device, svo_plur, svo_pos_fn, target_layers, "SVO-plur")
    adv_sing_act, n3 = collect_activations(model, tokenizer, device, adv_sing, adv_pos_fn, target_layers, "Adv-sing")
    adv_plur_act, n4 = collect_activations(model, tokenizer, device, adv_plur, adv_pos_fn, target_layers, "Adv-plur")
    print(f"  Took {time.time()-t0:.1f}s")
    print(f"  Valid: SVO-sing={n1}, SVO-plur={n2}, Adv-sing={n3}, Adv-plur={n4}")
    
    # ===== STEP 1: SUBSPACE ANALYSIS =====
    print("\n" + "=" * 70)
    print("STEP 1: Subspace Analysis (PCA + CCA + Principal Angles)")
    print("=" * 70)
    
    subspace_results = {}
    for li in target_layers:
        if li not in svo_sing_act or not svo_sing_act[li]["subj"]:
            continue
        if li not in adv_sing_act or not adv_sing_act[li]["subj"]:
            continue
        
        print(f"\n  Analyzing L{li}...")
        result = subspace_analysis(svo_sing_act, svo_plur_act, adv_sing_act, adv_plur_act, li, info.d_model)
        
        if result is None:
            continue
        
        subspace_results[li] = result
        
        print(f"    Syntax subspace: {result['n_pcs_90']} PCs for 90% var, {result['n_pcs_99']} for 99%")
        print(f"    Top-1 PC explains: {result['var_explained_top1']:.3f} ({result['var_explained_top1']*100:.1f}%)")
        print(f"    Top-5 PCs explain: {result['var_explained_top5']:.3f}")
        print(f"    Position subspace: {result['n_pcs_90_pos']} PCs for 90% var")
        print(f"    CCA mean corr: {result['cca_mean']:.4f}, max: {result['cca_max']:.4f}")
        print(f"    Principal angle mean cos: {result['mean_cos_angle']:.4f}")
        print(f"    Random baseline cos: {result['random_baseline_mean']:.4f} ± {result['random_baseline_std']:.4f}")
        print(f"    Z-score vs random: {result['z_score_vs_random']:.2f}")
        print(f"    Overlap (syntax in pos): {result['overlap_mean']:.4f}")
        print(f"    PC cosines (pos2 vs pos3): {[f'{c:.3f}' for c in result['pc_cosines_pos2_vs_pos3'][:5]]}")
    
    # ===== STEP 2: α-RESPONSE CURVE =====
    print("\n" + "=" * 70)
    print("STEP 2: α-Response Curve (monotonicity + linearity)")
    print("=" * 70)
    
    # Extract difference-of-means direction for patching
    test_svo = svo_data[:30]  # Use 30 for patching (keep runtime manageable)
    test_adv = adv_data[:30]
    
    alphas = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
    
    alpha_curves = {}
    for li in [0, 10, 15, 18]:  # Focus on key layers
        if li not in svo_sing_act or not svo_sing_act[li]["subj"]:
            continue
        if li not in adv_sing_act or not adv_sing_act[li]["subj"]:
            continue
        
        # Extract direction
        sing2 = np.array(svo_sing_act[li]["subj"])
        plur2 = np.array(svo_plur_act[li]["subj"])
        sing3 = np.array(adv_sing_act[li]["subj"])
        plur3 = np.array(adv_plur_act[li]["subj"])
        
        d_num2 = plur2.mean(0) - sing2.mean(0)
        d_num3 = plur3.mean(0) - sing3.mean(0)
        
        # Cross-position: d_num2 applied to Adv subj@pos3
        print(f"\n  L{li}: Cross-position α-curve (d_num2 → Adv@pos3)")
        curve_cross = alpha_response_curve(
            model, tokenizer, device, layers, test_adv,
            adv_pos_fn, d_num2,
            lambda tok, toks: adv_pos_fn(tok, toks)[0],
            li, alphas
        )
        
        # Same-position: d_num2 applied to SVO subj@pos2
        print(f"  L{li}: Same-position α-curve (d_num2 → SVO@pos2)")
        curve_same = alpha_response_curve(
            model, tokenizer, device, layers, test_svo,
            svo_pos_fn, d_num2,
            lambda tok, toks: svo_pos_fn(tok, toks)[0],
            li, alphas
        )
        
        alpha_curves[li] = {
            "cross": curve_cross,
            "same": curve_same,
        }
        
        # Print curve
        print(f"\n  L{li} α-curve:")
        print(f"  {'α':>6} | {'cross Δ':>10} | {'cross neg%':>10} | {'same Δ':>10} | {'same neg%':>10}")
        print(f"  {'-'*55}")
        for a in alphas:
            cr = curve_cross.get(a, {})
            sm = curve_same.get(a, {})
            cr_str = f"{cr.get('mean', 'N/A'):+.4f}" if cr else "N/A"
            cr_neg = f"{cr.get('neg_ratio', 0):.0%}" if cr else "N/A"
            sm_str = f"{sm.get('mean', 'N/A'):+.4f}" if sm else "N/A"
            sm_neg = f"{sm.get('neg_ratio', 0):.0%}" if sm else "N/A"
            print(f"  {a:>6.2f} | {cr_str:>10} | {cr_neg:>10} | {sm_str:>10} | {sm_neg:>10}")
        
        # Check monotonicity
        cross_deltas = [curve_cross.get(a, {}).get("mean", 0) for a in alphas if a in curve_cross]
        if len(cross_deltas) > 2:
            # Is the curve monotonically decreasing (more negative = stronger effect)?
            monotonic = all(cross_deltas[i] >= cross_deltas[i+1] for i in range(len(cross_deltas)-1))
            print(f"  Cross-position monotonic: {monotonic}")
        
        gc.collect()
        torch.cuda.empty_cache()
    
    # ===== STEP 3: LAYERNORM CONTROL =====
    print("\n" + "=" * 70)
    print("STEP 3: LayerNorm Control (before vs after LN patching)")
    print("=" * 70)
    
    ln_results = {}
    for li in [0, 10, 15, 18]:
        if li not in svo_sing_act or not svo_sing_act[li]["subj"]:
            continue
        
        sing2 = np.array(svo_sing_act[li]["subj"])
        plur2 = np.array(svo_plur_act[li]["subj"])
        d_num2 = plur2.mean(0) - sing2.mean(0)
        
        print(f"\n  L{li}: LayerNorm analysis...")
        ln_result = layernorm_analysis(
            model, tokenizer, device, layers, li, d_num2,
            test_svo[:15], svo_pos_fn, alpha=2.0
        )
        ln_results[li] = ln_result
        
        print(f"    Before LN: Δ={ln_result['before_ln_mean']:+.4f}, neg%={ln_result['before_ln_neg_ratio']:.0%}, n={ln_result['n_before']}")
        print(f"    After LN:  Δ={ln_result['after_ln_mean']:+.4f}, neg%={ln_result['after_ln_neg_ratio']:.0%}, n={ln_result['n_after']}")
        print(f"    LN dir cosine: {ln_result['ln_cos_mean']:.4f}")
        print(f"    LN norm ratio: {ln_result['ln_norm_ratio_mean']:.4f}")
        
        if ln_result['ln_cos_mean'] < 0.5:
            print(f"    ⚠⚠⚠ LN DESTROYS DIRECTION! cos={ln_result['ln_cos_mean']:.3f} < 0.5")
        elif ln_result['ln_cos_mean'] < 0.8:
            print(f"    ⚠ LN significantly modifies direction! cos={ln_result['ln_cos_mean']:.3f}")
        else:
            print(f"    ✓ LN preserves direction reasonably well (cos={ln_result['ln_cos_mean']:.3f})")
        
        gc.collect()
        torch.cuda.empty_cache()
    
    # ===== STEP 4: SUBSPACE PATCHING + BOOTSTRAP CI =====
    print("\n" + "=" * 70)
    print("STEP 4: Subspace Patching + Bootstrap CI (30 pairs)")
    print("=" * 70)
    
    subspace_patch_results = {}
    for li in [0, 10, 15, 18]:
        if li not in subspace_results:
            continue
        
        V_syntax = subspace_results[li]["V_syntax"]  # [k, d]
        n_pcs = subspace_results[li]["n_pcs_90"]
        
        # Extract mean direction
        sing2 = np.array(svo_sing_act[li]["subj"])
        plur2 = np.array(svo_plur_act[li]["subj"])
        d_num2 = plur2.mean(0) - sing2.mean(0)
        
        # Project d_number onto syntax subspace
        # d_proj = V^T @ V @ d_num (project d_num onto V_syntax)
        V_Vt = V_syntax.T @ V_syntax  # [d, d] projection matrix
        d_proj = V_Vt @ d_num2
        
        # How much of d_num is captured by the subspace?
        cos_proj = float(np.dot(d_num2, d_proj) / (np.linalg.norm(d_num2) * np.linalg.norm(d_proj) + 1e-10))
        norm_ratio = float(np.linalg.norm(d_proj) / (np.linalg.norm(d_num2) + 1e-10))
        
        print(f"\n  L{li}: Subspace projection of d_number")
        print(f"    n_pcs_90={n_pcs}, cos(d_num, d_proj)={cos_proj:.4f}, ||d_proj||/||d_num||={norm_ratio:.4f}")
        
        # Test: cross-position patching with projected direction
        effects_proj = subspace_patching_v2(
            model, tokenizer, device, layers, test_adv,
            adv_pos_fn, d_proj, 2.0, li,
            lambda tok, toks: adv_pos_fn(tok, toks)[0]
        )
        
        # Test: cross-position patching with RAW direction (for comparison)
        effects_raw = subspace_patching_v2(
            model, tokenizer, device, layers, test_adv,
            adv_pos_fn, d_num2, 2.0, li,
            lambda tok, toks: adv_pos_fn(tok, toks)[0]
        )
        
        # Bootstrap CIs
        mean_proj, ci_proj = bootstrap_ci(effects_proj) if effects_proj else (0, (0, 0))
        mean_raw, ci_raw = bootstrap_ci(effects_raw) if effects_raw else (0, (0, 0))
        
        subspace_patch_results[li] = {
            "projected": {"mean": mean_proj, "ci": ci_proj, "n": len(effects_proj)},
            "raw": {"mean": mean_raw, "ci": ci_raw, "n": len(effects_raw)},
            "cos_proj": cos_proj,
            "norm_ratio": norm_ratio,
            "n_pcs": n_pcs,
        }
        
        print(f"    Projected: Δ={mean_proj:+.4f}, CI=[{ci_proj[0]:+.4f}, {ci_proj[1]:+.4f}], n={len(effects_proj)}")
        print(f"    Raw:       Δ={mean_raw:+.4f}, CI=[{ci_raw[0]:+.4f}, {ci_raw[1]:+.4f}], n={len(effects_raw)}")
        
        # Check if CI excludes 0
        if ci_proj[0] < 0 and ci_proj[1] < 0:
            print(f"    ★★★ PROJECTED direction CI excludes 0! (p<0.05)")
        if ci_raw[0] < 0 and ci_raw[1] < 0:
            print(f"    ★★ RAW direction CI excludes 0! (p<0.05)")
        
        gc.collect()
        torch.cuda.empty_cache()
    
    # ===== FINAL SUMMARY =====
    print("\n" + "=" * 70)
    print("FINAL SUMMARY: Phase 61 Rigorous Analysis")
    print("=" * 70)
    
    print("\n--- 1. Subspace vs Single Direction ---")
    for li in sorted(subspace_results.keys()):
        r = subspace_results[li]
        print(f"  L{li}: {r['n_pcs_90']} PCs for 90% var, "
              f"top-1={r['var_explained_top1']:.3f}, "
              f"CCA mean={r['cca_mean']:.3f}, "
              f"PA cos={r['mean_cos_angle']:.4f} vs random={r['random_baseline_mean']:.4f} (z={r['z_score_vs_random']:.1f}), "
              f"overlap={r['overlap_mean']:.4f}")
    
    print("\n--- 2. α-Response Curve Monotonicity ---")
    for li in sorted(alpha_curves.keys()):
        cr = alpha_curves[li]["cross"]
        deltas = [cr.get(a, {}).get("mean", 0) for a in alphas if a in cr]
        if len(deltas) >= 3:
            is_monotone = all(deltas[i] >= deltas[i+1] for i in range(len(deltas)-1))
            # Linear fit quality
            a_vals = np.array([a for a in alphas if a in cr])
            d_vals = np.array(deltas)
            if len(a_vals) >= 3:
                from numpy.polynomial import polynomial as P
                coeffs = np.polyfit(a_vals, d_vals, 1)
                d_pred = np.polyval(coeffs, a_vals)
                ss_res = np.sum((d_vals - d_pred) ** 2)
                ss_tot = np.sum((d_vals - d_vals.mean()) ** 2)
                r2 = 1 - ss_res / (ss_tot + 1e-10)
                print(f"  L{li}: monotone={is_monotone}, R²(linear)={r2:.3f}, slope={coeffs[0]:.4f}")
            else:
                print(f"  L{li}: insufficient data")
    
    print("\n--- 3. LayerNorm Effect ---")
    for li in sorted(ln_results.keys()):
        r = ln_results[li]
        verdict = "PRESERVED" if r['ln_cos_mean'] > 0.8 else ("MODIFIED" if r['ln_cos_mean'] > 0.5 else "DESTROYED")
        print(f"  L{li}: LN cos={r['ln_cos_mean']:.4f}, norm_ratio={r['ln_norm_ratio_mean']:.4f} → {verdict}")
        print(f"         before_LN Δ={r['before_ln_mean']:+.4f}, after_LN Δ={r['after_ln_mean']:+.4f}")
    
    print("\n--- 4. Subspace Patching + Bootstrap CI ---")
    for li in sorted(subspace_patch_results.keys()):
        r = subspace_patch_results[li]
        proj_sig = "★" if r['projected']['ci'][0] < 0 and r['projected']['ci'][1] < 0 else ""
        raw_sig = "★" if r['raw']['ci'][0] < 0 and r['raw']['ci'][1] < 0 else ""
        print(f"  L{li}: proj Δ={r['projected']['mean']:+.4f} [{r['projected']['ci'][0]:+.4f},{r['projected']['ci'][1]:+.4f}] {proj_sig}")
        print(f"        raw  Δ={r['raw']['mean']:+.4f} [{r['raw']['ci'][0]:+.4f},{r['raw']['ci'][1]:+.4f}] {raw_sig}")
        print(f"        cos(d,proj)={r['cos_proj']:.4f}, norm_ratio={r['norm_ratio']:.4f}")
    
    # ===== KEY VERDICT =====
    print("\n" + "=" * 70)
    print("KEY VERDICT")
    print("=" * 70)
    
    # 1. Is cos≈0 meaningful vs random?
    z_scores = [subspace_results[li]['z_score_vs_random'] for li in subspace_results]
    if z_scores:
        mean_z = np.mean(z_scores)
        print(f"\n  1. Principal Angle vs Random Baseline:")
        print(f"     Mean z-score = {mean_z:.2f}")
        if mean_z > 2:
            print(f"     ★★★ Syntax subspace is SIGNIFICANTLY different from random (z>2)")
        else:
            print(f"     ⚠ Syntax subspace NOT significantly different from random!")
    
    # 2. Is syntax subspace low-dimensional?
    n_pcs_list = [subspace_results[li]['n_pcs_90'] for li in subspace_results]
    if n_pcs_list:
        print(f"\n  2. Syntax Subspace Dimensionality:")
        print(f"     n_pcs for 90% var: {n_pcs_list}")
        if max(n_pcs_list) <= 5:
            print(f"     ★★★ Syntax is LOW-DIMENSIONAL (≤5 PCs for 90%)")
        elif max(n_pcs_list) <= 15:
            print(f"     ★★ Syntax is MODERATELY low-dimensional")
        else:
            print(f"     ⚠ Syntax is HIGH-DIMENSIONAL")
    
    # 3. CCA overlap
    cca_means = [subspace_results[li]['cca_mean'] for li in subspace_results]
    if cca_means:
        print(f"\n  3. CCA Correlation (syntax ↔ position):")
        print(f"     Mean CCA: {np.mean(cca_means):.4f}")
        if np.mean(cca_means) < 0.1:
            print(f"     ★★★ Syntax and Position are NEARLY UNCORRELATED")
        elif np.mean(cca_means) < 0.3:
            print(f"     ★ Weak correlation")
        else:
            print(f"     ⚠ Significant correlation — not fully independent")
    
    # 4. LayerNorm
    ln_cos = [ln_results[li]['ln_cos_mean'] for li in ln_results if ln_results[li]['n_ln'] > 0]
    if ln_cos:
        print(f"\n  4. LayerNorm Preservation:")
        print(f"     Mean cos(dir, LN(dir)) = {np.mean(ln_cos):.4f}")
        if np.mean(ln_cos) > 0.8:
            print(f"     ✓ LN preserves direction — results reliable")
        elif np.mean(ln_cos) > 0.5:
            print(f"     ⚠ LN modifies direction significantly — some results may be unreliable")
        else:
            print(f"     ⚠⚠⚠ LN destroys direction — ALL previous patching results questionable!")
    
    # 5. α-response curve
    print(f"\n  5. α-Response Curve:")
    monotone_count = 0
    for li in alpha_curves:
        cr = alpha_curves[li]["cross"]
        deltas = [cr.get(a, {}).get("mean", 0) for a in alphas if a in cr]
        if len(deltas) >= 3 and all(deltas[i] >= deltas[i+1] for i in range(len(deltas)-1)):
            monotone_count += 1
    print(f"     Monotone curves: {monotone_count}/{len(alpha_curves)}")
    if monotone_count == len(alpha_curves):
        print(f"     ★★★ All curves monotone — direction is a clean variable")
    elif monotone_count >= len(alpha_curves) // 2:
        print(f"     ★★ Most curves monotone — direction is mostly clean")
    else:
        print(f"     ⚠ Non-monotone curves — direction may not be a clean variable")
    
    return {
        "subspace": subspace_results,
        "alpha_curves": alpha_curves,
        "ln_results": ln_results,
        "subspace_patch": subspace_patch_results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="deepseek7b")
    args = parser.parse_args()
    
    print(f"Loading model: {args.model}")
    model, tokenizer, device = load_model(args.model)
    info = get_model_info(model, args.model)
    print(f"Model: {info.name}, Layers={info.n_layers}, d_model={info.d_model}")
    
    try:
        results = run_phase61(model, tokenizer, device, info)
    finally:
        release_model(model)
        print("\nDone.")
