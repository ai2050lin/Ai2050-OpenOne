"""
Phase 62: Position-Controlled Syntax Analysis — Breaking Position-Syntax Confound
=================================================================================
Core insight: CCA=1.0 in Phase 61 was DATA ARTIFACT (English SVO: pos≈syntax).
Fix: CONDITIONAL analysis — extract syntax WITHIN each position, compare ACROSS positions.

5 templates with subject at different positions:
  T1: "The [N] [V]"                    — subj @ pos2
  T2: "Today, the [N] [V]"             — subj @ pos3
  T3: "Right now, the [N] [V]"         — subj @ pos4
  T4: "At noon, the big [N] [V]"       — subj @ pos5
  T5: "In the park, the big [N] [V]"   — subj @ pos6

Key tests:
  1. Within-position syntax direction extraction
  2. Cross-position direction comparison (cosine)
  3. Conditional CCA (within position, NOT across)
  4. Cross-position subspace patching with Bootstrap CI (50 pairs)
  5. After-LN patching (systematic, 30 pairs)

Data: 80 NVA pairs × 5 templates × 2 numbers = 800 sentences
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

NVA_PAIRS = [
    ("cat","cats","runs","run"),("dog","dogs","walks","walk"),("bird","birds","flies","fly"),
    ("girl","girls","reads","read"),("boy","boys","sings","sing"),("man","men","works","work"),
    ("horse","horses","jumps","jump"),("bear","bears","sleeps","sleep"),("snake","snakes","crawls","crawl"),
    ("frog","frogs","swims","swim"),("fox","foxes","hunts","hunt"),("king","kings","rules","rule"),
    ("student","students","studies","study"),("teacher","teachers","speaks","speak"),
    ("doctor","doctors","helps","help"),("tree","trees","grows","grow"),("car","cars","moves","move"),
    ("queen","queens","leads","lead"),("child","children","plays","play"),("wolf","wolves","howls","howl"),
    ("driver","drivers","drives","drive"),("worker","workers","builds","build"),
    ("player","players","wins","win"),("writer","writers","writes","write"),
    ("rabbit","rabbits","hops","hop"),("eagle","eagles","soars","soar"),
    ("tiger","tigers","stalks","stalk"),("monkey","monkeys","climbs","climb"),
    ("lion","lions","roars","roar"),("farmer","farmers","plants","plant"),
    ("fish","fish","swims","swim"),("knife","knives","cuts","cut"),
    ("mouse","mice","squeaks","squeak"),("goose","geese","flies","fly"),
    ("tooth","teeth","bites","bite"),("foot","feet","steps","step"),
    ("person","people","speaks","speak"),("hero","heroes","fights","fight"),
    ("potato","potatoes","grows","grow"),("boss","bosses","leads","lead"),
    ("baby","babies","cries","cry"),("lady","ladies","dances","dance"),
    ("story","stories","ends","end"),("city","cities","shines","shine"),
    ("party","parties","starts","start"),("apple","apples","falls","fall"),
    ("river","rivers","flows","flow"),("cloud","clouds","drifts","drift"),
    ("star","stars","shines","shine"),("moon","moons","orbits","orbit"),
    ("book","books","opens","open"),("phone","phones","rings","ring"),
    ("clock","clocks","ticks","tick"),("train","trains","arrives","arrive"),
    ("plane","planes","lands","land"),("ship","ships","sails","sail"),
    ("robot","robots","moves","move"),("wizard","wizards","casts","cast"),
    ("dragon","dragons","breathes","breathe"),("angel","angels","flies","fly"),
    ("soldier","soldiers","marches","march"),("nurse","nurses","cares","care"),
    ("artist","artists","paints","paint"),("dancer","dancers","spins","spin"),
    ("singer","singers","sings","sing"),("hunter","hunters","tracks","track"),
    ("thief","thieves","steals","steal"),("leaf","leaves","falls","fall"),
    ("life","lives","begins","begin"),("wife","wives","cooks","cook"),
    ("calf","calves","runs","run"),("half","halves","breaks","break"),
    ("self","selves","acts","act"),("shelf","shelves","holds","hold"),
    ("wolf","wolves","runs","run"),("calf","calves","walks","walk"),
    ("ox","oxen","pulls","pull"),("child","children","runs","run"),
    ("mouse","mice","runs","run"),("goose","geese","walks","walk"),
    ("louse","lice","crawls","crawl"),("datum","data","shows","show"),
]

NOUNS_SET = set()
for sn, pn, _, _ in NVA_PAIRS:
    NOUNS_SET.add(sn.lower())
    NOUNS_SET.add(pn.lower())

TEMPLATES = [
    ("The {n} {v}", "T1_pos2"),
    ("Today, the {n} {v}", "T2_pos3"),
    ("Right now, the {n} {v}", "T3_pos4"),
    ("At noon, the big {n} {v}", "T4_pos5"),
    ("In the park, the big {n} {v}", "T5_pos6"),
]

TARGET_LAYERS = [0, 10, 15, 18]


def find_noun_pos(tokenizer, tokens):
    """Find subject noun position (0-indexed from BOS)"""
    for i, t in enumerate(tokens):
        d = tokenizer.decode([t]).strip().lower()
        for n in NOUNS_SET:
            if d == n or d.startswith(n):
                return i + 1  # +1 BOS
    return None


def find_verb_pos(tokenizer, tokens, sv, pv):
    """Find verb position"""
    targets = {sv.lower(), pv.lower()}
    for i, t in enumerate(tokens):
        d = tokenizer.decode([t]).strip().lower()
        for vt in targets:
            if d == vt or d.startswith(vt):
                return i + 1
    return None


def collect_template_acts(model, tokenizer, device, tmpl_str, nva_pairs, 
                          target_layers, label=""):
    """Collect subj/verb activations for one template, separated by number"""
    layers = get_layers(model)
    sing_subj = defaultdict(list)  # layer -> [acts]
    plur_subj = defaultdict(list)
    sing_verb = defaultdict(list)
    plur_verb = defaultdict(list)
    n_valid = 0

    for si, (sn, pn, sv, pv) in enumerate(nva_pairs):
        if si % 20 == 0 and si > 0:
            print(f"  {label} {si}/{len(nva_pairs)}")

        for is_sing, (noun, verb) in [(True, (sn, sv)), (False, (pn, pv))]:
            sent = tmpl_str.format(n=noun, v=verb)
            toks = tokenizer.encode(sent, add_special_tokens=False)
            sp = find_noun_pos(tokenizer, toks)
            vp = find_verb_pos(tokenizer, toks, sv, pv)
            if sp is None or vp is None:
                continue

            captured = {}
            def mk_hook(li):
                def fn(m, inp, out):
                    captured[li] = (out[0] if isinstance(out, tuple) else out).detach().cpu()
                return fn

            hooks = []
            for li in target_layers:
                if li < len(layers):
                    hooks.append(layers[li].register_forward_hook(mk_hook(li)))

            ids = tokenizer(sent, return_tensors="pt").to(device)
            with torch.no_grad():
                try: model(**ids)
                except:
                    for h in hooks: h.remove()
                    continue
            for h in hooks: h.remove()

            for li in target_layers:
                if li in captured:
                    h = captured[li]
                    if sp < h.shape[1]:
                        target = sing_subj if is_sing else plur_subj
                        target[li].append(h[0, sp, :].float().numpy())
                    if vp < h.shape[1]:
                        target = sing_verb if is_sing else plur_verb
                        target[li].append(h[0, vp, :].float().numpy())
            n_valid += 1
            del captured; gc.collect(); torch.cuda.empty_cache()

    print(f"  {label}: {n_valid} valid")
    return sing_subj, plur_subj, sing_verb, plur_verb, n_valid


# =============================================
# STEP 1: WITHIN-POSITION SYNTAX DIRECTION
# =============================================
def within_position_syntax(sing_subj, plur_subj, li):
    """Extract syntax direction within one position (difference of means)"""
    s = np.array(sing_subj[li])  # [N_s, d]
    p = np.array(plur_subj[li])  # [N_p, d]
    if len(s) < 3 or len(p) < 3:
        return None, None, 0
    d_num = p.mean(axis=0) - s.mean(axis=0)
    d_num = d_num / (np.linalg.norm(d_num) + 1e-10)
    # Also compute PCA-based subspace
    from sklearn.decomposition import PCA
    diff = p - s.mean(axis=0)  # [N_p, d]
    diff2 = s - p.mean(axis=0)  # [N_s, d]
    all_diff = np.vstack([diff, diff2])
    pca = PCA(n_components=min(20, all_diff.shape[0]-1, all_diff.shape[1]))
    pca.fit(all_diff)
    return d_num, pca, len(s) + len(p)


# =============================================
# STEP 2: CROSS-POSITION DIRECTION COMPARISON
# =============================================
def cross_position_compare(directions, pcas, template_names, li):
    """Compare syntax directions across positions"""
    print(f"\n  Layer {li}: Cross-position syntax direction comparison")
    
    valid_dirs = {name: d for name, d in directions.items() if d is not None}
    valid_pcas = {name: p for name, p in pcas.items() if p is not None}
    
    if len(valid_dirs) < 2:
        print("    Not enough valid directions for comparison")
        return
    
    # Pairwise cosine similarities
    names = list(valid_dirs.keys())
    print(f"    Pairwise cosine similarities ({len(names)} directions):")
    cos_matrix = np.zeros((len(names), len(names)))
    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            c = float(np.dot(valid_dirs[n1], valid_dirs[n2]))
            cos_matrix[i, j] = c
            if i < j:
                print(f"      cos({n1}, {n2}) = {c:.4f}")
    
    # Random baseline: random unit vectors in same dimension
    d = len(list(valid_dirs.values())[0])
    n_random = 10000
    random_cos = []
    for _ in range(n_random):
        v1 = np.random.randn(d); v1 /= np.linalg.norm(v1)
        v2 = np.random.randn(d); v2 /= np.linalg.norm(v2)
        random_cos.append(abs(np.dot(v1, v2)))
    random_mean = np.mean(random_cos)
    random_std = np.std(random_cos)
    
    # Z-scores for each pair
    print(f"    Random baseline: |cos| = {random_mean:.4f} ± {random_std:.4f}")
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            z = (abs(cos_matrix[i,j]) - random_mean) / random_std
            print(f"      z({names[i]}, {names[j]}) = {z:.1f}")
    
    # Subspace overlap using PCA components
    if len(valid_pcas) >= 2:
        print(f"    Subspace overlap (PCA top-5):")
        pca_names = list(valid_pcas.keys())
        for i in range(len(pca_names)):
            for j in range(i+1, len(pca_names)):
                # Project one subspace onto another
                comps_i = valid_pcas[pca_names[i]].components_[:5]  # [5, d]
                comps_j = valid_pcas[pca_names[j]].components_[:5]  # [5, d]
                # Mean cosine between corresponding PCs
                pc_cos = []
                for k in range(min(5, comps_i.shape[0], comps_j.shape[0])):
                    pc_cos.append(abs(float(np.dot(comps_i[k], comps_j[k]))))
                print(f"      {pca_names[i]} vs {pca_names[j]}: "
                      f"mean|cos|={np.mean(pc_cos):.4f} [{', '.join(f'{c:.3f}' for c in pc_cos)}]")


# =============================================
# STEP 3: CONDITIONAL CCA (within position)
# =============================================
def conditional_cca(sing_subj, plur_subj, template_name, li):
    """CCA within a single position — NO position confound"""
    from sklearn.cross_decomposition import CCA
    
    s = np.array(sing_subj[li])  # [N, d]
    p = np.array(plur_subj[li])
    if len(s) < 5 or len(p) < 5:
        return None
    
    # Within this position, the ONLY variable is number (sing/plur)
    # So CCA should reveal number-related directions
    # X = representations, Y = number label (binary)
    X = np.vstack([s, p])  # [2N, d]
    y = np.array([0]*len(s) + [1]*len(p))  # sing=0, plur=1
    
    # Use CCA with 1 component (binary Y)
    # Actually, let's do PCA on the difference instead
    # Within one position, there's no position variance — only number variance
    d_num = p.mean(axis=0) - s.mean(axis=0)
    
    # Compute variance explained by number
    total_var = np.var(X, axis=0).sum()
    num_var = np.dot(d_num, d_num) * len(s) * len(p) / (len(s) + len(p))
    fraction = num_var / (total_var + 1e-10)
    
    # Bootstrap CI for this fraction
    n_boot = 1000
    fracs = []
    for _ in range(n_boot):
        idx_s = np.random.choice(len(s), len(s), replace=True)
        idx_p = np.random.choice(len(p), len(p), replace=True)
        d_boot = p[idx_p].mean(axis=0) - s[idx_s].mean(axis=0)
        X_boot = np.vstack([s[idx_s], p[idx_p]])
        tv = np.var(X_boot, axis=0).sum()
        nv = np.dot(d_boot, d_boot) * len(s) * len(p) / (len(s) + len(p))
        fracs.append(nv / (tv + 1e-10))
    
    ci_lo, ci_hi = np.percentile(fracs, [2.5, 97.5])
    
    return {
        "fraction": float(fraction),
        "ci": (float(ci_lo), float(ci_hi)),
        "n_sing": len(s),
        "n_plur": len(p),
        "d_num_norm": float(np.linalg.norm(d_num)),
    }


# =============================================
# STEP 4: CROSS-POSITION SUBSPACE PATCHING
# =============================================
def cross_position_patching(model, tokenizer, device, target_layers,
                            src_template, tgt_template, nva_pairs, 
                            d_num_src, alpha, n_test=50, patch_after_ln=False):
    """
    Patch with syntax direction from src_template onto tgt_template.
    Tests: can syntax direction transfer across positions?
    """
    layers = get_layers(model)
    deltas = []
    
    for si, (sn, pn, sv, pv) in enumerate(nva_pairs[:n_test]):
        # Target sentence: singular subject → we add plural direction → should shift toward plural verb
        tgt_sent = tgt_template.format(n=sn, v=sv)
        toks = tokenizer.encode(tgt_sent, add_special_tokens=False)
        tgt_sp = find_noun_pos(tokenizer, toks)
        tgt_vp = find_verb_pos(tokenizer, toks, sv, pv)
        if tgt_sp is None or tgt_vp is None:
            continue
        
        ids = tokenizer(tgt_sent, return_tensors="pt").to(device)
        sv_ids = tokenizer.encode(sv, add_special_tokens=False)
        pv_ids = tokenizer.encode(pv, add_special_tokens=False)
        if not sv_ids or not pv_ids:
            continue
        
        # Base logits
        with torch.no_grad():
            base_logits = model(**ids).logits.detach().cpu()
        base_agr = (base_logits[0, tgt_vp, sv_ids[0]] - base_logits[0, tgt_vp, pv_ids[0]]).item()
        
        # Patched: add d_num_src at subject position
        for li in target_layers:
            captured = {}
            applied = [False]
            
            if patch_after_ln:
                # Patch AFTER LayerNorm
                layer = layers[li]
                input_ln = None
                for name, mod in layer.named_children():
                    if 'input_layernorm' in name or name == 'ln_1':
                        input_ln = mod
                        break
                
                if input_ln is None:
                    continue
                
                def after_ln_hook(m, inp, out, _li=li, _sp=tgt_sp):
                    if not applied[0]:
                        out_mod = out[0] if isinstance(out, tuple) else out
                        p = out_mod.clone()
                        direction_t = torch.tensor(d_num_src, dtype=torch.float32, device=device)
                        p[:, _sp, :] += (alpha * direction_t).to(p.dtype)
                        applied[0] = True
                        return (p,) + out[1:] if isinstance(out, tuple) else p
                    return out
                
                hook = input_ln.register_forward_hook(after_ln_hook)
            else:
                # Patch BEFORE LayerNorm (at residual stream, which is what we've been doing)
                def res_hook(m, inp, out, _li=li, _sp=tgt_sp):
                    if not applied[0]:
                        out_mod = out[0] if isinstance(out, tuple) else out
                        p = out_mod.clone()
                        direction_t = torch.tensor(d_num_src, dtype=torch.float32, device=device)
                        p[:, _sp, :] += (alpha * direction_t).to(p.dtype)
                        applied[0] = True
                        return (p,) + out[1:] if isinstance(out, tuple) else p
                    return out
                
                hook = layers[li].register_forward_hook(res_hook)
            
            with torch.no_grad():
                patched_logits = model(**ids).logits.detach().cpu()
            hook.remove()
            
            patched_agr = (patched_logits[0, tgt_vp, sv_ids[0]] - 
                          patched_logits[0, tgt_vp, pv_ids[0]]).item()
            deltas.append(patched_agr - base_agr)
        
        gc.collect(); torch.cuda.empty_cache()
    
    return deltas


def bootstrap_ci(data, n_boot=2000, ci=0.95):
    if len(data) < 3:
        return float(np.mean(data)), (float(np.mean(data)), float(np.mean(data)))
    data = np.array(data)
    boots = [np.mean(np.random.choice(data, len(data), replace=True)) for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [(1-ci)/2*100, (1+ci)/2*100])
    return float(np.mean(data)), (float(lo), float(hi))


# =============================================
# STEP 5: UNSUPERVISED STRUCTURE DISCOVERY
# =============================================
def unsupervised_structure(all_subj_acts, li, template_names):
    """Try to find number clusters WITHOUT using labels"""
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, silhouette_score
    
    # Combine all templates for this layer
    all_acts = []
    all_labels = []  # 0=sing, 1=plur (for evaluation only)
    all_templates = []
    
    for tname in template_names:
        sing, plur = all_subj_acts[tname]
        s = np.array(sing[li])
        p = np.array(plur[li])
        if len(s) > 0:
            all_acts.append(s)
            all_labels.extend([0]*len(s))
            all_templates.extend([tname]*len(s))
        if len(p) > 0:
            all_acts.append(p)
            all_labels.extend([1]*len(p))
            all_templates.extend([tname]*len(p))
    
    if len(all_acts) == 0:
        return None
    
    X = np.vstack(all_acts)  # [N, d]
    y = np.array(all_labels)
    
    # PCA to reduce dimensionality
    pca = PCA(n_components=min(50, X.shape[0]-1, X.shape[1]))
    X_pca = pca.fit_transform(X)
    
    # KMeans with 2 clusters (sing/plur) — but WITHOUT using labels
    km = KMeans(n_clusters=2, n_init=10, random_state=42)
    pred = km.fit_predict(X_pca[:, :10])  # Use top-10 PCs
    
    # Evaluate against true labels (ARI)
    ari = adjusted_rand_score(y, pred)
    
    # Silhouette score (higher = better clusters)
    sil = silhouette_score(X_pca[:, :10], pred) if len(set(pred)) > 1 else 0
    
    # Control: cluster by template instead of number
    template_labels = [template_names.index(t) for t in all_templates]
    ari_template = adjusted_rand_score(template_labels, pred)
    
    return {
        "ari_number": float(ari),
        "ari_template": float(ari_template),
        "silhouette": float(sil),
        "n_samples": len(X),
        "top5_var": float(pca.explained_variance_ratio_[:5].sum()),
    }


# =============================================
# MAIN
# =============================================
def run_phase62(model, tokenizer, device, info):
    d_model = info.d_model
    layers = get_layers(model)
    
    print("=" * 70)
    print("★★★ Phase 62: Position-Controlled Syntax Analysis ★★★")
    print("Breaking the position-syntax confound")
    print("=" * 70)
    print(f"\nTemplates: {len(TEMPLATES)}, Pairs: {len(NVA_PAIRS)}, Layers: {TARGET_LAYERS}")
    
    # ===== Step 0: Collect activations per template =====
    print("\n" + "=" * 70)
    print("Step 0: Collecting activations per template")
    print("=" * 70)
    
    template_data = {}  # name -> (sing_subj, plur_subj, sing_verb, plur_verb, n_valid)
    
    for tmpl_str, tmpl_name in TEMPLATES:
        print(f"\n  Template: {tmpl_name} = \"{tmpl_str}\"")
        ss, ps, sv, pv, nv = collect_template_acts(
            model, tokenizer, device, tmpl_str, NVA_PAIRS, 
            TARGET_LAYERS, label=tmpl_name
        )
        template_data[tmpl_name] = {
            "sing_subj": ss, "plur_subj": ps,
            "sing_verb": sv, "plur_verb": pv, "n_valid": nv
        }
    
    # ===== Step 1: Within-position syntax directions =====
    print("\n" + "=" * 70)
    print("STEP 1: Within-position syntax direction extraction")
    print("=" * 70)
    
    all_directions = {}  # (li, tname) -> direction
    all_pcas = {}        # (li, tname) -> PCA
    
    for li in TARGET_LAYERS:
        print(f"\n  Layer {li}:")
        for tname in [tn for _, tn in TEMPLATES]:
            td = template_data[tname]
            d_num, pca, n = within_position_syntax(
                td["sing_subj"], td["plur_subj"], li
            )
            all_directions[(li, tname)] = d_num
            all_pcas[(li, tname)] = pca
            if d_num is not None:
                print(f"    {tname}: n={n}, ||d_num||={np.linalg.norm(d_num):.4f}, "
                      f"PCA top-5 var={pca.explained_variance_ratio_[:5].sum():.3f}")
            else:
                print(f"    {tname}: insufficient data")
    
    # ===== Step 2: Cross-position direction comparison =====
    print("\n" + "=" * 70)
    print("STEP 2: Cross-position syntax direction comparison")
    print("=" * 70)
    
    for li in TARGET_LAYERS:
        dirs = {tn: all_directions.get((li, tn)) for _, tn in TEMPLATES}
        pcas = {tn: all_pcas.get((li, tn)) for _, tn in TEMPLATES}
        tnames = [tn for _, tn in TEMPLATES]
        cross_position_compare(dirs, pcas, tnames, li)
    
    # ===== Step 3: Conditional CCA (within position) =====
    print("\n" + "=" * 70)
    print("STEP 3: Conditional number variance (within position — NO confound)")
    print("=" * 70)
    
    for li in TARGET_LAYERS:
        print(f"\n  Layer {li}:")
        for tname in [tn for _, tn in TEMPLATES]:
            td = template_data[tname]
            result = conditional_cca(td["sing_subj"], td["plur_subj"], tname, li)
            if result:
                print(f"    {tname}: number_var_fraction={result['fraction']:.6f} "
                      f"CI=[{result['ci'][0]:.6f}, {result['ci'][1]:.6f}] "
                      f"n={result['n_sing']}+{result['n_plur']} "
                      f"||d||={result['d_num_norm']:.2f}")
    
    # ===== Step 4: Cross-position subspace patching =====
    print("\n" + "=" * 70)
    print("STEP 4: Cross-position subspace patching + Bootstrap CI")
    print("=" * 70)
    
    # Use T1 (pos2) as source, patch to T2 (pos3) and T3 (pos4)
    # Also: T2 → T1 as reverse test
    patch_tests = [
        ("T1_pos2", "T2_pos3", "T1→T2 (pos2→pos3)"),
        ("T1_pos2", "T3_pos4", "T1→T3 (pos2→pos4)"),
        ("T2_pos3", "T1_pos2", "T2→T1 (pos3→pos2)"),
    ]
    
    for li in TARGET_LAYERS:
        print(f"\n  Layer {li}:")
        d_src = all_directions.get((li, "T1_pos2"))
        if d_src is None:
            print("    T1 direction unavailable, skipping")
            continue
        
        for src_name, tgt_name, desc in patch_tests:
            d = all_directions.get((li, src_name))
            if d is None:
                continue
            src_tmpl = [t for t, n in TEMPLATES if n == src_name][0]
            tgt_tmpl = [t for t, n in TEMPLATES if n == tgt_name][0]
            
            # Before-LN patching
            deltas_before = cross_position_patching(
                model, tokenizer, device, [li], src_tmpl, tgt_tmpl,
                NVA_PAIRS[:50], d, alpha=2.0, n_test=50, patch_after_ln=False
            )
            # After-LN patching
            deltas_after = cross_position_patching(
                model, tokenizer, device, [li], src_tmpl, tgt_tmpl,
                NVA_PAIRS[:30], d, alpha=2.0, n_test=30, patch_after_ln=True
            )
            
            if deltas_before:
                mean_b, ci_b = bootstrap_ci(deltas_before)
                neg_b = np.mean([d < 0 for d in deltas_before])
                sig_b = "★" if ci_b[1] < 0 else ""
                print(f"    {desc} before-LN: Δ={mean_b:.4f} [{ci_b[0]:.4f},{ci_b[1]:.4f}] "
                      f"neg={neg_b:.0%} n={len(deltas_before)} {sig_b}")
            else:
                print(f"    {desc} before-LN: no data")
            
            if deltas_after:
                mean_a, ci_a = bootstrap_ci(deltas_after)
                neg_a = np.mean([d < 0 for d in deltas_after])
                sig_a = "★" if ci_a[1] < 0 else ""
                print(f"    {desc} after-LN:  Δ={mean_a:.4f} [{ci_a[0]:.4f},{ci_a[1]:.4f}] "
                      f"neg={neg_a:.0%} n={len(deltas_after)} {sig_a}")
            else:
                print(f"    {desc} after-LN:  no data")
            
            gc.collect(); torch.cuda.empty_cache()
    
    # ===== Step 5: Unsupervised structure =====
    print("\n" + "=" * 70)
    print("STEP 5: Unsupervised structure discovery (no labels)")
    print("=" * 70)
    
    all_subj_for_cluster = {}
    for tname in [tn for _, tn in TEMPLATES]:
        td = template_data[tname]
        all_subj_for_cluster[tname] = (td["sing_subj"], td["plur_subj"])
    
    for li in TARGET_LAYERS:
        print(f"\n  Layer {li}:")
        result = unsupervised_structure(all_subj_for_cluster, li, 
                                         [tn for _, tn in TEMPLATES])
        if result:
            print(f"    ARI(number)={result['ari_number']:.4f}, "
                  f"ARI(template)={result['ari_template']:.4f}, "
                  f"silhouette={result['silhouette']:.4f}, "
                  f"top5_var={result['top5_var']:.3f}, "
                  f"n={result['n_samples']}")
            if result['ari_number'] > result['ari_template']:
                print(f"    ★ Number clusters > Template clusters → syntax signal exists beyond position")
            else:
                print(f"    ⚠ Template clusters > Number clusters → position dominates")
    
    # ===== FINAL SUMMARY =====
    print("\n" + "=" * 70)
    print("FINAL SUMMARY: Phase 62")
    print("=" * 70)
    
    # Cross-position direction consistency
    print("\n--- Cross-Position Direction Consistency ---")
    for li in TARGET_LAYERS:
        dirs = {tn: all_directions.get((li, tn)) for _, tn in TEMPLATES}
        valid = {k: v for k, v in dirs.items() if v is not None}
        if len(valid) >= 2:
            names = list(valid.keys())
            # Compute mean pairwise cosine
            cos_vals = []
            for i in range(len(names)):
                for j in range(i+1, len(names)):
                    cos_vals.append(abs(float(np.dot(valid[names[i]], valid[names[j]]))))
            print(f"  L{li}: mean|cos|={np.mean(cos_vals):.4f} "
                  f"min={np.min(cos_vals):.4f} max={np.max(cos_vals):.4f} "
                  f"({len(valid)} directions)")
        else:
            print(f"  L{li}: insufficient directions")
    
    # Number variance fraction (position-controlled)
    print("\n--- Number Variance Fraction (position-controlled) ---")
    for li in TARGET_LAYERS:
        fracs = []
        for tname in [tn for _, tn in TEMPLATES]:
            td = template_data[tname]
            result = conditional_cca(td["sing_subj"], td["plur_subj"], tname, li)
            if result:
                fracs.append(result['fraction'])
        if fracs:
            print(f"  L{li}: mean={np.mean(fracs):.6f} [{np.min(fracs):.6f}, {np.max(fracs):.6f}]")
    
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deepseek7b")
    args = parser.parse_args()
    
    model, tokenizer, device = load_model(args.model)
    info = get_model_info(model, args.model)
    try:
        run_phase62(model, tokenizer, device, info)
    finally:
        release_model(model)
