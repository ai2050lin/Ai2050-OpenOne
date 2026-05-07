"""
Phase 63: NCSS Verification — LayerNorm-Induced Control Failure + Nonlinear Intervention
========================================================================================

THEORETICAL BASIS:
  NCSS (Nonlinear Coupled Subspace System) v0.1:
  - Transformer representations are NOT linear subspace sums
  - They are nonlinear coupled systems under LayerNorm constraints
  - Syntax = conditional vector field v(x), NOT fixed direction
  - Linear additive interventions fail because LN distorts them

CORE THEOREM (LayerNorm-Induced Control Failure):
  For any non-degenerate x and direction v:
    LN(x + αv) ≠ LN(x) + αv
  
  In high dimensions (d >> 1):
    ∂/∂α G(LN(x + αv)) → 0
  
  i.e., linear additive interventions are systematically attenuated by LN.

THREE KEY EXPERIMENTS:

Experiment 1: LN-Aware Patching (Test the Control Failure Theorem)
  - Patch in post-LN space: x̃' = LN(x) + αv
  - Compare with pre-LN patching: x' = x + αv → LN(x')
  - If theorem is correct: post-LN patching should be MUCH stronger
  - Quantify the attenuation factor: |Δ_post-LN| / |Δ_pre-LN|

Experiment 2: Subspace Projection Control (Replace additive with projective)
  - Instead of x + αv, use: x' = x + P_syn(x) - x
  - Or: x' = P_syn(x) only (zero out position component)
  - Test: is syntax a functional component accessible via projection?

Experiment 3: Attention Routing Intervention (The real mechanism?)
  - Instead of modifying residual stream, modify attention patterns
  - Swap attention from subject position across templates
  - If syntax is routing: attention intervention should have LARGE effect

DATA: 82 NVA pairs × 5 templates × 2 numbers = 820 sentences
LARGER than Phase 62 to address power analysis concern.

Also: Separate regular vs irregular nouns for confound analysis.
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

# ===== NVA PAIRS (classified by regularity) =====
REGULAR_NVA = [
    ("cat","cats","runs","run"),("dog","dogs","walks","walk"),("bird","birds","flies","fly"),
    ("girl","girls","reads","read"),("boy","boys","sings","sing"),("horse","horses","jumps","jump"),
    ("bear","bears","sleeps","sleep"),("snake","snakes","crawls","crawl"),
    ("frog","frogs","swims","swim"),("fox","foxes","hunts","hunt"),("king","kings","rules","rule"),
    ("student","students","studies","study"),("teacher","teachers","speaks","speak"),
    ("doctor","doctors","helps","help"),("tree","trees","grows","grow"),("car","cars","moves","move"),
    ("queen","queens","leads","lead"),("driver","drivers","drives","drive"),
    ("worker","workers","builds","build"),("player","players","wins","win"),
    ("writer","writers","writes","write"),("rabbit","rabbits","hops","hop"),
    ("eagle","eagles","soars","soar"),("tiger","tigers","stalks","stalk"),
    ("monkey","monkeys","climbs","climb"),("lion","lions","roars","roar"),
    ("farmer","farmers","plants","plant"),("robot","robots","moves","move"),
    ("wizard","wizards","casts","cast"),("dragon","dragons","breathes","breathe"),
    ("angel","angels","flies","fly"),("soldier","soldiers","marches","march"),
    ("nurse","nurses","cares","care"),("artist","artists","paints","paint"),
    ("dancer","dancers","spins","spin"),("singer","singers","sings","sing"),
    ("hunter","hunters","tracks","track"),("apple","apples","falls","fall"),
    ("river","rivers","flows","flow"),("cloud","clouds","drifts","drift"),
    ("star","stars","shines","shine"),("moon","moons","orbits","orbit"),
    ("book","books","opens","open"),("phone","phones","rings","ring"),
    ("clock","clocks","ticks","tick"),("train","trains","arrives","arrive"),
    ("plane","planes","lands","land"),("ship","ships","sails","sail"),
]

IRREGULAR_NVA = [
    ("man","men","works","work"),("child","children","plays","play"),
    ("wolf","wolves","howls","howl"),("fish","fish","swims","swim"),
    ("knife","knives","cuts","cut"),("mouse","mice","squeaks","squeak"),
    ("goose","geese","flies","fly"),("tooth","teeth","bites","bite"),
    ("foot","feet","steps","step"),("person","people","speaks","speak"),
    ("hero","heroes","fights","fight"),("potato","potatoes","grows","grow"),
    ("boss","bosses","leads","lead"),("baby","babies","cries","cry"),
    ("lady","ladies","dances","dance"),("story","stories","ends","end"),
    ("city","cities","shines","shine"),("party","parties","starts","start"),
    ("thief","thieves","steals","steal"),("leaf","leaves","falls","fall"),
    ("life","lives","begins","begin"),("wife","wives","cooks","cook"),
    ("calf","calves","runs","run"),("half","halves","breaks","break"),
    ("self","selves","acts","act"),("shelf","shelves","holds","hold"),
    ("ox","oxen","pulls","pull"),("louse","lice","crawls","crawl"),
    ("datum","data","shows","show"),
]

ALL_NVA = REGULAR_NVA + IRREGULAR_NVA

NOUNS_SET = set()
for sn, pn, _, _ in ALL_NVA:
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


def get_input_ln(layer):
    """Get the input LayerNorm module for a layer (handles RMSNorm)"""
    for name, mod in layer.named_children():
        if 'input_layernorm' in name or name == 'ln_1':
            return mod
    return None


def get_post_attn_ln(layer):
    """Get the post-attention LayerNorm module"""
    for name, mod in layer.named_children():
        if 'post_attention_layernorm' in name or name == 'ln_2':
            return mod
    return None


def collect_acts_by_noun_type(model, tokenizer, device, tmpl_str, nva_pairs,
                               target_layers, label=""):
    """Collect activations, separated by regular/irregular and number"""
    layers_list = get_layers(model)
    result = {
        'regular': {'sing_subj': defaultdict(list), 'plur_subj': defaultdict(list)},
        'irregular': {'sing_subj': defaultdict(list), 'plur_subj': defaultdict(list)},
    }
    n_valid = 0

    regular_nouns = set()
    for sn, pn, _, _ in REGULAR_NVA:
        regular_nouns.add(sn.lower())
        regular_nouns.add(pn.lower())

    for si, (sn, pn, sv, pv) in enumerate(nva_pairs):
        if si % 20 == 0 and si > 0:
            print(f"  {label} {si}/{len(nva_pairs)}")

        # Determine regularity
        is_regular = sn.lower() in regular_nouns
        noun_type = 'regular' if is_regular else 'irregular'

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
                if li < len(layers_list):
                    hooks.append(layers_list[li].register_forward_hook(mk_hook(li)))

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
                        target = result[noun_type]['sing_subj'] if is_sing else result[noun_type]['plur_subj']
                        target[li].append(h[0, sp, :].float().numpy())
            n_valid += 1
            del captured; gc.collect(); torch.cuda.empty_cache()

    print(f"  {label}: {n_valid} valid")
    return result


# ============================================================
# EXPERIMENT 1: LN-Aware Patching (Control Failure Theorem)
# ============================================================
def ln_aware_patching(model, tokenizer, device, info, target_layers, nva_pairs, alpha=2.0, n_test=60):
    """
    Three-way comparison of intervention methods:
    
    A. Pre-LN patching (current method): x' = x + αv, then LN(x')
    B. Post-LN patching: x̃' = LN(x) + αv  (patch in model's reading space)
    C. LN-corrected patching: x' such that LN(x') = LN(x) + αv
    
    If NCSS/Control Failure Theorem is correct:
    - B should be MUCH stronger than A
    - C should be exactly right (if we can compute it)
    """
    layers_list = get_layers(model)
    d_model = info.d_model
    
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: LN-Aware Patching (Testing Control Failure Theorem)")
    print("=" * 70)
    print(f"  Testing: LN(x + αv) ≠ LN(x) + αv")
    print(f"  Prediction: post-LN patching >> pre-LN patching")
    print(f"  α = {alpha}, n_test = {n_test}")
    
    # Collect syntax direction from T1 template
    # Use ONLY regular nouns for direction extraction (cleaner signal)
    print("\n  Collecting syntax direction from T1 (regular nouns)...")
    sing_subj = defaultdict(list)
    plur_subj = defaultdict(list)
    
    for sn, pn, sv, pv in REGULAR_NVA:
        for is_sing, (noun, verb) in [(True, (sn, sv)), (False, (pn, pv))]:
            sent = f"The {noun} {verb}"
            toks = tokenizer.encode(sent, add_special_tokens=False)
            sp = find_noun_pos(tokenizer, toks)
            if sp is None:
                continue
            
            captured = {}
            def mk_hook(li):
                def fn(m, inp, out):
                    captured[li] = (out[0] if isinstance(out, tuple) else out).detach().cpu()
                return fn
            
            hooks = []
            for li in target_layers:
                if li < len(layers_list):
                    hooks.append(layers_list[li].register_forward_hook(mk_hook(li)))
            
            ids = tokenizer(sent, return_tensors="pt").to(device)
            with torch.no_grad():
                try: model(**ids)
                except:
                    for h in hooks: h.remove()
                    continue
            for h in hooks: h.remove()
            
            for li in target_layers:
                if li in captured and sp < captured[li].shape[1]:
                    target = sing_subj if is_sing else plur_subj
                    target[li].append(captured[li][0, sp, :].float().numpy())
            
            del captured; gc.collect(); torch.cuda.empty_cache()
    
    # Compute syntax directions per layer
    syntax_dirs = {}
    for li in target_layers:
        s = np.array(sing_subj[li])
        p = np.array(plur_subj[li])
        if len(s) >= 3 and len(p) >= 3:
            d_num = p.mean(axis=0) - s.mean(axis=0)
            d_num = d_num / (np.linalg.norm(d_num) + 1e-10)
            syntax_dirs[li] = d_num
            print(f"    L{li}: direction extracted, n_s={len(s)}, n_p={len(p)}")
    
    if not syntax_dirs:
        print("  ERROR: No syntax directions extracted!")
        return
    
    # Now test three patching methods
    # Use T2 template as target (different position from T1)
    print("\n  Testing three patching methods on T2 targets...")
    
    results = {li: {'pre_ln': [], 'post_ln': [], 'ln_corrected': []} for li in target_layers}
    
    for si, (sn, pn, sv, pv) in enumerate(nva_pairs[:n_test]):
        # Target: singular subject in T2 → add plural direction → should shift toward plural
        tgt_sent = f"Today, the {sn} {sv}"
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
        
        for li in target_layers:
            if li not in syntax_dirs:
                continue
            
            d_num = syntax_dirs[li]
            direction_t = torch.tensor(d_num, dtype=torch.float32, device=device)
            layer = layers_list[li]
            input_ln = get_input_ln(layer)
            if input_ln is None:
                continue
            
            # === Method A: Pre-LN patching (x + αv, then LN processes it) ===
            captured_a = {}
            applied_a = [False]
            
            def pre_ln_hook(m, inp, out, _li=li, _sp=tgt_sp, _dir=direction_t, _alpha=alpha):
                if not applied_a[0]:
                    out_mod = out[0] if isinstance(out, tuple) else out
                    p = out_mod.clone()
                    # Add direction to residual (pre-LN from next layer's perspective)
                    p[:, _sp, :] += (_alpha * _dir).to(p.dtype)
                    applied_a[0] = True
                    return (p,) + out[1:] if isinstance(out, tuple) else p
                return out
            
            hook_a = layer.register_forward_hook(pre_ln_hook)
            with torch.no_grad():
                patched_a_logits = model(**ids).logits.detach().cpu()
            hook_a.remove()
            patched_a_agr = (patched_a_logits[0, tgt_vp, sv_ids[0]] - 
                           patched_a_logits[0, tgt_vp, pv_ids[0]]).item()
            delta_a = patched_a_agr - base_agr
            results[li]['pre_ln'].append(delta_a)
            
            # === Method B: Post-LN patching (LN(x) + αv) ===
            applied_b = [False]
            
            def post_ln_hook(m, inp, out, _li=li, _sp=tgt_sp, _dir=direction_t, _alpha=alpha):
                if not applied_b[0]:
                    out_mod = out[0] if isinstance(out, tuple) else out
                    p = out_mod.clone()
                    # Add direction AFTER LayerNorm (in model's reading space)
                    p[:, _sp, :] += (_alpha * _dir).to(p.dtype)
                    applied_b[0] = True
                    return (p,) + out[1:] if isinstance(out, tuple) else p
                return out
            
            hook_b = input_ln.register_forward_hook(post_ln_hook)
            with torch.no_grad():
                patched_b_logits = model(**ids).logits.detach().cpu()
            hook_b.remove()
            patched_b_agr = (patched_b_logits[0, tgt_vp, sv_ids[0]] - 
                           patched_b_logits[0, tgt_vp, pv_ids[0]]).item()
            delta_b = patched_b_agr - base_agr
            results[li]['post_ln'].append(delta_b)
            
            # === Method C: LN-corrected patching ===
            # Compute x' such that LN(x') ≈ LN(x) + αv
            # Approximate: x' = x + αv * σ(x) / γ
            # This accounts for the variance scaling in LN
            # First capture the pre-LN activation
            captured_c_pre = {}
            
            def capture_pre_ln(m, inp, out, _li=li, _sp=tgt_sp):
                # Capture input to LN (= pre-LN residual)
                if isinstance(inp, tuple):
                    captured_c_pre[_li] = inp[0].detach().clone()
                else:
                    captured_c_pre[_li] = inp.detach().clone()
            
            hook_capture = input_ln.register_forward_hook(capture_pre_ln)
            with torch.no_grad():
                model(**ids)
            hook_capture.remove()
            
            if li in captured_c_pre:
                pre_ln_act = captured_c_pre[li]  # [1, seq, d]
                pre_vec = pre_ln_act[0, tgt_sp, :].float()  # [d]
                
                # Compute LN's scaling factor
                with torch.no_grad():
                    ln_out = input_ln(pre_ln_act)  # [1, seq, d]
                    ln_vec = ln_out[0, tgt_sp, :].float()  # [d]
                
                # The correction: if LN(x) = γ * (x - μ) / σ + β
                # Then to get LN(x') = LN(x) + αv, we need:
                # x' - μ' = (σ'/σ)(x - μ) + αv*σ'/γ
                # Approximation: σ' ≈ σ, μ' ≈ μ (for small perturbation)
                # Then: x' ≈ x + αv * σ / γ
                
                # Compute σ (standard deviation used by LN)
                mu = pre_vec.mean()
                sigma = pre_vec.std()
                
                # Get γ (gain) from LN
                gamma = input_ln.weight.float() if hasattr(input_ln, 'weight') else torch.ones(d_model, device=device)
                # RMSNorm doesn't have bias
                
                # Corrected direction: scale by sigma/gamma to compensate for LN's normalization
                corrected_dir = direction_t * (sigma / (gamma.to(device) + 1e-10))
                # Normalize to same total magnitude as original direction
                orig_mag = torch.norm(direction_t)
                corr_mag = torch.norm(corrected_dir)
                if corr_mag > 1e-10:
                    corrected_dir = corrected_dir * (orig_mag / corr_mag)
                
                applied_c = [False]
                
                def corrected_hook(m, inp, out, _li=li, _sp=tgt_sp, _cdir=corrected_dir, _alpha=alpha):
                    if not applied_c[0]:
                        out_mod = out[0] if isinstance(out, tuple) else out
                        p = out_mod.clone()
                        p[:, _sp, :] += (_alpha * _cdir).to(p.dtype)
                        applied_c[0] = True
                        return (p,) + out[1:] if isinstance(out, tuple) else p
                    return out
                
                hook_c = layer.register_forward_hook(corrected_hook)
                with torch.no_grad():
                    patched_c_logits = model(**ids).logits.detach().cpu()
                hook_c.remove()
                patched_c_agr = (patched_c_logits[0, tgt_vp, sv_ids[0]] - 
                               patched_c_logits[0, tgt_vp, pv_ids[0]]).item()
                delta_c = patched_c_agr - base_agr
                results[li]['ln_corrected'].append(delta_c)
            
            del captured_c_pre; gc.collect(); torch.cuda.empty_cache()
    
    # === ANALYSIS ===
    print("\n" + "=" * 70)
    print("EXPERIMENT 1 RESULTS: LN-Aware Patching Comparison")
    print("=" * 70)
    
    for li in target_layers:
        print(f"\n  Layer {li}:")
        for method in ['pre_ln', 'post_ln', 'ln_corrected']:
            data = results[li][method]
            if len(data) < 3:
                print(f"    {method}: insufficient data ({len(data)})")
                continue
            
            mean_d = np.mean(data)
            std_d = np.std(data)
            # Bootstrap CI
            n_boot = 2000
            boots = [np.mean(np.random.choice(data, len(data), replace=True)) for _ in range(n_boot)]
            ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
            # Signal-to-noise ratio
            snr = abs(mean_d) / (std_d + 1e-10)
            # Effect significance
            sig = "★" if (ci_lo > 0 or ci_hi < 0) else ""
            # Direction correctness (should be negative = shift toward plural)
            neg_pct = np.mean([d < 0 for d in data])
            
            print(f"    {method:15s}: Δ={mean_d:+.4f} std={std_d:.4f} "
                  f"CI=[{ci_lo:+.4f},{ci_hi:+.4f}] SNR={snr:.2f} "
                  f"neg={neg_pct:.0%} n={len(data)} {sig}")
        
        # Attenuation ratio (key prediction of Control Failure Theorem)
        pre = results[li]['pre_ln']
        post = results[li]['post_ln']
        if len(pre) >= 3 and len(post) >= 3:
            # If theorem correct: |post| > |pre|
            mean_pre = abs(np.mean(pre))
            mean_post = abs(np.mean(post))
            ratio = mean_post / (mean_pre + 1e-10)
            print(f"    *** Attenuation ratio |post-LN|/|pre-LN| = {ratio:.2f} ***")
            if ratio > 1.5:
                print(f"    ★★★ CONFIRMED: Post-LN patching is {ratio:.1f}x stronger than pre-LN!")
                print(f"    ★★★ This supports the Control Failure Theorem!")
            elif ratio < 0.67:
                print(f"    ⚠⚠⚠ SURPRISE: Pre-LN patching is stronger! Control Failure not confirmed.")
            else:
                print(f"    → Roughly equal strength. LN attenuation not the main factor.")
    
    return results


# ============================================================
# EXPERIMENT 2: Subspace Projection Control
# ============================================================
def subspace_projection_control(model, tokenizer, device, info, target_layers, n_test=60):
    """
    Instead of x + αv, project onto syntax subspace:
    
    Method 1: Zero-out position component
      x' = x - P_pos(x)  (remove position, keep syntax+residual)
    
    Method 2: Amplify syntax component
      x' = x + β * P_syn(x)  (amplify syntax in the projection)
    
    Method 3: Swap syntax between sentences
      x'_A = x_A - P_syn(x_A) + P_syn(x_B)  (replace A's syntax with B's)
    
    If syntax is a functional component (not just epiphenomenon):
    - Method 1 should degrade agreement accuracy
    - Method 2 should strengthen agreement
    - Method 3 should swap the agreement pattern
    """
    layers_list = get_layers(model)
    d_model = info.d_model
    
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Subspace Projection Control")
    print("=" * 70)
    print("  Testing: Is syntax a functional component accessible via projection?")
    
    # Collect sing/plur subspaces per layer
    print("\n  Collecting subspaces...")
    sing_subj = defaultdict(list)
    plur_subj = defaultdict(list)
    
    # Also collect position-specific activations for position subspace
    pos_acts = defaultdict(list)  # li -> [acts], labeled by template
    pos_labels = defaultdict(list)
    
    for tmpl_str, tmpl_name in TEMPLATES:
        for sn, pn, sv, pv in REGULAR_NVA:  # Use regular nouns only
            for is_sing, (noun, verb) in [(True, (sn, sv)), (False, (pn, pv))]:
                sent = tmpl_str.format(n=noun, v=verb)
                toks = tokenizer.encode(sent, add_special_tokens=False)
                sp = find_noun_pos(tokenizer, toks)
                if sp is None:
                    continue
                
                captured = {}
                def mk_hook(li):
                    def fn(m, inp, out):
                        captured[li] = (out[0] if isinstance(out, tuple) else out).detach().cpu()
                    return fn
                
                hooks = []
                for li in target_layers:
                    if li < len(layers_list):
                        hooks.append(layers_list[li].register_forward_hook(mk_hook(li)))
                
                ids = tokenizer(sent, return_tensors="pt").to(device)
                with torch.no_grad():
                    try: model(**ids)
                    except:
                        for h in hooks: h.remove()
                        continue
                for h in hooks: h.remove()
                
                for li in target_layers:
                    if li in captured and sp < captured[li].shape[1]:
                        act = captured[li][0, sp, :].float().numpy()
                        if is_sing:
                            sing_subj[li].append(act)
                        else:
                            plur_subj[li].append(act)
                        pos_acts[li].append(act)
                        pos_labels[li].append(tmpl_name)
                
                del captured; gc.collect(); torch.cuda.empty_cache()
    
    # Compute syntax subspace (difference-of-means direction) and position subspace (PCA)
    from sklearn.decomposition import PCA
    
    syntax_dirs = {}
    syntax_subspaces = {}  # PCA components
    position_subspaces = {}
    
    for li in target_layers:
        s = np.array(sing_subj[li])
        p = np.array(plur_subj[li])
        all_acts = np.vstack([s, p])
        
        # Syntax direction
        d_num = p.mean(axis=0) - s.mean(axis=0)
        d_num_norm = d_num / (np.linalg.norm(d_num) + 1e-10)
        syntax_dirs[li] = d_num_norm
        
        # Syntax subspace (PCA on within-class differences)
        diff = np.vstack([p - s.mean(axis=0), s - p.mean(axis=0)])
        n_comp = min(20, diff.shape[0]-1, diff.shape[1])
        pca_syn = PCA(n_components=n_comp)
        pca_syn.fit(diff)
        syntax_subspaces[li] = pca_syn.components_  # [n_comp, d]
        
        # Position subspace (PCA on all activations, labeled by template)
        X_pos = np.array(pos_acts[li])
        n_pos_comp = min(20, X_pos.shape[0]-1, X_pos.shape[1])
        pca_pos = PCA(n_components=n_pos_comp)
        pca_pos.fit(X_pos)
        position_subspaces[li] = pca_pos.components_  # [n_comp, d]
        
        # Number variance fraction
        total_var = np.var(all_acts, axis=0).sum()
        num_var = np.dot(d_num, d_num) * len(s) * len(p) / (len(s) + len(p))
        frac = num_var / (total_var + 1e-10)
        
        print(f"    L{li}: n_s={len(s)}, n_p={len(p)}, "
              f"syn_subspace={pca_syn.components_.shape[0]}, "
              f"pos_subspace={pca_pos.components_.shape[0]}, "
              f"num_var_frac={frac:.6f}")
    
    # Test projection interventions
    print("\n  Testing projection interventions on T2 targets...")
    
    results = {li: {'baseline': [], 'remove_pos': [], 'amplify_syn': [], 'swap_syn': []} 
               for li in target_layers}
    
    test_pairs = REGULAR_NVA[:n_test]
    
    for si, (sn, pn, sv, pv) in enumerate(test_pairs):
        if si % 20 == 0 and si > 0:
            print(f"    Testing {si}/{n_test}...")
        
        # Source sentence (plural) — we'll swap its syntax direction
        src_sent = f"The {pn} {pv}"
        src_toks = tokenizer.encode(src_sent, add_special_tokens=False)
        src_sp = find_noun_pos(tokenizer, src_toks)
        
        # Target sentence (singular) — should shift toward plural after swap
        tgt_sent = f"Today, the {sn} {sv}"
        tgt_toks = tokenizer.encode(tgt_sent, add_special_tokens=False)
        tgt_sp = find_noun_pos(tokenizer, tgt_toks)
        tgt_vp = find_verb_pos(tokenizer, tgt_toks, sv, pv)
        if tgt_sp is None or tgt_vp is None or src_sp is None:
            continue
        
        tgt_ids = tokenizer(tgt_sent, return_tensors="pt").to(device)
        sv_ids = tokenizer.encode(sv, add_special_tokens=False)
        pv_ids = tokenizer.encode(pv, add_special_tokens=False)
        if not sv_ids or not pv_ids:
            continue
        
        # Base logits for target
        with torch.no_grad():
            base_logits = model(**tgt_ids).logits.detach().cpu()
        base_agr = (base_logits[0, tgt_vp, sv_ids[0]] - base_logits[0, tgt_vp, pv_ids[0]]).item()
        
        for li in target_layers:
            results[li]['baseline'].append(base_agr)
            
            # Capture target activation at subject position
            captured_tgt = {}
            def mk_cap_hook(li_idx):
                def fn(m, inp, out):
                    captured_tgt[li_idx] = (out[0] if isinstance(out, tuple) else out).detach().cpu()
                return fn
            
            hooks = []
            for li2 in target_layers:
                if li2 < len(layers_list):
                    hooks.append(layers_list[li2].register_forward_hook(mk_cap_hook(li2)))
            
            with torch.no_grad():
                model(**tgt_ids)
            for h in hooks: h.remove()
            
            if li not in captured_tgt:
                continue
            
            tgt_act = captured_tgt[li][0, tgt_sp, :].float().numpy()  # [d]
            
            # === Method 1: Remove position component ===
            # Project out top-k position PCs
            pos_basis = position_subspaces[li][:5]  # top-5 position PCs
            P_pos = pos_basis.T @ pos_basis  # Projection matrix
            tgt_no_pos = tgt_act - P_pos @ tgt_act
            
            # Add back as residual modification
            delta_remove_pos = tgt_no_pos - tgt_act
            delta_t = torch.tensor(delta_remove_pos, dtype=torch.float32, device=device)
            
            applied = [False]
            def hook_remove_pos(m, inp, out, _sp=tgt_sp, _delta=delta_t):
                if not applied[0]:
                    out_mod = out[0] if isinstance(out, tuple) else out
                    p = out_mod.clone()
                    p[:, _sp, :] += _delta.to(p.dtype)
                    applied[0] = True
                    return (p,) + out[1:] if isinstance(out, tuple) else p
                return out
            
            hook = layers_list[li].register_forward_hook(hook_remove_pos)
            with torch.no_grad():
                patched_logits = model(**tgt_ids).logits.detach().cpu()
            hook.remove()
            patched_agr = (patched_logits[0, tgt_vp, sv_ids[0]] - 
                         patched_logits[0, tgt_vp, pv_ids[0]]).item()
            results[li]['remove_pos'].append(patched_agr - base_agr)
            
            # === Method 2: Amplify syntax component ===
            syn_basis = syntax_subspaces[li][:5]  # top-5 syntax PCs
            P_syn = syn_basis.T @ syn_basis
            syn_component = P_syn @ tgt_act
            # Amplify by factor β=3
            delta_amplify = 2.0 * syn_component  # β-1 times the projection
            delta_t2 = torch.tensor(delta_amplify, dtype=torch.float32, device=device)
            
            applied2 = [False]
            def hook_amplify(m, inp, out, _sp=tgt_sp, _delta=delta_t2):
                if not applied2[0]:
                    out_mod = out[0] if isinstance(out, tuple) else out
                    p = out_mod.clone()
                    p[:, _sp, :] += _delta.to(p.dtype)
                    applied2[0] = True
                    return (p,) + out[1:] if isinstance(out, tuple) else p
                return out
            
            hook2 = layers_list[li].register_forward_hook(hook_amplify)
            with torch.no_grad():
                patched_logits2 = model(**tgt_ids).logits.detach().cpu()
            hook2.remove()
            patched_agr2 = (patched_logits2[0, tgt_vp, sv_ids[0]] - 
                          patched_logits2[0, tgt_vp, pv_ids[0]]).item()
            results[li]['amplify_syn'].append(patched_agr2 - base_agr)
            
            # === Method 3: Swap syntax component ===
            # Need source (plural) activation
            captured_src = {}
            hooks_src = []
            for li2 in target_layers:
                if li2 < len(layers_list):
                    hooks_src.append(layers_list[li2].register_forward_hook(mk_cap_hook(li2)))
            
            src_ids = tokenizer(src_sent, return_tensors="pt").to(device)
            with torch.no_grad():
                model(**src_ids)
            for h in hooks_src: h.remove()
            
            if li in captured_src and src_sp < captured_src[li].shape[1]:
                src_act = captured_src[li][0, src_sp, :].float().numpy()
                
                # Swap: remove target's syntax, add source's syntax
                syn_tgt = P_syn @ tgt_act
                syn_src = P_syn @ src_act
                delta_swap = syn_src - syn_tgt
                delta_t3 = torch.tensor(delta_swap, dtype=torch.float32, device=device)
                
                applied3 = [False]
                def hook_swap(m, inp, out, _sp=tgt_sp, _delta=delta_t3):
                    if not applied3[0]:
                        out_mod = out[0] if isinstance(out, tuple) else out
                        p = out_mod.clone()
                        p[:, _sp, :] += _delta.to(p.dtype)
                        applied3[0] = True
                        return (p,) + out[1:] if isinstance(out, tuple) else p
                    return out
                
                hook3 = layers_list[li].register_forward_hook(hook_swap)
                with torch.no_grad():
                    patched_logits3 = model(**tgt_ids).logits.detach().cpu()
                hook3.remove()
                patched_agr3 = (patched_logits3[0, tgt_vp, sv_ids[0]] - 
                              patched_logits3[0, tgt_vp, pv_ids[0]]).item()
                results[li]['swap_syn'].append(patched_agr3 - base_agr)
            
            del captured_tgt, captured_src; gc.collect(); torch.cuda.empty_cache()
    
    # === ANALYSIS ===
    print("\n" + "=" * 70)
    print("EXPERIMENT 2 RESULTS: Subspace Projection Control")
    print("=" * 70)
    
    for li in target_layers:
        print(f"\n  Layer {li}:")
        for method in ['remove_pos', 'amplify_syn', 'swap_syn']:
            data = results[li][method]
            if len(data) < 3:
                print(f"    {method}: insufficient data ({len(data)})")
                continue
            
            mean_d = np.mean(data)
            std_d = np.std(data)
            n_boot = 2000
            boots = [np.mean(np.random.choice(data, len(data), replace=True)) for _ in range(n_boot)]
            ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
            sig = "★" if (ci_lo > 0 or ci_hi < 0) else ""
            neg_pct = np.mean([d < 0 for d in data])
            
            print(f"    {method:15s}: Δ={mean_d:+.4f} std={std_d:.4f} "
                  f"CI=[{ci_lo:+.4f},{ci_hi:+.4f}] neg={neg_pct:.0%} n={len(data)} {sig}")
        
        # Key comparison: swap should have LARGEST effect
        swap = results[li]['swap_syn']
        amplify = results[li]['amplify_syn']
        remove = results[li]['remove_pos']
        if len(swap) >= 3 and len(amplify) >= 3:
            print(f"    *** |swap|={abs(np.mean(swap)):.4f} vs |amplify|={abs(np.mean(amplify)):.4f} ***")
            if abs(np.mean(swap)) > abs(np.mean(amplify)) * 1.5:
                print(f"    ★ Syntax swap > syntax amplify → syntax is context-dependent (NCSS)")
            else:
                print(f"    → Syntax amplify ≥ swap → syntax direction is somewhat additive")
    
    return results


# ============================================================
# EXPERIMENT 3: Attention Routing Intervention
# ============================================================
def attention_routing_intervention(model, tokenizer, device, info, target_layers, n_test=40):
    """
    Test whether syntax is carried by attention routing rather than residual directions.
    
    Method: Modify attention weights to route subject information differently.
    
    Specifically:
    1. Capture attention pattern from a PLURAL subject sentence
    2. Replace attention pattern in a SINGULAR subject sentence
    3. If syntax is routing: this should shift verb agreement
    
    This is the most critical test: if attention routing carries syntax,
    the effect should be MUCH larger than residual stream patching.
    """
    layers_list = get_layers(model)
    
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Attention Routing Intervention")
    print("=" * 70)
    print("  Testing: Is syntax carried by attention routing?")
    print("  If yes: attention swap should have MUCH larger effect than residual patching")
    
    results = {li: {'attn_swap': [], 'residual_patch': []} for li in target_layers}
    
    for si, (sn, pn, sv, pv) in enumerate(REGULAR_NVA[:n_test]):
        if si % 10 == 0 and si > 0:
            print(f"    Testing {si}/{n_test}...")
        
        # Source: plural subject sentence
        src_sent = f"The {pn} {pv}"
        # Target: singular subject sentence
        tgt_sent = f"Today, the {sn} {sv}"
        
        tgt_toks = tokenizer.encode(tgt_sent, add_special_tokens=False)
        tgt_sp = find_noun_pos(tokenizer, tgt_toks)
        tgt_vp = find_verb_pos(tokenizer, tgt_toks, sv, pv)
        if tgt_sp is None or tgt_vp is None:
            continue
        
        tgt_ids = tokenizer(tgt_sent, return_tensors="pt").to(device)
        sv_ids = tokenizer.encode(sv, add_special_tokens=False)
        pv_ids = tokenizer.encode(pv, add_special_tokens=False)
        if not sv_ids or not pv_ids:
            continue
        
        # Base logits
        with torch.no_grad():
            base_logits = model(**tgt_ids).logits.detach().cpu()
        base_agr = (base_logits[0, tgt_vp, sv_ids[0]] - base_logits[0, tgt_vp, pv_ids[0]]).item()
        
        for li in target_layers:
            # Step 1: Capture source attention pattern
            src_attn_weights = {}
            
            def capture_src_attn(m, inp, out, _li=li):
                # out from attention module includes (hidden_states, attn_weights, ...)
                if isinstance(out, tuple) and len(out) >= 2:
                    # attn_weights shape: [batch, heads, seq_q, seq_k]
                    aw = out[1]
                    if aw is not None:
                        src_attn_weights[_li] = aw.detach().clone()
            
            sa = layers_list[li].self_attn
            src_ids = tokenizer(src_sent, return_tensors="pt").to(device)
            hook_src = sa.register_forward_hook(capture_src_attn)
            with torch.no_grad():
                model(**src_ids)
            hook_src.remove()
            
            if li not in src_attn_weights:
                continue
            
            src_aw = src_attn_weights[li]  # [1, heads, seq_q, seq_k]
            
            # Step 2: Replace target attention pattern with source's
            # Only replace at verb position row (where agreement is computed)
            # This targets: how much does the verb attend to the subject?
            
            replaced = [False]
            
            def swap_attn_hook(m, inp, out, _li=li, _src_aw=src_aw, _tgt_vp=tgt_vp, _tgt_sp=tgt_sp):
                if not replaced[0]:
                    if isinstance(out, tuple) and len(out) >= 2:
                        aw = out[1]
                        if aw is not None:
                            new_aw = aw.clone()
                            # Replace attention weights at verb position
                            # Source's verb row attention to source's subject
                            # But we need to be careful about sequence length differences
                            min_seq = min(new_aw.shape[-1], _src_aw.shape[-1])
                            min_seq_q = min(new_aw.shape[-2], _src_aw.shape[-2])
                            
                            # Scale source attention to match target's normalization
                            # Just replace the attention FROM verb TO subject
                            if _tgt_vp < min_seq_q and _tgt_sp < min_seq:
                                # Option A: Replace entire verb row attention
                                if _tgt_vp < _src_aw.shape[-2]:
                                    new_aw[0, :, _tgt_vp, :min_seq] = _src_aw[0, :, _tgt_vp, :min_seq]
                                
                                # Re-normalize to sum to 1
                                new_aw = new_aw / (new_aw.sum(dim=-1, keepdim=True) + 1e-10)
                            
                            replaced[0] = True
                            return (out[0], new_aw) + out[2:]
                    replaced[0] = True
                return out
            
            hook_swap = sa.register_forward_hook(swap_attn_hook)
            with torch.no_grad():
                try:
                    swapped_logits = model(**tgt_ids).logits.detach().cpu()
                except:
                    hook_swap.remove()
                    continue
            hook_swap.remove()
            
            swapped_agr = (swapped_logits[0, tgt_vp, sv_ids[0]] - 
                         swapped_logits[0, tgt_vp, pv_ids[0]]).item()
            results[li]['attn_swap'].append(swapped_agr - base_agr)
            
            # Also do residual patching for comparison (same sentence pair)
            # Get syntax direction
            d_num_dir = None
            # Quick direction from this pair
            tgt_act_data = {}
            src_act_data = {}
            
            def cap_tgt(m, inp, out, _li=li):
                tgt_act_data[_li] = (out[0] if isinstance(out, tuple) else out).detach().cpu()
            def cap_src(m, inp, out, _li=li):
                src_act_data[_li] = (out[0] if isinstance(out, tuple) else out).detach().cpu()
            
            hook_t = layers_list[li].register_forward_hook(cap_tgt)
            with torch.no_grad():
                model(**tgt_ids)
            hook_t.remove()
            
            hook_s = layers_list[li].register_forward_hook(cap_src)
            with torch.no_grad():
                model(**src_ids)
            hook_s.remove()
            
            if li in tgt_act_data and li in src_act_data:
                tgt_h = tgt_act_data[li]
                src_h = src_act_data[li]
                if tgt_sp < tgt_h.shape[1] and tgt_sp < src_h.shape[1]:
                    # Direction: from singular toward plural
                    d_res = (src_h[0, tgt_sp, :] - tgt_h[0, tgt_sp, :]).float()
                    d_norm = torch.norm(d_res)
                    if d_norm > 1e-10:
                        d_res = d_res / d_norm
                        d_res_t = d_res.to(device)
                        
                        applied_res = [False]
                        def hook_residual(m, inp, out, _sp=tgt_sp, _d=d_res_t):
                            if not applied_res[0]:
                                out_mod = out[0] if isinstance(out, tuple) else out
                                p = out_mod.clone()
                                p[:, _sp, :] += (2.0 * _d).to(p.dtype)
                                applied_res[0] = True
                                return (p,) + out[1:] if isinstance(out, tuple) else p
                            return out
                        
                        hook_r = layers_list[li].register_forward_hook(hook_residual)
                        with torch.no_grad():
                            res_logits = model(**tgt_ids).logits.detach().cpu()
                        hook_r.remove()
                        
                        res_agr = (res_logits[0, tgt_vp, sv_ids[0]] - 
                                 res_logits[0, tgt_vp, pv_ids[0]]).item()
                        results[li]['residual_patch'].append(res_agr - base_agr)
            
            del src_attn_weights, tgt_act_data, src_act_data
            gc.collect(); torch.cuda.empty_cache()
    
    # === ANALYSIS ===
    print("\n" + "=" * 70)
    print("EXPERIMENT 3 RESULTS: Attention Routing vs Residual Patching")
    print("=" * 70)
    
    for li in target_layers:
        print(f"\n  Layer {li}:")
        for method in ['attn_swap', 'residual_patch']:
            data = results[li][method]
            if len(data) < 3:
                print(f"    {method}: insufficient data ({len(data)})")
                continue
            
            mean_d = np.mean(data)
            std_d = np.std(data)
            n_boot = 2000
            boots = [np.mean(np.random.choice(data, len(data), replace=True)) for _ in range(n_boot)]
            ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
            sig = "★" if (ci_lo > 0 or ci_hi < 0) else ""
            neg_pct = np.mean([d < 0 for d in data])
            
            print(f"    {method:18s}: Δ={mean_d:+.4f} std={std_d:.4f} "
                  f"CI=[{ci_lo:+.4f},{ci_hi:+.4f}] neg={neg_pct:.0%} n={len(data)} {sig}")
        
        # Key comparison
        attn = results[li]['attn_swap']
        res = results[li]['residual_patch']
        if len(attn) >= 3 and len(res) >= 3:
            attn_eff = abs(np.mean(attn))
            res_eff = abs(np.mean(res))
            ratio = attn_eff / (res_eff + 1e-10)
            print(f"    *** |attn_swap|/|residual_patch| = {ratio:.2f} ***")
            if ratio > 3.0:
                print(f"    ★★★ CONFIRMED: Attention routing >> residual patching!")
                print(f"    ★★★ Syntax is primarily carried by attention routing, NOT residual directions!")
            elif ratio > 1.5:
                print(f"    ★ Attention routing > residual patching — partial support for routing hypothesis")
            elif ratio < 0.5:
                print(f"    → Residual patching > attention routing — syntax is in residual, not routing")
            else:
                print(f"    → Roughly equal — both mechanisms may contribute")
    
    return results


# ============================================================
# EXPERIMENT 4: Regular vs Irregular Noun Comparison
# ============================================================
def regular_vs_irregular(model, tokenizer, device, info, target_layers):
    """
    Address Hard Flaw #2: noun type confound
    Compare syntax direction consistency and patching effects between:
    - Regular nouns (cat/cats, dog/dogs)
    - Irregular nouns (man/men, child/children)
    
    If the difference is large, it means previous results were confounded.
    """
    layers_list = get_layers(model)
    
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Regular vs Irregular Noun Comparison")
    print("=" * 70)
    
    # Collect activations for both types
    for noun_type, nva_list in [("regular", REGULAR_NVA), ("irregular", IRREGULAR_NVA)]:
        print(f"\n  --- {noun_type.upper()} NOUNS ({len(nva_list)} pairs) ---")
        sing_subj = defaultdict(list)
        plur_subj = defaultdict(list)
        
        for sn, pn, sv, pv in nva_list:
            sent = f"The {sn} {sv}"  # Use simplest template
            toks = tokenizer.encode(sent, add_special_tokens=False)
            sp = find_noun_pos(tokenizer, toks)
            if sp is None:
                continue
            
            captured = {}
            def mk_hook(li):
                def fn(m, inp, out):
                    captured[li] = (out[0] if isinstance(out, tuple) else out).detach().cpu()
                return fn
            
            hooks = []
            for li in target_layers:
                if li < len(layers_list):
                    hooks.append(layers_list[li].register_forward_hook(mk_hook(li)))
            
            ids = tokenizer(sent, return_tensors="pt").to(device)
            with torch.no_grad():
                try: model(**ids)
                except:
                    for h in hooks: h.remove()
                    continue
            for h in hooks: h.remove()
            
            for li in target_layers:
                if li in captured and sp < captured[li].shape[1]:
                    sing_subj[li].append(captured[li][0, sp, :].float().numpy())
            
            del captured
            
            # Plural
            sent_p = f"The {pn} {pv}"
            toks_p = tokenizer.encode(sent_p, add_special_tokens=False)
            sp_p = find_noun_pos(tokenizer, toks_p)
            if sp_p is None:
                continue
            
            captured_p = {}
            def mk_hook_p(li):
                def fn(m, inp, out):
                    captured_p[li] = (out[0] if isinstance(out, tuple) else out).detach().cpu()
                return fn
            
            hooks_p = []
            for li in target_layers:
                if li < len(layers_list):
                    hooks_p.append(layers_list[li].register_forward_hook(mk_hook_p(li)))
            
            ids_p = tokenizer(sent_p, return_tensors="pt").to(device)
            with torch.no_grad():
                try: model(**ids_p)
                except:
                    for h in hooks_p: h.remove()
                    continue
            for h in hooks_p: h.remove()
            
            for li in target_layers:
                if li in captured_p and sp_p < captured_p[li].shape[1]:
                    plur_subj[li].append(captured_p[li][0, sp_p, :].float().numpy())
            
            del captured_p; gc.collect(); torch.cuda.empty_cache()
        
        # Compute directions and separability
        from sklearn.decomposition import PCA
        from sklearn.metrics import silhouette_score
        
        for li in target_layers:
            s = np.array(sing_subj[li])
            p = np.array(plur_subj[li])
            if len(s) < 3 or len(p) < 3:
                print(f"    L{li}: insufficient data")
                continue
            
            # Direction
            d_num = p.mean(axis=0) - s.mean(axis=0)
            d_norm = np.linalg.norm(d_num)
            d_num_unit = d_num / (d_norm + 1e-10)
            
            # Separability: projection onto direction
            proj_s = s @ d_num_unit
            proj_p = p @ d_num_unit
            # Effect size (Cohen's d)
            d_eff = (proj_p.mean() - proj_s.mean()) / (np.std(np.concatenate([proj_s, proj_p])) + 1e-10)
            
            # Silhouette
            X = np.vstack([s, p])
            y = np.array([0]*len(s) + [1]*len(p))
            # Project onto top-5 PCs
            pca = PCA(n_components=min(5, X.shape[0]-1, X.shape[1]))
            X_pca = pca.fit_transform(X)
            sil = silhouette_score(X_pca, y) if len(set(y)) > 1 else 0
            
            # Number variance fraction
            total_var = np.var(X, axis=0).sum()
            num_var = np.dot(d_num, d_num) * len(s) * len(p) / (len(s) + len(p))
            frac = num_var / (total_var + 1e-10)
            
            print(f"    L{li}: n={len(s)}+{len(p)}, ||d||={d_norm:.2f}, "
                  f"Cohen_d={d_eff:.3f}, silhouette={sil:.3f}, "
                  f"num_var_frac={frac:.6f}")
    
    # Cross-compare directions between regular and irregular
    print("\n  --- Cross-type Direction Comparison ---")
    
    # Re-extract directions
    reg_dirs = {}
    irreg_dirs = {}
    
    for noun_type, nva_list, dirs in [("regular", REGULAR_NVA, reg_dirs), 
                                       ("irregular", IRREGULAR_NVA, irreg_dirs)]:
        sing_subj = defaultdict(list)
        plur_subj = defaultdict(list)
        
        for sn, pn, sv, pv in nva_list:
            for is_sing, (noun, verb) in [(True, (sn, sv)), (False, (pn, pv))]:
                sent = f"The {noun} {verb}"
                toks = tokenizer.encode(sent, add_special_tokens=False)
                sp = find_noun_pos(tokenizer, toks)
                if sp is None:
                    continue
                
                captured = {}
                def mk_hook(li):
                    def fn(m, inp, out):
                        captured[li] = (out[0] if isinstance(out, tuple) else out).detach().cpu()
                    return fn
                
                hooks = []
                for li in target_layers:
                    if li < len(layers_list):
                        hooks.append(layers_list[li].register_forward_hook(mk_hook(li)))
                
                ids = tokenizer(sent, return_tensors="pt").to(device)
                with torch.no_grad():
                    try: model(**ids)
                    except:
                        for h in hooks: h.remove()
                        continue
                for h in hooks: h.remove()
                
                for li in target_layers:
                    if li in captured and sp < captured[li].shape[1]:
                        target = sing_subj if is_sing else plur_subj
                        target[li].append(captured[li][0, sp, :].float().numpy())
                
                del captured; gc.collect(); torch.cuda.empty_cache()
        
        for li in target_layers:
            s = np.array(sing_subj[li])
            p = np.array(plur_subj[li])
            if len(s) >= 3 and len(p) >= 3:
                d_num = p.mean(axis=0) - s.mean(axis=0)
                d_num = d_num / (np.linalg.norm(d_num) + 1e-10)
                dirs[li] = d_num
    
    for li in target_layers:
        if li in reg_dirs and li in irreg_dirs:
            cos_reg_irreg = float(np.dot(reg_dirs[li], irreg_dirs[li]))
            print(f"    L{li}: cos(regular, irregular) = {cos_reg_irreg:.4f}")
            if cos_reg_irreg > 0.8:
                print(f"      → Directions highly aligned → syntax structure is robust across noun types")
            elif cos_reg_irreg > 0.5:
                print(f"      → Directions moderately aligned → some confound from noun type")
            else:
                print(f"      ⚠ Directions poorly aligned → noun type is a major confound!")


# ============================================================
# EXPERIMENT 5: Power Analysis & SNR Estimation
# ============================================================
def power_analysis(results_dict, target_layers):
    """
    Address Hard Flaw #1: effect size without power analysis
    Compute SNR, minimum detectable effect, and statistical power.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Power Analysis & SNR Estimation")
    print("=" * 70)
    
    for li in target_layers:
        print(f"\n  Layer {li}:")
        
        # Use experiment 1 results if available
        if li in results_dict:
            for method, data in results_dict[li].items():
                if len(data) < 3:
                    continue
                
                data = np.array(data)
                mean_d = np.mean(data)
                std_d = np.std(data)
                n = len(data)
                
                # SNR
                snr = abs(mean_d) / (std_d + 1e-10)
                
                # Standard error
                se = std_d / np.sqrt(n)
                
                # Minimum detectable effect at 80% power, α=0.05
                # For two-tailed t-test: MDE = (z_α/2 + z_β) * σ / √n
                z_alpha = 1.96  # two-tailed 0.05
                z_beta = 0.84  # 80% power
                mde = (z_alpha + z_beta) * std_d / np.sqrt(n)
                
                # Actual power for observed effect
                if std_d > 1e-10:
                    z_obs = abs(mean_d) / se
                    from scipy import stats
                    power = 1 - stats.norm.cdf(z_alpha - z_obs)
                else:
                    power = 0
                
                print(f"    {method:15s}: mean={mean_d:+.4f}, σ={std_d:.4f}, "
                      f"SNR={snr:.3f}, MDE(80%)={mde:.4f}, power={power:.1%}, n={n}")
                
                if abs(mean_d) < mde:
                    print(f"      ⚠ Effect size ({abs(mean_d):.4f}) < MDE ({mde:.4f}) — UNDERPOWERED!")
                    print(f"      Need n ≥ {int((z_alpha + z_beta)**2 * std_d**2 / (abs(mean_d) + 1e-10)**2)} for 80% power")
                else:
                    print(f"      ✔ Adequately powered")


# ============================================================
# MAIN
# ============================================================
def run_phase63(model, tokenizer, device, info):
    d_model = info.d_model
    layers_list = get_layers(model)
    
    print("=" * 70)
    print("★★★ Phase 63: NCSS Verification — Control Failure Theorem ★★★")
    print("Testing: LayerNorm-induced Control Failure + Nonlinear Intervention")
    print("=" * 70)
    print(f"\nData: {len(ALL_NVA)} NVA pairs ({len(REGULAR_NVA)} regular + {len(IRREGULAR_NVA)} irregular)")
    print(f"Layers: {TARGET_LAYERS}")
    
    # Experiment 1: LN-Aware Patching
    exp1_results = ln_aware_patching(
        model, tokenizer, device, info, TARGET_LAYERS,
        nva_pairs=ALL_NVA, alpha=2.0, n_test=60
    )
    
    # Experiment 2: Subspace Projection Control
    exp2_results = subspace_projection_control(
        model, tokenizer, device, info, TARGET_LAYERS, n_test=60
    )
    
    # Experiment 3: Attention Routing Intervention
    exp3_results = attention_routing_intervention(
        model, tokenizer, device, info, TARGET_LAYERS, n_test=40
    )
    
    # Experiment 4: Regular vs Irregular
    regular_vs_irregular(model, tokenizer, device, info, TARGET_LAYERS)
    
    # Experiment 5: Power Analysis
    power_analysis(exp1_results, TARGET_LAYERS)
    
    # ===== FINAL SUMMARY =====
    print("\n" + "=" * 70)
    print("FINAL SUMMARY: Phase 63 — NCSS Verification")
    print("=" * 70)
    
    print("""
KEY QUESTIONS ANSWERED:

1. Does LN attenuate linear interventions? (Control Failure Theorem)
   → Compare |Δ_post-LN| vs |Δ_pre-LN|
   → If post-LN >> pre-LN: theorem confirmed, direction ≠ control variable
   
2. Is syntax a functional component accessible via projection?
   → Compare swap/amplify/remove projections
   → If swap works: syntax is a coherent subspace component
   
3. Is syntax carried by attention routing?
   → Compare attention swap vs residual patching
   → If attention >> residual: syntax = routing, not representation
   
4. Do regular vs irregular nouns have different syntax directions?
   → cos(regular_dir, irregular_dir)
   → If low: previous results were confounded by noun type
   
5. Are previous experiments adequately powered?
   → MDE vs observed effect
   → If underpowered: negative results are inconclusive
""")
    
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deepseek7b")
    args = parser.parse_args()
    
    model, tokenizer, device = load_model(args.model)
    info = get_model_info(model, args.model)
    try:
        run_phase63(model, tokenizer, device, info)
    finally:
        release_model(model)
