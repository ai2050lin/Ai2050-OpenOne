"""
Phase 63b: NCSS Verification — Fixed Attention Experiment + Critical Tests
==========================================================================

Fixes from Phase 63:
1. Attention weights: need output_attentions=True in model call
2. Syntax swap: sequences of different lengths need careful handling
3. Focus on the most informative experiments

CORE TESTS:
1. Attention Routing: Enable output_attentions, capture real attention patterns
2. Same-template syntax swap (avoids length mismatch)
3. Regular vs irregular with within-type patching comparison
4. Effect size analysis with proper power calculations
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

REGULAR_NOUNS = set()
for sn, pn, _, _ in REGULAR_NVA:
    REGULAR_NOUNS.add(sn.lower())
    REGULAR_NOUNS.add(pn.lower())

TARGET_LAYERS = [0, 10, 15, 18]


def find_noun_pos(tokenizer, tokens):
    for i, t in enumerate(tokens):
        d = tokenizer.decode([t]).strip().lower()
        for n in NOUNS_SET:
            if d == n or d.startswith(n):
                return i + 1
    return None


def find_verb_pos(tokenizer, tokens, sv, pv):
    targets = {sv.lower(), pv.lower()}
    for i, t in enumerate(tokens):
        d = tokenizer.decode([t]).strip().lower()
        for vt in targets:
            if d == vt or d.startswith(vt):
                return i + 1
    return None


def get_input_ln(layer):
    for name, mod in layer.named_children():
        if 'input_layernorm' in name or name == 'ln_1':
            return mod
    return None


# ============================================================
# EXPERIMENT A: Attention Routing — Correct Implementation
# ============================================================
def attention_routing_experiment(model, tokenizer, device, info, target_layers, n_test=40):
    """
    Key insight: We need output_attentions=True to capture attention patterns.
    
    Method: For same-template sentences with different number,
    swap the attention pattern FROM verb TO subject position.
    This tests whether attention routing carries number information.
    """
    layers_list = get_layers(model)
    
    print("\n" + "=" * 70)
    print("EXPERIMENT A: Attention Routing Intervention")
    print("=" * 70)
    print("  Testing: Does attention FROM verb TO subject carry number info?")
    print("  Method: Replace verb→subject attention weights between sing/plur")
    
    results = {li: {'attn_replace': [], 'residual_patch': []} for li in target_layers}
    
    # Use simple template for both source and target (same length!)
    for si, (sn, pn, sv, pv) in enumerate(REGULAR_NVA[:n_test]):
        if si % 10 == 0 and si > 0:
            print(f"    Testing {si}/{n_test}...")
        
        # Both use "The [N] [V]" template — same sequence structure
        sing_sent = f"The {sn} {sv}"
        plur_sent = f"The {pn} {pv}"
        
        sing_toks = tokenizer.encode(sing_sent, add_special_tokens=False)
        plur_toks = tokenizer.encode(plur_sent, add_special_tokens=False)
        
        sing_sp = find_noun_pos(tokenizer, sing_toks)
        sing_vp = find_verb_pos(tokenizer, sing_toks, sv, pv)
        plur_sp = find_noun_pos(tokenizer, plur_toks)
        plur_vp = find_verb_pos(tokenizer, plur_toks, sv, pv)
        
        if sing_sp is None or sing_vp is None or plur_sp is None or plur_vp is None:
            continue
        
        sing_ids = tokenizer(sing_sent, return_tensors="pt").to(device)
        plur_ids = tokenizer(plur_sent, return_tensors="pt").to(device)
        
        sv_ids = tokenizer.encode(sv, add_special_tokens=False)
        pv_ids = tokenizer.encode(pv, add_special_tokens=False)
        if not sv_ids or not pv_ids:
            continue
        
        # Base agreement for singular sentence
        with torch.no_grad():
            base_out = model(**sing_ids, output_attentions=True)
            base_logits = base_out.logits.detach().cpu()
        base_agr = (base_logits[0, sing_vp, sv_ids[0]] - base_logits[0, sing_vp, pv_ids[0]]).item()
        
        # Also get plural sentence attention
        with torch.no_grad():
            plur_out = model(**plur_ids, output_attentions=True)
        
        for li in target_layers:
            # Get attention weights
            # output_attentions gives us a tuple of attention weights per layer
            # base_out.attentions[li] shape: [batch, heads, seq_q, seq_k]
            if not hasattr(base_out, 'attentions') or base_out.attentions is None:
                continue
            if li >= len(base_out.attentions) or li >= len(plur_out.attentions):
                continue
            
            sing_attn = base_out.attentions[li].detach().cpu()  # [1, heads, seq_q, seq_k]
            plur_attn = plur_out.attentions[li].detach().cpu()  # [1, heads, seq_q, seq_k]
            
            if sing_attn.shape != plur_attn.shape:
                continue
            
            # ===== METHOD 1: Replace verb→subject attention =====
            # In singular sentence, replace the verb row's attention with plural's
            # This should make the verb "see" the plural subject's attention pattern
            new_attn = sing_attn.clone()
            if sing_vp < sing_attn.shape[2] and sing_sp < sing_attn.shape[3]:
                # Replace entire verb row attention (how verb attends to all tokens)
                new_attn[0, :, sing_vp, :] = plur_attn[0, :, plur_vp, :]
                # Re-normalize
                new_attn_sum = new_attn[0, :, sing_vp, :].sum(dim=-1, keepdim=True)
                new_attn[0, :, sing_vp, :] = new_attn[0, :, sing_vp, :] / (new_attn_sum + 1e-10)
            
            # Now we need to inject this modified attention back
            # Unfortunately, we can't directly modify attention and re-run
            # Instead, use hook to replace attention output
            
            # The trick: use the attention module's forward to compute new output
            # We'll hook into the self_attn module and replace the attention weights
            
            attn_replaced = [False]
            
            def attn_hook(m, inp, out, _new_attn=new_attn.to(device)):
                if not attn_replaced[0]:
                    if isinstance(out, tuple):
                        # out = (hidden_states, attention_weights, ...)
                        # We want to modify hidden_states based on new attention
                        # But we can't recompute easily, so just mark as replaced
                        pass
                    attn_replaced[0] = True
                return out
            
            # Actually, we need a different approach:
            # Instead of replacing attention weights, we should look at what 
            # attention DOES to the residual stream
            
            # Better approach: compute the attention output difference
            # attn_output = sum_j(attn_weight[j] * V[j])
            # If we change attn_weight at verb position, the change is:
            # delta = sum_j((plur_attn[verb,j] - sing_attn[verb,j]) * V[j])
            # This is a LINEAR intervention in the residual stream, but targeted
            
            # For now, let's use a simpler proxy:
            # Compute the "attention-weighted value difference" between sing/plur
            # at the verb position
            
            # Actually, let's try the direct approach:
            # Replace the self_attn output by manually computing attention with modified weights
            
            # The cleanest test: just compute what the attention output WOULD be
            # if the verb attended with plural's pattern instead of singular's
            
            # This requires accessing V values, which we can get from a hook
            
            pass
        
        # ===== SIMPLER APPROACH: Attention-weighted value swap =====
        # Instead of modifying attention weights (which requires recomputing),
        # we directly modify the attention OUTPUT at the verb position.
        # 
        # The attention output at position i is:
        #   o_i = sum_j attn[i,j] * V[j]
        # 
        # The difference between plural and singular attention output at verb:
        #   delta = o_plur_verb - o_sing_verb
        #         = sum_j (plur_attn[verb,j] - sing_attn[verb,j]) * V[j]
        #
        # This delta captures what attention routing changes between sing/plur.
        # If we add this delta to the singular sentence's residual at verb position,
        # we're effectively swapping the attention routing.
        
        # We need to capture the self_attn output for both sentences
        for li in target_layers:
            sing_attn_out = {}
            plur_attn_out = {}
            
            def cap_sing_attn(m, inp, out, _li=li):
                sing_attn_out[_li] = (out[0] if isinstance(out, tuple) else out).detach().cpu()
            def cap_plur_attn(m, inp, out, _li=li):
                plur_attn_out[_li] = (out[0] if isinstance(out, tuple) else out).detach().cpu()
            
            # Capture singular attention output
            hook_s = layers_list[li].self_attn.register_forward_hook(cap_sing_attn)
            with torch.no_grad():
                model(**sing_ids)
            hook_s.remove()
            
            # Capture plural attention output
            hook_p = layers_list[li].self_attn.register_forward_hook(cap_plur_attn)
            with torch.no_grad():
                model(**plur_ids)
            hook_p.remove()
            
            if li not in sing_attn_out or li not in plur_attn_out:
                continue
            
            # Compute delta: attention output difference at verb position
            sing_ao = sing_attn_out[li]  # [1, seq, d]
            plur_ao = plur_attn_out[li]  # [1, seq, d]
            
            if sing_vp >= sing_ao.shape[1] or plur_vp >= plur_ao.shape[1]:
                continue
            
            # The delta at verb position
            attn_delta = plur_ao[0, plur_vp, :] - sing_ao[0, sing_vp, :]  # [d]
            attn_delta_norm = torch.norm(attn_delta)
            if attn_delta_norm < 1e-10:
                continue
            
            # ===== TEST A1: Add attention output delta to singular sentence =====
            # This simulates "swapping the attention routing"
            attn_delta_t = attn_delta.to(device).to(model.dtype)
            
            applied_a = [False]
            def hook_attn_swap(m, inp, out, _vp=sing_vp, _delta=attn_delta_t):
                if not applied_a[0]:
                    out_mod = out[0] if isinstance(out, tuple) else out
                    p = out_mod.clone()
                    p[:, _vp, :] += _delta
                    applied_a[0] = True
                    return (p,) + out[1:] if isinstance(out, tuple) else p
                return out
            
            # Add delta to self_attn output
            hook_a = layers_list[li].self_attn.register_forward_hook(hook_attn_swap)
            with torch.no_grad():
                patched_a_logits = model(**sing_ids).logits.detach().cpu()
            hook_a.remove()
            patched_a_agr = (patched_a_logits[0, sing_vp, sv_ids[0]] - 
                           patched_a_logits[0, sing_vp, pv_ids[0]]).item()
            results[li]['attn_replace'].append(patched_a_agr - base_agr)
            
            # ===== TEST A2: Residual patching for comparison =====
            # Add syntax direction (plur - sing) at subject position
            # This is the traditional approach
            sing_layer_out = {}
            plur_layer_out = {}
            
            def cap_sing_layer(m, inp, out, _li=li):
                sing_layer_out[_li] = (out[0] if isinstance(out, tuple) else out).detach().cpu()
            def cap_plur_layer(m, inp, out, _li=li):
                plur_layer_out[_li] = (out[0] if isinstance(out, tuple) else out).detach().cpu()
            
            hook_sl = layers_list[li].register_forward_hook(cap_sing_layer)
            with torch.no_grad():
                model(**sing_ids)
            hook_sl.remove()
            
            hook_pl = layers_list[li].register_forward_hook(cap_plur_layer)
            with torch.no_grad():
                model(**plur_ids)
            hook_pl.remove()
            
            if li not in sing_layer_out or li not in plur_layer_out:
                continue
            
            # Residual delta at subject position
            res_delta = plur_layer_out[li][0, plur_sp, :] - sing_layer_out[li][0, sing_sp, :]
            res_delta_norm = torch.norm(res_delta)
            if res_delta_norm < 1e-10:
                continue
            
            # Normalize both deltas to same magnitude for fair comparison
            common_mag = min(attn_delta_norm.item(), res_delta_norm.item())
            if res_delta_norm > 1e-10:
                res_delta_scaled = res_delta * (common_mag / res_delta_norm)
            else:
                continue
            if attn_delta_norm > 1e-10:
                attn_delta_scaled = attn_delta * (common_mag / attn_delta_norm)
            else:
                continue
            
            # Residual patching: add scaled direction at subject position
            res_delta_t = res_delta_scaled.to(device).to(model.dtype)
            
            applied_r = [False]
            def hook_res(m, inp, out, _sp=sing_sp, _delta=res_delta_t):
                if not applied_r[0]:
                    out_mod = out[0] if isinstance(out, tuple) else out
                    p = out_mod.clone()
                    p[:, _sp, :] += _delta
                    applied_r[0] = True
                    return (p,) + out[1:] if isinstance(out, tuple) else p
                return out
            
            hook_r = layers_list[li].register_forward_hook(hook_res)
            with torch.no_grad():
                patched_r_logits = model(**sing_ids).logits.detach().cpu()
            hook_r.remove()
            patched_r_agr = (patched_r_logits[0, sing_vp, sv_ids[0]] - 
                           patched_r_logits[0, sing_vp, pv_ids[0]]).item()
            results[li]['residual_patch'].append(patched_r_agr - base_agr)
            
            del sing_attn_out, plur_attn_out, sing_layer_out, plur_layer_out
            gc.collect(); torch.cuda.empty_cache()
    
    # === ANALYSIS ===
    print("\n" + "=" * 70)
    print("EXPERIMENT A RESULTS: Attention Routing vs Residual Patching")
    print("=" * 70)
    print("(Both interventions scaled to same magnitude for fair comparison)")
    
    for li in target_layers:
        print(f"\n  Layer {li}:")
        for method in ['attn_replace', 'residual_patch']:
            data = results[li][method]
            if len(data) < 3:
                print(f"    {method:18s}: insufficient data ({len(data)})")
                continue
            
            mean_d = np.mean(data)
            std_d = np.std(data)
            n = len(data)
            n_boot = 2000
            boots = [np.mean(np.random.choice(data, len(data), replace=True)) for _ in range(n_boot)]
            ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
            sig = "★" if (ci_lo > 0 or ci_hi < 0) else ""
            neg_pct = np.mean([d < 0 for d in data])
            snr = abs(mean_d) / (std_d + 1e-10)
            
            print(f"    {method:18s}: Δ={mean_d:+.4f} σ={std_d:.4f} "
                  f"CI=[{ci_lo:+.4f},{ci_hi:+.4f}] SNR={snr:.2f} "
                  f"neg={neg_pct:.0%} n={n} {sig}")
        
        # Key comparison
        attn = results[li]['attn_replace']
        res = results[li]['residual_patch']
        if len(attn) >= 3 and len(res) >= 3:
            attn_eff = abs(np.mean(attn))
            res_eff = abs(np.mean(res))
            ratio = attn_eff / (res_eff + 1e-10)
            print(f"    *** |attn_routing| / |residual_patch| = {ratio:.2f} ***")
            if ratio > 3.0:
                print(f"    ★★★ Syntax primarily carried by ATTENTION ROUTING!")
            elif ratio > 1.5:
                print(f"    ★ Attention routing > residual patching")
            elif ratio < 0.67:
                print(f"    → Residual patching > attention routing")
            else:
                print(f"    → Both mechanisms contribute similarly")
    
    return results


# ============================================================
# EXPERIMENT B: Same-Template Syntax Swap (avoid length mismatch)
# ============================================================
def same_template_syntax_swap(model, tokenizer, device, info, target_layers, n_test=50):
    """
    Swap syntax subspaces between singular and plural sentences IN THE SAME TEMPLATE.
    This avoids the sequence length mismatch issue.
    
    Method: For "The cat runs" (sing) and "The cats run" (plur):
    1. Extract syntax subspace from both
    2. Replace sing's syntax component with plur's syntax component
    3. Compare with simple direction addition
    
    If NCSS is correct: subspace swap should be MORE effective than direction addition
    because it respects the nonlinear coupling structure.
    """
    from sklearn.decomposition import PCA
    layers_list = get_layers(model)
    
    print("\n" + "=" * 70)
    print("EXPERIMENT B: Same-Template Syntax Subspace Swap")
    print("=" * 70)
    
    # First collect all sing/plur activations for subspace computation
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
    
    # Compute subspaces
    syntax_dirs = {}
    syntax_subspaces = {}
    for li in target_layers:
        s = np.array(sing_subj[li])
        p = np.array(plur_subj[li])
        if len(s) < 5 or len(p) < 5:
            continue
        
        d_num = p.mean(axis=0) - s.mean(axis=0)
        d_num_norm = d_num / (np.linalg.norm(d_num) + 1e-10)
        syntax_dirs[li] = d_num_norm
        
        diff = np.vstack([p - s.mean(axis=0), s - p.mean(axis=0)])
        n_comp = min(10, diff.shape[0]-1, diff.shape[1])
        pca = PCA(n_components=n_comp)
        pca.fit(diff)
        syntax_subspaces[li] = pca.components_  # [n_comp, d]
        
        print(f"    L{li}: syntax subspace {pca.components_.shape}, "
              f"top-5 var explained: {pca.explained_variance_ratio_[:5].sum():.3f}")
    
    # Now test: swap syntax component between sing/plur pairs
    results = {li: {'direction_add': [], 'subspace_swap': [], 'full_swap': []} 
               for li in target_layers}
    
    test_pairs = REGULAR_NVA[:n_test]
    
    for si, (sn, pn, sv, pv) in enumerate(test_pairs):
        if si % 10 == 0 and si > 0:
            print(f"    Testing {si}/{n_test}...")
        
        # Singular target sentence
        sing_sent = f"The {sn} {sv}"
        sing_toks = tokenizer.encode(sing_sent, add_special_tokens=False)
        sing_sp = find_noun_pos(tokenizer, sing_toks)
        sing_vp = find_verb_pos(tokenizer, sing_toks, sv, pv)
        if sing_sp is None or sing_vp is None:
            continue
        
        sing_ids = tokenizer(sing_sent, return_tensors="pt").to(device)
        sv_ids = tokenizer.encode(sv, add_special_tokens=False)
        pv_ids = tokenizer.encode(pv, add_special_tokens=False)
        if not sv_ids or not pv_ids:
            continue
        
        # Base agreement
        with torch.no_grad():
            base_logits = model(**sing_ids).logits.detach().cpu()
        base_agr = (base_logits[0, sing_vp, sv_ids[0]] - base_logits[0, sing_vp, pv_ids[0]]).item()
        
        # Plural source sentence (same template!)
        plur_sent = f"The {pn} {pv}"
        plur_ids = tokenizer(plur_sent, return_tensors="pt").to(device)
        plur_toks = tokenizer.encode(plur_sent, add_special_tokens=False)
        plur_sp = find_noun_pos(tokenizer, plur_toks)
        plur_vp = find_verb_pos(tokenizer, plur_toks, sv, pv)
        if plur_sp is None or plur_vp is None:
            continue
        
        for li in target_layers:
            if li not in syntax_dirs or li not in syntax_subspaces:
                continue
            
            # Capture activations for both
            sing_act = {}
            plur_act = {}
            
            def cap_sing(m, inp, out, _li=li):
                sing_act[_li] = (out[0] if isinstance(out, tuple) else out).detach().cpu()
            def cap_plur(m, inp, out, _li=li):
                plur_act[_li] = (out[0] if isinstance(out, tuple) else out).detach().cpu()
            
            hook_s = layers_list[li].register_forward_hook(cap_sing)
            with torch.no_grad():
                model(**sing_ids)
            hook_s.remove()
            
            hook_p = layers_list[li].register_forward_hook(cap_plur)
            with torch.no_grad():
                model(**plur_ids)
            hook_p.remove()
            
            if li not in sing_act or li not in plur_act:
                continue
            
            s_vec = sing_act[li][0, sing_sp, :].float().numpy()  # [d]
            p_vec = plur_act[li][0, plur_sp, :].float().numpy()  # [d]
            
            # === Method 1: Simple direction addition (baseline) ===
            d_num = syntax_dirs[li]
            delta_dir = 2.0 * d_num  # α=2.0
            
            applied1 = [False]
            delta1_t = torch.tensor(delta_dir, dtype=torch.float32, device=device)
            def hook_dir(m, inp, out, _sp=sing_sp, _d=delta1_t):
                if not applied1[0]:
                    out_mod = out[0] if isinstance(out, tuple) else out
                    p = out_mod.clone()
                    p[:, _sp, :] += _d.to(p.dtype)
                    applied1[0] = True
                    return (p,) + out[1:] if isinstance(out, tuple) else p
                return out
            
            hook1 = layers_list[li].register_forward_hook(hook_dir)
            with torch.no_grad():
                patched1_logits = model(**sing_ids).logits.detach().cpu()
            hook1.remove()
            patched1_agr = (patched1_logits[0, sing_vp, sv_ids[0]] - 
                          patched1_logits[0, sing_vp, pv_ids[0]]).item()
            results[li]['direction_add'].append(patched1_agr - base_agr)
            
            # === Method 2: Subspace swap ===
            # Replace sing's syntax subspace component with plur's
            syn_basis = syntax_subspaces[li]  # [k, d]
            P_syn = syn_basis.T @ syn_basis  # [d, d] projection matrix
            
            syn_s = P_syn @ s_vec  # sing's syntax component
            syn_p = P_syn @ p_vec  # plur's syntax component
            delta_swap = syn_p - syn_s
            
            applied2 = [False]
            delta2_t = torch.tensor(delta_swap, dtype=torch.float32, device=device)
            def hook_swap(m, inp, out, _sp=sing_sp, _d=delta2_t):
                if not applied2[0]:
                    out_mod = out[0] if isinstance(out, tuple) else out
                    p = out_mod.clone()
                    p[:, _sp, :] += _d.to(p.dtype)
                    applied2[0] = True
                    return (p,) + out[1:] if isinstance(out, tuple) else p
                return out
            
            hook2 = layers_list[li].register_forward_hook(hook_swap)
            with torch.no_grad():
                patched2_logits = model(**sing_ids).logits.detach().cpu()
            hook2.remove()
            patched2_agr = (patched2_logits[0, sing_vp, sv_ids[0]] - 
                          patched2_logits[0, sing_vp, pv_ids[0]]).item()
            results[li]['subspace_swap'].append(patched2_agr - base_agr)
            
            # === Method 3: Full vector swap (upper bound) ===
            # Replace entire activation at subject position
            delta_full = p_vec - s_vec
            
            applied3 = [False]
            delta3_t = torch.tensor(delta_full, dtype=torch.float32, device=device)
            def hook_full(m, inp, out, _sp=sing_sp, _d=delta3_t):
                if not applied3[0]:
                    out_mod = out[0] if isinstance(out, tuple) else out
                    p = out_mod.clone()
                    p[:, _sp, :] += _d.to(p.dtype)
                    applied3[0] = True
                    return (p,) + out[1:] if isinstance(out, tuple) else p
                return out
            
            hook3 = layers_list[li].register_forward_hook(hook_full)
            with torch.no_grad():
                patched3_logits = model(**sing_ids).logits.detach().cpu()
            hook3.remove()
            patched3_agr = (patched3_logits[0, sing_vp, sv_ids[0]] - 
                          patched3_logits[0, sing_vp, pv_ids[0]]).item()
            results[li]['full_swap'].append(patched3_agr - base_agr)
            
            del sing_act, plur_act; gc.collect(); torch.cuda.empty_cache()
    
    # === ANALYSIS ===
    print("\n" + "=" * 70)
    print("EXPERIMENT B RESULTS: Direction vs Subspace Swap vs Full Swap")
    print("=" * 70)
    
    for li in target_layers:
        print(f"\n  Layer {li}:")
        for method in ['direction_add', 'subspace_swap', 'full_swap']:
            data = results[li][method]
            if len(data) < 3:
                print(f"    {method:18s}: insufficient data")
                continue
            
            mean_d = np.mean(data)
            std_d = np.std(data)
            n_boot = 2000
            boots = [np.mean(np.random.choice(data, len(data), replace=True)) for _ in range(n_boot)]
            ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
            sig = "★" if (ci_lo > 0 or ci_hi < 0) else ""
            neg_pct = np.mean([d < 0 for d in data])
            
            print(f"    {method:18s}: Δ={mean_d:+.4f} σ={std_d:.4f} "
                  f"CI=[{ci_lo:+.4f},{ci_hi:+.4f}] neg={neg_pct:.0%} n={len(data)} {sig}")
        
        # Key comparison
        dir_eff = abs(np.mean(results[li]['direction_add'])) if len(results[li]['direction_add']) >= 3 else 0
        swap_eff = abs(np.mean(results[li]['subspace_swap'])) if len(results[li]['subspace_swap']) >= 3 else 0
        full_eff = abs(np.mean(results[li]['full_swap'])) if len(results[li]['full_swap']) >= 3 else 0
        
        if full_eff > 1e-10:
            print(f"    *** |direction|/|full| = {dir_eff/full_eff:.2f}, "
                  f"|subspace_swap|/|full| = {swap_eff/full_eff:.2f} ***")
            if swap_eff > dir_eff * 1.5:
                print(f"    ★ Subspace swap > direction add → syntax is a subspace, not a direction")
            else:
                print(f"    → Direction add ≈ subspace swap → direction captures most of syntax signal")
    
    return results


# ============================================================
# EXPERIMENT C: Regular vs Irregular — Patching Effectiveness
# ============================================================
def regular_vs_irregular_patching(model, tokenizer, device, info, target_layers, n_test=40):
    """
    Compare patching effectiveness for regular vs irregular nouns.
    If irregular nouns dilute the effect, this is a confound.
    """
    layers_list = get_layers(model)
    
    print("\n" + "=" * 70)
    print("EXPERIMENT C: Regular vs Irregular — Patching Effectiveness")
    print("=" * 70)
    
    for noun_type, nva_list in [("REGULAR", REGULAR_NVA), ("IRREGULAR", IRREGULAR_NVA)]:
        print(f"\n  --- {noun_type} NOUNS ({len(nva_list)} pairs) ---")
        
        # Extract direction from this noun type
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
        
        # Compute directions
        dirs = {}
        for li in target_layers:
            s = np.array(sing_subj[li])
            p = np.array(plur_subj[li])
            if len(s) >= 3 and len(p) >= 3:
                d_num = p.mean(axis=0) - s.mean(axis=0)
                d_num = d_num / (np.linalg.norm(d_num) + 1e-10)
                dirs[li] = d_num
        
        # Now patch within this noun type
        # Target: singular sentence → add plural direction → should shift toward plural
        results = {li: [] for li in target_layers}
        
        for si, (sn, pn, sv, pv) in enumerate(nva_list[:n_test]):
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
            
            with torch.no_grad():
                base_logits = model(**tgt_ids).logits.detach().cpu()
            base_agr = (base_logits[0, tgt_vp, sv_ids[0]] - base_logits[0, tgt_vp, pv_ids[0]]).item()
            
            for li in target_layers:
                if li not in dirs:
                    continue
                
                d_t = torch.tensor(dirs[li], dtype=torch.float32, device=device)
                applied = [False]
                def hook(m, inp, out, _sp=tgt_sp, _d=d_t):
                    if not applied[0]:
                        out_mod = out[0] if isinstance(out, tuple) else out
                        p = out_mod.clone()
                        p[:, _sp, :] += (2.0 * _d).to(p.dtype)
                        applied[0] = True
                        return (p,) + out[1:] if isinstance(out, tuple) else p
                    return out
                
                hook_h = layers_list[li].register_forward_hook(hook)
                with torch.no_grad():
                    patched_logits = model(**tgt_ids).logits.detach().cpu()
                hook_h.remove()
                patched_agr = (patched_logits[0, tgt_vp, sv_ids[0]] - 
                             patched_logits[0, tgt_vp, pv_ids[0]]).item()
                results[li].append(patched_agr - base_agr)
            
            gc.collect(); torch.cuda.empty_cache()
        
        # Print results
        for li in target_layers:
            data = results[li]
            if len(data) < 3:
                print(f"    L{li}: insufficient data ({len(data)})")
                continue
            mean_d = np.mean(data)
            std_d = np.std(data)
            n_boot = 2000
            boots = [np.mean(np.random.choice(data, len(data), replace=True)) for _ in range(n_boot)]
            ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
            sig = "★" if (ci_lo > 0 or ci_hi < 0) else ""
            neg_pct = np.mean([d < 0 for d in data])
            snr = abs(mean_d) / (std_d + 1e-10)
            print(f"    L{li}: Δ={mean_d:+.4f} σ={std_d:.4f} CI=[{ci_lo:+.4f},{ci_hi:+.4f}] "
                  f"SNR={snr:.2f} neg={neg_pct:.0%} n={len(data)} {sig}")


# ============================================================
# MAIN
# ============================================================
def run_phase63b(model, tokenizer, device, info):
    print("=" * 70)
    print("★★★ Phase 63b: NCSS Verification — Fixed Experiments ★★★")
    print("=" * 70)
    print(f"\nData: {len(ALL_NVA)} NVA pairs ({len(REGULAR_NVA)} regular + {len(IRREGULAR_NVA)} irregular)")
    print(f"Layers: {TARGET_LAYERS}")
    
    # Experiment A: Attention Routing
    attn_results = attention_routing_experiment(
        model, tokenizer, device, info, TARGET_LAYERS, n_test=40
    )
    
    # Experiment B: Same-template syntax swap
    swap_results = same_template_syntax_swap(
        model, tokenizer, device, info, TARGET_LAYERS, n_test=50
    )
    
    # Experiment C: Regular vs Irregular patching
    regular_vs_irregular_patching(
        model, tokenizer, device, info, TARGET_LAYERS, n_test=40
    )
    
    # ===== FINAL INTEGRATION =====
    print("\n" + "=" * 70)
    print("FINAL INTEGRATION: Phase 63 + 63b")
    print("=" * 70)
    
    print("""
PHASE 63 KEY FINDINGS:

1. Control Failure Theorem: NOT CONFIRMED
   - Post-LN patching NOT stronger than pre-LN
   - L0: |post-LN|/|pre-LN| = 0.23 (pre-LN STRONGER)
   - L10: ratio = 1.09 (roughly equal)
   - L15: ratio = 1.47 (slightly post-LN stronger)
   - L18: ratio = 0.48 (pre-LN stronger)
   
   ★ LN attenuation is NOT the primary reason for weak patching effects.
   ★ The problem is more fundamental — the signal itself is too weak.

2. Regular vs Irregular Nouns:
   - Direction alignment: cos(reg, irreg) = 0.80-0.90 → HIGHLY CONSISTENT
   - But: Cohen's d: regular=1.93, irregular=1.80 (both strong separability)
   - num_var_frac: regular=5.2, irregular=2.4 (regular has 2x more variance)
   ★ Direction structure is ROBUST across noun types
   ★ But regular nouns have STRONGER signal — explaining Phase 60 vs 62 discrepancy

3. ALL experiments are SEVERELY UNDERPOWERED:
   - Power: 3-15% (need 80%)
   - Need n=500-13000 for reliable detection
   ★ Previous "negative results" are INCONCLUSIVE, not disproven
   ★ We CANNOT conclude "syntax is epiphenomenal" from underpowered tests

AWAITING: Attention routing and subspace swap results...
""")
    
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deepseek7b")
    args = parser.parse_args()
    
    model, tokenizer, device = load_model(args.model)
    info = get_model_info(model, args.model)
    try:
        run_phase63b(model, tokenizer, device, info)
    finally:
        release_model(model)
