"""
Phase 60b: 精确语法方向Patching — 解决same-position悖论
=====================================================

Phase 60关键发现:
  cos(num_pos2, num_pos3) = 0.80-0.95 → Number是位置不变的!
  cos(num, d_pos) ≈ 0 → Number与Position正交!
  跨位置方向patching有效 (L0-L18一致负delta)

但same-position方向patching给出正delta (悖论)!

可能解释:
  1. d_number包含了词汇层信息("cats" vs "cat" token差异)
  2. 同位置添加d_number破坏了该位置的表示结构
  3. LayerNorm效应: 下一层的LayerNorm可能"修正"了添加的方向

验证方案:
  A. 用pos3方向patch pos2 (交叉提取, 同位置应用)
  B. 用pos2方向patch pos3 (原始提取, 跨位置应用)
  C. 测试更大alpha值 (2.0, 5.0)
  D. 分析LayerNorm效应
  E. 用归一化方向 (只保留方向, 去掉幅度)
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
]

NOUNS_SET = set()
for sn, pn, _, _, _ in NVA_PAIRS:
    NOUNS_SET.add(sn.lower())
    NOUNS_SET.add(pn.lower())


def svo_pos_fn(tok, toks):
    decoded = [tok.decode([t]).strip() for t in toks]
    subj_pos = None
    for i, d in enumerate(decoded):
        if d.lower() in NOUNS_SET:
            subj_pos = i + 1
            break
    verb_pos = subj_pos + 1 if subj_pos else None
    return subj_pos, verb_pos


def adv_pos_fn(tok, toks):
    decoded = [tok.decode([t]).strip() for t in toks]
    subj_pos = None
    for i, d in enumerate(decoded):
        if d.lower() in NOUNS_SET:
            subj_pos = i + 1
            break
    verb_pos = subj_pos + 1 if subj_pos else None
    return subj_pos, verb_pos


def collect_activations(model, tokenizer, device, sentences, pos_fn, target_layers, label=""):
    layers = get_layers(model)
    results = defaultdict(lambda: {"subj": [], "verb": []})
    valid_count = 0
    
    for si, sent in enumerate(sentences):
        if si % 10 == 0 and si > 0:
            print(f"  {label} {si}/{len(sentences)}")
        
        toks = tokenizer.encode(sent, add_special_tokens=False)
        subj_pos, verb_pos = pos_fn(tokenizer, toks)
        if subj_pos is None or verb_pos is None:
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
        
        input_ids = tokenizer(sent, return_tensors="pt").to(device)
        with torch.no_grad():
            try:
                model(**input_ids)
            except:
                for h in hooks: h.remove()
                continue
        
        for h in hooks: h.remove()
        
        for li in target_layers:
            if li in captured:
                h = captured[li]
                if subj_pos < h.shape[1]:
                    results[li]["subj"].append(h[0, subj_pos, :].float().numpy())
                if verb_pos < h.shape[1]:
                    results[li]["verb"].append(h[0, verb_pos, :].float().numpy())
        
        valid_count += 1
        del captured
        gc.collect()
        torch.cuda.empty_cache()
    
    print(f"  {label}: {valid_count}/{len(sentences)} valid")
    return results, valid_count


def extract_directions(svo_sing, svo_plur, adv_sing, adv_plur, li):
    sing2 = np.array(svo_sing[li]["subj"])
    plur2 = np.array(svo_plur[li]["subj"])
    sing3 = np.array(adv_sing[li]["subj"])
    plur3 = np.array(adv_plur[li]["subj"])
    
    d_num2 = plur2.mean(0) - sing2.mean(0)
    d_num3 = plur3.mean(0) - sing3.mean(0)
    
    # Position direction
    all2 = np.vstack([sing2, plur2])
    all3 = np.vstack([sing3, plur3])
    d_pos = all2.mean(0) - all3.mean(0)
    
    # Clean directions (orthogonal to position)
    def ortho(v, u):
        return v - np.dot(v, u) / (np.dot(u, u) + 1e-10) * u
    
    d_clean2 = ortho(d_num2, d_pos)
    d_clean3 = ortho(d_num3, d_pos)
    
    # Unit directions
    def unit(v):
        n = np.linalg.norm(v)
        return v / n if n > 1e-10 else v
    
    return {
        "d_num2": d_num2, "d_num3": d_num3,
        "d_pos": d_pos,
        "d_clean2": d_clean2, "d_clean3": d_clean3,
        "u_num2": unit(d_num2), "u_num3": unit(d_num3),
        "u_clean2": unit(d_clean2), "u_clean3": unit(d_clean3),
        "u_pos": unit(d_pos),
        "num2_norm": float(np.linalg.norm(d_num2)),
        "num3_norm": float(np.linalg.norm(d_num3)),
        "pos_norm": float(np.linalg.norm(d_pos)),
    }


def direction_patch(model, tokenizer, device, layers, sent, sv, pv,
                    subj_pos, verb_pos, direction, alpha, patch_pos, layer_idx):
    """Additive direction patching at specific layer and position"""
    direction_t = torch.tensor(direction, dtype=torch.float16, device=device)
    
    sv_ids = tokenizer.encode(sv, add_special_tokens=False)
    pv_ids = tokenizer.encode(pv, add_special_tokens=False)
    if not sv_ids or not pv_ids:
        return None
    
    input_ids = tokenizer(sent, return_tensors="pt").to(device)
    
    # Baseline
    with torch.no_grad():
        base_logits = model(**input_ids).logits.detach().cpu()
    
    if verb_pos >= base_logits.shape[1]:
        return None
    
    base_agr = (base_logits[0, verb_pos, sv_ids[0]] - 
               base_logits[0, verb_pos, pv_ids[0]]).item()
    base_top = base_logits[0, verb_pos].argmax().item()
    
    # Patched
    applied = [False]
    def add_hook(m, inp, out):
        if not applied[0]:
            if isinstance(out, tuple):
                p = out[0].clone()
                p[:, patch_pos, :] += (alpha * direction_t).to(p.dtype)
                applied[0] = True
                return (p,) + out[1:]
            else:
                p = out.clone()
                p[:, patch_pos, :] += (alpha * direction_t).to(p.dtype)
                applied[0] = True
                return p
        return out
    
    hook = layers[layer_idx].register_forward_hook(add_hook)
    with torch.no_grad():
        patched_logits = model(**input_ids).logits.detach().cpu()
    hook.remove()
    
    patched_agr = (patched_logits[0, verb_pos, sv_ids[0]] - 
                  patched_logits[0, verb_pos, pv_ids[0]]).item()
    patched_top = patched_logits[0, verb_pos].argmax().item()
    
    return {
        "delta": patched_agr - base_agr,
        "base_agr": base_agr,
        "patched_agr": patched_agr,
        "verb_changed": base_top != patched_top,
    }


def run_phase60b(model, tokenizer, device, info):
    print("=" * 70)
    print("★★★ Phase 60b: Resolving the Same-Position Paradox ★★★")
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
    
    # Collect activations
    print("\nStep 1: Collecting activations...")
    t0 = time.time()
    
    svo_sing_act, _ = collect_activations(model, tokenizer, device, svo_sing, svo_pos_fn, target_layers, "SVO-sing")
    svo_plur_act, _ = collect_activations(model, tokenizer, device, svo_plur, svo_pos_fn, target_layers, "SVO-plur")
    adv_sing_act, _ = collect_activations(model, tokenizer, device, adv_sing, adv_pos_fn, target_layers, "Adv-sing")
    adv_plur_act, _ = collect_activations(model, tokenizer, device, adv_plur, adv_pos_fn, target_layers, "Adv-plur")
    
    print(f"  Took {time.time()-t0:.1f}s")
    
    # Extract directions
    print("\nStep 2: Direction extraction...")
    all_dirs = {}
    for li in target_layers:
        if li in svo_sing_act and svo_sing_act[li]["subj"]:
            all_dirs[li] = extract_directions(svo_sing_act, svo_plur_act, adv_sing_act, adv_plur_act, li)
    
    # ===== Key Experiments =====
    test_svo = svo_data[:20]
    test_adv = adv_data[:20]
    
    # Alpha values to test
    alphas = [1.0, 2.0, 5.0]
    
    results = {}
    
    for li in target_layers:
        if li not in all_dirs:
            continue
        
        dirs = all_dirs[li]
        
        for alpha in alphas:
            key = (li, alpha)
            results[key] = {}
            
            # === Exp 1: Same-position, SAME-extraction direction (original) ===
            # Add d_num_pos2 to SVO subj@pos2
            effects = []
            for sing, plur, sv, pv, sn, pn in test_svo:
                toks = tokenizer.encode(sing, add_special_tokens=False)
                sp, vp = svo_pos_fn(tokenizer, toks)
                if sp is None or vp is None: continue
                
                r = direction_patch(model, tokenizer, device, layers, sing, sv, pv,
                                   sp, vp, dirs["d_num2"], alpha, sp, li)
                if r: effects.append(r)
            
            if effects:
                results[key]["same_same"] = {
                    "delta": np.mean([e["delta"] for e in effects]),
                    "verb_pct": np.mean([e["verb_changed"] for e in effects]) * 100,
                    "neg_ratio": np.mean([e["delta"] < 0 for e in effects]),
                    "n": len(effects),
                }
            
            # === Exp 2: Same-position, CROSS-extraction direction ===
            # Add d_num_pos3 to SVO subj@pos2
            effects = []
            for sing, plur, sv, pv, sn, pn in test_svo:
                toks = tokenizer.encode(sing, add_special_tokens=False)
                sp, vp = svo_pos_fn(tokenizer, toks)
                if sp is None or vp is None: continue
                
                r = direction_patch(model, tokenizer, device, layers, sing, sv, pv,
                                   sp, vp, dirs["d_num3"], alpha, sp, li)
                if r: effects.append(r)
            
            if effects:
                results[key]["same_cross"] = {
                    "delta": np.mean([e["delta"] for e in effects]),
                    "verb_pct": np.mean([e["verb_changed"] for e in effects]) * 100,
                    "neg_ratio": np.mean([e["delta"] < 0 for e in effects]),
                    "n": len(effects),
                }
            
            # === Exp 3: Cross-position, same-extraction direction ===
            # Add d_num_pos2 to Adv subj@pos3
            effects = []
            for sing, plur, sv, pv, sn, pn in test_adv:
                toks = tokenizer.encode(sing, add_special_tokens=False)
                sp, vp = adv_pos_fn(tokenizer, toks)
                if sp is None or vp is None: continue
                
                r = direction_patch(model, tokenizer, device, layers, sing, sv, pv,
                                   sp, vp, dirs["d_num2"], alpha, sp, li)
                if r: effects.append(r)
            
            if effects:
                results[key]["cross_same"] = {
                    "delta": np.mean([e["delta"] for e in effects]),
                    "verb_pct": np.mean([e["verb_changed"] for e in effects]) * 100,
                    "neg_ratio": np.mean([e["delta"] < 0 for e in effects]),
                    "n": len(effects),
                }
            
            # === Exp 4: Cross-position, clean direction (position-removed) ===
            # Add d_clean_pos2 to Adv subj@pos3
            effects = []
            for sing, plur, sv, pv, sn, pn in test_adv:
                toks = tokenizer.encode(sing, add_special_tokens=False)
                sp, vp = adv_pos_fn(tokenizer, toks)
                if sp is None or vp is None: continue
                
                r = direction_patch(model, tokenizer, device, layers, sing, sv, pv,
                                   sp, vp, dirs["d_clean2"], alpha, sp, li)
                if r: effects.append(r)
            
            if effects:
                results[key]["cross_clean"] = {
                    "delta": np.mean([e["delta"] for e in effects]),
                    "verb_pct": np.mean([e["verb_changed"] for e in effects]) * 100,
                    "neg_ratio": np.mean([e["delta"] < 0 for e in effects]),
                    "n": len(effects),
                }
            
            # === Exp 5: Same-position, CLEAN direction ===
            # Add d_clean_pos2 to SVO subj@pos2
            effects = []
            for sing, plur, sv, pv, sn, pn in test_svo:
                toks = tokenizer.encode(sing, add_special_tokens=False)
                sp, vp = svo_pos_fn(tokenizer, toks)
                if sp is None or vp is None: continue
                
                r = direction_patch(model, tokenizer, device, layers, sing, sv, pv,
                                   sp, vp, dirs["d_clean2"], alpha, sp, li)
                if r: effects.append(r)
            
            if effects:
                results[key]["same_clean"] = {
                    "delta": np.mean([e["delta"] for e in effects]),
                    "verb_pct": np.mean([e["verb_changed"] for e in effects]) * 100,
                    "neg_ratio": np.mean([e["delta"] < 0 for e in effects]),
                    "n": len(effects),
                }
            
            # === Exp 6: UNIT direction, same-position ===
            # Add unit(d_num_pos2) * alpha * num2_norm to SVO subj@pos2
            scaled_dir = dirs["u_num2"] * dirs["num2_norm"]  # Same magnitude as d_num2
            effects = []
            for sing, plur, sv, pv, sn, pn in test_svo:
                toks = tokenizer.encode(sing, add_special_tokens=False)
                sp, vp = svo_pos_fn(tokenizer, toks)
                if sp is None or vp is None: continue
                
                r = direction_patch(model, tokenizer, device, layers, sing, sv, pv,
                                   sp, vp, scaled_dir, alpha, sp, li)
                if r: effects.append(r)
            
            if effects:
                results[key]["same_unit"] = {
                    "delta": np.mean([e["delta"] for e in effects]),
                    "verb_pct": np.mean([e["verb_changed"] for e in effects]) * 100,
                    "neg_ratio": np.mean([e["delta"] < 0 for e in effects]),
                    "n": len(effects),
                }
        
        gc.collect()
        torch.cuda.empty_cache()
    
    # ===== Print Results =====
    print("\n" + "=" * 70)
    print("RESULTS: Direction Patching Comparison")
    print("=" * 70)
    
    for alpha in alphas:
        print(f"\n--- α = {alpha} ---")
        print(f"{'Layer':>6} | {'same+same':>12} | {'same+cross':>12} | {'same+clean':>12} | {'cross+same':>12} | {'cross+clean':>12}")
        print("-" * 80)
        
        for li in target_layers:
            key = (li, alpha)
            if key not in results:
                continue
            
            r = results[key]
            vals = []
            for exp_name in ["same_same", "same_cross", "same_clean", "cross_same", "cross_clean"]:
                if exp_name in r:
                    d = r[exp_name]["delta"]
                    neg = r[exp_name]["neg_ratio"]
                    vals.append(f"{d:+.4f}({neg:.0%})")
                else:
                    vals.append("N/A")
            
            print(f"  L{li:>3} | {vals[0]:>12} | {vals[1]:>12} | {vals[2]:>12} | {vals[3]:>12} | {vals[4]:>12}")
    
    # ===== Key Analysis =====
    print("\n" + "=" * 70)
    print("KEY ANALYSIS: Same-Position Paradox")
    print("=" * 70)
    
    print("\nQuestion: Why does same-position patching give wrong-direction delta?")
    print("Hypothesis: d_number contains position-correlated contamination")
    
    for li in [10, 15, 20]:
        if (li, 2.0) not in results:
            continue
        r = results[(li, 2.0)]
        
        same_same = r.get("same_same", {})
        same_cross = r.get("same_cross", {})
        same_clean = r.get("same_clean", {})
        cross_same = r.get("cross_same", {})
        cross_clean = r.get("cross_clean", {})
        
        print(f"\n  L{li} (α=2.0):")
        print(f"    same+same:  Δ={same_same.get('delta', 'N/A'):+.4f}, neg%={same_same.get('neg_ratio', 0):.0%}")
        print(f"    same+cross: Δ={same_cross.get('delta', 'N/A'):+.4f}, neg%={same_cross.get('neg_ratio', 0):.0%}")
        print(f"    same+clean: Δ={same_clean.get('delta', 'N/A'):+.4f}, neg%={same_clean.get('neg_ratio', 0):.0%}")
        print(f"    cross+same: Δ={cross_same.get('delta', 'N/A'):+.4f}, neg%={cross_same.get('neg_ratio', 0):.0%}")
        print(f"    cross+clean:Δ={cross_clean.get('delta', 'N/A'):+.4f}, neg%={cross_clean.get('neg_ratio', 0):.0%}")
        
        # Check if cross-extraction fixes same-position
        if same_same.get('delta', 0) > 0 and same_cross.get('delta', 0) < 0:
            print(f"    ★ CROSS-EXTRACTION FIXES SAME-POSITION! → d_num_pos2 has position contamination")
        elif same_same.get('delta', 0) > 0 and same_clean.get('delta', 0) < 0:
            print(f"    ★ CLEAN DIRECTION FIXES SAME-POSITION! → Position component was the problem")
        elif same_same.get('delta', 0) > 0:
            print(f"    ⚠ Same-position still broken → Issue is not position contamination")
        
        # Check if cross-position works
        if cross_same.get('delta', 0) < -0.03 or cross_clean.get('delta', 0) < -0.03:
            print(f"    ★ CROSS-POSITION WORKS! → Number is position-invariant")
    
    # Cosine diagnostic
    print("\n" + "=" * 70)
    print("DIRECTION DIAGNOSTICS")
    print("=" * 70)
    
    for li in target_layers:
        if li not in all_dirs:
            continue
        dirs = all_dirs[li]
        
        def cos(a, b):
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            return float(np.dot(a, b) / (na * nb)) if na > 1e-10 and nb > 1e-10 else 0
        
        d2, d3, dp = dirs["d_num2"], dirs["d_num3"], dirs["d_pos"]
        dc2 = dirs["d_clean2"]
        
        print(f"\n  L{li}:")
        print(f"    cos(d_num2, d_num3) = {cos(d2, d3):.4f}")
        print(f"    cos(d_num2, d_pos)  = {cos(d2, dp):.4f}")
        print(f"    cos(d_clean2, d_num2) = {cos(dc2, d2):.4f}")
        print(f"    ||d_num2|| = {dirs['num2_norm']:.2f}, ||d_num3|| = {dirs['num3_norm']:.2f}, ||d_pos|| = {dirs['pos_norm']:.2f}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="deepseek7b")
    args = parser.parse_args()
    
    print(f"Loading model: {args.model}")
    model, tokenizer, device = load_model(args.model)
    info = get_model_info(model, args.model)
    print(f"Model: {info.name}, Layers={info.n_layers}, d_model={info.d_model}")
    
    try:
        results = run_phase60b(model, tokenizer, device, info)
    finally:
        release_model(model)
        print("\nDone.")
