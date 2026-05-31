"""Phase 308 cross-model analysis"""
import json, numpy as np
from pathlib import Path
from collections import defaultdict

models = ["qwen3", "glm4", "deepseek7b"]
model_layers = {"qwen3": 18, "glm4": 20, "deepseek7b": 14}

all_data = {}
for m in models:
    p = Path(f"results/phase308_scope_causal/{m}_scope_causal.json")
    if p.exists():
        all_data[m] = json.load(open(p, encoding='utf-8'))

# ============ SCOPE SUMMARY ============
print("=" * 80)
print("SCOPE ANALYSIS: cos(O_narrow, O_wide) by scope type")
print("=" * 80)

for m in models:
    if m not in all_data:
        continue
    d = all_data[m]
    ml = str(model_layers[m])
    st = d.get("scope_type_summary", {})
    if ml not in st:
        # Try nearest layer
        available = [int(k) for k in st.keys()]
        ml = str(min(available, key=lambda x: abs(x - model_layers[m])))
    
    if ml in st:
        print(f"\n{m.upper()} (L{ml}):")
        for stype, data in st[ml].items():
            print(f"  {stype:25s}: cos(O_n,O_w)={data['avg_cos_O_narrow_O_wide']:+.3f} "
                  f"|O_n|={data['avg_O_narrow_norm']:.1f} |O_w|={data['avg_O_wide_norm']:.1f} "
                  f"|S|={data['avg_S_scope_norm']:.1f} S/O_n={data['scope_ratio']:.3f}")

# ============ O-C ORTHOGONALITY ============
print("\n" + "=" * 80)
print("O-C ORTHOGONALITY: cos(O_not, C) and cos(O_not, C_pc1)")
print("=" * 80)

for m in models:
    if m not in all_data:
        continue
    d = all_data[m]
    oc = d.get("oc_ortho_results", {})
    ml = str(model_layers[m])
    if ml not in oc:
        available = [int(k) for k in oc.keys()]
        ml = str(min(available, key=lambda x: abs(x - model_layers[m])))
    
    if ml in oc:
        data = oc[ml]
        print(f"\n{m.upper()} (L{ml}):")
        print(f"  cos(O_not, C)       = {data['cos_O_C']:+.3f}")
        print(f"  cos(O_not, C_pc1)   = {data['cos_O_Cpc1']:+.3f} (C_pc1_var={data['C_pc1_var']:.1%})")
        print(f"  O_clean/C ratio     = {data['O_clean_C_ratio']:.3f}")
        print(f"  O_clean/Cpc1 ratio  = {data['O_clean_Cpc1_ratio']:.3f}")
        
        # Causal effects
        eff = data.get("causal_effects", {})
        if eff:
            print(f"  Causal effects (→ 'not'):")
            for pname, e in eff.items():
                if e and "not" in e:
                    print(f"    {pname:20s}: not={e['not']:+.4f}")

# ============ CROSS-FORM NEGATION ============
print("\n" + "=" * 80)
print("CROSS-FORM NEGATION: cos(O_not, O_other)")
print("=" * 80)

for m in models:
    if m not in all_data:
        continue
    d = all_data[m]
    cf = d.get("crossform_results", {})
    ml = str(model_layers[m])
    if ml not in cf:
        available = [int(k) for k in cf.keys()]
        ml = str(min(available, key=lambda x: abs(x - model_layers[m])))
    
    if ml in cf:
        cross_cos = cf[ml].get("cross_cos", {})
        print(f"\n{m.upper()} (L{ml}):")
        for pair_name, cos_per_role in cross_cos.items():
            adj_str = f"adj={cos_per_role['adj']:+.3f}" if 'adj' in cos_per_role else "adj=N/A"
            verb_str = f"verb={cos_per_role['verb']:+.3f}" if 'verb' in cos_per_role else "verb=N/A"
            print(f"  {pair_name:25s}: {adj_str}, {verb_str}")

# ============ SCOPE CAUSAL EFFECTS ============
print("\n" + "=" * 80)
print("SCOPE CAUSAL EFFECTS: O_shared, S_scope, O_narrow, O_wide → 'not'")
print("=" * 80)

for m in models:
    if m not in all_data:
        continue
    d = all_data[m]
    sc = d.get("scope_causal_results", {})
    ml = str(model_layers[m])
    if ml not in sc:
        available = [int(k) for k in sc.keys()]
        ml = str(min(available, key=lambda x: abs(x - model_layers[m])))
    
    if ml in sc:
        print(f"\n{m.upper()} (L{ml}):")
        layer_data = sc[ml]
        
        # Average across all test pairs
        avg_effects = defaultdict(list)
        for key, effects in layer_data.items():
            for pname, eff in effects.items():
                if eff and "not" in eff:
                    avg_effects[pname].append(eff["not"])
        
        for pname in ["O_shared", "S_scope", "O_narrow", "O_wide", "random",
                       "O_narrow_local", "O_wide_local", "S_scope_local"]:
            if pname in avg_effects:
                vals = avg_effects[pname]
                print(f"  {pname:20s}: mean→not={np.mean(vals):+.4f} (n={len(vals)})")

# ============ ACROSS-LAYER SCOPE TRENDS ============
print("\n" + "=" * 80)
print("ACROSS-LAYER SCOPE TRENDS")
print("=" * 80)

scope_types = ["quantifier_scope", "adverb_scope", "infinitive_scope"]
for m in models:
    if m not in all_data:
        continue
    d = all_data[m]
    st = d.get("scope_type_summary", {})
    print(f"\n{m.upper()}:")
    for stype in scope_types:
        trend = []
        for li_str in sorted(st.keys(), key=int):
            if stype in st[li_str]:
                trend.append((int(li_str), st[li_str][stype]["avg_cos_O_narrow_O_wide"]))
        if trend:
            vals_str = " ".join([f"L{l}:{v:+.2f}" for l, v in trend])
            print(f"  {stype:25s}: {vals_str}")

from collections import defaultdict
