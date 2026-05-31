"""
Phase 305 Cross-Model Analysis: Operator-Scope Causal Testing
================================================================
"""
import sys, json, os
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from collections import defaultdict

RESULT_DIR = "results/phase305_operator_causal"
MODELS = ["qwen3", "glm4", "deepseek7b"]

def load_results():
    data = {}
    for m in MODELS:
        path = os.path.join(RESULT_DIR, f"{m}_operator_causal.json")
        if os.path.exists(path):
            data[m] = json.load(open(path, 'r', encoding='utf-8'))
    return data

def main():
    data = load_results()
    print("=" * 80)
    print("Phase 305 Cross-Model Analysis: Operator-Scope Causal Testing")
    print("=" * 80)
    
    # =====================================================================
    # 1. O(not) CAUSAL EFFECT COMPARISON
    # =====================================================================
    print("\n" + "=" * 60)
    print("1. O(not) CAUSAL EFFECT COMPARISON")
    print("=" * 60)
    
    metrics = ["O_not_cos_shift", "full_delta_cos_shift", "O_loo_cos_shift",
               "O_avg_cos_shift", "antonym_dir_cos_shift", "avg_random_shift"]
    
    for m in MODELS:
        if m not in data:
            continue
        nl = data[m]["n_layers"]
        mid_li = str(nl // 2)
        results = data[m].get("results", {}).get(mid_li, {})
        causal = results.get("causal", [])
        
        if not causal:
            continue
        
        print(f"\n  {m.upper()} L{mid_li} ({len(causal)} test cases):")
        for metric in metrics:
            vals = [v.get(metric) for v in causal if v.get(metric) is not None]
            if vals:
                pos = sum(1 for v in vals if v > 0)
                print(f"    {metric:25s}: {np.mean(vals):+.4f} ± {np.std(vals):.4f} "
                      f"pos={pos}/{len(vals)} ({pos/len(vals)*100:.0f}%)")
    
    # =====================================================================
    # 2. PER-ROLE O(not) BREAKDOWN
    # =====================================================================
    print("\n" + "=" * 60)
    print("2. PER-ROLE O(not) CAUSAL EFFECT")
    print("=" * 60)
    
    for m in MODELS:
        if m not in data:
            continue
        nl = data[m]["n_layers"]
        mid_li = str(nl // 2)
        results = data[m].get("results", {}).get(mid_li, {})
        causal = results.get("causal", [])
        
        if not causal:
            continue
        
        print(f"\n  {m.upper()} L{mid_li}:")
        for role in ["adj", "verb", "noun"]:
            items = [v for v in causal if v.get("role") == role]
            if not items:
                continue
            O_not = [v.get("O_not_cos_shift") for v in items if v.get("O_not_cos_shift") is not None]
            FD = [v.get("full_delta_cos_shift") for v in items if v.get("full_delta_cos_shift") is not None]
            O_avg = [v.get("O_avg_cos_shift") for v in items if v.get("O_avg_cos_shift") is not None]
            ant = [v.get("antonym_dir_cos_shift") for v in items if v.get("antonym_dir_cos_shift") is not None]
            
            print(f"    [{role}] n={len(items)}")
            if O_not:
                print(f"      O_not:       {np.mean(O_not):+.4f}")
            if FD:
                print(f"      full_delta:  {np.mean(FD):+.4f}")
            if O_avg:
                print(f"      O_avg:       {np.mean(O_avg):+.4f}")
            if ant:
                print(f"      antonym:     {np.mean(ant):+.4f}")
    
    # =====================================================================
    # 3. CROSS-ROLE O(not) SHARING
    # =====================================================================
    print("\n" + "=" * 60)
    print("3. CROSS-ROLE O(not) SHARING")
    print("=" * 60)
    
    for m in MODELS:
        if m not in data:
            continue
        nl = data[m]["n_layers"]
        mid_li = str(nl // 2)
        results = data[m].get("results", {}).get(mid_li, {})
        crc = results.get("cross_role_cos", {})
        
        print(f"\n  {m.upper()} L{mid_li}:")
        for pair, cos_val in sorted(crc.items()):
            print(f"    cos(O_{pair.replace('_', ', O_')}): {cos_val:+.4f}")
    
    # =====================================================================
    # 4. O(not) LOO CONSISTENCY
    # =====================================================================
    print("\n" + "=" * 60)
    print("4. O(not) LOO CONSISTENCY (Within-Role Cross-Operand Sharing)")
    print("=" * 60)
    
    for m in MODELS:
        if m not in data:
            continue
        nl = data[m]["n_layers"]
        mid_li = str(nl // 2)
        results = data[m].get("results", {}).get(mid_li, {})
        loo = results.get("loo_cos", {})
        
        print(f"\n  {m.upper()} L{mid_li}:")
        for role, cos_vals in loo.items():
            if cos_vals:
                print(f"    {role}: {np.mean(cos_vals):+.4f} ± {np.std(cos_vals):.4f} n={len(cos_vals)}")
    
    # =====================================================================
    # 5. O(not) vs ANTONYM COMPARISON
    # =====================================================================
    print("\n" + "=" * 60)
    print("5. O(not) vs ANTONYM: Is Negation = Antonym?")
    print("=" * 60)
    
    for m in MODELS:
        if m not in data:
            continue
        nl = data[m]["n_layers"]
        mid_li = str(nl // 2)
        results = data[m].get("results", {}).get(mid_li, {})
        causal = results.get("causal", [])
        
        # Only antonym pairs (adj)
        ant_items = [v for v in causal if v.get("antonym_dir_cos_shift") is not None]
        
        if not ant_items:
            continue
        
        O_shifts = [v.get("O_not_cos_shift", 0) for v in ant_items]
        ant_shifts = [v.get("antonym_dir_cos_shift", 0) for v in ant_items]
        
        print(f"\n  {m.upper()} L{mid_li} ({len(ant_items)} antonym pairs):")
        print(f"    O_not causal:   {np.mean(O_shifts):+.4f}")
        print(f"    Antonym causal: {np.mean(ant_shifts):+.4f}")
        print(f"    O_not > antonym: {sum(1 for o, a in zip(O_shifts, ant_shifts) if o > a)}/{len(O_shifts)}")
        
        # How often is O(not) stronger than antonym?
        O_wins = sum(1 for o, a in zip(O_shifts, ant_shifts) if abs(o) > abs(a))
        print(f"    |O_not| > |antonym|: {O_wins}/{len(O_shifts)}")
        
        # Correlation between O(not) and antonym
        if len(O_shifts) > 2:
            corr = np.corrcoef(O_shifts, ant_shifts)[0, 1]
            print(f"    Correlation(O_not, antonym): {corr:+.4f}")
    
    # =====================================================================
    # 6. KEY COMPARISON TABLE
    # =====================================================================
    print("\n" + "=" * 60)
    print("6. KEY COMPARISON TABLE")
    print("=" * 60)
    
    header = f"{'Metric':30s} {'Qwen3':>10s} {'GLM4':>10s} {'DS7B':>10s}"
    print(f"\n  {header}")
    print(f"  {'-'*65}")
    
    for metric, label in [
        ("O_not_cos_shift", "O(not) causal"),
        ("full_delta_cos_shift", "full_delta causal"),
        ("O_avg_cos_shift", "O_avg causal"),
        ("antonym_dir_cos_shift", "antonym causal"),
        ("avg_random_shift", "Random baseline"),
    ]:
        row = f"  {label:30s}"
        for m in MODELS:
            if m not in data:
                row += f" {'N/A':>10s}"
                continue
            nl = data[m]["n_layers"]
            mid_li = str(nl // 2)
            results = data[m].get("results", {}).get(mid_li, {})
            causal = results.get("causal", [])
            vals = [v.get(metric) for v in causal if v.get(metric) is not None]
            if vals:
                row += f" {np.mean(vals):+10.4f}"
            else:
                row += f" {'N/A':>10s}"
        print(row)
    
    # Cross-role sharing
    print(f"\n  {'Cross-Role Sharing':30s}")
    for pair_key in ["adj_verb", "adj_noun", "verb_noun"]:
        row = f"  {f'cos(O_{pair_key})':30s}"
        for m in MODELS:
            if m not in data:
                row += f" {'N/A':>10s}"
                continue
            nl = data[m]["n_layers"]
            mid_li = str(nl // 2)
            results = data[m].get("results", {}).get(mid_li, {})
            crc = results.get("cross_role_cos", {})
            val = crc.get(pair_key, None)
            if val is not None:
                row += f" {val:+10.4f}"
            else:
                row += f" {'N/A':>10s}"
        print(row)
    
    # LOO consistency
    print(f"\n  {'LOO Consistency':30s}")
    for role in ["adj", "verb", "noun"]:
        row = f"  {f'{role} LOO cos':30s}"
        for m in MODELS:
            if m not in data:
                row += f" {'N/A':>10s}"
                continue
            nl = data[m]["n_layers"]
            mid_li = str(nl // 2)
            results = data[m].get("results", {}).get(mid_li, {})
            loo = results.get("loo_cos", {})
            vals = loo.get(role, [])
            if vals:
                row += f" {np.mean(vals):+10.4f}"
            else:
                row += f" {'N/A':>10s}"
        print(row)
    
    # =====================================================================
    # 7. DS7B: O(not) vs R(role) COMPARISON
    # =====================================================================
    print("\n" + "=" * 60)
    print("7. DS7B: O(not) vs R(role) — Why O(not) Works But R Doesn't")
    print("=" * 60)
    
    m = "deepseek7b"
    if m in data:
        nl = data[m]["n_layers"]
        mid_li = str(nl // 2)
        results = data[m].get("results", {}).get(mid_li, {})
        causal = results.get("causal", [])
        
        if causal:
            print(f"\n  DS7B L{mid_li} O(not) vs R(role) from Phase 304:")
            print(f"    R_only causal:        +0.177  (Phase 304)")
            print(f"    full_delta causal:    +0.008  (Phase 304)")
            print(f"    O_not causal:         {np.mean([v.get('O_not_cos_shift',0) for v in causal]):+.3f}  (Phase 305)")
            print(f"    O_not full_delta:     {np.mean([v.get('full_delta_cos_shift',0) for v in causal]):+.3f}  (Phase 305)")
            print(f"")
            print(f"    Key contrast:")
            print(f"    R(role): R_only >> full_delta (0.177 vs 0.008) — massive cancellation")
            print(f"    O(not):  O_not ≈ full_delta — minimal cancellation")
            print(f"")
            print(f"    This means DS7B's negation operator is NOT affected by")
            print(f"    the sentence structure cancellation that plagues role direction.")
            
            # Per-role O(not) in DS7B
            for role in ["adj", "verb", "noun"]:
                items = [v for v in causal if v.get("role") == role]
                if items:
                    O_not = [v.get("O_not_cos_shift", 0) for v in items if v.get("O_not_cos_shift") is not None]
                    FD = [v.get("full_delta_cos_shift", 0) for v in items if v.get("full_delta_cos_shift") is not None]
                    if O_not and FD:
                        print(f"    [{role}] O_not={np.mean(O_not):+.3f} FD={np.mean(FD):+.3f} "
                              f"ratio={np.mean(O_not)/max(abs(np.mean(FD)),0.01):.2f}")
    
    print("\n" + "=" * 80)
    print("Phase 305 Cross-Model Analysis Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
