"""Cross-model comparison for Phase 384 results."""
import sys, json, numpy as np
sys.stdout.reconfigure(encoding='utf-8')

models = ["qwen3", "deepseek7b", "glm4"]
all_data = {}
for m in models:
    f = f'results/phase384_obj_residualized_category/{m}_phase384.json'
    all_data[m] = json.load(open(f, 'r', encoding='utf-8'))

print("=" * 80)
print("Phase 384 Cross-Model Summary")
print("=" * 80)

# 1. Unique R² comparison
print("\n### 1. Unique R² vs Individual R² ###\n")
print(f"{'Model/Layer':<15} {'Cat Indiv':>10} {'Cat Unique':>10} {'Cat Shared':>10} {'Cat%Unique':>10} {'Obj Unique':>10} {'NR Unique':>10}")
for m in models:
    for l_str in sorted(all_data[m]['partial_r2'].keys(), key=int):
        p = all_data[m]['partial_r2'][l_str]
        cat_ind = p['individual_r2'].get('category', 0)
        cat_uni = p['unique_r2'].get('category', 0)
        cat_shr = p['shared_r2'].get('category', 0)
        obj_uni = p['unique_r2'].get('object_identity', 0)
        nr_uni = p['unique_r2'].get('norm_ratio', 0)
        pct = cat_uni / max(cat_ind, 1e-10) * 100
        print(f"{m+' L'+l_str:<15} {cat_ind:>10.4f} {cat_uni:>10.4f} {cat_shr:>10.4f} {pct:>9.1f}% {obj_uni:>10.4f} {nr_uni:>10.4f}")

# 2. Residualization effect
print("\n### 2. Residualization Effect ###\n")
print(f"{'Model/Layer':<15} {'R2_raw':>8} {'R2_resid':>8} {'Ratio':>8} {'Acc_raw':>8} {'Acc_resid':>8}")
for m in models:
    for l_str in sorted(all_data[m]['causal_test'].keys(), key=int):
        r = all_data[m]['causal_test'][l_str]
        ratio = r['r2_resid'] / max(r['r2_raw'], 1e-10)
        print(f"{m+' L'+l_str:<15} {r['r2_raw']:>8.4f} {r['r2_resid']:>8.4f} {ratio:>8.2%} {r['acc_raw']:>8.4f} {r['acc_resid']:>8.4f}")

# 3. Causal comparison: raw vs clean
print("\n### 3. Causal: Raw vs Clean Category ###\n")
print(f"{'Model/Layer':<15} {'Raw_add':>10} {'Clean_add':>10} {'Raw_rem':>10} {'Clean_rem':>10} {'Raw_swap_d':>10} {'Clean_swap_d':>12}")
for m in models:
    for l_str in sorted(all_data[m]['causal_test'].keys(), key=int):
        r = all_data[m]['causal_test'][l_str]
        ra = r.get('raw_add_effect', {}).get('mean', 0)
        ca = r.get('clean_add_effect', {}).get('mean', 0)
        rr = r.get('raw_remove_effect', {}).get('mean', 0)
        cr = r.get('clean_remove_effect', {}).get('mean', 0)
        rs = r.get('raw_swap_effect', {}).get('diff', 0)
        cs = r.get('clean_swap_effect', {}).get('diff', 0)
        print(f"{m+' L'+l_str:<15} {ra:>+10.4f} {ca:>+10.4f} {rr:>+10.4f} {cr:>+10.4f} {rs:>+10.4f} {cs:>+12.4f}")

# 4. Key insight: direction correctness
print("\n### 4. Direction Correctness (add>0, rem<0 = correct) ###\n")
print(f"{'Model/Layer':<15} {'Raw_add_dir':>12} {'Clean_add_dir':>14} {'Raw_rem_dir':>12} {'Clean_rem_dir':>14}")
for m in models:
    for l_str in sorted(all_data[m]['causal_test'].keys(), key=int):
        r = all_data[m]['causal_test'][l_str]
        ra = r.get('raw_add_effect', {}).get('mean', 0)
        ca = r.get('clean_add_effect', {}).get('mean', 0)
        rr = r.get('raw_remove_effect', {}).get('mean', 0)
        cr = r.get('clean_remove_effect', {}).get('mean', 0)
        raw_add_ok = "OK" if ra > 0 else "WRONG"
        clean_add_ok = "OK" if ca > 0 else "WRONG"
        raw_rem_ok = "OK" if rr < 0 else "WRONG"
        clean_rem_ok = "OK" if cr < 0 else "WRONG"
        print(f"{m+' L'+l_str:<15} {raw_add_ok:>12} {clean_add_ok:>14} {raw_rem_ok:>12} {clean_rem_ok:>14}")
