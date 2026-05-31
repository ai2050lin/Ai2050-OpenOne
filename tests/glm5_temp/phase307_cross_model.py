"""
Phase 307 Cross-Model Analysis: Operator Orthogonality + Multi-Operator
"""
import json, numpy as np
from pathlib import Path
from collections import defaultdict

RESULT_DIR = Path("results/phase307_operator_ortho")
MODELS = ["qwen3", "glm4", "deepseek7b"]

def load_results():
    data = {}
    for m in MODELS:
        path = RESULT_DIR / f"{m}_operator_ortho.json"
        if path.exists():
            data[m] = json.load(open(path, "r", encoding="utf-8"))
    return data

def main():
    data = load_results()
    print("=" * 70)
    print("Phase 307 Cross-Model: Operator Orthogonality + Multi-Operator")
    print("=" * 70)
    
    # 1. O vs R Orthogonality
    print("\n" + "=" * 70)
    print("1. O vs R Orthogonality (cos(O_not, R))")
    print("=" * 70)
    
    for m in MODELS:
        if m not in data:
            continue
        ortho = data[m].get("orthogonality_results", {})
        print(f"\n--- {m.upper()} ---")
        for li_str, entries in ortho.items():
            not_entries = {k: v for k, v in entries.items() if v["operator"] == "not"}
            if not_entries:
                for key, val in not_entries.items():
                    print(f"  L{li_str} not/{val['role']}: cos(O,R)={val['cos_O_R']:+.3f}")
    
    # 2. Cross-Operator Similarity Matrix
    print("\n" + "=" * 70)
    print("2. Cross-Operator Similarity (avg across roles)")
    print("=" * 70)
    
    for m in MODELS:
        if m not in data:
            continue
        cross = data[m].get("cross_operator_results", {})
        # Use middle layer
        target_layers = {"qwen3": 18, "glm4": 20, "deepseek7b": 14}
        tl = str(target_layers.get(m))
        if tl in cross:
            print(f"\n--- {m.upper()} L{tl} ---")
            ops = set()
            for key, val in cross[tl].items():
                ops.add(val["op1"])
                ops.add(val["op2"])
            ops = sorted(ops)
            
            # Build matrix
            mat = {}
            for key, val in cross[tl].items():
                mat[(val["op1"], val["op2"])] = val["avg_cos"]
                mat[(val["op2"], val["op1"])] = val["avg_cos"]
            
            # Print matrix
            header = "          " + "  ".join(f"{op:<8}" for op in ops)
            print(header)
            for op1 in ops:
                row = f"{op1:<10}"
                for op2 in ops:
                    if op1 == op2:
                        row += f"  {'1.000':<8}"
                    elif (op1, op2) in mat:
                        row += f"  {mat[(op1,op2)]:+.3f}  "
                    else:
                        row += f"  {'N/A':<8}"
                print(row)
    
    # 3. Negation Cluster (not vs never)
    print("\n" + "=" * 70)
    print("3. Negation Cluster: not vs never similarity")
    print("=" * 70)
    
    for m in MODELS:
        if m not in data:
            continue
        cross = data[m].get("cross_operator_results", {})
        for li_str, entries in cross.items():
            for key, val in entries.items():
                if (val["op1"] == "not" and val["op2"] == "never") or \
                   (val["op1"] == "never" and val["op2"] == "not"):
                    per_role = val.get("per_role_cos", {})
                    roles_str = " ".join(f"{r}={c:+.3f}" for r, c in per_role.items())
                    print(f"  {m} L{li_str}: not vs never avg={val['avg_cos']:+.3f} ({roles_str})")
    
    # 4. LOO Consistency per Operator
    print("\n" + "=" * 70)
    print("4. LOO Consistency per Operator (adj only, middle layer)")
    print("=" * 70)
    
    for m in MODELS:
        if m not in data:
            continue
        loo = data[m].get("loo_results", {})
        target_layers = {"qwen3": 18, "glm4": 20, "deepseek7b": 14}
        tl = str(target_layers.get(m))
        if tl in loo:
            print(f"\n--- {m.upper()} L{tl} ---")
            adj_loo = {k: v for k, v in loo[tl].items() if v["role"] == "adj"}
            for key, val in sorted(adj_loo.items(), key=lambda x: x[1]["loo_consistency"], reverse=True):
                print(f"  {val['operator']}: LOO={val['loo_consistency']:.3f}")
    
    # 5. Causal Test Summary
    print("\n" + "=" * 70)
    print("5. Causal Test per Operator")
    print("=" * 70)
    
    for m in MODELS:
        if m not in data:
            continue
        causal = data[m].get("causal_results", {})
        target_layers = {"qwen3": 18, "glm4": 20, "deepseek7b": 14}
        tl = str(target_layers.get(m))
        if tl in causal:
            print(f"\n--- {m.upper()} L{tl} ---")
            op_shifts = defaultdict(list)
            for val in causal[tl].values():
                op_shifts[val["operator"]].append(val["causal_shift"])
            for op, shifts in sorted(op_shifts.items()):
                print(f"  {op}: avg={np.mean(shifts):+.4f} n={len(shifts)}")
    
    # 6. Summary Table: cos(O_not, R) across models
    print("\n" + "=" * 70)
    print("6. Summary: cos(O_not, R) Across Models")
    print("=" * 70)
    print(f"\n{'Model':<12} {'Layer':<8} {'adj':<10} {'verb':<10} {'noun':<10}")
    print("-" * 50)
    
    for m in MODELS:
        if m not in data:
            continue
        ortho = data[m].get("orthogonality_results", {})
        target_layers = {"qwen3": 18, "glm4": 20, "deepseek7b": 14}
        tl = str(target_layers.get(m))
        if tl in ortho:
            not_cos = {}
            for key, val in ortho[tl].items():
                if val["operator"] == "not":
                    not_cos[val["role"]] = val["cos_O_R"]
            adj_c = not_cos.get("adj", 0)
            verb_c = not_cos.get("verb", 0)
            noun_c = not_cos.get("noun", 0)
            print(f"{m:<12} {tl:<8} {adj_c:<10.3f} {verb_c:<10.3f} {noun_c:<10.3f}")

if __name__ == "__main__":
    main()
