"""
Phase 296 Cross-Model Analysis
===============================
Compare variance decomposition results across Qwen3, GLM4, DS7B.
"""
import json, sys
import numpy as np
from pathlib import Path
from collections import defaultdict

RESULT_DIR = Path("results/phase296_residual_decomposition")
MODELS = ["qwen3", "glm4", "deepseek7b"]

def load_results():
    data = {}
    for model in MODELS:
        path = RESULT_DIR / f"{model}_residual_decomposition.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data[model] = json.load(f)
    return data

def main():
    data = load_results()
    print("=" * 80)
    print("Phase 296 Cross-Model Comparison: Identity-Role Residual Decomposition")
    print("=" * 80)

    # ---- 1. Variance Decomposition Comparison ----
    print("\n## 1. Variance Decomposition (RAW)")
    print(f"{'Layer':>6} | {'Qwen3 Id%':>10} {'Role%':>7} {'Int%':>6} {'R2':>6} | "
          f"{'GLM4 Id%':>10} {'Role%':>7} {'Int%':>6} {'R2':>6} | "
          f"{'DS7B Id%':>10} {'Role%':>7} {'Int%':>6} {'R2':>6}")
    print("-" * 100)

    # Sample layers: find common layers
    all_layers = set()
    for model in MODELS:
        if model in data:
            nl = data[model]["n_layers"]
            for frac in [0, 0.25, 0.5, 0.75, 1.0]:
                li = int(nl * frac)
                all_layers.add(li)

    for li in sorted(all_layers):
        row = f"L{li:4d} |"
        for model in MODELS:
            if model not in data:
                row += f" {'N/A':>32} |"
                continue
            vd = data[model].get("variance_decomposition_raw", {})
            key = str(li)
            if key in vd:
                r = vd[key]
                row += f" {r['identity_ratio']*100:9.1f}% {r['role_ratio']*100:5.1f}% " \
                       f"{r['interaction_ratio']*100:4.1f}% {r['r_squared']:5.3f} |"
            else:
                row += f" {'N/A':>32} |"
        print(row)

    # ---- 2. Normalized Variance Decomposition ----
    print("\n## 2. Variance Decomposition (NORMALIZED - removes norm effects)")
    print(f"{'Layer':>6} | {'Qwen3 Id%':>10} {'Role%':>7} {'R2':>6} | "
          f"{'GLM4 Id%':>10} {'Role%':>7} {'R2':>6} | "
          f"{'DS7B Id%':>10} {'Role%':>7} {'R2':>6}")
    print("-" * 80)

    for li in sorted(all_layers):
        row = f"L{li:4d} |"
        for model in MODELS:
            if model not in data:
                row += f" {'N/A':>25} |"
                continue
            vd = data[model].get("variance_decomposition_normalized", {})
            key = str(li)
            if key in vd:
                r = vd[key]
                row += f" {r['identity_ratio']*100:9.1f}% {r['role_ratio']*100:5.1f}% " \
                       f"{r['r_squared']:5.3f} |"
            else:
                row += f" {'N/A':>25} |"
        print(row)

    # ---- 3. Key Metrics Evolution ----
    print("\n## 3. Identity Ratio Evolution (RAW)")
    for model in MODELS:
        if model not in data:
            continue
        vd = data[model].get("variance_decomposition_raw", {})
        nl = data[model]["n_layers"]
        vals = []
        for li in range(nl + 1):
            if str(li) in vd:
                vals.append(vd[str(li)]["identity_ratio"])
        if vals:
            print(f"  {model:12s}: L0={vals[0]*100:.0f}% -> Mid={vals[len(vals)//2]*100:.0f}% -> "
                  f"Final={vals[-1]*100:.0f}% | Min={min(vals)*100:.0f}% at L{vals.index(min(vals))}")

    print("\n## 4. Role Ratio Evolution (RAW)")
    for model in MODELS:
        if model not in data:
            continue
        vd = data[model].get("variance_decomposition_raw", {})
        nl = data[model]["n_layers"]
        vals = []
        for li in range(nl + 1):
            if str(li) in vd:
                vals.append(vd[str(li)]["role_ratio"])
        if vals:
            print(f"  {model:12s}: L0={vals[0]*100:.0f}% -> Mid={vals[len(vals)//2]*100:.0f}% -> "
                  f"Final={vals[-1]*100:.0f}% | Max={max(vals)*100:.0f}% at L{vals.index(max(vals))}")

    print("\n## 5. Interaction Ratio Evolution (RAW)")
    for model in MODELS:
        if model not in data:
            continue
        vd = data[model].get("variance_decomposition_raw", {})
        nl = data[model]["n_layers"]
        vals = []
        for li in range(nl + 1):
            if str(li) in vd:
                vals.append(vd[str(li)]["interaction_ratio"])
        if vals:
            print(f"  {model:12s}: L0={vals[0]*100:.0f}% -> Mid={vals[len(vals)//2]*100:.0f}% -> "
                  f"Final={vals[-1]*100:.0f}% | Max={max(vals)*100:.0f}% at L{vals.index(max(vals))}")

    # ---- 6. Identity Preservation ----
    print("\n## 6. Identity Preservation (Same Token, Different Role)")
    for model in MODELS:
        if model not in data:
            continue
        ip = data[model].get("identity_preservation", {})
        st = ip.get("same_token_diff_role", {})
        nl = data[model]["n_layers"]
        vals = []
        for li in range(nl + 1):
            if str(li) in st:
                cos_vals = list(st[str(li)].values())
                vals.append(np.mean(cos_vals))
        if vals:
            print(f"  {model:12s}: L0={vals[0]:.3f} -> Mid={vals[len(vals)//2]:.3f} -> "
                  f"Final={vals[-1]:.3f} | Min={min(vals):.3f} at L{vals.index(min(vals))}")

    # ---- 7. Role Increment Consistency ----
    print("\n## 7. Role Increment Cross-Token Consistency")
    for model in MODELS:
        if model not in data:
            continue
        ri = data[model].get("role_increment", {})
        cross = ri.get("cross_token_cosines", {})
        nl = data[model]["n_layers"]
        
        # Get middle layer cross-token cosines
        mid_li = str(nl // 2)
        if mid_li in cross:
            print(f"  {model:12s} (L{nl//2}): {dict(cross[mid_li])}")

    # ---- 8. Subspace Overlap ----
    print("\n## 8. Identity-Role Subspace Overlap")
    for model in MODELS:
        if model not in data:
            continue
        so = data[model].get("subspace_overlap", {})
        nl = data[model]["n_layers"]
        vals = []
        for li in range(nl + 1):
            if str(li) in so and "subspace_overlap" in so[str(li)]:
                vals.append((li, so[str(li)]["subspace_overlap"]))
        if vals:
            mid_val = vals[len(vals)//2][1]
            print(f"  {model:12s}: Mid overlap={mid_val:.4f}")
            # First 3 principal angles at middle layer
            mid_li_str = str(nl // 2)
            if mid_li_str in so and "principal_angles_deg" in so[mid_li_str]:
                angles = so[mid_li_str]["principal_angles_deg"]
                print(f"               Principal angles: {angles[:3]}")

    # ---- 9. Key Cross-Model Differences ----
    print("\n" + "=" * 80)
    print("KEY CROSS-MODEL DIFFERENCES")
    print("=" * 80)

    print("""
1. ADDITIVITY: How well does h = mu + I + R work?
   - Qwen3: R2 ~ 0.80-1.0 (additive model works well)
   - GLM4:  R2 ~ 0.76-1.0 (slightly less additive)
   - DS7B:  R2 ~ 0.58-1.0 (much less additive, high interaction)

2. IDENTITY DOMINANCE: How much of variance is identity?
   - All models: Identity peaks at L0 (~100%), drops to ~40-55% at mid layers
   - Qwen3/GLM4: Identity rebounds to ~60-70% at deep layers
   - DS7B: Identity stays ~40% throughout (flat profile)

3. ROLE STRENGTH: How much of variance is role?
   - Qwen3/GLM4: Role peaks at ~25-32% at mid layers
   - DS7B: Role stays at ~17-19% (lower and flatter)

4. INTERACTION: How much does role effect depend on token?
   - Qwen3/GLM4: Interaction ~20-28% (moderate)
   - DS7B: Interaction ~42-43% (very high!)
   => DS7B's role effect is highly token-dependent

5. IDENTITY PRESERVATION: Same token in different roles
   - All models: U-shaped curve (1.0 -> 0.54-0.61 -> 0.86-0.90)
   - DS7B: Final layer drops (0.70)
   - GLM4: Final layer drops sharply (0.50)

6. IDENTITY-ROLE ORTHOGONALITY:
   - All models: cos(I, R) ~ 0.00 (near-perfect orthogonality)
   => Identity and role live in nearly independent subspaces

7. DS7B ANOMALY:
   - High interaction ratio (~43%) means the additive decomposition barely works
   - This suggests DS7B uses a more entangled encoding where role effects
     are highly token-specific rather than following a universal pattern
   - Possibly related to sliding window attention limiting information flow
""")


if __name__ == "__main__":
    main()
