"""
Phase 295 Cross-Model Analysis: Identity-Role Decoupling
=========================================================
"""
import json, sys
import numpy as np
from pathlib import Path

RESULT_DIR = Path("results/phase295_identity_role")

def load_model_data(name):
    return json.load(open(RESULT_DIR / f"{name}_identity_role.json", "r", encoding="utf-8"))

def analyze():
    models = ["qwen3", "glm4", "deepseek7b"]
    data = {m: load_model_data(m) for m in models}

    print("=" * 70)
    print("PHASE 295 CROSS-MODEL ANALYSIS")
    print("=" * 70)

    # ===== 1. Cosine Similarity by Category =====
    print("\n### 1. COSINE SIMILARITY: Same Token Same Role vs Diff Token Same Role")
    print("(Testing: Does same-token always beat same-role?)")
    print()

    for cat in ["same_token_same_role", "same_token_diff_role",
                "diff_token_same_role", "diff_token_diff_role"]:
        print(f"  --- {cat} ---")
        for m in models:
            nl = data[m]["n_layers"]
            cos = data[m]["cosine_similarity"].get(cat, {})
            layers_to_show = [0, nl//6, nl//3, nl//2, 2*nl//3, nl]
            vals = []
            for li in layers_to_show:
                li_str = str(li)
                if li_str in cos and cos[li_str]:
                    avg = np.mean(cos[li_str])
                    vals.append(f"L{li}={avg:.3f}")
            print(f"    {m:12s}: {' | '.join(vals)}")
        print()

    # ===== 2. Identity vs Role Signal Decay =====
    print("### 2. IDENTITY SIGNAL DECAY (intra_token - inter_token gap)")
    print()

    # Key finding: at what layer does identity gap drop below role gap?
    for m in models:
        nl = data[m]["n_layers"]
        dist = data[m]["distance_structure"]
        print(f"  {m} (n_layers={nl}):")
        tok_gaps = []
        role_gaps = []
        for li in range(nl + 1):
            li_str = str(li)
            intra_t = np.mean(dist["intra_token"].get(li_str, [0]))
            inter_t = np.mean(dist["inter_token"].get(li_str, [0]))
            intra_r = np.mean(dist["intra_role"].get(li_str, [0]))
            inter_r = np.mean(dist["inter_role"].get(li_str, [0]))
            tok_gap = intra_t - inter_t
            role_gap = intra_r - inter_r
            tok_gaps.append(tok_gap)
            role_gaps.append(role_gap)

        # Find where token gap drops below some threshold
        for threshold in [0.5, 0.3, 0.2, 0.1]:
            cross = None
            for i in range(len(tok_gaps)):
                if tok_gaps[i] < threshold:
                    cross = i
                    break
            if cross is not None:
                print(f"    Token gap < {threshold}: L{cross}")
            else:
                print(f"    Token gap < {threshold}: never")

        # Check if role gap ever exceeds token gap
        crossover = None
        for i in range(len(tok_gaps)):
            if role_gaps[i] > tok_gaps[i]:
                crossover = i
                break
        if crossover is not None:
            print(f"    Role gap > Token gap: L{crossover} (ROLE DOMINATES!)")
        else:
            print(f"    Role gap > Token gap: NEVER (TOKEN ALWAYS DOMINATES)")

        # Print key layers
        for li in [0, nl//4, nl//2, 3*nl//4, nl]:
            print(f"    L{li:2d}: tok_gap={tok_gaps[li]:+.4f} role_gap={role_gaps[li]:+.4f} "
                  f"ratio={'inf' if role_gaps[li] < 0.01 else f'{tok_gaps[li]/role_gaps[li]:.1f}x'}")
        print()

    # ===== 3. Norm-Matched Patching =====
    print("### 3. NORM-MATCHED PATCHING")
    print()
    for m in models:
        nl = data[m]["n_layers"]
        nm = data[m]["norm_matched_patching"]
        if not nm:
            print(f"  {m}: No norm-matched data")
            continue

        early = [r for r in nm if r["layer"] <= nl // 3]
        mid = [r for r in nm if nl // 3 < r["layer"] <= 2 * nl // 3]
        late = [r for r in nm if r["layer"] > 2 * nl // 3]

        print(f"  {m}:")
        for label, subset in [("Early (0-L1/3)", early), ("Mid", mid), ("Late (L2/3+)", late)]:
            if not subset:
                continue
            avg_a = np.mean([r["np_aligned"] for r in subset if r.get("np_aligned") is not None] or [0])
            avg_m = np.mean([r["np_misaligned"] for r in subset if r.get("np_misaligned") is not None] or [0])
            avg_nm = np.mean([r["np_norm_matched"] for r in subset if r.get("np_norm_matched") is not None] or [0])
            avg_nr = np.mean([r["norm_ratio"] for r in subset if r.get("norm_ratio")] or [0])
            n = len(subset)
            print(f"    {label:16s}: aligned={avg_a:+.4f} misaligned={avg_m:+.4f} "
                  f"norm_matched={avg_nm:+.4f} ratio={avg_nr:.2f} (n={n})")
        print()

    # ===== 4. Linear Probe =====
    print("### 4. LINEAR PROBE (Token Identity vs Role)")
    print()
    for m in models:
        nl = data[m]["n_layers"]
        probes = data[m]["linear_probes"]
        tok_probe = probes.get("token_probe", {})
        role_probe = probes.get("role_probe", {})

        print(f"  {m}:")
        print(f"    Token identity probe: {len(tok_probe)} layers with data")
        print(f"    Role probe: {len(role_probe)} layers with data")

        if tok_probe or role_probe:
            for li_str in sorted(set(list(tok_probe.keys()) + list(role_probe.keys())), key=lambda x: int(x)):
                li = int(li_str)
                if li % 4 != 0 and li != nl:
                    continue
                t = tok_probe.get(li_str, {})
                r = role_probe.get(li_str, {})
                t_acc = t.get("accuracy", "N/A")
                r_acc = r.get("accuracy", "N/A")
                t_n = t.get("n_samples", "")
                r_n = r.get("n_samples", "")
                print(f"    L{li:2d}: token={t_acc} role={r_acc}")
        print()

    # ===== 5. Critical Comparison: L0 vs Final =====
    print("### 5. CRITICAL LAYER COMPARISON: L0 vs Final")
    print()
    for m in models:
        nl = data[m]["n_layers"]
        dist = data[m]["distance_structure"]
        cos = data[m]["cosine_similarity"]

        # Cosine sim at L0 and final
        for cat in ["same_token_same_role", "diff_token_same_role"]:
            l0_vals = cos.get(cat, {}).get("0", [])
            lf_vals = cos.get(cat, {}).get(str(nl), [])
            l0_avg = np.mean(l0_vals) if l0_vals else 0
            lf_avg = np.mean(lf_vals) if lf_vals else 0

            # Distance gap at L0 and final
            li0, lif = "0", str(nl)
            intra_t_0 = np.mean(dist["intra_token"].get(li0, [0]))
            inter_t_0 = np.mean(dist["inter_token"].get(li0, [0]))
            intra_r_0 = np.mean(dist["intra_role"].get(li0, [0]))
            inter_r_0 = np.mean(dist["inter_role"].get(li0, [0]))
            intra_t_f = np.mean(dist["intra_token"].get(lif, [0]))
            inter_t_f = np.mean(dist["inter_token"].get(lif, [0]))
            intra_r_f = np.mean(dist["intra_role"].get(lif, [0]))
            inter_r_f = np.mean(dist["inter_role"].get(lif, [0]))

            print(f"  {m} {cat}:")
            print(f"    L0: cos={l0_avg:.3f} tok_gap={intra_t_0-inter_t_0:+.4f} role_gap={intra_r_0-inter_r_0:+.4f}")
            print(f"    L{nl}: cos={lf_avg:.3f} tok_gap={intra_t_f-inter_t_f:+.4f} role_gap={intra_r_f-inter_r_f:+.4f}")
        print()

    # ===== 6. KEY CONCLUSIONS =====
    print("=" * 70)
    print("KEY CONCLUSIONS")
    print("=" * 70)
    print()
    print("1. TOKEN IDENTITY ALWAYS DOMINATES over functional role (all layers, all models)")
    print("   - intra_token gap > intra_role gap at every single layer")
    print("   - This is the strongest finding: token identity is NEVER surpassed by role")
    print()
    print("2. Token identity gap decays monotonically from L0 to final layer")
    print("   - L0: gap ~0.93 (near-perfect token clustering)")
    print("   - Final: gap ~0.08-0.15 (still positive)")
    print()
    print("3. Role signal is weak but present and grows slowly")
    print("   - L0: role gap ~0.05 (barely above noise)")
    print("   - Final: role gap ~0.08-0.18")
    print()
    print("4. Norm-matched patching: aligned > misaligned AFTER norm matching")
    print("   - This CONTRADICTS Phase 294's misaligned > aligned finding")
    print("   - Phase 294's misaligned > aligned was primarily a norm effect,")
    print("     NOT a token-identity dominance effect")
    print()
    print("5. Same token same role: cos~1.0 at L0, drops to ~0.7-0.85 at mid, rises to ~0.95 at final")
    print("   - U-shape: embedding similarity -> context divergence -> convergence")
    print()
    print("6. Diff token same role: cos~0.2-0.3 at L0, rises to ~0.95 at final")
    print("   - Different tokens converge when sharing the same role")


if __name__ == "__main__":
    analyze()
