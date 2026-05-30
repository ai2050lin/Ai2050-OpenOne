"""Phase 297 Cross-Model Comparison"""
import sys, os, json, numpy as np
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
from collections import defaultdict

RESULT_DIR = Path("results/phase297_role_frame")

models = ["qwen3", "glm4", "deepseek7b"]
data = {}
for m in models:
    p = RESULT_DIR / f"{m}_role_frame.json"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            data[m] = json.load(f)

print("=" * 70)
print("Phase 297: Cross-Model Comparison — Identity-Role-Frame Decomposition")
print("=" * 70)

# ===== 1. Frame variance within role =====
print("\n[A] Frame (Pair) Variance within Role (RAW, mid-layer)")
print("-" * 70)
for m in models:
    if m not in data:
        continue
    nl = data[m]["n_layers"]
    mid = nl // 2
    apr = data[m].get("anova_per_role_raw", {})
    ms = apr.get(str(mid), {}).get("roles", {})
    parts = []
    for role in ["adj", "verb", "noun"]:
        r = ms.get(role, {})
        if r:
            parts.append(f"{role}: Id={r['identity_ratio']*100:.1f}% Pair={r['pair_ratio']*100:.1f}% Interact={r['interaction_ratio']*100:.1f}%")
    print(f"  {m:12s} L{mid}: {' | '.join(parts)}")

# ===== 2. Pair-averaged cross-role ANOVA =====
print("\n[B] Pair-Averaged Cross-Role ANOVA (RAW)")
print("-" * 70)
print(f"  {'Model':12s} {'Layer':6s} {'Id%':6s} {'Role%':6s} {'Interact%':9s} {'R²':6s}")
for m in models:
    if m not in data:
        continue
    nl = data[m]["n_layers"]
    cra = data[m].get("cross_role_anova_raw", {})
    for li_str in sorted(cra.keys(), key=int):
        li = int(li_str)
        if li in [0, nl//4, nl//2, 3*nl//4, nl-1, nl]:
            r = cra[li_str]
            print(f"  {m:12s} L{li:4d} {r['identity_ratio']*100:5.1f} {r['role_ratio']*100:5.1f} "
                  f"{r['interaction_ratio']*100:8.1f} {r['r_squared']:.3f}")

# ===== 3. Cross-pair consistency =====
print("\n[C] Cross-Pair Consistency of Role Increment")
print("-" * 70)
print(f"  {'Model':12s} {'Layer':6s} {'SharedRatio':12s} {'CrossPairCos':12s}")
for m in models:
    if m not in data:
        continue
    nl = data[m]["n_layers"]
    rid = data[m].get("role_increment_decomposition", {})
    sr_data = rid.get("shared_role_ratio", {})
    cc_data = rid.get("cross_pair_cos", {})
    for li_str in sorted(sr_data.keys(), key=int):
        li = int(li_str)
        if li in [0, nl//4, nl//2, 3*nl//4, nl-1, nl]:
            sr = list(sr_data[li_str].values())
            cc = list(cc_data.get(li_str, {}).values())
            if sr and cc:
                print(f"  {m:12s} L{li:4d} {np.mean(sr):.4f}       {np.mean(cc):.4f}")

# ===== 4. Identity preservation =====
print("\n[D] Identity Preservation (Same Pair vs Diff Pair)")
print("-" * 70)
print(f"  {'Model':12s} {'Layer':6s} {'SamePair':10s} {'DiffPair':10s} {'SameRoleDiffPair':16s}")
for m in models:
    if m not in data:
        continue
    nl = data[m]["n_layers"]
    ipc = data[m].get("identity_preservation_controlled", {})
    sp_data = ipc.get("same_pair_id_pres", {})
    dp_data = ipc.get("diff_pair_id_pres", {})
    sr_data = ipc.get("same_role_pair_pres", {})
    for li_str in sorted(sp_data.keys(), key=int):
        li = int(li_str)
        if li in [0, nl//4, nl//2, 3*nl//4, nl-1, nl]:
            sp = list(sp_data[li_str].values())
            dp = list(dp_data.get(li_str, {}).values())
            sr = list(sr_data.get(li_str, {}).values())
            if sp and dp and sr:
                print(f"  {m:12s} L{li:4d} {np.mean(sp):.4f}     {np.mean(dp):.4f}     {np.mean(sr):.4f}")

# ===== 5. Key comparison: Phase 296 vs 297 =====
print("\n[E] Phase 296 vs 297: Role% Comparison")
print("-" * 70)
p296_dir = Path("results/phase296_residual_decomposition")
for m in models:
    p296_path = p296_dir / f"{m}_residual_decomposition.json"
    if not p296_path.exists() or m not in data:
        continue
    with open(p296_path, "r", encoding="utf-8") as f:
        d296 = json.load(f)
    nl = data[m]["n_layers"]
    mid = nl // 2
    
    vd296 = d296.get("variance_decomposition_raw", {})
    r296_mid = vd296.get(str(mid), {})
    role_296 = r296_mid.get("role_ratio", 0) * 100
    inter_296 = r296_mid.get("interaction_ratio", 0) * 100
    
    cra = data[m].get("cross_role_anova_raw", {})
    r297_mid = cra.get(str(mid), {})
    role_297 = r297_mid.get("role_ratio", 0) * 100
    inter_297 = r297_mid.get("interaction_ratio", 0) * 100
    
    print(f"  {m:12s} L{mid}: Phase296 Role={role_296:.1f}% Interact={inter_296:.1f}% "
          f"| Phase297 Role={role_297:.1f}% Interact={inter_297:.1f}%")

# ===== 6. DS7B special analysis =====
print("\n[F] DS7B Special: Near-Zero Cross-Pair Role Consistency")
print("-" * 70)
m = "deepseek7b"
if m in data:
    rid = data[m].get("role_increment_decomposition", {})
    cc_data = rid.get("cross_pair_cos", {})
    sr_data = rid.get("shared_role_ratio", {})
    print(f"  Layer | SharedRatio | CrossPairCos | Interpretation")
    for li_str in sorted(cc_data.keys(), key=int):
        li = int(li_str)
        cc = list(cc_data[li_str].values())
        sr = list(sr_data.get(li_str, {}).values())
        if cc and sr:
            avg_cc = np.mean(cc)
            avg_sr = np.mean(sr)
            if avg_cc < 0.1:
                interp = "NO consistent role direction"
            elif avg_cc < 0.3:
                interp = "Weak role direction"
            else:
                interp = "Moderate role direction"
            print(f"  L{li:4d} | {avg_sr:.4f}       | {avg_cc:+.4f}       | {interp}")

print("\n" + "=" * 70)
print("Phase 297 Cross-Model Comparison Complete!")
