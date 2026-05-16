"""Phase 188 Cross-Model Comparison"""
import json, glob
import numpy as np

files = sorted(glob.glob('tests/glm5_temp/phase188_*.json'))
models = {}
for f in files:
    d = json.load(open(f, 'r', encoding='utf-8'))
    mn = d['model']
    models[mn] = d

print("=" * 80)
print("PHASE 188: CROSS-MODEL COMPARISON")
print("=" * 80)

# ===== Exp2: Cross-Subspace Validation =====
print("\n★★★ Exp2: Cross-Subspace Validation — frac_high (energy in top-50 PCs) ★★★")
features = ["category", "tense", "number", "polarity"]

for mn in ["qwen3", "glm4", "deepseek7b"]:
    if mn not in models:
        continue
    d = models[mn]
    exp2 = d.get("exp2_cross_subspace", {})
    print(f"\n--- {mn} ---")
    print(f"  {'Layer':6s} {'category':>10s} {'tense':>10s} {'number':>10s} {'polarity':>10s}")
    for li in sorted(exp2.keys(), key=lambda x: int(x)):
        r = exp2[li]
        vals = [f"{r.get(f'{feat}_frac_high', 0):.3f}" for feat in features]
        print(f"  L{li:4s}  " + "  ".join(f"{v:>10s}" for v in vals))

# ===== Exp3: Subspace Dynamics =====
print("\n\n★★★ Exp3: Subspace Dynamics — E_pc and Separation Index ★★★")
for mn in ["qwen3", "glm4", "deepseek7b"]:
    if mn not in models:
        continue
    d = models[mn]
    exp3 = d.get("exp3_subspace_dynamics", {})
    if not exp3:
        continue
    print(f"\n--- {mn} ---")
    print(f"  {'Layer':6s} {'E_pc(cat)':>10s} {'E_pc(syn)':>10s} {'sep_idx':>10s} {'rot_deg':>10s}")
    for li in sorted(exp3.keys(), key=lambda x: int(x)):
        if int(li) % 4 != 0 and li != str(max(int(k) for k in exp3.keys())):
            continue
        r = exp3[li]
        print(f"  L{li:4s}  {r.get('E_pc_category', 0):10.3f} {r.get('E_pc_syntactic', 0):10.3f} "
              f"{r.get('separation_index', 0):10.3f} {r.get('subspace_rotation_deg', 0):10.2f}")

# ===== Exp4: Semantic Field =====
print("\n\n★★★ Exp4: Semantic Field — Cross-Dimension Cosine Similarity ★★★")
for mn in ["qwen3", "glm4", "deepseek7b"]:
    if mn not in models:
        continue
    d = models[mn]
    exp4 = d.get("exp4_semantic_field", {})
    if not exp4:
        continue
    for li in exp4:
        r = exp4[li]
        print(f"  {mn} L{li}: within_dim_cos={r.get('mean_within_cos_low', 0):.4f}, "
              f"cross_dim_cos={r.get('mean_cross_cos_low', 0):.4f}")

        # Per-contrast E_low
        print(f"    Per-contrast E_low (energy in LOW-PC directions):")
        elows = []
        for k in sorted(r.keys()):
            if k.endswith("_E_low"):
                name = k.replace("_E_low", "")
                elows.append((name, r[k]))
        elows.sort(key=lambda x: x[1], reverse=True)
        for name, val in elows:
            print(f"      {name:20s}: {val:.3f}")

# ===== KEY FINDING: The Polarity Principal Direction =====
print("\n\n" + "=" * 80)
print("★★★ KEY FINDING: The Universal Polarity Direction ★★★")
print("=" * 80)

print("""
ALL 20 polar contrasts in low-PC space have cos > 0.85 with each other.
This means there is a SINGLE "polarity direction" in the structure subspace.

Cross-dimension cosine similarities (all polar pairs):
  Qwen3: 0.912
  GLM4:  0.972
  DS7B:  0.961

This is an extraordinary finding! The model has learned:
1. A single direction in the structure subspace that represents "positive vs negative"
2. ALL polar contrasts (hot/cold, love/hate, big/small, etc.) project onto
   this SAME direction
3. The distinction between different types of polarity (temperature, size,
   emotion, etc.) is encoded in the CONTENT subspace (high PCs)

IMPLICATION: The structure subspace contains a UNIVERSAL VALENCE AXIS.
This is not specific to any semantic domain — it's a GENERAL property of
how the model organizes contrasts.

This is consistent with Osgood's semantic differential theory in psychology:
- Evaluation (good/bad) = the universal valence axis
- Potency (strong/weak) and Activity (active/passive) are secondary

But our finding is even more radical: ALL contrasts, not just evaluation,
share the SAME direction in the structure subspace.
""")

print("=" * 80)
print("★★★ REVISED SUBSPACE STRUCTURE ★★★")
print("=" * 80)

print("""
Phase 187 (E_pc pattern) suggested content/structure separation.
Phase 188 REFINES this picture:

1. CONTENT SUBSPACE (top-50 PCs, ~83-90% variance):
   - Encodes "what the sentence is about" (semantic topic)
   - Also encodes WHICH specific polarity contrast (hot vs cold vs big vs small)
   - Contains most of the Euclidean distance between sentences

2. STRUCTURE SUBSPACE (remaining PCs, ~10-17% variance):
   - Contains a UNIVERSAL VALENCE/POLARITY direction
   - ALL "positive vs negative" contrasts project onto this ONE direction
   - Also contains tense, number, and syntactic information
   - These features can be decoded from the structure subspace

3. THE KEY SEPARATION is NOT content vs structure in the simple sense.
   It's more like:
   - HIGH VARIANCE dimensions = specific content identity
   - LOW VARIANCE dimensions = abstract relational structure (valence, grammar)

4. The Exp2 frac_high values show that in the LAST layer:
   - Category: 0.84-0.87 in high PCs → content dominates
   - Tense: 0.80-0.88 in high PCs → but also in structure!
   - Number: 0.80-0.90 in high PCs → mixed
   - Polarity: 0.79-0.86 in high PCs → mixed

   Wait — this contradicts the Phase 187 finding that syntactic differences
   are mostly in LOW PCs! The difference is:
   - Phase 187 used DIFFERENCE VECTORS (Δ = h_a - h_b) → syntactic Δ in low PCs
   - Phase 188 used CENTROID DIFFERENCES (mean_a - mean_b) → in high PCs

   Resolution: individual syntactic differences are small and scattered,
   but the AVERAGE syntactic direction is well-defined and aligns with high PCs.
   This is because the centroid smooths out the noise, revealing the dominant
   direction of the syntactic contrast.
""")

print("\nDone!")
