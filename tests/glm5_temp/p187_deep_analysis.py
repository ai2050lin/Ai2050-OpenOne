"""Phase 187 Deep Analysis: Understanding the Results"""
import json, glob
import numpy as np

files = sorted(glob.glob('tests/glm5_temp/phase187_*.json'))
models = {}
for f in files:
    d = json.load(open(f, 'r', encoding='utf-8'))
    mn = d['model']
    models[mn] = d

print("=" * 80)
print("PHASE 187: DEEP ANALYSIS")
print("=" * 80)

# ===== 1. The E_pc Pattern: Content vs Structure Subspace =====
print("\n" + "=" * 80)
print("★★★ ANALYSIS 1: Energy in Principal Subspace — The REAL Signal ★★★")
print("=" * 80)

print("""
KEY OBSERVATION: The E_pc metric reveals a UNIVERSAL pattern across all 3 models:

| Type       | E_pc (Qwen3) | E_pc (GLM4) | E_pc (DS7B) | Pattern     |
|------------|---------------|-------------|--------------|-------------|
| category   | 0.816         | 0.834       | 0.847        | HIGH        |
| random     | 0.911         | 0.925       | 0.962        | VERY HIGH   |
| subordinate| 0.300         | 0.253       | 0.259        | LOW         |
| syntactic  | 0.251         | 0.228       | 0.156        | VERY LOW    |
| paraphrase | 0.310         | 0.262       | 0.360        | LOW-MEDIUM  |

INTERPRETATION:
- Category & Random: differences are ALONG the principal subspace → CONTENT differences
- Subordinate & Syntactic: differences are ORTHOGONAL to principal subspace → STRUCTURE differences
- The principal subspace (top-50 PCs) captures ~85-90% of variance → this is the "content axis"
- The orthogonal subspace captures ~10-15% of variance → this is the "structure axis"

THIS IS THE KEY FINDING: The model's representation space has TWO distinct subspaces:
1. CONTENT subspace (high PCs): captures "what the sentence is about"
2. STRUCTURE subspace (low PCs): captures "how the sentence expresses it"

The "isometric compression" finding was a statistical artifact because:
- Global Euclidean distance ||Δ|| mixes both subspaces
- Content differences dominate the norm (80-90% of energy)
- Structure differences are buried in the noise floor (10-20% of energy)
""")

# ===== 2. DS7B Anisotropy =====
print("\n" + "=" * 80)
print("★★★ ANALYSIS 2: DS7B's Unique Anisotropy — Why? ★★★")
print("=" * 80)

# Check DS7B's PCA variance concentration
d = models['deepseek7b']
exp1 = d['exp1_diff_amplification_spectrum']
sample_layers = exp1['sample_layers']
print("\nDS7B PCA cumulative variance at each layer:")
for li in sample_layers:
    li_str = str(li)
    if li_str in exp1.get('pca_cumvar', {}):
        top10 = exp1['pca_cumvar'][li_str].get('top10', 0)
        top50 = exp1['pca_cumvar'][li_str].get('top50', 0)
        print(f"  L{li}: top10={top10:.3f}, top50={top50:.3f}")

print("""
DS7B's PCA shows ANOMALOUS variance concentration:
- L5-L20: top10 PCs capture 98.2-98.8% of variance!
- This is MUCH higher than Qwen3 (37-43%) or GLM4 (38-43%)
- This means DS7B's representation is extremely low-rank in middle layers

WHY does DS7B show anisotropy?
- Random differences align almost entirely with the dominant PCs (E_pc=0.962)
- Category differences have some component in orthogonal directions (E_pc=0.847)
- When the representation is extremely low-rank, the amplification is dominated
  by the principal subspace → random differences get amplified MORE
- This is NOT "semantic anisotropy" but "rank anisotropy"

CONCLUSION: DS7B's anisotropy is an artifact of its extreme low-rank
representation, not evidence for difference renormalization.
""")

# ===== 3. The Norm Scaling Pattern =====
print("\n" + "=" * 80)
print("★★★ ANALYSIS 3: Norm Scaling — Range Normalization ★★★")
print("=" * 80)

print("""
All three models show the SAME norm scaling pattern:

Qwen3: ||Δ_L|| / √||Δ_0|| ≈ 271-322 (roughly constant)
GLM4:  ||Δ_L|| / √||Δ_0|| ≈ 258-466 (roughly constant)
DS7B:  ||Δ_L|| / √||Δ_0|| ≈ 320-1047 (less constant, due to anisotropy)

The scaling is approximately proportional to √||Δ_0||, not linear.
This means: ||Δ_final|| ≈ C × √||Δ_initial||

INTERPRETATION:
- The model has a "target dynamic range" for differences
- Large initial differences are compressed (amplification < proportional)
- Small initial differences are expanded (amplification > proportional)
- The net effect: all differences end up at a similar absolute scale

This is "range normalization" — not isometric compression, not anisotropy.
The model maintains a finite dynamic range, pulling all differences toward
a common absolute magnitude.
""")

# ===== 4. Cross-Lingual Alignment =====
print("\n" + "=" * 80)
print("★★★ ANALYSIS 4: Cross-Lingual Direction Alignment ★★★")
print("=" * 80)

print("""
Cross-lingual direction alignment cos(Δ_en, Δ_zh):

| Model  | L0     | L_last | Slope    | Pattern      |
|--------|--------|--------|----------|-------------|
| Qwen3  | 0.2145 | 0.3401 | +0.00359 | CONVERGING   |
| GLM4   | 0.0370 | 0.3199 | +0.00725 | CONVERGING   |
| DS7B   | 0.0327 | 0.2324 | +0.00740 | CONVERGING   |

All three models show CONVERGING alignment, but the absolute values are LOW (0.23-0.34).
This is PARTIAL alignment, not full alignment.

Per-contrast analysis shows a CLEAR pattern:
- HIGH alignment: hot_vs_cold (0.43-0.57), apple_vs_banana (0.55-0.63), love_vs_hate (0.20-0.53)
- LOW alignment: tall_vs_short (0.01-0.06), can_vs_cannot (0.07-0.17), cat_vs_dog (0.12-0.42)

The contrast pairs with high alignment share a key property:
→ They are POLAR OPPOSITES with strong semantic fields
→ The contrast direction is well-defined in the embedding space

The pairs with low alignment have:
→ They involve RELATIVE or CONTEXT-DEPENDENT contrasts
→ "tall", "short", "big", "small" are relative adjectives, not absolute
→ "can/cannot" involves modality, which is encoded differently across languages
""")

# ===== 5. The Jacobian Isotropy Paradox =====
print("\n" + "=" * 80)
print("★★★ ANALYSIS 5: The Jacobian Isotropy Paradox ★★★")
print("=" * 80)

print("""
ALL three models show ISOTROPIC Jacobian (p > 0.9):
- Qwen3: g_semantic=57.247, g_random=57.296, p=0.9948
- GLM4:  g_semantic=73.682, g_random=73.565, p=0.9017
- DS7B:  g_semantic=77.672, g_random=76.738, p=0.9430

This seems to CONTRADICT the "difference renormalization" hypothesis.

BUT: The Jacobian test measures LOCAL LINEAR amplification at a point.
The "difference renormalization" may be a GLOBAL NONLINEAR effect.

THREE possible explanations:
1. The difference renormalization is a second-order (curvature) effect,
   not a first-order (Jacobian) effect
2. The anisotropy exists in the nonlinear dynamics (attention pattern changes),
   not in the linear amplification
3. The "difference renormalization" hypothesis is WRONG — the model really
   does treat all directions equally, and the apparent pattern in cumul_amp
   is just due to initial difference magnitude

OCCAM'S RAZOR: Explanation 3 is the simplest. The data supports:
- Isometric compression at the linear (Jacobian) level
- Range normalization at the nonlinear (cumulative) level
- Content vs Structure subspace separation (the E_pc pattern)
""")

# ===== 6. Revised Theoretical Framework =====
print("\n" + "=" * 80)
print("★★★ ANALYSIS 6: Revised Theoretical Framework ★★★")
print("=" * 80)

print("""
BEFORE Phase 187:
  Hypothesis: "Difference Renormalization"
  - Some directions amplified, some compressed
  - Anisotropic encoding

AFTER Phase 187:
  REVISED: "Subspace-Structured Range Normalization"
  1. The representation space has TWO subspaces:
     a. CONTENT subspace (top PCs): captures semantic content
     b. STRUCTURE subspace (low PCs): captures grammatical/structural info
  2. The linear dynamics (Jacobian) are ISOTROPIC — all directions amplified equally
  3. The nonlinear dynamics produce RANGE NORMALIZATION:
     - Large initial differences → less relative amplification
     - Small initial differences → more relative amplification
     - Net: all differences converge to a similar absolute scale
  4. The apparent "anisotropy" in cumul_amp is due to:
     - Different difference types having different initial magnitudes
     - Range normalization compressing/expanding to a common scale
  5. The E_pc pattern (content vs structure) is the REAL discovery:
     - This is the subspace separation that enables language encoding
     - Content differences are in high-variance directions
     - Structure differences are in low-variance directions

KEY INSIGHT FOR AGI:
The model doesn't "choose" which differences to amplify.
Instead, it organizes the representation space so that:
- Content (high-variance) and Structure (low-variance) are SEPARATED
- This separation is what enables downstream tasks (classification, generation)
- The "encoding" is the SEPARATION, not the AMPLIFICATION
""")

print("\n" + "=" * 80)
print("★★★ NEXT STEPS ★★★")
print("=" * 80)

print("""
Phase 188 should investigate:
1. **Subspace Structure Validation**: Is the content/structure separation
   universal across more sentence types, longer texts, and more models?
2. **The Structure Subspace**: What information is encoded in the low-PC
   directions? Can we decode it? Is it grammatical, logical, relational?
3. **The Range Normalization Mechanism**: How does the model achieve
   range normalization? Is it the residual stream norm growth?
   Or is it an active process in attention/MLP?
4. **Semantic Field Theory**: The cross-lingual alignment suggests that
   "polar opposite" contrasts (hot/cold, love/hate) have well-defined
   direction vectors. This hints at a "semantic field" structure.
5. **Dynamic Subspace Tracking**: Track how content/structure subspaces
   evolve across layers — when does the separation emerge? Is it gradual
   or sudden? Does it correlate with attention head specialization?
""")

print("\nDone!")
