"""Append Phase 197 results to AGI_GLM5_MEMO.md"""
import os

memo_path = r"research\glm5\docs\AGI_GLM5_MEMO.md"

content = """
## Phase 197: Trajectory-Level Constraint Dynamics -- Cross-Model Verification [2026-05-16 21:00]

### Core Shift: From Single-Step KL to Multi-Step Trajectory Analysis

User's key critiques are correct: KL is not a semantic measure, output constraints != internal computation, KL additivity may be mathematical artifact.
Correct direction: study P(x_{t:t+k}) not P(x_{t+1}), i.e., trajectory-level constraint dynamics.

### Experiment Design

4 experiments:
1. Multi-step Distribution Evolution: autoregressive 8-12 steps, entropy/branching factor per step
2. Trajectory Divergence Rate: KL/JS/overlap at each step (base vs constraint)
3. Attractor Basin Structure: sample continuations, analyze first-token basins
4. Conditional Delayed Effect: does conditional KL increase over steps?

### Three-Model Core Results Comparison

| Metric | Qwen3 (25 sent x 12 steps x 40 samples) | GLM4 (8 sent x 8 steps x 15 samples) | DS7B (8 sent x 8 steps) |
|--------|------------------------------------------|---------------------------------------|--------------------------|
| KL slope (negation) | 1.07 INCREASING | 1.00 INCREASING | 0.47 INCREASING |
| KL slope (question) | 0.56 INCREASING | 0.54 INCREASING | 0.54 INCREASING |
| KL slope (role_binding) | 0.84 INCREASING | 0.78 INCREASING | 0.70 INCREASING |
| Conditional KL slope | 0.76 INCREASING | 1.64 INCREASING | 1.64 INCREASING |

### Key Findings

#### 1. All constraint KLs increase monotonically over steps (3-model consensus, HIGH confidence)

Negation: KL from ~1.5 to ~17 (Qwen3), ~8.8 (GLM4)
Question: KL from ~8-16 sustained high (Qwen3), ~5.5->10.7 (GLM4)
Role_binding: KL from ~3-6 to ~16 (Qwen3), ~9.1 (GLM4)

**Interpretation**: Constraints are not instantaneous effects but **accumulate and amplify** during autoregressive generation. Each generated token increases context divergence, making subsequent distributions more different. This confirms the user's core insight: semantics should be understood at the **trajectory level**, not single-step.

#### 2. Question opens more attractor basins (Qwen3 confirmed, GLM4 partially confirmed)

**Qwen3**:
- base: 4.3 basins, concentration=0.393, entropy=2.54
- question: 5.1 basins, concentration=0.260, entropy=3.24
- neg+rb: 2.9 basins, concentration=0.490, entropy=2.14 (most convergent!)

**GLM4**:
- base: 35 basins, concentration=0.150, entropy=2.98
- question: 13 basins, concentration=0.508, entropy=1.50
- question first tokens: Yes(33), Solution(6), It(4)

**Analysis**: GLM4's question mode is highly concentrated (top1=0.508), first tokens are Yes/No, indicating question compresses generation space to "answer space". Qwen3 shows question opens more branches (5.1 vs 4.3), indicating question opens multiple answer basins. Both phenomena are not contradictory: question changes **discourse mode**, not simply increases/decreases entropy.

#### 3. Conditional delayed constraint effect (3-model consensus!)

**Qwen3**: conditional KL slope=0.76, significantly INCREASING
**GLM4**: conditional KL slope=1.64, strongest increase!
**DS7B**: KL from 0.41->9.53, slope=1.64

**Core insight**: "If" clause has small KL at step 0 (next token not yet constrained by condition), but as generation proceeds, conditional constraints start truly affecting subsequent tokens, KL increases sharply. This is exactly the **delayed constraint** the user described!

#### 4. Negation is the strongest "probability mass suppressor"

**Qwen3**: neg+rb condition (2.9 basins, 0.490 concentration) is most convergent among all conditions.
**First tokens**: comma(169), 's(155), period(107) -- almost all structural tokens.

This shows negation is not "adding a negative vector" but **suppressing all contentful continuations**, forcing the model to only generate structural filler tokens.

### Theoretical Analysis

#### Correctness of User's Critiques

1. KL is not a semantic measure -- but KL's **dynamics** (INCREASING) indeed reflects constraint accumulation on trajectories.
2. Should not equate output constraints with internal computation -- correct. But comparing KL profiles across constraints lets us infer internal **propagation dynamics**.
3. KL additivity may be mathematical artifact -- not yet verified at trajectory level.
4. Autoregressive coupling is key -- **this experiment's core finding IS autoregressive coupling!** KL increasing over steps is because autoregressive coupling amplifies early constraints.

#### New Core Understanding

Semantic operations' essence:
- **Negation** = probability mass suppression + structural token redirection
- **Question** = discourse mode switching + answer space opening
- **Role binding** = dependency topology restructuring + semantic role reassignment
- **Conditional** = delayed constraint + trajectory-level accumulation

These are not vector operations, but **dynamic reachability control over future generation trajectories**.

#### Hard Issues and Problems

1. **GLM4 "question highly concentrated" vs Qwen3 "question opens more basins"** -- needs deeper analysis. May be architectural difference or sampling/data difference.

2. **KL increase may be tautology** -- two autoregressive processes from different initial conditions naturally produce increasingly different distributions. This may not reflect "semantic constraint propagation" but basic chaotic system property. CRITICAL CONTROL NEEDED: does random token perturbation also cause KL increase?

3. **Attractor basin token-level analysis too coarse** -- only looking at first token ignores subsequent token coherence.

4. **Conditional delayed effect may be tokenization artifact** -- "If" token changes syntactic structure, subsequent KL increase may just be because two parse trees diverge.

### Next Steps

#### Critical Verification: Random Perturbation Control Experiment

**Most critical next step**: If random token perturbation also causes KL increase, then KL increase is tautology not discovery. Need to design:
- Add a random token (e.g., "the", "and", "xyz") to base sentence end, check if KL also increases
- If random token KL also increases -> increase is basic chaotic system property
- If semantic constraint KL increase rate significantly higher than random perturbation -> increase indeed reflects constraint propagation

#### Trajectory Semantics vs Token Semantics

Need to upgrade from "first token analysis" to "full generation trajectory analysis":
- Sample 20-token continuations, compute trajectory distances using semantic embeddings
- Compare "trajectory space" topology under different constraints

#### Constraint Propagation Layer Analysis

How do hidden state changes at different layers predict the final KL profile? This bridges "internal computation" and "output constraints".

Scripts:
- tests/glm5/phase197_trajectory_dynamics.py (Qwen3 full version)
- tests/glm5/phase197b_lite_verify.py (GLM4/DS7B lite version)

Logs:
- tests/glm5_temp/phase197_qwen3_log.txt (57KB, Qwen3 complete)
- tests/glm5_temp/phase197b_glm4_v2_log.txt (GLM4 lite)
- tests/glm5_temp/phase197b_ds7b_log.txt (DS7B partial)
"""

with open(memo_path, 'a', encoding='utf-8') as f:
    f.write(content)
print("MEMO updated successfully")
