"""Append Phase 198 results to AGI_GLM5_MEMO.md"""
import os

memo_path = r"research\glm5\docs\AGI_GLM5_MEMO.md"

content = r"""
## Phase 198: Mode Transition & Random Perturbation Control [2026-05-16 22:30]

### Core Question: Is KL increase real semantic propagation or autoregressive tautology?

### Three-Model Results: Sem/Rand KL Slope Ratio

| Metric | Qwen3 (25sent x 12step x 40sample) | GLM4 (5sent x 6step x 10sample) | DS7B (5sent x 6step x 10sample) |
|--------|--------------------------------------|----------------------------------|----------------------------------|
| Sem/Rand ratio | **1.00x** | **0.92x** | **1.30x** |
| negation slope | 0.946 | 1.385 | 1.851 |
| rand_token slope | 0.927 | 1.410 | 0.798 |
| question slope | 0.339 | 1.210 | 0.223 |
| question KL[0] | 10.60 | 5.21 | 12.49 |

### CRITICAL FINDING: KL slope increase is autoregressive tautology

The ratio of semantic-to-random KL slope is ~1.0x across all three models.
This means: KL increasing over autoregressive steps is NOT specific to semantic constraints.
It is a basic property of any autoregressive system: different initial conditions naturally diverge.

**Negation KL profile (1.1 -> 11.3) is nearly identical to random token (1.1 -> 11.3)!**

### What IS real: The delay spectrum

Despite the slope tautology, there are genuine semantic signals in the INITIAL divergence and temporal structure:

| Constraint | KL[0] | Type | Models confirming |
|------------|-------|------|-------------------|
| question | 5.3-12.5 | IMMEDIATE | Qwen3+GLM4+DS7B (3/3) |
| negation | 1.1-1.8 | MODERATE | Qwen3+GLM4+DS7B (3/3) |
| conditional | 0.37-0.44 | DELAYED | Qwen3+GLM4 (2/3) |
| rand_token | 1.1-2.1 | MODERATE | Qwen3+GLM4+DS7B (3/3) |
| rand_period | 9.0 | IMMEDIATE | Qwen3 only |

Key distinction: **question has KL[0]=10.6 but rand_period also has KL[0]=9.0**.
The high initial KL for question is partly due to syntactic restructuring (punctuation, word order),
similar to how adding a period fundamentally changes the sentence.

### What distinguishes semantic from random perturbation

NOT the KL slope (that's tautology), but:

1. **Attractor basin structure**: 
   - question: 23 basins, top1=0.242, entropy=2.58 (opens answer space)
   - rand_token: 23 basins, top1=0.300, entropy=2.16 (just noise)
   - rand_period: 13 basins, top1=0.833, entropy=0.81 (collapses to period continuation)
   - Basin count is similar but the TOKENS are completely different!

2. **Conditional delayed effect**: 
   - Conditional KL[0]=0.44, slope=1.10 (Qwen3) — genuinely delayed!
   - Rand_token KL[0]=1.10, slope=0.93 — immediate divergence
   - The DELAY is real: conditional constraints don't act immediately

3. **Mode transition effects** (Qwen3):
   - CoT trigger: KL slope=0.09 (minimal trajectory change), but entropy shift=-1.27
   - Translation: KL slope=1.06 (strong divergence), entropy shift=+0.39
   - Contrast ("but"): KL slope=0.70, entropy shift=+1.62 (opens alternative path!)
   - Coding: KL slope=0.52, entropy shift=-0.23 (compresses to code structure)

### Theory Upgrade: Three-Layer Structure

Based on the evidence:

**Level 1: Mode (Global Computational Mode)**
- Determines what PROGRAM the network runs
- CoT, translation, QA, narrative, coding — these switch the entire generation dynamics
- NOT captured by KL slope (which is tautology)
- Captured by: entropy dynamics, attractor basin token types, trajectory topology

**Level 2: Constraint (Probability Field Modulation)**
- Determines which continuations are permissible
- Negation, question, conditional, role binding
- Captured by: KL[0] (initial divergence), delay spectrum, basin structure

**Level 3: Autoregressive Chaos (Tautological)**
- Any perturbation causes KL to increase over steps
- NOT semantic — it's just chaotic amplification
- Must be subtracted out to see real signal

### The Right Metric is NOT KL slope, but KL[0] x Basin Structure

The real semantic signal is in:
1. **How much the FIRST step diverges** (KL[0])
2. **What the divergence looks like** (which tokens become more/less likely)
3. **Whether the divergence is delayed** (conditional shows genuine temporal structure)

### Hard Issues

1. **question KL[0]=10.6 but rand_period KL[0]=9.0** — Are they doing the same thing? 
   Answer: No. Question opens answer basins (What/Yes/No), period collapses to structural tokens.
   The KL magnitude is similar but the STRUCTURE is completely different.

2. **Conditional "delayed" might just be "weak initial"** — if conditional KL[0] is small simply because "If" doesn't change the next token much, the "delay" is just the natural autoregressive amplification, not a genuine temporal deferral mechanism.
   Counter: Conditional slope (1.10) is HIGHER than rand_token slope (0.93), meaning conditional diverges FASTER than random. This suggests genuine constraint propagation.

3. **Mode transitions**: CoT has very low KL slope (0.09) — it barely changes the trajectory!
   But it causes large entropy shifts (-1.27). This means CoT doesn't change WHERE you go but HOW you go there (more structured, lower entropy). This is a genuine mode effect.

### Next Steps: The Real Frontier

The critical insight from this experiment: **KL slope is tautology, but attractor topology is real.**

We need to develop:
1. **Attractor topology metrics** — not just "how many basins" but the SHAPE of the basin landscape
2. **Token-structured KL** — decompose KL into which TOKEN GROUPS diverge (content vs structural vs answer)
3. **Mode detection from trajectory** — can we infer the current mode from the generation dynamics alone?
4. **Constraint delay spectrum refinement** — conditional vs negation delay difference is real and needs deeper analysis

Scripts:
- tests/glm5/phase198_mode_transition.py (Qwen3 full)
- tests/glm5/phase198b_lite_mode.py (GLM4/DS7B lite)

Logs:
- tests/glm5_temp/phase198_qwen3_log.txt (Qwen3)
- tests/glm5_temp/phase198b_glm4_log.txt (GLM4)
- tests/glm5_temp/phase198b_ds7b_log.txt (DS7B)
"""

with open(memo_path, 'a', encoding='utf-8') as f:
    f.write(content)
print("MEMO updated successfully")
