# -*- coding: utf-8 -*-
"""p3087 patch scan: generate
phase3087_omega_p84_glm4_layer_scan.py from
phase3084_omega_p81_3b_layer_scan.py.
Every replacement is count-asserted."""
import io

SRC = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
       r'\phase3084_omega_p81_3b_layer_scan.py')
DST = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
       r'\phase3087_omega_p84_glm4_layer_scan.py')

src = io.open(SRC, encoding='utf-8').read()

# ---- 1. docstring wholesale ----
parts = src.split('"""')
assert len(parts) >= 3, len(parts)
DOC = '''Phase 3087 A1: Omega-P84 GLM4-9B layer scan
=========================================================
Menu A of 3086 (continuum confirmed): the fourth
spectral point is GLM4-9B (Zhipu, non-Qwen arch:
GlmForCausalLM, 40L / 4096H / 32 heads / 2 kv,
vocab 151552, head_dim 128, untied), the only
non-Qwen-family checkpoint on disk.  Before the
full four-way arbitration at one layer (A2), the
3083/3084 lesson (family-C degeneration that
turned out to be a LAYER-POSITION effect,
rescued at L28/L34) requires a layer-position
scan on GLM4 first.

Question: over 4 candidate injection layers,
where (if anywhere) is the GLM4 focus machinery
healthy on all three frozen families?

Design (3084-identical pipeline, GLM4-tuned):
- bf16 eager full load on the 16GB card (18.84GB
  allocated, driver sysmem fallback for ~1.75GB,
  measured viable in p3087_bf16_timing: no OOM
  at seq<=74, 0.16-0.59s/forward).
- L_INJ in (31, 34, 37, 38) - depth 0.78 / 0.85
  / 0.93 / 0.95 of 40 layers (proportional
  mapping of the 3B candidates 28/31/33/34 of
  36); L_POST = L_INJ + 1.
- E1 repV ladder (24 pairs, med_c reference,
  b4/b8 per layer), 32-head single swap scan
  (24 pairs) -> r1 = median cos - med_c;
  n_neg from the 3071 criterion; focal top8;
  R_ALL full 32-head swap.
- b7a identity attn self-swap anchor moved to
  the neutral layer 36 (GLM4 has no 3083-style
  degenerate layer; any healthy layer verifies
  the swap mechanism bit-free).

Verdict (preregistered, 3084-identical):
- setup/anchor fail -> setup_failed
- some L in CAND with n_neg >= 8 on ALL three
  families -> layer_rescue (best = argmax_L
  min-family n_neg; ties -> lower L, matching
  the 3084 min-family tie-break)
- else family C n_neg >= 8 at some L ->
  layer_partial
- else -> layer_absent

Output: tests/glm5/result/
rdc_query_construction_20260913/phase3087/
omega_p84_glm4_layer_scan/
'''
src = '"""' + parts[0][3:] if False else src
pre, body, rest = src.split('"""', 2)
src = '"""' + DOC + '"""' + rest

# ---- 2. exact replacements ----
REPS = [
    ('PHASE = 3084', 'PHASE = 3087', 1),
    ("NAME = 'omega_p81_3b_layer_scan'",
     "NAME = 'omega_p84_glm4_layer_scan'", 1),
    ("'qwen2.5-3b-instruct')",
     "'glm4-9b-chat-hf')", 1),
    ('NL, HID, KV_HEAD, NQ = 36, 2048, 2, 16',
     'NL, HID, KV_HEAD, NQ = 40, 4096, 2, 32',
     1),
    ('SEED_MAIN = 3084', 'SEED_MAIN = 3087', 1),
    ('SEED = 3084', 'SEED = 3087', 1),
    ('L_CAND = (28, 31, 33, 34)',
     'L_CAND = (31, 34, 37, 38)', 1),
    ('L_CAND = (34,)', 'L_CAND = (38,)', 1),
    ('attn_swaps=[(33, ALLH,',
     'attn_swaps=[(36, ALLH,', 1),
    ('BH[base_i, 33])])[0]',
     'BH[base_i, 36])])[0]', 1),
    ('# b7a identity attn self-swap at '
     'L_INJ=33',
     '# b7a identity attn self-swap at '
     'neutral layer 36', 1),
    ('assert NVOC == 151936',
     'assert NVOC == 151552', 1),
    ("log('qwen2.5-3b (qwen2.5-3b-instruct) "
     "loaded '",
     "log('glm4-9b (glm4-9b-chat-hf) "
     "loaded '", 1),
    ("'phase3084', NAME)", "'phase3087', NAME)",
     1),
    ("'mode': 'single model qwen2.5-3b-"
     "instruct '",
     "'mode': 'single model glm4-9b-chat-hf '",
     1),
    ('bf16 eager seed 3084;',
     'bf16 eager seed 3087;', 1),
    ('layer L_INJ in (28, 31, 33, 34) ',
     'layer L_INJ in (31, 34, 37, 38) ', 1),
    ("'16-head single swap scan (24 '",
     "'32-head single swap scan (24 '", 1),
    ("'criterion, R_ALL full 16-head '",
     "'criterion, R_ALL full 32-head '", 1),
    ("log('[%s] L%d: full 16-head swap '",
     "log('[%s] L%d: full 32-head swap '", 1),
    ("'L_INJ_3083': np.int64(33),",
     "'B7A_LAYER': np.int64(36),", 1),
]
for old, new, want in REPS:
    got = src.count(old)
    assert got == want, (old, got, want)
    src = src.replace(old, new)

# ---- 3. multi-line PREREG blocks ----
Q_OLD = """    'question': '3084 A (menu of 3083): is the '
                '3083 family-C degeneration '
                '(n_neg=2/16 at L_INJ=33) a '
                'LAYER-POSITION effect?  Scan '
                'the focus machinery over 4 '
                'candidate layers and record '
                'per-(layer, family) n_neg / r1 '
                'magnitudes / capture8.',"""
Q_NEW = """    'question': '3087 A (menu of 3086): does '
                'the fourth spectral point '
                'GLM4-9B (non-Qwen arch, 40L/'
                '4096H/32H/2kv) need a layer '
                'rescue scan before its full '
                'arbitration, and which layer '
                'is its focus-optimal L_INJ?  '
                'Scan the focus machinery over '
                '4 candidate layers and record '
                'per-(layer, family) n_neg / r1 '
                'magnitudes / capture8.',"""
I_OLD = """    'independence': 'the verdict tree '
                    '(layer_rescue / '
                    'layer_partial / '
                    'layer_absent with n_neg>=8 '
                    'thresholds) was frozen '
                    'before any layer-scan '
                    'observation; all qwen2.5-3b '
                    'activations from 3083 are '
                    'prior data but only at '
                    'L_INJ=33 (recollected here '
                    'under seed 3084)',"""
I_NEW = """    'independence': 'the verdict tree '
                    '(layer_rescue / '
                    'layer_partial / '
                    'layer_absent with n_neg>=8 '
                    'thresholds) was frozen '
                    'before any GLM4 observation; '
                    '3B layer-scan results (3084) '
                    'are prior data but concern '
                    'qwen2.5-3b only; GLM4 has no '
                    'prior injections',"""
LC_OLD = """    'layer_candidates': 'L_INJ in (28, 31, 33, '
                        '34): depth 0.78/0.86/'
                        '0.92/0.94 of 36 layers; '
                        '33 is the 3083 position; '
                        '34 matches qwen3-4b (2nd '
                        'from last), 28/31 probe '
                        'the mid-deep band; '
                        'L_POST = L_INJ + 1',"""
LC_NEW = """    'layer_candidates': 'L_INJ in (31, 34, 37, '
                        '38): depth 0.78/0.85/'
                        '0.93/0.95 of 40 layers '
                        '(proportional mapping of '
                        'the 3B candidates 28/31/'
                        '33/34 of 36); 37/38 are '
                        'the 3rd/2nd from last '
                        '(the 3080/3084 rescue '
                        'band), 31/34 probe the '
                        'mid-deep band; b7a anchor '
                        'at the neutral layer 36; '
                        'L_POST = L_INJ + 1',"""
B7_OLD = """        'b7a': 'identity attn self-swap (all '
               'heads, L_INJ) bit 0.0 per '
               'family (L_INJ=33 bank check '
               'only)',"""
B7_NEW = """        'b7a': 'identity attn self-swap (all '
               'heads, neutral layer 36) bit '
               '0.0 per family (bank check '
               'only)',"""
for old, new in ((Q_OLD, Q_NEW), (I_OLD, I_NEW),
                 (LC_OLD, LC_NEW),
                 (B7_OLD, B7_NEW)):
    assert src.count(old) == 1, old[:60]
    src = src.replace(old, new)

# ---- 4. sanity: no bad residue ----
for bad, want in (
        ('omega_p81', 0),
        ('(28, 31, 33, 34)', 0),
        ('L_CAND = (34,)', 0),
        ('seed 3084', 0),
        ('SEED = 3084', 0),
        ('151936', 0),
        ('16-head', 0),
        ('L_INJ=33', 0),
        ('attn_swaps=[(33', 0),
        ('BH[base_i, 33', 0),
        ('phase3084', 0),
        ('qwen2.5-3b-instruct', 0),
        ('3084 A', 0)):
    got = src.count(bad)
    assert got == want, (bad, got, want)

# ---- 5. positive counts ----
for good, want in (
        ('3087', 9),
        ('glm4-9b-chat-hf', 3),
        ('32-head', 5),
        ('L_CAND = (31, 34, 37, 38)', 1),
        ('31, 34, 37, 38', 3),
        ('neutral layer 36', 4),
        ('GlmForCausalLM', 1)):
    got = src.count(good)
    assert got == want, (good, got, want)

with io.open(DST, 'w', encoding='utf-8') as f:
    f.write(src)
print('PATCH_OK chars=%d' % len(src))
