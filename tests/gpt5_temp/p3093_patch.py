# -*- coding: utf-8 -*-
"""Patch 3087 A1 (GLM4 layer scan) ->
phase3093_omega_p90_qwen14b_layer_scan.py.

Order: docstring wholesale -> exact multi-line
REPS -> simple replaces (longer-first) ->
bad/want -> frozen-span identity -> compile."""
import io
import py_compile

SRC = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
       r'\phase3087_omega_p84_glm4_layer_scan.py')
DST = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
       r'\phase3093_omega_p90_qwen14b_layer_'
       r'scan.py')

src = io.open(SRC, encoding='utf-8').read()
orig = src

# ---- 0. docstring wholesale ----
i0 = src.index('"""')
i1 = src.index('"""', i0 + 3) + 3
DOC = '''"""Phase 3093 A1: Omega-P90 Qwen3-14B layer scan
=========================================================
Menu A of 3092 (gate sensitivity formalized): the
fifth spectral point is Qwen3-14B (Qwen3 family,
40L / 5120H / 40 heads / 8 kv GQA, head_dim 128,
vocab 151936, untied, bf16 ~29.6GB - the largest
checkpoint on disk; driver sysmem fallback for
~12.5GB, viability measured by the p3093 timing
probe before this run).  Spectrum so far: 4B
(36L), 3B (36L), DS7B (28L), GLM4-9B (40L,
L37/L38) - the 14B point completes the Qwen3
scale ladder and fills the biggest R1
reuse-inventory gap.

Question: over 4 candidate injection layers,
where (if anywhere) is the Qwen3-14B focus
machinery healthy on all three frozen families?

Design (3087-identical pipeline, 14B-tuned):
- bf16 eager full load (same protocol as GLM4
  p3087; sysmem-fallback share much larger -
  timing probed in p3093_timing_report).
- L_INJ in (31, 34, 37, 38) - depth 0.78 / 0.85
  / 0.93 / 0.95 of 40 layers (proportional
  mapping of the 3B candidates 28/31/33/34 of
  36); L_POST = L_INJ + 1.
- E1 repV ladder (24 pairs, med_c reference,
  b4/b8 per layer), 40-head single swap scan
  (24 pairs) -> r1 = median cos - med_c;
  n_neg from the 3071 criterion; focal top8;
  R_ALL full 40-head swap.
- b7a identity attn self-swap anchor at the
  neutral layer 36 (same as the GLM4 scan).

Verdict (preregistered, 3087-identical):
- setup/anchor fail -> setup_failed
- some L in CAND with n_neg >= 8 on ALL three
  families -> layer_rescue (best = argmax_L
  min-family n_neg; ties -> lower L, matching
  the 3084 min-family tie-break)
- else family C n_neg >= 8 at some L ->
  layer_partial
- else -> layer_absent

Output: tests/glm5/result/
rdc_query_construction_20260913/phase3093/
omega_p90_qwen14b_layer_scan/
"""'''
src = DOC + src[i1:]

# ---- 1. exact multi-line REPS ----
REPS = [
    # R1 PREREG mode (2 lines)
    ("    'mode': 'single model glm4-9b-chat-hf '\n"
     "            'bf16 eager seed 3087; three "
     "frozen '",
     "    'mode': 'single model Qwen3-14B '\n"
     "            'bf16 eager seed 3093; three "
     "frozen '"),
    # R2 PREREG scan head count
    ("            '32-head single swap scan (24 '",
     "            '40-head single swap scan (24 '"),
    # R3 PREREG R_ALL head count
    ("            'criterion, R_ALL full 32-head '",
     "            'criterion, R_ALL full 40-head '"),
    # R4 question header (7 lines)
    ("    'question': '3087 A (menu of 3086): does '\n"
     "                'the fourth spectral point '\n"
     "                'GLM4-9B (non-Qwen arch, 40L/'\n"
     "                '4096H/32H/2kv) need a layer '\n"
     "                'rescue scan before its full '\n"
     "                'arbitration, and which layer '\n"
     "                'is its focus-optimal L_INJ?  '",
     "    'question': '3093 (menu of 3092): does '\n"
     "                'the fifth spectral point '\n"
     "                'Qwen3-14B (Qwen3 family, 40L/'\n"
     "                '5120H/40H/8kv GQA) need a layer '\n"
     "                'rescue scan before its full '\n"
     "                'arbitration, and which layer '\n"
     "                'is its focus-optimal L_INJ?  '"),
    # R5 independence (5 lines)
    ("                    'before any GLM4 observation; '\n"
     "                    '3B layer-scan results (3084) '\n"
     "                    'are prior data but concern '\n"
     "                    'qwen2.5-3b only; GLM4 has no '\n"
     "                    'prior injections',",
     "                    'before any qwen3-14b '\n"
     "                    'observation; GLM4 layer-scan '\n"
     "                    'results (3087) are prior data '\n"
     "                    'but concern GLM4 only; '\n"
     "                    'qwen3-14b has no prior '\n"
     "                    'injections (same Qwen3 '\n"
     "                    'family as qwen3-4b but 3.5x '\n"
     "                    'scale)',"),
    # R6 limitations tied -> untied
    ("        'texts; tied embeddings model '",
     "        'texts; untied embeddings model '"),
    # R7 run string model name
    ("    'run': 'run1 authoritative (qwen2.5-3b '",
     "    'run': 'run1 authoritative (qwen3-14b '"),
    # R8 load log line
    ("log('glm4-9b (glm4-9b-chat-hf) loaded '",
     "log('qwen3-14b (Qwen3-14B) loaded '"),
]
for old, new in REPS:
    assert src.count(old) == 1, ('REP not unique', old[:60], src.count(old))
    src = src.replace(old, new)

# ---- 2. simple replaces (longer first) ----
SIMPLE = [
    ("assert NVOC == 151552",
     "assert NVOC == 151936"),
    ("MDIR = os.path.join(ROOT, 'models', 'hf',\n"
     "                    'glm4-9b-chat-hf')",
     "MDIR = os.path.join(ROOT, 'models', 'hf',\n"
     "                    'Qwen3-14B')"),
    ("NL, HID, KV_HEAD, NQ = 40, 4096, 2, 32",
     "NL, HID, KV_HEAD, NQ = 40, 5120, 8, 40"),
    ("PHASE = 3087", "PHASE = 3093"),
    ("NAME = 'omega_p84_glm4_layer_scan'",
     "NAME = 'omega_p90_qwen14b_layer_scan'"),
    ("    'phase3087', NAME)",
     "    'phase3093', NAME)"),
    ("SEED_MAIN = 3087", "SEED_MAIN = 3093"),
    ("SEED = 3087", "SEED = 3093"),
    ("full 32-head swap",
     "full 40-head swap"),
]
for old, new in SIMPLE:
    assert src.count(old) == 1, ('SIMPLE not unique', old[:60], src.count(old))
    src = src.replace(old, new)

# ---- 3. bad list (must be absent) ----
BAD = ['glm4-9b-chat-hf', 'omega_p84',
       'phase3087', '151552', '4096H',
       'qwen2.5-3b', '32-head', '; tied embeddings']
bad_left = [b for b in BAD if b in src]
assert not bad_left, ('BAD strings remain', bad_left)

# ---- 4. want list (must be present) ----
WANT = ['omega_p90_qwen14b_layer_scan', 'Qwen3-14B',
        '151936', 'NL, HID, KV_HEAD, NQ = 40, 5120, 8, 40',
        'PHASE = 3093', 'SEED_MAIN = 3093', 'seed 3093',
        '40-head single swap scan', 'full 40-head swap',
        'untied embeddings model', "'phase3093', NAME)",
        'qwen3-14b (Qwen3-14B) loaded']
missing = [w for w in WANT if w not in src]
assert not missing, ('WANT missing', missing)

# ---- 5. frozen-span identity vs original ----
def span(text, a, b):
    return text[text.index(a):text.index(b)]


for tag, a, b in [
    ('FAM', 'FAM = {', 'PREREG = {'),
    ('hooks_fw', 'def hook_v(st, cp):',
     'ridx = np.arange(FRONT)'),
    ('banks', 'def run_family_banks(fkey):',
     'def run_layer(B, L_INJ):'),
    ('verdict', 'if not setup_ok_all:',
     '# ==== npz ===='),
    ('npz_save', "save = {\n    'VERDICT'",
     "np.savez(npz_path, **save)"),
]:
    assert span(orig, a, b) == span(src, a, b), \
        ('frozen span changed', tag)

# ---- 6. compile & write ----
io.open(DST, 'w', encoding='utf-8').write(src)
py_compile.compile(DST, doraise=True)
print('PATCH_OK bytes=%d lines=%d'
      % (len(src.encode('utf-8')),
         src.count('\n') + 1))
