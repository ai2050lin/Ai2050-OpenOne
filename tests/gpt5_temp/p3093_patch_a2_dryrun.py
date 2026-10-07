# -*- coding: utf-8 -*-
"""Patch 3087 A2 (GLM4 L37 full arbitration)
-> phase3093_omega_p91_qwen14b_l{L}_full_
arbitration.py.

Reads the 3093 A1 result.json for the actual
layer-scan grid; refuses to run unless the A1
verdict is layer_rescue.  Docstring + PREREG
model-specific fields regenerated with real A1
numbers BEFORE any A2 observation (prereg
honesty).  Statistical core spans asserted
byte-identical."""
import ast
import io
import py_compile

ROOT = r'D:\AI2050\Ai2050-OpenOne'
SRC = (ROOT + r'\tests\glm5'
       r'\phase3087_omega_p85_glm4_l37_full_'
       r'arbitration.py')
DST = (ROOT + r'\tests\gpt5_temp\p3093_dry_a2_l%d_full_'
       r'arbitration.py')  # % L
A1RES = (ROOT + r'\tests\gpt5_temp\p3093_mock_a1_result.json')

import json
a1 = json.load(io.open(A1RES, encoding='utf-8'))
assert a1['verdict'] == 'layer_rescue', \
    ('A1 verdict not layer_rescue - A2 design '
     'must be re-decided', a1['verdict'])
LB = int(a1['stats']['rescue_best'])
LP = LB + 1
assert LB in (31, 34, 37, 38)
FL = a1['stats']['families_layers']
FKS = ('A', 'B', 'C')


def row(L):
    d = FL[str(L)]
    nn = '/'.join(str(d[f]['n_neg'])
                  for f in FKS)
    mc = '/'.join('%.2f' % d[f]['med_c']
                  for f in FKS)
    ra = '/'.join('%+.2f' % d[f]['r_all']
                  for f in FKS)
    return ('L%d n_neg %s med_c %s R_ALL %s'
            % (L, nn, mc, ra))


minfam = {L: min(FL[str(L)][f]['n_neg']
                 for f in FKS)
          for L in (31, 34, 37, 38)}
L_WEAK = min(minfam, key=lambda L: (minfam[L], L))
GRID_DETAIL = '; '.join(row(L)
                        for L in (31, 34, 37, 38))
BEST = FL[str(LB)]
GRID_SUMMARY = (
    'L%d rescues all three families '
    '(n_neg %s, med_c %s, R_ALL %s); '
    'L%d is the weakest (min-family n_neg %d)'
    % (LB,
       '/'.join(str(BEST[f]['n_neg'])
                for f in FKS),
       '/'.join('%.2f' % BEST[f]['med_c']
                for f in FKS),
       '/'.join('%+.2f' % BEST[f]['r_all']
                for f in FKS),
       L_WEAK, minfam[L_WEAK]))

NAME_NEW = ('omega_p91_qwen14b_l%d_full_'
            'arbitration') % LB

DOC = '''# -*- coding: utf-8 -*-
"""Phase 3093: Omega-P91 full-pipeline
four-way arbitration at L_INJ=%d on
Qwen3-14B (the fifth spectrum point;
largest checkpoint in the RDC arbitration
chain).

Question (3093 A2, menu of 3092): the 3093
A1 layer scan probed the Qwen3-14B causal
loading machinery on the 40-layer
equal-ratio grid L31/34/37/38 (mapping the
3084 3B grid): %s.  The full
3083/3085 pipeline (E1 repV ladder + E3
per-head scan + focal top8 + E4 255-subset
sweep + spectrum_struct + G_DS/f2~T_AB
migration gate + cross statistics) is
rerun at L_INJ=%d to obtain the four-way
trunk/dispersed x migrates verdict for the
fifth spectrum point and to extend the
3086 spectrum-transfer continuum to n=15
units (Qwen3 family scale transfer).

Pipeline (3085-identical except model,
L_INJ and the repro anchor source): three
frozen 3076 families A/B/C, 8 bodies x 4
prefixes = 32 prompts each; V-banks +
last-position banks (LG/VB/PB/BX/BA/BM/
BH); b-anchors b0/b1/b3/b4/b6/b7a/b8; E1
repV injection ladder at L_INJ=%d (24
pairs); E3 per-head single-swap scan
(40 heads x 24 pairs) -> r1; per-family
focal top8 (3071 criterion); E4 ALL 255
non-empty subsets swapped jointly + full
40-head swap; in-run spectrum (3082 E3
bit-identical column-centered SVD -> PR /
k_eff / top3); cross statistics (seed
3093, 20000 perms); G_DS main gate
(count>=4 AND min sp>0 over the 6 f2
tests); Stouffer z and Bonferroni
reported.

NEW vs 3085: L%d deterministic
reproduction anchor vs the 3093 A1
layer-scan npz (same model, same frozen
texts, same layer): n_neg and top8 must
match EXACTLY and med_c / CS1H / R_ALL
within 1e-9; mismatch -> setup failure;
smoke skips the anchor.

verdict (preregistered, tree unchanged
from 3083/3085):  setup/b-anchor/repro
failure -> fifth_setup_failed;  any
family top8 degenerate (n_neg < 8) ->
fifth_top8_degenerate;  spectrum class
from CS top3 (thresholds in the 3082
gap): all top3 >= 0.9 -> trunk; all top3
<= 0.5 -> dispersed; else mixed;
migration state: G_DS AND f2~T_AB
p<0.05 -> locked; G_DS -> partial; else
absent;  combined: trunk + locked/
partial -> fifth_trunk_migrates;
dispersed + locked/partial ->
fifth_dispersed_migrates (PR diagnosis
falsified); trunk + absent ->
fifth_trunk_no_migrate; dispersed +
absent -> fifth_dispersed_no_migrate
(3082 prediction holds); mixed ->
fifth_mixed_<state>

arbitration reading (frozen):
fifth_trunk_migrates -> trunk anatomy
with locked head migration extends WITHIN
the Qwen3 family across 3.9x parameter
scale (qwen3-4b -> 14B);
fifth_dispersed_migrates -> 3082 PR
diagnostic falsified at scale (dispersed
anatomy coexists with locked migration in
the largest Qwen3 checkpoint);
fifth_trunk_no_migrate -> trunk-with-
migration remains qwen3-4b-specific even
within the family (scale breaks the
anatomy); fifth_dispersed_no_migrate ->
3082 prediction holds (trunk anatomy
absent at 14B scale).

model adaptation (frozen before run):
Qwen3-14B (Qwen3ForCausalLM, hidden
5120, 40 layers, 40 heads, 8 kv heads,
head_dim 128, inter 17408, vocab 151936,
bf16 ~29.5 GB allocated on a 17.1 GB
card, ~12.5 GB driver sysmem fallback,
timing probed non-OOM at 0.60s/forward);
L_INJ=%d, L_POST=%d
(injection at the layer-%d V,
observation at the next layer; layer %d
downstream; the 3093 A1 scan selected
L%d as the rescue layer); E2H/lens
machinery omitted as in 3081/3083/3085
(all cosines from forward logits); input
embeddings are UNTIED
(tie_word_embeddings=False); same frozen
3076 texts; Qwen3-14B is the same family
as qwen3-4b at 3.9x parameters, 40L/
5120H/40H/8kv vs 36L/2560H/32H/8kv - the
family-internal scale-transfer point for
the 3086 continuum (n 12 -> 15 units).

limitations (recorded): L_INJ=%d chosen
from the 3093 A1 descriptive layer scan
(%s); the rescue band boundary
remains unmapped (coarse 4-layer grid on
a 40-layer stack); qwen3-14b shares the
Qwen3 family with qwen3-4b - the scale
transfer point in the chain; instruct-
tuned like the other models; same 3076
texts; n=24 pairs per family pair;
partial spearman first-order on n=24 is
orientation evidence; causal-connective
paradigm, shared syntactic frame;
migration defined on the per-family
focal top-8 subset frame; spectrum
classification thresholds (0.9 / 0.5)
chosen in the 3082 empirical gap BEFORE
this run.

memory discipline: single model; families
sequential; big banks fp64 CPU freed after
each family (del + gc + empty_cache); no
W40, no Wo%d/Wo%d (E2H omitted).
"""''' % (LB, GRID_SUMMARY, LB, LB, LB, LB,
          LP, LB, LP, LB, LB, GRID_DETAIL, LB,
          LP)

src = io.open(SRC, encoding='utf-8').read()
orig = src

# ---- 0. docstring wholesale ----
i0 = src.index('"""Phase 3087:')
i1 = src.index('"""', i0 + 3) + 3
src = DOC + src[i1:]


# ---- 1. span replaces (PREREG fields) ----
def span_rep(text, start_marker, end_marker,
             new_text):
    assert text.count(start_marker) == 1, \
        ('span start not unique', start_marker[:50])
    i0 = text.index(start_marker)
    j = text.index(end_marker, i0)
    i1 = j + len(end_marker)
    return text[:i0] + new_text + text[i1:]


NEW_Q = ("    'question': '3093 A2 (menu of "
         "3092): the '\n"
         "                '3093 A1 layer scan "
         "probed the '\n"
         "                'Qwen3-14B causal "
         "loading '\n"
         "                'machinery on the "
         "40-layer '\n"
         "                'equal-ratio grid "
         "L31/34/37/38 '\n"
         "                '(mapping the 3084 "
         "3B grid): '\n"
         "                '%s. The four-way '\n"
         "                'trunk/dispersed x "
         "migrates '\n"
         "                'arbitration is "
         "OPENED here for '\n"
         "                'the fifth spectrum "
         "point: '\n"
         "                'rerun the full "
         "3083/3085 '\n"
         "                'pipeline (E1 + E3 "
         "+ E4 + '\n"
         "                'spectrum + cross "
         "statistics) '\n"
         "                'at L_INJ=%d and "
         "combine the '\n"
         "                'in-run spectrum "
         "class with the '\n"
         "                'migration gate "
         "into the '\n"
         "                'four-way verdict, "
         "extending '\n"
         "                'the 3086 "
         "spectrum-transfer '\n"
         "                'continuum to n=15 "
         "units (Qwen3 '\n"
         "                'family scale "
         "transfer).'," % (GRID_SUMMARY, LB))
src = span_rep(
    src,
    "    'question': '3087 A2 (menu of 3086): the '",
    "'non-Qwen model).',", NEW_Q)

NEW_ARB = ("    'arbitration': "
           "'fifth_trunk_migrates -> '\n"
           "        'trunk anatomy with locked "
           "head '\n"
           "        'migration extends WITHIN "
           "the '\n"
           "        'Qwen3 family across 3.9x "
           "scale '\n"
           "        '(qwen3-4b -> 14B); '\n"
           "        'fifth_dispersed_migrates "
           "-> 3082 PR '\n"
           "        'diagnostic falsified at "
           "scale '\n"
           "        '(dispersed anatomy "
           "coexists with '\n"
           "        'locked migration in the "
           "largest '\n"
           "        'Qwen3 checkpoint); '\n"
           "        'fifth_trunk_no_migrate -> "
           "trunk-with-'\n"
           "        'migration remains "
           "qwen3-4b-specific '\n"
           "        'even within the family "
           "(scale '\n"
           "        'breaks the anatomy); '\n"
           "        'fifth_dispersed_no_migrate "
           "-> 3082 '\n"
           "        'prediction holds (trunk "
           "anatomy '\n"
           "        'absent at 14B scale)',")
src = span_rep(
    src,
    "    'arbitration': 'fourth_trunk_migrates -> '",
    "'rare outside the Qwen family)',", NEW_ARB)

NEW_LM = ("    'layer_mapping': 'L_INJ=%d (V "
          "injection + '\n"
          "                     'attn swaps), "
          "L_POST=%d '\n"
          "                     '(block-output "
          "continuity '\n"
          "                     'check); 40-layer "
          "stack, '\n"
          "                     'layer %d "
          "downstream of '\n"
          "                     'the injection; "
          "selected by '\n"
          "                     'the 3093 A1 "
          "layer scan '\n"
          "                     '(L31/34/37/38 "
          "equal-ratio '\n"
          "                     'mapping of the "
          "3084 3B grid: '\n"
          "                     '%s); qwen3-4b "
          "used '\n"
          "                     'L34/L35 of 36 "
          "(2nd from '\n"
          "                     'last / last); "
          "qwen3-14b grid '\n"
          "                     'kept the same "
          "4-point '\n"
          "                     'spacing',"
          % (LB, LP, LP, GRID_DETAIL))
src = span_rep(
    src,
    "    'layer_mapping': 'L_INJ=37 (V injection + '",
    "'spacing',", NEW_LM)

NEW_LIM = ("    'limitations': 'qwen3-14b is the "
           "same '\n"
           "        'family as qwen3-4b at 3.9x "
           "parameters '\n"
           "        '- a family-internal "
           "scale-transfer '\n"
           "        'point, NOT an "
           "architecturally '\n"
           "        'independent one (GLM4-9B "
           "remains the '\n"
           "        'most independent point in "
           "the chain '\n"
           "        'so far); instruct-tuned "
           "like the '\n"
           "        'other instruct models; "
           "same '\n"
           "        '3076 texts (model "
           "independence, '\n"
           "        'not text independence); "
           "n=24 pairs '\n"
           "        'per family pair; "
           "first-order '\n"
           "        'partial on n=24 is "
           "orientation '\n"
           "        'only; causal-connective "
           "paradigm, '\n"
           "        'shared syntactic frame; "
           "focal '\n"
           "        'top-8 subset frame; "
           "spectrum '\n"
           "        'classification thresholds "
           "(0.9 / '\n"
           "        '0.5) sit in the 3082 gap "
           "but are '\n"
           "        'two-point calibrated; "
           "Qwen3-14B '\n"
           "        'keeps input embeddings "
           "untied '\n"
           "        '(tie_word_embeddings="
           "False) - the '\n"
           "        'forwards/logits-only "
           "protocol is '\n"
           "        'unaffected; L_INJ=%d was "
           "'\n"
           "        'selected from the 3093 A1 "
           "layer '\n"
           "        'scan (%s) - layer-choice '\n"
           "        'dependence is handled "
           "explicitly '\n"
           "        'but the rescue-band "
           "boundary '\n"
           "        'remains unmapped (coarse "
           "4-layer '\n"
           "        'grid on a 40-layer "
           "stack); bf16 '\n"
           "        'weights exceed VRAM by "
           "~12.5 GB and '\n"
           "        'rely on the verified "
           "driver sysmem '\n"
           "        'fallback (non-OOM at "
           "0.60s/fw in '\n"
           "        'the p3093 loading probe; "
           "identical '\n"
           "        'behavior recorded in "
           "A1)'," % (LB, GRID_DETAIL))
src = span_rep(
    src,
    "    'limitations': 'glm4-9b shares no "
    "lineage '",
    "'behavior recorded in A1)',", NEW_LIM)

# ---- 2. exact REPS (mode + repro + defs) ----
REPS = [
    ("    'mode': 'single model "
     "glm4-9b-chat-hf '\n"
     "            'bf16 eager seed 3087; "
     "three '",
     "    'mode': 'single model Qwen3-14B '\n"
     "            'bf16 eager seed 3093; "
     "three '"),
    ("'W32 - all cosines from forward '",
     "'W40 - all cosines from forward '"),
    ("            'ladder at L_INJ=37 (24 "
     "pairs, '",
     "            'ladder at L_INJ=%d (24 "
     "pairs, '" % LB),
    ("            'med_c reference), 32-head "
     "single '",
     "            'med_c reference), 40-head "
     "single '"),
    ("            'jointly (24 pairs each), "
     "full '\n"
     "            '32-head swap; b-anchors "
     "b0/b1/b3/'",
     "            'jointly (24 pairs each), "
     "full '\n"
     "            '40-head swap; b-anchors "
     "b0/b1/b3/'"),
    ("            'L37 deterministic "
     "reproduction '\n"
     "            'anchor vs the 3087 A1 "
     "layer-scan '",
     "            'L%d deterministic "
     "reproduction '\n"
     "            'anchor vs the 3093 A1 "
     "layer-scan '" % LB),
    ("                    'at L37 are new "
     "data (new '\n"
     "                    'forwards, seed "
     "3087); the '",
     "                    'at L%d are new "
     "data (new '\n"
     "                    'forwards, seed "
     "3093); the '" % LB),
    ("                    'were fixed before "
     "any L37 '",
     "                    'were fixed before "
     "any L%d '" % LB),
    ("                    'observation; 3087 "
     "A1 '",
     "                    'observation; 3093 "
     "A1 '"),
    ("                    'layer -> L37 "
     "values must '\n"
     "                    'match the 3087 A1 "
     "npz per '",
     "                    'layer -> L%d "
     "values must '\n"
     "                    'match the 3093 A1 "
     "npz per '" % LB),
    ("'before any glm4-9b '",
     "'before any qwen3-14b '"),
    ("                   'pair k (32 heads)',",
     "                   'pair k (40 heads)',"),
    ("                'AND min(sp) > 0; seed "
     "3087, '",
     "                'AND min(sp) > 0; seed "
     "3093, '"),
    ("        'two-sided permutation p (seed "
     "3087, '",
     "        'two-sided permutation p (seed "
     "3093, '"),
    ("        'freed between families; no "
     "W32 / '\n"
     "        'Wo37 / Wo38 (E2H and lens "
     "probes '",
     "        'freed between families; no "
     "W40 / '\n"
     "        'Wo%d / Wo%d (E2H and lens "
     "probes '" % (LB, LP)),
    ("log('glm4-9b (glm4-9b-chat-hf) loaded '",
     "log('qwen3-14b (Qwen3-14B) loaded '"),
    ("        'K3/NP_USE differ from the "
     "3087 A1 '",
     "        'K3/NP_USE differ from the "
     "3093 A1 '"),
    ("# ==== L37 deterministic reproduction\n"
     "# anchor (vs 3087 A1 layer-scan npz) "
     "====",
     "# ==== L%d deterministic reproduction\n"
     "# anchor (vs 3093 A1 layer-scan npz) "
     "====" % LB),
    ("        log('repro[%s] vs 3087A1 L37: '",
     "        log('repro[%s] vs 3093A1 L%d: '"
     % ('%s', LB)),
    ("    'run': 'run1 authoritative (glm4-9b "
     "bf16 '",
     "    'run': 'run1 authoritative "
     "(qwen3-14b bf16 '"),
]
for old, new in REPS:
    assert src.count(old) == 1, \
        ('REP not unique', old[:60],
         src.count(old))
    src = src.replace(old, new)

# ---- 3. GSUBS ----
GSUBS = [
    ('fourth_', 'fifth_'),
    ('omega_p84_glm4_layer_scan',
     'omega_p90_qwen14b_layer_scan'),
    ('phase3087', 'phase3093'),
    ('glm4-9b-chat-hf', 'Qwen3-14B'),
    ('omega_p85_glm4_l37_full_arbitration',
     NAME_NEW),
    ('MED_C_L37_', 'MED_C_L%d_' % LB),
    ('N_NEG_L37_', 'N_NEG_L%d_' % LB),
    ('TOP8_L37_', 'TOP8_L%d_' % LB),
    ('CS1H_L37_', 'CS1H_L%d_' % LB),
    ('R_ALL_L37_', 'R_ALL_L%d_' % LB),
    ('L_INJ = 37', 'L_INJ = %d' % LB),
    ('PHASE = 3087', 'PHASE = 3093'),
    ('SEED_MAIN = 3087', 'SEED_MAIN = 3093'),
    ('SEED = 3087', 'SEED = 3093'),
    ('NL, HID, KV_HEAD, NQ = 40, 4096, 2, 32',
     'NL, HID, KV_HEAD, NQ = 40, 5120, 8, 40'),
    ('assert NVOC == 151552',
     'assert NVOC == 151936'),
]
for old, new in GSUBS:
    n = src.count(old)
    assert n >= 1, ('GSUBS zero-count', old)
    src = src.replace(old, new)

# ---- 4. bad / want ----
BAD = ['glm4', 'Glm', '4096',
       '151552', '3087', 'omega_p84',
       'omega_p85_glm4', 'fourth',
       '32-head', '32 heads',
       'Wo37', 'W32', 'qwen2.5-3b',
       'glm4-9b']
# note: 'L37' NOT in BAD - the 4-point grid
# labels (L31..L38) legitimately contain it
bad_left = [b for b in BAD if b in src]
import io as _io
_lns = src.splitlines()
_ctx = []
for _b in bad_left:
    _hit = [ln for ln in _lns if _b in ln][:3]
    _ctx.append('BAD %s => %s'
              % (_b, ' ## '
                 .join(_hit)))
_io.open(r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\p3093_bad_ctx.txt', 'w',
         encoding='utf-8').write('\n'.join(_ctx) + '\n')
assert not bad_left, ('BAD remain', bad_left)

WANT = [NAME_NEW, 'Qwen3-14B', '151936',
        'NL, HID, KV_HEAD, NQ = 40, 5120, 8, 40',
        'PHASE = 3093', 'SEED_MAIN = 3093',
        'seed 3093', 'L_INJ = %d' % LB,
        'fifth_trunk_migrates',
        'fifth_setup_failed',
        'omega_p90_qwen14b_layer_scan',
        'MED_C_L%d_' % LB,
        '40-head single', '40 heads',
        'qwen3-14b (Qwen3-14B) loaded',
        "'phase3093', NAME)",
        'L_INJ=%d' % LB, 'L_POST=%d' % LP,
        'no W40', 'Wo%d' % LB]
missing = [w for w in WANT if w not in src]
assert not missing, ('WANT missing', missing)


# ---- 5. frozen-span identity ----
def span(text, a, b):
    return text[text.index(a):text.index(b)]


for tag, a, b in [
    ('FAM', 'FAM = {', 'PREREG = {'),
    ('hooks_fw', 'def hook_v(st, cp):',
     'ALLH = np.arange(NQW)'),
    ('spectrum', 'def spectrum_struct(',
     '# ==== load model ===='),
    ('repro_pre', "REPRO = {}\n"
     "REPRO_OK = True", 'if SMOKE:'),
    ('repro_tail',
     "REPRO[fk] = {'medc_diff': d_mc,",
     "'rall_diff': d_ra,"),
]:
    assert span(orig, a, b) == span(src, a, b), \
        ('frozen span changed', tag)

# ---- 6. compile & write ----
out = DST % LB
io.open(out, 'w', encoding='utf-8').write(src)
py_compile.compile(out, doraise=True)
print('PATCH_A2_OK L=%d bytes=%d lines=%d'
      % (LB, len(src.encode('utf-8')),
         src.count('\n') + 1))
