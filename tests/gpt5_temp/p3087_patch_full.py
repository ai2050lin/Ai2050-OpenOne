# -*- coding: utf-8 -*-
"""Phase 3087 A2 generator (p3087_patch_full.py):
patch phase3085_omega_p82_l34_full_arbitration.py ->
phase3087_omega_p85_glm4_l{L}_full_arbitration.py.

Reads the authoritative 3087 A1 layer-scan npz,
selects the rescue layer (argmax over min-family
n_neg, tie-break max mean med_c), rewrites the
docstring and the eight narrative PREREG keys
wholesale with A1 numbers, patches constants /
load asserts / repro anchor / run description,
then applies the global subs (third_ -> fourth_,
'seed 3085' -> 'seed 3087', '(16 heads)' ->
'(32 heads)', qwen spectrum note -> glm4 note).
Asserts: every bad residue count == 0, want
counts present, py_compile ok.
GEN_CHECK=1 -> diagnose only, do not write.
"""
import io
import os
import py_compile
import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
SRC = os.path.join(
    ROOT, 'tests', 'glm5',
    'phase3085_omega_p82_l34_full_arbitration.py')
A1_NPZ = os.path.join(
    ROOT, 'tests', 'glm5', 'result',
    'rdc_query_construction_20260913',
    'phase3087', 'omega_p84_glm4_layer_scan',
    'omega_p84_glm4_layer_scan.npz')
DST_T = os.path.join(
    ROOT, 'tests', 'glm5',
    'phase3087_omega_p85_glm4_l{L}_'
    'full_arbitration.py')
CHECK = os.environ.get('GEN_CHECK', '0') == '1'

FKEYS = ('A', 'B', 'C')
L_CAND = (31, 34, 37, 38)

z = np.load(A1_NPZ, allow_pickle=False)
verdict = str(z['VERDICT'])
assert verdict != 'setup_failed', \
    'A1 setup_failed - A2 generation aborted'
stats = {}
for L in L_CAND:
    nn = [int(z['N_NEG_L%d_%s' % (L, fk)])
          for fk in FKEYS]
    mc = [float(z['MED_C_L%d_%s' % (L, fk)])
          for fk in FKEYS]
    ra = [float(z['R_ALL_L%d_%s' % (L, fk)])
          for fk in FKEYS]
    stats[L] = (nn, mc, ra)

def fmt(v, nd=2):
    return ('%.' + str(nd) + 'f') % v

# rescue selection (A1 tree: argmax over
# min-family n_neg; tie-break max mean med_c)
best = max(
    L_CAND,
    key=lambda L: (min(stats[L][0]),
                   sum(stats[L][1]) / 3.0))
L_RES = int(best)
L_POST = L_RES + 1

nn_res, mc_res, ra_res = stats[L_RES]
nn_txt = '/'.join(str(v) for v in nn_res)
mc_txt = '/'.join(fmt(v) for v in mc_res)
ra_txt = '/'.join(fmt(v) for v in ra_res)
A1S = ('L%d rescues all three families '
       '(n_neg %s, med_c %s, R_ALL %s)'
       % (L_RES, nn_txt, mc_txt, ra_txt))
weak_L = min(L_CAND,
             key=lambda L: min(stats[L][0]))
weak_txt = ('L%d is the weakest (min-family '
            'n_neg %d)'
            % (weak_L, min(stats[weak_L][0])))
grid_txt = '; '.join(
    'L%d n_neg %s med_c %s'
    % (L,
       '/'.join(str(v) for v in stats[L][0]),
       '/'.join(fmt(v) for v in stats[L][1]))
    for L in L_CAND)
SELS = ('strongest rescue magnitudes, '
        'min-family n_neg %d' % min(nn_res))
SEL_txt = SELS + ' on grid ' + grid_txt

# ================= new docstring =================
NEW_DOC = """Phase 3087: Omega-P85 full-pipeline
four-way arbitration at L_INJ={L} on
glm4-9b-chat-hf (the fourth spectrum point;
first non-Qwen model in the RDC arbitration
chain).

Question (3087 A2, menu of 3086): the 3087
A1 layer scan probed the GLM4-9B causal
loading machinery on the 40-layer
equal-ratio grid L31/34/37/38 (mapping the
3084 3B grid): {A1S}; {weak_txt}.  The full
3083/3085 pipeline (E1 repV ladder + E3
per-head scan + focal top8 + E4 255-subset
sweep + spectrum_struct + G_DS/f2~T_AB
migration gate + cross statistics) is
rerun at L_INJ={L} to obtain the four-way
trunk/dispersed x migrates verdict for the
fourth spectrum point and to extend the
3086 spectrum-transfer continuum to n=12
units.

Pipeline (3085-identical except model,
L_INJ and the repro anchor source): three
frozen 3076 families A/B/C, 8 bodies x 4
prefixes = 32 prompts each; V-banks +
last-position banks (LG/VB/PB/BX/BA/BM/
BH); b-anchors b0/b1/b3/b4/b6/b7a/b8; E1
repV injection ladder at L_INJ={L} (24
pairs); E3 per-head single-swap scan
(32 heads x 24 pairs) -> r1; per-family
focal top8 (3071 criterion); E4 ALL 255
non-empty subsets swapped jointly + full
32-head swap; in-run spectrum (3082 E3
bit-identical column-centered SVD -> PR /
k_eff / top3); cross statistics (seed
3087, 20000 perms); G_DS main gate
(count>=4 AND min sp>0 over the 6 f2
tests); Stouffer z and Bonferroni
reported.

NEW vs 3085: L{L} deterministic
reproduction anchor vs the 3087 A1
layer-scan npz (same model, same frozen
texts, same layer): n_neg and top8 must
match EXACTLY and med_c / CS1H / R_ALL
within 1e-9; mismatch -> setup failure;
smoke skips the anchor.

verdict (preregistered, tree unchanged
from 3083/3085):  setup/b-anchor/repro
failure -> fourth_setup_failed;  any
family top8 degenerate (n_neg < 8) ->
fourth_top8_degenerate;  spectrum class
from CS top3 (thresholds in the 3082
gap): all top3 >= 0.9 -> trunk; all top3
<= 0.5 -> dispersed; else mixed;
migration state: G_DS AND f2~T_AB
p<0.05 -> locked; G_DS -> partial; else
absent;  combined: trunk + locked/
partial -> fourth_trunk_migrates;
dispersed + locked/partial ->
fourth_dispersed_migrates (PR diagnosis
falsified); trunk + absent ->
fourth_trunk_no_migrate; dispersed +
absent -> fourth_dispersed_no_migrate
(3082 prediction holds); mixed ->
fourth_mixed_<state>

arbitration reading (frozen):
fourth_trunk_migrates -> trunk anatomy
with locked head migration generalizes
beyond the Qwen lineage (DS7B outlier
reading withdrawn);
fourth_dispersed_migrates -> 3082 PR
diagnostic falsified (dispersed anatomy
coexists with locked migration in a
non-Qwen model); fourth_trunk_no_
migrate -> qwen3-4b specificity deepened
(trunk is Qwen-lineage anatomy);
fourth_dispersed_no_migrate -> 3082
prediction holds (trunk anatomy rare
outside the Qwen family).

model adaptation (frozen before run):
glm4-9b-chat-hf (GlmForCausalLM, hidden
4096, 40 layers, 32 heads, 2 kv heads,
head_dim 128, inter 13696, vocab 151552,
bf16 ~18.8 GB allocated, verified
non-OOM at seq<=74 via driver sysmem
fallback); L_INJ={L}, L_POST={LP}
(injection at the layer-{L} V,
observation at the next layer; layer {LP}
downstream; the 3087 A1 scan selected
L{L} as the rescue layer); E2H/lens
machinery omitted as in 3081/3083/3085
(all cosines from forward logits); input
embeddings are UNTIED
(tie_word_embeddings=False); same frozen
3076 texts; GLM4-9B is architecturally
outside the Qwen2 lineage - the
independent fourth spectrum point for
the 3086 continuum (n 9 -> 12 units).

limitations (recorded): L_INJ={L} chosen
from the 3087 A1 descriptive layer scan
({SEL_txt}); the rescue band boundary
remains unmapped (coarse 4-layer grid on
a 40-layer stack); glm4-9b shares no
lineage with the DS7B backbone or the
Qwen-family models - the most
architecturally independent model in the
chain so far; chat-tuned like the other
instruct models; same 3076 texts; n=24
pairs per family pair; partial spearman
first-order on n=24 is orientation
evidence; causal-connective paradigm,
shared syntactic frame; migration
defined on the per-family focal top-8
subset frame; spectrum classification
thresholds (0.9 / 0.5) chosen in the
3082 empirical gap BEFORE this run.

memory discipline: single model; families
sequential; big banks fp64 CPU freed after
each family (del + gc + empty_cache); no
W32, no Wo{L}/Wo{LP} (E2H omitted).
""".format(L=L_RES, LP=L_POST, A1S=A1S,
           weak_txt=weak_txt, SEL_txt=SEL_txt)

# ============ new PREREG key texts ============
NEW_MODE = """    'mode': 'single model glm4-9b-chat-hf '
            'bf16 eager seed 3087; three '
            'frozen 3076 families processed '
            'sequentially in one process '
            '(A, B, C; big banks freed between '
            'families); per-family protocol '
            '3081-identical minus E2H/lens (no '
            'W32 - all cosines from forward '
            'logits): repV 3065-identical E1 '
            'ladder at L_INJ={L} (24 pairs, '
            'med_c reference), 32-head single '
            'swap scan, per-family focal top8 '
            'by the 3071 signed criterion, ALL '
            '255 non-empty subsets swapped '
            'jointly (24 pairs each), full '
            '32-head swap; b-anchors b0/b1/b3/'
            'b4/b6/b7a/b8 (b5/b7c/a2lens '
            'omitted with the lens machinery); '
            'in-run: per-family causal '
            'spectrum (3082 E3 bit-identical '
            'column-centered SVD of CS and '
            'CS1H: PR / k_eff / top3); '
            'L{L} deterministic reproduction '
            'anchor vs the 3087 A1 layer-scan '
            'npz (n_neg/top8 exact; '
            'med_c/CS1H/R_ALL within 1e-9; '
            'feeds setup_ok; smoke skips); '
            'smoke mode optional (SMOKE=1: 12 '
            'masks, K3=4 pairs, NP_USE=8, '
            'cross stats off)',
""".format(L=L_RES)

NEW_QUESTION = """    'question': '3087 A2 (menu of 3086): the '
                '3087 A1 layer scan probed the '
                'GLM4-9B causal loading '
                'machinery on the 40-layer '
                'equal-ratio grid L31/34/37/38 '
                '(mapping the 3084 3B grid): '
                '{A1S}; {weak_txt}. The four-way '
                'trunk/dispersed x migrates '
                'arbitration is OPENED here for '
                'the fourth spectrum point: '
                'rerun the full 3083/3085 '
                'pipeline (E1 + E3 + E4 + '
                'spectrum + cross statistics) '
                'at L_INJ={L} and combine the '
                'in-run spectrum class with the '
                'migration gate into the '
                'four-way verdict, extending '
                'the 3086 spectrum-transfer '
                'continuum to n=12 units (first '
                'non-Qwen model).',
""".format(L=L_RES, A1S=A1S, weak_txt=weak_txt)

NEW_ARBITRATION = """    'arbitration': 'fourth_trunk_migrates -> '
        'trunk anatomy with locked head '
        'migration generalizes beyond the '
        'Qwen lineage (DS7B outlier reading '
        'withdrawn); '
        'fourth_dispersed_migrates -> 3082 PR '
        'diagnostic falsified (dispersed '
        'anatomy coexists with locked '
        'migration in a non-Qwen model); '
        'fourth_trunk_no_migrate -> qwen3-4b '
        'specificity deepened (trunk is '
        'Qwen-lineage anatomy); '
        'fourth_dispersed_no_migrate -> 3082 '
        'prediction holds (trunk anatomy '
        'rare outside the Qwen family)',
"""

NEW_INDEPENDENCE = """    'independence': 'f2 values and CS spectra '
                    'at L{L} are new data (new '
                    'forwards, seed 3087); the '
                    'gate G_DS, the spectrum '
                    'thresholds (0.9 / 0.5) and '
                    'the combined verdict tree '
                    'were fixed before any L{L} '
                    'observation; 3087 A1 '
                    'priors (n_neg/med_c/R_ALL '
                    'descriptives) guided the '
                    'LAYER CHOICE ONLY - all '
                    'verdict inputs (f2/T/U, '
                    'spectrum class, gates) are '
                    'computed fresh in-run; '
                    'family texts are the same '
                    'frozen 3076 texts',
""".format(L=L_RES)

NEW_REPRO = """    'repro_anchor': 'activation collection is '
                    'deterministic: same model, '
                    'same frozen texts, same '
                    'layer -> L{L} values must '
                    'match the 3087 A1 npz per '
                    'family (n_neg exact, top8 '
                    'exact, med_c/CS1H/R_ALL '
                    'abs diff <= 1e-9); the 1e-9 '
                    'tolerance covers the fp32-'
                    'roundtrip noise class '
                    'measured by the 3083-vs-'
                    '3084 L33 probe (max 5.8e-13 '
                    'med_c); mismatch -> '
                    'fourth_setup_failed; smoke '
                    'skips the anchor (K3/NP_USE '
                    'differ)',
""".format(L=L_RES)

NEW_LAYER_MAPPING = """    'layer_mapping': 'L_INJ={L} (V injection + '
                     'attn swaps), L_POST={LP} '
                     '(block-output continuity '
                     'check); 40-layer stack, '
                     'layer {LP} downstream of '
                     'the injection; selected by '
                     'the 3087 A1 layer scan '
                     '(L31/34/37/38 equal-ratio '
                     'mapping of the 3084 3B grid: '
                     '{grid_txt}); qwen3-4b used '
                     'L34/L35 of 36 (2nd from '
                     'last / last); GLM4 grid '
                     'kept the same 4-point '
                     'spacing',
""".format(L=L_RES, LP=L_POST, grid_txt=grid_txt)

NEW_LIMITATIONS = """    'limitations': 'glm4-9b shares no lineage '
        'with the DS7B backbone or the '
        'Qwen-family models - the most '
        'architecturally independent point '
        'in the chain so far (fourth '
        'spectrum point); chat-tuned like '
        'the other instruct models; same '
        '3076 texts (model independence, '
        'not text independence); n=24 pairs '
        'per family pair; first-order '
        'partial on n=24 is orientation '
        'only; causal-connective paradigm, '
        'shared syntactic frame; focal '
        'top-8 subset frame; spectrum '
        'classification thresholds (0.9 / '
        '0.5) sit in the 3082 gap but are '
        'two-point calibrated; GLM4-9B '
        'keeps input embeddings untied '
        '(tie_word_embeddings=False) - the '
        'forwards/logits-only protocol is '
        'unaffected; L_INJ={L} was '
        'selected from the 3087 A1 layer '
        'scan ({SEL_txt}) - layer-choice '
        'dependence is handled explicitly '
        'but the rescue-band boundary '
        'remains unmapped (coarse 4-layer '
        'grid on a 40-layer stack); bf16 '
        'weights exceed VRAM by ~1.7 GB and '
        'rely on the verified driver sysmem '
        'fallback (non-OOM at seq<=74 in '
        'the loading probe; identical '
        'behavior recorded in A1)',
""".format(L=L_RES, SEL_txt=SEL_txt)

NEW_MEMORY = """    'memory_discipline': 'single model; '
        'families sequential with big banks '
        'freed between families; no W32 / '
        'Wo{L} / Wo{LP} (E2H and lens probes '
        'omitted); del + gc + empty_cache '
        'at family end and run end',
""".format(L=L_RES, LP=L_POST)

NEW = {'mode': NEW_MODE,
       'question': NEW_QUESTION,
       'arbitration': NEW_ARBITRATION,
       'independence': NEW_INDEPENDENCE,
       'repro_anchor': NEW_REPRO,
       'layer_mapping': NEW_LAYER_MAPPING,
       'limitations': NEW_LIMITATIONS,
       'memory_discipline': NEW_MEMORY}

KEYS_ORDER = ['mode', 'question', 'arbitration',
              'independence', 'repro_anchor',
              'families', 'layer_mapping',
              'top8_criterion', 'spectrum',
              'definitions', 'anchors', 'gates',
              'verdict', 'statistics_discipline',
              'limitations', 'memory_discipline']

# ================== load source ==================
with io.open(SRC, 'r', encoding='utf-8') as f:
    src = f.read()
orig_len = len(src)

# ---- 1. docstring wholesale ----
parts = src.split('"""', 2)
assert len(parts) == 3 and \
    parts[0] == '# -*- coding: utf-8 -*-\n' \
    and parts[1].startswith('Phase 3085'), \
    'docstring split unexpected'
src = parts[0] + '"""' + NEW_DOC + '"""' \
    + parts[2]

# ---- 2. PREREG wholesale (8 keys) ----
p_start = src.find('PREREG = {')
assert p_start > 0
p_end = src.find('\n}', p_start)
assert p_end > 0
cuts = []
pos = p_start
for k in KEYS_ORDER:
    i = src.find("    '%s':" % k, pos)
    assert i > 0, 'key not found: ' + k
    cuts.append((k, i))
    pos = i + 4
cuts_sorted = sorted(cuts, key=lambda t: t[1])
assert [k for k, _ in cuts_sorted] \
    == KEYS_ORDER, 'key order mismatch'
buf = [src[p_start:cuts_sorted[0][1]]]
for idx, (k, i) in enumerate(cuts_sorted):
    nxt = cuts_sorted[idx + 1][1] \
        if idx + 1 < len(cuts_sorted) else p_end
    if k in NEW:
        buf.append(NEW[k])
    else:
        buf.append(src[i:nxt])
new_prereg = ''.join(buf)
src = src[:p_start] + new_prereg + src[p_end:]

# ---- 3. exact replacements ----
REPS = [
    ("PHASE = 3085", "PHASE = 3087"),
    ("NAME = 'omega_p82_l34_full_arbitration'",
     "NAME = 'omega_p85_glm4_l%d_"
     "full_arbitration'" % L_RES),
    ("    'phase3085', NAME)",
     "    'phase3087', NAME)"),
    ("                    "
     "'qwen2.5-3b-instruct')",
     "                    'glm4-9b-chat-hf')"),
    ("NL, HID, KV_HEAD, NQ = 36, 2048, 2, 16",
     "NL, HID, KV_HEAD, NQ = 40, 4096, 2, 32"),
    ("SEED_MAIN = 3085", "SEED_MAIN = 3087"),
    ("L_INJ = 34", "L_INJ = %d" % L_RES),
    ("L_POST = 35", "L_POST = %d" % L_POST),
    ("SEED = 3085", "SEED = 3087"),
    ("assert NVOC == 151936",
     "assert NVOC == 151552"),
    ("log('tie_word_embeddings=%s (tied '\n"
     "    'embeddings: forwards/logits-only '\n"
     "    'protocol unaffected, recorded as an '\n"
     "    'adaptation)' % TIED)",
     "log('tie_word_embeddings=%s (untied '\n"
     "    'embeddings: no tie consideration '\n"
     "    'needed; protocol unaffected)'"
     " % TIED)"),
    ("log('qwen2.5-3b (qwen2.5-3b-instruct) "
     "loaded '",
     "log('glm4-9b (glm4-9b-chat-hf) loaded '"),
    ("# ==== L34 deterministic reproduction",
     "# ==== L%d deterministic reproduction"
     % L_RES),
    ("# anchor (vs 3084 layer-scan npz) ====",
     "# anchor (vs 3087 A1 layer-scan npz)"
     " ===="),
    ("    z84 = np.load(os.path.join(",
     "    zA1 = np.load(os.path.join("),
    ("        'phase3084',",
     "        'phase3087',"),
    ("        'omega_p81_3b_layer_scan',",
     "        'omega_p84_glm4_layer_scan',"),
    ("        'omega_p81_3b_layer_scan.npz'),",
     "        'omega_p84_glm4_layer_scan."
     "npz'),"),
    ("        'K3/NP_USE differ from the 3084 '",
     "        'K3/NP_USE differ from the "
     "3087 A1 '"),
    ("- float(z84['MED_C_L34_'",
     "- float(zA1['MED_C_L%d_'" % L_RES),
    ("== int(z84['N_NEG_L34_' + fk]))",
     "== int(zA1['N_NEG_L%d_' + fk]))" % L_RES),
    ("in z84['TOP8_L34_' + fk]])",
     "in zA1['TOP8_L%d_' + fk]])" % L_RES),
    ("- z84['CS1H_L34_' + fk])))",
     "- zA1['CS1H_L%d_' + fk])))" % L_RES),
    ("- float(z84['R_ALL_L34_'",
     "- float(zA1['R_ALL_L%d_'" % L_RES),
    ("log('repro[%s] vs 3084 L34: '",
     "log('repro[%%s] vs 3087A1 L%d: '"
     % L_RES),
    ("'run': 'run1 authoritative (qwen2.5-3b "
     "bf16 '",
     "'run': 'run1 authoritative (glm4-9b "
     "bf16 '"),
    ("    # full 28-head swap",
     "    # full attn swap (all heads via "
     "NQW coords)"),
    ("log('[%s] full 28-head swap recov=%+.4f '"
     "\n        '(forwards=%d)'"
     "\n        % (fkey, R_ALL, FW[0]))",
     "log('[%s] full attn swap recov=%+.4f '"
     "\n        '(forwards=%d)'"
     "\n        % (fkey, R_ALL, FW[0]))"),
]
for old, new in REPS:
    cnt = src.count(old)
    assert cnt == 1, \
        'rep count %d for %r' % (cnt, old[:60])
    src = src.replace(old, new)

# ---- 4. global subs ----
GSUBS = [
    ("third_", "fourth_"),
    ("seed 3085", "seed 3087"),
    ("(16 heads)", "(32 heads)"),
    ("'before any qwen2.5-3b '",
     "'before any glm4-9b '"),
]
for old, new in GSUBS:
    src = src.replace(old, new)

# ---- 5. bad residues (must be 0) ----
BAD = ['qwen2.5', 'z84', 'omega_p81',
       '151936', '11008', 'third_',
       "'pair k (16 heads)'", "'phase3084'",
       'vs 3084', "'qwen2.5-3b-instruct')",
       'PHASE = 3085', 'SEED = 3085',
       'SEED_MAIN = 3085', "'phase3085', NAME)",
       'NL, HID, KV_HEAD, NQ = 36, 2048, '
       '2, 16', '2048', '16 heads',
       'tie_word_embeddings=True']
if L_RES != 34:
    BAD += ['L_INJ = 34']
if L_POST != 35:
    BAD += ['L_POST = 35']
bad_report = []
for b in BAD:
    c = src.count(b)
    if c:
        bad_report.append((b, c))

# ---- 6. want counts ----
WANT = [('glm4-9b-chat-hf', 3),
        ('151552', 1),
        ('omega_p84_glm4_layer_scan', 2),
        ('fourth_', 30),
        ('32-head', 3),
        ('32 heads', 1),
        ('omega_p85_glm4_l%d_full_'
         'arbitration' % L_RES, 1),
        ('tie_word_embeddings=False', 1),
        ('GlmForCausalLM', 1),
        ('L_INJ = %d' % L_RES, 1),
        ('seed 3087', 2),
        ('3086', 2),
        ('sysmem', 2)]
want_report = []
for w, n in WANT:
    c = src.count(w)
    if c < n:
        want_report.append((w, c, n))

print('A1 verdict :', verdict)
for L in L_CAND:
    print('  L%d n_neg %s med_c %s R_ALL %s'
          % (L,
             '/'.join(str(v)
                      for v in stats[L][0]),
             '/'.join(fmt(v)
                      for v in stats[L][1]),
             '/'.join(fmt(v)
                      for v in stats[L][2])))
print('rescue layer: L%d (L_POST=%d)'
      % (L_RES, L_POST))
print('doc+prereg+reps done; chars %d -> %d'
      % (orig_len, len(src)))
print('bad residues:', bad_report
      if bad_report else 'ALL CLEAR')
print('want shortfall:', want_report
      if want_report else 'ALL OK')

if bad_report or want_report:
    print('GEN_FAILED - not written')
    raise SystemExit(1)

DST = DST_T.format(L=L_RES)
if CHECK:
    print('GEN_CHECK ok (not written)', DST)
    raise SystemExit(0)

with io.open(DST, 'w', encoding='utf-8') as f:
    f.write(src)
py_compile.compile(DST, doraise=True)
import hashlib
h = hashlib.sha256(
    io.open(DST, 'rb').read()).hexdigest()[:8]
print('WROTE', DST)
print('sha8', h, 'chars', len(src))
