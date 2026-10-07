# -*- coding: utf-8 -*-
"""p3089 generator: phase3087_omega_p85_glm4_l37
-> phase3089_omega_p87_glm4_l38_full_arbitration.

Order matters: GSUBS on the code body FIRST
(fourth_ -> fourth_l38_ is single-pass; new
inserted texts already carry final names),
then docstring rewrite, then PREREG block
rewrite, then targeted REPS, then BAD/WANT,
then compile.  Output log -> p3089_patch_out.txt
"""
import io
import os

SRC = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
       r'\phase3087_omega_p85_glm4_l37_full_'
       r'arbitration.py')
DST = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
       r'\phase3089_omega_p87_glm4_l38_full_'
       r'arbitration.py')
OUTF = (r'D:\AI2050\Ai2050-OpenOne\tests'
        r'\gpt5_temp\p3089_patch_out.txt')
o = []
ok = True

s = io.open(SRC, encoding='utf-8').read()
parts = s.split('"""', 2)
assert len(parts) == 3, len(parts)
assert parts[1].lstrip().startswith(
    'Phase 3087'), parts[1][:40]
head, body = parts[0], parts[2]

# ---- 1. GSUBS on body ----
GSUBS = [('fourth_', 'fourth_l38_'),
         ('omega_p85', 'omega_p87')]
for old, new in GSUBS:
    c = body.count(old)
    o.append('GSUB %-12r count=%d' % (old, c))
    body = body.replace(old, new)

# ---- 2. targeted REPS ----
REPS = [
    ('PHASE = 3087', 'PHASE = 3089'),
    ('glm4_l37_full_arbitration',
     'glm4_l38_full_arbitration'),
    ("    'phase3087', NAME)",
     "    'phase3089', NAME)"),
    ('SEED_MAIN = 3087', 'SEED_MAIN = 3089'),
    ('SEED = 3087', 'SEED = 3089'),
    ('L_INJ = 37', 'L_INJ = 38'),
    ('L_POST = 38', 'L_POST = 39'),
    ("'MED_C_L37_'", "'MED_C_L38_'"),
    ("'N_NEG_L37_'", "'N_NEG_L38_'"),
    ("'TOP8_L37_'", "'TOP8_L38_'"),
    ("'CS1H_L37_'", "'CS1H_L38_'"),
    ("'R_ALL_L37_'", "'R_ALL_L38_'"),
    ('vs 3087A1 L37: ', 'vs 3087A1 L38: '),
    ('# ==== L37 deterministic reproduction',
     '# ==== L38 deterministic reproduction'),
]
for old, new in REPS:
    c = body.count(old)
    if c != 1:
        ok = False
        o.append('REP %-40r count=%d EXPECT 1 '
                 'ABORT' % (old, c))
        continue
    body = body.replace(old, new)
    o.append('REP %-40r count=1 OK' % old)

# ---- 3. docstring rewrite ----
NEW_DOC = u'''
Phase 3089: Omega-P87 full-pipeline
four-way arbitration at L_INJ=38 on
glm4-9b-chat-hf (L38 replica - the
criterion-split discriminant from the
3087 A1 layer scan).

Question (3089 A, menu of 3088): the 3087
A1 layer scan found the rescue band
L34/37/38 with a CRITERION SPLIT: the
n_neg criterion selected L37 (n_neg 12/
24/14) while the E1 magnitude criterion
selected L38 (med_c 0.192/0.200/0.132,
R_ALL -0.177/-0.120/-0.125, 4-6x
stronger).  3087 A2 ran the arbitration
at L37 (fourth_mixed_absent, same cell
as 3085) and 3088 built the n=12
continuum on the L37 GLM4 units.  This
phase reruns the IDENTICAL pipeline at
L_INJ=38 to test whether the GLM4
four-way verdict and the continuum unit
values are robust to the layer-choice
criterion: same cell -> criterion split
closed, GLM4 conclusion layer-robust;
different cell -> layer sensitivity
demonstrated, and 3089 B re-runs the
continuum with L38 units (forward-free).

Pipeline (3087-identical except L_INJ
and the repro-anchor layer): three
frozen 3076 families A/B/C, 8 bodies x 4
prefixes = 32 prompts each; V-banks +
last-position banks (LG/VB/PB/BX/BA/BM/
BH); b-anchors b0/b1/b3/b4/b6/b7a/b8; E1
repV injection ladder at L_INJ=38 (24
pairs); E3 per-head single-swap scan
(32 heads x 24 pairs) -> r1; per-family
focal top8 (3071 criterion); E4 ALL 255
non-empty subsets swapped jointly + full
32-head swap; in-run spectrum (3082 E3
bit-identical column-centered SVD -> PR /
k_eff / top3); cross statistics (seed
3089, 20000 perms); G_DS main gate
(count>=4 AND min sp>0 over the 6 f2
tests); Stouffer z and Bonferroni
reported.

NEW vs 3087: L38 deterministic
reproduction anchor vs the SAME 3087 A1
layer-scan npz (L38 keys; same model,
same frozen texts, same layer): n_neg
and top8 must match EXACTLY and med_c /
CS1H / R_ALL within 1e-9; mismatch ->
setup failure; smoke skips the anchor.

verdict (preregistered, tree unchanged
from 3083/3085/3087, names suffixed
_l38):  setup/b-anchor/repro failure ->
fourth_l38_setup_failed;  any family
top8 degenerate (n_neg < 8) ->
fourth_l38_top8_degenerate;  spectrum
class from CS top3 (thresholds in the
3082 gap): all top3 >= 0.9 -> trunk;
all top3 <= 0.5 -> dispersed; else
mixed;  migration state: G_DS AND
f2~T_AB p<0.05 -> locked; G_DS ->
partial; else absent;  combined:
trunk + locked/partial ->
fourth_l38_trunk_migrates; dispersed +
locked/partial -> fourth_l38_dispersed_
migrates (PR diagnosis falsified);
trunk + absent -> fourth_l38_trunk_no_
migrate; dispersed + absent ->
fourth_l38_dispersed_no_migrate (3082
prediction holds); mixed ->
fourth_l38_mixed_<state>

arbitration reading (frozen):
cell == the 3087 L37 cell
(fourth_l38_mixed_absent) -> the
criterion split is closed; the GLM4
conclusion and the 3088 n=12 continuum
verdict are layer-robust;
any migrate cell -> locked/partial
migration appears at the
magnitude-deep layer (layer-sensitive
loading; the continuum needs the L38
re-run);
fourth_l38_top8_degenerate -> the L38
top8 selection degrades under the full
pipeline (A1 descriptives did not
transfer).

model adaptation (frozen before run):
glm4-9b-chat-hf (GlmForCausalLM, hidden
4096, 40 layers, 32 heads, 2 kv heads,
head_dim 128, inter 13696, vocab 151552,
bf16 ~18.8 GB allocated, verified
non-OOM at seq<=74 via driver sysmem
fallback); L_INJ=38, L_POST=39
(injection at the layer-38 V,
observation at the next layer; layer 39
downstream; L38 selected by the E1
magnitude criterion of the 3087 A1
scan); E2H/lens machinery omitted as in
3081/3083/3085/3087 (all cosines from
forward logits); input embeddings are
UNTIED (tie_word_embeddings=False);
same frozen 3076 texts.

limitations (recorded): L_INJ=38 chosen
by the E1 magnitude criterion from the
3087 A1 descriptive layer scan (L38
n_neg 15/12/10 med_c 0.19/0.20/0.13
R_ALL -0.18/-0.12/-0.12 vs L37 n_neg
12/24/14 med_c 0.04/0.03/0.02 R_ALL
-0.05/0.05/-0.04); L38 has the WEAKEST
min-family n_neg (10) of the rescue
band - the top8-degenerate exit is a
live possibility and is itself
informative; this is a REPLICA phase:
the verdict space and gates were fixed
in 3087 and the A1 L38 descriptives
were known before the run (test of
robustness, not blind discovery); glm4-
9b shares no lineage with the DS7B
backbone or the Qwen-family models;
chat-tuned like the other instruct
models; same 3076 texts; n=24 pairs per
family pair; partial spearman
first-order on n=24 is orientation
evidence; causal-connective paradigm,
shared syntactic frame; migration
defined on the per-family focal top-8
subset frame; spectrum classification
thresholds (0.9 / 0.5) chosen in the
3082 empirical gap.

memory discipline: single model; families
sequential; big banks fp64 CPU freed after
each family (del + gc + empty_cache); no
W32, no Wo38/Wo39 (E2H omitted).
'''
parts[1] = NEW_DOC

# ---- 4. PREREG block rewrite ----
NEW_PREREG = '''PREREG = {
    'mode': 'single model glm4-9b-chat-hf '
            'bf16 eager seed 3089; three '
            'frozen 3076 families processed '
            'sequentially in one process '
            '(A, B, C; big banks freed between '
            'families); per-family protocol '
            '3081-identical minus E2H/lens (no '
            'W32 - all cosines from forward '
            'logits): repV 3065-identical E1 '
            'ladder at L_INJ=38 (24 pairs, '
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
            'L38 deterministic reproduction '
            'anchor vs the 3087 A1 layer-scan '
            'npz L38 keys (n_neg/top8 exact; '
            'med_c/CS1H/R_ALL within 1e-9; '
            'feeds setup_ok; smoke skips); '
            'smoke mode optional (SMOKE=1: 12 '
            'masks, K3=4 pairs, NP_USE=8, '
            'cross stats off)',
    'question': '3089 A (menu of 3088): the '
                '3087 A1 layer scan found the '
                'GLM4 rescue band L34/37/38 '
                'with a criterion split - n_neg '
                'criterion selected L37 (12/24/'
                '14) vs E1 magnitude criterion '
                'selecting L38 (med_c 0.19/0.20/'
                '0.13, R_ALL -0.18/-0.12/-0.12, '
                '4-6x).  Rerun the full 3083/'
                '3085/3087 pipeline at L_INJ=38 '
                'and combine the in-run spectrum '
                'class with the migration gate '
                'into the four-way verdict: same '
                'cell as L37 -> criterion split '
                'closed and the 3088 continuum '
                'verdict is layer-robust; '
                'different cell -> layer '
                'sensitivity demonstrated and '
                'the continuum is re-run with '
                'L38 units (forward-free).',
    'arbitration': 'fourth_l38_mixed_absent '
        '(== 3087 L37 cell) -> criterion '
        'split closed, GLM4 conclusion '
        'layer-robust; '
        'fourth_l38_trunk_migrates -> trunk '
        'anatomy with locked head migration '
        'generalizes beyond the Qwen lineage '
        'at the magnitude layer (DS7B outlier '
        'reading withdrawn); '
        'fourth_l38_dispersed_migrates -> '
        '3082 PR diagnostic falsified; '
        'fourth_l38_trunk_no_migrate -> '
        'qwen3-4b specificity deepened; '
        'fourth_l38_dispersed_no_migrate -> '
        '3082 prediction holds',
    'independence': 'f2 values and CS spectra '
                    'at L38 are new data (new '
                    'forwards, seed 3089); the '
                    'gate G_DS, the spectrum '
                    'thresholds (0.9 / 0.5) and '
                    'the combined verdict tree '
                    'were fixed in 3087 before '
                    'any L38 observation; 3087 '
                    'A1 priors (n_neg/med_c/'
                    'R_ALL descriptives at BOTH '
                    'L37 and L38) guided the '
                    'LAYER CHOICE ONLY - all '
                    'verdict inputs (f2/T/U, '
                    'spectrum class, gates) are '
                    'computed fresh in-run; '
                    'family texts are the same '
                    'frozen 3076 texts',
    'repro_anchor': 'activation collection is '
                    'deterministic: same model, '
                    'same frozen texts, same '
                    'layer -> L38 values must '
                    'match the 3087 A1 npz L38 '
                    'keys per family (n_neg '
                    'exact, top8 exact, med_c/'
                    'CS1H/R_ALL abs diff <= 1e-9); '
                    'the 1e-9 tolerance covers '
                    'the fp32-roundtrip noise '
                    'class measured by the '
                    '3083-vs-3084 L33 probe (max '
                    '5.8e-13 med_c); mismatch -> '
                    'fourth_l38_setup_failed; '
                    'smoke skips the anchor '
                    '(K3/NP_USE differ)',
    'families': {
        fk: {'domain': FAM[fk]['domain'],
             'bodies': list(FAM[fk]['bodies']),
             'targets': list(TARGETS),
             'prefixes': list(FAM[fk]
                              ['prefixes'])}
        for fk in FKEYS},
    'layer_mapping': 'L_INJ=38 (V injection + '
                     'attn swaps), L_POST=39 '
                     '(block-output continuity '
                     'check); 40-layer stack, '
                     'layer 39 downstream of '
                     'the injection; selected by '
                     'the E1 magnitude criterion '
                     'on the 3087 A1 scan '
                     '(L31/34/37/38 equal-ratio '
                     'mapping of the 3084 3B grid: '
                     'L31 n_neg 14/4/19 med_c '
                     '0.08/0.11/0.08; L34 n_neg '
                     '14/9/19 med_c 0.05/0.06/'
                     '0.08; L37 n_neg 12/24/14 '
                     'med_c 0.04/0.03/0.02 R_ALL '
                     '-0.05/0.05/-0.04; L38 n_neg '
                     '15/12/10 med_c 0.19/0.20/'
                     '0.13 R_ALL -0.18/-0.12/'
                     '-0.12); L37 was the n_neg-'
                     'criterion choice (3087); '
                     'L38 is the magnitude-'
                     'criterion choice (this '
                     'phase)',
    'top8_criterion': 'per family (3071-'
                      'identical): order = '
                      'np.argsort(r1) ASCENDING '
                      '(most negative first), '
                      'topk = min(8, n_neg), '
                      'top8 = order[:topk]; '
                      'top8_sel_ok = (topk == 8 '
                      'and all r1[top8] < 0); '
                      'if any family degenerate '
                      '(n_neg < 8) the cross '
                      'section is skipped and '
                      'the verdict is '
                      'fourth_l38_top8_degenerate '
                      '(frozen before the run)',
    'spectrum': {
        'definition': 'per family: '
            'spectrum_struct(CS) with CS the '
            '(n_masks, 24) cosine-similarity '
            'matrix of the E4 subset sweep; '
            'column-centered SVD; PR = '
            '(sum lam)^2 / sum lam^2, k_eff '
            '= exp(entropy of lam/tot), top3 '
            '= sum(lam[:3])/tot (3082 E3 '
            'bit-identical); CS1H spectrum '
            'recorded in parallel',
        'classification': 'trunk if min over '
            'the 3 families of CS top3 >= '
            '0.9; dispersed if max <= 0.5; '
            'else mixed; thresholds chosen '
            'in the 3082 empirical gap (4B '
            '0.957-0.964 vs DS7B 0.27-0.32) '
            'before any glm4-9b '
            'observation',
    },
    'definitions': {
        'T_fg[k]': 'sp(CS1H_f[:,k], CS1H_g[:,k]) '
                   'head-level migration of '
                   'pair k (32 heads)',
        'U_fg[k]': 'sp(CS_f[:,k], CS_g[:,k]) '
                   'subset-level migration '
                   '(255 subsets)',
        'f1_sTT[k]': 'sp(TT_f[k], TT_g[k]) '
                     'rank shape (3079 main, '
                     'control here)',
        'f2_cTT[k]': 'cos(TT_f[k], TT_g[k]) '
                     'direction angle (3080 '
                     'locked carrier, MAIN '
                     'here)',
        'f3/f4': 'logits shape similarity at '
                 'pref/base (controls)',
        'f5_amp[k]': 'TT norm ratio control',
        'mig_fg': 'sp(R1_f, R1_g) family '
                  'migration reference '
                  '(3079 SP_R1 analog)',
        'partial': 'first-order spearman '
                   'partial (pearson on ranks)',
    },
    'anchors': {
        'b0': 'bank recapture (4 prompts) bit '
              '0.0 per family',
        'b1': 'sham self-V-replacement bit '
              '0.0 per family',
        'b3': 'all banks finite',
        'b4': 'delta-x at L_INJ bit 0.0 '
              '(injection must not move the '
              'layer input)',
        'b6': '(x+a)+m=h2 bf16 identity per '
              'family',
        'b7a': 'identity attn self-swap '
               '(all heads, L_INJ) bit 0.0',
        'b8': 'block-output continuity '
              'zX[L_POST] == zP[L_INJ] bit '
              '0.0',
    },
    'gates': {
        'G1': 'f1 control (3079-aligned): max '
              'over (pair, response) of '
              'sp(f1, resp) >= 0.5 AND p < '
              '0.05; recorded, NOT gating',
        'G_DS': 'MAIN (3080 G2-aligned, f2): '
                'count over the 6 f2 tests of '
                '(sp > 0 AND p < 0.05) >= 4 '
                'AND min(sp) > 0; seed 3089, '
                '20000 perms; Stouffer '
                'one-sided z and Bonferroni '
                '0.05/6 = 0.00833 reported',
    },
    'verdict': 'setup/b-anchor fail -> '
               'fourth_l38_setup_failed; top8 '
               'degenerate -> '
               'fourth_l38_top8_degenerate; '
               'spectrum class (frozen): all '
               'CS top3 >= 0.9 -> trunk, all '
               '<= 0.5 -> dispersed, else '
               'mixed; migration: G_DS AND '
               'f2~T_AB p<0.05 -> locked, '
               'G_DS -> partial, else absent; '
               'combined -> '
               'fourth_l38_trunk_migrates / '
               'fourth_l38_dispersed_migrates '
               '/ fourth_l38_trunk_no_migrate '
               '/ fourth_l38_dispersed_no_'
               'migrate / '
               'fourth_l38_mixed_<state>',
    'statistics_discipline': 'manual 3077-'
        'series spearman for all statistics; '
        'two-sided permutation p (seed 3089, '
        'vectorized); TT/LG stored as fp32 '
        'copies and promoted to fp64 for the '
        'family comparisons (3079 data path, '
        'identical rounding for every '
        'comparison); partial spearman '
        'permuted on the response column '
        'only; family level n=3 orientation '
        'only; no post-hoc model changes',
    'limitations': 'REPLICA phase: verdict '
        'space and gates fixed in 3087; A1 '
        'L38 descriptives known before the '
        'run (robustness test, not blind '
        'discovery); L38 has the weakest '
        'min-family n_neg (10) of the rescue '
        'band - the top8-degenerate exit is '
        'a live possibility and itself '
        'informative; glm4-9b shares no '
        'lineage with the DS7B backbone or '
        'the Qwen-family models; chat-tuned '
        'like the other instruct models; '
        'same 3076 texts (model independence, '
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
        'unaffected; bf16 weights exceed '
        'VRAM by ~1.7 GB and rely on the '
        'verified driver sysmem fallback '
        '(non-OOM at seq<=74 in the loading '
        'probe; identical behavior recorded '
        'in A1 and 3087 A2)',
    'memory_discipline': 'single model; '
        'families sequential with big banks '
        'freed between families; no W32 / '
        'Wo38 / Wo39 (E2H and lens probes '
        'omitted); del + gc + empty_cache '
        'at family end and run end',

}'''
i = body.index('PREREG = {')
j = body.index('\nos.makedirs', i)
body = body[:i] + NEW_PREREG + body[j:]
o.append('PREREG block rewritten (chars %d)'
         % len(NEW_PREREG))

final = head + '"""' + parts[1] + '"""' + body

# ---- 5. BAD tokens ----
BAD = ["PHASE = 3087", "omega_p85",
       "l37_full", "'phase3087', NAME)",
       "SEED = 3087", "SEED_MAIN = 3087",
       "L_INJ = 37", "L_POST = 38",
       "'MED_C_L37_'", "'N_NEG_L37_'",
       "'TOP8_L37_'", "'CS1H_L37_'",
       "'R_ALL_L37_'", "vs 3087A1 L37",
       "'fourth_mixed_' + mig_state",
       "at L_INJ=37", "seed 3087",
       "no Wo37/Wo38", "'L37_' + fk"]
for tok in BAD:
    c = final.count(tok)
    good = (c == 0)
    ok = ok and good
    o.append('BAD %-38r count=%d %s'
             % (tok, c,
                'OK' if good else 'VIOLATION'))

# ---- 6. WANT tokens ----
WANT = [("PHASE = 3089", 1),
        ("omega_p87_glm4_l38_full_arbitration",
         1),
        ("'phase3089', NAME)", 1),
        ("L_INJ = 38", 1),
        ("L_POST = 39", 1),
        ("SEED = 3089", 1),
        ("SEED_MAIN = 3089", 1),
        ("'MED_C_L38_'", 1),
        ("'N_NEG_L38_'", 1),
        ("'TOP8_L38_'", 1),
        ("'CS1H_L38_'", 1),
        ("'R_ALL_L38_'", 1),
        ("vs 3087A1 L38", 1),
        ("fourth_l38_", None),
        ("fourth_l38_mixed_", None),
        ("'omega_p84_glm4_layer_scan'", 1),
        ("'omega_p84_glm4_layer_scan.npz'", 1),
        ("'phase3087'", 1),
        ("seed 3089", None),
        ("L_INJ=38", None),
        ("Wo38 / Wo39", 1),
        ("L_POST=39", None)]
for tok, want in WANT:
    c = final.count(tok)
    if want is None:
        good = c >= 2
    else:
        good = (c == want)
    ok = ok and good
    o.append('WANT %-42r count=%d want=%s %s'
             % (tok, c, want,
                'OK' if good else 'MISMATCH'))

# ---- 7. compile ----
try:
    compile(final, DST, 'exec')
    o.append('compile OK')
except SyntaxError as e:
    ok = False
    o.append('compile FAIL: %r' % e)

if ok:
    with io.open(DST, 'w',
                 encoding='utf-8') as f:
        f.write(final)
    o.append('written %s (%d chars)'
             % (DST, len(final)))
o.append('PATCH_%s' % ('PASS' if ok else 'FAIL'))
io.open(OUTF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
