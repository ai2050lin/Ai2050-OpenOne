# -*- coding: utf-8 -*-
"""Generate phase3083_omega_p80_third_model_arbitration.py
from the 3081 blueprint via asserted replacements.
Writes a report; never relies on stdout."""
import io
import os
import py_compile
import sys

ROOT = r'D:\AI2050\Ai2050-OpenOne'
SRC = os.path.join(
    ROOT, 'tests', 'glm5',
    'phase3081_omega_p78_ds7b_crossmodel.py')
DST = os.path.join(
    ROOT, 'tests', 'glm5',
    'phase3083_omega_p80_third_model_arbitration.py')
REP = os.path.join(
    ROOT, 'tests', 'gpt5_temp',
    'p3083_patch_report.txt')

src = io.open(SRC, encoding='utf-8').read()
rep = []
n_repl = 0


def sub1(old, new, tag):
    """Replace exactly one occurrence."""
    global src, n_repl
    c = src.count(old)
    assert c == 1, (tag, c)
    src = src.replace(old, new)
    n_repl += 1
    rep.append('OK  %s' % tag)


NEW_DOC = '''"""Phase 3083: Omega-P80 third-model
arbitration of the 3082 PR spectrum
diagnosis (qwen2.5-3b-instruct bf16,
single model).

Question (3083 A, menu of 3082): 3082
anatomized the DS7B migration absence into
a SPECTRAL split - qwen3-4b carries one
shared low-rank causal trunk per family
(CS column-centered SVD top3 0.957-0.964,
PR 1.3-2.0) while DS7B is high-rank
dispersed (top3 0.27-0.32, PR 13.6-17.1)
with random head-level focal overlap.
Preregistered falsifiable prediction:
trunk-type models should show the head
migration / cos lock; dispersed-type
should not.  qwen2.5-3b-instruct is the
arbitrator: Qwen2 architecture (same
family as the DS7B backbone) but
instruct-tuned (same tuning class as
qwen3-4b) and NOT an R1 distill - it
arbitrates whether DS7B was the outlier
(distill specialization) or qwen3-4b was.

Pipeline (3076/3081-identical protocol):
three frozen 3076 families A/B/C, 8 bodies
x 4 prefixes = 32 prompts each; V-banks +
last-position banks (LG/VB/PB/BX/BA/BM/BH);
b-anchors b0/b1/b3/b4/b6/b7a/b8; E1 repV
injection ladder at L_INJ (24 pairs); E3
per-head single-swap scan (16 heads x K3
pairs) -> r1; per-family focal top8 (3071
criterion); E4 ALL 255 non-empty subsets
swapped jointly + full 16-head swap.
NEW vs 3081: per-family causal spectrum
(3082 E3 bit-identical column-centered SVD
of CS and CS1H -> PR / k_eff / top3) is
computed in-run and preregistered as a
pre-check that combines with the migration
gate into the verdict.

Cross statistics (3079/3080 machinery,
seed 3083, 20000 perms): T/U responses x
f1-f5 predictors, 30 tests; G_DS main gate
(count>=4 AND min sp>0 over the 6 f2
tests); G1 f1 control; coupling
sp(U,T)/sp(f1,f2); partial sp(f2|f1),
sp(f5|f2); family orientation (n=3).

verdict (preregistered):
  setup/b-anchor failure ->
      third_setup_failed
  any family top8 degenerate (n_neg < 8)
      -> third_top8_degenerate
  spectrum class from CS top3 (thresholds
      in the 3082 gap): all top3 >= 0.9 ->
      trunk; all top3 <= 0.5 -> dispersed;
      else mixed
  migration state: G_DS AND f2~T_AB p<0.05
      -> locked; G_DS -> partial; else
      absent
  combined: trunk + locked/partial ->
      third_trunk_migrates; dispersed +
      locked/partial ->
      third_dispersed_migrates (PR
      diagnosis falsified); trunk + absent
      -> third_trunk_no_migrate;
      dispersed + absent ->
      third_dispersed_no_migrate (3082
      prediction holds, 4B-side outlier
      deepened); mixed ->
      third_mixed_<state>

arbitration reading (frozen before run):
  third_trunk_migrates        -> DS7B was
      the outlier (R1 distill
      specialization)
  third_dispersed_migrates    -> 3082 PR
      diagnostic falsified
  third_trunk_no_migrate      -> qwen3-4b
      specificity deepened
  third_dispersed_no_migrate  -> 3082
      prediction holds (4B-side trunk is
      the rare anatomy)

model adaptation (frozen before run):
qwen2.5-3b-instruct (Qwen2ForCausalLM,
hidden 2048, 36 layers, 16 heads, 2 kv
heads, inter 11008, vocab 151936, bf16,
6.18 GB); L_INJ=33, L_POST=34 (injection
at the 3rd-from-last layer V, observation
at the 2nd-from-last; layers 34-35
downstream; qwen3-4b used L34/L35 of 36,
DS7B used L25/L26 of 28); E2H/lens
machinery omitted as in 3081 (all cosines
from forward logits); same frozen 3076
texts.

limitations (recorded): qwen2.5-3b shares
the Qwen2 lineage with the DS7B backbone -
not a maximally independent model;
instruct-tuned like qwen3-4b; same 3076
texts; n=24 pairs per family pair; partial
spearman first-order on n=24 is
orientation evidence; causal-connective
paradigm, shared syntactic frame;
migration defined on the per-family focal
top-8 subset frame; spectrum
classification thresholds (0.9 / 0.5)
chosen in the 3082 empirical gap BEFORE
this run.

memory discipline: single model; families
sequential; big banks fp64 CPU freed after
each family (del + gc + empty_cache); no
W32, no Wo34/Wo35 (E2H omitted).
"""'''

NEW_PREREG = '''PREREG = {
    'mode': 'single model qwen2.5-3b-instruct '
            'bf16 eager seed 3083; three '
            'frozen 3076 families processed '
            'sequentially in one process '
            '(A, B, C; big banks freed between '
            'families); per-family protocol '
            '3081-identical minus E2H/lens (no '
            'W32 - all cosines from forward '
            'logits): repV 3065-identical E1 '
            'ladder at L_INJ=33 (24 pairs, '
            'med_c reference), 16-head single '
            'swap scan, per-family focal top8 '
            'by the 3071 signed criterion, ALL '
            '255 non-empty subsets swapped '
            'jointly (24 pairs each), full '
            '16-head swap; b-anchors b0/b1/b3/'
            'b4/b6/b7a/b8 (b5/b7c/a2lens '
            'omitted with the lens machinery); '
            'NEW in-run: per-family causal '
            'spectrum (3082 E3 bit-identical '
            'column-centered SVD of CS and '
            'CS1H: PR / k_eff / top3); smoke '
            'mode optional (SMOKE=1: 12 '
            'masks, K3=4 pairs, NP_USE=8, '
            'cross stats off)',
    'question': '3083 A (menu of 3082): 3082 '
                'anatomized the DS7B migration '
                'absence into a SPECTRAL split: '
                'qwen3-4b carries one shared '
                'low-rank causal trunk per '
                'family (CS top3 0.957-0.964) '
                'while DS7B is high-rank '
                'dispersed (top3 0.27-0.32) '
                'with random head-level focal '
                'overlap.  Preregistered '
                'falsifiable prediction: '
                'trunk-type models should show '
                'head migration / cos lock, '
                'dispersed-type should not.  '
                'Run the full 3081 protocol on '
                'qwen2.5-3b-instruct (Qwen2 '
                'architecture like the DS7B '
                'backbone, instruct-tuned like '
                'qwen3-4b, NOT an R1 distill) '
                'and combine the in-run spectrum '
                'class with the migration gate '
                'into the verdict.',
    'arbitration': 'third_trunk_migrates -> '
        'DS7B was the outlier (R1 distill '
        'specialization); '
        'third_dispersed_migrates -> 3082 PR '
        'diagnostic falsified; '
        'third_trunk_no_migrate -> qwen3-4b '
        'specificity deepened; '
        'third_dispersed_no_migrate -> 3082 '
        'prediction holds (4B-side trunk is '
        'the rare anatomy)',
    'independence': 'f2 values and CS spectra '
                    'on qwen2.5-3b are new data '
                    '(different model, new '
                    'forwards); the gate G_DS, '
                    'the spectrum thresholds '
                    '(0.9 / 0.5) and the '
                    'combined verdict tree were '
                    'fixed in this prereg before '
                    'any qwen2.5-3b observation; '
                    'family texts are the same '
                    'frozen 3076 texts - '
                    'independence is across '
                    'models, not texts',
    'families': {
        fk: {'domain': FAM[fk]['domain'],
             'bodies': list(FAM[fk]['bodies']),
             'targets': list(TARGETS),
             'prefixes': list(FAM[fk]
                              ['prefixes'])}
        for fk in FKEYS},
    'layer_mapping': 'L_INJ=33 (V injection + '
                     'attn swaps), L_POST=34 '
                     '(block-output continuity '
                     'check); 36-layer stack, '
                     'layers 34-35 downstream of '
                     'the injection; DS7B used '
                     'L25/L26 of 28 (3rd/2nd '
                     'from last), qwen3-4b used '
                     'L34/L35 of 36 (2nd from '
                     'last / last)',
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
                      'third_top8_degenerate '
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
            'before any qwen2.5-3b '
            'observation',
    },
    'definitions': {
        'T_fg[k]': 'sp(CS1H_f[:,k], CS1H_g[:,k]) '
                   'head-level migration of '
                   'pair k (16 heads)',
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
                'AND min(sp) > 0; seed 3083, '
                '20000 perms; Stouffer '
                'one-sided z and Bonferroni '
                '0.05/6 = 0.00833 reported',
    },
    'verdict': 'setup/b-anchor fail -> '
               'third_setup_failed; top8 '
               'degenerate -> '
               'third_top8_degenerate; '
               'spectrum class (frozen): all '
               'CS top3 >= 0.9 -> trunk, all '
               '<= 0.5 -> dispersed, else '
               'mixed; migration: G_DS AND '
               'f2~T_AB p<0.05 -> locked, '
               'G_DS -> partial, else absent; '
               'combined -> '
               'third_trunk_migrates / '
               'third_dispersed_migrates / '
               'third_trunk_no_migrate / '
               'third_dispersed_no_migrate / '
               'third_mixed_<state>',
    'statistics_discipline': 'manual 3077-'
        'series spearman for all statistics; '
        'two-sided permutation p (seed 3083, '
        'vectorized); TT/LG stored as fp32 '
        'copies and promoted to fp64 for the '
        'family comparisons (3079 data path, '
        'identical rounding for every '
        'comparison); partial spearman '
        'permuted on the response column '
        'only; family level n=3 orientation '
        'only; no post-hoc model changes',
    'limitations': 'qwen2.5-3b shares the '
        'Qwen2 lineage with the DS7B '
        'backbone (related architecture '
        'family, not maximally '
        'independent); instruct-tuned like '
        'qwen3-4b; same 3076 texts (model '
        'independence, not text '
        'independence); n=24 pairs per '
        'family pair; first-order partial '
        'on n=24 is orientation only; '
        'causal-connective paradigm, '
        'shared syntactic frame; focal '
        'top-8 subset frame; spectrum '
        'classification thresholds (0.9 / '
        '0.5) sit in the 3082 gap but are '
        'two-point calibrated',
    'memory_discipline': 'single model; '
        'families sequential with big banks '
        'freed between families; no W32 / '
        'Wo34 / Wo35 (E2H and lens probes '
        'omitted); del + gc + empty_cache '
        'at family end and run end',
}

'''

SPECTRUM_FN = '''

def spectrum_struct(M):
    """Column-centered SVD structure of M
    (3082 E3 bit-identical)."""
    M = np.asarray(M, dtype=np.float64)
    Mc = M - M.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Mc,
                      compute_uv=False)
    lam = s * s
    tot = float(lam.sum())
    pr = float(lam.sum() ** 2
               / (lam * lam).sum())
    p = lam / tot
    p = p[p > 0]
    H = float(-(p * np.log(p)).sum())
    keff = float(np.exp(H))
    top3 = float(lam[:3].sum() / tot)
    return pr, keff, top3
'''

SPEC_BLOCK = '''

# ==== per-family causal spectrum (3082 E3
# bit-identical: column-centered SVD) ====
SPEC = {}
SPEC1H = {}
for fk in FKEYS:
    pr, keff, t3 = spectrum_struct(
        RES_F[fk]['CS'])
    SPEC[fk] = {'pr': pr, 'keff': keff,
                'top3': t3}
    prh, keffh, t3h = spectrum_struct(
        RES_F[fk]['CS1H'])
    SPEC1H[fk] = {'pr': prh,
                  'keff': keffh,
                  'top3': t3h}
    log('spectrum[%s]: CS PR=%.2f k_eff='
        '%.2f top3=%.4f | CS1H PR=%.2f '
        'k_eff=%.2f top3=%.4f'
        % (fk, pr, keff, t3, prh, keffh,
           t3h))
t3s = [SPEC[fk]['top3'] for fk in FKEYS]
if min(t3s) >= 0.9:
    spec_class = 'trunk'
elif max(t3s) <= 0.5:
    spec_class = 'dispersed'
else:
    spec_class = 'mixed'
log('spec_class=%s (CS top3: %s)'
    % (spec_class,
       ' '.join('%s=%.4f' % (fk, t)
                for fk, t in zip(FKEYS,
                                 t3s))))
'''

# ---- 1. docstring wholesale ----
i1 = src.index('"""Phase 3081')
i2 = src.index('"""', i1 + 10) + 3
src = src[:i1] + NEW_DOC + src[i2:]
n_repl += 1
rep.append('OK  docstring wholesale')

# ---- 2. PREREG wholesale ----
j1 = src.index('PREREG = {')
j2 = src.index('os.makedirs(OUT, exist_ok=True)')
src = src[:j1] + NEW_PREREG + src[j2:]
n_repl += 1
rep.append('OK  PREREG wholesale')

# ---- 3. constants ----
sub1("PHASE = 3081", "PHASE = 3083", "PHASE")
sub1("NAME = 'omega_p78_ds7b_crossmodel'",
     "NAME = 'omega_p80_third_model_arbitration'",
     "NAME")
sub1("'phase3081', NAME)",
     "'phase3083', NAME)", "OUT dir")
sub1("                    "
     "'deepseek-r1-distill-qwen-7b')",
     "                    "
     "'qwen2.5-3b-instruct')", "MDIR")
sub1("NL, HID, KV_HEAD, NQ = 28, 3584, 4, 28",
     "NL, HID, KV_HEAD, NQ = 36, 2048, 2, 16",
     "NL/HID/KV/NQ")
sub1("SEED_MAIN = 3081", "SEED_MAIN = 3083",
     "SEED_MAIN")
sub1("L_INJ = 25", "L_INJ = 33", "L_INJ")
sub1("L_POST = 26", "L_POST = 34", "L_POST")
sub1("SEED = 3081", "SEED = 3083", "SEED")
sub1("assert NVOC == 152064",
     "assert NVOC == 151936", "NVOC assert")
sub1("log('ds7b (deepseek-r1-distill-qwen-7b) "
     "loaded '",
     "log('qwen2.5-3b (qwen2.5-3b-instruct) "
     "loaded '", "model log line")
sub1("'run': 'run1 authoritative (ds7b bf16 '",
     "'run': 'run1 authoritative (qwen2.5-3b "
     "bf16 '", "run label")

# ---- 4. spectrum_struct function insert ----
anchor = "\n\n\n# ==== load model ===="
c = src.count(anchor)
assert c == 1, ('spectrum fn anchor', c)
src = src.replace(
    anchor, SPECTRUM_FN
    + "\n\n\n# ==== load model ====")
n_repl += 1
rep.append('OK  spectrum_struct fn insert')

# ---- 5. SPEC block after RES_F loop ----
anchor2 = ("RES_F = {}\n"
           "for fk in FKEYS:\n"
           "    RES_F[fk] = run_family(fk)\n"
           "    gc.collect()\n"
           "    torch.cuda.empty_cache()\n")
c = src.count(anchor2)
assert c == 1, ('RES_F loop anchor', c)
src = src.replace(anchor2, anchor2 + SPEC_BLOCK)
n_repl += 1
rep.append('OK  SPEC block insert')

# ---- 6. verdict block ----
OLD_VERDICT = """        # verdict
        p_f2_TAB = E2['T_AB']['p']
        sp_f2_TAB = E2['T_AB']['sp']
        if g_ds and p_f2_TAB < 0.05:
            verdict = 'ds7b_cos_locked'
        elif g_ds:
            verdict = 'ds7b_cos_partial'
        else:
            verdict = 'ds7b_cos_absent'
"""
NEW_VERDICT = """        # verdict (combined: migration state
        # x 3082 PR spectrum class)
        p_f2_TAB = E2['T_AB']['p']
        sp_f2_TAB = E2['T_AB']['sp']
        if g_ds and p_f2_TAB < 0.05:
            mig_state = 'locked'
        elif g_ds:
            mig_state = 'partial'
        else:
            mig_state = 'absent'
        if spec_class == 'mixed':
            verdict = 'third_mixed_' + mig_state
        elif mig_state == 'absent':
            verdict = ('third_trunk_no_migrate'
                       if spec_class == 'trunk'
                       else
                       'third_dispersed_no_'
                       'migrate')
        else:
            verdict = ('third_trunk_migrates'
                       if spec_class == 'trunk'
                       else
                       'third_dispersed_'
                       'migrates')
"""
sub1(OLD_VERDICT, NEW_VERDICT, "verdict block")

# ---- 7. VERDICT log line ----
sub1("""        log('VERDICT: %s (G_DS=%s count=%d/6 '
            'min_sp=%.4f stouffer=%.3f; '
            'f2~T_AB sp=%+.4f p=%.5f; '
            'G1=%s best f1 sp=%+.4f)'
            % (verdict, g_ds, cnt_pos,
               min_sp2, stouffer,
               sp_f2_TAB, p_f2_TAB, g1,
               best[1]))""",
     """        log('VERDICT: %s (spec=%s G_DS=%s '
            'count=%d/6 min_sp=%.4f '
            'stouffer=%.3f; f2~T_AB sp='
            '%+.4f p=%.5f; G1=%s best f1 '
            'sp=%+.4f)'
            % (verdict, spec_class, g_ds,
               cnt_pos, min_sp2, stouffer,
               sp_f2_TAB, p_f2_TAB, g1,
               best[1]))""",
     "VERDICT log")

# ---- 8. gates dict ----
sub1("""                 'G1_pair': best[4],
                 'f1_sign_positive': f1_pos}""",
     """                 'G1_pair': best[4],
                 'f1_sign_positive': f1_pos,
                 'spec_class': spec_class,
                 'spectrum': {
                     fk: dict(SPEC[fk])
                     for fk in FKEYS}}""",
     "gates dict")

# ---- 9. npz initial save dict ----
sub1("""    'L_INJ': np.int64(L_INJ),
    'L_POST': np.int64(L_POST),
}""",
     """    'L_INJ': np.int64(L_INJ),
    'L_POST': np.int64(L_POST),
    'SPEC_CLASS': np.array(spec_class),
}""",
     "npz SPEC_CLASS")

# ---- 10. npz per-family spectrum keys ----
sub1("""    save['MED_C_' + fk] = np.float64(
        R['med_c'])""",
     """    save['MED_C_' + fk] = np.float64(
        R['med_c'])
    save['E3_PR_CS_' + fk] = np.float64(
        SPEC[fk]['pr'])
    save['E3_KEFF_CS_' + fk] = np.float64(
        SPEC[fk]['keff'])
    save['E3_TOP3_CS_' + fk] = np.float64(
        SPEC[fk]['top3'])
    save['E3_PR_CS1H_' + fk] = np.float64(
        SPEC1H[fk]['pr'])
    save['E3_KEFF_CS1H_' + fk] = \\
        np.float64(SPEC1H[fk]['keff'])
    save['E3_TOP3_CS1H_' + fk] = \\
        np.float64(SPEC1H[fk]['top3'])""",
     "npz spectrum keys")

# ---- 11. result cross additions ----
sub1("""            'sp_fam_as_mig': f64(sp_fam_as),
            'sp_fam_tt_mig': f64(sp_fam_tt),""",
     """            'sp_fam_as_mig': f64(sp_fam_as),
            'sp_fam_tt_mig': f64(sp_fam_tt),
            'spec_class': spec_class,
            'spectrum': {
                fk: {kk: f64(vv) for kk, vv
                     in SPEC[fk].items()}
                for fk in FKEYS},""",
     "result cross spectrum")

# ---- 12. legacy key/log renames ----
sub1("save['R1_ALL28_' + fk]",
     "save['R1_ALLNH_' + fk]", "R1_ALLNH key")
sub1("E3 scan r1_all28: min=%.4f ",
     "E3 scan r1_allnh: min=%.4f ",
     "r1 log label")

# ---- 13. leftover verdict strings ----
for old, new, tag in (
        ('ds7b_setup_failed',
         'third_setup_failed', 'setup_failed'),
        ('ds7b_top8_degenerate',
         'third_top8_degenerate',
         'top8_degenerate')):
    c = src.count(old)
    assert c == 3, (tag, c)
    src = src.replace(old, new)
    n_repl += 1
    rep.append('OK  %s (x%d)' % (tag, c))

# ---- final sanity scan ----
for bad in ('ds7b', 'deepseek', '152064',
            'omega_p78', 'phase3081',
            'L_INJ = 25', 'L_POST = 26'):
    if bad in src:
        idxs = []
        i = src.find(bad)
        while i >= 0 and len(idxs) < 5:
            ln = src.count('\n', 0, i) + 1
            idxs.append(ln)
            i = src.find(bad, i + 1)
        raise AssertionError(
            'BAD substring %r at lines %s'
            % (bad, idxs))
rep.append('SANITY no stale substrings')

counts = {
    'spectrum_struct': src.count(
        'spectrum_struct'),
    'third_setup_failed': src.count(
        'third_setup_failed'),
    'third_top8_degenerate': src.count(
        'third_top8_degenerate'),
    'SPEC_CLASS': src.count('SPEC_CLASS'),
    'E3_PR_CS_': src.count('E3_PR_CS_'),
    'spec_class': src.count('spec_class'),
}
rep.append('COUNTS %s' % counts)
assert counts['spectrum_struct'] == 4
assert counts['third_setup_failed'] == 5
assert counts['third_top8_degenerate'] == 6
assert counts['SPEC_CLASS'] == 1
assert counts['E3_PR_CS_'] == 1

io.open(DST, 'w', encoding='utf-8',
        newline='\n').write(src)
rep.append('WROTE %s (%d chars)' % (DST,
                                    len(src)))

py_compile.compile(DST, doraise=True)
rep.append('PY_COMPILE OK')
rep.append('TOTAL replacements %d' % n_repl)

io.open(REP, 'w', encoding='utf-8').write(
    '\n'.join(rep) + '\n')
print('PATCH_OK')
