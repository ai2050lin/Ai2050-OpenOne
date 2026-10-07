# -*- coding: utf-8 -*-
"""Phase 3085: Omega-P82 full-pipeline
four-way arbitration at L_INJ=34 on
qwen2.5-3b-instruct (the 3083 arbitration
re-opened at the 3084-confirmed loading
layer).

Question (3085 A, menu of 3084): the 3084
layer scan showed that the 3083 family-C
degeneration (n_neg=2/16 at L_INJ=33) is a
LAYER-POSITION effect: L28 and L34 rescue
all three families, and L34 (2nd from last,
the qwen3-4b protocol position) is the
strongest layer (n_neg 12/12/11, med_c
0.22-0.36, R_ALL -0.23..-0.30).  The 3083
four-way trunk/dispersed x migrates
arbitration was postponed by the L33
degeneration; it is re-opened here by
rerunning the FULL 3083 pipeline (E1 repV
ladder + E3 per-head scan + focal top8 + E4
255-subset sweep + spectrum_struct +
G_DS/f2~T_AB migration gate + cross
statistics) at L_INJ=34.

Pipeline (3083-identical except L_INJ and
the new repro anchor): three frozen 3076
families A/B/C, 8 bodies x 4 prefixes = 32
prompts each; V-banks + last-position banks
(LG/VB/PB/BX/BA/BM/BH); b-anchors
b0/b1/b3/b4/b6/b7a/b8; E1 repV injection
ladder at L_INJ=34 (24 pairs); E3 per-head
single-swap scan (16 heads x 24 pairs) ->
r1; per-family focal top8 (3071 criterion);
E4 ALL 255 non-empty subsets swapped
jointly + full 16-head swap; in-run
spectrum (3082 E3 bit-identical
column-centered SVD -> PR / k_eff / top3);
cross statistics (seed 3085, 20000 perms);
G_DS main gate (count>=4 AND min sp>0 over
the 6 f2 tests); Stouffer z and Bonferroni
reported.

NEW vs 3083: L34 deterministic reproduction
anchor vs the 3084 layer-scan npz (same
model, same frozen texts, same layer):
n_neg and top8 must match EXACTLY and
med_c / CS1H / R_ALL within 1e-9 (the
fp32-roundtrip noise measured by the
3083-vs-3084 L33 probe is <= 3e-12;
tolerance is 3 orders above); mismatch ->
setup failure; smoke skips the anchor.
Activation determinism was already
demonstrated by the 3084 L33 replay of the
3083 state (top8 bit-equal, med_c within
6e-13).

verdict (preregistered, unchanged from
3083):  setup/b-anchor/repro failure ->
third_setup_failed;  any family top8
degenerate (n_neg < 8) ->
third_top8_degenerate;  spectrum class
from CS top3 (thresholds in the 3082 gap):
all top3 >= 0.9 -> trunk; all top3 <= 0.5
-> dispersed; else mixed;  migration
state: G_DS AND f2~T_AB p<0.05 -> locked;
G_DS -> partial; else absent;  combined:
trunk + locked/partial ->
third_trunk_migrates; dispersed +
locked/partial -> third_dispersed_migrates
(PR diagnosis falsified); trunk + absent
-> third_trunk_no_migrate; dispersed +
absent -> third_dispersed_no_migrate (3082
prediction holds, 4B-side outlier
deepened); mixed -> third_mixed_<state>

arbitration reading (frozen, unchanged):
third_trunk_migrates -> DS7B was the
outlier (R1 distill specialization);
third_dispersed_migrates -> 3082 PR
diagnostic falsified; third_trunk_no_
migrate -> qwen3-4b specificity deepened;
third_dispersed_no_migrate -> 3082
prediction holds (4B-side trunk is the
rare anatomy).

model adaptation (frozen before run):
qwen2.5-3b-instruct (Qwen2ForCausalLM,
hidden 2048, 36 layers, 16 heads, 2 kv
heads, inter 11008, vocab 151936, bf16,
6.18 GB); L_INJ=34, L_POST=35 (injection
at the 2nd-from-last layer V, observation
at the last; layer 35 downstream; the 3084
scan selected L34 as the strongest rescue
layer); E2H/lens machinery omitted as in
3081/3083 (all cosines from forward
logits); input embeddings are TIED to the
unembed in this model
(tie_word_embeddings=True; forwards/
logits protocol unaffected, recorded as
an adaptation); same frozen 3076 texts.

limitations (recorded): L_INJ=34 chosen
from the 3084 descriptive layer scan
(strongest rescue magnitudes); the rescue
band boundary remains unmapped (coarse
4-layer grid); qwen2.5-3b shares the Qwen2
lineage with the DS7B backbone - not a
maximally independent model; instruct-
tuned like qwen3-4b; same 3076 texts; n=24
pairs per family pair; partial spearman
first-order on n=24 is orientation
evidence; causal-connective paradigm,
shared syntactic frame; migration defined
on the per-family focal top-8 subset
frame; spectrum classification thresholds
(0.9 / 0.5) chosen in the 3082 empirical
gap BEFORE this run.

memory discipline: single model; families
sequential; big banks fp64 CPU freed after
each family (del + gc + empty_cache); no
W32, no Wo34/Wo35 (E2H omitted).
"""
import gc
import hashlib
import io
import itertools
import json
import os
import time
from statistics import NormalDist

import numpy as np
import torch
from transformers import AutoTokenizer, \
    AutoModelForCausalLM

PHASE = 3085
NAME = 'omega_p82_l34_full_arbitration'
ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUT = os.path.join(
    ROOT, 'tests', 'glm5', 'result',
    'rdc_query_construction_20260913',
    'phase3085', NAME)
SMOKE = os.environ.get('SMOKE', '0') == '1'
if SMOKE:
    OUT = os.path.join(OUT, 'smoke')

MDIR = os.path.join(ROOT, 'models', 'hf',
                    'qwen2.5-3b-instruct')
NL, HID, KV_HEAD, NQ = 36, 2048, 2, 16
HDIM = 128
KVW = KV_HEAD * HDIM
NQW = NQ * HDIM
FRONT = 4
SEED_MAIN = 3085
L_INJ = 34
L_POST = 35
NH = NQ
SEED = 3085
REPRO_TOL = 1e-9
N_PERM = 1000 if SMOKE else 20000
G1_SP = 0.5
G1_P = 0.05
FKEYS = ('A', 'B', 'C')
CPAIRS = (('A', 'B'), ('A', 'C'), ('B', 'C'))
TARGETS = ('so', 'because', 'therefore',
           'however', 'while', 'yet',
           'although', 'thus')
FAM = {
    'A': {
        'domain': 'everyday-causal (3074 texts '
                  'rerun, bit-anchor family)',
        'bodies': (
            'The weather was cold, so',
            'He studied every night because',
            'The experiment failed, therefore',
            'He missed the train, however',
            'The garden grows quickly while',
            'The price was high, yet',
            'She speaks French, although',
            'The road was closed, thus',),
        'prefixes': ('', 'In a formal style,',
                     'In Shakespearean style,',
                     'Regarding the weather,'),
    },
    'B': {
        'domain': 'science-causal',
        'bodies': (
            'The solution turned acidic, so',
            'The sample was heated because',
            'The catalyst degraded, therefore',
            'The vacuum leaked, however',
            'The crystals formed while',
            'The pressure dropped, yet',
            'The alloy expanded, although',
            'The circuit overheated, thus',),
        'prefixes': ('', 'In a formal style,',
                     'In Shakespearean style,',
                     'Regarding the experiment,'),
    },
    'C': {
        'domain': 'social-emotional-causal',
        'bodies': (
            'She felt deeply betrayed, so',
            'He apologized to her because',
            'They reconciled after the '
            'quarrel, therefore',
            'She stormed out of the room, '
            'however',
            'He listened quietly to every '
            'word while',
            'The gift was cheap and hasty, '
            'yet',
            'She forgave him in the end, '
            'although',
            'The friendship ended without '
            'warning, thus',),
        'prefixes': ('', 'In a formal style,',
                     'In Shakespearean style,',
                     'Regarding the '
                     'conversation,'),
    },
}

PREREG = {
    'mode': 'single model qwen2.5-3b-instruct '
            'bf16 eager seed 3085; three '
            'frozen 3076 families processed '
            'sequentially in one process '
            '(A, B, C; big banks freed between '
            'families); per-family protocol '
            '3081-identical minus E2H/lens (no '
            'W32 - all cosines from forward '
            'logits): repV 3065-identical E1 '
            'ladder at L_INJ=34 (24 pairs, '
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
            'CS1H: PR / k_eff / top3); NEW '
            'vs 3083: L34 deterministic '
            'reproduction anchor vs the '
            '3084 layer-scan npz (n_neg/'
            'top8 exact; med_c/CS1H/R_ALL '
            'within 1e-9; feeds setup_ok; '
            'smoke skips); smoke mode '
            'optional (SMOKE=1: 12 masks, '
            'K3=4 pairs, NP_USE=8, cross '
            'stats off)',
    'question': '3085 A (menu of 3084): the '
                '3084 layer scan located the '
                '3B causal loading machinery: '
                'L33 (the 3083 position) is a '
                'valley (family C n_neg=2/16, '
                'med_c lowest, R_ALL near '
                'zero) while L34 is the '
                'strongest rescue layer (n_neg '
                '12/12/11, med_c 0.22-0.36, '
                'R_ALL -0.23..-0.30, the '
                'qwen3-4b protocol position). '
                ' The 3083 four-way '
                'trunk/dispersed x migrates '
                'arbitration was postponed by '
                'the L33 degeneration - it is '
                'RE-OPENED here: rerun the '
                'full 3083 pipeline (E1 + E3 + '
                'E4 + spectrum + cross '
                'statistics) at L_INJ=34 and '
                'combine the in-run spectrum '
                'class with the migration gate '
                'into the four-way verdict.',
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
                    'at L34 are new data (new '
                    'forwards, seed 3085); the '
                    'gate G_DS, the spectrum '
                    'thresholds (0.9 / 0.5) and '
                    'the combined verdict tree '
                    'were fixed before any L34 '
                    'observation; 3084 L34 '
                    'priors (n_neg/med_c/R_ALL '
                    'descriptives) guided the '
                    'LAYER CHOICE ONLY - all '
                    'verdict inputs (f2/T/U, '
                    'spectrum class, gates) are '
                    'computed fresh in-run; '
                    'family texts are the same '
                    'frozen 3076 texts',
    'repro_anchor': 'activation collection is '
                    'deterministic: same model, '
                    'same frozen texts, same '
                    'layer -> L34 values must '
                    'match the 3084 npz per '
                    'family (n_neg exact, top8 '
                    'exact, med_c/CS1H/R_ALL '
                    'abs diff <= 1e-9); the 1e-9 '
                    'tolerance covers the fp32-'
                    'roundtrip noise measured by '
                    'the 3083-vs-3084 L33 probe '
                    '(max 5.8e-13 med_c, 3.0e-12 '
                    'CS1H, 1.4e-13 R_ALL); '
                    'mismatch -> '
                    'third_setup_failed; smoke '
                    'skips the anchor (K3/NP_USE '
                    'differ)',
    'families': {
        fk: {'domain': FAM[fk]['domain'],
             'bodies': list(FAM[fk]['bodies']),
             'targets': list(TARGETS),
             'prefixes': list(FAM[fk]
                              ['prefixes'])}
        for fk in FKEYS},
    'layer_mapping': 'L_INJ=34 (V injection + '
                     'attn swaps), L_POST=35 '
                     '(block-output continuity '
                     'check); 36-layer stack, '
                     'layer 35 downstream of '
                     'the injection; selected by '
                     'the 3084 layer scan '
                     '(L28/31/33/34: L34 strongest '
                     'rescue, L33 the 3083 '
                     'valley); DS7B used L25/L26 '
                     'of 28 (3rd/2nd from last), '
                     'qwen3-4b used L34/L35 of '
                     '36 (2nd from last / last)',
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
                'AND min(sp) > 0; seed 3085, '
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
        'two-sided permutation p (seed 3085, '
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
        'two-point calibrated; qwen2.5-3b '
        'ties input embeddings to the '
        'unembed (tie_word_embeddings='
        'True) - the forwards/logits-only '
        'protocol is unaffected, recorded '
        'as an adaptation; L_INJ=34 was '
        'selected from the 3084 '
        'descriptive layer scan (strongest '
        'rescue magnitudes, n_neg 12/12/'
        '11) - layer-choice dependence is '
        'handled explicitly but the '
        'rescue-band boundary remains '
        'unmapped (coarse 4-layer grid)',
    'memory_discipline': 'single model; '
        'families sequential with big banks '
        'freed between families; no W32 / '
        'Wo34 / Wo35 (E2H and lens probes '
        'omitted); del + gc + empty_cache '
        'at family end and run end',
}

os.makedirs(OUT, exist_ok=True)
for fn in (NAME + '.npz', 'run_log.txt',
           'execution.json', 'result.json',
           'seal.json'):
    p = os.path.join(OUT, fn)
    if os.path.exists(p):
        os.remove(p)
t0 = time.time()
created = time.strftime('%Y-%m-%d %H:%M:%S')
execution = {'phase': PHASE, 'name': NAME,
             'created': created, 'prereg': PREREG,
             'smoke': SMOKE}
with io.open(os.path.join(OUT, 'execution.json'),
             'w', encoding='utf-8') as f:
    json.dump(execution, f, ensure_ascii=False,
              indent=1)

lines = []


def log(msg):
    lines.append(str(msg))
    with io.open(LOG, 'a', encoding='utf-8') as f:
        f.write(str(msg) + '\n')


def sha8(path):
    with io.open(path, 'rb') as f:
        return hashlib.sha256(
            f.read()).hexdigest()[:8]


LOG = os.path.join(OUT, 'run_log.txt')
log('execution.json written (prereg frozen) %s '
    'smoke=%s' % (created, SMOKE))
torch.manual_seed(SEED_MAIN)
np.random.seed(SEED_MAIN)


def cosv(a, b):
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(a @ b) / (na * nb)


def spearman(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ra = np.argsort(np.argsort(a)) \
        .astype(np.float64)
    rb = np.argsort(np.argsort(b)) \
        .astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra * ra).sum()
                  * (rb * rb).sum())
    if den == 0:
        return 0.0
    return float((ra * rb).sum() / den)


def perm_p(a, b, n_perm=N_PERM, seed=SEED):
    rng = np.random.default_rng(seed)
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    obs = abs(spearman(a, b))
    if n_perm <= 0:
        return 1.0
    ra = np.argsort(np.argsort(a)) \
        .astype(np.float64)
    ra -= ra.mean()
    B = np.tile(b, (n_perm, 1))
    B = rng.permuted(B, axis=1)
    rb = np.argsort(np.argsort(B, axis=1),
                    axis=1).astype(np.float64)
    rb -= rb.mean(axis=1, keepdims=True)
    num = (rb * ra[None, :]).sum(axis=1)
    den = np.sqrt(
        (rb * rb).sum(axis=1)
        * float((ra * ra).sum()))
    den[den == 0] = 1.0
    stats = np.abs(num / den)
    return float((stats >= obs - 1e-12)
                 .mean())


def pearson(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a * a).sum()
                  * (b * b).sum())
    if den == 0:
        return 0.0
    return float((a * b).sum() / den)


def ranks(x):
    x = np.asarray(x, np.float64)
    return np.argsort(np.argsort(x)) \
        .astype(np.float64)


def partial_sp(x, y, z):
    """First-order partial spearman: pearson
    on ranks of x,y controlling rank(z)."""
    rx = ranks(x)
    ry = ranks(y)
    rz = ranks(z)
    r_xy = pearson(rx, ry)
    r_xz = pearson(rx, rz)
    r_yz = pearson(ry, rz)
    den = np.sqrt(
        (1.0 - r_xz * r_xz)
        * (1.0 - r_yz * r_yz))
    if den == 0:
        return 0.0
    return float((r_xy - r_xz * r_yz) / den)


def partial_sp_perm_p(x, y, z,
                      n_perm=N_PERM,
                      seed=SEED):
    """Perm p for partial_sp: permute y only
    (x, z structure fixed)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    z = np.asarray(z, np.float64)
    obs = abs(partial_sp(x, y, z))
    if n_perm <= 0:
        return 1.0
    Y = np.tile(y, (n_perm, 1))
    Y = rng.permuted(Y, axis=1)
    rx = ranks(x)
    rz = ranks(z)
    r_xz = pearson(rx, rz)
    cnt = 0
    for i in range(n_perm):
        ry = ranks(Y[i])
        r_xy = pearson(rx, ry)
        r_yz = pearson(ry, rz)
        den = np.sqrt(
            (1.0 - r_xz * r_xz)
            * (1.0 - r_yz * r_yz))
        val = 0.0 if den == 0 else \
            (r_xy - r_xz * r_yz) / den
        if abs(val) >= obs - 1e-12:
            cnt += 1
    return float(cnt) / n_perm

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



# ==== load model ====
tok = AutoTokenizer.from_pretrained(MDIR)
model = AutoModelForCausalLM.from_pretrained(
    MDIR, torch_dtype=torch.bfloat16,
    attn_implementation='eager').to('cuda') \
    .eval()
layers = model.model.layers
assert len(layers) == NL
assert int(model.config.num_key_value_heads) \
    == KV_HEAD
assert int(model.config.num_attention_heads) \
    == NQ
assert int(model.config.hidden_size) == HID
TIED = bool(getattr(
    model.config, 'tie_word_embeddings',
    False))
log('tie_word_embeddings=%s (tied '
    'embeddings: forwards/logits-only '
    'protocol unaffected, recorded as an '
    'adaptation)' % TIED)
INTER = int(model.config.intermediate_size)
NVOC = int(model.config.vocab_size)
assert NVOC == 151936
log('qwen2.5-3b (qwen2.5-3b-instruct) loaded '
    'bf16 (vocab=%d inter=%d layers=%d) '
    'gpu=%.2f GB / total %.2f GB'
    % (NVOC, INTER, NL,
       torch.cuda.memory_allocated() / 1e9,
       torch.cuda.get_device_properties(0)
       .total_memory / 1e9))

FW = [0]
stateV = {li: {'repl': None, 'mask': None}
          for li in range(NL)}
stateATN = {li: {'repl': None, 'mask': None}
            for li in range(NL)}
capV = {li: {'rec': False, 'orig': None}
        for li in range(NL)}
capP = {li: {'rec': False, 'v': None}
        for li in range(NL)}
capX = {li: {'rec': False, 'v': None}
        for li in range(NL)}
capA = {li: {'rec': False, 'v': None}
        for li in range(NL)}
capM = {li: {'rec': False, 'v': None}
        for li in range(NL)}
capH = {li: {'rec': False, 'v': None}
        for li in range(NL)}


def hook_v(st, cp):
    def h(module, inp, out):
        if cp['rec']:
            cp['orig'] = out[0].detach().clone()
        if st['repl'] is not None:
            out[0][st['mask']] = st['repl']
        return out
    return h


def hook_post(cp):
    def h(module, inp, out):
        if cp['rec']:
            cp['v'] = out[0].detach().clone()
        return out
    return h


def hook_last(cp):
    def h(module, inp, out):
        if cp['rec']:
            t = out[0] \
                if isinstance(out, tuple) else out
            cp['v'] = t[0, -1].detach().clone()
        return out
    return h


def hook_in_last(cp):
    def h(module, inp, out):
        if cp['rec']:
            cp['v'] = inp[0][0, -1] \
                .detach().clone()
        return out
    return h


def hook_pre_last(st):
    def h(module, args):
        if st['repl'] is None:
            return None
        a = args[0].clone()
        a[0, -1, st['mask']] = st['repl']
        return (a,)
    return h


for li in range(NL):
    layers[li].self_attn.v_proj \
        .register_forward_hook(hook_v(
            stateV[li], capV[li]))
    layers[li].register_forward_hook(hook_post(
        capP[li]))
    layers[li].register_forward_hook(hook_in_last(
        capX[li]))
    layers[li].self_attn \
        .register_forward_hook(hook_last(capA[li]))
    layers[li].mlp \
        .register_forward_hook(hook_last(capM[li]))
    layers[li].self_attn.o_proj \
        .register_forward_hook(hook_in_last(
            capH[li]))
    layers[li].self_attn.o_proj \
        .register_forward_pre_hook(hook_pre_last(
            stateATN[li]))


def reset_all():
    for li in range(NL):
        stateV[li]['repl'] = None
        stateV[li]['mask'] = None
        capV[li]['rec'] = False
        capV[li]['orig'] = None
        capP[li]['rec'] = False
        capP[li]['v'] = None
        capX[li]['rec'] = False
        capX[li]['v'] = None
        capA[li]['rec'] = False
        capA[li]['v'] = None
        capM[li]['rec'] = False
        capM[li]['v'] = None
        capH[li]['rec'] = False
        capH[li]['v'] = None
        stateATN[li]['repl'] = None
        stateATN[li]['mask'] = None


def forward_gen(ids, repl=None, attn_swaps=None):
    """3076-identical V replacement; optional
    attn swaps [(layer, idx, vals)] at o_proj
    inputs (last position).  Returns fp64 lg
    (NVOC,), vb (NL, seq, KVW), po
    (NL, seq, HID), zX/zA/zM (NL, HID), zH
    (NL, NQW) on CPU numpy."""
    reset_all()
    FW[0] += 1
    m_all = torch.ones(
        len(ids), dtype=torch.bool,
        device='cuda')
    if repl is not None:
        rt = torch.tensor(
            np.ascontiguousarray(repl),
            dtype=torch.bfloat16,
            device='cuda')
        for li in range(NL):
            stateV[li]['repl'] = rt[li]
            stateV[li]['mask'] = m_all
    if attn_swaps is not None:
        for li_, midx, mval in attn_swaps:
            stateATN[li_]['mask'] = torch.tensor(
                np.ascontiguousarray(midx),
                dtype=torch.long,
                device='cuda')
            stateATN[li_]['repl'] = torch.tensor(
                np.ascontiguousarray(mval),
                dtype=torch.bfloat16,
                device='cuda')
    for li in range(NL):
        capV[li]['rec'] = True
        capP[li]['rec'] = True
        capX[li]['rec'] = True
        capA[li]['rec'] = True
        capM[li]['rec'] = True
        capH[li]['rec'] = True
    with torch.no_grad():
        out = model(torch.tensor(
            [ids], device='cuda'),
            use_cache=False)
    lg = out.logits[0, -1].detach() \
        .double().cpu().numpy()
    vb = np.stack([capV[li]['orig'].double()
                   .cpu().numpy()
                   for li in range(NL)])
    po = np.stack([capP[li]['v'].double()
                   .cpu().numpy()
                   for li in range(NL)])
    zX = torch.stack([capX[li]['v']
                      for li in range(NL)])
    zA = torch.stack([capA[li]['v']
                      for li in range(NL)])
    zM = torch.stack([capM[li]['v']
                      for li in range(NL)])
    zH = torch.stack([capH[li]['v']
                      for li in range(NL)])
    reset_all()
    return lg, vb, po, zX.double().cpu() \
        .numpy(), zA.double().cpu().numpy(), \
        zM.double().cpu().numpy(), \
        zH.double().cpu().numpy()


ridx = np.arange(FRONT)
ALLH = np.arange(NQW)
HEAD_IDX = [np.arange(h * HDIM, (h + 1) * HDIM)
            for h in range(NH)]


def popcount(m):
    return bin(m).count('1')


def run_family(fkey):
    fam = FAM[fkey]
    bodies = fam['bodies']
    prefixes = fam['prefixes']
    log('[%s] ==== family begin (%s) ===='
        % (fkey, fam['domain']))

    # ==== assembly (3065/3074/3076-identical)
    word_tok = {}
    for w in TARGETS:
        wi = tok(' ' + w,
                 add_special_tokens=False)[
            'input_ids']
        assert len(wi) == 1, (fkey, w, wi)
        word_tok[w] = int(wi[0])
    assembled = []
    for bi in range(len(bodies)):
        for ci in range(len(prefixes)):
            s = (prefixes[ci] + ' '
                 + bodies[bi]) if prefixes[ci] \
                else bodies[bi]
            ids = [int(x) for x in tok(
                s, add_special_tokens=False)[
                'input_ids']]
            t = word_tok[TARGETS[bi]]
            assert ids.count(t) == 1, \
                (fkey, bi, ci)
            assembled.append(
                {'ids': ids, 'cond': ci,
                 'body': bi})
    n_pr = len(assembled)
    assert n_pr == 32
    idx_of = {}
    for i in range(n_pr):
        idx_of[(assembled[i]['cond'],
                assembled[i]['body'])] = i
    for i in range(n_pr):
        ci = assembled[i]['cond']
        if ci == 0:
            assembled[i]['off'] = 0
        else:
            bid = assembled[idx_of[(0,
                assembled[i]['body'])]]['ids']
            pid = assembled[i]['ids']
            off = len(pid) - len(bid)
            assert off > 0, (fkey, i)
            assert list(pid[off + 1:]) \
                == list(bid[1:]), (fkey, i)
            w0b = tok.decode([bid[0]]).strip()
            w0p = tok.decode(
                [pid[off]]).strip()
            assert w0b == w0p, (fkey, i)
            assembled[i]['off'] = off
    LENS = np.array([len(assembled[i]['ids'])
                     for i in range(n_pr)])
    NMAX = int(LENS.max())
    log('[%s] assembled 32 prompts (lens %d-%d)'
        % (fkey, int(LENS.min()), NMAX))

    cidx = []
    bidx = []
    for ci in (1, 2, 3):
        for bi in range(len(bodies)):
            cidx.append(ci)
            bidx.append(bi)
    cidx = np.array(cidx)
    bidx = np.array(bidx)
    NP_ = 24
    NP_USE = 8 if SMOKE else NP_
    K3 = 4 if SMOKE else NP_

    def pair_idx(k):
        b = int(bidx[k])
        c = int(cidx[k])
        base_i = idx_of[(0, b)]
        pref_i = idx_of[(c, b)]
        off = assembled[pref_i]['off']
        return b, base_i, pref_i, off

    BASE_K = np.array([pair_idx(k)[1]
                       for k in range(NP_)])
    for k in range(NP_):
        b, base_i, pref_i, off = pair_idx(k)
        assert int(LENS[base_i]) >= FRONT, \
            (fkey, k, int(LENS[base_i]))
    log('[%s] base length check ok (all >= '
        'FRONT=%d)' % (fkey, FRONT))

    # ==== banks ====
    LG = np.zeros((n_pr, NVOC))
    VB = np.zeros((n_pr, NL, NMAX, KVW))
    PB = np.zeros((n_pr, NL, NMAX, HID))
    BX = np.zeros((n_pr, NL, HID))
    BA = np.zeros((n_pr, NL, HID))
    BM = np.zeros((n_pr, NL, HID))
    BH = np.zeros((n_pr, NL, NQW))
    for i in range(n_pr):
        lg, vb, po, zX, zA, zM, zH = \
            forward_gen(assembled[i]['ids'])
        n = int(LENS[i])
        if i == 0:
            assert vb.shape == (NL, n, KVW), \
                vb.shape
            assert po.shape == (NL, n, HID), \
                po.shape
        LG[i] = lg
        VB[i, :, :n, :] = vb
        PB[i, :, :n, :] = po
        BX[i] = zX
        BA[i] = zA
        BM[i] = zM
        BH[i] = zH
    log('[%s] banks: LG%s VB%s PB%s BH%s '
        '(forwards=%d)'
        % (fkey, LG.shape, VB.shape, PB.shape,
           BH.shape, FW[0]))

    b0_diff = 0.0
    for si in (0, 9, 17, 31):
        lg, vb, po, zX, zA, zM, zH = \
            forward_gen(assembled[si]['ids'])
        n2 = int(LENS[si])
        b0_diff = max(b0_diff, float(np.max(
            np.abs(LG[si] - lg))))
        b0_diff = max(b0_diff, float(np.max(
            np.abs(VB[si, :, :n2, :] - vb))))
        b0_diff = max(b0_diff, float(np.max(
            np.abs(PB[si, :, :n2, :] - po))))
        b0_diff = max(b0_diff, float(np.max(
            np.abs(BX[si] - zX))))
        b0_diff = max(b0_diff, float(np.max(
            np.abs(BA[si] - zA))))
        b0_diff = max(b0_diff, float(np.max(
            np.abs(BM[si] - zM))))
        b0_diff = max(b0_diff, float(np.max(
            np.abs(BH[si] - zH))))
    b0_diff = float(b0_diff)
    b0_ok = bool(b0_diff == 0.0)
    log('[%s] b0 recapture diff=%.3e ok=%s'
        % (fkey, b0_diff, b0_ok))

    TT = np.stack([
        LG[idx_of[(int(cidx[k]),
                   int(bidx[k]))]]
        - LG[idx_of[(0, int(bidx[k]))]]
        for k in range(NP_)])
    TT32 = TT.astype(np.float32)
    LG32 = LG.astype(np.float32)
    med_tn = float(np.median(
        np.linalg.norm(TT, axis=1)))

    base0 = idx_of[(0, int(bidx[0]))]
    n0 = int(LENS[base0])
    selfV = VB[base0, :, :n0, :].copy()
    lg_s = forward_gen(
        assembled[base0]['ids'], repl=selfV)[0]
    b1_diff = float(np.max(np.abs(lg_s
                                  - LG[base0])))
    b1_ok = bool(b1_diff == 0.0)
    log('[%s] b1 sham self-replacement diff='
        '%.3e ok=%s' % (fkey, b1_diff, b1_ok))

    b3_ok = bool(np.isfinite(LG).all()
                 and np.isfinite(VB).all()
                 and np.isfinite(PB).all()
                 and np.isfinite(TT).all()
                 and np.isfinite(BX).all()
                 and np.isfinite(BA).all()
                 and np.isfinite(BM).all()
                 and np.isfinite(BH).all())
    log('[%s] b3 finite=%s' % (fkey, b3_ok))

    b6_diff = 0.0
    for i in range(n_pr):
        n = int(LENS[i])
        x_ = torch.tensor(BX[i],
                          device='cuda') \
            .to(torch.bfloat16)
        a_ = torch.tensor(BA[i],
                          device='cuda') \
            .to(torch.bfloat16)
        m_ = torch.tensor(BM[i],
                          device='cuda') \
            .to(torch.bfloat16)
        p_ = torch.tensor(PB[i, :, n - 1, :],
                          device='cuda') \
            .to(torch.bfloat16)
        h2 = (x_ + a_) + m_
        b6_diff = max(b6_diff, float(
            (h2 - p_).abs().max()))
    b6_diff = float(b6_diff)
    b6_ok = bool(b6_diff == 0.0)
    log('[%s] b6 (x+a)+m=h2 bf16 diff=%.3e '
        'ok=%s' % (fkey, b6_diff, b6_ok))

    # b7a identity attn self-swap
    b, base_i, pref_i, off = pair_idx(0)
    lg7a = forward_gen(
        assembled[base_i]['ids'],
        attn_swaps=[(L_INJ, ALLH,
                     BH[base_i, L_INJ])])[0]
    b7a_diff = float(np.max(np.abs(
        lg7a - LG[base_i])))
    b7a_ok = bool(b7a_diff == 0.0)
    log('[%s] b7a identity attn self-swap '
        'diff=%.3e ok=%s'
        % (fkey, b7a_diff, b7a_ok))

    def repv_of(k):
        b, base_i, pref_i, off = pair_idx(k)
        nb = int(LENS[base_i])
        repV = VB[base_i, :, :nb, :].copy()
        repV[L_INJ, ridx, :] = VB[pref_i][
            L_INJ, off + ridx, :]
        return base_i, repV

    def run_cond(mk_kwargs):
        cs = np.full(K3, np.nan)
        for k in range(K3):
            base_i, repV = repv_of(k)
            lg = forward_gen(
                assembled[base_i]['ids'],
                repl=repV,
                **mk_kwargs(base_i))[0]
            cs[k] = cosv(lg - LG[base_i],
                         TT[k])
        return cs

    # ==== E1 repV injection ladder at L_INJ
    COS_LAD = np.full(NP_USE, np.nan)
    b4_diff = 0.0
    b8_diff = 0.0
    for k in range(NP_USE):
        b, base_i, pref_i, off = pair_idx(k)
        nb = int(LENS[base_i])
        base_i, repV = repv_of(k)
        lg, _, po_i, zX, _, _, _ = \
            forward_gen(
                assembled[base_i]['ids'],
                repl=repV)
        dlg = lg - LG[base_i]
        COS_LAD[k] = cosv(dlg, TT[k])
        b4_diff = max(b4_diff, float(np.max(
            np.abs(zX[L_INJ] - BX[base_i,
                                   L_INJ]))))
        b8_diff = max(b8_diff, float(np.max(
            np.abs(zX[L_POST]
                   - po_i[L_INJ, nb - 1]))))
    b4_ok = bool(b4_diff == 0.0)
    b8_ok = bool(b8_diff == 0.0)
    log('[%s] b4 delta-x at L_INJ diff=%.3e '
        'ok=%s' % (fkey, b4_diff, b4_ok))
    log('[%s] b8 zX[L_POST] vs zP[L_INJ] '
        'diff=%.3e ok=%s'
        % (fkey, b8_diff, b8_ok))

    med_c = float(np.median(COS_LAD))
    log('[%s] E1 med_c(L_INJ=%d)=%.4f '
        '(forwards=%d)'
        % (fkey, L_INJ, med_c, FW[0]))

    # ==== E3 per-head single swap scan ====
    CS1H = np.full((NH, K3), np.nan)
    for h in range(NH):
        idx = HEAD_IDX[h]
        CS1H[h] = run_cond(
            lambda bi, idx=idx: {
                'attn_swaps': [
                    (L_INJ, idx,
                     BH[bi, L_INJ][idx])]})
        if (h + 1) % 7 == 0 or h == NH - 1:
            log('[%s] E3 scan progress %d/%d '
                '(forwards=%d)'
                % (fkey, h + 1, NH, FW[0]))
    r1_nh = np.median(CS1H, axis=1) - med_c
    log('[%s] E3 scan r1_allnh: min=%.4f '
        'max=%.4f n_neg=%d'
        % (fkey, float(r1_nh.min()),
           float(r1_nh.max()),
           int((r1_nh < 0).sum())))

    # ==== per-family focal top8 (3071
    # criterion: argsort(r1) ascending) ====
    n_neg = int((r1_nh < 0).sum())
    order = np.argsort(r1_nh)
    topk = min(8, n_neg)
    top8 = [int(h) for h in order[:topk]]
    top8_sel_ok = bool(
        topk == 8
        and bool((r1_nh[top8] < 0).all()))
    neg_sum = float(r1_nh[r1_nh < 0].sum())
    capture8 = abs(float(
        r1_nh[top8].sum())) \
        / abs(neg_sum) if abs(neg_sum) > 1e-9 \
        else float('nan')
    log('[%s] top8 (3071 criterion)=%s '
        'n_neg=%d capture8=%.4f sel_ok=%s'
        % (fkey, top8, n_neg, capture8,
           top8_sel_ok))

    # ==== E4 subset saturation sweep ====
    NT = len(top8)
    NM_ALL = 1 << NT
    M_FULL = NM_ALL - 1
    if SMOKE:
        MASKS = [m for m in (1, 2, 3, 5, 7, 15,
                             31, 63, 127, 255, 24,
                             85) if m < NM_ALL]
    else:
        MASKS = list(range(1, NM_ALL))
    NM = len(MASKS)
    log('[%s] E4 sweep: NT=%d masks=%d K3=%d'
        % (fkey, NT, NM, K3))

    CS = np.full((NM, K3), np.nan)
    for mi, m in enumerate(MASKS):
        sel = [ti for ti in range(NT)
               if m >> ti & 1]
        idx = np.concatenate(
            [HEAD_IDX[top8[ti]] for ti in sel])
        CS[mi] = run_cond(
            lambda bi, idx=idx: {
                'attn_swaps': [
                    (L_INJ, idx,
                     BH[bi, L_INJ][idx])]})
        if (mi + 1) % 64 == 0 or mi == NM - 1:
            log('[%s] E4 sweep progress %d/%d '
                '(forwards=%d)'
                % (fkey, mi + 1, NM, FW[0]))
    R_S = np.median(CS, axis=1) - med_c
    A_S = -R_S
    y_u8 = float(-R_S[MASKS.index(M_FULL)]) \
        if M_FULL in MASKS else float('nan')
    log('[%s] E4 sweep R(S) collected: min='
        '%.4f max=%.4f (most negative at '
        'mask %d)'
        % (fkey, float(R_S.min()),
           float(R_S.max()),
           int(MASKS[int(np.argmin(R_S))])))

    # full 28-head swap
    lg_all = run_cond(lambda bi: {
        'attn_swaps': [(L_INJ, ALLH,
                        BH[bi, L_INJ])]})
    R_ALL = float(np.median(lg_all)
                  - med_c)
    log('[%s] full 28-head swap recov=%+.4f '
        '(forwards=%d)'
        % (fkey, R_ALL, FW[0]))

    # free the big banks before analysis
    del VB, PB, BX, BA, BM, BH, selfV
    gc.collect()
    torch.cuda.empty_cache()
    log('[%s] big banks freed (forwards=%d)'
        % (fkey, FW[0]))

    setup_ok_f = bool(b0_ok and b1_ok
                      and b3_ok and b4_ok
                      and b6_ok and b7a_ok
                      and b8_ok)
    log('[%s] ==== family end: setup_ok=%s '
        'forwards=%d ===='
        % (fkey, setup_ok_f, FW[0]))

    return {
        'fkey': fkey,
        'domain': fam['domain'],
        'med_c': med_c,
        'med_tt_norm': med_tn,
        'COS_LAD': COS_LAD,
        'r1_nh': r1_nh,
        'CS1H': CS1H,
        'n_neg': n_neg, 'topk': topk,
        'top8': top8,
        'top8_sel_ok': top8_sel_ok,
        'capture8': capture8,
        'MASKS': MASKS, 'CS': CS,
        'R_S': R_S, 'A_S': A_S,
        'R_ALL': R_ALL,
        'y_u8': y_u8,
        'TT32': TT32, 'LG32': LG32,
        'b0_diff': b0_diff, 'b0_ok': b0_ok,
        'b1_diff': b1_diff, 'b1_ok': b1_ok,
        'b3_ok': b3_ok,
        'b4_diff': b4_diff, 'b4_ok': b4_ok,
        'b6_diff': b6_diff, 'b6_ok': b6_ok,
        'b7a_diff': b7a_diff,
        'b7a_ok': b7a_ok,
        'b8_diff': b8_diff, 'b8_ok': b8_ok,
        'setup_ok_f': setup_ok_f,
    }


RES_F = {}
for fk in FKEYS:
    RES_F[fk] = run_family(fk)
    gc.collect()
    torch.cuda.empty_cache()


# ==== L34 deterministic reproduction
# anchor (vs 3084 layer-scan npz) ====
REPRO = {}
REPRO_OK = True
if SMOKE:
    log('repro anchor skipped (smoke: '
        'K3/NP_USE differ from the 3084 '
        'reference)')
else:
    z84 = np.load(os.path.join(
        ROOT, 'tests', 'glm5', 'result',
        'rdc_query_construction_20260913',
        'phase3084',
        'omega_p81_3b_layer_scan',
        'omega_p81_3b_layer_scan.npz'),
        allow_pickle=False)
    for fk in FKEYS:
        d_mc = abs(float(RES_F[fk]['med_c'])
                   - float(z84['MED_C_L34_'
                                + fk]))
        nn_ok = bool(
            int(RES_F[fk]['n_neg'])
            == int(z84['N_NEG_L34_' + fk]))
        t8_ok = bool(
            [int(h) for h
             in RES_F[fk]['top8']]
            == [int(h) for h
                in z84['TOP8_L34_' + fk]])
        d_cs = float(np.max(np.abs(
            RES_F[fk]['CS1H']
            - z84['CS1H_L34_' + fk])))
        d_ra = abs(float(RES_F[fk]['R_ALL'])
                   - float(z84['R_ALL_L34_'
                                + fk]))
        ok = bool(d_mc <= REPRO_TOL
                  and nn_ok and t8_ok
                  and d_cs <= REPRO_TOL
                  and d_ra <= REPRO_TOL)
        REPRO[fk] = {'medc_diff': d_mc,
                     'nneg_ok': nn_ok,
                     'top8_ok': t8_ok,
                     'cs1h_diff': d_cs,
                     'rall_diff': d_ra,
                     'ok': ok}
        REPRO_OK = REPRO_OK and ok
        log('repro[%s] vs 3084 L34: '
            'medc_diff=%.3e nneg_ok=%s '
            'top8_ok=%s cs1h_diff=%.3e '
            'rall_diff=%.3e ok=%s'
            % (fk, d_mc, nn_ok, t8_ok,
               d_cs, d_ra, ok))

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

# ==== cross-model statistics ====
setup_ok_all = bool(all(
    RES_F[fk]['setup_ok_f'] for fk in FKEYS)
    and REPRO_OK)
top8_all_ok = bool(all(
    RES_F[fk]['top8_sel_ok']
    for fk in FKEYS))
verdict = 'smoke_pending'
gates = {'G_DS': None, 'G1': None}
if SMOKE:
    log('cross-model statistics skipped '
        '(smoke)')
    log('VERDICT: smoke_pending')
else:
    if not setup_ok_all:
        verdict = 'third_setup_failed'
        log('VERDICT: %s (setup_ok=%s)'
            % (verdict, setup_ok_all))
    elif not top8_all_ok:
        verdict = 'third_top8_degenerate'
        log('VERDICT: %s (top8_sel=%s)'
            % (verdict, top8_all_ok))
    else:
        log('E5 cross-model statistics '
            '(seed %d, n_perm %d)'
            % (SEED, N_PERM))
        TT64 = {f: RES_F[f]['TT32']
                .astype(np.float64)
                for f in FKEYS}
        LG64 = {f: RES_F[f]['LG32']
                .astype(np.float64)
                for f in FKEYS}
        CIDX = np.array([k // 8 + 1
                         for k in range(24)])
        BIDX = np.array([k % 8
                         for k in range(24)])
        PREF_K = BIDX * 4 + CIDX
        BASE_KX = BIDX * 4
        T = {}
        U = {}
        F = {fid: {} for fid in
             ('f1_sTT', 'f2_cTT', 'f3_sLGp',
              'f4_sLGb', 'f5_amp')}
        for fa, fb in CPAIRS:
            key = fa + fb
            T[key] = np.array([
                spearman(RES_F[fa]['CS1H'][:, k],
                         RES_F[fb]['CS1H'][:, k])
                for k in range(24)])
            U[key] = np.array([
                spearman(RES_F[fa]['CS'][:, k],
                         RES_F[fb]['CS'][:, k])
                for k in range(24)])
            F['f1_sTT'][key] = np.array([
                spearman(TT64[fa][k],
                         TT64[fb][k])
                for k in range(24)])
            F['f2_cTT'][key] = np.array([
                cosv(TT64[fa][k], TT64[fb][k])
                for k in range(24)])
            F['f3_sLGp'][key] = np.array([
                spearman(LG64[fa][PREF_K[k]],
                         LG64[fb][PREF_K[k]])
                for k in range(24)])
            F['f4_sLGb'][key] = np.array([
                spearman(LG64[fa][BASE_KX[k]],
                         LG64[fb][BASE_KX[k]])
                for k in range(24)])
            n_fa = np.linalg.norm(TT64[fa],
                                  axis=1)
            n_fb = np.linalg.norm(TT64[fb],
                                  axis=1)
            F['f5_amp'][key] = np.minimum(
                n_fa, n_fb) / np.maximum(
                n_fa, n_fb)
        for key in ('AB', 'AC', 'BC'):
            log('  %s: T med=%.4f [%+.3f, '
                '%+.3f] n_pos=%d/24 | '
                'U med=%.4f [%+.3f, %+.3f]'
                % (key,
                   float(np.median(T[key])),
                   float(T[key].min()),
                   float(T[key].max()),
                   int((T[key] > 0).sum()),
                   float(np.median(U[key])),
                   float(U[key].min()),
                   float(U[key].max())))

        # E3 tests: 5 predictors x 2 responses
        # x 3 pairs = 30
        test_keys = []
        E3 = {}
        for resp_name, RESP in (('T', T),
                                ('U', U)):
            for fid in ('f1_sTT', 'f2_cTT',
                        'f3_sLGp', 'f4_sLGb',
                        'f5_amp'):
                for key in ('AB', 'AC', 'BC'):
                    s_ = spearman(F[fid][key],
                                  RESP[key])
                    p_ = perm_p(F[fid][key],
                                RESP[key])
                    E3['%s~%s_%s'
                       % (fid, resp_name,
                          key)] = {'sp': s_,
                                   'p': p_}
                    test_keys.append(
                        (fid, resp_name, key))
        log('E3 tests sp (p):')
        for tk in test_keys:
            e = E3['%s~%s_%s' % tk]
            log('  %s~%s_%s: %+.4f (%.5f)%s'
                % (tk[0], tk[1], tk[2],
                   e['sp'], e['p'],
                   ' <-- MAIN' if tk[0]
                   == 'f2_cTT' else ''))

        # f2 main gate G_DS (3080 G2-aligned)
        E2 = {}
        for rn, RESP in (('T', T), ('U', U)):
            for key in ('AB', 'AC', 'BC'):
                E2['%s_%s' % (rn, key)] = \
                    E3['f2_cTT~%s_%s'
                       % (rn, key)]
        cnt_pos = sum(
            1 for v in E2.values()
            if v['sp'] > 0 and v['p'] < 0.05)
        n_bonf = sum(
            1 for v in E2.values()
            if v['p'] < 0.05 / 6)
        min_sp2 = min(v['sp']
                      for v in E2.values())
        g_ds = bool(cnt_pos >= 4
                    and min_sp2 > 0)
        zlist = [NormalDist().inv_cdf(
            min(1.0 - 1e-12, 1.0 - v['p']))
            for v in E2.values()]
        stouffer = float(np.sum(zlist)
                         / np.sqrt(len(zlist)))
        log('  G_DS: count=%d/6 (bonf %d) '
            'min_sp=%.4f -> %s | Stouffer '
            'z(one-sided)=%.3f'
            % (cnt_pos, n_bonf, min_sp2,
               g_ds, stouffer))

        # f1 control gate G1 (3079-aligned)
        best = None
        for key in ('AB', 'AC', 'BC'):
            for rn in ('T', 'U'):
                e = E3['f1_sTT~%s_%s'
                       % (rn, key)]
                cand = (abs(e['sp']), e['sp'],
                        e['p'], rn, key)
                if best is None \
                        or cand[0] > best[0]:
                    best = cand
        g1 = bool(best[0] >= G1_SP
                  and best[2] < G1_P)
        f1_pos = bool(g1 and best[1] > 0)
        log('  G1 (f1 control): best sp=%+.4f '
            'p=%.5f on %s_%s -> %s '
            '(f1_sign_positive=%s)'
            % (best[1], best[2], best[3],
               best[4], g1, f1_pos))

        # coupling + family orientation
        SP_UT = {}
        sp_f1f2 = {}
        mig = {}
        sp_as = {}
        tt_med = {}
        for fa, fb in CPAIRS:
            key = fa + fb
            SP_UT[key] = spearman(U[key],
                                  T[key])
            sp_f1f2[key] = spearman(
                F['f1_sTT'][key],
                F['f2_cTT'][key])
            mig[key] = spearman(
                RES_F[fa]['r1_nh'],
                RES_F[fb]['r1_nh'])
            sp_as[key] = spearman(
                RES_F[fa]['A_S'],
                RES_F[fb]['A_S'])
            tt_med[key] = float(np.median(
                [spearman(TT64[fa][k],
                          TT64[fb][k])
                 for k in range(24)]))
            log('  %s: sp(U,T)=%.4f '
                'sp(f1,f2)=%.4f mig=%.4f '
                'sp(A_S)=%.4f tt_med=%.4f'
                % (key, SP_UT[key],
                   sp_f1f2[key], mig[key],
                   sp_as[key],
                   tt_med[key]))
        sp_fam_as = spearman(
            [sp_as[k] for k in
             ('AB', 'AC', 'BC')],
            [mig[k] for k in
             ('AB', 'AC', 'BC')])
        sp_fam_tt = spearman(
            [tt_med[k] for k in
             ('AB', 'AC', 'BC')],
            [mig[k] for k in
             ('AB', 'AC', 'BC')])
        log('  family orientation (n=3): '
            'sp(A_S sim, mig)=%+.3f '
            'sp(tt_med, mig)=%+.3f'
            % (sp_fam_as, sp_fam_tt))

        # E3.1 AB per-pair table (exploratory)
        NORM = {f: np.linalg.norm(TT64[f],
                                  axis=1)
                for f in FKEYS}
        log('E3.1 AB per-pair table: '
            'k(bi,ci) normA normB f5 f1 f2 '
            'T_AB U_AB')
        for k in range(24):
            log('  %2d(%d,%d) %9.1f %9.1f '
                '%.3f %+.3f %+.3f %+.3f '
                '%+.3f'
                % (k, BIDX[k], CIDX[k],
                   NORM['A'][k], NORM['B'][k],
                   F['f5_amp']['AB'][k],
                   F['f1_sTT']['AB'][k],
                   F['f2_cTT']['AB'][k],
                   T['AB'][k], U['AB'][k]))
        kmax = int(np.argmax(T['AB']))
        kmin = int(np.argmin(T['AB']))
        log('  top T_AB pair: k=%d (bi=%d '
            'ci=%d) f1=%+.3f f2=%+.3f'
            % (kmax, BIDX[kmax],
               CIDX[kmax],
               F['f1_sTT']['AB'][kmax],
               F['f2_cTT']['AB'][kmax]))
        log('  bottom T_AB pair: k=%d (bi=%d '
            'ci=%d) f1=%+.3f f2=%+.3f'
            % (kmin, BIDX[kmin],
               CIDX[kmin],
               F['f1_sTT']['AB'][kmin],
               F['f2_cTT']['AB'][kmin]))

        # partial spearman (orientation)
        log('E3.2 partial spearman '
            '(orientation only):')
        PSP = {}
        for key in ('AB', 'AC'):
            for rn, RESP in (('T', T),
                             ('U', U)):
                tag = '%s_%s' % (rn, key)
                p21 = partial_sp(
                    F['f2_cTT'][key],
                    RESP[key],
                    F['f1_sTT'][key])
                p21p = partial_sp_perm_p(
                    F['f2_cTT'][key],
                    RESP[key],
                    F['f1_sTT'][key])
                p52 = partial_sp(
                    F['f5_amp'][key],
                    RESP[key],
                    F['f2_cTT'][key])
                p52p = partial_sp_perm_p(
                    F['f5_amp'][key],
                    RESP[key],
                    F['f2_cTT'][key])
                PSP['f2g1_' + tag] = p21
                PSP['f2g1p_' + tag] = p21p
                PSP['f5g2_' + tag] = p52
                PSP['f5g2p_' + tag] = p52p
                log('  %s: sp(f2|f1)=%+.3f '
                    '(p=%.4f) sp(f5|f2)='
                    '%+.3f (p=%.4f)'
                    % (tag, p21, p21p, p52,
                       p52p))

        # verdict (combined: migration state
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
        gates = {'G_DS': g_ds,
                 'count_sig_pos': cnt_pos,
                 'n_bonf': n_bonf,
                 'min_sp': min_sp2,
                 'stouffer_z': stouffer,
                 'f2_TAB_sp': sp_f2_TAB,
                 'f2_TAB_p': p_f2_TAB,
                 'G1': g1,
                 'G1_sp': best[1],
                 'G1_p': best[2],
                 'G1_resp': best[3],
                 'G1_pair': best[4],
                 'f1_sign_positive': f1_pos,
                 'spec_class': spec_class,
                 'spectrum': {
                     fk: dict(SPEC[fk])
                     for fk in FKEYS}}
        log('VERDICT: %s (spec=%s G_DS=%s '
            'count=%d/6 min_sp=%.4f '
            'stouffer=%.3f; f2~T_AB sp='
            '%+.4f p=%.5f; G1=%s best f1 '
            'sp=%+.4f)'
            % (verdict, spec_class, g_ds,
               cnt_pos, min_sp2, stouffer,
               sp_f2_TAB, p_f2_TAB, g1,
               best[1]))

# ==== npz ====
npz_path = os.path.join(OUT, NAME + '.npz')
save = {
    'VERDICT': np.array(verdict),
    'ELAPSED': np.float64(time.time() - t0),
    'SMOKE': np.bool_(SMOKE),
    'FORWARDS': np.int64(FW[0]),
    'SETUP_OK': np.bool_(setup_ok_all),
    'TOP8_ALL_OK': np.bool_(top8_all_ok),
    'L_INJ': np.int64(L_INJ),
    'L_POST': np.int64(L_POST),
    'SPEC_CLASS': np.array(spec_class),
    'TIED': np.bool_(TIED),
    'REPRO_OK': np.bool_(REPRO_OK),
}
for fk in FKEYS:
    if fk in REPRO:
        save['REPRO_MEDC_DIFF_' + fk] = \
            np.float64(
                REPRO[fk]['medc_diff'])
        save['REPRO_CS1H_DIFF_' + fk] = \
            np.float64(
                REPRO[fk]['cs1h_diff'])
        save['REPRO_RALL_DIFF_' + fk] = \
            np.float64(
                REPRO[fk]['rall_diff'])
        save['REPRO_NNEG_OK_' + fk] = \
            np.bool_(REPRO[fk]['nneg_ok'])
        save['REPRO_TOP8_OK_' + fk] = \
            np.bool_(REPRO[fk]['top8_ok'])
    R = RES_F[fk]
    save['MED_C_' + fk] = np.float64(
        R['med_c'])
    save['E3_PR_CS_' + fk] = np.float64(
        SPEC[fk]['pr'])
    save['E3_KEFF_CS_' + fk] = np.float64(
        SPEC[fk]['keff'])
    save['E3_TOP3_CS_' + fk] = np.float64(
        SPEC[fk]['top3'])
    save['E3_PR_CS1H_' + fk] = np.float64(
        SPEC1H[fk]['pr'])
    save['E3_KEFF_CS1H_' + fk] = \
        np.float64(SPEC1H[fk]['keff'])
    save['E3_TOP3_CS1H_' + fk] = \
        np.float64(SPEC1H[fk]['top3'])
    save['COS_LAD_' + fk] = R['COS_LAD']
    save['R1_ALLNH_' + fk] = R['r1_nh']
    save['CS1H_' + fk] = R['CS1H']
    save['TOP8_' + fk] = np.array(
        R['top8'], dtype=np.int64)
    save['N_NEG_' + fk] = np.int64(R['n_neg'])
    save['CAPTURE8_' + fk] = np.float64(
        R['capture8'])
    save['TOP8_SEL_OK_' + fk] = np.bool_(
        R['top8_sel_ok'])
    save['MASKS_' + fk] = np.array(
        R['MASKS'], dtype=np.int64)
    save['CS_' + fk] = R['CS']
    save['R_S_' + fk] = R['R_S']
    save['A_S_' + fk] = R['A_S']
    save['R_ALL_' + fk] = np.float64(
        R['R_ALL'])
    save['TT_NORM_' + fk] = np.linalg.norm(
        R['TT32'].astype(np.float64), axis=1)
    save['MED_TT_NORM_' + fk] = np.float64(
        R['med_tt_norm'])
    save['TT_' + fk] = R['TT32']
    save['LG_' + fk] = R['LG32']
    for bk in ('b0', 'b1', 'b4', 'b6',
               'b7a', 'b8'):
        save[(bk.upper() + '_DIFF_' + fk)] = \
            np.float64(R[bk + '_diff'])
        save[(bk.upper() + '_OK_' + fk)] = \
            np.bool_(R[bk + '_ok'])
    save['B3_OK_' + fk] = np.bool_(R['b3_ok'])
if not SMOKE and verdict not in (
        'third_setup_failed',
        'third_top8_degenerate'):
    for key in ('AB', 'AC', 'BC'):
        save['T_' + key] = T[key]
        save['U_' + key] = U[key]
        for fid in F:
            save['%s_%s' % (fid.upper(),
                            key)] = F[fid][key]
        save['MIG_' + key] = np.float64(
            mig[key])
        save['SP_AS_' + key] = np.float64(
            sp_as[key])
        save['TT_MED_' + key] = np.float64(
            tt_med[key])
        save['SP_UT_' + key] = np.float64(
            SP_UT[key])
        save['SP_F1F2_' + key] = np.float64(
            sp_f1f2[key])
    for tk in test_keys:
        e = E3['%s~%s_%s' % tk]
        save['E3_%s_%s_%s'
             % (tk[0].upper(), tk[1],
                tk[2])] = np.float64(e['sp'])
        save['E3P_%s_%s_%s'
             % (tk[0].upper(), tk[1],
                tk[2])] = np.float64(e['p'])
    save['GDS_COUNT'] = np.int64(cnt_pos)
    save['GDS_N_BONF'] = np.int64(n_bonf)
    save['GDS_MIN_SP'] = np.float64(min_sp2)
    save['STOUFFER_Z'] = np.float64(stouffer)
    for tag in PSP:
        save['PSP_' + tag.upper()] = \
            np.float64(PSP[tag])
    save['SP_FAM_AS_MIG'] = np.float64(
        sp_fam_as)
    save['SP_FAM_TT_MIG'] = np.float64(
        sp_fam_tt)
np.savez(npz_path, **save)
log('npz saved %s' % npz_path)

# ==== result.json ====
def f64(x):
    x = float(x)
    return x if np.isfinite(x) else None


fam_stats = {}
for fk in FKEYS:
    R = RES_F[fk]
    fam_stats[fk] = {
        'domain': R['domain'],
        'med_c': f64(R['med_c']),
        'med_tt_norm': f64(R['med_tt_norm']),
        'r_all': f64(R['R_ALL']),
        'a_u8': f64(R['y_u8']),
        'n_neg': int(R['n_neg']),
        'top8': R['top8'],
        'top8_sel_ok': bool(R['top8_sel_ok']),
        'capture8': f64(R['capture8']),
        'b_anchors': {
            'b0_diff': f64(R['b0_diff']),
            'b0_ok': bool(R['b0_ok']),
            'b1_diff': f64(R['b1_diff']),
            'b1_ok': bool(R['b1_ok']),
            'b3_ok': bool(R['b3_ok']),
            'b4_diff': f64(R['b4_diff']),
            'b4_ok': bool(R['b4_ok']),
            'b6_diff': f64(R['b6_diff']),
            'b6_ok': bool(R['b6_ok']),
            'b7a_diff': f64(R['b7a_diff']),
            'b7a_ok': bool(R['b7a_ok']),
            'b8_diff': f64(R['b8_diff']),
            'b8_ok': bool(R['b8_ok']),
            'setup_ok': bool(
                R['setup_ok_f'])},
    }
result = {
    'phase': PHASE, 'name': NAME,
    'created': created,
    'elapsed': time.time() - t0,
    'forwards': int(FW[0]),
    'run': 'run1 authoritative (qwen2.5-3b bf16 '
           'single model, families A/B/C '
           'sequential)' if not SMOKE
    else 'smoke',
    'prereg': PREREG,
    'stats': {
        'repro': (None if SMOKE
                  else REPRO),
        'families': fam_stats,
        'cross': (None if SMOKE or
                  verdict in (
                      'third_setup_failed',
                      'third_top8_degenerate')
                  else {
            'e3': {k: {'sp': f64(E3[k]['sp']),
                       'p': f64(E3[k]['p'])}
                   for k in E3},
            'g_ds_count': int(cnt_pos),
            'g_ds_n_bonf': int(n_bonf),
            'g_ds_min_sp': f64(min_sp2),
            'stouffer_z': f64(stouffer),
            'sp_ut': {k: f64(v) for k, v
                      in SP_UT.items()},
            'sp_f1f2': {k: f64(v) for k, v
                        in sp_f1f2.items()},
            'mig': {k: f64(v) for k, v
                    in mig.items()},
            'sp_as': {k: f64(v) for k, v
                      in sp_as.items()},
            'tt_med': {k: f64(v) for k, v
                       in tt_med.items()},
            'sp_fam_as_mig': f64(sp_fam_as),
            'sp_fam_tt_mig': f64(sp_fam_tt),
            'spec_class': spec_class,
            'spectrum': {
                fk: {kk: f64(vv) for kk, vv
                     in SPEC[fk].items()}
                for fk in FKEYS},
            'partial': {k: f64(v) for k, v
                        in PSP.items()},
            'ab_top_bottom': {
                'kmax': int(kmax),
                'bi_max': int(BIDX[kmax]),
                'ci_max': int(CIDX[kmax]),
                'kmin': int(kmin),
                'bi_min': int(BIDX[kmin]),
                'ci_min': int(CIDX[kmin])},
        }),
    },
    'gates': gates,
    'verdict': verdict,
}
with io.open(os.path.join(OUT, 'result.json'),
             'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False,
              indent=1)

seal = {
    'phase': PHASE, 'name': NAME,
    'created': created,
    'npz_sha256_8': sha8(npz_path),
    'result_sha256_8': sha8(
        os.path.join(OUT, 'result.json')),
    'exec_sha256_8': sha8(os.path.join(
        OUT, 'execution.json')),
    'script_sha256_8': sha8(os.path.abspath(
        __file__)),
    'verdict': verdict,
    'setup_ok': setup_ok_all,
}
with io.open(os.path.join(OUT, 'seal.json'),
             'w', encoding='utf-8') as f:
    json.dump(seal, f, ensure_ascii=False,
              indent=1)
log('sealed npz8=%s result8=%s exec8=%s '
    'script8=%s elapsed=%.1fs'
    % (seal['npz_sha256_8'],
       seal['result_sha256_8'],
       seal['exec_sha256_8'],
       seal['script_sha256_8'],
       time.time() - t0))
log('sealed')
print('RUN_COMPLETE %s' % verdict)

del model, layers, tok
gc.collect()
torch.cuda.empty_cache()
