# -*- coding: utf-8 -*-
"""p3149 patch1: header (docstring, NAME,
SMOKE env, D48, OUT, ckpt). Idempotent."""
import io

FP = (r'D:\AI2050\Ai2050-OpenOne\tests'
      r'\glm5\phase3149_omega_p147_'
      r'carrier_dlogit_poslate_kdose_'
      r'v3amp.py')
s = io.open(FP, encoding='utf-8').read()


def rep(old, new, tag, req=True):
    global s
    n = s.count(old)
    if n == 0 and not req:
        print('SKIP', tag)
        return
    assert n == 1, (tag, n)
    s = s.replace(old, new)
    print('OK', tag)


# 1) docstring
OLD_DOC = '''# -*- coding: utf-8 -*-
"""Phase 3148 (Omega-P146): tail sign
crossing fill (TAIL25 pos/neg mid-dose
d{2.5,3,3.5} -> locate the sign-crossing
dose d* between d2 (pos>neg, 3145/3147)
and d4 (pos<neg, 3146) + fstep/first
structure of pos_d1 vs neg_d4 flip
rows -> format-channel vs answer-side
switching) + top-2 super-source
write-side identity (order_ex[:2]=
{2530,3755} solo+pair dose d{0.5,1,2}
@L17 + L17 per-head o_proj-input
ablation over 32 heads -> per-head
contribution to the two coordinates +
dvec29/dvec19 top-50 overlap) +
unembed-cancel causality (neg d0.5
@L38 allstep + L39 projection removal
of w131401 vs random dir vs w_dn ->
is the format shift a sufficient cause
of the flip) + v1 perturbation
symmetry (amp alpha{-0.25,-0.5} vs
clip alpha{+0.25,+0.5} @L38 all ->
amplitude dependence vs direction
specificity).

Preregistered in 3147 closeout
(Omega-P146). 18 bit-anchor replays:
11 from 3147 (b_pc1/b_dvec29/b_joint,
d_tbot15_d4/d_ttop10_d4, d_co50ex
d{1,2,4}, t_pc1_inst, e_pcres38_d1
pos/neg) + v2_clip_a025/a050 (3146
windows, 2nd replay) + tail sign
curve bits (s_tailpos_d1.0/d_tail_d4.0
3rd cross-phase, s_tailpos_d4.0 +
d_tail_d1.0/d_tail_d2.0 2nd replay).
Bank shards REUSED from 3138. Frozen
dvecs from 3135 (sha-anchored).
Session baselines P+A1 with xphase
recording. Frozen before observation."""'''
NEW_DOC = '''# -*- coding: utf-8 -*-
"""Phase 3149 (Omega-P147): flip-carrier
dlogit per-token decomposition (neg_d0.5
8 flip rows + pos_d1 13 flip rows, step
0-1 dlogit top-20 token sets -> common
elevated tokens beyond 131401) + pos_d1
late-onset mechanism (per-step L38/39
capture: w_dn proj vs dlogit131401 vs
|dh|, + firstk prefix-cut gen k{0..4} ->
readout competition vs injection delay
vs format growth) + coordinate-dose
interchange (co50ex top-5/top-10 @
d{0.5,1,2} vs full dose curve ->
breadth-strength exchangeability) + v1
micro-amp fill alpha{-0.05,-0.10,-0.15}
@L38 all -> bias dose dependence vs
3147 micro clip.

Preregistered in 3148 closeout
(Omega-P147). 8 bit-anchor replays:
b_pc1/b_dvec29/b_joint (3142 chain),
d_co50ex_d2.0 (3147), s2_tailpos_d1/
s2_tailneg_d1 (3146/3147), v2_clip_a025
(3146), n_neg_d0.5 (3146). Bank shards
REUSED from 3138. Frozen dvecs from
3135 (sha-anchored). Session baselines
P+A1 with xphase recording. Frozen
before observation."""'''
rep(OLD_DOC, NEW_DOC, 'docstring')

# 2) NAME
rep("NAME = ('omega_p146_xcross_headsrc_'\n"
    "        'uncancel_v1sym')",
    "NAME = ('omega_p147_carrier_dlogit_'\n"
    "        'poslate_kdose_v3amp')",
    'NAME')

# 3) SMOKE env
rep("SMOKE = os.environ.get('P3148_SMOKE',\n"
    "                       '') == '1'",
    "SMOKE = os.environ.get('P3149_SMOKE',\n"
    "                       '') == '1'",
    'SMOKE env')

# 4) D48 after D47
rep("      r'negmech_v1micro'\nMDIR_G",
    "      r'negmech_v1micro'\n"
    "D48 = RDIR + r'\\phase3148' \\\n"
    "      r'\\omega_p146_xcross_headsrc_' \\\n"
    "      r'uncancel_v1sym'\nMDIR_G",
    'D48')

# 5) OUT phase dir
rep("OUT = os.path.join(RDIR, "
    "'phase3148', NAME)",
    "OUT = os.path.join(RDIR, "
    "'phase3149', NAME)",
    'OUT', req=False)
rep("OUT = os.path.join(RDIR, "
    "'phase3148',\n"
    "                   NAME)",
    "OUT = os.path.join(RDIR, "
    "'phase3149',\n"
    "                   NAME)",
    'OUT-2', req=False)

# 6) ckpt name
rep("CKPTF = os.path.join(OUT, "
    "'p146_ckpt.pkl')",
    "CKPTF = os.path.join(OUT, "
    "'p147_ckpt.pkl')",
    'CKPTF')

io.open(FP, 'w', encoding='utf-8',
        newline='\n').write(s)
print('PATCH1A_DONE')
