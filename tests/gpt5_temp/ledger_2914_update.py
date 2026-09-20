# -*- coding: utf-8 -*-
"""ledger_2914_update.py -- register M2914 + refine L14."""
import hashlib
import json

P = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas'
     r'\atlas_ledger.json')
with open(P, encoding='utf-8') as f:
    L = json.load(f)

m_ids = [m.get('meas_id') for m in L['measurements']]
assert 'M2914_head_identity_replication' not in m_ids, 'already in'
assert m_ids[-1] == 'M2913_perhead_wvo_decomposition', m_ids[-1]

m2914 = {
    "meas_id": "M2914_head_identity_replication",
    "type": "head_identity_replication",
    "verdict": (
        "head_identity_reproduced - Run B re-ran the 2913 per-head "
        "protocol verbatim (SEED=2896, eps=1.0, pos 1, window "
        "[26,36), 57 words verbatim 2887, o_proj-input capture): "
        "B_heads cross-run rel err 3.16e-08 (fp32 capture domain, "
        "same magnitude as the 2913 e2run 3.88e-08), Spearman AND "
        "Kendall of margins_h both exactly 1.000000, top2 {7,8} "
        "identical, family gate reproduced (h7 0.17710 > p95 "
        "0.17474, same numbers as 2913), {7,8} subset margin "
        "0.280360 vs ref 0.28036 (absdiff 0.0), flips_h identical "
        "(P2 equality, p_flip_min 0.089844), P3 selection-corrected "
        "p=0.01493 reproduced (k=2 heads [7,8]). Anchors 5/5: a1 "
        "block identity 1.22e-15; a2 margin -0.02139 vs 2903 stored "
        "-0.01946 (1 word); a3 composite W_VO spectra vs 2903 "
        "weights_descriptive all 10 layers, worst diff 4.992e-06 "
        "(2903 caliber verbatim: M fp64 zeros, fp32@fp64->fp64 "
        "matmul, fp64 svd); a4a per-head zf identity sum_h va_h vs "
        "composite 3.08e-15; a4b thin-SVD self-check 2.96e-15. "
        "EXECUTION HISTORY: run1 void - naive sv comparison "
        "broadcast (128,) vs (200,) (A@Bm 300x200 has 72 exact-zero "
        "tail singular values; fixed by comparing leading-128). "
        "P4 (descriptive, 10-layer window): carrier heads carry NO "
        "static spectral identity - t12 rank h7=30/32 h8=4/32, PR "
        "rank h7=18 h8=32, zf_gain rank h7=14 h8=30, zf_cos rank "
        "h7=16 h8=18; Spearman(spectral stat, margins_h) ~ 0 (t12 "
        "0.0, PR -0.0114, zf_gain -0.3046, zf_cos -0.1785): the "
        "per-head margin carriers are determined by RUNTIME "
        "response (data-dependent o_proj input r_c), not by static "
        "weight spectra. KEY READING: h7/h8 head identity now "
        "formally confirmed (2913 hard-constraint clause "
        "discharged) - the attn channel's positive language "
        "separation is carried by two specific, exactly "
        "reproducible heads whose identity is invisible in the "
        "static weights and lives only in the data-dependent "
        "response"),
    "source": {
        "path": "phase2914/head_identity_replication/result.json",
        "sha256_8": "a0142e60",
        "phase": 2914
    }
}
L['measurements'].append(m2914)

l14 = None
for c in L['linkage']:
    if c.get('link_id') == 'L14_readout_spectrum_cross_model':
        l14 = c
        break
assert l14 is not None
assert 'M2914_head_identity_replication' not in l14['connects']
l14['connects'].append('M2914_head_identity_replication')
l14['notes'] += (
    " | confirmed by M2914: 2913 per-head carriers h7/h8 reproduce "
    "EXACTLY under protocol-verbatim re-run (Spearman/Kendall 1.0, "
    "top2 {7,8}, {7,8} reconstruction absdiff 0.0, relB 3.16e-08, "
    "gate h7 0.17710 > 0.17474 same numbers) - head-level margin "
    "attribution formally confirmed; P4: carriers have no static "
    "W_VO spectral identity (spectra-vs-margins rho ~ 0 across "
    "t12/PR/zf_gain/zf_cos), identity is runtime-response-"
    "determined, invisible in static weights")
l14['phase_updated'] = 2914

with open(P, 'w', encoding='utf-8') as f:
    json.dump(L, f, indent=1, ensure_ascii=False)

h = hashlib.sha256(open(P, 'rb').read()).hexdigest()[:8]
out = ('ledger updated: measurements %d, L14 connects %d, '
       'file sha256_8 %s'
       % (len(L['measurements']), len(l14['connects']), h))
with open(r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\ledger_2914_report.txt', 'w',
          encoding='utf-8') as f:
    f.write(out + '\n')
print('OK ledger 2914')
