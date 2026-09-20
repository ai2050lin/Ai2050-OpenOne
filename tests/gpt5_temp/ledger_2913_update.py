# -*- coding: utf-8 -*-
"""ledger_2913_update.py -- register M2913 + refine L14."""
import hashlib
import json

P = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas'
     r'\atlas_ledger.json')
with open(P, encoding='utf-8') as f:
    L = json.load(f)

m_ids = [m.get('meas_id') for m in L['measurements']]
assert 'M2913_perhead_wvo_decomposition' not in m_ids, 'already in'
assert m_ids[-1] == 'M2912_sign_balance_zigzag', m_ids[-1]

m2913 = {
    "meas_id": "M2913_perhead_wvo_decomposition",
    "type": "perhead_wvo_decomposition",
    "verdict": (
        "margin_heads_present_alternation_absent - protocol 2903 "
        "verbatim forward per-head jacobian (o_proj-input pre-hook "
        "capture, 32 heads x 128 dim, GQA 8kv, eps=1.0, pos 1, "
        "conds same/func/null, SEED=2896, window [26,36)). "
        "EXECUTION HISTORY: runs 1-3 void on implementation bugs "
        "(o_proj-input numpy .cpu misuse; anchor Dfull built from "
        "the language direction in the wrong domain; result "
        "assembly referencing unassigned probe variables); run4 "
        "completed probes but failed the v1 cross-phase anchor "
        "gate (e2 vs 2903 B_attn 6.07e-2 >> 1e-3) - audit showed "
        "the 1e-3 caliber was MLP-channel-only in 2903 PREREG and "
        "does not transfer; v2 (runs 5) re-froze anchors as: a1 "
        "input-domain block-vs-full identity 1e-9 (measured "
        "1.22e-15), a3 margin 5e-3 / acc 2 words (measured 1.93e-3 "
        "/ 1 word), cross-phase quantities logged WITHOUT gate. "
        "Run-internal separation: e2run (this-run bf16 "
        "module-output B vs 2903 B_attn, same domain) = 3.88e-08 - "
        "the forward is deterministic across runs and the 6.07e-2 "
        "gap is RESPONSE-EXTRACTION-DOMAIN QUANTIZATION (bf16 "
        "module-output differencing vs fp32 o_proj-input "
        "differencing), not run drift; the fp32 input-domain "
        "margin (-0.02139) is the cleaner estimate of the attn "
        "channel margin. P1 (head margin spectrum, family "
        "max-null over 200 SEED=2896 label permutations): PRESENT "
        "- head 7 margin 0.17710 > p95_max 0.17474; top-5 h7 "
        "0.1771 / h8 0.1343 / h6 0.0789 / h22 0.0601 / h21 "
        "0.0554. P2 (per-head 10-layer d=1 sign-margin flips, "
        "exact binomial(9,.5), BH-FDR q=0.05): ABSENT - best head "
        "7/9 flips p=0.0898, FDR set empty; the AGGREGATE 8/9 "
        "alternation (2911 P1, p=0.0195) is stronger than any "
        "single head - alternation is a cross-head aggregate "
        "phenomenon. P3 (greedy top-k reconstruction with full "
        "selection-corrected label-permutation null): SIGNIFICANT "
        "- top-2 heads {7,8} margin 0.28036 vs full-channel "
        "-0.02139, p=0.01493; two heads flip the attn channel "
        "from negative to a margin exceeding qwen_mlp (0.1799). "
        "P4 descriptive: top sign-balance-gap heads 17/30/8 "
        "(gap_mean 0.23/0.19/0.19, pf0_range up to 0.55). KEY "
        "READING: the negative attn margin is a HEAD-LEVEL "
        "STRUCTURE - two heads carry mlp-scale positive language "
        "separation diluted by the remaining 30 heads' negative "
        "contributions; L14 spectrum refined: weak channel margin "
        "= strong heads + canceling heads, not uniformly weak "
        "response"),
    "source": {
        "path": "phase2913/perhead_wvo_decomposition/result.json",
        "sha256_8": "552a2c66",
        "phase": 2913
    }
}
L['measurements'].append(m2913)

l14 = None
for c in L['linkage']:
    if c.get('link_id') == 'L14_readout_spectrum_cross_model':
        l14 = c
        break
assert l14 is not None
assert 'M2913_perhead_wvo_decomposition' not in l14['connects']
l14['connects'].append('M2913_perhead_wvo_decomposition')
l14['notes'] += (
    " | 2913 per-head decomposition: the attn negative margin is "
    "head-level structure - h7 (0.177) and h8 (0.134) carry "
    "mlp-scale language separation, top-2 subset margin 0.280 "
    "(selection-corrected p=0.015) vs full -0.021; the 2911 "
    "margin-sign alternation is NOT single-head (best head 7/9 "
    "vs aggregate 8/9) - cross-head aggregate phenomenon; "
    "methodological constants: attn-channel anchor thresholds do "
    "not transfer from mlp (bf16 extraction-domain quantization "
    "gap 6.07e-2 vs true cross-run drift 3.88e-8, forward "
    "deterministic)")
l14['phase_updated'] = 2913

with open(P, 'w', encoding='utf-8') as f:
    json.dump(L, f, indent=1, ensure_ascii=False)

h = hashlib.sha256(open(P, 'rb').read()).hexdigest()[:8]
out = ('ledger updated: measurements %d, L14 connects %d, '
       'file sha256_8 %s'
       % (len(L['measurements']), len(l14['connects']), h))
with open(r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\ledger_2913_report.txt', 'w',
          encoding='utf-8') as f:
    f.write(out + '\n')
print('OK ledger 2913')
