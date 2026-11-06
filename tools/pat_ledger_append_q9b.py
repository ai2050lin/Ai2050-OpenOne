# -*- coding: utf-8 -*-
"""追加 Q9-B（Qwen 侧）一条测量 + 三条 negative。幂等（按 name 去重）。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / 'research' / 'gpt5' / 'atlas' / 'atlas_ledger.json'

ENTRY = {
    'phase': 'Q9-B',
    'name': 'pat_q9b_qwen_patch_w6',
    'line': 'G',
    'date': '2026-11-03',
    'verdict': 'register_direction_is_sufficient_not_necessary_and_component_agnostic_'
               'weight_space_gate_surgery_is_nonnull_but_negligible_glm_arms_pending',
    'res_files': ['tests/gpt5/result/pat_q9b_qwen_patch.json',
                  'tests/gpt5/result/pat_q9b_qwen_w6.json',
                  'tests/gpt5/result/pat_q9b_w6_precheck.json'],
    'note': (
        'Q9-B Qwen3-4B arms (bf16, frozen Q9-A material + frozen npz directions; GLM W4/W5 pending). '
        'Readout probe: 3-way register LR (C=1.0, max_iter=3000) fitted on discovery resid_in[35], '
        'applied to 180 confirm samples; intervention at last-token position, MLP/attn module outputs, '
        'mid band L18-27; alpha calibrated in units of median |mlp_out . v| per layer. '
        'PRE-CHECK (CPU, no forward): c_j = v . W_out[:,j] on real down_proj weights. Register direction '
        'energy in top-64 of 9728 columns S64=0.116+-0.016 vs random-direction null 0.063-0.066 '
        '(ratio 1.5-2.3x, z~22 but effect small); participation ratio 1050-2217 columns, k50~950, '
        'k80~2750. => DISTRIBUTED_WRITE_CANDIDATE, NOT a rank-1 gate. '
        'W1 sufficiency (MLP patch): dose response in d_margin +2.83/+5.69/+11.21/+20.11 for '
        'alpha=0.5/1/2/4; KL 0.024/0.095/0.373/1.124 vs random-direction null KL p95=0.003 '
        '(~30x above null at alpha=1); 3-way flip toward +s reaches 0.333 at alpha>=2 with acc '
        'dropping 1.000->0.667. SUFFICIENT. '
        'W2 necessity (ablate v-projection from mid-band MLP outputs): acc stays 1.000, d_margin '
        '-0.92, KL 0.072 (null KL 0.001) => direction is READ but NOT NECESSARY. '
        'W2b joint ablation (MLP+attention simultaneously): acc 0.989, d_margin -1.58, KL 0.114 - '
        'removing the direction from both components still barely changes the readout. '
        'W3 read/write separation: attention patch with the SAME direction and alpha=1 gives '
        'd_margin +6.20 (vs MLP +5.69) and KL 0.129 (vs 0.095): attention writes the register as '
        'effectively as MLP at the residual level => ATT_ALSO_WRITES, hypothesis H-RW REFUTED in its '
        'strong form. W3b attention ablation d_margin -0.61, KL 0.025. '
        'W6 weight-space gate surgery (WSP-inspired): W_out <- W_out - u u^T W_out on L18-27, '
        'dose by column count k=64/512/2048 vs random-column control. Edit is NON-NULL but tiny: '
        'KL 0.0251/0.0578/0.0792 vs random 0.0009/0.0015/0.0073 (11-39x); probe acc stays 1.000 and '
        'class flip 0.000 at every dose including full removal (KL 0.0908). Capability gate '
        '(440-item 4-set battery, forced-choice at prompt-final position): 0.693 base -> 0.698 edited '
        '(305->307/440), no damage (McNemar n.s.). '
        'Verdict: SUFFICIENT_NOT_NECESSARY + ATT_ALSO_WRITES + DISTRIBUTED_WRITE. '
        'Interpretation: register/style is an additive READABLE direction in the residual stream, not a '
        'closable rank-1 gate; the WSP single-direction surgery works for degenerate self-loop failures '
        '(their Qwen loop) but not for register style, exactly matching their own Gemma failure case. '
        'Consistent with Q2 (no necessary unit), Q5-A (weak-evidence accumulation) and Q9-A (hs[0] '
        'already carries 2/3 of the register discriminability).'
    ),
}

NEGATIVES = [
    {
        'name': 'pat_q9b_neg_register_is_a_closable_gate',
        'phase': 'Q9-B', 'date': '2026-11-03',
        'claim_refuted': 'The register/style direction can be removed from the weights like the '
                         'WSP loop direction (rank-1 surgery, 0.016%-style edit) so that behaviour '
                         'switches off.',
        'evidence': 'Weight-space projection removal on Qwen3-4B L18-27 leaves the 3-way readout '
                    'unchanged (acc 1.000, class flip 0.000) and only shifts next-token KL by '
                    '0.025-0.091 vs 0.001-0.007 for random-column edits; the 440-item capability '
                    'battery is unchanged (0.693 -> 0.698). Pre-check shows the direction energy is '
                    'spread over ~950-2750 of 9728 columns (S64=0.116 vs null 0.065), not a rank-1 '
                    'structure.',
        'what_stands': 'The direction is measurable in the weights (edit above random control) and '
                       'sufficient when ADDED at activation level (dose-dependent d_margin up to 20, '
                       'KL 1.12). Readability and additivity stand; closability does not.',
    },
    {
        'name': 'pat_q9b_neg_attention_reads_mlp_writes',
        'phase': 'Q9-B', 'date': '2026-11-03',
        'claim_refuted': 'Read/write separation: attention only reads contextual register cues while '
                         'MLP writes the register increment (so patching MLP should move the readout '
                         'much more than patching attention).',
        'evidence': 'Same direction, same calibrated alpha=1, same layers: attention patch d_margin '
                    '+6.20 / KL 0.129 vs MLP patch +5.69 / KL 0.095 - attention is at least as '
                    'effective. Ablations: MLP-only -0.92, attention-only -0.61, joint -1.58 '
                    '(acc 0.989).',
        'what_stands': 'Q9-A share result stands (natural write increment in mid band is MLP-'
                       'dominated, raw 0.11-0.39). What falls is the inference from share to '
                       'component-specific controllability: the residual stream is additive, so any '
                       'component injected with the same vector moves the readout similarly. '
                       'Component attribution must be done by ablation, not by patch magnitude.',
    },
    {
        'name': 'pat_q9b_neg_register_direction_is_necessary',
        'phase': 'Q9-B', 'date': '2026-11-03',
        'claim_refuted': 'The register direction is necessary for the model to represent/compute '
                         'register: removing its projection from the mid-band writes should damage '
                         'register classification (pre-registered W2 gate: acc drop >= 0.10).',
        'evidence': 'MLP ablation acc 1.000 (drop 0.000, gate FAIL); joint MLP+attention ablation '
                    'acc 0.989 (drop 0.011); weight-space removal acc 1.000. Margin shifts are '
                    '10-100x above the random-direction null but far below the pre-registered gate.',
        'what_stands': 'The direction is causally READ (non-null ablation effect, KL 0.072-0.114 vs '
                       'null 0.001) but redundant: other channels (hs[0] identity, remaining layers, '
                       'positions) carry the same information. Record as NULL_BOUNDED for necessity, '
                       'not as evidence of no effect.',
    },
]


def main() -> int:
    doc = json.loads(LEDGER.read_text(encoding='utf-8'))
    names = {m.get('name') for m in doc['measurements']}
    if ENTRY['name'] not in names:
        prev = doc['ledger_sha256_8']
        e = dict(ENTRY)
        e['prev_post'] = prev
        doc['measurements'].append(e)
        print('appended measurement %s (prev=%s)' % (e['name'], prev))
    else:
        print('measurement already present, skipped')
        prev = doc['measurements'][-1]['prev_post']

    have = {n.get('name') for n in doc['negatives']}
    for n in NEGATIVES:
        if n['name'] not in have:
            doc['negatives'].append(n)
            print('appended negative %s' % n['name'])
        else:
            print('negative %s already present' % n['name'])

    doc['ledger_sha256_8'] = prev
    doc['ledger_sha256_8'] = hashlib.sha256(
        json.dumps(doc, ensure_ascii=False, indent=1).encode('utf-8')).hexdigest()[:8]
    LEDGER.write_text(json.dumps(doc, ensure_ascii=False, indent=1),
                      encoding='utf-8', newline='')
    chk = json.loads(LEDGER.read_text(encoding='utf-8'))
    tmp = dict(chk)
    tmp['ledger_sha256_8'] = prev
    rec = hashlib.sha256(json.dumps(tmp, ensure_ascii=False, indent=1).encode('utf-8')).hexdigest()[:8]
    print('n=%d neg=%d head=%s recomputed=%s match=%s'
          % (len(chk['measurements']), len(chk['negatives']), chk['ledger_sha256_8'], rec,
             rec == chk['ledger_sha256_8']))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
