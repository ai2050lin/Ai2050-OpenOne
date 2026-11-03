# -*- coding: utf-8 -*-
"""追加 Q9-A 一条测量 + 三条 negative。幂等（按 name 去重），链头纪律与 Q8-D 相同。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / 'research' / 'gpt5' / 'atlas' / 'atlas_ledger.json'

ENTRY = {
    'phase': 'Q9-A',
    'name': 'pat_q9a',
    'line': 'G',
    'date': '2026-11-01',
    'verdict': 'register_style_is_partly_token_identity_at_hs0_2of3_with_early_computed_increment_mid_band_mlp_dominantly_writes_the_style_increment_and_register_axis_becomes_language_abstract_by_l8_glm_late_band_rewires_language_specific',
    'res_sha8': '3f470a5b',
    'res_sha8_glm': '6df1b476',
    'note': (
        'First component-localization phase of the three-block integration (embedding / MLP / '
        'attention). 60 facts x 6 register renderings (zh/en x colloquial/academic/code-comment), '
        'deterministic fact-level disc/conf split; per layer captured resid_in, self_attn module '
        'output and mlp module output at last-token position A and control position B=A-1; '
        'residual identity asserted on all samples/layers both positions (fp16 storage floor '
        'rel<=0.0057, float64 verify identical). Qwen3-4B sha8 3f470a5b (143.6s), GLM-4-9B 8-bit '
        'sha8 6df1b476 (340.8s). Independent verify pat_q9a_verify.py: float64 accumulation, '
        'nearest-centroid classifier, norm-normalized component ratios; A-position numbers '
        'reproduce bit-identically across two runs and two npz versions. '
        'P1/V1: hs[0] 3-way 0.667 both models (null 0.32-0.34), saturates by l=1-2 (LR) / l=8 '
        '(NC curve); position B l=0 0.833 both models -> the last token (sentence punctuation) '
        'carries NO register at hs[0], style must be COMPUTED; verdict '
        'STYLE_HAS_COMPUTED_INCREMENT concentrated in first ~8 layers. Char 2-gram bag = 1.000 '
        'both models: probe absolutes are uninformative, all conclusions rest on within-stream '
        'decomposition and transfer curves. '
        'P3/V2 component attribution (style axis = discovery mean difference, confirm projections): '
        'raw share |attn|:|mlp| in the mid band (1/2..3/4 depth) is 0.39/0.25 (Qwen zh/en) and '
        '0.17/0.11 (GLM zh/en), all p<=0.01 paired permutation; norm-normalized efficiency still '
        '0.33-0.74 i.e. MLP writes more register per unit norm there too. Periphery is '
        'MODEL-SPECIFIC: GLM final quarter flips to attention (raw 0.80-0.93, normed 1.37-1.38) '
        'while Qwen keeps MLP (0.26-0.49). Verdict narrowed per D27 to '
        'MID_BAND_MLP_CARRIES + PERIPHERY_MODEL_SPECIFIC. '
        'P4/V3 transfer (nearest-centroid per-layer curves): register axis becomes '
        'LANGUAGE-ABSTRACT by l~8 in BOTH models (zh2en and en2zh reach 1.0); GLM late band '
        '(l>=28) shows en2zh decay 1.0->0.5 while zh2en stays 1.0 - late-layer re-language-'
        'specialization coinciding with its attention flip. Main-script V3 (LR at argmax-first-'
        'saturated layer l=1/2) is SUPERSEDED by the curves: transfer is layer-dependent, LR '
        'zh2en 0.50 vs NC 1.00 at the same layer - classifier choice reverses the verdict '
        '(new rules R-MET-38, R-MET-39). '
        'E1 profile, no intervention: "MLP writes" is a share of the write-increment, not '
        'computational necessity (Q2-C1 lesson). Consequence for the SAE program: mount the '
        'first dictionary on mid-band MLP outputs (Qwen L18-27 / GLM L20-30).'
    ),
}

NEGATIVES = [
    {
        'name': 'pat_q9a_neg_attention_is_the_register_carrier',
        'phase': 'Q9-A', 'date': '2026-11-01',
        'claim_refuted': 'Attention is the underlying carrier of language-pattern families, '
                         'specifically register/style (colloquial vs academic vs code) is '
                         'written into the residual stream mainly by attention.',
        'why_refuted': 'In the mid band (1/2..3/4 depth) the attention write-increment projects '
                       'onto the register axis only 0.11-0.39 of what the MLP increment does '
                       '(raw share), p<=0.01 paired permutation, in BOTH models and BOTH '
                       'languages; norm-normalized the ratio is still 0.33-0.74. Attention '
                       'dominance appears only in periphery bands and is model-specific '
                       '(GLM final quarter).',
        'what_stands': 'A narrower reading survives: attention may carry conditional READING '
                       '(fetching register cues from context) while MLP carries WRITING. This '
                       'read/write split is the preregistered next test (Q9-B), not a finding '
                       'of this phase.',
    },
    {
        'name': 'pat_q9a_neg_style_is_pure_token_identity',
        'phase': 'Q9-A', 'date': '2026-11-01',
        'claim_refuted': 'Register/style is fully present in the token embeddings (hs[0]); no '
                         'dedicated computation is needed to represent it.',
        'why_refuted': 'hs[0] 3-way register probe is 0.667 (both models, null 0.32-0.34) and '
                       'only 0.833 at the content position B; saturation needs the first ~8 '
                       'layers (+0.333 increment, 3x null sd). At position A the final token is '
                       'punctuation whose hs[0] carries no register at all (0.667) yet later '
                       'layers reach 1.000, so the missing information is computed, not stored.',
        'what_stands': 'Token identity does carry a large register prior (2/3); the claim '
                       'refuted is only completeness. Char 2-gram bag-of-words also reaches '
                       '1.000, so register is lexically visible - the computed increment is '
                       'about binding it into the residual stream, not about lexical knowledge.',
    },
    {
        'name': 'pat_q9a_neg_register_axis_language_abstract_at_every_layer',
        'phase': 'Q9-A', 'date': '2026-11-01',
        'claim_refuted': 'Once a register axis is readable, cross-language transfer of that axis '
                         'is a layer-uniform property (STYLE_ABSTRACT can be decided at any '
                         'convenient layer).',
        'why_refuted': 'Transfer is strongly layer-dependent: both models start at 0.5 at l=0 '
                       'and reach 1.0 by l~8, and GLM then DECAYS back to 0.5 for en2zh over '
                       'l=28..40 while zh2en stays 1.0. The main script picked the first '
                       'saturated layer (l=1/2) and got PARTIAL/ABSTRACT - both wrong about the '
                       'profile. LR vs nearest-centroid also reverses single-layer verdicts '
                       '(0.50 vs 1.00 at the same layer).',
        'what_stands': 'The layer-uniform claim is refuted, not the abstraction itself: by '
                       'l~8 both models DO transfer perfectly; GLM late-band re-specialization '
                       'is a model fingerprint pending full-precision re-check (8-bit caveat).',
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
