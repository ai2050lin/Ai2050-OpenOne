# -*- coding: utf-8 -*-
"""追加 Q8-B 一条测量 + 三条 negative（幂等）。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / 'research' / 'gpt5' / 'atlas' / 'atlas_ledger.json'

ENTRY = {
    'phase': 'Q8-B', 'name': 'pat_q8b', 'line': 'G', 'date': '2026-11-01',
    'verdict': 'frequency_is_a_low_dim_dense_subspace_with_log_scale_and_orthogonal_to_norm_and_semantics',
    'res_sha8': '318259ee',
    'res_sha8_glm': '060de60e',
    'note': (
        'Dimensionalising the "frequency knob" that Q8-A left as a rank correlation. Seven '
        'pre-registered criteria (PAT_Q8B_PREREG_v1.md), pure static, no forward pass: Qwen3-4B '
        '63 s (n=24421, d=2560) and GLM-4-9B 84 s (n=27215, d=4096); independent verify 20/20 on '
        'both. POOL = 8049 single CJK chars + 16372 whole CJK words that have a wordfreq zipf '
        '(1724 byte fragments excluded from every regression because zipf is undefined for them). '
        'CORRECTION OF Q8-A (the main product of this round): the single-coordinate |rho|=0.802 was '
        'a MATERIALS artefact. It came from merging four hand-picked sets (52 colour + 50 fruit + 74 '
        'rare + 52 common, n~228) whose between-set frequencies span four orders of magnitude, so any '
        'coordinate that merely separates the groups scores high. On the full pool the best single '
        'coordinate is 0.520 (Pearson) / 0.548 (Spearman) in Qwen and 0.590 / 0.598 in GLM, i.e. '
        'R^2 0.195 against 0.826 for the full-space direction - a 4.2x gap. "One knob" is a group '
        'label read as a coordinate. New rule R-MET-34. '
        'B1 SCALE: linear fit of zipf on the frequency direction gives held-out R^2 0.8263 (Qwen) / '
        '0.7214 (GLM); the same X and folds fitted on RAW FREQUENCY give 0.0312 / 0.0365, a gap of '
        '0.795 / 0.685. The scale is LOGARITHMIC - the table does not order by probability, it '
        'orders by log-probability. Quadratic 0.8268/0.7205 and 12-bin 0.8118/0.6916 both fail to '
        'beat linear by the pre-registered 0.02 (delta -0.0144 / -0.0298) => LINEAR_LOG_SCALAR on '
        'both models. '
        'B2 DIMENSION: two instruments disagree and both are reported. Coordinate-sparse curve needs '
        'k=512-1024 coordinates to reach 0.69-0.77 and never saturates; the Krylov subspace (PLS-1 '
        'equivalent) saturates at k=4 already (0.805 Qwen / 0.736 GLM) and is flat afterwards '
        '(0.827 at k=32). Verdict AMBIGUOUS_DIM with sub-label LOW_DIM_DENSE: the frequency '
        'direction is a 4-16 dimensional DENSE subspace - energy concentrated in a few combinations, '
        'but no single coordinate carries it. '
        'B3 NOT NORM: rho(log||e||, zipf)=+0.011 (GLM +0.315), cos(w_freq,w_norm)=+0.083 (+0.242), '
        'and after regressing log||e|| out of X the R^2 KEEPS 1.00 (Qwen 0.8261->0.8268) / 1.03 '
        '(GLM 0.7214->0.7416). The "norm predicts readability" result of the project concerns the '
        'CONTEXTUAL residual stream (Q5-A pos0 register), not the static table: here frequency is a '
        'direction, not a magnitude. '
        'B6 SEMANTICS IS NOT THE FREQUENCY RESIDUAL: colour within 0.1988->0.1982 and fruit '
        '0.0975->0.0975 after removing the frequency component (retain 100%; random CJK null '
        '0.0040/0.0037); GLM 0.1436->0.1430 and 0.0772->0.0773 (null 0.0156/0.0155). Static '
        'embeddings carry a statistical structure and a semantic structure in nearly orthogonal '
        'subspaces. '
        'B5 TURNING THE KNOB IS A MODEL FINGERPRINT, NOT A UNIVERSAL: pushing a token along '
        'w_freq by 0.25/0.5/1/2 zipf units and reading the nearest neighbour (search set = rows '
        'that have a defined frequency + fragments) gives a monotone dose-response in Qwen '
        '(+0.085/+0.330/+0.670/+1.440, sign-correct 0.53/0.59/0.65/0.72) => SMOOTH_KNOB (one unit '
        'of push moves you 0.67 units), but in GLM every dose lands at +3.0~3.3 zipf with sign '
        'correctness 0.86-1.00 => KNOB_NOT_LOCAL: GLM frequency direction is DECOUPLED from '
        'neighbourhood structure, it points at the high-frequency wall (function-word band) '
        'regardless of dose. '
        'B7 TYPE IS ONLY PARTLY FREQUENCY, AND ONLY IN QWEN: a single scalar t separates CHAR from '
        'WORD with AUC 0.748 in Qwen (=> about 3/4 of the distinction is frequency) but only 0.578 '
        'in GLM (=> essentially none of it is). Q8-A said "the category backbone is mostly '
        'frequency"; that sentence must be narrowed to "about 3/4, and only in Qwen". '
        'CROSS-MODEL: B1/B2/B3/B6 all isomorph (same verdicts, |dR^2|<0.15) => SHAPE_ISOMORPHIC; '
        'B5 and B7 diverge => post-hoc label FUNCTION_MODEL_SPECIFIC. How the table is laid out is '
        'shared; whether the knob can be turned is each model own business. '
        'LATIN control (n=1500, 80/20 single split): R^2 0.5102 (Qwen) / 0.5347 (GLM), same '
        'LINEAR_LOG_SCALAR shape => the log scale is not CJK-specific, but it is 0.6-0.7x weaker. '
        'TWO SELF-INFLICTED BUGS FOUND BY THE VERIFIER and both fixed before the numbers above '
        '(see errata): (1) B1/Krylov had the train-test mask inverted, so every B1 R^2 was '
        'in-sample (0.8650 -> 0.8263 after the fix); (2) the Gram cache key did not include the '
        'fold assignment, so the null distribution (which uses a different fold split) poisoned '
        '(\'f\',\'main\',j) and the full-dimensional R^2 read 0.776 instead of 0.826. New rules '
        'R-MTH-24 (cache keys must carry the fold fingerprint) and R-MTH-25 (every held-out R^2 '
        'needs an independent recomputation through a different code path, and the gap must be '
        'reported). All E1.'
    ),
}

NEGATIVES = [
    {
        'name': 'pat_q8b_neg_single_coordinate_carries_frequency', 'phase': 'Q8-B',
        'date': '2026-11-01',
        'claim_refuted': 'Frequency is carried by a single embedding coordinate (|rho|~0.80), so the '
                         'table has a one-parameter frequency backbone.',
        'why_refuted': 'The 0.802 of Q8-A came from four hand-picked material sets whose between-set '
                       'frequencies span four orders of magnitude. On the full pool of tokens that '
                       'actually have a wordfreq entry (24421 Qwen / 27215 GLM) the best single '
                       'coordinate reaches Pearson -0.520 / -0.590 (Spearman -0.548 / -0.598), '
                       'R^2 0.195, against 0.826 / 0.721 for the full-space direction. Coordinate '
                       'curves need 512-1024 coordinates to reach 0.69-0.77 and never saturate.',
        'what_stands': 'A frequency direction exists and is strongly linear in log frequency (B1); it '
                       'is just not sparse in the coordinate basis.',
    },
    {
        'name': 'pat_q8b_neg_knob_turns_in_every_model', 'phase': 'Q8-B', 'date': '2026-11-01',
        'claim_refuted': 'Because the embedding encodes frequency as a direction, translating a token '
                         'along that direction moves it to a frequency neighbour, in any model.',
        'why_refuted': 'Dose-response (0.25/0.5/1/2 zipf units) gives a monotone, sign-correct '
                       'response in Qwen (+0.085/+0.330/+0.670/+1.440) but a DOSE-INDEPENDENT '
                       'constant +3.0~3.3 zipf in GLM with sign correctness 0.86-1.00. GLM lands on '
                       'the high-frequency wall no matter how far it is pushed, so the direction is '
                       'not attached to the neighbourhood structure there.',
        'what_stands': 'In Qwen one unit of push moves the nearest neighbour about 0.67 zipf, which is '
                       'the only positive evidence that a causal intervention (Q8-D) is worth trying.',
    },
    {
        'name': 'pat_q8b_neg_semantics_is_frequency_residual', 'phase': 'Q8-B', 'date': '2026-11-01',
        'claim_refuted': 'Once the frequency direction is removed, what is left of the categorical '
                         'structure (colour, fruit) is small - semantics is the residual of a '
                         'statistical scalar.',
        'why_refuted': 'Removing the frequency component leaves the within-set similarity essentially '
                       'untouched: colour 0.1988 -> 0.1982 and fruit 0.0975 -> 0.0975 in Qwen '
                       '(retain 100%), 0.1436 -> 0.1430 and 0.0772 -> 0.0773 in GLM. Random CJK '
                       'subsets of the same size sit at 0.0040/0.0037 and 0.0156/0.0155.',
        'what_stands': 'Statistical and semantic structure occupy nearly orthogonal subspaces in the '
                       'static table, so "the category backbone is mostly frequency" is a statement '
                       'about separability (AUC 0.748), not about content.',
    },
]

ERRATA = {
    'corrects': 'Q8-A',
    'phase': 'Q8-B',
    'date': '2026-11-01',
    'note': ('Q8-A P7 reported |rho(single coordinate, zipf)| = 0.802 (GLM 0.774) and concluded '
             '"frequency is one explicit knob" and "the category backbone is mostly frequency". '
             'Q8-B shows both numbers are materials artefacts: on the full pool of frequency-bearing '
             'tokens the single-coordinate correlation is 0.520-0.590 (R^2 0.195) and the full-space '
             'direction gives 0.826/0.721; the type-separability AUC is 0.748 in Qwen but 0.578 in '
             'GLM. Q8-A wording is narrowed to: frequency is a 4-16 dimensional dense subspace with a '
             'logarithmic scale; "mostly frequency" holds to about 3/4 and only in Qwen.'),
}


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

    have = {n.get('name') for n in doc['negatives']}
    for n in NEGATIVES:
        if n['name'] not in have:
            doc['negatives'].append(n)
            print('appended negative %s' % n['name'])
        else:
            print('negative %s already present' % n['name'])

    if not any(e.get('phase') == 'Q8-B' and e.get('corrects') == 'Q8-A'
               for e in doc.get('errata_ledger', [])):
        doc['errata_ledger'].append(ERRATA)
        print('appended errata')

    # 所有内容改动完成后再算链头
    prev = doc['measurements'][-1]['prev_post']
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
