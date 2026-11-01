# -*- coding: utf-8 -*-
"""追加 Q7-A 一条测量 + 三条 negative。幂等（按 (phase,name)/(name) 去重）。"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / 'research' / 'gpt5' / 'atlas' / 'atlas_ledger.json'

NOTE = (
    'Is the relation between two clauses (sequence / contrast / cause) represented as a rule that can be '
    'PREDICTED from the residual stream independently of what the clauses are about? This is the first '
    'phase in the project that abandons the location question entirely: twelve earlier rounds all asked '
    '"where is X stored" and all returned NULL, so Q7-A asks "what rule is executed" with a predictive '
    'criterion whose error cannot be rescued by slicing (D24). '
    'DESIGN (frozen, PAT_Q7A_PREREG_v1.md): 3 relations x 18 contents x 3 surface forms = 162 sentences. '
    'Each content has two topic words that appear in clause A and in ALL THREE clause-B variants, so the '
    'topic is matched and only the predicate stance varies; the 18 topic pairs are pairwise disjoint, '
    'machine-asserted (0 conflicts). Surface S1 = explicit connective + comma, S2 = connective + period, '
    'S3 = implicit, NO connective at all. Connectives rotate by content index so train and test folds '
    'never share a connective for the same relation. '
    'MEASURED at 3 positions x 37 layers: P_END (sequence final token, primary), P_CONN (connective '
    'first token, explicit only), P_AE (end of clause A). '
    'PROTOCOL: linear ridge on standardised hidden states, 6 folds by content (135 train / 27 test), '
    'alpha frozen from an inner 5-fold on layer 17, l* selected by inner 5-fold INSIDE each outer training '
    'set, cluster bootstrap over content (18 units). '
    'ALIGNMENT GATE J0 PASSES EXACTLY: at P_AE the three relation variants of a (content, surface) cell '
    'have a byte-identical prefix, so accuracy must be exactly 1/3 - measured 0.333333 on all three '
    'surfaces, deviation 0.00e+00. '
    'RESULT J1: held-out-content 3-class accuracy 0.9136, cluster CI [0.8272, 0.9815], +0.5802 above the '
    'three-way floor 0.3333. Per relation: sequence 0.9815, contrast 0.8889, cause 0.8704. Per surface: '
    'S1 0.9074, S2 0.9444, S3 0.8889. Selected layers per fold: 15/13/16/17/20/14. '
    'RESULT J2 (surface decoupling): train on S1+S2 (with connectives), test on S3 (connectives removed), '
    'same content folds -> 0.8889, CI [0.7778, 0.9815]. Reverse 0.9167. Punctuation minimal pair '
    'S1->S2 = 0.9444. So the code is NOT the connective token: removing every connective costs 2.5 '
    'points, while changing only a comma to a period costs nothing. P_CONN itself is 1.0000, as it must '
    'be (the token IS the marker), which is exactly why P_CONN cannot be the evidence. '
    'RESULT J3 (reuse structure): truncating the fitted probe W (d x 3) to rank 1 gives 0.6111, rank 2 '
    'gives 0.9136 = full rank 3. One axis carries 61% of the 91%, a SECOND dimension is required for the '
    'rest. Verdict PARTIAL_SHARING. '
    'NULL: 10 permutations that re-assign labels independently per content (so no consistent relabeling '
    'exists) give mean 0.3265, p95 0.4318 - the pipeline collapses to the floor as it should. '
    'SURFACE BASELINES, added after the fact and reported in full: character bag over the whole sentence '
    'reaches 0.8025 with the same folds; a two-number length baseline reaches 0.3704 (so length is not '
    'the cue); a SINGLE binary "clause B contains a negation character" reaches 0.6481. '
    'PAIRED TEST: hidden state 148 correct vs char bag 130 correct, discordant 28 vs 10, McNemar exact '
    'p = 0.0051. The residual stream carries real information beyond surface lexicon, but only 11 points '
    'of it. '
    'LAYER PROFILE (all 37 values recorded, D25): layer 0 = 0.3333, layer 1 already 0.7840, peak 0.9506 '
    'at layers 16 and 22, still 0.8951 at layer 35. The relation is largely available from layer 1, which '
    'is consistent with the char-bag baseline; the compositional gain over the raw bag is concentrated in '
    'layers 2-16 (+0.17). '
    'GEOMETRY (descriptive, full-fit, layer 16): the three class centroids sit at the vertices of an '
    'EQUILATERAL triangle - pairwise cosines relative to the centroid -0.501/-0.499/-0.500, first-axis '
    'variance share 0.501. The three relations are symmetric and equidistant; none sits "between" the '
    'others, and there is no ordering axis. This is the geometric counterpart of J3 rank-2. NOTE: this '
    '-0.5 is the signature of a centred equilateral triple and must NOT be conflated with Q6-B2\'s -0.5 '
    'isotropic baseline for difference vectors; same number, unrelated quantity. '
    'VERDICT: DECODABLE_NOT_SHARED. A content-invariant, surface-invariant, predictive linear readout of '
    'discourse relation exists; it is NOT a single shared direction (one axis stops at 0.61) but a '
    'two-dimensional, three-pole arrangement.'
)

NEG_RANK3 = dict(
    name='pat_q7a_neg_3class_rank3_is_degenerate',
    phase='Q7-A', date='2026-11-01',
    claim_refuted='A rank-k truncation curve over a 3-class probe can adjudicate SHARED_AXIS vs '
                  'PARTIAL_SHARING vs PRIVATE_DIRECTIONS, because k=3 means "three private directions".',
    why_refuted='Three class centroids always span at most TWO dimensions of discriminative subspace, so '
                'rank 3 is structurally unreachable for a 3-class problem (it equals rank 2 unless the '
                'class covariances differ). The pre-registered table therefore could never output '
                'PRIVATE_DIRECTIONS and was biased toward PARTIAL_SHARING. Measured rank2 = 0.9136 = '
                'rank3. The only informative comparison is rank-1 (0.6111) vs rank-2, which is what the '
                'conclusion actually uses.',
    fix='Rank curves must be stated relative to the number of classes minus one; for C classes the '
        'saturated rank is C-1 and must be measured as such, or the design must use >= 4 classes.',
)

NEG_G0 = dict(
    name='pat_q7a_neg_behavioral_gate_shopping',
    phase='Q7-A', date='2026-11-01',
    claim_refuted='A behavioural readout can certify that the hand-written stimuli really express the '
                  'intended discourse relation, i.e. can serve as a pass/fail validity gate before the '
                  'representation measurement.',
    why_refuted='Three behaviourally reasonable readouts of the SAME 54 pairs disagreed: metalinguistic '
                '3-way choice 0.5926 (but contrast only 0.278 - the model almost never emits the label '
                '"contrast"), connective fit by full-sequence log-likelihood 0.4444, and connective fit '
                'by clause-B conditional log-likelihood after removing the per-connective bias 0.4815 '
                '(contrast 0.611). The first is dominated by a label prior, the second and third by how '
                'little each connective constrains the continuation ("then" constrains least and wins by '
                'default). Any threshold placed on one of them would have been gate shopping.',
    fix='Behavioural checks are diagnostics, not gates. They may be invoked to interpret a NULL, never to '
        'certify a PASS, and all readouts must be reported together so the disagreement is visible.',
)

NEG_LEX = dict(
    name='pat_q7a_neg_internal_probe_beats_bag_means_abstract',
    phase='Q7-A', date='2026-11-01',
    claim_refuted='If a hidden-state linear probe beats a character-bag baseline on held-out content, the '
                  'relation must be encoded abstractly, decoupled from the words.',
    why_refuted='The margin is 0.9136 vs 0.8025 = 11 points (McNemar p=0.005), while layer 1 ALREADY '
                'reads 0.7840 and a single binary "clause B contains a negation character" reads 0.6481 '
                '(contrast variants contain a negation in 94.4% of cases). Most of the decodability is '
                'present before any compositional processing. The defensible claim is "real information '
                'beyond surface lexicon", not "abstract relation code".',
    fix='Every predictive probe result must ship with (a) a bag-of-words baseline on the same folds, '
        '(b) a length baseline, (c) the dominant single-feature baseline, and (d) the layer-1 value.',
)


def main() -> int:
    doc = json.loads(LEDGER.read_text(encoding='utf-8'))
    have_m = {(m.get('phase'), m.get('name')) for m in doc['measurements']}
    have_n = {n.get('name') for n in doc.get('negatives', [])}
    bak = LEDGER.with_suffix('.json.bak_q7a')
    if not bak.exists():
        shutil.copy2(LEDGER, bak)
        print('备份 ->', bak.name)

    added = 0
    entry = {'phase': 'Q7-A', 'name': 'pat_q7a', 'line': 'G', 'date': '2026-11-01',
             'verdict': 'discourse_relation_is_predictively_decodable_content_and_connective_invariant_'
                        'but_three_pole_two_dimensional_not_one_axis',
             'res_sha8': '378498be', 'note': NOTE}
    if (entry['phase'], entry['name']) not in have_m:
        entry['prev_post'] = doc['ledger_sha256_8']
        doc['measurements'].append(entry)
        doc['ledger_sha256_8'] = hashlib.sha256(
            json.dumps(doc, ensure_ascii=False, indent=1).encode('utf-8')).hexdigest()[:8]
        added += 1
        print('+ measurement pat_q7a -> head %s (n=%d)'
              % (doc['ledger_sha256_8'], len(doc['measurements'])))

    for n in (NEG_RANK3, NEG_G0, NEG_LEX):
        if n['name'] in have_n:
            continue
        doc.setdefault('negatives', []).append(n)
        doc['ledger_sha256_8'] = hashlib.sha256(
            json.dumps(doc, ensure_ascii=False, indent=1).encode('utf-8')).hexdigest()[:8]
        added += 1
        print('+ negative   %-52s -> head %s (neg=%d)'
              % (n['name'], doc['ledger_sha256_8'], len(doc['negatives'])))

    if added:
        LEDGER.write_text(json.dumps(doc, ensure_ascii=False, indent=1),
                          encoding='utf-8', newline='')
    print('n=%d head=%s neg=%d added=%d'
          % (len(doc['measurements']), doc['ledger_sha256_8'],
             len(doc.get('negatives', [])), added))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
