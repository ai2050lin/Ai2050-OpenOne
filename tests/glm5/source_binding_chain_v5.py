"""Source-binding chain summary, revision 5 (Claude Phase 425): cards and edges after Phases 422-424.

No model is run.  Revisions 1-4 are imported unchanged and amended:

  * the reading of the name tokens by the answer position (edge N1) is now carried by 16 heads of blocks 23-27
    (23.36-38, 24.17/19/33/36, 25.15/16/18/24/38, 26.18, 27.0/11/13); random heads do nothing (Phase 424);
  * the reading ends by block 27 and predicts the broad exchange in all nine transition cells (archive name, entity name
    or both, exchanged at block 23, 24 or 25): 0.931 with every head, 0.913 with the 16 heads (Phases 423, 424);
  * the fidelity record gets Phase 422 (revision-4 rule at block 24) and the transition cells; the revision-3 class
    rule and follow-rate table are superseded as predictors of the transition.

Checks as in revisions 1-4: cited result files exist with provenance and a matching report hash, verdicts are read from
the result files, every typed number lies within 0.00055 of a number in the report of its phase.

    python tests/glm5/source_binding_chain_v5.py
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import source_binding_chain as V1  # noqa: E402
import source_binding_chain_v2 as V2  # noqa: E402
import source_binding_chain_v3 as V3  # noqa: E402
import source_binding_chain_v4 as V4  # noqa: E402

ROOT, RES, PV, n = V1.ROOT, V1.RES, V1.PV, V1.n
RUN = 'SOURCE-BINDING-CHAIN-20261113-v5'
SOURCES = {**V4.SOURCES,
           422: ('2026-11-12/223218_SOURCE-BINDING-fidelity-v4-holdout.json', 'source_binding_fidelity_v4'),
           423: ('2026-11-12/232210_SOURCE-BINDING-sites-holdout.json', 'source_binding_sites_v1'),
           424: ('2026-11-13/002707_SOURCE-BINDING-readheads-holdout.json', 'source_binding_readheads_v1')}
NARROW = ('narrow edit on top of the natural run: only the attention row of the answer position; named heads read donor keys / values '
          '(unedited donor run) at the exchanged name tokens. Predicted object: the broad exchange of these tokens as residual states '
          'at the input of block 23, 24 or 25')
READ16 = [[23, 36], [23, 37], [23, 38], [24, 17], [24, 19], [24, 33], [24, 36], [25, 15], [25, 16], [25, 18], [25, 24], [25, 38],
          [26, 18], [27, 0], [27, 11], [27, 13]]

AMEND = {
    'SB.H.25.18': {
        'heads': [15, 16, 18, 24, 38],
        'role': 'block-25 members of the read set SB.H.read16 (25.18 and 25.15 weigh most; 25.16 and 25.38 added in Phase 424): carry '
                'a position pointer from the question name tokens to the answer position. The write is an intermediate quantity '
                '(edge N6) and its size predicts the broad exchange edit by edit',
        'causal': [n(424, 0.9333, 'exchange at block 25, nine-cell test: 25.15, 25.18, 25.24 and block 27 read the donor names: '
                                  'agreement with the broad edit (every head 0.9427, no effect 0.8349)'),
                   n(424, 0.5945, 'exchange at block 23: the same heads alone (the full read set 0.9123)')],
        'verdicts': [(424, 'R3'), (423, 'X4')],
        'counter': 'as in revision 4. In addition: sufficient alone only for exchanges at block 25; for earlier exchanges the heads '
                   'of blocks 23-24 are needed (0.5945 against 0.9123). Keys and values are shared inside groups of 5 heads '
                   '(25.15-25.19), so neighbours may differ only by their queries'},
}
EDGE_AMEND = {
    'N1': {'via': 'SB.H.read16 (16 heads of blocks 23-27; block-25 members on card SB.H.25.18)',
           'blocks': '23-27 (the reading ends by block 27: donor keys / values up to the last block change nothing)',
           'numbers': [n(423, 0.9312, 'nine transition cells: the answer when the answer position reads the donor names in blocks L..27 '
                                      'through every head agrees with the broad exchange (revision-3 rule 0.828, no effect 0.6694)'),
                       n(423, 0.0003, 'same, donor keys / values up to block 39: difference'),
                       n(424, 0.9128, 'through the 16 heads only (random heads 0.6705 = no effect)')],
           'status': 'supported as the route that carries the broad exchange in the transition (0.91-0.93 of the edits); located by '
                     'head; routes through other positions make up the rest'},
    'N7': {'numbers': [n(422, 0.1638, 'both names at block 24: agreement gained when block 27 also reads the donor names (0.6667 -> 0.8306)')],
           'status': 'supported: opposes when the names hold the base content, helps when they were exchanged; the heads that read the '
                     'names in block 27 (27.0, 27.11, 27.13) overlap the checking heads (27.0, 27.3, 27.12, 27.14) in 27.0 only'},
}

NEW_CARDS = [
    {
        'id': 'SB.H.read16', 'kind': 'attention_head_set', 'block': '23-27', 'heads': READ16,
        'position': 'final position (outputs the answer)',
        'role': 'read the question name tokens at the answer position: 23.36-38 start in block 23, block 25 is the main stage '
                '(25.18, 25.15), 27.0, 27.11, 27.13 read last. What these heads read decides the answer after an exchange of the '
                'name tokens in blocks 23-25',
        'causal': [n(424, 0.9128, 'nine transition cells (36,864 edits): only these heads read the donor names: agreement with the broad '
                                  'exchange'),
                   n(424, 0.9336, 'every head of blocks L..27 reads the donor names'),
                   n(424, 0.6705, 'as many random heads per block read them (5 sets; equals no effect 0.6705)'),
                   n(424, 0.9099, 'without 24.33, 24.36, 25.38, 26.18 (12 heads)'),
                   n(424, 0.8875, 'without block 24 and 25.16 as well (9 heads)'),
                   n(424, 0.3178, 'exchange at block 23: gain over 25.15, 25.18, 25.24 and block 27 alone'),
                   n(424, 0.0906, 'gain over the revision-3 rule')],
        'verdicts': [(424, 'R1'), (424, 'R2'), (424, 'R3'), (424, 'R4'), (423, 'X1'), (423, 'X3')],
        'counter': 'the 12-head subset does as well overall (0.9099; the four extra heads help only for exchanges at block 24). '
                   'Screened by leaving one head out of the all-heads rule: redundant heads are missed, 0.0208 remain to the '
                   'all-heads rule. Keys and values are shared inside groups of 5 heads; QK and OV are exchanged together. One cell '
                   '(archive name at block 23) is predicted better by the revision-3 rule in Phase 423 (0.905 against 0.9297)',
        'grade': 'in-scope intervention, held-out prediction on new entities, pre-registered',
        'configuration': NARROW,
    },
]

FIDELITY = {'phases': V4.FIDELITY['phases'] + [422, 423, 424],
            'numbers': V4.FIDELITY['numbers'] + [
                n(422, 0.8306, 'both names at block 24: revision-4 rule (narrow edit through N1 and N7); revision 3 0.6082, no effect 0.5527'),
                n(423, 0.9312, 'nine transition cells, every head of blocks L..27 reads the donor names; revision 3 0.828'),
                n(423, 0.876, 'lowest of the nine cells (both names at block 24)'),
                n(424, 0.9128, 'nine transition cells, the 16 heads of SB.H.read16; random heads 0.6705'),
                n(424, 0.9123, 'exchange at block 23'), n(424, 0.9051, 'at block 24'), n(424, 0.921, 'at block 25')],
            'rule_revision_5': 'name tokens (archive, entity or both) exchanged at block L in 23-25: the answer is the one given when '
                               'the heads of SB.H.read16 in blocks >= L read the donor keys / values at these tokens (answer row '
                               'only). No fitted number. Other blocks as in revision 3. Supersedes the revision-3 class rule and '
                               'follow-rate table and the revision-4 rule for block 24',
            'composed_estimate': 'over the edit family of Phase 416 (name tokens at blocks 8, 14, 18, 22, 23, 24, 25, 26; 98,304 edits) '
                                 'revision 5 would agree in (61,440 x 0.9809 + 36,864 x 0.9128) / 98,304 = 0.955 (revision 3: 0.9222 '
                                 'measured). The two parts were measured on different panels; this is arithmetic, not a measurement'}

GAPS = [g for g in V4.GAPS if not g.startswith('fidelity: name-token state exchanges only')] + [
    'what each head of SB.H.read16 reads and writes (QK / OV not separated; archive-name and entity-name heads not tested apart)',
    'the routes through other positions that make up the 0.07 between the direct reading and the broad exchange; the archive name '
    'exchanged at block 23 goes partly through the holders (Phase 423)',
    'fidelity: name-token state exchanges only, four candidate words; the prediction of the transition is a narrow intervention '
    'followed by a forward pass, not a computation on the graph alone',
]


def amend(cards, edges, A, E, tag, phase):
    for c in cards:
        a = A.get(c['id'])
        if a:
            c[f'role_{tag}'], c[f'counter_{tag}'] = c['role'], c['counter']
            c.update({k: v for k, v in a.items() if k not in ('causal', 'verdicts')})
            c['revised_in'] = phase
            c['causal'] = c['causal'] + a['causal']
            c['verdicts'] = list(c['verdicts']) + a['verdicts']
    for e in edges:
        a = E.get(e['id'])
        if a:
            e[tag] = {k: e[k] for k in a if k != 'numbers'}
            e['numbers'] = e['numbers'] + a.get('numbers', [])
            e.update({k: v for k, v in a.items() if k != 'numbers'})
            e['revised_in'] = phase


def main():
    cards, edges = copy.deepcopy(V1.CARDS), copy.deepcopy(V1.EDGES)
    for c in cards:                                   # revision 2
        a = V2.AMEND.get(c['id'], {})
        if 'role' in a:
            c['role_revision_1'], c['role'] = c['role'], a['role']
        c['causal'] = c['causal'] + a.get('causal', [])
        c['verdicts'] = list(c['verdicts']) + a.get('verdicts', [])
        c['configuration'] = V2.CONFIG[c['id']]
        c['revised_in'] = 412 if a or V2.CONFIG[c['id']] == V2.ISOLATED else None
    for e in edges:
        e['configuration'] = V2.CONFIG[e['id']]
    cards += copy.deepcopy(V2.NEW_CARDS)
    edges += copy.deepcopy(V2.NEW_EDGES)
    amend(cards, edges, V3.AMEND, V3.EDGE_AMEND, 'revision_2', 416)
    cards += copy.deepcopy(V3.NEW_CARDS)
    edges += copy.deepcopy(V3.NEW_EDGES)
    amend(cards, edges, V4.AMEND, V4.EDGE_AMEND, 'revision_3', 422)
    cards += copy.deepcopy(V4.NEW_CARDS)
    edges += copy.deepcopy(V4.NEW_EDGES)
    amend(cards, edges, AMEND, EDGE_AMEND, 'revision_4', 425)
    cards += copy.deepcopy(NEW_CARDS)
    assert len({c['id'] for c in cards}) == len(cards) and len({e['id'] for e in edges}) == len(edges)

    sources, pool, verd = {}, {}, {}
    for phase, (rel, folder) in SOURCES.items():
        doc = json.loads((ROOT / 'results' / rel).read_text(encoding='utf-8'))
        assert not PV.check_provenance(doc), rel
        payload = doc['payload']
        assert payload.get('gate_passed') is True, rel
        report = RES / folder / 'report.json'
        sha = hashlib.sha256(report.read_bytes()).hexdigest()
        assert sha == payload['report_sha256'], (rel, sha)
        got = set()
        V1.floats(json.loads(report.read_text(encoding='utf-8')), got)
        pool[phase], verd[phase] = got, payload['verdicts']
        sources[str(phase)] = {'result_file': 'results/' + rel, 'result_sha8': doc['sha8'], 'report': f'tests/glm5/result/{folder}/report.json',
                               'report_sha256': sha, 'run': doc['provenance']['tool'], 'git_sha': doc['provenance']['git_sha'],
                               'seed': doc['provenance']['seed'], 'verdicts': payload['verdicts']}

    missing, checked = [], 0
    for item in cards + edges + [dict(FIDELITY, id='fidelity')]:
        for num in item.get('causal', []) + item.get('numbers', []):
            checked += 1
            if min(abs(abs(num['value']) - abs(f)) for f in pool[num['phase']]) > 0.00055:
                missing.append((item['id'], num['phase'], num['value']))
    assert not missing, missing

    for c in cards:
        c['verdicts'] = [{'phase': p, 'key': k, 'verdict': V1.verdict(verd[p], k)} for p, k in c['verdicts']]
        c.update({'model': 'Qwen3-14B bf16', 'scope': V1.SCOPE, 'natural_corpus_statistics': 'not measured',
                  'unembedding_readout': 'not measured' if c['id'] != 'SB.S.final.centre'
                  else 'output-weight subspace deletion keeps 0.969: the answer is not mainly in that subspace', 'formal_promotion': 0})
    for e in edges:
        e.update({'scope': V1.SCOPE, 'formal_promotion': 0})

    payload = {'run': RUN, 'kind': 'summary of Claude Phases 396-424, revision 5 of the Phase 408 summary; no model run',
               'supersedes': 'results/2026-11-12/222216_SOURCE-BINDING-chain-summary-v4.json (kept; revision 4)',
               'task': 'SOURCE-BINDING-20261110-v1', 'scope': V1.SCOPE,
               'natural_short_chain': ['E1', 'E2', 'N1', 'N6', 'N2', 'E8'], 'check_route': ['N7'],
               'backup_routes': ['E3', 'E4', 'E5', 'E6', 'E7', 'N3', 'N4', 'N5'],
               'fidelity_verdicts': {str(p): verd[p] for p in (422, 423, 424)},
               'cards': cards, 'edges': edges, 'fidelity': FIDELITY, 'gaps': GAPS, 'sources': sources,
               'checks': {'numbers_checked_against_reports': checked, 'reports_hash_matched': len(sources),
                          'verdicts_read_from_result_files': sum(len(c['verdicts']) for c in cards)},
               'formal_promotion': 0}
    out = PV.write_result('SOURCE-BINDING-chain-summary-v5', payload, model_path=V1.MODEL, seed=0, tool=RUN)
    print(json.dumps({'result': str(out), **payload['checks'], 'cards': len(cards), 'edges': len(edges)}, ensure_ascii=False))
    for c in cards:
        if c.get('revised_in') == 425 or c['id'] == 'SB.H.read16':
            print(c['id'], c.get('heads'), [(v['phase'], v['key'], v['verdict']) for v in c['verdicts'] if v['phase'] >= 422])
    v4 = json.loads((ROOT / 'results/2026-11-12/222216_SOURCE-BINDING-chain-summary-v4.json').read_text(encoding='utf-8'))['payload']
    same = lambda a, b, keys: all(a.get(k) == b.get(k) for k in keys)
    by = {c['id']: c for c in v4['cards']}
    kept = [c['id'] for c in cards if c['id'] in by and c['id'] not in AMEND and same(c, by[c['id']], ('role', 'causal', 'verdicts', 'counter'))]
    print(json.dumps({'cards_identical_to_revision_4': len(kept), 'cards_in_revision_4': len(by)}))


if __name__ == '__main__':
    main()
