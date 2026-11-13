"""Source-binding chain summary, revision 4 (Claude Phase 422): cards and edges after Phases 416-421.

No model is run.  Revisions 1-3 are imported unchanged and amended:

  * the write of the block-25 heads (25.18, 25.24, 25.15) is an intermediate quantity: it has no direct path to the output
    and no single later component turns it into the answer; blocks 25-28 and the attention of blocks 34-37 act in relay,
    and random component sets of the same size do as well as the chosen one (Phase 420);
  * four heads of block 27 (27.0, 27.3, 27.12, 27.14) check the pointer against the content of the name tokens: they pull
    back when the names still hold the base content and push when the names were exchanged (Phase 421);
  * per-edit prediction inside the block-24 transition: the pointer score after a narrow exchange in the block-25 heads
    (Phase 417), its edit-free linear read-out (Phase 419), and the same score with donor names in block 27 (Phase 421);
  * the fidelity record gets Phase 416 (revision 3) and the per-edit predictors.

Checks as in revisions 1-3: cited result files exist with provenance and a matching report hash, verdicts are read from
the result files, every typed number lies within 0.00055 of a number in the report of its phase.

    python tests/glm5/source_binding_chain_v4.py
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

ROOT, RES, PV, n = V1.ROOT, V1.RES, V1.PV, V1.n
RUN = 'SOURCE-BINDING-CHAIN-20261112-v4'
SOURCES = {**V3.SOURCES,
           416: ('2026-11-12/065415_SOURCE-BINDING-fidelity-v3-holdout.json', 'source_binding_fidelity_v3'),
           417: ('2026-11-12/074913_SOURCE-BINDING-peredit-holdout.json', 'source_binding_peredit_v1'),
           419: ('2026-11-12/082456_SOURCE-BINDING-readout-holdout.json', 'source_binding_readout_v1'),
           420: ('2026-11-12/101310_SOURCE-BINDING-downstream-holdout.json', 'source_binding_downstream_v1'),
           421: ('2026-11-12/221351_SOURCE-BINDING-attn27-holdout.json', 'source_binding_attn27_v1')}
NARROW = ('narrow edit on top of the natural run: in block 25, heads 25.18, 25.24, 25.15, attention row of the answer position, keys '
          'and values of name tokens and holders taken from a donor (Phases 415 / 417); components after it responsive or frozen')
CONFLICT = V3.CONFLICT

AMEND = {
    'SB.H.25.18': {
        'role': 'heads 25.18, 25.24, 25.15 at the answer position: carry a position pointer from the question name tokens. What they '
                'write is an intermediate quantity: it reaches the output only through later components (edge N6) and its size '
                'predicts, edit by edit, whether a broad exchange of the name tokens at block 24 moves the answer',
        'causal': [n(417, 0.9046, 'AUC of score(pointer record) - score(base) after the narrow exchange, for the outcome of the broad '
                                  'block-24 exchange (unedited score 0.6369)'),
                   n(417, 0.8232, 'accuracy of the frozen-threshold rule (class rule 0.6468)'),
                   n(419, 0.7796, 'AUC of a linear read-out of what the three heads write in the two unedited runs (no edit; unedited '
                                  'score 0.6427, narrow edit 0.8881)'),
                   n(420, 0.0013, 'share of the score shift that reaches the output with every later component frozen (direct path)')],
        'verdicts': [(417, 'P1'), (417, 'P2'), (417, 'P3'), (419, 'U1'), (419, 'U2'), (419, 'U3'), (420, 'N1')],
        'counter': 'as in revision 3. In addition: the predictor is itself an intervention; without any edit a fitted linear read-out '
                   'recovers about half of the gap (0.7796 against 0.8881). The write alone moves nothing at the output (0.0013)'},
}
EDGE_AMEND = {
    'N1': {'numbers': [n(417, 0.9046, 'per-edit prediction of the block-24 exchange from the narrow exchange in the block-25 heads (AUC)'),
                       n(421, 0.9458, 'the same with donor names in block 27 (AUC)')],
           'status': 'supported as a sufficient route; in the natural run one of several sources that are merged by strength '
                     '(SB.S.final.merge); the merge is predicted edit by edit from the write of the block-25 heads (Phases 417, 421)'},
}

NEW_CARDS = [
    {
        'id': 'SB.C.relay', 'kind': 'component_route', 'block': '25-37',
        'position': 'final position (outputs the answer)',
        'role': 'turns the write of the block-25 heads into the score of the pointed record in two stages: blocks 25-28 (MLP 25-27, '
                'attention 26 and 28 weigh most) and the attention of blocks 34-37; neither stage acts alone',
        'causal': [n(420, 0.0013, 'all 29 later components frozen to the base run: share of the full score shift (direct path)'),
                   n(420, 0.0653, 'largest share of any single component responding alone'),
                   n(420, 0.1639, 'only the attention layers respond'), n(420, 0.0273, 'only the MLPs respond'),
                   n(420, 0.0326, 'only blocks 25-28 respond'), n(420, 0.1409, 'only attention 34-37 responds'),
                   n(420, 0.6221, 'both respond (11 of 29 components)'),
                   n(420, 0.4486, 'synergy: both minus the sum of each alone'),
                   n(420, 0.5607, 'share lost when blocks 25-28 are frozen'), n(420, 0.501, 'share lost when attention 34-37 is frozen'),
                   n(420, 0.8601, 'blocks 25-33 and attention 34-37 respond (MLP 34-37 and blocks 38-39 frozen)'),
                   n(420, 0.8542, 'AUC of the score with that set for the outcome of the broad exchange (all responsive 0.8932)')],
        'verdicts': [(420, 'N1'), (420, 'N2'), (420, 'N3'), (420, 'N4'), (420, 'N5'), (420, 'N7')],
        'counter': 'not a privileged set: one of five random sets of the same size and type mix reaches 0.6662 (the chosen one 0.6221). '
                   'Shares of interacting components do not add up. Heads and MLP units inside the components are not located. '
                   'Freezing to the base run is an intervention',
        'grade': 'in-scope intervention (path patching by component), held-out entities, pre-registered',
        'configuration': NARROW,
    },
    {
        'id': 'SB.H.27.K', 'kind': 'attention_head_set', 'block': 27, 'heads': [0, 3, 12, 14],
        'position': 'final position (outputs the answer)',
        'role': 'check the pointer of the answer position against the content of the name tokens: with the names unchanged their '
                'attention pattern shifts and pulls the score back to the base answer; with the names exchanged the layer pushes '
                'towards the new pointer',
        'causal': [n(421, 1.8239, 'narrow edit, whole attention output of block 27 frozen: score shift in units of the narrow edit'),
                   n(421, 1.7307, 'attention weights of the base run, values as computed'),
                   n(421, 1.8975, 'outputs of the four heads frozen (largest of 5 random sets of 4 heads: gain 0.1349)'),
                   n(421, 0.8904, 'only the four heads respond'),
                   n(421, 1.6663, 'block 27 reads donor keys / values at the name tokens'),
                   n(421, 0.9934, 'records and headers hidden from the answer row of block 27'),
                   n(421, 0.1081, 'broad exchange at block 24: drop of the share following the new pointer when the names are hidden '
                                  'from the answer row of block 27')],
        'verdicts': [(421, 'Q1'), (421, 'Q2'), (421, 'Q3'), (421, 'Q4'), (421, 'Q5'), (420, 'N6')],
        'counter': 'the four heads put little attention on the name tokens (sum 0.18 -> 0.23) and move it away from the records; '
                   'hiding the names alone gains less (1.2667) than donor names. That they read the names directly is not shown; '
                   'the path from the name content to their attention pattern is not located. Attention 31 and 33 also oppose '
                   'slightly (Phase 420)',
        'grade': 'in-scope intervention, held-out entities, pre-registered',
        'configuration': NARROW + '; the broad exchange for the last number',
    },
]
NEW_EDGES = [
    {'id': 'N6', 'source': 'SB.H.25.18 (write at the answer position)', 'target': 'final position (score of the pointed record)',
     'via': 'SB.C.relay', 'blocks': '25-37', 'type': 'in-scope intervention (path patching by component)',
     'numbers': [n(420, 0.0013, 'direct path'), n(420, 0.6221, 'blocks 25-28 and attention 34-37 respond'),
                 n(420, 0.8601, 'plus blocks 29-33'), n(420, 0.4486, 'synergy of the two stages')],
     'status': 'supported as a distributed relay; no single component and no privileged small set', 'configuration': NARROW},
    {'id': 'N7', 'source': 'question name tokens (content)', 'target': 'final position (pointer check)', 'via': 'SB.H.27.K',
     'blocks': '27', 'type': 'in-scope intervention (attention weights, key / value exchange, head outputs, attention mask)',
     'numbers': [n(421, 1.7307, 'narrow edit, base-run attention weights in block 27'), n(421, 1.8975, 'four heads frozen'),
                 n(421, 1.6663, 'donor names in block 27'), n(421, 0.1081, 'broad exchange, names hidden in block 27: drop')],
     'status': 'supported: opposes when the names hold the base content, helps when they were exchanged; path from the names to the '
               'heads not located', 'configuration': NARROW + '; ' + CONFLICT},
]

FIDELITY = {'phases': [409, 412, 416, 417, 419, 421],
            'numbers': V3.FIDELITY['numbers'] + [
                n(416, 0.9809, 'third new panel, name tokens at blocks 8-22 and 26: agreement of the graph'),
                n(416, 0.8735, 'name tokens at block 23: revision-3 graph'),
                n(416, 0.7663, 'at block 24: revision-3 class rule (no effect 0.7227, revision 2 0.609)'),
                n(416, 0.8334, 'at block 25'), n(416, 0.9222, 'all 98,304 edits'),
                n(417, 0.8232, 'both names at block 24, conflict edits: per-edit rule from the narrow exchange (class rule 0.6468)'),
                n(419, 0.704, 'same, edit-free linear read-out'),
                n(421, 0.9458, 'AUC of the narrow exchange with donor names in block 27 (narrow exchange alone 0.9069)')],
            'rule_revision_4': 'both names exchanged at block 24: the answer is the one given after the narrow exchange through edges N1 '
                               'and N7 (heads 25.18, 25.24, 25.15 read donor name tokens and holders; block 27 reads donor keys / '
                               'values at the name tokens; answer row only). No fitted number. Tested in Phase 422 '
                               '(source_binding_fidelity4.py); other edits as in revision 3'}

GAPS = [g for g in V3.GAPS if not g.startswith(('how the pointer written by the block-25 heads', 'fidelity: name-token state'))] + [
    'inside the relay (blocks 25-28, attention 34-37): which heads and MLP units, and what form the write takes after blocks 25-28; '
    'head 28.21 (revision 2) lies in the first stage but its share of the relay is not measured',
    'how the content of the name tokens reaches the attention pattern of heads 27.0, 27.3, 27.12, 27.14, and what these heads write',
    'the per-edit predictor is an intervention; edit-free, a fitted linear read-out reaches about half of it (Phase 419)',
    'fidelity: name-token state exchanges only, four candidate words; per-edit rule only for both names at block 24 (single-name '
    'sites and blocks 23, 25 use the revision-3 class rule and follow-rate table)',
]


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
    for c in cards:                                   # revision 3
        a = V3.AMEND.get(c['id'])
        if a:
            c['role_revision_2'], c['counter_revision_2'] = c['role'], c['counter']
            c.update({'role': a['role'], 'counter': a['counter'], 'heads': a['heads'], 'revised_in': 416})
            c['causal'] = c['causal'] + a['causal']
            c['verdicts'] = list(c['verdicts']) + a['verdicts']
    for e in edges:
        a = V3.EDGE_AMEND.get(e['id'])
        if a:
            e['revision_2'] = {k: e[k] for k in a if k != 'numbers'}
            e['numbers'] = e['numbers'] + a.get('numbers', [])
            e.update({k: v for k, v in a.items() if k != 'numbers'})
            e['revised_in'] = 416
    cards += copy.deepcopy(V3.NEW_CARDS)
    edges += copy.deepcopy(V3.NEW_EDGES)
    for c in cards:                                   # revision 4
        a = AMEND.get(c['id'])
        if a:
            c['role_revision_3'], c['counter_revision_3'] = c['role'], c['counter']
            c.update({'role': a['role'], 'counter': a['counter'], 'revised_in': 422})
            c['causal'] = c['causal'] + a['causal']
            c['verdicts'] = list(c['verdicts']) + a['verdicts']
    for e in edges:
        a = EDGE_AMEND.get(e['id'])
        if a:
            e['revision_3'] = {k: e[k] for k in a if k != 'numbers'}
            e['numbers'] = e['numbers'] + a.get('numbers', [])
            e.update({k: v for k, v in a.items() if k != 'numbers'})
            e['revised_in'] = 422
    cards += copy.deepcopy(NEW_CARDS)
    edges += copy.deepcopy(NEW_EDGES)
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

    payload = {'run': RUN, 'kind': 'summary of Claude Phases 396-421, revision 4 of the Phase 408 summary; no model run',
               'supersedes': 'results/2026-11-12/063401_SOURCE-BINDING-chain-summary-v3.json (kept; revision 3)',
               'task': 'SOURCE-BINDING-20261110-v1', 'scope': V1.SCOPE,
               'natural_short_chain': ['E1', 'E2', 'N1', 'N6', 'N2', 'E8'], 'check_route': ['N7'],
               'backup_routes': ['E3', 'E4', 'E5', 'E6', 'E7', 'N3', 'N4', 'N5'],
               'fidelity_verdicts_416': verd[416],
               'cards': cards, 'edges': edges, 'fidelity': FIDELITY, 'gaps': GAPS, 'sources': sources,
               'checks': {'numbers_checked_against_reports': checked, 'reports_hash_matched': len(sources),
                          'verdicts_read_from_result_files': sum(len(c['verdicts']) for c in cards)},
               'formal_promotion': 0}
    out = PV.write_result('SOURCE-BINDING-chain-summary-v4', payload, model_path=V1.MODEL, seed=0, tool=RUN)
    print(json.dumps({'result': str(out), **payload['checks'], 'cards': len(cards), 'edges': len(edges)}, ensure_ascii=False))
    for c in cards:
        if c.get('revised_in') == 422 or c['id'] in ('SB.C.relay', 'SB.H.27.K'):
            print(c['id'], [(v['phase'], v['key'], v['verdict']) for v in c['verdicts'] if v['phase'] >= 417])


if __name__ == '__main__':
    main()
