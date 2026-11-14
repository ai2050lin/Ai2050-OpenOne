"""Source-binding chain summary, revision 6 (Claude Phase 435): cards and edges after Phases 426, 433 and 434.

No model is run.  Revisions 1-5 are imported unchanged and amended:

  * the read-head set SB.H.read16 is not specific to the original task: frozen and re-applied without selection, it carries
    the direct reading of the asked names in an English task of the same structure (colours), but not in a Chinese one
    (Phase 426);
  * in the Chinese task the names reach the answer through one template position, the newline after <|im_end|> (edge N8):
    the block 23-25 members of SB.H.read16 write it (N8a), 27.11 and 30.35 / 30.38 / 30.39 with blocks 34-39 read it
    (N8b) (Phases 433, 434);
  * cards and edges that now cover more than the original task carry their own scope.

Checks as in revisions 1-5: cited result files exist with provenance and a matching report hash, verdicts are read from
the result files, every typed number lies within 0.00055 of a number in the report of its phase.

    python tests/glm5/source_binding_chain_v6.py
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
import source_binding_chain_v5 as V5  # noqa: E402

ROOT, RES, PV, n = V1.ROOT, V1.RES, V1.PV, V1.n
RUN = 'SOURCE-BINDING-CHAIN-20261114-v6'
SOURCES = {**V5.SOURCES,
           426: ('2026-11-13/015601_SOURCE-BINDING-transfer-holdout.json', 'source_binding_transfer_v1'),
           433: ('2026-11-13/223926_SOURCE-BINDING-zhroute-holdout.json', 'source_binding_zhroute_v1'),
           434: ('2026-11-14/000436_SOURCE-BINDING-eotheads-holdout.json', 'source_binding_eotheads_v1')}
SCOPE3 = ('Qwen3-14B bf16; the source-binding task and two new tasks of the same structure (two sources x two items, four records; '
          'English colours, Chinese animals); new word pools; one model')
RESTRICT = ('broad exchange of the question names at block 23 or 24, restricted by position: positions outside the free set are reset '
            'to the base run after every block (Phase 433); head restrictions on the attention rows of the relay token and the answer '
            'position (Phase 434). Chinese task')

AMEND = {
    'SB.H.read16': {
        'role': 'read the question name tokens. In English prompts (the source-binding task and the colours task) they read them at '
                'the answer position (23.36-38 first, block 25 main, 27.0 / 27.11 / 27.13 last). In the Chinese task their block 23-25 '
                'members read them at the newline after <|im_end|> instead, from where 27.11 and 30.35 / 30.38 / 30.39 pass them on '
                '(edge N8)',
        'causal': [n(426, 0.914, 'colours task, frozen set, no new selection: agreement with the broad exchange (every head 0.9274, '
                                 'random heads 0.7182 = no effect)'),
                   n(426, 0.1958, 'colours: gain over no effect'),
                   n(426, 0.8365, 'Chinese task: agreement (no effect 0.7746): the direct reading carries little there'),
                   n(426, 0.0619, 'Chinese: gain over no effect'),
                   n(434, 0.9252, 'Chinese: the relay token reads the names only through the 13 block 23-25 members (random 13 heads '
                                  '0.8227 = no relay)')],
        'verdicts': [(426, 'C1'), (426, 'C2'), (426, 'C3'), (426, 'Z1'), (426, 'Z2'), (434, 'K2')],
        'counter': 'as in revision 5. In addition: in the Chinese task the direct reading at the answer position is a minor route '
                   '(0.8365 against no effect 0.7746); the same heads act there at another position. Why the position changes with '
                   'the language (or the name form or question structure, which also differ) is not separated',
        'scope': SCOPE3,
    },
}
EDGE_AMEND = {
    'N1': {'status': 'supported as the main route of the transition in English prompts (source-binding task 0.913, colours task 0.914 '
                     'with the frozen heads); in the Chinese task a minor route (0.8365, no effect 0.7746) that is needed together with '
                     'edge N8',
           'numbers': [n(426, 0.914, 'colours task, frozen heads'), n(426, 0.8365, 'Chinese task, frozen heads'),
                       n(433, 0.4826, 'Chinese task: share of the score shift reached by the direct reading alone (English 0.8915)')],
           'scope': SCOPE3},
}

NEW_CARDS = [
    {
        'id': 'SB.S.eot.relay', 'kind': 'residual_state', 'block': '23-39',
        'position': 'the newline after <|im_end|> (second of the five end-of-turn template tokens)',
        'role': 'in the Chinese task, collects the asked names from block 23 on and hands them to the answer position',
        'causal': [n(433, 0.7924, 'positions between the names and the answer reset to the base run: agreement with the broad exchange '
                                  '(direct reading only; no effect 0.7437)'),
                   n(433, 0.9541, 'the five end-of-turn tokens free'), n(433, 0.9703, 'the whole template tail free'),
                   n(433, 0.8115, 'everything free except the five end-of-turn tokens'),
                   n(434, 0.9589, 'only the newline after <|im_end|> free (all five 0.9602; new word pool)'),
                   n(434, 0.8228, 'nothing free (new word pool)')],
        'verdicts': [(433, 'Z1'), (433, 'Z2'), (433, 'Z3'), (433, 'C1'), (433, 'C2'), (434, 'K1')],
        'counter': 'in the English colours task the direct reading already suffices (0.9336) and the relay adds little (0.9428), although '
                   'the end-of-turn tokens carry nearly all of the score shift there too (0.9974): both languages may collect the names '
                   'at this position. What the state holds is not decoded. Resetting positions is an intervention',
        'grade': 'in-scope intervention, held-out prediction on new word pools, pre-registered',
        'configuration': RESTRICT, 'scope': SCOPE3,
    },
    {
        'id': 'SB.H.eot.read', 'kind': 'attention_head_set', 'block': '27-39', 'heads': [[27, 11], [30, 35], [30, 38], [30, 39]],
        'position': 'final position (outputs the answer)',
        'role': 'read the relay token for the answer position: 27.11 and 30.35 / 30.38 / 30.39, with a spread reading by every head of '
                'blocks 34-39',
        'causal': [n(434, 0.9615, 'the answer row reads the relay token only through these four heads and blocks 34-39'),
                   n(434, 0.8817, 'only through blocks 34-39'),
                   n(434, 0.0798, 'gain of the four heads over blocks 34-39 alone (random four heads: none)'),
                   n(434, 0.9287, 'write through the 13 read heads and read through these heads together (direct reading 0.8228)'),
                   n(434, 0.1059, 'gain of that sparse relay over the direct reading')],
        'verdicts': [(434, 'K3'), (434, 'K4')],
        'counter': '27.11 is also a member of SB.H.read16 (it reads the names directly at the answer position). The checking heads 27.12 '
                   'and 27.14 oppose the read in the leave-one-out screen of the pilot (not tested on its own). Selected on an opened '
                   'panel by leave-one-out in blocks 27 and 30 only; QK and OV are not separated',
        'grade': 'in-scope intervention, held-out prediction on a new word pool, pre-registered',
        'configuration': RESTRICT, 'scope': SCOPE3,
    },
]
NEW_EDGES = [
    {'id': 'N8a', 'source': 'question name tokens', 'target': 'SB.S.eot.relay', 'via': 'SB.H.read16 (block 23-25 members, 13 heads)',
     'blocks': '23-25', 'type': 'in-scope intervention (position reset, head restriction on the relay row)',
     'numbers': [n(434, 0.9252, 'the relay token reads the names only through the 13 heads'), n(434, 0.9589, 'through every head'),
                 n(434, 0.8228, 'not at all (direct reading only)')],
     'status': 'supported in the Chinese task; the same heads that read the names at the answer position in English prompts',
     'configuration': RESTRICT, 'scope': SCOPE3},
    {'id': 'N8b', 'source': 'SB.S.eot.relay', 'target': 'final position (answer)', 'via': 'SB.H.eot.read',
     'blocks': '27-39', 'type': 'in-scope intervention (position reset, head restriction on the answer row)',
     'numbers': [n(434, 0.9615, 'read through 27.11, 30.35 / 30.38 / 30.39 and blocks 34-39'), n(434, 0.8817, 'through blocks 34-39 only')],
     'status': 'supported in the Chinese task; needed together with the direct reading (edge N1)', 'configuration': RESTRICT, 'scope': SCOPE3},
]

FIDELITY = {'phases': V5.FIDELITY['phases'] + [426, 433, 434],
            'numbers': V5.FIDELITY['numbers'] + [
                n(426, 0.914, 'colours task (English, same structure), transition cells, frozen read heads'),
                n(426, 0.8365, 'Chinese task, frozen read heads (no effect 0.7746)'),
                n(434, 0.9287, 'Chinese task, new word pool: sparse relay N8a + N8b with the direct reading')],
            'rule_revision_6': 'name tokens exchanged at block L in 23-25. English prompts: as in revision 5 (the heads of SB.H.read16 in '
                               'blocks >= L read the new names at the answer position). Chinese prompt: in addition the relay token '
                               '(newline after <|im_end|>) reads them through the block 23-25 members of SB.H.read16 and the answer row '
                               'reads the relay token through 27.11, 30.35 / 30.38 / 30.39 and blocks 34-39. Measured for blocks 23 and '
                               '24 in the Chinese task (0.9287)'}

GAPS = [g for g in V5.GAPS] + [
    'why the Chinese prompt routes the names through the relay token: language, name form (two-character names, two-token team names) '
    'and question structure differ together between the tasks',
    'what the relay token holds and what the read heads take from it; whether the same heads write the relay token in English prompts '
    '(the end-of-turn tokens carry the score shift there too)',
    'the Chinese task for exchanges at block 25; tasks of another structure; other models',
]


def build():
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
    V5.amend(cards, edges, V3.AMEND, V3.EDGE_AMEND, 'revision_2', 416)
    cards += copy.deepcopy(V3.NEW_CARDS)
    edges += copy.deepcopy(V3.NEW_EDGES)
    V5.amend(cards, edges, V4.AMEND, V4.EDGE_AMEND, 'revision_3', 422)
    cards += copy.deepcopy(V4.NEW_CARDS)
    edges += copy.deepcopy(V4.NEW_EDGES)
    V5.amend(cards, edges, V5.AMEND, V5.EDGE_AMEND, 'revision_4', 425)
    cards += copy.deepcopy(V5.NEW_CARDS)
    for x in cards + edges:                           # revision 6 records a scope per item; earlier items keep the one-task scope
        x.setdefault('scope', V1.SCOPE)
    V5.amend(cards, edges, AMEND, EDGE_AMEND, 'revision_5', 435)
    cards += copy.deepcopy(NEW_CARDS)
    edges += copy.deepcopy(NEW_EDGES)
    assert len({c['id'] for c in cards}) == len(cards) and len({e['id'] for e in edges}) == len(edges)
    return cards, edges


def main():
    cards, edges = build()
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
        c.update({'model': 'Qwen3-14B bf16', 'scope': c.get('scope', V1.SCOPE), 'natural_corpus_statistics': 'not measured',
                  'unembedding_readout': 'not measured' if c['id'] != 'SB.S.final.centre'
                  else 'output-weight subspace deletion keeps 0.969: the answer is not mainly in that subspace', 'formal_promotion': 0})
    for e in edges:
        e.update({'scope': e.get('scope', V1.SCOPE), 'formal_promotion': 0})
    assert all(v['verdict'] == 'supported' for c in cards for v in c['verdicts'] if v['phase'] >= 426)

    payload = {'run': RUN, 'kind': 'summary of Claude Phases 396-434, revision 6 of the Phase 408 summary; no model run',
               'supersedes': 'results/2026-11-13/003412_SOURCE-BINDING-chain-summary-v5.json (kept; revision 5)',
               'task': 'SOURCE-BINDING-20261110-v1 (and the two same-structure tasks of Phase 426)', 'scope': SCOPE3,
               'natural_short_chain': ['E1', 'E2', 'N1', 'N6', 'N2', 'E8'], 'chinese_relay': ['N8a', 'N8b'], 'check_route': ['N7'],
               'backup_routes': ['E3', 'E4', 'E5', 'E6', 'E7', 'N3', 'N4', 'N5'],
               'fidelity_verdicts': {str(p): verd[p] for p in (426, 433, 434)},
               'cards': cards, 'edges': edges, 'fidelity': FIDELITY, 'gaps': GAPS, 'sources': sources,
               'checks': {'numbers_checked_against_reports': checked, 'reports_hash_matched': len(sources),
                          'verdicts_read_from_result_files': sum(len(c['verdicts']) for c in cards)},
               'formal_promotion': 0}
    out = PV.write_result('SOURCE-BINDING-chain-summary-v6', payload, model_path=V1.MODEL, seed=0, tool=RUN)
    print(json.dumps({'result': str(out), **payload['checks'], 'cards': len(cards), 'edges': len(edges)}, ensure_ascii=False))
    for c in cards:
        if c.get('revised_in') == 435 or c['id'] in [x['id'] for x in NEW_CARDS]:
            print(c['id'], [(v['phase'], v['key'], v['verdict']) for v in c['verdicts'] if v['phase'] >= 426])
    v5 = json.loads((ROOT / 'results/2026-11-13/003412_SOURCE-BINDING-chain-summary-v5.json').read_text(encoding='utf-8'))['payload']
    by = {c['id']: c for c in v5['cards']}
    same = lambda a, b: all(a.get(k) == b.get(k) for k in ('role', 'causal', 'verdicts', 'counter'))
    kept = [c['id'] for c in cards if c['id'] in by and c['id'] not in AMEND and same(c, by[c['id']])]
    print(json.dumps({'cards_identical_to_revision_5': len(kept), 'cards_in_revision_5': len(by)}))


if __name__ == '__main__':
    main()
