"""Source-binding chain summary, revision 3 (Claude Phase 416): cards and edges after Phases 412-415.

No model is run.  Revisions 1 and 2 are imported unchanged and amended:

  * the reading of the name tokens by the answer position is no longer a window with a winner: Phase 413 shows a graded
    merge of the freshly read pointer, what the answer position read before block 24 and what the holder positions hand
    over; the natural reading starts by block 23 (blocks 24-25 remain the earliest window that suffices alone);
  * the block-25 heads (25.18, 25.24, 25.15) carry a position pointer from the name tokens; from the holders they
    carry content that does nothing alone and they lose attention share to them (Phases 414-415);
  * the answer backup (Phase 411, 'via not located') is located: blocks 30-36, heads 30.35, 36.1, 30.38, 36.25 (Phase 414);
  * the fidelity record gets Phase 412 (replication, backups, the failed block-24 prediction).

Checks as in revisions 1-2: cited result files exist with provenance and a matching report hash, verdicts are read from
the result files, every typed number lies within 0.00055 of a number in the report of its phase.

    python tests/glm5/source_binding_chain_v3.py
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

ROOT, RES, PV, n = V1.ROOT, V1.RES, V1.PV, V1.n
RUN = 'SOURCE-BINDING-CHAIN-20261112-v3'
SOURCES = {**V2.SOURCES,
           412: ('2026-11-12/012454_SOURCE-BINDING-fidelity-v2-holdout.json', 'source_binding_fidelity_v2'),
           413: ('2026-11-12/045831_SOURCE-BINDING-arbitration-holdout.json', 'source_binding_arbitration_v1'),
           414: ('2026-11-12/053306_SOURCE-BINDING-backup-holdout.json', 'source_binding_backup_v1'),
           415: ('2026-11-12/061325_SOURCE-BINDING-h2518-holdout.json', 'source_binding_h2518_v1')}
NATURAL = V2.NATURAL
CONFLICT = 'conflict configuration (name tokens exchanged at block 24, Phase 413); attention edits on the answer position only'
V_REGIME = 'direct fetch switched off (code words hidden from the answer position from block 26 on)'

AMEND = {
    'SB.H.25.18': {
        'heads': [18, 24, 15],
        'role': 'heads 25.18, 25.24, 25.15 at the answer position: carry a position pointer from the question name tokens; they also read '
                'the holder positions, whose content does nothing alone but takes attention share',
        'causal': [n(415, 5.7462, 'name-token keys / values of the three heads taken from a donor: shift of score(pointer record) - score(base)'),
                   n(415, 0.0716, 'same exchange: share of answers that change (the other routes still hold the base answer)'),
                   n(415, 2.8262, 'same exchange, head 25.18 alone'),
                   n(415, 0.3408, 'holder keys / values exchanged instead: score shift (answers changed: 0)'),
                   n(415, 0.0011, 'both sources exchanged for three random heads of block 25: score shift'),
                   n(415, 0.3668, 'conflict: the three heads lose the names: drop of the share following the new pointer'),
                   n(414, 0.1307, 'conflict: the three heads lose the holders: gain of the share following the new pointer (random heads -0.0027)'),
                   n(415, 0.0829, 'conflict: holder values zeroed, attention kept: gain (carried content)'),
                   n(415, 0.0524, 'conflict: hiding minus zeroing (attention share)')],
        'verdicts': [(415, 'D1'), (415, 'D2'), (415, 'D3'), (415, 'D4'), (415, 'D5'), (415, 'D6'), (414, 'B5')],
        'counter': 'compensated in the natural run (Phase 410). The pointer shift is large but changes 7 % of the answers only. The '
                   'shift towards the record the donor names is also positive (4.209; the two records often share a factor), so only '
                   'pointer > identity is claimed. Keys and values were exchanged together: QK and OV are not separated'},
}
EDGE_AMEND = {
    'N1': {'blocks': '23-25 (natural reading starts by block 23; blocks 24-25 are the earliest window that suffices alone)',
           'numbers': [n(413, 0.8105, 'both names exchanged at block 23: share following the donor pointer'),
                       n(413, 0.4164, 'at block 24'), n(413, 0.2375, 'at block 25'), n(413, 0.0024, 'at block 26')],
           'status': 'supported as a sufficient route; in the natural run one of several sources that are merged by strength (SB.S.final.merge)'},
    'N3': {'via': 'SB.H.backup (edge N4)', 'status': 'descriptive; route located in Phase 414 (edge N4)'},
}

NEW_CARDS = [
    {
        'id': 'SB.S.final.merge', 'kind': 'state_structure', 'block': '23-25',
        'position': 'final position (outputs the answer)',
        'role': 'which record is fetched is a graded merge of three sources: the pointer read from the name tokens from block 24 on, what '
                'the position itself read before block 24, and what the holder positions hand over; no source has priority',
        'causal': [n(413, 0.4164, 'both names exchanged at block 24: share following the donor pointer'),
                   n(413, 0.0283, 'the answer position cannot read the exchanged tokens from block 24 on'),
                   n(413, 0.5722, 'the answer position cannot read the holders from block 24 on'),
                   n(413, 0.65, 'the answer position could not read the names before block 24'),
                   n(413, 0.7483, 'both removals'),
                   n(413, 0.6591, 'names and all holders exchanged together (only the own state keeps the base)')],
        'verdicts': [(413, 'A0'), (413, 'A1'), (413, 'A2'), (413, 'A3'), (413, 'A4')],
        'counter': 'with both old sources removed 0.129 still keep the base answer (what the position read from the holders before '
                   'block 24 cannot be isolated: hiding the holders in all blocks destroys the task). Donor relations differ widely '
                   '(0.15-0.91); the two removals add up slightly less than additively in the score. Hiding is an intervention',
        'grade': 'in-scope intervention, held-out entities, pre-registered',
        'configuration': CONFLICT,
    },
    {
        'id': 'SB.H.backup', 'kind': 'attention_head', 'block': '30-36', 'heads': [[30, 35], [36, 1], [30, 38], [36, 25], [32, 35], [36, 6]],
        'position': 'final position (outputs the answer)',
        'role': 'copy a ready answer from the holder positions when the direct fetch is off; the first four alone suffice',
        'causal': [n(414, 0.3346, 'code words and holders hidden from block 26 on (floor)'),
                   n(414, 0.9323, 'holders visible only in blocks 30-36'),
                   n(414, 0.832, 'in that window only heads 30.35, 36.1, 30.38, 36.25 read the holders (random heads 0.3646)'),
                   n(414, 0.6302, 'in that window the six heads lose the holders (random heads 0.9284)'),
                   n(414, 0.9883, 'the six heads lose the holders, everything else free')],
        'verdicts': [(414, 'B0'), (414, 'B0w'), (414, 'B1'), (414, 'B2'), (414, 'B3'), (414, 'B6')],
        'counter': 'acts only with the direct fetch off: in the natural run hiding all holders from block 26 on keeps 1.000. Compensated '
                   'outside the window (blocks 37-39, other heads). Not the heads of the pull-back in the conflict (-0.0207). What the '
                   'holders hold is not decomposed',
        'grade': 'in-scope intervention, held-out entities, pre-registered',
        'configuration': V_REGIME,
    },
]
NEW_EDGES = [
    {'id': 'N4', 'source': 'holder positions (ready answer)', 'target': 'final position (answer value)', 'via': 'SB.H.backup',
     'blocks': '30-36', 'type': 'in-scope intervention (attention mask, per head)',
     'numbers': [n(414, 0.9323, 'holders visible only in blocks 30-36'), n(414, 0.832, 'four heads alone'), n(414, 0.3346, 'floor')],
     'status': 'supported as a sufficient route with the direct fetch off; no effect in the natural run', 'configuration': V_REGIME},
    {'id': 'N5', 'source': 'holder positions', 'target': 'final position (pointer stage)', 'via': 'SB.H.25.18 (25.18, 25.24, 25.15)',
     'blocks': '25', 'type': 'in-scope intervention (attention mask, value zeroing, key / value exchange; per head)',
     'numbers': [n(414, 0.1436, 'conflict: holders hidden in blocks 24-25: gain of the share following the new pointer'),
                 n(414, 0.0247, 'holders hidden from block 26 on'),
                 n(414, 0.1307, 'only the three heads lose the holders'),
                 n(415, 0.0829, 'holder values zeroed, attention kept'), n(415, 0.0524, 'hiding minus zeroing')],
     'status': 'supported in the conflict: about 60 % carried content, about 40 % attention share; no effect of the content alone in '
               'the natural run', 'configuration': CONFLICT},
]

FIDELITY = {'phases': [409, 412],
            'numbers': V2.FIDELITY['numbers'] + [
                n(412, 0.9824, 'second new panel, name tokens at blocks 8-22 and 26: agreement of the graph (best baseline 0.7167)'),
                n(412, 1.0, 'R, think block, both together, other question tokens exchanged: agreement of no effect (24,576 edits)'),
                n(412, 0.5962, 'name tokens at block 24: revision-2 graph (position rule)'),
                n(412, 0.7309, 'same edits: no effect - the revision-2 prediction failed')]}

GAPS = [g for g in V2.GAPS if not g.startswith(('what head 25.18 reads', 'where the answer at the think', 'arbitration between',
                                                'fidelity was measured'))] + [
    'how the pointer written by the block-25 heads becomes the query of head 28.21 (blocks 26-27; MLP role not tested)',
    'where the ready answer at the holder positions comes from, and what exactly the backup heads copy',
    'why donor relations differ in the block-24 conflict (0.15-0.91) and why holder content counts in the conflict but not alone',
    'QK and OV of the block-25 heads are not separated; what the answer position read before block 24 is not located by head',
    'fidelity: name-token state exchanges only, four candidate words; blocks 23-25 are a graded transition that a deterministic rule '
    'predicts only partly (Phase 416)',
]


def main():
    cards, edges = copy.deepcopy(V1.CARDS), copy.deepcopy(V1.EDGES)
    for c in cards:
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
    for c in cards:
        a = AMEND.get(c['id'])
        if a:
            c['role_revision_2'], c['counter_revision_2'] = c['role'], c['counter']
            c.update({'role': a['role'], 'counter': a['counter'], 'heads': a['heads'], 'revised_in': 416})
            c['causal'] = c['causal'] + a['causal']
            c['verdicts'] = list(c['verdicts']) + a['verdicts']
    for e in edges:
        a = EDGE_AMEND.get(e['id'])
        if a:
            e['revision_2'] = {k: e[k] for k in a if k != 'numbers'}
            e['numbers'] = e['numbers'] + a.get('numbers', [])
            e.update({k: v for k, v in a.items() if k != 'numbers'})
            e['revised_in'] = 416
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

    payload = {'run': RUN, 'kind': 'summary of Claude Phases 396-415, revision 3 of the Phase 408 summary; no model run',
               'supersedes': 'results/2026-11-12/010635_SOURCE-BINDING-chain-summary-v2.json (kept; revision 2)',
               'task': 'SOURCE-BINDING-20261110-v1', 'scope': V1.SCOPE,
               'natural_short_chain': ['E1', 'E2', 'N1', 'N2', 'E8'], 'backup_routes': ['E3', 'E4', 'E5', 'E6', 'E7', 'N3', 'N4', 'N5'],
               'fidelity_verdicts_412': verd[412],
               'cards': cards, 'edges': edges, 'fidelity': FIDELITY, 'gaps': GAPS, 'sources': sources,
               'checks': {'numbers_checked_against_reports': checked, 'reports_hash_matched': len(sources),
                          'verdicts_read_from_result_files': sum(len(c['verdicts']) for c in cards)},
               'formal_promotion': 0}
    out = PV.write_result('SOURCE-BINDING-chain-summary-v3', payload, model_path=V1.MODEL, seed=0, tool=RUN)
    print(json.dumps({'result': str(out), **payload['checks'], 'cards': len(cards), 'edges': len(edges)}, ensure_ascii=False))
    for c in cards:
        if c.get('revised_in') == 416 or c['id'] in ('SB.S.final.merge', 'SB.H.backup'):
            print(c['id'], [(v['phase'], v['key'], v['verdict']) for v in c['verdicts'] if v['phase'] >= 413])


if __name__ == '__main__':
    main()
