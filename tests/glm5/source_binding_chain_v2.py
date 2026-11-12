"""Source-binding chain summary, revision 2 (Claude Phase 412): cards and edges after Phases 409-411.

No model is run.  Revision 1 (source_binding_chain.py, Phase 408) is imported unchanged and amended, not rewritten:

  * every card and edge of revision 1 gets a `configuration` field.  The part that passes through the relay position R
    (Phases 397-405) was measured with the final position restricted to R; Phases 409-411 show that in the natural run
    it is one of several parallel routes, so it is marked 'isolated configuration; backup route in the natural run';
  * the entity-name card is amended: the tokens carry a map archive block -> line, not one line (Phase 409);
  * new cards and edges for the natural short chain: the answer position reads the name tokens (blocks 24-25, head
    25.18; Phase 410) and fetches the code word (block 28, head 28.21; Phase 411); the redundancy between the direct
    fetch, R and the think positions is recorded as a structure with its counter-examples.

The checks are those of revision 1: cited result files exist with provenance and a matching report hash, verdicts are
read from the result files, every typed number lies within 0.00055 of a number in the report of its phase.

    python tests/glm5/source_binding_chain_v2.py
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import source_binding_chain as V1  # noqa: E402

ROOT, RES, PV, n = V1.ROOT, V1.RES, V1.PV, V1.n
RUN = 'SOURCE-BINDING-CHAIN-20261112-v2'
SOURCES = {**V1.SOURCES,
           409: ('2026-11-11/213220_SOURCE-BINDING-fidelity-holdout.json', 'source_binding_fidelity_v1'),
           410: ('2026-11-12/000153_SOURCE-BINDING-readers-holdout.json', 'source_binding_readers_v1'),
           411: ('2026-11-12/004654_SOURCE-BINDING-fetch-holdout.json', 'source_binding_fetch_v1')}

NATURAL = 'natural run (nothing restricted except the edit itself)'
ISOLATED = ('isolated configuration (final position restricted to R); in the natural run a backup route: '
            'exchanging the state of R never changes the answer (Phase 409), only R visible to the final position still gives 1.000 (Phase 411)')
CONFIG = {'SB.H.21.5': NATURAL, 'SB.H.early8': NATURAL, 'SB.S.qsrc.block': NATURAL, 'SB.S.qent.line': NATURAL,
          'SB.S.R.pointer': ISOLATED, 'SB.S.R.query': ISOLATED, 'SB.H.28.21': ISOLATED, 'SB.H.30.4': ISOLATED,
          'SB.S.final.centre': NATURAL,
          'E1': NATURAL, 'E2': NATURAL, 'E3': ISOLATED, 'E4': ISOLATED, 'E5': ISOLATED, 'E6': ISOLATED, 'E7': ISOLATED, 'E8': NATURAL}

AMEND = {
    'SB.S.qent.line': {
        'role': 'carries, for each archive block, the line that holds the requested entity (a map block -> line), not one line',
        'causal': [n(409, 0.9299, 'entity-name tokens exchanged at block 22: agreement of the map reading'),
                   n(409, 0.4544, 'same edits: agreement of the single-line reading')],
        'verdicts': [(409, 'F1'), (409, 'F2')]},
    'SB.S.qsrc.block': {
        'causal': [n(409, 0.9801, 'archive-name token exchanged at block 18, 15 donor relations: agreement of the graph'),
                   n(409, 0.9974, 'same at block 22')],
        'verdicts': [(409, 'F3')]},
    'SB.H.28.21': {
        'causal': [n(411, 0.9388, 'the same head at the FINAL position, natural run: alone sufficient for the direct fetch (card SB.H.28.21.final)')],
        'verdicts': []},
}

NEW_CARDS = [
    {
        'id': 'SB.H.25.18', 'kind': 'attention_head', 'block': 25, 'heads': [18],
        'position': 'final position (outputs the answer)',
        'role': 'reads the archive-name token and the entity-name tokens of the question directly; the same head for both',
        'causal': [n(410, 0.9805, 'archive name: only the final position may read it, only in blocks 24-25, only head 25.18 (random heads 0.4987)'),
                   n(410, 0.9049, 'entity name: same (random heads 0.5)'),
                   n(410, 0.5391, 'archive name: blocks 24-25 open, heads 25.18 / 25.15 / 25.16 cut (random heads 1.0)'),
                   n(410, 0.6536, 'entity name: blocks 24-25 open, heads 25.18 / 25.15 cut')],
        'verdicts': [(410, 'qsrc.H0'), (410, 'qsrc.H0w'), (410, 'qsrc.H1'), (410, 'qsrc.H2'), (410, 'qsrc.H3'),
                     (410, 'qent.H0'), (410, 'qent.H0w'), (410, 'qent.H1'), (410, 'qent.H2'), (410, 'qent.H3')],
        'counter': 'compensated in the natural run: cutting these heads alone keeps 1.000 / 0.974; with the final position unable to read '
                   'the names at all 0.969 / 0.781 remain (other positions read them). What the head reads (pointer or name) is not separated',
        'grade': 'in-scope intervention, held-out entities, pre-registered',
        'configuration': NATURAL,
    },
    {
        'id': 'SB.H.28.21.final', 'kind': 'attention_head', 'block': 28, 'heads': [21],
        'position': 'final position (outputs the answer)',
        'role': 'attends to the four code-word tokens and selects the one of the requested record (direct fetch)',
        'causal': [n(411, 0.9388, 'in blocks >= 26 the final position sees only the four code words, only in block 28, only head 28.21 (random heads 0.3203)'),
                   n(411, 0.6901, 'block 28 open, six heads cut (random six heads 0.9974)'),
                   n(411, 0.0352, 'natural run, only the correct code word hidden: the head copies another record')],
        'verdicts': [(411, 'final.H0'), (411, 'final.H0w'), (411, 'final.H1'), (411, 'final.H2'), (411, 'final.H3')],
        'counter': 'compensated in the natural run: cutting the six heads keeps 0.997, hiding all four code words keeps 1.000. '
                   'QK / OV not decomposed. The necessary-set verdict passes close to its threshold (0.690 against 0.70)',
        'grade': 'in-scope intervention, held-out entities, pre-registered',
        'configuration': NATURAL,
    },
    {
        'id': 'SB.S.answer.copies', 'kind': 'state_structure', 'block': '26-39',
        'position': 'relay R; the four tokens of the empty think block',
        'role': 'each already holds an answer the final position can copy; with the direct fetch three mutually redundant sources',
        'causal': [n(411, 1.0, 'in blocks >= 26 the final position sees only R'), n(411, 1.0, 'only the think block'),
                   n(411, 0.9948, 'only the four code words'), n(411, 0.8685, 'code words and R hidden'),
                   n(411, 0.418, 'code words, R and the think block hidden'), n(411, 0.3203, 'everything hidden')],
        'verdicts': [],
        'counter': 'descriptive arms, no pre-registered verdict. Where the answer at the think positions comes from is not tested. '
                   'With the sources in conflict the direct fetch wins (0.0352 when only the correct code word is hidden)',
        'grade': 'in-scope intervention, held-out entities, descriptive',
        'configuration': NATURAL,
    },
]

NEW_EDGES = [
    {'id': 'N1', 'source': 'SB.S.qsrc.block, SB.S.qent.line', 'target': 'final position (pointer read)', 'via': 'SB.H.25.18 (25.15, 25.16 assist)',
     'blocks': '24-25', 'type': 'in-scope intervention (attention mask, per head)',
     'numbers': [n(410, 1.0, 'only the final position may read the name token (both tokens)'), n(410, 0.9805, 'head 25.18 alone, archive name'),
                 n(410, 0.9049, 'head 25.18 alone, entity name')],
     'status': 'supported as a sufficient route; redundant in the natural run', 'configuration': NATURAL},
    {'id': 'N2', 'source': 'code-word token of the pointed record', 'target': 'final position (answer value)', 'via': 'SB.H.28.21.final',
     'blocks': '28', 'type': 'in-scope intervention (attention mask, per head)',
     'numbers': [n(411, 0.9974, 'the four code words visible only in block 28'), n(411, 0.9388, 'head 28.21 alone')],
     'status': 'supported as a sufficient route; redundant in the natural run', 'configuration': NATURAL},
    {'id': 'N3', 'source': 'SB.S.answer.copies', 'target': 'final position (answer value)', 'via': 'not located',
     'blocks': '26-39', 'type': 'in-scope intervention (attention mask, all heads), descriptive',
     'numbers': [n(411, 1.0, 'only R visible'), n(411, 0.418, 'all three sources hidden')],
     'status': 'descriptive; replaces the reading of E7 as the main route', 'configuration': NATURAL},
]

FIDELITY = {'phase': 409, 'numbers': [n(409, 0.9833, 'name-token edits never run before: agreement of the graph'),
                                      n(409, 0.7045, 'same edits: best baseline (name only)'),
                                      n(409, 0.5907, 'R and other question tokens: agreement of the revision-1 graph (no effect: 1.0)')]}

GAPS = [g for g in V1.GAPS if not g.startswith(('which heads carry the value from R', 'no graph-only prediction'))] + [
    'what head 25.18 reads from the name tokens (position pointer or name) and how the final position turns it into the query of 28.21',
    'where the answer at the think positions comes from; which heads let the final position copy from R and from the think block',
    'arbitration between the three sources is known only for one conflict (correct code word hidden: the direct fetch wins)',
    'fidelity was measured for one family of edits (state exchange of name tokens and R), four candidate words; see Phases 409 and 412',
]


def main():
    cards, edges = copy.deepcopy(V1.CARDS), copy.deepcopy(V1.EDGES)
    for c in cards:
        a = AMEND.get(c['id'], {})
        if 'role' in a:
            c['role_revision_1'], c['role'] = c['role'], a['role']
        c['causal'] = c['causal'] + a.get('causal', [])
        c['verdicts'] = list(c['verdicts']) + a.get('verdicts', [])
        c['configuration'] = CONFIG[c['id']]
        c['revised_in'] = 412 if a or CONFIG[c['id']] == ISOLATED else None
    for e in edges:
        e['configuration'] = CONFIG[e['id']]
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
            if min(abs(abs(num['value']) - f) for f in pool[num['phase']]) > 0.00055:
                missing.append((item['id'], num['phase'], num['value']))
    assert not missing, missing

    for c in cards:
        c['verdicts'] = [{'phase': p, 'key': k, 'verdict': V1.verdict(verd[p], k)} for p, k in c['verdicts']]
        c.update({'model': 'Qwen3-14B bf16', 'scope': V1.SCOPE, 'natural_corpus_statistics': 'not measured',
                  'unembedding_readout': 'not measured' if c['id'] != 'SB.S.final.centre'
                  else 'output-weight subspace deletion keeps 0.969: the answer is not mainly in that subspace', 'formal_promotion': 0})
    for e in edges:
        e.update({'scope': V1.SCOPE, 'formal_promotion': 0})

    payload = {'run': RUN, 'kind': 'summary of Claude Phases 396-411, revision 2 of the Phase 408 summary; no model run',
               'supersedes': 'results/2026-11-11/202044_SOURCE-BINDING-chain-summary.json (kept; revision 1)',
               'task': 'SOURCE-BINDING-20261110-v1', 'scope': V1.SCOPE,
               'natural_short_chain': ['E1', 'E2', 'N1', 'N2', 'E8'], 'backup_routes': ['E3', 'E4', 'E5', 'E6', 'E7', 'N3'],
               'cards': cards, 'edges': edges, 'fidelity': FIDELITY, 'gaps': GAPS, 'sources': sources,
               'checks': {'numbers_checked_against_reports': checked, 'reports_hash_matched': len(sources),
                          'verdicts_read_from_result_files': sum(len(c['verdicts']) for c in cards)},
               'formal_promotion': 0}
    out = PV.write_result('SOURCE-BINDING-chain-summary-v2', payload, model_path=V1.MODEL, seed=0, tool=RUN)
    print(json.dumps({'result': str(out), **payload['checks'], 'cards': len(cards), 'edges': len(edges)}, ensure_ascii=False))
    for c in cards[len(V1.CARDS):]:
        print(c['id'], [(v['phase'], v['key'], v['verdict']) for v in c['verdicts']])


if __name__ == '__main__':
    main()
