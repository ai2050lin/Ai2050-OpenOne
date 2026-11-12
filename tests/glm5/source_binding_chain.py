"""Source-binding chain summary (Claude Phase 408): feature cards and attribution edges.

No model is run. The script assembles the findings of Claude Phases 396-407
(Qwen3-14B, SOURCE-BINDING-20261110-v1) into one structured file and checks it
against the stored products:

  * every cited result file must exist, carry the five provenance fields, and its
    report_sha256 must match the report.json on disk;
  * every verdict a card or edge cites is read from the result file, not typed;
  * every number typed below must lie within 0.00055 of a number in the report.json of
    the phase it is attributed to.

Numbers are transcribed from the research record; the checks guard against
transcription errors, they are not a re-analysis.  Evidence grades are
descriptive.  Formal promotion: 0.

    python tests/glm5/source_binding_chain.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools import provenance as PV  # noqa: E402

RUN = 'SOURCE-BINDING-CHAIN-20261111-v1'
MODEL = ROOT / 'models' / 'hf' / 'Qwen3-14B'
RES = ROOT / 'tests' / 'glm5' / 'result'

# phase -> (result file under results/, product directory)
SOURCES = {
    396: ('2026-11-10/175250_SOURCE-BINDING-ablation-holdout.json', 'source_binding_ablation_v2'),
    397: ('2026-11-10/224039_SOURCE-BINDING-relay-holdout.json', 'source_binding_relay_v1'),
    399: ('2026-11-11/003104_SOURCE-BINDING-upstream-holdout.json', 'source_binding_upstream_v1'),
    400: ('2026-11-11/025431_SOURCE-BINDING-heads-holdout.json', 'source_binding_heads_v1'),
    401: ('2026-11-11/034145_SOURCE-BINDING-qkov-holdout.json', 'source_binding_qkov_v1'),
    402: ('2026-11-11/045534_SOURCE-BINDING-query-holdout.json', 'source_binding_query_v1'),
    403: ('2026-11-11/052237_SOURCE-BINDING-request-holdout.json', 'source_binding_request_v1'),
    404: ('2026-11-11/055856_SOURCE-BINDING-twodim-holdout.json', 'source_binding_twodim_v1'),
    405: ('2026-11-11/061629_SOURCE-BINDING-pointer-holdout.json', 'source_binding_pointer_v1'),
    406: ('2026-11-11/064309_SOURCE-BINDING-nameloc-holdout.json', 'source_binding_nameloc_v1'),
    407: ('2026-11-11/105440_SOURCE-BINDING-writers-holdout.json', 'source_binding_writers_v1'),
}

SCOPE = ('Qwen3-14B bf16; source-binding four-way choice (chance 0.25); new entity names, '
         'seen answer words and frames; one model, one task')


def n(phase, value, what):
    """A number attributed to a phase; checked against that phase's report.json."""
    return {'phase': phase, 'value': value, 'what': what}


# ---------------------------------------------------------------- feature cards
CARDS = [
    {
        'id': 'SB.H.21.5', 'kind': 'attention_head', 'block': 21, 'heads': [5, 7],
        'position': 'entity-name tokens of the question',
        'role': 'reads the four records into the entity-name tokens; either head alone suffices',
        'causal': [
            n(407, 0.921, 'only head 21.5 may use the edge (floor 0.626, random heads 0.62-0.65)'),
            n(407, 0.686, 'necessary set (32 heads, rule not met, flagged) blocked and edge closed after window; random 1.000'),
        ],
        'verdicts': [(407, 'qent.H1_sufficient'), (407, 'qent.H2_necessary_in_window'),
                     (407, 'qent.H3_no_later_compensation')],
        'counter': 'blocking 21.5 or 21.7 alone has no effect (mutual backup); no small necessary set found; '
                   'H2 passed narrowly (0.686 against 0.70)',
        'grade': 'in-scope intervention, held-out entities, pre-registered',
    },
    {
        'id': 'SB.H.early8', 'kind': 'attention_head_set', 'block': '4-11',
        'heads': [[9, 3], [9, 2], [4, 10], [5, 15], [7, 4], [9, 1], [9, 37], [11, 38]],
        'position': 'archive-name token of the question',
        'role': 'reads the header line of the requested archive into the archive-name token',
        'causal': [
            n(407, 0.814, 'only these 8 heads may use the edge (floor 0.569, random heads 0.57)'),
            n(407, 0.587, '12-head necessary set blocked and edge closed after window; random 0.89'),
            n(407, 1.0, '12-head set blocked, later blocks untouched: no loss'),
        ],
        'verdicts': [(407, 'qsrc.H1_sufficient'), (407, 'qsrc.H2_necessary_in_window'),
                     (407, 'qsrc.H3_no_later_compensation')],
        'counter': 'later blocks compensate (H3 not supported); window end rule not met (flagged); '
                   'strongest single head reaches only 0.744',
        'grade': 'in-scope intervention, held-out entities, pre-registered; distributed and redundant in time',
    },
    {
        'id': 'SB.S.qsrc.block', 'kind': 'residual_state', 'block': '16-22',
        'position': 'archive-name token of the question',
        'role': 'carries which archive block (first / second) holds the requested record',
        'causal': [
            n(406, 0.809, 'state swapped from other-archive-order donor at block 16: other-archive answers'),
            n(406, 0.981, 'same at block 22'),
        ],
        'verdicts': [(406, 'N1_carriers'), (406, 'N2_non_carriers')],
        'counter': 'the pre-registered position prediction for this series was wrong (it carries only the block '
                   'dimension), so N2 is vacuous for it; effect gone by block 26',
        'grade': 'in-scope intervention, held-out entities, natural configuration',
    },
    {
        'id': 'SB.S.qent.line', 'kind': 'residual_state', 'block': 22,
        'position': 'entity-name tokens of the question',
        'role': 'carries which line inside an archive block holds the requested entity',
        'causal': [n(406, 0.874, 'state swapped from other-entity-order donor at block 22: other-entity answers'),
                   n(406, 0.996, 'both names swapped at block 22: answer follows position')],
        'verdicts': [(406, 'N1_carriers')],
        'counter': 'absent before block 22 (0.000-0.008 at blocks 16-20); 0.408 at block 24',
        'grade': 'in-scope intervention, held-out entities, natural configuration',
    },
    {
        'id': 'SB.S.R.pointer', 'kind': 'residual_state', 'block': '20-24',
        'position': 'relay R (line break after the user turn)',
        'role': 'two separable dimensions: archive block and line; mostly ordinal position, partly content',
        'causal': [
            n(404, 0.999, 'only archive name readable: archive right'),
            n(404, 0.514, 'only archive name readable: entity at chance'),
            n(404, 0.932, 'archive dimension swapped at block 24: other-archive answers'),
            n(404, 0.927, 'entity dimension swapped: other-entity answers'),
            n(404, 0.755, 'two single-dimension shifts added: both-wrong record (random vector keeps 0.892)'),
            n(405, 0.893, 'conflict donor (both layout factors): follows position'),
            n(405, 0.02, 'same donor: follows content'),
        ],
        'verdicts': [(403, 'W1'), (404, 'K1'), (404, 'K2'), (404, 'X1'), (404, 'X2'), (404, 'X3'),
                     (405, 'L1'), (405, 'L2'), (405, 'L3'), (405, 'L4')],
        'counter': 'line dimension only about two thirds positional (0.66-0.68 against 0.32-0.34 content); '
                   'measured in the isolated configuration; in the natural one R also reads the question at blocks 24-27',
        'grade': 'in-scope intervention, held-out entities and held-out layout factor, pre-registered',
    },
    {
        'id': 'SB.S.R.query', 'kind': 'residual_state', 'block': '24-25',
        'position': 'relay R',
        'role': 'the query that selects the record for head 30.4; formed from several redundant sources',
        'causal': [
            n(402, 0.331, 'R reads nothing from block 24 on'),
            n(402, 0.994, 'R reads nothing from block 26 on: query already formed'),
            n(402, 0.273, 'four source groups blocked together'),
            n(402, 0.919, 'only the four records readable'),
        ],
        'verdicts': [(402, 'V1'), (402, 'V2'), (402, 'V3'), (402, 'A1'), (402, 'A2')],
        'counter': 'no single group is necessary; the necessary set is not complete',
        'grade': 'in-scope intervention, held-out entities, pre-registered',
    },
    {
        'id': 'SB.H.28.21', 'kind': 'attention_head', 'block': 28, 'heads': [21],
        'position': 'relay R',
        'role': 'assists head 30.4; attends to the correct record',
        'causal': [n(400, 0.958, 'only 30.4 and 28.21 may read the correct record (random heads 0.000)'),
                   n(400, 0.949, '28.21 alone blocked in window')],
        'verdicts': [(400, 'H1'), (401, 'P1')],
        'counter': 'alone neither sufficient (0.005) nor necessary (0.949)',
        'grade': 'in-scope intervention, held-out entities, pre-registered',
    },
    {
        'id': 'SB.H.30.4', 'kind': 'attention_head', 'block': 30, 'heads': [4],
        'position': 'relay R',
        'role': 'reads the value token of the record the query points to and writes the value into R',
        'causal': [
            n(400, 0.29, 'blocked inside the window (random head of block 30: 0.997)'),
            n(400, 0.216, 'blocked, everything else untouched'),
            n(401, 0.964, 'attention mass on the correct record'),
            n(401, 0.957, 'query swapped from a donor asking for another record: answer follows donor'),
            n(401, 0.953, 'head output swapped: answer follows donor'),
            n(401, 0.997, 'may read only the value token'),
            n(401, 0.232, 'may not read the value token'),
        ],
        'verdicts': [(400, 'H1'), (400, 'H2'), (400, 'H3'), (401, 'P1'), (401, 'P2a'), (401, 'P2b'), (401, 'P3'), (401, 'P4')],
        'counter': 'alone sufficient only to 0.217; a same-request donor with reversed archive order moves '
                   'only 0.42 of the attention mass (the query is positional)',
        'grade': 'in-scope intervention, held-out entities, pre-registered; necessary single head',
    },
    {
        'id': 'SB.S.final.centre', 'kind': 'residual_subspace', 'block': 36,
        'position': 'final position',
        'role': 'the answer sits in the 15-dimensional subspace spanned by the answer-word class means, '
                'not in the output-weight subspace',
        'causal': [
            n(396, 0.279, 'centre subspace deleted and all context blocked'),
            n(396, 0.806, 'norm-matched nuisance subspace deleted and all context blocked'),
            n(396, 0.969, 'output-weight subspace deleted and all context blocked'),
            n(396, 0.856, 'centre subspace deleted only: later blocks re-fetch'),
        ],
        'verdicts': [(396, '36.Q2b'), (396, '36.Q1'), (396, '36.Q2')],
        'counter': 'single-layer deletion is not a test of necessity (Q1 inconclusive); main layer chosen after the '
                   'discovery pilot; seen answer words only',
        'grade': 'in-scope intervention, held-out entities, pre-registered main layer',
    },
]

# ---------------------------------------------------------------- attribution edges
EDGES = [
    {
        'id': 'E1', 'source': 'header line of the requested archive', 'target': 'SB.S.qsrc.block',
        'via': 'SB.H.early8', 'blocks': '4-11', 'type': 'in-scope intervention (attention edge, per head)',
        'numbers': [n(407, 0.814, 'sufficient set only'), n(407, 0.587, 'necessary set blocked, edge closed later')],
        'status': 'supported, but redundant in time: blocks from 12 on can re-read the header line',
    },
    {
        'id': 'E2', 'source': 'the four records', 'target': 'SB.S.qent.line',
        'via': 'SB.H.21.5', 'blocks': '18-25', 'type': 'in-scope intervention (attention edge, per head)',
        'numbers': [n(407, 0.921, 'head 21.5 only'), n(407, 0.686, '32 heads blocked')],
        'status': 'supported; 21.5 and 21.7 back each other up',
    },
    {
        'id': 'E3', 'source': 'SB.S.qsrc.block', 'target': 'SB.S.R.pointer',
        'via': 'not located (R attends to the archive-name token)', 'blocks': '20-23',
        'type': 'in-scope intervention (attention mask, all heads; state swap)',
        'numbers': [n(404, 0.999, 'only archive name readable: archive right'),
                    n(403, 0.999, 'natural configuration: blocking the question in 20-23 costs nothing')],
        'status': 'supported in the isolated configuration; in the natural one a direct route at blocks 24-27 replaces it',
    },
    {
        'id': 'E4', 'source': 'SB.S.qent.line', 'target': 'SB.S.R.pointer',
        'via': 'not located (R attends to the entity-name tokens)', 'blocks': '20-23',
        'type': 'in-scope intervention (attention mask, all heads; state swap)',
        'numbers': [n(404, 0.587, 'only entity name readable: archive at chance, entity 1.000')],
        'status': 'supported in the isolated configuration',
    },
    {
        'id': 'E5', 'source': 'SB.S.R.pointer', 'target': 'SB.S.R.query',
        'via': 'not decomposed', 'blocks': '24-25', 'type': 'in-scope intervention (state swap, query swap)',
        'numbers': [n(404, 0.932, 'pointer state swapped: other record chosen'),
                    n(401, 0.957, 'query of head 30.4 swapped: answer follows')],
        'status': 'supported end to end; the map from pointer state to the query vector is not decomposed',
    },
    {
        'id': 'E6', 'source': 'value token of the pointed record', 'target': 'relay R (answer value)',
        'via': 'SB.H.30.4 with SB.H.28.21', 'blocks': '28-31',
        'type': 'in-scope intervention (attention edge per head; query and output swaps)',
        'numbers': [n(399, 0.884, 'R may read only the correct record'),
                    n(399, 0.004, 'R may not read the correct record: a definite wrong value'),
                    n(400, 0.958, 'two heads only'), n(400, 0.29, 'head 30.4 blocked in window')],
        'status': 'supported; single route without redundancy',
    },
    {
        'id': 'E7', 'source': 'relay R (answer value)', 'target': 'SB.S.final.centre',
        'via': 'not located (final position attends to R)', 'blocks': '24-36',
        'type': 'in-scope intervention (attention mask, all heads)',
        'numbers': [n(397, 1.0, 'final position may read only R'),
                    n(397, 0.401, 'R, the line break after </think> and the records blocked together'),
                    n(397, 0.262, 'all context blocked')],
        'status': 'supported; one of three redundant routes, the joint block leaves 0.401',
    },
    {
        'id': 'E8', 'source': 'SB.S.final.centre', 'target': 'output answer word',
        'via': 'blocks 36-39 and unembedding', 'blocks': '36-39',
        'type': 'in-scope intervention (subspace deletion)',
        'numbers': [n(396, 0.279, 'centre deleted'), n(396, 0.806, 'norm-matched control deleted')],
        'status': 'supported at the pre-registered main layer',
    },
]

GAPS = [
    'how reading the header line (blocks 4-11) becomes the block pointer on the archive-name token (blocks 14-16)',
    'which heads move the pointers from the name tokens to R (blocks 20-23) and from the question to R directly (24-27)',
    'how the pointer state at R is turned into the query vector of head 30.4 (blocks 24-27); MLP role not tested',
    'which heads carry the value from R to the final position; shares of the three redundant routes in the natural run',
    'the residual of every step (sufficient sets reach 0.81-0.96, not 1.00)',
    'no dictionary features, no natural-corpus activation statistics, no unembedding read-out of the heads',
    'one model, one task, seen answer words and frames; GLM-4-9B did not pass the behaviour gate of this task',
    'no graph-only prediction of held-out outputs was computed: this is a chain of interventions, not an atlas fidelity score',
]


def floats(obj, out):
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        out.add(abs(float(obj)))
    elif isinstance(obj, dict):
        for v in obj.values():
            floats(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            floats(v, out)


def verdict(tree, key):
    """Find a verdict by dotted key; the last part may be a prefix (e.g. 'H1' for 'H1_sufficient')."""
    node = tree
    parts = key.split('.')
    for i, part in enumerate(parts):
        if not isinstance(node, dict):
            raise KeyError(key)
        hit = [k for k in node if k == part or k.startswith(part + '_')]
        if len(hit) != 1:
            raise KeyError(f'{key}: {sorted(node)}')
        node = node[hit[0]]
    if isinstance(node, dict):
        node = node.get('verdict', node)
    return node


def main():
    sources, pool, verd = {}, {}, {}
    for phase, (rel, folder) in SOURCES.items():
        path = ROOT / 'results' / rel
        doc = json.loads(path.read_text(encoding='utf-8'))
        problems = PV.check_provenance(doc)
        assert not problems, (rel, problems)
        payload = doc['payload']
        assert payload.get('gate_passed') is True, rel
        report = RES / folder / 'report.json'
        sha = hashlib.sha256(report.read_bytes()).hexdigest()
        assert sha == payload['report_sha256'], (rel, sha, payload['report_sha256'])
        got = set()
        floats(json.loads(report.read_text(encoding='utf-8')), got)
        pool[phase] = got
        verd[phase] = payload['verdicts']
        sources[str(phase)] = {
            'result_file': 'results/' + rel, 'result_sha8': doc['sha8'],
            'report': f'tests/glm5/result/{folder}/report.json', 'report_sha256': sha,
            'run': doc['provenance']['tool'], 'git_sha': doc['provenance']['git_sha'],
            'seed': doc['provenance']['seed'], 'verdicts': payload['verdicts'],
        }

    missing, checked = [], 0
    for item in CARDS + EDGES:
        for num in item.get('causal', []) + item.get('numbers', []):
            checked += 1
            if min(abs(abs(num['value']) - f) for f in pool[num['phase']]) > 0.00055:
                missing.append((item['id'], num['phase'], num['value']))
    assert not missing, missing

    for card in CARDS:
        card['verdicts'] = [{'phase': p, 'key': k, 'verdict': verdict(verd[p], k)} for p, k in card['verdicts']]
        card.update({'model': 'Qwen3-14B bf16', 'scope': SCOPE,
                     'natural_corpus_statistics': 'not measured',
                     'unembedding_readout': 'not measured' if card['id'] != 'SB.S.final.centre'
                     else 'output-weight subspace deletion keeps 0.969: the answer is not mainly in that subspace',
                     'formal_promotion': 0})
    for edge in EDGES:
        edge.update({'scope': SCOPE, 'formal_promotion': 0})

    payload = {
        'run': RUN, 'kind': 'summary of Claude Phases 396-407; no model run',
        'task': 'SOURCE-BINDING-20261110-v1', 'scope': SCOPE,
        'cards': CARDS, 'edges': EDGES, 'gaps': GAPS, 'sources': sources,
        'checks': {'numbers_checked_against_reports': checked, 'reports_hash_matched': len(sources),
                   'verdicts_read_from_result_files': sum(len(c['verdicts']) for c in CARDS)},
        'formal_promotion': 0,
    }
    out = PV.write_result('SOURCE-BINDING-chain-summary', payload, model_path=MODEL, seed=0, tool=RUN)
    print(json.dumps({'result': str(out), **payload['checks'], 'cards': len(CARDS), 'edges': len(EDGES)},
                     ensure_ascii=False))
    for card in CARDS:
        print(card['id'], [(v['phase'], v['key'], v['verdict']) for v in card['verdicts']])


if __name__ == '__main__':
    main()
