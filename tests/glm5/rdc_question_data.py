"""Read only declared native split/field boundaries for Phase2748 algorithms."""
from scipy import sparse
from rdc_question_common import *


def index(key, splits):
    assert set(splits) <= {'train', 'validation', 'diagnostic', 'confirmation'}
    confirmation = 'confirmation' in splits
    assert not confirmation or set(splits) == {'confirmation'}
    contract, manifest, rows, groups = material(key, confirmation)
    rows = sorted([r for r in rows if r['split'] in splits],
                  key=lambda r: (r['cohort'], r['group_id'], r['within_context_index']))
    scope = 'confirmation' if confirmation else 'nonconfirmation'
    final = read(OUT/f'native/{key}/{scope}/result.json')
    assert final['all_passed'] and final['execution']['material_manifest_sha256'] == sha(OUT/'material/manifest.json')
    needed = {r['group_id'] for r in rows}
    groupdata, questiondata = {}, {}
    for gid in needed:
        g = read(OUT/f'native/{key}/{scope}/groups/{gid}.json')
        assert g['split'] in splits and g['execution'] == final['execution']
        groupdata[gid] = g
        for q in g['questions']:
            questiondata[q['question_id']] = q
    assert len(questiondata) == len(rows)
    return rows, groupdata, questiondata


def field(reference, names):
    path = ROOT/reference['path']
    assert sha(path) == reference['sha256']
    with np.load(path) as z:
        return {k: (unbits(z[k]).astype(np.float64) if z[k].dtype == np.uint16 else z[k].astype(np.float64)) for k in names}


def native_features(key, rows, groupdata, questiondata):
    result = {k: [] for k in ['H12', 'question_H0', 'context_H0', 'context_H12', 'read', 'shuffle_read']}
    contexts = {}
    for gid, g in groupdata.items():
        contexts[gid] = field(g['context_field'], ['context_H0_mean', 'context_H12_mean'])
    for r in rows:
        values = field(questiondata[r['question_id']]['field'],
            ['H12_last_BF16', 'question_H0_mean', 'native_source_read_BF16', 'source_value_pair_shuffle_BF16'])
        result['H12'].append(values['H12_last_BF16'])
        result['question_H0'].append(values['question_H0_mean'])
        result['read'].append(values['native_source_read_BF16'])
        result['shuffle_read'].append(values['source_value_pair_shuffle_BF16'])
        result['context_H0'].append(contexts[r['group_id']]['context_H0_mean'])
        result['context_H12'].append(contexts[r['group_id']]['context_H12_mean'])
    return {k: np.stack(v) for k, v in result.items()}


def targets(rows, questiondata, target):
    assert target in {'postnorm', 'H24', 'H32', 'H_last', 'MLP24_product'}
    name = 'postnorm_BF16' if target == 'postnorm' else 'block24_product_BF16' if target == 'MLP24_product' else 'hidden_BF16'
    result = []
    for row in rows:
        value = field(questiondata[row['question_id']]['field'], [name])[name]
        if target.startswith('H'):
            value = value[-1 if target == 'H_last' else int(target[1:])]
        result.append(value)
    return np.stack(result)


def lexical_features(key, rows):
    width = read(ROOT/'models/hf'/MODELS[key]/'config.json')['vocab_size']
    triplets = {'question': ([], [], []), 'context': ([], [], [])}
    aux = []
    for i, r in enumerate(rows):
        token = r['tokens']
        ids = np.asarray(token['input_ids'])
        for role in ['question', 'context']:
            positions = token[role+'_token_positions']
            selected = ids[positions]
            countids, counts = np.unique(selected, return_counts=True)
            a, b, v = triplets[role]
            a.extend([i]*len(countids)); b.extend(countids.tolist()); v.extend((counts/len(selected)).tolist())
        q = token['question_token_positions']
        # Registered inputs are all known before the predicted response.
        aux.append([len(q)/1024, len(token['context_token_positions'])/1024,
            len(ids)/1024, token['context_prefix_length']/1024, q[0]/1024, q[-1]/1024])
    matrices = {}
    for role, (a, b, v) in triplets.items():
        matrices[role] = sparse.csr_matrix((v, (a, b)), shape=(len(rows), width), dtype=np.float64)
    return sparse.hstack([matrices['question'], sparse.csr_matrix(aux)], format='csr'), matrices['context']


def variant_features(key, rows, values, variant):
    zero = np.zeros((len(rows), 1), dtype=np.float64)
    if variant == 'lexical_position':
        return lexical_features(key, rows)
    choices = {
        'early_query_context': (values['H12'], values['context_H12']),
        'early_query_only': (values['H12'], zero),
        'context_only': (zero, values['context_H12']),
        'question_embedding_only': (values['question_H0'], zero),
        'embedding_question_context': (values['question_H0'], values['context_H0']),
        'native_source_read': (np.concatenate([values['H12'], values['read']], axis=1), values['context_H12']),
        'source_value_pair_shuffle': (np.concatenate([values['H12'], values['shuffle_read']], axis=1), values['context_H12']),
        'within_context_target_pair_shuffle': (values['H12'], values['context_H12']),
    }
    return choices[variant]


def target_permutation(rows):
    locations = {(r['group_id'], r['within_context_index']): i for i, r in enumerate(rows)}
    order = np.array([locations[(r['group_id'], (r['within_context_index']+1) % 4)] for r in rows])
    assert len(np.unique(order)) == len(rows)
    return order
