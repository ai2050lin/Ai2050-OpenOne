"""Freeze substantially covered learning data and source-separated diagnostics."""
from collections import Counter, defaultdict
import unicodedata
from rdc_formation_common import *

CLASS_NAMES = ['special', 'newline', 'whitespace', 'decimal', 'punctuation_symbol',
               'binary_English', 'binary_Chinese', 'CJK_only', 'Latin_word', 'other']


def classify(tok, token_id):
    if token_id in tok.all_special_ids:
        return 0
    value = tok.decode([token_id], skip_special_tokens=False)
    stripped = value.strip()
    if '\n' in value or '\r' in value:
        return 1
    if value.isspace() or not value:
        return 2
    if stripped.isdecimal():
        return 3
    if stripped and all(unicodedata.category(c)[0] in 'PS' for c in stripped):
        return 4
    if stripped.casefold() in ['yes', 'no']:
        return 5
    if stripped in ['是', '否']:
        return 6
    if stripped and all('\u3400' <= c <= '\u9fff' for c in stripped):
        return 7
    if stripped and all(c.isalpha() and 'LATIN' in unicodedata.name(c, '') for c in stripped):
        return 8
    return 9


def natural_point(row, position, split):
    ids = row['prompt_ids']
    return {'sample_id': row.get('sample_id', 'source_' + rank(row['source_id'])[:20]) + '_content_' + str(position),
        'source_group': row['source_group'], 'source_id': row['source_id'],
        'cohort': row['cohort'], 'family': 'natural_' + row['cohort'], 'language': row['language'],
        'kind': 'natural_content', 'split': split, 'ids': ids[:position+1], 'target': ids[position+1],
        'position': position, 'original_text': row['text'], 'component_ids': row.get('component_ids', []),
        'source_input_ids': ids, 'annotation_scope': 'Authentic next input token at a declared corpus boundary; not model-generated or a synthetic suffix.'}


def control_point(row, tok, split):
    label = tok(row['target'], add_special_tokens=False)['input_ids']
    assert len(label) == 1
    return {**row, 'kind': 'controlled_relation', 'split': split, 'ids': row['prompt_ids'],
            'target_text': row['target'], 'target': label[0], 'cohort': row['family'],
            'position': len(row['prompt_ids']) - 1}


def freeze():
    path = OUT / 'material/protocol.json'
    if path.exists():
        return read(path), gzread(OUT / 'material/rows.json.gz')
    from transformers import AutoTokenizer
    import phase2740_rdc_query_material as old
    import phase2728_rdc_law_material as source
    start = time.monotonic()
    tok = AutoTokenizer.from_pretrained(ROOT / 'models/hf/qwen3-4b', local_files_only=True)
    natural = gzread(OLD / 'material/natural.json.gz')
    # Existing document split is immutable. Each seed will consume ALL2048
    # examples exactly once:128updates x16examples, not an advertised unused pool.
    train = []
    for cohort in ['gum', 'ewt', 'cmrc']:
        selected = old.round_robin([r for r in natural if r['cohort'] == cohort and r['split'] == 'train'], 288)
        assert len(selected) == 288
        for r in selected:
            for p in [len(r['prompt_ids'])//2, 2*len(r['prompt_ids'])//3]:
                train.append(natural_point(r, p, 'train'))
    controls = gzread(BASE / 'material.json.gz')['models']['qwen4']['rows']
    train += [control_point(r, tok, 'train') for r in controls]
    assert len(train) == 2048
    validation = []
    diagnostic = []
    for split, dest in [('validation', validation), ('test', diagnostic)]:
        for cohort in ['gum', 'ewt', 'cmrc']:
            selected = old.round_robin([r for r in natural if r['cohort'] == cohort and r['split'] == split], 64)
            assert len(selected) == 64
            for r in selected:
                dest.append(natural_point(r, 2*len(r['prompt_ids'])//3, split))
    # Exposed2746 confirmation remains a fixed diagnostic after new design.
    previous_confirmation = gzread(BASE / 'phase2746/history_prediction/confirmation/material.json.gz')
    diagnostic += [control_point(r, tok, 'old_exposed_new_wording_diagnostic') for r in previous_confirmation if r['kind'] == 'controlled']
    inventory, inventory_paths = old.prior_inventory()
    inventory += natural + gzread(OLD / 'followup/material.json.gz') + previous_confirmation
    groups = {r['source_group'] for r in inventory}
    used = {tuple(r['prompt_ids']) for r in inventory}
    components = {c for r in inventory for c in r.get('component_ids', [])}
    source_manifest = []
    pools = {}
    for bank in ['gum', 'ewt']:
        pool = []
        for part in ['train', 'dev', 'test']:
            source_path = (source.JOINT if bank == 'gum' else source.BASE) / 'sources' / f'{bank}_{part}.conllu.gz'
            assert source_path.exists(), 'No implicit network acquisition'
            pool += source.natural_candidates(tok, bank, part, source_manifest)
        pools[bank] = pool
    pools['cmrc'] = source.encyclopedia_pool(tok, 'cmrc_train', source_manifest) + source.encyclopedia_pool(tok, 'cmrc_dev', source_manifest)
    remaining = defaultdict(dict)
    for bank, pool in pools.items():
        for r in pool:
            g = r['source_group']
            if g in groups or tuple(r['prompt_ids']) in used or set(r.get('component_ids', [])) & components:
                continue
            if g not in remaining[bank] or rank(r['source_id']) < rank(remaining[bank][g]['source_id']):
                remaining[bank][g] = dict(r, cohort=bank)
    availability = {bank: len(remaining[bank]) for bank in ['gum', 'ewt', 'cmrc']}
    fresh = []
    for bank in ['gum', 'ewt', 'cmrc']:
        # All remaining English documents up to192;192Chinese docs. Count and
        # asymmetry are recorded, not invented to create a balanced claim.
        for g in sorted(remaining[bank], key=lambda x: rank('formation_fresh/' + x))[:192]:
            r = remaining[bank][g]
            fresh.append(natural_point(r, 2*len(r['prompt_ids'])//3, 'prospective_source_confirmation'))
    assert fresh, 'No new source document available'
    splits = [train, validation, diagnostic, fresh]
    group_sets = [{r['source_group'] for r in part} for part in splits]
    for i in range(len(splits)):
        for j in range(i):
            assert not group_sets[i] & group_sets[j], (i, j)
    vocab_size = read(ROOT / 'models/hf/qwen3-4b/config.json')['vocab_size']
    classes = np.array([classify(tok, i) for i in range(vocab_size)], dtype=np.int16)
    rng = np.random.default_rng(2747001)
    groups_for_permutation = defaultdict(list)
    for i, r in enumerate(train):
        r['surface_class'] = int(classes[r['target']])
        groups_for_permutation[(r['kind'], r['cohort'], r['language'], r['surface_class'])].append(i)
    permutation = np.arange(len(train))
    for key, ix in groups_for_permutation.items():
        permutation[ix] = rng.permutation(ix)
    for i, r in enumerate(train):
        r['permuted_target'] = train[int(permutation[i])]['target']
        assert classes[r['target']] == classes[r['permuted_target']]
    # Full training-side corpus prior, not the old576-position prior. Window
    # overlap remains explicit; source-balanced counts are also preserved.
    count = np.zeros(vocab_size, np.int64)
    balanced = np.zeros(vocab_size, np.float64)
    windows = [r for r in natural if r['split'] == 'train']
    by_doc = Counter(r['source_group'] for r in windows)
    for r in windows:
        c = np.bincount(r['prompt_ids'], minlength=vocab_size)
        count += c
        balanced += c / by_doc[r['source_group']]
    data = {'train': train, 'validation': validation, 'diagnostic': diagnostic, 'fresh': fresh}
    compressed(OUT / 'material/rows.json.gz', data)
    compressed(OUT / 'material/excluded_inventory.json.gz',
        [{k: r.get(k) for k in ['sample_id', 'source_id', 'source_group', 'prompt_ids', 'component_ids']} for r in inventory])
    prior_receipt = commit_array('material', 'vocabulary', classes=classes, prior_count=count,
                                prior_source_balanced_count=balanced, permutation=permutation)
    p = {'timestamp': stamp(), 'source': snapshot(__file__), 'phase': 2747,
        'material_sha256': sha(OUT / 'material/rows.json.gz'),
        'counts': {k: len(v) for k, v in data.items()},
        'source_groups': {k: len({r['source_group'] for r in v}) for k, v in data.items()},
        'train_composition': dict(Counter(r['family'] for r in train)),
        'fresh_source_availability': availability, 'fresh_selected': dict(Counter(r['cohort'] for r in fresh)),
        'excluded_source_groups': len(groups), 'source_manifest': source_manifest,
        'source_code_snapshots': [snapshot(old.__file__), snapshot(source.__file__)],
        'surface_classes': CLASS_NAMES, 'vocabulary_receipt': prior_receipt,
        'label_permutation_changed_fraction': float(np.mean([r['target'] != r['permuted_target'] for r in train])),
        'permutation_scope': 'Within kind/cohort/language/coarse surface class; exact empirical target and class marginals preserved. Same-label coincidences retained and counted.',
        'supervision_conditions': ['true_token', 'within_surface_class_permuted_token', 'surface_class_mass'],
        'format_boundary': 'Finite decoded-token surface classes, not pure format or proven semantics-free supervision. Binary answer identities pooled by language; all classes keep full vocabulary denominators.',
        'nexttoken_alignment': 'Natural gold is exactly ids[position+1], supplied only to training/scoring. Controlled gold is one-token answer to recorded full chat prompt.',
        'train_consumption_plan': '128updates x16gradient-accumulated examples, each2048-example training permutation consumed exactly once per seed/condition.',
        'prior': {'natural_train_windows': len(windows), 'source_groups': len(by_doc),
                  'token_occurrences': int(count.sum()), 'overlap': 'Within-source windows overlap; counts are occurrences, not independent tokens. Both raw and document-weighted fullvocabulary counts saved.'},
        'selection': 'All train data and prior use original train documents. Validation selects future calibration only. Exposed2746controlled observations are diagnostic, not prospective confirmation of a new design.',
        'unseen_scope': 'Fresh source groups/exact inputs/components absent from explicit inventory, not asserted absent from pretraining or every historical Phase. No fresh English source counts fabricated.',
        'seconds': time.monotonic() - start}
    immutable(path, p)
    ledger('phase2747_material', p['seconds'])
    return p, data


if __name__ == '__main__':
    p, data = freeze()
    print('FORMATION2747_MATERIAL', p['counts'], p['fresh_source_availability'], p['prior'], flush=True)
