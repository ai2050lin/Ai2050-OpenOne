"""Source-only history exclusion and explicitly approximate near-duplicate screening."""
import re
from collections import Counter, defaultdict
from rdc_construction_common import *
from rdc_question_material import canonical_title, exact_answer_signature, text_id

OUT = BASE/'phase2748'
HISTORY_PATHS = [
    'rdc_operator_atlas_20260911/material.json.gz',
    'rdc_operator_atlas_20260911/qa_material.json.gz',
    'rdc_operator_atlas_20260911/qa_balanced_material.json.gz',
    'rdc_operator_atlas_20260911/qa_multihop_material.json.gz',
    'rdc_law_campaign_20260911/material.json.gz',
    'rdc_law_campaign_20260911/confirmation_material.json.gz',
    'rdc_binding_campaign_20260912/natural_discovery.json.gz',
    'rdc_binding_campaign_20260912/natural_confirmation.json.gz',
    'rdc_binding_campaign_20260912/signed_source/identity_recovery/resolved_material.json.gz',
    'rdc_update_campaign_20260913/natural_material.json.gz',
    'rdc_update_campaign_20260913/fresh_graph/material.json.gz',
    'rdc_query_campaign_20260913/material/natural.json.gz',
    'rdc_query_campaign_20260913/followup/material.json.gz',
    'rdc_query_construction_20260913/phase2746/history_prediction/confirmation/material.json.gz',
    'rdc_query_construction_20260913/phase2747/material/rows.json.gz',
]
TEXT_KEYS = {'text', 'full_context', 'context', 'passage', 'original_text', 'actual_input'}


def norm(text):
    return ' '.join(text.casefold().split())


def ranked(value):
    return text_id('natural_questions2748/' + str(value))


def four_questions(rows):
    """Stable source-ID ordering; no native behavior and no random retry."""
    unique = {}
    for row in sorted(rows, key=lambda r: ranked(r['question_id'])):
        unique.setdefault(norm(row['question']), row)
    ordered = list(unique.values())
    signatures = [{exact_answer_signature(a) for a in r['answer_annotations']} for r in ordered]
    def search(start, chosen, used):
        if len(chosen) == 4:
            return [ordered[i] for i in chosen]
        if len(ordered)-start < 4-len(chosen):
            return None
        for i in range(start, len(ordered)):
            if signatures[i] & used:
                continue
            result = search(i+1, chosen+[i], used | signatures[i])
            if result is not None:
                return result
        return None
    return search(0, [], set())


def main():
    target = OUT/'material/context_inventory.json'
    if target.exists():
        assert read(target)['all_source_checks_passed']
        return read(target)
    assert read(OUT/'protocol.json')['status'] == 'frozen_before_model_observation'
    start = time.monotonic()
    old_texts, old_titles = {}, set()
    manifests = []
    def walk(value):
        if isinstance(value, dict):
            title = value.get('title')
            if isinstance(title, str) and title.strip():
                old_titles.add(canonical_title(title))
            if isinstance(title, list):
                old_titles.update(canonical_title(t) for t in title if isinstance(t, str) and t.strip())
            for key, child in value.items():
                if key in TEXT_KEYS and isinstance(child, str) and len(child) >= 80:
                    normalized = norm(child)
                    old_texts.setdefault(text_id(normalized), normalized)
                elif isinstance(child, dict) or (isinstance(child, list) and child and isinstance(child[0], (dict, list))):
                    walk(child)
        elif isinstance(value, list):
            for child in value:
                if isinstance(child, (dict, list)):
                    walk(child)
    for name in HISTORY_PATHS:
        path = RESULT/name
        assert path.exists(), ('Declared historical material missing; do not silently narrow exclusion', path)
        walk(gzread(path))
        manifests.append({'path': path.relative_to(ROOT).as_posix(), 'sha256': sha(path), 'scope': 'Registered historical material'})
    for name in ['squad_train', 'squad_dev']:
        path = RESULT/'rdc_operator_atlas_20260911/sources'/(name+'.json.gz')
        walk(gzread(path))
        manifests.append({'path': path.relative_to(ROOT).as_posix(), 'sha256': sha(path),
            'scope': 'Conservative exclusion of complete cached SQuAD source. Prior Phase2728 recorded all490titles previously exposed; not every source paragraph asserted previously run.'})
    rows_by_context = defaultdict(list)
    source_manifests = []
    for cohort in ['quoref', 'drop']:
        source = read(OUT/'sources'/(cohort+'_inventory.json'))
        path = ROOT/source['rows']['path']
        assert sha(path) == source['rows']['sha256']
        for row in gzread(path):
            rows_by_context[row['context_id']].append(row)
        source_manifests.append({'cohort': cohort, 'rows': source['rows']})
    nodes = []
    passages = {}
    for context_id, rows in sorted(rows_by_context.items()):
        first = rows[0]
        assert all(r['passage'] == first['passage'] and r['cohort'] == first['cohort'] for r in rows)
        article_keys = {canonical_title(r['title']) for r in rows if r.get('title', '').strip()}
        for row in rows:
            if row['article_id']:
                article_keys.add(row['article_id'].split(':', 1)[1])
        normalized = norm(first['passage'])
        passage_key = text_id(normalized)
        passages.setdefault(passage_key, normalized)
        selected = four_questions(rows)
        reasons = []
        if article_keys & old_titles:
            reasons.append('registered_historical_title_overlap')
        if passage_key in old_texts:
            reasons.append('registered_historical_normalized_text_equal')
        if selected is None:
            reasons.append('fewer_than_four_distinct_questions_with_nonoverlapping_complete_answer_signatures')
        nodes.append({'context_id': context_id, 'cohort': first['cohort'], 'context_text_sha256': first['context_text_sha256'],
            'normalized_text_sha256': passage_key, 'article_keys': sorted(article_keys),
            'article_id': first['article_id'], 'article_identity_scope': first['article_identity_scope'],
            'question_count': len(rows), 'four_initial_question_ids': [r['question_id'] for r in selected] if selected else [],
            'exclusion_reasons': reasons, 'original_splits': sorted({r['source_split'] for r in rows})})
    print('NATURAL_CONTEXT_EXACT_INVENTORY', len(nodes), len(old_texts), len(old_titles), flush=True)
    # This is source-text screening, NOT a compression/selection of hidden coordinates.
    # Sixteen 4-row bands retrieve candidates; exact set intersections confirm edges.
    # Candidate retrieval is approximate and is explicitly not an exhaustive no-near-duplicate proof.
    rng = np.random.default_rng(2748003)
    a = rng.integers(1, 2**64-1, 64, dtype=np.uint64) | np.uint64(1)
    b = rng.integers(0, 2**64-1, 64, dtype=np.uint64)
    buckets = defaultdict(list)
    signatures = {}
    shingle_sets = {}
    near_edges = []
    historical_keys = set(old_texts)
    all_texts = {**old_texts, **passages}
    # Historical entries are indexed before new entries. Only pairs involving a new
    # context are tested; there is no need to rediscover every old-old pair.
    order = sorted(old_texts) + sorted(set(passages)-historical_keys)
    for number, key in enumerate(order):
        words = re.findall(r'\w+|[^\w\s]', all_texts[key], re.UNICODE)
        shingles = {int.from_bytes(hashlib.sha256('\x1f'.join(words[i:i+5]).encode()).digest()[:8], 'little')
                    for i in range(max(0, len(words)-4))}
        if len(shingles) < 32:
            continue
        values = np.fromiter(shingles, dtype=np.uint64)
        with np.errstate(over='ignore'):
            signature = (values[:, None]*a[None, :]+b[None, :]).min(axis=0)
        bands = [(j, signature[4*j:4*j+4].tobytes()) for j in range(16)]
        candidates = set()
        if key in passages:
            for band in bands:
                candidates.update(buckets[band])
        for other in sorted(candidates):
            other_set = shingle_sets[other]
            common = len(shingles & other_set)
            jaccard = common/(len(shingles)+len(other_set)-common)
            containment = common/min(len(shingles), len(other_set))
            if jaccard >= .8 or containment >= .9:
                near_edges.append({'left_normalized_sha256': key, 'right_normalized_sha256': other,
                    'left_shingles': len(shingles), 'right_shingles': len(other_set), 'intersection': common,
                    'jaccard': jaccard, 'shorter_containment': containment,
                    'historical_endpoint': other in historical_keys or key in historical_keys})
        shingle_sets[key] = shingles
        signatures[key] = signature.tolist()
        for band in bands:
            buckets[band].append(key)
        if number % 2000 == 0:
            print('NATURAL_CONTEXT_NEAR_SCREEN', number, len(order), len(near_edges), flush=True)
    parent = {key: key for key in all_texts}
    def root(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key
    for edge in near_edges:
        left, right = root(edge['left_normalized_sha256']), root(edge['right_normalized_sha256'])
        parent[max(left, right)] = min(left, right)
    old_components = {root(key) for key in historical_keys}
    for node in nodes:
        node['near_context_component'] = root(node['normalized_text_sha256'])
        if node['near_context_component'] in old_components and 'registered_historical_normalized_text_equal' not in node['exclusion_reasons']:
            node['exclusion_reasons'].append('detected_near_context_component_has_historical_material')
        node['eligible_before_tokenization'] = not node['exclusion_reasons']
    inventory_path = OUT/'material/context_nodes.json.gz'
    compressed(inventory_path, nodes)
    screening_path = OUT/'material/near_context_screen.json.gz'
    compressed(screening_path, {'edges': near_edges, 'minhash_signatures': signatures,
        'hash_a': a.tolist(), 'hash_b': b.tolist(), 'historical_normalized_sha256': sorted(historical_keys),
        'historical_title_keys': sorted(old_titles)})
    counts = []
    for cohort in ['quoref', 'drop']:
        selected = [r for r in nodes if r['cohort'] == cohort]
        eligible = [r for r in selected if r['eligible_before_tokenization']]
        counts.append({'cohort': cohort, 'original_exact_contexts': len(selected),
            'eligible_contexts_before_tokenization': len(eligible),
            'eligible_known_articles': len({r['article_id'] for r in eligible if r['article_id']}),
            'exclusion_counts_nonexclusive': dict(Counter(reason for r in selected for reason in r['exclusion_reasons']))})
    value = {'timestamp': stamp(), 'source': snapshot(__file__), 'all_source_checks_passed': True,
        'protocol_sha256': sha(OUT/'protocol.json'), 'historical_material': manifests,
        'sources': source_manifests, 'counts': counts,
        'nodes': {'path': inventory_path.relative_to(ROOT).as_posix(), 'sha256': sha(inventory_path)},
        'near_screen': {'path': screening_path.relative_to(ROOT).as_posix(), 'sha256': sha(screening_path),
            'method': '64deterministic uint64 minhashes of all5-token word/punctuation shingles,16bands x4rows; candidate edges checked by exact Jaccard>=.8 or shorter-set containment>=.9. At least32unique shingles.',
            'limitation': 'Approximate candidate retrieval, not exhaustive absence of every near duplicate or contained historic prefix. Detected connected components are kept together; any old endpoint excludes the full detected component.'},
        'seconds': time.monotonic()-start,
        'scope': 'Source-only qualification, not native language results. DROP has no verified article identity. Selected answers have different normalized source annotations, not a proven semantic contrast or necessary reasoning type.'}
    immutable(target, value)
    print('NATURAL_CONTEXT_INVENTORY_DONE', counts, flush=True)
    return value


if __name__ == '__main__':
    main()
