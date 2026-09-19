"""Independent full selected-material, source-row and token-boundary verification."""
from collections import Counter, defaultdict
from transformers import AutoTokenizer
from rdc_construction_common import *
from rdc_question_inputs import qualify_context
from rdc_question_material import exact_answer_signature, text_id
from phase2748_rdc_context_inventory import ranked, norm


OUT = BASE/'phase2748'


def main():
    start = time.monotonic()
    manifest = read(OUT/'material/manifest.json')
    rows = gzread(ROOT/manifest['rows']['path'])
    groups = gzread(ROOT/manifest['context_groups']['path'])
    inventory = read(OUT/'material/context_inventory.json')
    nodes = {r['context_id']: r for r in gzread(ROOT/inventory['nodes']['path'])}
    sources = {}
    for src in inventory['sources']:
        for row in gzread(ROOT/src['rows']['path']):
            sources[row['question_id']] = row
    by_context = defaultdict(list)
    for row in rows:
        original = sources[row['question_id']]
        assert all(row[k] == v for k, v in original.items())
        assert text_id(row['passage']) == row['context_text_sha256']
        by_context[row['group_id']].append(row)
    assert len({r['question_id'] for r in rows}) == len(rows) == 1600
    assert len(groups) == len(by_context) == 400
    assert len({g['near_context_component'] for g in groups}) == 400
    known = [g['article_id'] for g in groups if g['article_id']]
    assert len(set(known)) == len(known) == 200
    for group in groups:
        rr = by_context[group['group_id']]
        node = nodes[group['context_id']]
        assert node['eligible_before_tokenization']
        assert len(rr) == 4 and {r['split'] for r in rr} == {group['split']}
        assert [r['question_id'] for r in rr] == node['four_initial_question_ids']
        assert len({norm(r['question']) for r in rr}) == 4
        signatures = [{exact_answer_signature(a) for a in r['answer_annotations']} for r in rr]
        assert all(not signatures[i] & signatures[j] for i in range(4) for j in range(i))
    for cohort in ['quoref', 'drop']:
        ordered = sorted((g for g in groups if g['cohort'] == cohort), key=lambda r: ranked('split/'+r['context_id']))
        expected = ['train']*96+['validation']*24+['diagnostic']*48+['confirmation']*32
        assert [g['split'] for g in ordered] == expected
    checks = []
    for key, ref in manifest['model_token_files'].items():
        assert sha(ROOT/ref['path']) == ref['sha256']
        tokenrows = gzread(ROOT/ref['path'])
        assert [r['question_id'] for r in tokenrows] == [r['question_id'] for r in rows]
        by_id = {r['question_id']: r for r in tokenrows}
        meta = manifest['tokenizer_metadata'][key]
        assert all(sha(ROOT/f['path']) == f['sha256'] for f in meta['files'])
        tok = AutoTokenizer.from_pretrained(ROOT/'models/hf'/MODELS[key], local_files_only=True, use_fast=True, trust_remote_code=True)
        for group in groups:
            actual = qualify_context(tok, by_context[group['group_id']], meta['native_stop_ids'])
            for new in actual:
                old = by_id[new['question_id']]
                assert all(old[k] == v for k, v in new.items())
                assert len(old['input_ids']) <= 1024 and len(old['teacher_ids_including_EOS']) <= 64
        checks.append({'model': key, 'full_native_prompt_teacher_retokenizations': len(tokenrows),
            'every_field_equal': True, 'all_group_prefixes_equal': True})
        print('NATURAL_MATERIAL_AUDIT', key, len(tokenrows), flush=True)
    original = read(BASE/'protocol.json')
    raw_memo = MEMO.read_bytes()
    assert hashlib.sha256(raw_memo[:original['memo_original_bytes']]).hexdigest() == original['memo_original_sha256']
    value = {'timestamp': stamp(), 'source': snapshot(__file__), 'all_passed': True,
        'material_manifest_sha256': sha(OUT/'material/manifest.json'), 'context_groups': len(groups),
        'source_questions_unchanged': len(rows), 'known_articles_unique': len(known),
        'detected_near_components_unique': len(groups), 'tokenizer_checks': checks,
        'memo_original_prefix_unchanged': True, 'confirmation_native_unobserved': True,
        'scope': 'Exhaustive selected-field/source-byte identity and tokenizer replay; near-duplicate retrieval remains approximate and no pretraining/article exclusion is inferred for DROP.',
        'seconds': time.monotonic()-start}
    save(OUT/'material'/('audit_'+str(time.time_ns())+'.json'), value)
    save(OUT/'material/audit_current.json', value)
    print('NATURAL_MATERIAL_AUDIT_PASSED', flush=True)


if __name__ == '__main__':
    main()
