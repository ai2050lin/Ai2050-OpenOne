"""Select/freeze authentic material and all native token IDs before observation."""
from collections import Counter, defaultdict
import transformers
from transformers import AutoTokenizer
from rdc_construction_common import *
from rdc_question_inputs import INSTRUCTION, qualify_context
from rdc_question_material import exact_answer_signature, text_id
from phase2748_rdc_context_inventory import ranked, norm

OUT = BASE/'phase2748'


def freeze_rules():
    path = OUT/'material/selection_rules.json'
    if path.exists():
        return read(path)
    value = {'timestamp': stamp(), 'source': snapshot(__file__),
        'input_algorithm': snapshot(Path(__file__).with_name('rdc_question_inputs.py')),
        'protocol_sha256': sha(OUT/'protocol.json'),
        'context_inventory_sha256': sha(OUT/'material/context_inventory.json'),
        'status': 'frozen_before_tokenizer_selection_and_native_observation',
        'cohort_order': ['quoref', 'drop'], 'model_order': ['qwen4', 'qwen14', 'glm4'],
        'instruction': INSTRUCTION,
        'candidate_order': 'Stable SHA256 natural_questions2748/context_id; only source-eligible nodes.',
        'question_rule': 'Use exactly the four_initial_question_ids already fixed by context_inventory. If any model exceeds caps, reject entire context; no alternative questions retried.',
        'identity_rule': 'At most one accepted context per known article and per detected near-context connected component across both cohorts.',
        'assignment': 'After first200eligible contexts per cohort, order SHA256 natural_questions2748/split/context_id and assign consecutive train96, validation24, diagnostic48, confirmation32.',
        'requested_contexts_per_cohort': {'train': 96, 'validation': 24, 'diagnostic': 48, 'confirmation': 32},
        'prompt_token_cap': 1024, 'complete_teacher_token_cap_including_EOS': 64,
        'EOS_rule': 'Native tokenizer.eos_token_id must be in original generation_config.eos_token_id. All native stop IDs terminate free generation; no answer or EOS is forced.',
        'teacher_rule': 'First original complete annotation, all required spans in original order, json.dumps(ensure_ascii=False); separately tokenized and native tokenizer EOS appended.',
        'split_scope': 'Original train/dev may be mixed in new whole-context experimental splits; this is not an official benchmark submission. Quoref known-article separated; DROP article identity unavailable.',
        'confirmation': 'Only source and tokenizer metadata processed here; no native target, learned fit, model output, or behavioral selection.',
        'length_selection_bias': 'Longer passages or answers and contexts with a long member of the preselected four are excluded; preserve all inspected rejection records.'}
    immutable(path, value)
    return value


def main():
    target = OUT/'material/manifest.json'
    if target.exists():
        value = read(target)
        for item in [value['rows'], value['context_groups'], *value['model_token_files'].values()]:
            assert sha(ROOT/item['path']) == item['sha256']
        return value
    start = time.monotonic()
    guard()
    rules = freeze_rules()
    inventory = read(OUT/'material/context_inventory.json')
    assert sha(ROOT/inventory['nodes']['path']) == inventory['nodes']['sha256']
    nodes = [n for n in gzread(ROOT/inventory['nodes']['path']) if n['eligible_before_tokenization']]
    required_ids = {q for n in nodes for q in n['four_initial_question_ids']}
    source_rows = {}
    for source in inventory['sources']:
        assert sha(ROOT/source['rows']['path']) == source['rows']['sha256']
        for row in gzread(ROOT/source['rows']['path']):
            if row['question_id'] in required_ids:
                assert row['question_id'] not in source_rows
                source_rows[row['question_id']] = row
    assert required_ids == set(source_rows)
    tokenizers, metadata, stops = {}, {}, {}
    for key in rules['model_order']:
        folder = ROOT/'models/hf'/MODELS[key]
        tok = AutoTokenizer.from_pretrained(folder, local_files_only=True, use_fast=True, trust_remote_code=True)
        assert tok.is_fast, ('Offset-aligned fast tokenizer required', key)
        generation = read(folder/'generation_config.json')
        stop = generation['eos_token_id']
        stops[key] = stop if isinstance(stop, list) else [stop]
        assert tok.eos_token_id in stops[key]
        tokenizers[key] = tok
        files = [folder/name for name in ['config.json', 'generation_config.json', 'tokenizer_config.json', 'tokenizer.json']]
        assert all(f.is_file() for f in files)
        metadata[key] = {'tokenizer_class': type(tok).__name__, 'is_fast': tok.is_fast,
            'transformers_version': transformers.__version__, 'vocab_length': len(tok),
            'native_stop_ids': stops[key], 'native_stop_token_strings': tok.convert_ids_to_tokens(stops[key]),
            'teacher_EOS_id': tok.eos_token_id, 'teacher_EOS_string': tok.eos_token,
            'chat_template': tok.chat_template, 'files': [{'path': str(f.relative_to(ROOT)), 'sha256': sha(f)} for f in files]}
    accepted, rejected, used_articles, used_components = [], [], set(), set()
    encoded_by_model = {k: {} for k in tokenizers}
    for cohort in rules['cohort_order']:
        count = 0
        for node in sorted((n for n in nodes if n['cohort'] == cohort), key=lambda n: ranked(n['context_id'])):
            if count == 200:
                break
            base = {'context_id': node['context_id'], 'cohort': cohort}
            if node['article_id'] and node['article_id'] in used_articles:
                rejected.append({**base, 'reason': 'known_article_already_selected'})
                continue
            if node['near_context_component'] in used_components:
                rejected.append({**base, 'reason': 'detected_near_component_already_selected'})
                continue
            rows = [source_rows[q] for q in node['four_initial_question_ids']]
            assert all(row['context_id'] == node['context_id'] for row in rows)
            assert len({norm(row['question']) for row in rows}) == 4
            signatures = [{exact_answer_signature(a) for a in row['answer_annotations']} for row in rows]
            assert all(not signatures[i] & signatures[j] for i in range(4) for j in range(i))
            encoded = {k: qualify_context(tok, rows, stops[k]) for k, tok in tokenizers.items()}
            lengths = {k: {'prompt_tokens': [len(r['input_ids']) for r in batch],
                'teacher_tokens_including_EOS': [len(r['teacher_ids_including_EOS']) for r in batch]}
                for k, batch in encoded.items()}
            if any(max(v['prompt_tokens']) > rules['prompt_token_cap'] or
                   max(v['teacher_tokens_including_EOS']) > rules['complete_teacher_token_cap_including_EOS']
                   for v in lengths.values()):
                rejected.append({**base, 'reason': 'actual_token_cap', 'all_model_lengths': lengths})
                continue
            accepted.append({**node, 'all_model_lengths': lengths})
            if node['article_id']:
                used_articles.add(node['article_id'])
            used_components.add(node['near_context_component'])
            for key, batch in encoded.items():
                for item in batch:
                    encoded_by_model[key][item['question_id']] = item
            count += 1
            if count % 25 == 0:
                print('NATURAL_TOKEN_QUALIFIED', cohort, count, 'excluded', len(rejected), flush=True)
        assert count == 200, ('Insufficient frozen-rule material; revise before any model run', cohort, count)
    groups, rows = [], []
    for cohort in rules['cohort_order']:
        ordered = sorted((n for n in accepted if n['cohort'] == cohort), key=lambda n: ranked('split/'+n['context_id']))
        cursor = 0
        for split, count in rules['requested_contexts_per_cohort'].items():
            for node in ordered[cursor:cursor+count]:
                group = {**node, 'split': split, 'group_id': node['context_id']}
                groups.append(group)
                for index, qid in enumerate(node['four_initial_question_ids']):
                    row = {**source_rows[qid], 'split': split, 'group_id': node['context_id'],
                           'within_context_index': index, 'near_context_component': node['near_context_component']}
                    rows.append(row)
                    for key in encoded_by_model:
                        encoded_by_model[key][qid].update({'split': split, 'group_id': node['context_id'], 'within_context_index': index})
            cursor += count
        assert cursor == 200
    assert len(groups) == 400 and len(rows) == 1600
    def register(path, value):
        compressed(path, value)
        return {'path': path.relative_to(ROOT).as_posix(), 'sha256': sha(path), 'bytes': path.stat().st_size}
    rows_file = register(OUT/'material/rows.json.gz', rows)
    groups_file = register(OUT/'material/context_groups.json.gz', groups)
    rejects_file = register(OUT/'material/selection_rejections.json.gz', rejected)
    model_files = {k: register(OUT/'material'/(k+'_tokens.json.gz'), [data[row['question_id']] for row in rows])
                   for k, data in encoded_by_model.items()}
    value = {'timestamp': stamp(), 'source': snapshot(__file__),
        'input_algorithm': snapshot(Path(__file__).with_name('rdc_question_inputs.py')),
        'status': 'actual_material_and_tokenization_frozen_no_model_observation',
        'selection_rules_sha256': sha(OUT/'material/selection_rules.json'),
        'context_inventory_sha256': sha(OUT/'material/context_inventory.json'),
        'all_checks_passed': True, 'contexts': len(groups), 'questions': len(rows),
        'rows': rows_file, 'context_groups': groups_file, 'rejections': rejects_file,
        'model_token_files': model_files, 'tokenizer_metadata': metadata,
        'counts': [{'cohort': cohort, 'split': split,
            'contexts': sum(g['cohort'] == cohort and g['split'] == split for g in groups),
            'questions': sum(r['cohort'] == cohort and r['split'] == split for r in rows)}
            for cohort in rules['cohort_order'] for split in rules['requested_contexts_per_cohort']],
        'original_source_splits': dict(Counter(r['cohort']+'/'+r['source_split'] for r in rows)),
        'answer_types': dict(Counter(r['cohort']+'/'+r['answer_annotations'][0]['type'] for r in rows)),
        'rejection_counts': dict(Counter(r['cohort']+'/'+r['reason'] for r in rejected)),
        'actual_token_lengths': {k: {name: {'min': int(min(values)), 'median': float(np.median(values)), 'max': int(max(values))}
            for name, values in {'prompt': [len(v['input_ids']) for v in data.values()],
                                 'teacher_with_EOS': [len(v['teacher_ids_including_EOS']) for v in data.values()]}.items()}
            for k, data in encoded_by_model.items()},
        'selection_inputs_exclude_native_behavior': True, 'confirmation_native_unobserved': True,
        'seconds': time.monotonic()-start}
    immutable(target, value)
    print('NATURAL_MATERIAL_FROZEN', len(groups), len(rows), value['actual_token_lengths'], flush=True)
    return value


if __name__ == '__main__':
    main()
