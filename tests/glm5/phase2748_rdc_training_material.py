"""Freeze every complete-answer draw, target frequency and exact surface partition."""
from collections import defaultdict
from transformers import AutoTokenizer
from rdc_question_common import *
from phase2747_rdc_material import classify, CLASS_NAMES


def freeze():
    path = OUT/'training/material_manifest.json'
    if path.exists():
        return read(path)
    start = time.monotonic()
    contract, manifest, rows, _ = material('qwen4')
    rows = [r for r in rows if r['split'] == 'train']
    assert len(rows) == 768
    by_group = defaultdict(list)
    for row in rows:
        by_group[row['group_id']].append(row)
    permuted_source = {}
    for group, rr in by_group.items():
        rr.sort(key=lambda r: r['within_context_index'])
        assert [r['within_context_index'] for r in rr] == [0, 1, 2, 3]
        for i, row in enumerate(rr):
            permuted_source[row['question_id']] = rr[(i+1) % 4]
    old = read(BASE/'phase2747/material/protocol.json')
    reference = old['vocabulary_receipt']
    assert sha(ROOT/reference['field_path']) == reference['field_sha256']
    with np.load(ROOT/reference['field_path']) as z:
        classes = z['classes'].copy()
    tok = AutoTokenizer.from_pretrained(ROOT/'models/hf/qwen3-4b', local_files_only=True, use_fast=True, trust_remote_code=True)
    actual = np.array([classify(tok, i) for i in range(len(classes))], dtype=classes.dtype)
    assert np.array_equal(actual, classes) and len(classes) == read(ROOT/'models/hf/qwen3-4b/config.json')['vocab_size']
    configs, draws = [], []
    frequencies = {}
    for seed in contract['learning']['seeds']:
        order = np.random.default_rng(seed).permutation(len(rows))
        for condition in contract['learning']['conditions']:
            name = condition+'_'+str(seed)
            raw_count = np.zeros(len(classes), dtype=np.int64)
            example_normalized = np.zeros(len(classes), dtype=np.float64)
            per_run = []
            for index, selected in enumerate(order):
                row = rows[int(selected)]
                teacher = permuted_source[row['question_id']] if condition == 'within_context_permuted_complete_answer' else row
                target_ids = teacher['tokens']['teacher_ids_including_EOS']
                count = np.bincount(target_ids, minlength=len(classes))
                raw_count += count
                example_normalized += count/len(target_ids)
                per_run.append({'run': name, 'seed': seed, 'condition': condition,
                    'draw': index, 'step': index//8+1, 'within_batch': index % 8,
                    'question_id': row['question_id'], 'group_id': row['group_id'], 'cohort': row['cohort'],
                    'teacher_from_question_id': teacher['question_id'], 'teacher_text': teacher['tokens']['teacher_text'],
                    'teacher_ids_including_EOS': target_ids, 'surface_class_ids': classes[target_ids].tolist(),
                    'actual_prompt_tokens': len(row['tokens']['input_ids']),
                    'teacher_token_count_including_EOS': len(target_ids),
                    'effective_input_token_count': len(row['tokens']['input_ids'])+len(target_ids)-1})
            assert len(per_run) == 768 and len({r['question_id'] for r in per_run}) == 768
            frequencies[name+'_raw'] = raw_count
            frequencies[name+'_example_normalized'] = example_normalized
            draws += per_run
            configs.append({'run': name, 'questions': len(per_run), 'steps': 96,
                'total_teacher_tokens': int(raw_count.sum()),
                'teacher_token_counts_by_cohort': {c: sum(r['teacher_token_count_including_EOS'] for r in per_run if r['cohort'] == c) for c in ['quoref', 'drop']},
                'maximum_effective_input_tokens': max(r['effective_input_token_count'] for r in per_run)})
    first = contract['learning']['conditions'][0]+'_'+str(contract['learning']['seeds'][0])
    for run in configs:
        assert np.array_equal(frequencies[run['run']+'_raw'], frequencies[first+'_raw'])
        assert np.allclose(frequencies[run['run']+'_example_normalized'], frequencies[first+'_example_normalized'], rtol=0, atol=1e-10)
    fields = commit_arrays(Path('training/material'), 'vocabulary_and_exposure', {'classes': classes, **frequencies})
    drawpath = OUT/'training/draws.json.gz'
    compressed(drawpath, draws)
    value = {'timestamp': stamp(), 'source': snapshot(__file__), 'all_passed': True,
        'effective_contract_sha256': sha(OUT/'effective_experiment_contract.json'),
        'material_manifest_sha256': sha(OUT/'material/manifest.json'),
        'draws': {'path': drawpath.relative_to(ROOT).as_posix(), 'sha256': sha(drawpath)},
        'partition_and_exposure': fields, 'original_partition_receipt': reference,
        'partition_algorithm': snapshot(Path(__file__).with_name('phase2747_rdc_material.py')),
        'surface_class_names': CLASS_NAMES, 'every_native_vocabulary_class_recomputed_equal': len(classes),
        'run_inventory': configs, 'total_planned_draws': len(draws),
        'all_conditions_and_seeds_whole_teacher_token_multisets_equal': True,
        'all_example_normalized_target_frequencies_equal_to_roundoff': True,
        'execution_status': 'Draws/material frozen; no optimizer update executed by this script.',
        'scope': 'Whole answers including EOS are permuted only within the same context. True and surface-class conditions keep true teacher histories; the class loss is not content-free. Equal marginal targets do not imply equal conditional teacher states or gradients.',
        'seconds': time.monotonic()-start}
    immutable(path, value)
    print('NATURAL_TRAINING_EXPOSURES_FROZEN', len(draws), configs[0]['total_teacher_tokens'], flush=True)
    return value


if __name__ == '__main__':
    freeze()
