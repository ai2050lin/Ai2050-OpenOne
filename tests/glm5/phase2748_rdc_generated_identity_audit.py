"""Independent stored-identity/score audit of real learned and own-history runs.

No decoder, unembedding, fitting, parameter update or new semantic scoring.
"""
import argparse
from collections import Counter
from transformers import AutoTokenizer
from rdc_question_common import *
from rdc_question_material import conservative_complete_answer
from phase2748_rdc_identity_audit import Fields


BASE_HISTORY = {'generated_ids', 'positions', 'statistics'}
NATIVE_HISTORY = {'postnorm_BF16', 'H12_last_BF16', 'native_source_read_BF16', 'all_hidden_BF16'}
PREDICTED_HISTORY = {'H12_last_BF16', 'native_source_read_BF16',
                     'predicted_postnorm_FP64', 'predicted_postnorm_BF16'}


def check_history(row, record, arrays, tokenizer, maximum):
    """Check actually stored IDs, text, stop/cap, positions and original score."""
    assert all(record[k] == row[k] for k in ['question_id', 'group_id', 'cohort', 'split'])
    ids = record['generated_ids']; stops = set(row['tokens']['native_stop_ids'])
    assert 0 < len(ids) <= maximum and record['maximum_new_tokens'] == maximum
    assert not any(x in stops for x in ids[:-1])
    eos = ids[-1] in stops
    assert record['native_EOS'] == eos
    assert record['censored'] == (not eos and len(ids) == maximum)
    assert arrays['generated_ids'].dtype == np.int64 and arrays['generated_ids'].tolist() == ids
    assert arrays['positions'].dtype == np.int64
    assert np.array_equal(arrays['positions'], np.arange(len(ids)) + len(row['tokens']['input_ids']) - 1)
    assert arrays['statistics'].shape == (len(ids), 2) and arrays['statistics'].dtype == np.float64
    assert np.isfinite(arrays['statistics']).all()
    assert np.all(arrays['statistics'][:, 0] >= 0) and np.all(arrays['statistics'][:, 1] <= 0)
    text = tokenizer.decode(ids, skip_special_tokens=True)
    assert text == record['generated_text']
    assert conservative_complete_answer(text, row['answer_annotations'], eos) == record['score']
    return len(ids)


def check_teacher(row, record, arrays, first, history):
    ids = row['tokens']['teacher_ids_including_EOS']
    assert record['question_id'] == row['question_id']
    assert record['teacher_text'] == row['tokens']['teacher_text']
    assert record['teacher_ids_including_EOS'] == ids == arrays['teacher_ids'].tolist()
    assert record['tokens'] == len(ids) and ids[-1] in row['tokens']['native_stop_ids']
    assert set(arrays) == {'teacher_ids', 'NLL', 'argmax', 'postnorm_BF16', 'entropy_FP64'}
    assert arrays['teacher_ids'].dtype == arrays['argmax'].dtype == np.int64
    for name in ['NLL', 'entropy_FP64']:
        assert arrays[name].shape == (len(ids),) and arrays[name].dtype == np.float64
        assert np.isfinite(arrays[name]).all() and np.all(arrays[name] >= 0)
    assert arrays['argmax'].shape == (len(ids),)
    assert arrays['postnorm_BF16'].shape == (len(ids), len(first['postnorm_BF16']))
    assert arrays['postnorm_BF16'].dtype == np.uint16
    assert np.array_equal(arrays['postnorm_BF16'][0], first['postnorm_BF16'])
    assert arrays['argmax'][0] == history['generated_ids'][0]
    assert arrays['entropy_FP64'][0] == history['statistics'][0, 0]
    assert record['mean_token_NLL'] == float(arrays['NLL'].mean())
    assert record['sum_token_NLL'] == float(arrays['NLL'].sum())
    assert record['teacher_forced_argmax_token_accuracy'] == float(np.mean(arrays['argmax'] == arrays['teacher_ids']))
    return len(ids)


def setup(key, confirmation, kind):
    contract, manifest, rows, groups = material(key, confirmation)
    allowed = {'confirmation'} if confirmation else {'diagnostic'} if kind == 'prospective' else {'validation', 'diagnostic'}
    rows = [r for r in rows if r['split'] in allowed]; groups = [g for g in groups if g['split'] in allowed]
    for ref in manifest['tokenizer_metadata'][key]['files']: assert sha(ROOT/ref['path']) == ref['sha256']
    tok = AutoTokenizer.from_pretrained(ROOT/'models/hf'/MODELS[key], local_files_only=True,
                                       use_fast=True, trust_remote_code=True)
    config = read(ROOT/'models/hf'/MODELS[key]/'config.json')
    return contract, rows, groups, tok, config


def audit(kind, key, run=None, confirmation=False):
    start = time.monotonic()
    assert kind in {'learned', 'prospective'} and (kind != 'learned' or key == 'qwen4')
    assert (kind == 'learned') == (run is not None)
    unit = read(OUT/'unit/generated_identity_current.json')
    assert unit['all_passed'] and unit['audit_sha256'] == sha(__file__)
    contract, rows, groups, tok, config = setup(key, confirmation, kind)
    scope = 'confirmation' if confirmation else 'diagnostic' if kind == 'prospective' else 'nonconfirmation'
    if kind == 'learned':
        assert run in [r['run'] for r in read(OUT/'training/material_manifest.json')['run_inventory']]
        names = [run]; execution = OUT/'learned/execution.json'
    else:
        selection = read(OUT/'fit'/key/'validation_selection.json')
        names = [selection['primary_rule'], selection['control_rule']]
        execution = OUT/'prospective'/key/'execution.json'
    folders = {name: OUT/kind/(run if kind == 'learned' else key)/
               (scope if kind == 'learned' else name+'/'+scope) for name in names}
    results = {name: read(folder/'result.json') for name, folder in folders.items()}
    assert all(r['all_passed'] and r['execution_sha256'] == sha(execution) for r in results.values())
    revision = {'source': snapshot(__file__), 'field_reader': snapshot(Path(__file__).with_name('phase2748_rdc_identity_audit.py')),
                'scoring_source': snapshot(Path(__file__).with_name('rdc_question_material.py')),
                'material_manifest_sha256': sha(OUT/'material/manifest.json'), 'execution_sha256': sha(execution),
                'result_sha256': {name: sha(folder/'result.json') for name, folder in folders.items()}}
    target = OUT/'verification'/('generated_'+kind+'_'+(run if run else key)+'_'+scope+'.json')
    if target.exists():
        old = read(target); assert old['all_passed'] and old['execution'] == revision
        print('NATURAL_GENERATED_IDENTITY_ALREADY_COMPLETE', kind, run or key, scope, flush=True); return old
    fields = Fields(); by_id = {r['question_id']: r for r in rows}
    full = set(contract['capture']['full_history_context_ids']); maximum = contract['scoring']['greedy_maximum_new_tokens']
    width, depth = config['hidden_size'], config['num_hidden_layers']; inventories = []
    native_scope = 'confirmation' if confirmation else 'nonconfirmation'
    for name, folder in folders.items():
        result = results[name]; counts = Counter(); seen = set(); receipts = []
        assert result['contexts'] == len(groups) and result['questions'] == len(rows)
        if kind == 'learned':
            assert result['run'] == run and result['split_scope'] == scope
            checkpoint_sha = sha(OUT/'training'/run/'checkpoint96.json')
            assert result['checkpoint_sha256'] == checkpoint_sha
        else:
            assert result['model'] == key and result['variant'] == name and result['split'] == scope
            deployed = OUT/'fit'/key/'deployed'/(name+'.json'); deployed_record = read(deployed)
            expected_predictor = {'variant': name, 'record_sha256': sha(deployed),
                                  'field_sha256': deployed_record['field']['sha256'],
                                  'selection_sha256': sha(OUT/'fit'/key/'validation_selection.json')}
            assert result['predictor'] == expected_predictor
        for group in groups:
            path = folder/'groups'/(group['group_id']+'.json'); record = read(path)
            receipts.append({'path': path.relative_to(ROOT).as_posix(), 'sha256': sha(path)})
            assert all(record[k] == group[k] for k in ['group_id', 'cohort', 'split'])
            assert record['execution_sha256'] == sha(execution)
            assert [q['question_id'] for q in record['questions']] == group['four_initial_question_ids']
            if kind == 'learned':
                assert record['run'] == run and record['checkpoint_sha256'] == checkpoint_sha
                native = read(OUT/'native'/key/native_scope/'groups'/(group['group_id']+'.json'))
                native_questions = {q['question_id']: q for q in native['questions']}
            else: assert record['predictor'] == expected_predictor
            for q in record['questions']:
                qid = q['question_id']; assert qid not in seen; seen.add(qid); row = by_id[qid]
                retained = row['group_id'] in full
                history = q if kind == 'prospective' else q['history']
                arrays = fields.read(history['field'], list(history['field']['arrays']))
                steps = check_history(row, history, arrays, tok, maximum)
                counts['free_histories'] += 1; counts['free_tokens'] += steps
                counts['complete_match_and_stop'] += int(history['score']['whole_response_exact_and_stopped'])
                counts['full_hidden_or_predicted_trajectories'] += int(retained)
                if kind == 'prospective':
                    assert history['predictor'] == expected_predictor and history['native_late_layers_executed'] is False
                    assert history['full_coordinate_trajectories_retained'] == retained
                    assert set(arrays) == BASE_HISTORY | {'casting_statistics'} | (PREDICTED_HISTORY if retained else set())
                    assert arrays['casting_statistics'].shape == (steps, 4) and np.isfinite(arrays['casting_statistics']).all()
                    assert np.all(arrays['casting_statistics'] >= 0) and np.all(arrays['casting_statistics'][:, 3] == 0)
                    if retained:
                        for k in PREDICTED_HISTORY: assert arrays[k].shape == (steps, width)
                        for k in PREDICTED_HISTORY-{'predicted_postnorm_FP64'}: assert arrays[k].dtype == np.uint16
                        predicted = arrays['predicted_postnorm_FP64']; cast = unbits(arrays['predicted_postnorm_BF16']).astype(np.float64)
                        assert predicted.dtype == np.float64 and np.isfinite(predicted).all() and np.isfinite(cast).all()
                        error = cast-predicted
                        rebuilt = np.stack([np.mean(error**2, axis=1), np.max(np.abs(error), axis=1),
                                            np.linalg.norm(predicted, axis=1), np.zeros(steps)], axis=1)
                        assert np.allclose(rebuilt, arrays['casting_statistics'], rtol=2e-14, atol=1e-14)
                else:
                    assert history['checkpoint_sha256'] == q['teacher']['checkpoint_sha256'] == checkpoint_sha
                    assert q['original_H12_and_source_read_bit_equal'] and history['full_H_all_layers_every_generated_step'] == retained
                    assert all(q[k] == row[k] for k in ['question_id', 'group_id', 'split', 'cohort'])
                    first = fields.read(q['field'], list(q['field']['arrays']))
                    assert first['postnorm_BF16'].shape == (width,) and first['postnorm_BF16'].dtype == np.uint16
                    original = fields.read(native_questions[qid]['field'], ['H12_last_BF16', 'native_source_read_BF16'])
                    for k in original:
                        assert np.array_equal(first[k], original[k]) and first[k].shape == (width,)
                    assert set(arrays) == BASE_HISTORY | (NATIVE_HISTORY if retained else set())
                    if retained:
                        assert first['hidden_BF16'].shape == (depth+1, width)
                        assert arrays['all_hidden_BF16'].shape == (steps, depth+1, width)
                        assert np.array_equal(arrays['all_hidden_BF16'][0], first['hidden_BF16'])
                        for k in ['postnorm_BF16', 'H12_last_BF16', 'native_source_read_BF16']:
                            assert arrays[k].shape == (steps, width) and np.array_equal(arrays[k][0], first[k])
                    else: assert set(first) == {'postnorm_BF16', 'H12_last_BF16', 'native_source_read_BF16'}
                    teacher = fields.read(q['teacher']['field'], list(q['teacher']['field']['arrays']))
                    counts['teacher_tokens'] += check_teacher(row, q['teacher'], teacher, first, arrays)
            if len(receipts) % 24 == 0: print('NATURAL_GENERATED_IDENTITY', kind, name, scope, len(receipts), len(groups), flush=True)
        assert seen == set(by_id)
        assert counts['free_tokens'] == result['generated_tokens' if kind == 'prospective' else 'free_generated_tokens']
        if kind == 'learned': assert counts['teacher_tokens'] == result['teacher_tokens']
        else: assert counts['full_hidden_or_predicted_trajectories'] == result['full_coordinate_trajectory_questions']
        assert counts['full_hidden_or_predicted_trajectories'] == sum(r['group_id'] in full for r in rows)
        inventories.append({'name': name, 'counts': dict(counts), 'groups': receipts})
    value = {'timestamp': stamp(), 'all_passed': True, 'kind': kind, 'model': key, 'run': run, 'scope': scope,
             'execution': revision, 'inventories': inventories, 'field_files_SHA256_checked': len(fields.checked),
             'seconds': time.monotonic()-start,
             'scope_note': 'Re-decodes all actually emitted token IDs and re-runs the unchanged complete-response scorer; verifies native token positions, stop/cap, stored field membership, teacher summaries, learned first-state early locality and retained prediction casting statistics. No new decoder/head replay, no semantic equivalence judgment, and no field equivalence claim after histories diverge.'}
    immutable(target, value)
    print('NATURAL_GENERATED_IDENTITY_PASS', kind, run or key, scope, round(value['seconds'], 2), flush=True)
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--kind', choices=['learned', 'prospective'], required=True)
    parser.add_argument('--model', choices=['qwen4', 'qwen14', 'glm4'], required=True)
    parser.add_argument('--run'); parser.add_argument('--confirmation', action='store_true')
    args = parser.parse_args(); audit(args.kind, args.model, args.run, args.confirmation)
