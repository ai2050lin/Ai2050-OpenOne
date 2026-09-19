"""Restartable complete native atlas and full histories, with measured B1 waves."""
import argparse
from collections import defaultdict
from rdc_question_common import *
from rdc_question_native import QuestionNative
from rdc_question_history import teacher_likelihood, greedy_histories
from phase2748_rdc_native_capture import revision, context_requests, question_requests
from phase2748_rdc_context_inventory import ranked
from rdc_operator_model import memory


def history_qualified(key):
    pointer = read(OUT/'history_cost'/key/'current.json')
    assert sha(ROOT/pointer['path']) == pointer['sha256']
    result = read(ROOT/pointer['path'])
    assert result['all_passed']
    assert result['execution']['history']['sha256'] == sha(Path(__file__).with_name('rdc_question_history.py'))
    return result, pointer


def deduplicated_pilot_fields(key):
    pointer = read(OUT/'pilot'/key/'current.json')
    assert sha(ROOT/pointer['path']) == pointer['sha256']
    pilot_result = read(ROOT/pointer['path'])
    result = {}
    for record in pilot_result['records']:
        identifier = record['group_id'] if record['kind'] == 'shared_context' else record['question_id']
        result[identifier] = record['field']
    return result


def reused_or_committed(folder, name, arrays, previous=None):
    if previous is None:
        return commit_arrays(folder, name, arrays)
    assert sha(ROOT/previous['path']) == previous['sha256']
    with np.load(ROOT/previous['path']) as z:
        assert set(z.files) == set(arrays)
        assert all(np.array_equal(z[k], v) for k, v in arrays.items()), 'New B1wave differs from its original native pilot field'
    # An exact reference, not a copied field or a replacement of the new forward.
    return previous


def memory_forecast(model, groups, by_id, histories):
    c = model.config
    element = 2*c.num_hidden_layers*c.num_key_value_heads*c.head_dim*2
    source_tokens = sum(len(by_id[g['four_initial_question_ids'][0]]['tokens']['context_prefix_ids']) for g in groups)
    prompt_tokens = sum(len(by_id[q]['tokens']['input_ids']) for g in groups for q in g['four_initial_question_ids'])
    extra_teacher = sum(len(by_id[q]['tokens']['teacher_ids_including_EOS'])-1 for g in groups for q in g['four_initial_question_ids'])
    questions = 4*len(groups)
    live_tokens = source_tokens+prompt_tokens
    if histories:
        # During teacher scoring both the untouched free prefix and its teacher
        # clone exist. During free generation all caches are conservatively
        # charged the128token cap, even if some requests terminate sooner.
        live_tokens = max(source_tokens+2*prompt_tokens+extra_teacher,
                          source_tokens+prompt_tokens+128*questions)
    activation_bytes = sum(len(by_id[g['four_initial_question_ids'][0]]['tokens']['context_prefix_ids'])
                           for g in groups)*c.hidden_size*2*4
    current_layer_reserve = 0 if model.config.hidden_size == 2560 else 1024**3
    workspace = 1024**3
    return {'contexts': len(groups), 'questions': questions, 'source_tokens': source_tokens,
        'prompt_tokens': prompt_tokens, 'KV_bytes_per_token_all_layers': element,
        'maximum_simultaneously_live_cache_tokens': live_tokens, 'maximum_cache_bytes': live_tokens*element,
        'live_activation_reserve_bytes': activation_bytes, 'current_layer_reserve_bytes': current_layer_reserve,
        'workspace_reserve_bytes': workspace,
        'additional_cuda_bytes': live_tokens*element+activation_bytes+current_layer_reserve+workspace,
        'no_early_stop_savings_assumed': True}


def select_wave(model, pending, by_id):
    import torch
    histories = pending[0]['split'] != 'train'
    torch.cuda.empty_cache()
    free, _ = torch.cuda.mem_get_info()
    selected = []
    forecast = None
    for group in pending[:16]:
        if (group['split'] != 'train') != histories or group['split'] != pending[0]['split']:
            break
        candidate = selected+[group]
        estimate = memory_forecast(model, candidate, by_id, histories)
        if estimate['additional_cuda_bytes'] > free-256*1024**2:
            break
        selected, forecast = candidate, estimate
    assert selected, ('No one-context safe B1wave fits', free, memory_forecast(model, pending[:1], by_id, histories))
    forecast['actual_free_cuda_before_wave'] = free
    forecast['mode'] = 'independent_B1_layer_major_no_arithmetic_batch_merge'
    return selected, forecast


def fixed_resource_forecast():
    """Full-stage estimate uses actual pilots; it is not a guaranteed token bound."""
    path = OUT/'resources/acquisition_admission.json'
    if path.exists():
        return read(path)
    contract = effective_contract()
    from phase2748_rdc_retention_contract import freeze as retention_freeze
    retention_freeze()
    entries = []
    first_fields = 0
    history_fields = 0
    for key in contract['models']:
        p = read(OUT/'pilot'/key/'current.json')
        assert sha(ROOT/p['path']) == p['sha256']
        prefix = read(ROOT/p['path'])
        own, hp = history_qualified(key)
        first_fields += prefix['projected_first_prefix_field_bytes']
        # All8cost examples retain everyH; formal only48/832do so. Estimate the
        # actual components individually from raw declared arrays with measured
        # whole-field compression ratio. Preserve a visible uncertainty margin.
        raw_ordinary = raw_full = raw_teacher = compressed_total = raw_total = 0
        for row in own['records']:
            field = row['field']
            full = field['arrays'].get('all_hidden_BF16')
            full_bytes = int(np.prod(full['shape']))*np.dtype(full['dtype']).itemsize if full else 0
            raw_full += full_bytes
            raw_ordinary += field['uncompressed_bytes']-full_bytes
            raw_teacher += row['teacher']['field']['uncompressed_bytes']
            compressed_total += field['bytes']+row['teacher']['field']['bytes']
            raw_total += field['uncompressed_bytes']+row['teacher']['field']['uncompressed_bytes']
        ratio = compressed_total/raw_total
        estimate = ratio*((raw_ordinary+raw_teacher)/8*832+raw_full/8*48)
        history_fields += estimate
        entries.append({'model': key, 'first_prefix_pointer': p, 'history_pointer': hp,
            'first_prefix_field_estimate': prefix['projected_first_prefix_field_bytes'],
            'native_history_field_estimate': estimate, 'observed_history_compression_ratio': ratio,
            'pilot_generated_tokens': own['native_generated_tokens'],
            'eight_request_free_seconds': own['first_free_seconds'],
            'timing_scope': 'Observed8-request wave only; larger safe independentB1waves may amortize weight I/O. Not a guarantee of future runtime.'})
    # Final fullFP32 and BF16 checkpoint storage uses the measured lossless-codec
    # projection; its uncompressed size is also reported. The fixed
    # remainder reserves selected deployable coefficients, full-vocabulary scores,
    # training diagnostics and learned/predicted histories. Actual writes retain
    # a4GiB physical reserve and are re-estimated from committed data.
    codec = read(OUT/'unit/checkpoint_codec_current.json')
    assert codec['all_passed']
    parameters = codec['suggested_six_run_storage_projection_bytes']
    remainder = 1600*1024**2
    estimate = int(first_fields+history_fields+parameters+remainder)
    free = shutil.disk_usage(FIELDS).free
    value = {'timestamp': stamp(), 'source': snapshot(__file__), 'models': entries,
        'first_prefix_fields_estimate_bytes': first_fields, 'native_history_fields_estimate_bytes': history_fields,
        'six_final_FP32_plus_BF16_parameters_uncompressed_bytes': 6*74711040*6,
        'six_final_lossless_parameter_storage_projection_bytes': parameters,
        'historical_complete_parameter_codec_receipt_sha256': sha(OUT/'unit/checkpoint_codec_current.json'),
        'retention_contract_sha256': sha(OUT/'retention_contract.json'),
        'remaining_fit_training_and_generated_history_estimate_bytes': remainder,
        'full_stage_initial_estimate_bytes': estimate, 'actual_free_C_bytes': free,
        'minimum_free_C_bytes': 4*1024**3,
        'estimated_headroom_after_all_new_files_and_reserve': free-estimate-4*1024**3,
        'within_initial_estimate': free-estimate > 4*1024**3,
        'uncertainty': 'Small cost pilots and content-dependent generation lengths. Not a physical worst-case proof; all new writes re-check actualfree4GiB. If the projection is exceeded, pause expansion for an explicit retention/storage audit, never delete key data or silently truncate histories.',
        'all_hidden_coordinates_retained': True,
        'coefficient_retention': 'Persist exact selected deployment coefficients and all freeze hyperparameters/source arrays. Secondary/alternative fits may be deterministically reconstructed from retained native full-coordinate targets and exact fit recipes; do not duplicate enormous all-target coefficient matrices only for storage convenience.'}
    immutable(path, value)
    assert value['within_initial_estimate'], ('Full-stage storage forecast needs an explicit revision before expansion', value)
    print('NATURAL_FULL_STAGE_STORAGE_ADMITTED', estimate, value['estimated_headroom_after_all_new_files_and_reserve'], flush=True)
    return value


def main(key, confirmation=False, limit_contexts=None):
    import torch
    from rdc_native_tail import cuda_singleton
    from rdc_formation_readout import CUDA_FORMATION
    cuda_singleton(set(CUDA_FORMATION) | {'phase2748_rdc_native_capture.py', 'phase2748_rdc_history_cost.py',
        'phase2748_rdc_acquire.py', 'phase2748_rdc_training.py', 'phase2748_rdc_prospective.py'})
    contract, manifest, rows, groups = material(key, confirmation)
    admission = fixed_resource_forecast()
    assert admission['within_initial_estimate']
    own_qualification, _ = history_qualified(key)
    assert own_qualification['execution']['qualification'] == revision(key)
    phase = 'confirmation' if confirmation else 'nonconfirmation'
    folder = Path('native')/key/phase
    rev = {'source': snapshot(__file__), 'native_qualification': revision(key),
        'history': snapshot(Path(__file__).with_name('rdc_question_history.py')),
        'resource_admission_sha256': sha(OUT/'resources/acquisition_admission.json'),
        'material_manifest_sha256': sha(OUT/'material/manifest.json')}
    protocol_path = OUT/folder/'execution.json'
    if protocol_path.exists():
        assert read(protocol_path) == rev, 'Acquisition source changed: preserve old run and audit a versioned continuation'
    else:
        immutable(protocol_path, rev)
    target = OUT/folder/'result.json'
    if target.exists():
        assert read(target)['all_passed']
        return read(target)
    by_id = {r['question_id']: r for r in rows}
    pilot_fields = deduplicated_pilot_fields(key)
    order = {'train': 0, 'validation': 1, 'diagnostic': 2, 'confirmation': 3}
    groups = sorted(groups, key=lambda g: (order[g['split']], ranked('acquire/'+g['group_id'])))
    committed = []
    for group in groups:
        receipt = OUT/folder/'groups'/(group['group_id']+'.json')
        if receipt.exists():
            record = read(receipt)
            assert record['execution'] == rev
            refs = [record['context_field']]+[q['field'] for q in record['questions']]
            refs += [q[name]['field'] for q in record['questions'] for name in ['teacher', 'history'] if name in q]
            assert all(sha(ROOT/ref['path']) == ref['sha256'] for ref in refs)
            committed.append(record)
    done = {g['group_id'] for g in committed}
    pending = [g for g in groups if g['group_id'] not in done]
    start = time.monotonic()
    model = engine = None
    try:
        storage_guard()
        model, tok = load(key, OUT/folder/'loader', gpu_limit=6 if key == 'qwen14' else 11 if key == 'glm4' else None)
        engine = QuestionNative(model, contract['capture']['selected_MLP_blocks'][key])
        new_contexts = 0
        while pending:
            batch, forecast = select_wave(model, pending, by_id)
            if limit_contexts is not None:
                batch = batch[:max(0, limit_contexts-new_contexts)]
                if not batch:
                    break
                forecast = memory_forecast(model, batch, by_id, batch[0]['split'] != 'train')
            storage_guard()
            mem = memory()
            assert mem['host_available_bytes'] > 2*1024**3 and mem.get('system_commit_headroom', 100*1024**3) > 2*1024**3
            tick = time.monotonic()
            batch_rows = [by_id[qid] for g in batch for qid in g['four_initial_question_ids']]
            contexts = context_requests(key, batch, rows, model.config, contract)
            source_outputs = engine.forward(contexts)
            requests = question_requests(key, batch, rows, contexts, model.config)
            first_outputs = engine.forward(requests)
            context_refs = [reused_or_committed(folder/'contexts', g['group_id'], o['fields'], pilot_fields.get(g['group_id']))
                            for g, o in zip(batch, source_outputs)]
            question_records = {}
            for row, output in zip(batch_rows, first_outputs):
                qid = row['question_id']
                ref = reused_or_committed(folder/'questions', qid.replace(':', '_'), output['fields'], pilot_fields.get(qid))
                question_records[qid] = {'question_id': qid, 'group_id': row['group_id'], 'split': row['split'], 'cohort': row['cohort'],
                    'field': ref, 'statistics': complete_vocabulary(model, output['fields']['postnorm_BF16'], row['tokens']['teacher_ids_including_EOS'][0])}
            if batch[0]['split'] != 'train':
                last_progress = time.monotonic()
                def progress(stage, step, tokens, seconds):
                    nonlocal last_progress
                    if step == 1 or step % 8 == 0 or time.monotonic()-last_progress > 25:
                        print('NATURAL_ACQUIRE_HISTORY', key, len(committed), stage, step, tokens, round(seconds, 1), flush=True)
                        last_progress = time.monotonic()
                teachers, teacher_seconds = teacher_likelihood(model, engine, requests, first_outputs, batch_rows, progress)
                for (arrays, record) in teachers:
                    qid = record['question_id']
                    question_records[qid]['teacher'] = {**record, 'field': commit_arrays(folder/'teachers', qid.replace(':', '_'), arrays)}
                del teachers
                generated, generation_seconds = greedy_histories(model, tok, engine, requests, first_outputs, batch_rows, contract, progress)
                for arrays, record in generated:
                    qid = record['question_id']
                    question_records[qid]['history'] = {**record, 'field': commit_arrays(folder/'histories', qid.replace(':', '_'), arrays)}
                del generated
            else:
                teacher_seconds = generation_seconds = 0.
            elapsed = time.monotonic()-tick
            for group, reference in zip(batch, context_refs):
                record = {'timestamp': stamp(), 'group_id': group['group_id'], 'cohort': group['cohort'], 'split': group['split'],
                    'execution': rev, 'context_field': reference,
                    'questions': [question_records[qid] for qid in group['four_initial_question_ids']],
                    'wave_context_ids': [g['group_id'] for g in batch], 'wave_memory_forecast': forecast,
                    'wave_elapsed_seconds': elapsed, 'wave_teacher_seconds': teacher_seconds, 'wave_generation_seconds': generation_seconds}
                immutable(OUT/folder/'groups'/(group['group_id']+'.json'), record)
                committed.append(record)
            del requests, first_outputs, contexts, source_outputs, question_records
            gc.collect()
            torch.cuda.empty_cache()
            done = {g['group_id'] for g in committed}
            pending = [g for g in groups if g['group_id'] not in done]
            new_contexts += len(batch)
            status = {'timestamp': stamp(), 'model': key, 'split_scope': phase, 'contexts': len(committed), 'total_contexts': len(groups),
                'first_prefix_questions': 4*len(committed), 'total_first_prefix_questions': len(rows),
                'seconds_this_process': time.monotonic()-start, 'memory': memory(), 'CUDA_allocated': torch.cuda.memory_allocated(),
                'native_MLP_product_checked_scalars_this_process': engine.product_checked_scalars,
                'native_attention_replay_checks_this_process': engine.attention_replay_checks,
                'goal_complete': False}
            save(OUT/folder/'progress.json', status)
            print('NATURAL_ACQUIRE', key, len(committed), len(groups), round(status['seconds_this_process'], 1), flush=True)
        if pending:
            print('NATURAL_ACQUIRE_BOUNDED_PROCESS_SAVED', key, len(committed), len(groups), flush=True)
            return
        flat = [q for g in committed for q in g['questions']]
        result = {'timestamp': stamp(), 'all_passed': True, 'execution': rev, 'model': key, 'split_scope': phase,
            'contexts': len(committed), 'questions': len(flat), 'native_free_histories': sum('history' in q for q in flat),
            'native_free_tokens': sum(len(q['history']['generated_ids']) for q in flat if 'history' in q),
            'teacher_tokens': sum(q['teacher']['tokens'] for q in flat if 'teacher' in q),
            'group_receipts': [(folder/'groups'/(g['group_id']+'.json')).as_posix() for g in committed],
            'seconds_final_process': time.monotonic()-start,
            'scope': 'Complete declared native acquisition for this model/split scope; fitted mechanism and learned/prospective behavior require their separate stages.'}
        immutable(target, result)
        print('NATURAL_ACQUIRE_COMPLETE', key, phase, len(flat), result['native_free_tokens'], flush=True)
        return result
    except Exception as exc:
        failure(OUT/folder, start, exc)
        raise
    finally:
        if engine is not None:
            engine.close()
        if model is not None:
            del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['qwen4', 'qwen14', 'glm4'], required=True)
    parser.add_argument('--confirmation', action='store_true')
    parser.add_argument('--limit-contexts', type=int)
    args = parser.parse_args()
    main(args.model, args.confirmation, args.limit_contexts)
