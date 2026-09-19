"""CUDA qualification and native same-context question-prefix atlas acquisition."""
import argparse
from collections import defaultdict
from rdc_question_common import *
from rdc_question_native import QuestionNative, native_whole_B1
from rdc_operator_model import memory
from phase2748_rdc_context_inventory import ranked


def revision(key):
    engine = Path(__file__).with_name('rdc_question_native.py')
    unit = read(OUT/'unit/native_engine_current.json')
    assert unit['all_passed'] and unit['engine']['sha256'] == sha(engine)
    return {'engine': snapshot(engine), 'source': snapshot(__file__),
        'common': snapshot(Path(__file__).with_name('rdc_question_common.py')),
        'effective_contract_sha256': sha(OUT/'effective_experiment_contract.json'),
        'unit_sha256': sha(OUT/'unit/native_engine_current.json'), 'model': key}


def context_requests(key, batch, rows, config, contract):
    import torch
    by_id = {r['question_id']: r for r in rows}
    requests = []
    for group in batch:
        row = by_id[group['four_initial_question_ids'][0]]
        token = row['tokens']
        requests.append({'group_id': group['group_id'], 'input_ids': torch.tensor([token['context_prefix_ids']], device='cuda'),
            'mode': 'context', 'context_positions_local': token['context_token_positions'],
            'collect_hidden': True, 'full_prefix_H': group['group_id'] in contract['capture']['full_source_H_context_ids']})
    return requests


def question_requests(key, batch, rows, contexts, config, reverse=False):
    import torch
    by_id = {r['question_id']: r for r in rows}
    requests = []
    for group, context in zip(batch, contexts):
        for qid in group['four_initial_question_ids']:
            row = by_id[qid]
            token = row['tokens']
            split = token['context_prefix_length']
            requests.append({'question_id': qid, 'group_id': group['group_id'], 'mode': 'question',
                'input_ids': torch.tensor([token['question_branch_ids']], device='cuda'),
                'cache': clone_cache(context['cache'], config), 'collect_hidden': True, 'collect_units': True,
                'question_positions_local': [p-split for p in token['question_token_positions']],
                'account_attention': True, 'save_source_details': True,
                'source_positions': token['context_token_positions'], 'context_prefix_length': split,
                'value_permutation_seed': int(ranked('value_pair/'+key+'/'+group['group_id'])[:16], 16)})
    return requests[::-1] if reverse else requests


def equality(left, right):
    assert set(left) == set(right)
    mismatch = [name for name in left if not np.array_equal(left[name], right[name])]
    assert not mismatch, ('Repeated native field differs', mismatch)


def compare_shape(model, whole, segmented):
    import torch
    a = unbits(whole['hidden_BF16']).astype(float)
    b = unbits(segmented['hidden_BF16']).astype(float)
    pa = unbits(whole['postnorm_BF16']).astype(float)
    pb = unbits(segmented['postnorm_BF16']).astype(float)
    with torch.inference_mode():
        ah = tensor_bits_to_cuda(whole['postnorm_BF16'])[None]
        bh = tensor_bits_to_cuda(segmented['postnorm_BF16'])[None]
        la = model.lm_head(ah).float()[0].double().log_softmax(-1)
        lb = model.lm_head(bh).float()[0].double().log_softmax(-1)
    return {'whole_vs_segmented_H_bit_equal': bool(np.array_equal(whole['hidden_BF16'], segmented['hidden_BF16'])),
        'layer_MSE': np.mean((a-b)**2, axis=-1).tolist(), 'postnorm_MSE': float(np.mean((pa-pb)**2)),
        'whole_to_segmented_full_vocabulary_KL': float((la.exp()*(la-lb)).sum()),
        'whole_argmax': int(la.argmax()), 'segmented_argmax': int(lb.argmax()),
        'scope': 'Native arithmetic execution-shape control; this difference is not attributed to question semantics.'}


def pilot(model, tok, engine, key, contract, rows, groups, rev):
    import torch
    folder = Path('pilot')/key/rev['engine']['sha256'][:12]
    target = OUT/folder/'result.json'
    if target.exists():
        value = read(target)
        assert value['all_passed'] and value['execution'] == rev
        return value
    start = time.monotonic()
    selected = [next(g for g in groups if g['group_id'] == gid) for gid in contract['capture']['pilot_context_ids']]
    by_id = {r['question_id']: r for r in rows}
    pilot_rows = [by_id[qid] for g in selected for qid in g['four_initial_question_ids']]
    whole, controls = {}, []
    for row in pilot_rows:
        ids = torch.tensor([row['tokens']['input_ids']], device='cuda')
        reference = native_whole_B1(model, ids, weights=engine.weights)
        request = {'input_ids': ids, 'mode': 'plain', 'collect_hidden': True}
        wave = engine.forward([request])[0]['fields']
        equality(reference, {k: wave[k] for k in reference})
        whole[row['question_id']] = reference
        del request, wave
        print('NATURAL_NATIVE_WHOLE_CHECK', key, len(whole), 8, flush=True)
    context = context_requests(key, selected, rows, model.config, contract)
    context_fields = engine.forward(context)
    initial_cache_ids = [cache_id(r['cache']) for r in context]
    requests = question_requests(key, selected, rows, context, model.config)
    outputs = engine.forward(requests)
    question_fields = {r['question_id']: output['fields'] for r, output in zip(requests, outputs)}
    del requests, outputs
    reverse = question_requests(key, selected, rows, context, model.config, reverse=True)
    repeated = engine.forward(reverse)
    for request, output in zip(reverse, repeated):
        equality(question_fields[request['question_id']], output['fields'])
    assert initial_cache_ids == [cache_id(r['cache']) for r in context]
    del reverse, repeated
    repeated_context_requests = context_requests(key, selected[::-1], rows, model.config, contract)
    repeated_context = engine.forward(repeated_context_requests)[::-1]
    for first, second in zip(context_fields, repeated_context):
        equality(first['fields'], second['fields'])
    del repeated_context_requests, repeated_context
    records = []
    for group, output in zip(selected, context_fields):
        field = commit_arrays(folder/'contexts', group['group_id'], output['fields'])
        records.append({'kind': 'shared_context', 'group_id': group['group_id'], 'field': field})
    for row in pilot_rows:
        qid = row['question_id']
        fields = question_fields[qid]
        field = commit_arrays(folder/'questions', qid.replace(':', '_'), fields)
        whole_field = commit_arrays(folder/'whole_B1', qid.replace(':', '_'), whole[qid])
        stats = complete_vocabulary(model, fields['postnorm_BF16'], row['tokens']['teacher_ids_including_EOS'][0])
        control = compare_shape(model, whole[qid], fields)
        controls.append({'question_id': qid, **control})
        records.append({'kind': 'question', 'question_id': qid, 'field': field, 'whole_B1_field': whole_field, 'statistics': stats})
    # Pilot full-prefix fixtures are deliberately larger than ordinary context
    # fields; estimate their incremental fullH separately rather than multiplying
    # that exceptional retention by every context.
    context_bytes = sum(r['field']['bytes'] for r in records if r['kind'] == 'shared_context')
    question_bytes = sum(r['field']['bytes'] for r in records if r['kind'] == 'question')
    ordinary_source_raw = sum(sum(v.nbytes for k, v in o['fields'].items() if k != 'full_prefix_hidden_BF16') for o in context_fields)
    field_forecast = ordinary_source_raw/2*400 + question_bytes/8*1600 + context_bytes
    storage_guard(int(field_forecast*1.15))
    value = {'timestamp': stamp(), 'all_passed': True, 'execution': rev,
        'model': key, 'native_whole_B1_allH_postnorm_equal': 8,
        'segmented_reverse_order_all_saved_fields_bit_equal': 8,
        'repeated_shared_context_every_saved_field_bit_equal': 2,
        'shared_full_layer_cache_unchanged_after_all_branches': True,
        'shape_controls': controls, 'records': records,
        'native_attention_replay_checks': engine.attention_replay_checks,
        'native_MLP_product_checked_scalars': engine.product_checked_scalars,
        'projected_first_prefix_field_bytes': field_forecast,
        'projection_scope': '400contexts/1600question first-prefix fields for this model only. Full-history and training-checkpoint storage require separate actual forecasts; not an all-phase completion/resource proof.',
        'memory': memory(), 'cuda_allocated': torch.cuda.memory_allocated(),
        'cuda_peak_allocated': torch.cuda.max_memory_allocated(),
        'seconds': time.monotonic()-start, 'scope': 'Native CUDA numerical/capture qualification on8training questions, not heldout mechanism results.'}
    immutable(target, value)
    save(OUT/'pilot'/key/'current.json', {'path': target.relative_to(ROOT).as_posix(), 'sha256': sha(target)})
    print('NATURAL_NATIVE_PILOT_PASSED', key, round(value['seconds'], 2), int(field_forecast), flush=True)
    return value


def main(key, pilot_only=True):
    import torch
    from rdc_native_tail import cuda_singleton
    from rdc_formation_readout import CUDA_FORMATION
    cuda_singleton(set(CUDA_FORMATION) | {'phase2748_rdc_native_capture.py', 'phase2748_rdc_training.py', 'phase2748_rdc_prospective.py'})
    contract, manifest, rows, groups = material(key)
    rev = revision(key)
    start = time.monotonic()
    model = engine = None
    folder = OUT/'native'/key/rev['engine']['sha256'][:12]
    try:
        storage_guard()
        model, tok = load(key, folder/'loader', gpu_limit=6 if key == 'qwen14' else 11 if key == 'glm4' else None)
        engine = QuestionNative(model, contract['capture']['selected_MLP_blocks'][key])
        value = pilot(model, tok, engine, key, contract, rows, groups, rev)
        assert pilot_only, 'Formal acquisition is enabled only after pilot evidence review and explicit full-scope resource assessment'
        return value
    except Exception as exc:
        failure(folder, start, exc)
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
    args = parser.parse_args()
    main(args.model)
