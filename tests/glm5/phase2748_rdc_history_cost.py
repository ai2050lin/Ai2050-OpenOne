"""Full-cap native history cost/replay qualification on fixed validation contexts."""
import argparse
from rdc_question_common import *
from rdc_question_native import QuestionNative
from rdc_question_history import teacher_likelihood, greedy_histories
from phase2748_rdc_native_capture import revision, pilot, context_requests, question_requests, equality
from phase2748_rdc_context_inventory import ranked
from rdc_operator_model import memory


def main(key):
    import torch
    from rdc_native_tail import cuda_singleton
    from rdc_formation_readout import CUDA_FORMATION
    cuda_singleton(set(CUDA_FORMATION) | {'phase2748_rdc_native_capture.py', 'phase2748_rdc_history_cost.py',
                                         'phase2748_rdc_acquire.py', 'phase2748_rdc_training.py'})
    contract, manifest, rows, groups = material(key)
    for model_key in contract['models']:
        pointer = read(OUT/'pilot'/model_key/'current.json')
        assert sha(ROOT/pointer['path']) == pointer['sha256'] and read(ROOT/pointer['path'])['all_passed']
    history_source = Path(__file__).with_name('rdc_question_history.py')
    rev = {'source': snapshot(__file__), 'history': snapshot(history_source), 'qualification': revision(key)}
    folder = Path('history_cost')/key/sha(history_source)[:12]
    target = OUT/folder/'result.json'
    if target.exists():
        assert read(target)['execution'] == rev and read(target)['all_passed']
        return read(target)
    start = time.monotonic()
    selected = [min((g for g in groups if g['cohort'] == c and g['split'] == 'validation'
                     and g['group_id'] in contract['capture']['full_history_context_ids']), key=lambda g: ranked(g['group_id']))
                for c in ['quoref', 'drop']]
    immutable(OUT/folder/'material_registration.json', {'timestamp': stamp(),
        'context_ids': [g['group_id'] for g in selected], 'execution': rev,
        'choice': 'One fixed hash-minimum validation full-Hhistory context per cohort; before their model outcomes. Caps and scoring unchanged. Used for cost/numerical replay, not to change feature/objective choices.'})
    model = engine = None
    try:
        storage_guard()
        model, tok = load(key, OUT/folder/'loader', gpu_limit=6 if key == 'qwen14' else 11 if key == 'glm4' else None)
        engine = QuestionNative(model, contract['capture']['selected_MLP_blocks'][key])
        by_id = {r['question_id']: r for r in rows}
        selected_rows = [by_id[qid] for g in selected for qid in g['four_initial_question_ids']]
        contexts = context_requests(key, selected, rows, model.config, contract)
        source_output = engine.forward(contexts)
        requests = question_requests(key, selected, rows, contexts, model.config)
        first = engine.forward(requests)
        prefix_ids = [cache_id(r['cache']) for r in requests]
        last_progress = time.monotonic()
        def progress(stage, step, tokens, seconds):
            nonlocal last_progress
            if step == 1 or step % 8 == 0 or time.monotonic()-last_progress > 25:
                print('NATURAL_HISTORY_COST', key, stage, step, tokens, round(seconds, 2), flush=True)
                last_progress = time.monotonic()
        teachers, teacher_seconds = teacher_likelihood(model, engine, requests, first, selected_rows, progress)
        assert [cache_id(r['cache']) for r in requests] == prefix_ids, 'Teacher scoring mutated free-history initial cache'
        packets, seconds = greedy_histories(model, tok, engine, requests, first, selected_rows, contract, progress)
        del requests, first
        repeated_requests = question_requests(key, selected, rows, contexts, model.config, reverse=True)
        repeated_first = engine.forward(repeated_requests)
        again, repeat_seconds = greedy_histories(model, tok, engine, repeated_requests, repeated_first,
                                                 selected_rows[::-1], contract, progress)
        for (a, ar), (b, br) in zip(packets, again[::-1]):
            equality(a, b)
            assert ar == br
        del repeated_requests, repeated_first, again
        records = []
        for (arrays, record), (teacher_arrays, teacher_record) in zip(packets, teachers):
            name = record['question_id'].replace(':', '_')
            field = commit_arrays(folder/'histories', name, arrays)
            teacher_field = commit_arrays(folder/'teachers', name, teacher_arrays)
            records.append({**record, 'field': field, 'teacher': {**teacher_record, 'field': teacher_field}})
        native_steps = sum(len(r['generated_ids']) for r in records)
        result = {'timestamp': stamp(), 'all_passed': True, 'execution': rev, 'model': key,
            'records': records, 'native_histories': len(records), 'native_generated_tokens': native_steps,
            'same_complete_histories_every_field_replayed_in_reverse_order': True,
            'teacher_scoring_left_free_prefix_cache_bit_identical': True,
            'teacher_seconds': teacher_seconds, 'first_free_seconds': seconds, 'repeat_free_seconds': repeat_seconds,
            'field_bytes': sum(r['field']['bytes']+r['teacher']['field']['bytes'] for r in records),
            'allocated_generation_seconds_per_token': seconds/native_steps,
            'cuda_peak_allocated': torch.cuda.max_memory_allocated(), 'memory': memory(),
            'seconds': time.monotonic()-start,
            'scope': 'Actual full-cap eight-question cost and replay qualification; these remain validation examples. One small cost sample cannot establish a maximum token count or aggregate heldout ability.'}
        immutable(target, result)
        save(OUT/'history_cost'/key/'current.json', {'path': target.relative_to(ROOT).as_posix(), 'sha256': sha(target)})
        print('NATURAL_HISTORY_COST_PASSED', key, native_steps, round(seconds, 2), flush=True)
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
    args = parser.parse_args()
    main(args.model)
