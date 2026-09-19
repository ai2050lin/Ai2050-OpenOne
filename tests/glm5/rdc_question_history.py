"""Complete teacher likelihood and independent native greedy histories.

Teacher tokens are given conditioning, never reported as free generation.
Every free-history continuation consumes only its own previously emitted ID.
"""
from rdc_question_common import *
from rdc_question_material import conservative_complete_answer


def distribution(model, postnorm):
    import torch
    with torch.inference_mode():
        lp = model.lm_head(postnorm).float()[0].double().log_softmax(-1)
        choice = int(lp.argmax())
        return lp, choice, float(-(lp.exp()*lp).sum())


def step_request(request, token, collect_hidden=False):
    import torch
    return {'input_ids': torch.tensor([[token]], device=request['input_ids'].device),
        'cache': request['cache'], 'mode': 'history', 'collect_hidden': collect_hidden,
        'collect_units': False, 'account_attention': False, 'save_source_details': False}


def teacher_likelihood(model, engine, requests, first_outputs, rows, progress=None):
    """B1 sequential teacher forcing from cloned actual question-prefix cache."""
    import torch
    state = []
    for request, output, row in zip(requests, first_outputs, rows):
        target = row['tokens']['teacher_ids_including_EOS']
        state.append({'request': {'cache': clone_cache(request['cache'], model.config),
                                 'input_ids': request['input_ids']}, 'output': output,
            'target': target, 'NLL': [], 'argmax': [], 'postnorm': [], 'entropy': []})
    started = time.monotonic()
    for step in range(max(len(s['target']) for s in state)):
        active = [s for s in state if step < len(s['target'])]
        if step:
            next_requests = [step_request(s['request'], s['target'][step-1]) for s in active]
            outputs = engine.forward(next_requests)
            for s, request, output in zip(active, next_requests, outputs):
                s['request'], s['output'] = request, output
        for s in active:
            lp, choice, entropy = distribution(model, s['output']['postnorm'])
            s['NLL'].append(float(-lp[s['target'][step]]))
            s['argmax'].append(choice)
            s['entropy'].append(entropy)
            s['postnorm'].append(s['output']['fields']['postnorm_BF16'].copy())
            if step+1 == len(s['target']):
                s['request']['cache'] = None
            del lp
        if progress:
            progress('teacher', step+1, sum(len(s['NLL']) for s in state), time.monotonic()-started)
    packets = []
    for row, s in zip(rows, state):
        assert len(s['NLL']) == len(s['target'])
        arrays = {'teacher_ids': np.asarray(s['target'], np.int64),
            'NLL': np.asarray(s['NLL'], np.float64), 'argmax': np.asarray(s['argmax'], np.int64),
            'postnorm_BF16': np.stack(s['postnorm']), 'entropy_FP64': np.asarray(s['entropy'], np.float64)}
        record = {'question_id': row['question_id'], 'teacher_text': row['tokens']['teacher_text'],
            'teacher_ids_including_EOS': s['target'], 'tokens': len(s['target']),
            'mean_token_NLL': float(np.mean(s['NLL'])), 'sum_token_NLL': float(np.sum(s['NLL'])),
            'teacher_forced_argmax_token_accuracy': float(np.mean(np.asarray(s['argmax']) == s['target'])),
            'scope': 'Given complete-answer history, sequential nativeB1 prediction, native EOS included. Not free-generation accuracy.'}
        packets.append((arrays, record))
    return packets, time.monotonic()-started


def greedy_histories(model, tok, engine, requests, first_outputs, rows, contract, progress=None):
    import torch
    stop = model.generation_config.eos_token_id or tok.eos_token_id
    stops = set(stop if isinstance(stop, list) else [stop])
    maximum = contract['scoring']['greedy_maximum_new_tokens']
    full_groups = set(contract['capture']['full_history_context_ids'])
    state = []
    for request, output, row in zip(requests, first_outputs, rows):
        state.append({'request': request, 'output': output, 'row': row, 'done': False,
            'full': row['group_id'] in full_groups, 'tokens': [], 'postnorm': [], 'H12': [],
            'source_read': [], 'statistics': [], 'hidden': []})
    started = time.monotonic()
    for step in range(maximum):
        active = [s for s in state if not s['done']]
        if not active:
            break
        if step:
            next_requests = [step_request(s['request'], s['tokens'][-1], s['full']) for s in active]
            outputs = engine.forward(next_requests)
            for s, request, output in zip(active, next_requests, outputs):
                s['request'], s['output'] = request, output
        for s in active:
            value = s['output']
            fields = value['fields']
            lp, choice, entropy = distribution(model, value['postnorm'])
            s['tokens'].append(choice)
            s['statistics'].append([entropy, float(lp[choice])])
            for dest, name in [('postnorm', 'postnorm_BF16'), ('H12', 'H12_last_BF16'), ('source_read', 'native_source_read_BF16')]:
                s[dest].append(fields[name].copy())
            if s['full']:
                s['hidden'].append(fields['hidden_BF16'].copy())
            s['done'] = choice in stops or len(s['tokens']) == maximum
            if s['done']:
                s['request']['cache'] = None
            del lp
        if progress:
            progress('greedy', step+1, sum(len(s['tokens']) for s in state), time.monotonic()-started)
    packets = []
    for s in state:
        row = s['row']
        ids = s['tokens']
        eos = ids[-1] in stops
        text = tok.decode(ids, skip_special_tokens=True)
        arrays = {'generated_ids': np.asarray(ids, np.int64), 'postnorm_BF16': np.stack(s['postnorm']),
            'H12_last_BF16': np.stack(s['H12']), 'native_source_read_BF16': np.stack(s['source_read']),
            'statistics': np.asarray(s['statistics'], np.float64),
            'positions': np.arange(len(ids), dtype=np.int64)+len(row['tokens']['input_ids'])-1}
        if s['full']:
            arrays['all_hidden_BF16'] = np.stack(s['hidden'])
        assert all(np.isfinite(unbits(v) if v.dtype == np.uint16 else v).all() for v in arrays.values())
        record = {'question_id': row['question_id'], 'group_id': row['group_id'], 'cohort': row['cohort'],
            'split': row['split'], 'generated_ids': ids, 'generated_text': text, 'native_EOS': eos,
            'maximum_new_tokens': maximum, 'censored': not eos and len(ids) == maximum,
            'score': conservative_complete_answer(text, row['answer_annotations'], eos),
            'full_H_all_layers_every_generated_step': s['full'],
            'statistics_columns': ['complete_vocabulary_entropy_FP64', 'chosen_token_log_probability_FP64'],
            'alignment': 'All saved fields at step t precede the emission of generated_ids[t]. Step0is the registered segmented question prefix; latersteps consume one own previously emitted token.',
            'history_scope': 'Native free generation, no teacher/other question answer fed into history.'}
        packets.append((arrays, record))
    return packets, time.monotonic()-started
