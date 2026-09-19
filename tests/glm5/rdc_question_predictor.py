"""Deploy frozen full-coordinate predictors using only permitted early inputs."""
from rdc_question_common import *


class FrozenPredictor:
    def __init__(self, key, variant):
        folder = OUT/'fit'/key
        self.record = read(folder/'deployed'/(variant+'.json'))
        selection = read(folder/'validation_selection.json')
        assert self.record['selection_sha256'] == sha(folder/'validation_selection.json')
        assert variant in [selection['primary_rule'], selection['control_rule']]
        reference = self.record['field']
        assert sha(ROOT/reference['path']) == reference['sha256']
        with np.load(ROOT/reference['path']) as z:
            self.values = {k:z[k].copy() for k in z.files}
        self.variant, self.key = variant, key
        self.rho = self.record['choice']['rho']
        self.identity = {'variant':variant,'record_sha256':sha(folder/'deployed'/(variant+'.json')),
            'field_sha256':reference['sha256'],'selection_sha256':self.record['selection_sha256']}

    def predict(self, H12, native_read, context_H12):
        h = np.atleast_2d(np.asarray(H12,dtype=np.float64))
        readout = np.atleast_2d(np.asarray(native_read,dtype=np.float64))
        context = np.atleast_2d(np.asarray(context_H12,dtype=np.float64))
        query = np.concatenate([h,readout],axis=1) if self.variant == 'native_source_read' else h
        assert self.variant in {'native_source_read','early_query_context','within_context_target_pair_shuffle'}
        state = self.values
        q = (query-state['query_mean'])/state['query_scale']
        c = (context-state['context_mean'])/state['context_scale']
        kq = q @ state['train_query'].T/q.shape[1]
        kc = c @ state['train_context'].T/c.shape[1]
        raw = kq+kc+self.rho*kq*kc
        centered = raw-(raw @ state['weights'])[:,None]-state['kernel_column_mean'][None,:]+state['kernel_grand_mean']
        predicted = centered @ state['coefficients']+state['target_mean']
        assert np.isfinite(predicted).all()
        return predicted


def early_contexts(key, groups, rows, model, engine, contract):
    from phase2748_rdc_native_capture import context_requests
    requests = context_requests(key,groups,rows,model.config,contract)
    for request in requests:
        request.update(collect_hidden=False,full_prefix_H=False)
    outputs = engine.forward(requests,stop_after=13)
    assert all(o['postnorm'] is None and 'postnorm_BF16' not in o['fields'] for o in outputs)
    contexts = {r['group_id']:o['fields']['context_H12_mean'] for r,o in zip(requests,outputs)}
    return requests, contexts


def early_cache_id(cache):
    assert all(layer.keys is None and layer.values is None for layer in cache.layers[13:])
    return [{'block':i,'keys':identity(bits(layer.keys)),'values':identity(bits(layer.values))}
            for i,layer in enumerate(cache.layers[:13])]


def clone_early_cache(cache,config):
    from transformers.cache_utils import DynamicCache
    early_cache_id(cache)
    # Native config creates uninitialized late cache slots. Clone the13real
    # layers only; do not initialize nonexistent late states or run late layers.
    return DynamicCache([(layer.keys.clone(),layer.values.clone())for layer in cache.layers[:13]],config=config)


def early_questions(key, groups, rows, contexts, model, engine, reverse=False):
    import torch
    by_id={r['question_id']:r for r in rows}
    requests=[]
    for group,context in zip(groups,contexts):
        for qid in group['four_initial_question_ids']:
            token=by_id[qid]['tokens']
            requests.append({'question_id':qid,'group_id':group['group_id'],'mode':'question',
                'input_ids':torch.tensor([token['question_branch_ids']],device=context['input_ids'].device),
                'cache':clone_early_cache(context['cache'],model.config),
                'collect_hidden':False,'collect_units':False,'account_attention':False,'save_source_details':False,
                'question_positions_local':[p-token['context_prefix_length']for p in token['question_token_positions']]})
    if reverse:requests.reverse()
    outputs = engine.forward(requests,stop_after=13)
    assert all(o['postnorm'] is None and 'postnorm_BF16' not in o['fields'] for o in outputs)
    return requests, outputs


def own_histories(model, tok, engine, predictor, requests, outputs, rows, context_means, contract, progress=None):
    import torch
    from rdc_question_history import step_request, distribution
    from rdc_question_material import conservative_complete_answer
    stops = set(rows[0]['tokens']['native_stop_ids'])
    assert all(set(row['tokens']['native_stop_ids']) == stops for row in rows)
    full = set(contract['capture']['full_history_context_ids'])
    maximum = contract['scoring']['greedy_maximum_new_tokens']
    states = [{'row':row,'request':request,'output':output,'ids':[],'statistics':[],
        'casting':[],'full':row['group_id'] in full,'fields':default_fields(),'done':False}
        for row,request,output in zip(rows,requests,outputs)]
    started = time.monotonic()
    for step in range(maximum):
        active = [s for s in states if not s['done']]
        if not active:
            break
        if step:
            next_requests = [step_request(s['request'],s['ids'][-1],False) for s in active]
            outputs = engine.forward(next_requests,stop_after=13)
            for s,r,o in zip(active,next_requests,outputs):
                s['request'],s['output'] = r,o
        for s in active:
            fields = s['output']['fields']
            assert s['output']['postnorm'] is None and 'postnorm_BF16' not in fields
            predicted = predictor.predict(unbits(fields['H12_last_BF16']),unbits(fields['native_source_read_BF16']),
                                          context_means[s['row']['group_id']])[0]
            actual = torch.from_numpy(predicted.copy()).to(device='cuda',dtype=torch.bfloat16)[None]
            assert bool(torch.isfinite(actual).all())
            cast = bits(actual[0])
            error = unbits(cast).astype(float)-predicted
            lp, choice, entropy = distribution(model,actual)
            s['ids'].append(choice)
            s['statistics'].append([entropy,float(lp[choice])])
            s['casting'].append([float(np.mean(error**2)),float(np.max(np.abs(error))),float(np.linalg.norm(predicted)),0.])
            if s['full']:
                s['fields']['H12_last_BF16'].append(fields['H12_last_BF16'].copy())
                s['fields']['native_source_read_BF16'].append(fields['native_source_read_BF16'].copy())
                s['fields']['predicted_postnorm_FP64'].append(predicted.copy())
                s['fields']['predicted_postnorm_BF16'].append(cast.copy())
            s['done'] = choice in stops or len(s['ids']) == maximum
            if s['done']:
                s['request']['cache'] = None
            del lp,actual
        if progress:
            progress(step+1,sum(len(s['ids']) for s in states),time.monotonic()-started)
    packets = []
    for s in states:
        row = s['row']
        ids = s['ids']; eos = ids[-1] in stops
        text = tok.decode(ids,skip_special_tokens=True)
        arrays = {'generated_ids':np.asarray(ids,np.int64),'statistics':np.asarray(s['statistics'],np.float64),
            'casting_statistics':np.asarray(s['casting'],np.float64),
            'positions':np.arange(len(ids),dtype=np.int64)+len(row['tokens']['input_ids'])-1}
        if s['full']:
            arrays.update({k:np.stack(v) for k,v in s['fields'].items()})
        record = {'question_id':row['question_id'],'group_id':row['group_id'],'split':row['split'],'cohort':row['cohort'],
            'generated_ids':ids,'generated_text':text,'native_EOS':eos,'censored':not eos and len(ids)==maximum,
            'maximum_new_tokens':maximum,'score':conservative_complete_answer(text,row['answer_annotations'],eos),
            'statistics_columns':['complete_vocabulary_entropy_FP64','chosen_token_log_probability_FP64'],
            'casting_columns':['full_coordinate_MSE','maximum_coordinate_error','unrounded_prediction_L2','nonfinite_count'],
            'full_coordinate_trajectories_retained':s['full'],'predictor':predictor.identity,
            'native_late_layers_executed':False,'native_layers_executed':[0,12],
            'history_scope':'Actual own generated past only; original native postnorm/laterH never computed or fed to predictor. First-prefix fit reused out of support on later steps.',
            'alignment':'Fields at step t precede emitted ID t; H12 is input to block12, read is its actual attention write. Final probability from predicted postnorm and original full BF16unembedding.'}
        packets.append((arrays,record))
    return packets,time.monotonic()-started


def default_fields():
    return {k:[] for k in ['H12_last_BF16','native_source_read_BF16','predicted_postnorm_FP64','predicted_postnorm_BF16']}
