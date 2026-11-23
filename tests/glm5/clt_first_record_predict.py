"""Freeze a compact, same-block attention-to-MLP path and test new families."""
from pathlib import Path
import json
import os
import time

import numpy as np

from clt_first_record_discovery import ROOT, HERE
from deploy.storage_policy import check_write
from deploy.test_archive_client import runtime_identity, save
from shared.dist_protocol.test_archive import CORE_ROLES, digest, validate_plan
from source_binding_design import read, sha

OUT = HERE/'result/clt_first_record_path_v1'
MATERIAL = OUT/'materials'
OLD = HERE/'result/clt_b26_competition_v1/confirm-materials.json'
SCRIPT = HERE/'clt_first_record_predict.py'
PLAN = OUT/'prediction-plan.json'
QUESTIONS = OUT/'prediction-materials.json'
SEED = 202611190315
BUDGET = 512
UNITS = 128
BATCH = 384
SOURCE_GATE = .90
GAIN_GATE = .20
CAP = .02


def dependencies():
    from clt_first_record_discovery import code_identity
    return {**code_identity(), str(SCRIPT.relative_to(ROOT)): sha(SCRIPT),
            'tests/glm5/clt_first_record_desktop.py': sha(HERE/'clt_first_record_desktop.py'),
            'tests/glm5/run_routine_test.py': sha(HERE/'run_routine_test.py'),
            'deploy/test_archive_client.py': sha(ROOT/'deploy/test_archive_client.py')}


def build_plan(discovery_record):
    discovery_record = discovery_record.resolve(strict=True)
    if not discovery_record.is_relative_to(HERE/'result/routine'):
        raise ValueError('发现记录必须来自本机登记工作区')
    review = read(discovery_record/'review.json')
    receipt = read(discovery_record/'server-receipt.json')
    if (review['archive_state'] != 'server_verified' or receipt.get('archive_state') != 'server_verified'
            or not receipt.get('readback_verified') or receipt.get('run_id')!=review.get('server_run_id')):
        raise ValueError('发现采集须先完成服务器核验')
    manifest = read(MATERIAL/'manifest.json')
    splits = {}
    for name in ('pilot', 'confirm'):
        part = manifest['splits'][name]
        path = MATERIAL/part['file']
        if sha(path) != part['material_sha256']:
            raise ValueError(f'{name} 冻结题目哈希不符')
        splits[name] = read(path)
    old = read(OLD)
    files = dependencies()
    # Exposed fit examples are uniformly sampled by family before reading activation data.
    fit = [dict(row, source_id=row['id'], id='fit-'+row['id']) for row in old if row['family'] % 4 == 0]
    rows = [dict(row, split=split) for split, batch in [('fit', fit), *splits.items()] for row in batch]
    save(QUESTIONS, rows)
    selection = {'head': 'one highest positive paired physical-first log-mass shift minus name/label, weighted by correct head ranking on the fixed H36 residual slice; exclude B26/H36',
                 'projections': '512 original columns each, discovery variance times squared native projection norm',
                 'mlp': '128 same-block units, original activation variance times squared down-column norm',
                 'comparison': ['original', 'equally-sized random projection columns', 'fit mean filling'],
                 'diagnostics_only':'Q-only / K-only selection; V-only O error, with the other two projection branches held native',
                 'fits': '32 exposed families divisible by four, never fit on new pilot or confirm',
                 'decision': {'source_agreement': SOURCE_GATE, 'improvement': GAIN_GATE, 'pilot_each_cell': .90},
                 'reference_record': str(discovery_record), 'reference_run': review['server_run_id'],
                 'discovery_receipt_sha256': sha(discovery_record/'server-receipt.json'),
                 'data_manifest_sha256': sha(MATERIAL/'manifest.json')}
    entry, file_sha = sha(SCRIPT), sha(QUESTIONS)
    plan = {
        'schema': 'research_test_plan.v1', 'kind': 'real',
        'title': 'Compact source → V/O → same-block MLP prediction',
        'objective': 'Predict original source choice and local path writes on new first-record families.',
        'principle': 'Select a head from independently archived discovery traces; approximate three projections from original columns; propagate into real same-block RMSNorm and MLP parameters.',
        'hypothesis': 'A small native path can predict the selected parallel head and downstream MLP states.',
        'alternatives': 'Distributed source lookup, incidental routing, low H36 values carrying the answer, or downstream signals dominated by fixed other-head contributions.',
        'criteria': 'Stop unless every pilot cell reaches 90%. On physical-first confirmation rows require ≥90% five-record agreement and ≥20% improvement against both controls for centered log-mass, O-write and MLP errors, with positive bootstrap lower bounds by family.',
        'domains': ['KN', 'RS'], 'family_ids': ['CLT-FIRST-RECORD-PATH', 'SOURCE-BINDING-20261110'],
        'dataset': {'id': 'first-record-fit-pilot-confirm-v1', 'revision': 'native-path-prediction-v1',
                    'content_sha256': file_sha, 'questions_file_sha256': file_sha,
                    'split_manifest_sha256': sha(MATERIAL/'manifest.json'), 'case_count': len(rows),
                    'exposure': 'Exposed discovery-only fit and independent pilot and confirmation families.'},
        'entrypoint_sha256': entry, 'execution_args': [],
        'acquisition': {'model_id': 'Qwen3-14B', 'model_snapshot': manifest['model_snapshot'],
                        'tokenizer_sha256': manifest['tokenizer_sha256'], 'entrypoint_sha256': entry,
                        'questions_file_sha256': file_sha,
                        'inputs_sha256': digest([[r['split'], r['id'], r['tokens']] for r in rows]),
                        'code_sha256': digest(files), 'environment_sha256': digest(runtime_identity()),
                        'precision': 'BF16',
                        'seed': SEED, 'generation': {'script_args': [], 'one_step': True, 'sample': False},
                        'capture': {'layers': ['single data-selected block from 0–39, frozen before confirmation'],
                                    'readouts': ['last input token'],
                                    'boundary': 'selected scaled dot-product route and V → O reconstruction; original other-head inputs and residual preserved prior to same-block RMSNorm/SwiGLU'}},
        'analysis': {'method_version': 'clt_first_record_predict.v1', 'code_sha256': digest(files),
                     'scoring_sha256': digest(selection), 'selection': selection, 'required_artifact_roles': sorted(CORE_ROLES | {'evidence'})},
    }
    validate_plan(plan)
    save(PLAN, plan)


def pick_head(record, metadata):
    if sha(record/'server-receipt.json') != metadata['discovery_receipt_sha256']:
        raise ValueError('发现回执已改变')
    receipt = read(record/'server-receipt.json')
    work = Path(read(record/'review.json')['scratch'])
    proved = {a['name']:a for a in receipt['artifacts'] if a['availability']=='uploaded'}
    cases_path=work/'case_results.jsonl'
    if sha(cases_path)!=proved[cases_path.name]['sha256']:
        raise ValueError('逐题发现记录与中心回执不符')
    with cases_path.open(encoding='utf-8') as stream:
        cases = [json.loads(line) for line in stream]
    by_shard = {}
    for case in cases:
        by_shard.setdefault(case['native_shard'], []).append(case)
    contrasts, rates = [], []
    for key in sorted(by_shard):
        path=work/key
        if Path(key).name!=key or sha(path)!=proved[key]['sha256']:
            raise ValueError('发现分片与中心回执不符')
        with np.load(path, allow_pickle=False) as data:
            mass = data['mass']
        if mass.shape[0] % 12 or mass.shape[1:] != (40, 40, 6):
            raise ValueError('发现分片形状不匹配')
        residual = [r['native_row'] for r in by_shard[key] if r['sample_slice']=='head36_residual']
        if residual:
            target = 3+((np.array(residual)%12)//3)%2
            rates.append(mass[residual,:,:,:5].argmax(-1)==target[:,None,None])
        ratios = np.log(np.maximum(mass[..., 4], 1e-9)) - np.log(np.maximum(mass[..., 3], 1e-9))
        ratios = ratios.reshape(-1, 2, 2, 3, 40, 40)
        swap = (ratios[:, :, 1] - ratios[:, :, 0]).mean(axis=1)
        contrasts.append(swap[:, 2] - (swap[:, 0] + swap[:, 1])*.5)
    x = np.concatenate(contrasts)
    if len(x) != 128:
        raise ValueError('发现分片未覆盖全部家族')
    success=np.concatenate(rates).mean(axis=0)
    mean = np.maximum(0,x.mean(axis=0))*success
    mean[26,36] = -np.inf
    b, h = map(int, np.unravel_index(mean.argmax(), mean.shape))
    if mean[b,h] <= 0:
        raise ValueError('发现集没有正向并行头候选')
    return b,h,{'paired_log_shift':float(x[:,b,h].mean()),'residual_source_agreement':float(success[b,h])}


def capture(rt, rows, block, head, progress, candidate=None):
    """Use the original layer and compute the candidate locally before observing its targets."""
    t = rt.t
    F = t.nn.functional
    owner = rt.block
    saved = {}

    def put(name, value):
        saved.setdefault(name, []).append(value.float().cpu().numpy())

    def hooks(number):
        layer = owner(number)
        if number != block:
            return layer
        att = layer.self_attn
        group = head//(rt.heads//rt.kv)
        H = slice(head*rt.hd, (head+1)*rt.hd)
        G = slice(group*rt.hd, (group+1)*rt.hd)
        W = {'q': att.q_proj.weight[H], 'k': att.k_proj.weight[G], 'v': att.v_proj.weight[G]}
        ctx = {'offset': 0}

        def residual(_, args):
            ctx['residual'] = args[0][:, -1]

        def prepare(_, args, kw):
            ctx.update(z=kw['hidden_states'], position=kw['position_embeddings'], mask=kw['attention_mask'][:, 0, -1])

        def project(key, col):
            def take(_, args, out):
                ctx[key] = out[..., col]
            return take

        def score(q, k, v, spans):
            q, k = rt.mod.apply_rotary_pos_emb(att.q_norm(q).unsqueeze(1), att.k_norm(k).unsqueeze(1), *ctx['position'])
            s = (q[:, 0, -1, None].float()*k[:, 0].float()).sum(-1)*att.scaling
            a = s.masked_fill(~ctx['mask'], float('-inf')).softmax(-1)
            return (a[:, None]*spans).sum(-1), (a.unsqueeze(-1)*v.float()).sum(1)

        layer.register_forward_pre_hook(residual)
        att.register_forward_pre_hook(prepare, with_kwargs=True)
        if candidate is None:
            def fit_state(_, args):
                z, mask = ctx['z'].float(), ctx['mask']
                kv = z.masked_fill(~mask.unsqueeze(-1), 0)
                put('last', z[:, -1])
                put('sum', kv.sum(1))
                put('sqsum', kv.square().sum(1))
                put('count', mask.sum(1))
                for key, weight in W.items():
                    if key not in saved:
                        put(key, weight.float().square().sum(0)[None])

            att.o_proj.register_forward_pre_hook(fit_state)
            layer.mlp.down_proj.register_forward_pre_hook(lambda _, args: put('units', args[0][:, -1]))
            return layer

        device = W['q'].device
        units = t.as_tensor(candidate['units'], device=device, dtype=t.long)
        mean = {key: t.as_tensor(candidate['mean_'+key], device=device, dtype=t.float32) for key in W}
        indices = {name: {key: t.as_tensor(candidate[f'{name}_{key}'], device=device, dtype=t.long) for key in W}
                   for name in ('selected', 'random')}
        offsets = {key: F.linear(mean[key], w.float()) for key,w in W.items()}
        # The target uses the original same-block MLP. Prediction sees the actual residual
        # and other head inputs at this boundary, plus only the approximated selected head.
        for key, sites in [('q', H), ('k', G), ('v', G)]:
            getattr(att, f'{key}_proj').register_forward_hook(project(key, sites))

        def account(_, args):
            raw = args[0][:, -1]
            z = ctx['z']
            n, length, _ = z.shape
            start = ctx['offset']
            ctx['offset'] += n
            spans = t.zeros((n, 5, length), device=z.device, dtype=t.bool)
            for j, row in enumerate(rows[start:start+n]):
                pad = length-len(row['tokens'])
                for i, tokens in enumerate(row['records']):
                    spans[j, i, [pad+k for k in tokens]] = True
            original_mass, original_value = score(ctx['q'], ctx['k'], ctx['v'], spans)
            put('mass_native', original_mass)
            put('native_head_scale_error', (original_value-raw[:, H].float()).abs().amax(1)/(1+raw[:, H].float().abs().amax(1)))
            native_write = F.linear(raw[:, H].float(), att.o_proj.weight[:, H].float())
            put('write_native_power', native_write.square().mean(1))
            for name in ('selected', 'random', 'mean'):
                projected = {}
                for key, w in W.items():
                    base = offsets[key]
                    if name == 'mean':
                        value = base.expand(n, length, -1)
                    else:
                        chosen = indices[name][key]
                        value = base + F.linear(z[..., chosen].float()-mean[key][chosen], w[:, chosen].float())
                    projected[key] = value.to(z.dtype)
                pm, pv = score(projected['q'], projected['k'], projected['v'], spans)
                put('mass_'+name, pm)
                put('write_error_'+name, (F.linear(pv, att.o_proj.weight[:, H].float())-native_write).square().mean(1))
                if name=='selected':
                    q_mass, _ = score(projected['q'],ctx['k'],ctx['v'],spans)
                    k_mass, _ = score(ctx['q'],projected['k'],ctx['v'],spans)
                    _, v_only = score(ctx['q'],ctx['k'],projected['v'],spans)
                    put('mass_q_only',q_mass)
                    put('mass_k_only',k_mass)
                    put('write_error_v_only',(F.linear(v_only,att.o_proj.weight[:,H].float())-native_write).square().mean(1))
                changed = raw.clone()
                changed[:, H] = pv.to(raw.dtype)
                x = layer.post_attention_layernorm(ctx['residual']+F.linear(changed, att.o_proj.weight))
                gates = layer.mlp.act_fn(F.linear(x, layer.mlp.gate_proj.weight[units]))
                values = gates*F.linear(x, layer.mlp.up_proj.weight[units])
                put('mlp_'+name, values)

        att.o_proj.register_forward_pre_hook(account)
        layer.mlp.down_proj.register_forward_pre_hook(lambda _, args: put('mlp_native', args[0][:, -1, units]))
        return layer

    def mark(number, layer, x, y, batch, width, offset=0, **_):
        if number >= 0 and offset+len(batch) == len(rows) and (number+1) % 5 == 0:
            progress(number+1)

    rt.block = hooks
    try:
        _, logits = rt.forward(rows, mark)
    finally:
        rt.block = owner
    scores = []
    for row, logit in zip(rows, logits, strict=True):
        values = logit[row['choice_ids']]
        output = int(logit.argmax())
        scores.append({'id': row['id'], 'forced': row['target_record'] == int(values.argmax()),
                       'output': output, 'correct': output == row['gold_id'], 'logits': values.cpu().tolist()})
    return {key: np.concatenate(parts) for key,parts in saved.items()}, scores


def fit(groups, rt, block):
    data = {k: np.concatenate([d[k] for d in groups]) for k in groups[0]}
    last, pop = data['last'].astype(np.float64), data['count'].sum()
    mean = {'q': last.mean(0), 'k': data['sum'].astype(np.float64).sum(0)/pop}
    var = {'q': last.var(0), 'k': data['sqsum'].astype(np.float64).sum(0)/pop-mean['k']**2}
    mean['v'],var['v'] = mean['k'],var['k']
    rng = np.random.default_rng(SEED)
    result = {}
    for key in ('q','k','v'):
        result['mean_'+key] = mean[key]
        result['selected_'+key] = np.argsort(-(var[key]*data[key][0]), kind='stable')[:BUDGET]
        result['random_'+key] = rng.choice(rt.dim, BUDGET, replace=False)
    down = rt.weight(f'model.layers.{block}.mlp.down_proj.weight', 'cpu').float().numpy()
    result['units'] = np.argsort(-(data['units'].var(0)*np.square(down).sum(0)), kind='stable')[:UNITS]
    return result


def native_scalars(rt, block, head, candidate):
    group=head//(rt.heads//rt.kv)
    scope=f'model.layers.{block}.'
    out={}
    for key,row in [('q',head*rt.hd),('k',group*rt.hd),('v',group*rt.hd)]:
        col=int(candidate['selected_'+key][0])
        w=rt.weight(scope+f'self_attn.{key}_proj.weight','cpu')
        output=row+int(w[row:row+rt.hd,col].abs().argmax())
        out[key]={'row':output,'col':col,'weight':float(w[output,col])}
    col=head*rt.hd
    w=rt.weight(scope+'self_attn.o_proj.weight','cpu')
    output=int(w[:,col].abs().argmax())
    out['o']={'row':output,'col':col,'weight':float(w[output,col])}
    neuron=int(candidate['units'][0])
    for key in ('gate','up'):
        w=rt.weight(scope+f'mlp.{key}_proj.weight','cpu')
        col=int(w[neuron].abs().argmax())
        out[key]={'row':neuron,'col':col,'weight':float(w[neuron,col])}
    w=rt.weight(scope+'mlp.down_proj.weight','cpu')
    output=int(w[:,neuron].abs().argmax())
    out['down']={'row':output,'col':neuron,'weight':float(w[output,neuron])}
    out['mlp_unit']=neuron
    return out


def behavior(rows, scores):
    return {f'{variant}/{cell}': float(np.mean([b['correct'] for r,b in zip(rows,scores,strict=True)
                if r['frame']==variant and r['cell']==cell])) for variant in sorted({r['frame'] for r in rows}) for cell in range(12)}


def evaluate(rows, data):
    mask = np.array([r['rule']==2 for r in rows])
    actual = data['mass_native'][mask]

    def centered(x):
        logged = np.log(np.maximum(x, 1e-9))
        return logged-logged.mean(axis=1,keepdims=True)

    ground = centered(actual)
    scores, errors = {}, {}
    for name in ('selected','random','mean'):
        mass = data['mass_'+name][mask]
        terms = {'source_log_mass': np.square(centered(mass)-ground).mean(1),
                 'vo_write': data['write_error_'+name][mask],
                 'mlp_units': np.square(data['mlp_'+name][mask]-data['mlp_native'][mask]).mean(1)}
        errors[name] = terms
        scores[name] = {'agreement': float(np.mean(mass.argmax(1)==actual.argmax(1))),
                        **{key:float(value.mean()) for key,value in terms.items()}}
    contrasts = {}
    rng = np.random.default_rng(SEED+1)
    n = len(actual)//4
    bootstrap = rng.integers(n,size=(10000,n))
    for base in ('random','mean'):
        for term, chosen in errors['selected'].items():
            other = errors[base][term]
            family = (other-chosen).reshape(n,4).mean(1)
            gain = float(1-chosen.mean()/max(other.mean(),1e-20))
            low = float(np.quantile(family[bootstrap].mean(1),.0125))
            contrasts[f'{base}/{term}'] = {'gain':gain,'lower':low,'passed':gain>=GAIN_GATE and low>0}
    frames=np.array([r['frame'] for r in rows])[mask]
    agrees=data['mass_selected'][mask].argmax(1)==actual.argmax(1)
    by_frame={f:{'agreement':float(agrees[frames==f].mean()),'rows':int((frames==f).sum())}
              for f in sorted(set(frames))}
    branches={key+'_only_agreement':float(np.mean(data['mass_'+key+'_only'][mask].argmax(1)==actual.argmax(1)))
              for key in ('q','k')}
    branches['v_only_write_mse']=float(data['write_error_v_only'][mask].mean())
    return {'scores':scores,'contrasts':contrasts,'by_frame':by_frame,'diagnostics':branches,
            'gate_passed':scores['selected']['agreement']>=SOURCE_GATE and all(x['passed'] for x in contrasts.values())}


def registered_context(identity=dependencies, entrypoint=None):
    work = Path(os.environ.get('AI2050_TEST_WORK_DIR',''))
    questions = Path(os.environ.get('AI2050_TEST_QUESTIONS',''))
    if not work.is_absolute() or not work.is_dir() or not questions.is_file():
        raise ValueError('须通过正式归档登记启动')
    record = questions.parent
    plan, preflight, review = [read(record/name) for name in ('protocol.json','preflight.json','review.json')]
    if (not preflight.get('preflight_id') or preflight.get('plan_sha256') != digest(plan)
            or not review.get('server_run_id') or Path(review['scratch']).resolve()!=work.resolve()
            or plan['entrypoint_sha256']!=sha(entrypoint or Path(__file__)) or plan['dataset']['questions_file_sha256']!=sha(questions)
            or plan['acquisition']['code_sha256']!=digest(identity())):
        raise ValueError('预检、运行或冻结材料身份不符')
    return work,questions,plan,review['server_run_id']


def execute():
    from shared.dist_protocol.mechanism_trace import GpuLease
    from shared.dist_protocol.natural_register_runner import snapshot
    from source_binding_design import MODELS
    from source_binding_resources import resources
    from source_binding_runtime import Runtime

    work, questions, plan, run_id = registered_context()
    if snapshot(MODELS['qwen14']) != plan['acquisition']['model_snapshot']:
        raise ValueError('模型快照不匹配')
    rows = read(questions)
    metadata = plan['analysis']['selection']
    block, head, selectivity = pick_head(Path(metadata['reference_record']), metadata)
    batches = {k:[r for r in rows if r['split']==k] for k in ('fit','pilot','confirm')}
    rt = Runtime('qwen14')
    t = rt.t
    t.backends.cuda.matmul.allow_tf32 = False
    started, completed = time.monotonic(), 0
    n_blocks = 40*sum((len(b)+BATCH-1)//BATCH for b in batches.values())
    record = {'run_id':run_id,'block':block,'head':head,'discovery_selectivity':selectivity,
              'source_run':metadata['reference_run']}

    def collect(name, candidate=None):
        nonlocal completed
        arrays, scores = [], []
        population = batches[name]
        for lo in range(0,len(population),BATCH):
            def progress(blocks):
                done = completed+blocks
                elapsed = time.monotonic()-started
                print(json.dumps({'stage':name,'completed':done,'total':n_blocks,'elapsed':elapsed,
                                  'eta':elapsed*(n_blocks-done)/done if done else None,
                                  'cuda_peak':t.cuda.max_memory_allocated(),**resources()}),flush=True)
            progress(0)
            d,s = capture(rt,population[lo:lo+BATCH],block,head,progress,candidate)
            arrays.append(d)
            scores.extend(s)
            completed += 40
        return arrays,scores

    entries = []
    with GpuLease(), t.inference_mode(), (work/'case_results.jsonl').open('x',encoding='utf-8',buffering=1) as stream:
        fitted, fit_scores = collect('fit')
        for item in fit_scores:
            stream.write(json.dumps(dict(item,split='fit',run_id=run_id))+'\n')
        candidate = fit(fitted,rt,block)
        record['sample_native_scalars'] = native_scalars(rt,block,head,candidate)
        sealed = work/'candidate.npz'
        check_write(sealed,2**20,operation='Frozen native predictor')
        with sealed.open('xb') as file:
            np.savez(file,**candidate)
        entries.append({'path':sealed.name,'role':'evidence','retention':'important'})
        record['candidate_sha256'] = sha(sealed)
        for name in ('pilot','confirm'):
            arrays, scores = collect(name,candidate)
            combined = {key:np.concatenate([d[key] for d in arrays]) for key in arrays[0]}
            for item in scores:
                stream.write(json.dumps(dict(item,split=name,run_id=run_id))+'\n')
            accuracy = behavior(batches[name],scores)
            err = float(combined['native_head_scale_error'].max())
            record[name] = {'cells':accuracy,'native_error':err,'passed':min(accuracy.values())>=.90 and err<=CAP}
            file = work/f'{name}-path.npz'
            check_write(file, 32*2**20,operation='First-record path trace')
            with file.open('xb') as f:
                np.savez(f, **combined)
            entries.append({'path':file.name,'role':'evidence','retention':'important'})
            save(work/'artifacts.json', {'schema':'research_test_files.v1','files':entries})
            if not record[name]['passed']:
                break
            if name=='confirm':
                record['prediction']=evaluate(batches[name],combined)
                actual=combined['mass_native'].argmax(1)
                record['readout']={str(rule):float(np.mean([hits==row['target_record']
                       for hits,row in zip(actual,batches[name],strict=True) if row['rule']==rule])) for rule in range(3)}
    record['evidence'] = {'readable':record.get('readout'),
                          'predictable':record.get('prediction',{}).get('gate_passed',False),
                          'causal':'pending only if the prediction gate passed'}
    save(work/'conclusion.json',record)


if __name__ == '__main__':
    execute()
