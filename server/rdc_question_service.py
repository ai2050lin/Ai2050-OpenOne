"""Read-only natural-question atlas; no model, training, or CUDA request path."""
import gzip
import hashlib
import json
import struct
from functools import lru_cache
from pathlib import Path
import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from server.rdc_feature_service import ROOT, read, decode
from server.rdc_relation_service import field_response

BASE = ROOT/'tests/glm5/result/rdc_query_construction_20260913/phase2748'
PHYSICAL = Path('C:/AI2050-RDC-Archive/rdc_query_construction_20260913/phase2748')
MODELS = ('qwen4', 'qwen14', 'glm4')
router = APIRouter(prefix='/questions', tags=['Natural same-context question atlas'])
MODEL_FOLDERS={'qwen4':'qwen3-4b','qwen14':'Qwen3-14B','glm4':'glm4-9b-chat-hf'}


def guard():
    if BASE.resolve() != PHYSICAL.resolve():
        raise HTTPException(409, 'Registered Phase2748 storage identity changed')


def key(value):
    if value not in MODELS:
        raise HTTPException(422, 'Unknown registered model')
    guard()
    return value


@lru_cache(maxsize=256)
def verified_hash(path, size, mtime_ns, expected):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise HTTPException(409, 'Committed artifact SHA256 mismatch')
    return expected


def checked(ref):
    # References originate in a registered result, never in user-supplied paths.
    guard()
    path = ROOT/ref['path']
    if not path.resolve().is_relative_to(PHYSICAL.resolve()) or not path.is_file():
        raise HTTPException(409, 'Registered artifact unavailable or outside archive')
    stat = path.stat()
    verified_hash(str(path.resolve()), stat.st_size, stat.st_mtime_ns, ref['sha256'])
    return path


@lru_cache(maxsize=4)
def gzartifact(path, expected):
    return json.loads(gzip.decompress(checked({'path': path, 'sha256': expected}).read_bytes()))


def manifest():
    guard()
    return read(BASE/'material/manifest.json')


def confirmation_open():
    cert = read(BASE/'confirmation/freeze_certificate.json', {})
    if not cert.get('all_passed'):
        return False
    path = BASE/'material/manifest.json'
    stat = path.stat()
    verified_hash(str(path), stat.st_size, stat.st_mtime_ns, cert['material_manifest_sha256'])
    return True


def materials():
    ref = manifest()['rows']
    return gzartifact(ref['path'], ref['sha256'])


def tokens(model):
    ref = manifest()['model_token_files'][key(model)]
    return {r['question_id']: r for r in gzartifact(ref['path'], ref['sha256'])}


def source(question):
    row = next((r for r in materials() if r['question_id'] == question), None)
    if row is None:
        raise HTTPException(404, 'Unknown frozen question ID')
    if row['split'] == 'confirmation' and not confirmation_open():
        raise HTTPException(409, 'Confirmation remains sealed; no material or target exposure')
    return row


def committed(model, row):
    scope = 'confirmation' if row['split'] == 'confirmation' else 'nonconfirmation'
    folder = BASE/'native'/key(model)/scope
    group = read(folder/'groups'/(row['group_id']+'.json'), {})
    if not group:
        raise HTTPException(409, 'Complete four-question native group not committed')
    execution = read(folder/'execution.json', {})
    if not execution or group['execution'] != execution:
        raise HTTPException(409, 'Native execution receipt mismatch')
    questions = {r['question_id']: r for r in group['questions']}
    if len(questions) != 4 or row['question_id'] not in questions:
        raise HTTPException(409, 'Incomplete native group')
    return group, questions[row['question_id']]


def array(ref, name):
    if name not in ref['arrays']:
        raise HTTPException(409, 'Requested field axis was not retained; no substitution')
    with np.load(checked(ref), allow_pickle=False) as z:
        value = z[name]
        return decode(value) if value.dtype == np.uint16 else value.astype(float)


def show(value, labels, view='raw', **extra):
    value = np.asarray(value, float)
    if value.ndim != 2 or len(labels) != value.shape[0]:
        raise HTTPException(409, 'Invalid retained field shape')
    if view == 'row_RMS':
        value = value/np.sqrt(np.mean(value**2, axis=-1, keepdims=True)).clip(1e-12)
    elif view != 'raw':
        raise HTTPException(422, 'Unknown numerical view')
    return field_response(value, labels, 0, value.shape[-1],
        view+'; all original coordinate indices retained; no Top-K or coordinate reordering.', **extra)


@router.get('/overview')
def overview():
    info = manifest()
    models = {}
    for model in MODELS:
        fit = read(BASE/'fit'/model/'result.json', {})
        paired = read(BASE/'fit'/model/'paired_analysis.json', {})
        behavior = read(BASE/'behavior'/model/'nonconfirmation/result.json', {})
        history_fit = read(BASE/'native_history_prediction'/model/'result.json', {})
        history_description = read(BASE/'native_history_prediction'/model/'descriptive_report_v2.json', {})
        geometry=read(BASE/'relation_geometry'/model/'result.json',{})
        geometry_qa=read(BASE/'relation_geometry'/model/'visual_QA.json',{})
        qa = read(BASE/'atlas'/model/'visual_QA_final.json', {})
        models[model] = {
            'progress': read(BASE/'native'/model/'nonconfirmation/progress.json', {}),
            'native_complete': read(BASE/'native'/model/'nonconfirmation/result.json', {}).get('all_passed', False),
            'fit': fit, 'behavior': behavior.get('summaries', []),
            'fit_summaries': paired.get('summaries', []),
            'primary_rule': fit.get('primary_rule'),
            'native_history_prediction': {
                'complete': history_fit.get('all_passed', False),
                'native_generated_tokens': history_fit.get('native_generated_tokens'),
                'summaries': [r for r in history_description.get('summaries', []) if r['split']=='diagnostic' and r['variant']==fit.get('primary_rule')],
                'paired_comparisons': [r for r in history_fit.get('paired_comparisons', []) if r['split']=='diagnostic'],
                'rules': [r for r in history_fit.get('records', []) if r['split']=='diagnostic'],
                'limits': history_fit.get('limits'),
            },
            'figures': read(BASE/'atlas'/model/'display_v2.json', {}).get('figures', []) if qa.get('all_passed') else [],
            'readout_complete': read(BASE/'fit'/model/'readout/nonconfirmation/result.json', {}).get('all_passed', False),
            'relation_geometry':{'complete':geometry.get('all_passed',False),'labels':geometry.get('labels',[]),
                'fixed_pairs':[r for r in geometry.get('fixed_display_pairs',[])if r['split']=='diagnostic'],
                'limits':geometry.get('limits'),
                'figures':read(BASE/'relation_geometry'/model/'display.json',{}).get('figures',[])if geometry_qa.get('all_passed')else[]},
        }
    return {'phase': 2748, 'models': models, 'contexts': info['contexts'], 'questions': info['questions'],
        'confirmation_open': confirmation_open(), 'training': read(BASE/'training/result.json', {}),
        'limits': 'Partial Phase. Native behavior, retrospective atlas, statistical prediction, training and own-history deployment are separate evidence. First token is often JSON format. No claim of solved language mechanism.'}


@router.get('/samples')
def samples(model: str='qwen4', split: str='diagnostic', cohort: str='all'):
    key(model)
    if split not in ('train', 'validation', 'diagnostic', 'confirmation') or cohort not in ('drop', 'quoref', 'all'):
        raise HTTPException(422, 'Unknown split/cohort')
    if split == 'confirmation' and not confirmation_open():
        raise HTTPException(409, 'Confirmation remains sealed')
    scope = 'confirmation' if split == 'confirmation' else 'nonconfirmation'
    captured = {p.stem for p in (BASE/'native'/model/scope/'groups').glob('*.json')}
    return [{k: r[k] for k in ('question_id', 'group_id', 'split', 'cohort', 'within_context_index', 'question')} |
        {'captured': r['group_id'] in captured} for r in sorted(materials(), key=lambda r:(r['cohort'], r['group_id'], r['within_context_index']))
        if r['split'] == split and (cohort == 'all' or r['cohort'] == cohort)]


@router.get('/analysis')
def analysis(kind: str='readout', model: str='qwen4', scope: str='nonconfirmation'):
    """Expose only completed, identity-checked analyses; never trigger a job."""
    key(model)
    if scope not in ('nonconfirmation', 'confirmation'):
        raise HTTPException(422, 'Unknown analysis scope')
    if scope == 'confirmation' and not confirmation_open():
        raise HTTPException(409, 'Confirmation analysis remains sealed')
    if kind == 'readout':
        folder = BASE/'readout_analysis'/model/scope
        execution = BASE/'readout_analysis/execution.json'
        dependencies = {'readout_result_sha256': BASE/'fit'/model/'readout'/scope/'result.json',
                        'selection_sha256': BASE/'fit'/model/'validation_selection.json'}
    elif kind == 'prospective':
        folder = BASE/'prospective_analysis'/model/('confirmation' if scope == 'confirmation' else 'diagnostic')
        execution = BASE/'prospective_analysis/execution.json'
        dependencies = {'fit_selection_sha256': BASE/'fit'/model/'validation_selection.json'}
    elif kind == 'learning':
        if model != 'qwen4':
            raise HTTPException(422, 'The six training runs are registered for qwen4 only')
        folder = BASE/'learning_analysis'/scope
        execution = BASE/'learning_analysis/execution.json'
        dependencies = {}
    elif kind == 'parameter':
        if model != 'qwen4' or scope != 'nonconfirmation':
            raise HTTPException(422, 'Parameter formation is a qwen4 training endpoint, not a confirmation outcome')
        folder = BASE/'parameter_formation'
        execution = folder/'execution.json'
        dependencies = {'original_manifest_sha256': BASE/'training/original_checkpoint_manifest.json'}
    elif kind == 'training_bridge':
        if model != 'qwen4' or scope != 'nonconfirmation':
            raise HTTPException(422, 'Zero-update bridge is a qwen4 validation-only diagnostic')
        folder = BASE/'training_bridge_baseline'
        execution = folder/'execution.json'
        dependencies = {'acquisition_sha256': folder/'acquisition.json',
                        'training_result_sha256': BASE/'training/result.json'}
    else:
        raise HTTPException(422, 'Unknown registered analysis')
    path = folder/'result.json'
    result = read(path, {})
    base = {'kind': kind, 'model': model, 'scope': scope}
    if not result:
        return {**base, 'complete': False, 'status': 'No completed analysis receipt; qualified or prepared code is not a result.'}
    if not result.get('all_passed'):
        raise HTTPException(409, 'Analysis receipt did not pass its checks')
    checked({'path': execution.relative_to(ROOT).as_posix(), 'sha256': result['execution_sha256']})
    for name, target in dependencies.items():
        checked({'path': target.relative_to(ROOT).as_posix(), 'sha256': result[name]})
    if kind == 'prospective':
        for entry in result['rules']:
            target = BASE/'prospective'/model/entry['variant']/folder.name/'result.json'
            checked({'path': target.relative_to(ROOT).as_posix(), 'sha256': entry['prospective_result_sha256']})
    elif kind == 'learning':
        for entry in result['runs']:
            target = BASE/'learned'/entry['run']/scope/'result.json'
            checked({'path': target.relative_to(ROOT).as_posix(), 'sha256': entry['learned_result_sha256']})
    elif kind == 'parameter':
        for run, digest in result['checkpoint_receipts'].items():
            target = BASE/'training'/run/'checkpoint96.json'
            checked({'path': target.relative_to(ROOT).as_posix(), 'sha256': digest})
    # Verify referenced metric matrices/output files before returning their summaries.
    def verify_references(value):
        if isinstance(value, dict):
            if 'path' in value and 'sha256' in value:
                checked(value)
            for name, entry in value.items():
                if name != 'source':
                    verify_references(entry)
        elif isinstance(value, list):
            for entry in value:
                verify_references(entry)
    verify_references({k: v for k, v in result.items() if k != 'source'})
    return {**base, 'complete': True, 'result': result,
            'receipt': {'path': path.relative_to(ROOT).as_posix(),
                        'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}}


@router.get('/sample')
def sample(model: str='qwen4', question: str=''):
    row = source(question)
    group, q = committed(model, row)
    group_rows = [r for r in materials() if r['group_id'] == row['group_id']]
    return {'material': row, 'tokens': tokens(model)[question], 'record': q,
        'context_field': group['context_field'], 'execution': group['execution'],
        'siblings': [{'material': r, 'record': next(x for x in group['questions'] if x['question_id'] == r['question_id'])} for r in group_rows],
        'scope': 'Only committed native B1 results. Teacher scoring supplies gold tokens; free generation consumes its own history. Conservative nonmatch is not automatically a reasoning error.'}


@router.get('/field')
def field(model: str='qwen4', question: str='', mode: str='first_hidden', view: str='raw',
          block: int=Query(12, ge=0, le=80), step: int=Query(0, ge=0, le=127), start: int=Query(0, ge=0)):
    row = source(question)
    group, q = committed(model, row)
    axes = 'Columns are native residual coordinates; H0 embedding, Hn after n blocks, postnorm separate. Each state precedes its output token.'
    extra = {'model': model, 'question_id': question, 'mode': mode}
    if mode == 'first_hidden':
        value = array(q['field'], 'hidden_BF16')
        value = np.concatenate([value, array(q['field'], 'postnorm_BF16')[None]], axis=0)
        labels = ['H'+str(i) for i in range(len(value)-1)]+['postnorm']
    elif mode == 'MLP':
        names = [f'block{block}_{name}_BF16' for name in ('gate', 'up', 'product')]
        value = np.stack([array(q['field'], name) for name in names])
        labels = [f'block{block} '+name for name in ('gate', 'up', 'product')]
        axes = 'Columns are all native MLP intermediate units, not residual coordinates or individual parameters. Rows gate/up/SiLU(gate)*up.'
    elif mode == 'attention':
        value = array(q['field'], 'block12_attention_BF16')
        labels = ['block12 attention head '+str(i) for i in range(len(value))]
        axes = 'Columns are every actual source token position including the question; rows are query attention heads. Not residual coordinates.'
        extra['input_ids'] = tokens(model)[question]['input_ids']
    elif mode == 'source_H12':
        full = array(group['context_field'], 'source_H12_BF16')
        if start >= len(full):
            raise HTTPException(422, 'Source position outside retained field')
        end = min(start+64, len(full))
        value = full[start:end]
        labels = ['context source token '+str(i) for i in range(start, end)]
        extra.update({'token_start': start, 'token_end': end, 'total_source_tokens': len(full)})
        axes = 'Explicit source-token page, up to64positions; all H12 residual coordinates preserved for each position. Other token pages remain queryable.'
    elif mode in ('history_postnorm', 'history_H12', 'history_read', 'history_all_hidden', 'teacher_postnorm'):
        history = q.get('teacher' if mode == 'teacher_postnorm' else 'history')
        if not history:
            raise HTTPException(409, 'This training question has no registered complete history')
        name = {'history_postnorm':'postnorm_BF16', 'history_H12':'H12_last_BF16',
            'history_read':'native_source_read_BF16', 'history_all_hidden':'all_hidden_BF16', 'teacher_postnorm':'postnorm_BF16'}[mode]
        value = array(history['field'], name)
        if mode == 'history_all_hidden':
            if step >= len(value):
                raise HTTPException(422, 'Generation step outside actual history')
            value = value[step]
            labels = ['H'+str(i) for i in range(len(value))]
            extra['generated_id'] = history['generated_ids'][step]
            extra['step'] = step
        else:
            labels = ['teacher step '+str(i) if mode == 'teacher_postnorm' else f"step {i}: emits ID {history['generated_ids'][i]}" for i in range(len(value))]
        axes = ('Teacher supplied history; not free generation. ' if mode == 'teacher_postnorm' else 'Actual own-token history, no gold answer supplied. ')+axes
    else:
        raise HTTPException(422, 'Unknown registered field mode')
    return show(value, labels, view, axes=axes, **extra)


@router.get('/atlas')
def atlas(model: str='qwen4', split: str='diagnostic', cohort: str='drop', channel: str='hidden',
          statistic: str='within_RMS', block: int=Query(12, ge=0, le=80)):
    key(model)
    if split not in ('train','validation','diagnostic') or cohort not in ('drop','quoref'):
        raise HTTPException(422, 'Unknown atlas split/cohort')
    result = read(BASE/'atlas'/model/'result.json', {})
    if not result.get('all_passed'):
        raise HTTPException(409, 'Full native coordinate aggregate not committed')
    if channel not in ('hidden','MLP') or statistic not in ('within_RMS','train_standardized_RMS','mean'):
        raise HTTPException(422, 'Unknown declared atlas statistic')
    suffix = {'within_RMS':'within_variance','train_standardized_RMS':'within_RMS_train_standardized','mean':'mean'}[statistic]
    name = 'H_and_postnorm' if channel == 'hidden' else 'selected_MLP_gate_up_product'
    value = array(result['fields'], '__'.join([split, cohort, name, suffix]))
    if statistic == 'within_RMS': value = np.sqrt(value)
    if channel == 'MLP':
        if block not in result['MLP_blocks']:
            raise HTTPException(409, 'MLP block not collected')
        value = value[result['MLP_blocks'].index(block)]
        labels = [f'block{block} '+name for name in result['MLP_channel_order']]
    else:
        labels = ['H'+str(i) for i in range(len(value)-1)]+['postnorm']
    return show(value, labels, model=model, split=split, cohort=cohort, statistic=statistic,
        axes='All native '+('residual coordinates' if channel=='hidden' else 'MLP units')+'; original order. Same-context question means are retrospective statistics, never predictor inputs.',
        statistic_definition=result['estimand'], normalization_definition='Raw mean/RMS, or RMS divided by own-cohort training coordinate SD with1e-8floor. No diagnostic fitting.')


@router.get('/figure')
def figure(model: str='qwen4', name: str='raw_RMS'):
    key(model)
    folder = BASE/'atlas'/model
    qa = read(folder/'visual_QA_final.json', {})
    if not qa.get('all_passed'):
        raise HTTPException(409, 'Actual figure QA not completed')
    display = checked({'path': (folder/'display_v2.json').relative_to(ROOT).as_posix(), 'sha256':qa['display_sha256']})
    item = next((r for r in read(display)['figures'] if r['name'] == name), None)
    if item is None or item['sha256'] not in qa['actually_viewed_and_reviewed_figures']:
        raise HTTPException(404, 'Unknown QA-verified figure')
    return FileResponse(checked(item), media_type='image/png')


@router.get('/relation-geometry')
def relation_geometry(model: str='qwen4',split: str='diagnostic',cohort: str='drop',metric: str='excess'):
    key(model)
    if split not in ('train','validation','diagnostic')or cohort not in ('drop','quoref')or metric not in ('excess','similarity','permutation_mean'):
        raise HTTPException(422,'Unknown nonconfirmation relation query')
    result=read(BASE/'relation_geometry'/model/'result.json',{})
    if not result.get('all_passed'):raise HTTPException(409,'Full relation analysis not completed')
    prefix=split+'__'+cohort
    with np.load(checked(result['field']),allow_pickle=False)as z:
        values=z[prefix+'__'+metric+'_mean'];counts=z[prefix+'__valid_contexts']
    return {'model':model,'split':split,'cohort':cohort,'metric':metric,'labels':result['labels'],
        'values':[[float(v)if n>0 else None for v,n in zip(row,count)]for row,count in zip(values,counts)],
        'valid_context_counts':counts.tolist(),'source_field':result['field'],
        'axes':'Rows and columns are observed spaces, not native hidden coordinates. Each4questionGram contracts all original coordinates. Null means zero-energy/undefined.',
        'limits':result['limits']}


@router.get('/relation-figure')
def relation_figure(model: str='qwen4',name: str='excess'):
    key(model);folder=BASE/'relation_geometry'/model
    qa=read(folder/'visual_QA.json',{})
    if not qa.get('all_passed'):raise HTTPException(409,'Actual relation figure QA not completed')
    display=checked({'path':(folder/'display.json').relative_to(ROOT).as_posix(),'sha256':qa['display_sha256']})
    entry=next((r for r in read(display)['figures']if r['name']==name),None)
    if entry is None or entry['sha256']not in qa['actually_viewed_and_reviewed_figures']:
        raise HTTPException(404,'Unknown QA-reviewed relation figure')
    return FileResponse(checked(entry),media_type='image/png')


@router.get('/source-coupling')
def source_coupling(model: str='qwen4', split: str='diagnostic', cohort: str='drop',
                    coordinate: str='permuted_minus_native_read_mean_square'):
    key(model)
    allowed = ('native_read_mean', 'permuted_read_mean', 'permuted_minus_native_read_mean_square')
    if split not in ('train', 'validation', 'diagnostic') or cohort not in ('drop', 'quoref') or coordinate not in allowed:
        raise HTTPException(422, 'Unknown nonconfirmation source-coupling query')
    path = BASE/'source_coupling'/model/'result.json'; result = read(path, {})
    if not result.get('all_passed'):
        return {'model': model, 'complete': False, 'status': '实际全来源分析尚未完成；不从部分原场推算结果。'}
    execution = BASE/'source_coupling/execution.json'; stat = execution.stat()
    verified_hash(str(execution), stat.st_size, stat.st_mtime_ns, result['execution_sha256'])
    native = BASE/'native'/model/'nonconfirmation/result.json'; stat = native.stat()
    verified_hash(str(native), stat.st_size, stat.st_mtime_ns, result['native_result_sha256'])
    if not read(native).get('all_passed'):
        raise HTTPException(409, 'Native source collection incomplete')
    prefix = split+'__'+cohort
    with np.load(checked(result['field']), allow_pickle=False) as z:
        heads = z[prefix+'__head_means']; counts = z[prefix+'__effective_source_valid_questions']
        values = z[prefix+'__'+coordinate]
    if heads.shape != (result['head_count'], len(result['head_columns'])) or values.shape != (result['native_width'],):
        raise HTTPException(409, 'Source-coupling axes mismatch')
    effective = result['head_columns'].index('passage_conditional_effective_source_fraction')
    table = [[None if i == effective and count == 0 else float(x) for i, x in enumerate(row)] for row,count in zip(heads, counts)]
    relative_path = BASE/'source_coupling'/model/'relative_strength.json'
    relative = read(relative_path, {}); relative_response = {'complete': False}
    if relative.get('all_passed'):
        if relative['source_coupling_result_sha256'] != hashlib.sha256(path.read_bytes()).hexdigest():
            raise HTTPException(409, 'Relative source-coupling identity mismatch')
        rel_execution = BASE/'source_coupling/relative_execution.json'; stat = rel_execution.stat()
        verified_hash(str(rel_execution), stat.st_size, stat.st_mtime_ns, relative['execution_sha256'])
        checked(relative['field'])
        for ref in relative['frozen_normalization_sources']: checked(ref)
        relative_response = {'complete': True, 'summaries': [r for r in relative['summaries'] if r['split'] == split and r['cohort'] == cohort],
            'receipt': {'path': relative_path.relative_to(ROOT).as_posix(), 'sha256': hashlib.sha256(relative_path.read_bytes()).hexdigest()},
            'field': relative['field'], 'protocol': read(rel_execution), 'limits': relative['limits']}
    return {'model': model, 'complete': True, 'split': split, 'cohort': cohort, 'coordinate': coordinate,
        'summary': next(r for r in result['summaries'] if r['split'] == split and r['cohort'] == cohort),
        'head_columns': result['head_columns'], 'head_values': table, 'effective_source_valid_questions': counts.tolist(),
        'field': show(values[None], [coordinate], axes='Every original post-O residual coordinate; native index unchanged. '+('Squared change, not signed response or semantic importance.' if coordinate.endswith('square') else 'Mean native BF16-decoded attention write; not semantic importance.')),
        'source_field': result['field'], 'receipt': {'path': path.relative_to(ROOT).as_posix(), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()},
        'protocol': read(execution), 'limits': result['limits'], 'relative_strength': relative_response}


def native_tensor(model,name):
    """Read-only original BF16 view; only fixed tensor names are built by callers."""
    folder=ROOT/'models/hf'/MODEL_FOLDERS[key(model)]
    mapping=read(folder/'model.safetensors.index.json')['weight_map']
    if name not in mapping:raise HTTPException(409,'Original tensor not registered')
    file=folder/mapping[name]
    with file.open('rb')as stream:
        length=struct.unpack('<Q',stream.read(8))[0]
        item=json.loads(stream.read(length))[name]
    if item['dtype']!='BF16':raise HTTPException(409,'Expected original native BF16 tensor')
    offset=8+length+item['data_offsets'][0]
    return np.memmap(file,mode='r',dtype='<u2',offset=offset,shape=tuple(item['shape'])),{
        'parameter':name,'shard':mapping[name],'data_offset':offset,'shape':item['shape'],'dtype':item['dtype']}


@router.get('/parameter-path')
def parameter_path(model: str='qwen4', question: str='', block: int=Query(12,ge=0,le=80),
                   unit: int=Query(0,ge=0), output: int=Query(0,ge=0)):
    row=source(question);group,_=committed(model,row)
    config=read(ROOT/'models/hf'/MODEL_FOLDERS[model]/'config.json')
    width,middle=config['hidden_size'],config['intermediate_size']
    if unit>=middle or output>=width or block>=config['num_hidden_layers']:
        raise HTTPException(422,'Index outside original parameter axes')
    # Qualify observed block before attempting any parameter lookup.
    field=group['questions'][0]['field']
    if f'block{block}_MLP_input_BF16'not in field['arrays']:
        raise HTTPException(409,'This block was not part of the frozen natural MLP observation')
    prefix=f'model.layers.{block}.mlp.'
    refs=[]
    if model=='glm4':
        joined,ref=native_tensor(model,prefix+'gate_up_proj.weight')
        if joined.shape!=(2*middle,width):raise HTTPException(409,'Packed native GLM MLP axes mismatch')
        gate=decode(joined[unit]).astype(float);up=decode(joined[middle+unit]).astype(float)
        refs.append({**ref,'gate_row':unit,'up_row':middle+unit})
    else:
        g,ref=native_tensor(model,prefix+'gate_proj.weight');refs.append({**ref,'row':unit})
        u,ref=native_tensor(model,prefix+'up_proj.weight');refs.append({**ref,'row':unit})
        gate=decode(g[unit]).astype(float);up=decode(u[unit]).astype(float)
    down,ref=native_tensor(model,prefix+'down_proj.weight');refs.append({**ref,'output_row':output,'unit_column':unit})
    downrow=decode(down[output]).astype(float);downcolumn=decode(down[:,unit]).astype(float)
    assert gate.shape==up.shape==downcolumn.shape==(width,)and downrow.shape==(middle,)
    inputs=[];writes=[];labels=[];records=[]
    material_index={r['question_id']:r for r in materials()if r['group_id']==row['group_id']}
    for q in group['questions']:
        reference=q['field'];x=array(reference,f'block{block}_MLP_input_BF16')
        nativegate=array(reference,f'block{block}_gate_BF16')
        nativeup=array(reference,f'block{block}_up_BF16')
        nativeproduct=array(reference,f'block{block}_product_BF16')
        nativewrite=array(reference,f'block{block}_MLP_write_BF16')
        gterms=gate*x;uterms=up*x;wterms=downrow*nativeproduct
        number=material_index[q['question_id']]['within_context_index']+1
        inputs.extend([gterms,uterms]);labels.extend([f'question{number} all gate input terms',f'question{number} all up input terms'])
        writes.append(wterms)
        gi,ui=float(nativegate[unit]),float(nativeup[unit])
        product64=gi*np.exp(-np.logaddexp(0.,-gi))*ui
        records.append({'question_id':q['question_id'],'question':material_index[q['question_id']]['question'],
            'gate_FP64_sum':float(gterms.sum()),'native_gate_BF16':gi,
            'up_FP64_sum':float(uterms.sum()),'native_up_BF16':ui,
            'product_from_native_gate_up_FP64':float(product64),'native_product_BF16':float(nativeproduct[unit]),
            'all_unit_write_FP64_sum':float(wterms.sum()),'native_MLP_write_BF16_at_output':float(nativewrite[output]),
            'write_FP64_minus_native_BF16':float(wterms.sum()-nativewrite[output]),
            'selected_unit_write_term_FP64':float(wterms[unit]),'source_field_sha256':reference['sha256']})
    factor_values=np.stack([gate,up,downcolumn])
    model_folder=ROOT/'models/hf'/MODEL_FOLDERS[model]
    return {'model':model,'group_id':row['group_id'],'selected_question_id':question,'block':block,'unit':unit,'output_coordinate':output,
        'native_parameter_addresses':refs,'records':records,
        'factor_decoded_FP64_sha256':hashlib.sha256(factor_values.tobytes()).hexdigest(),
        'current_checkpoint_config_sha256':hashlib.sha256((model_folder/'config.json').read_bytes()).hexdigest(),
        'current_checkpoint_index_sha256':hashlib.sha256((model_folder/'model.safetensors.index.json').read_bytes()).hexdigest(),
        'factors':show(factor_values,['All gate-read weights','All up-read weights','All down-write weights'],axes='Every native residual coordinate; same fixed parameters across all four questions.'),
        'input_terms':show(np.stack(inputs),labels,axes='All input-coordinate products Wgate[k,i]*x[i] and Wup[k,i]*x[i], computed in FP64 from original BF16 words.'),
        'write_terms':show(np.stack(writes),[f"question{material_index[q['question_id']]['within_context_index']+1} all unit write terms"for q in group['questions']],axes='Every native MLP unit contribution Wdown[j,k]*native_product[k], no selected-unit truncation of the sum.'),
        'arithmetic':'CPU FP64 descriptive contraction of actual BF16 parameter/state words. Native BF16 matrix multiplication, SiLU and rounding differ; both values and their discrepancy shown, not claimed bit-identical replay.',
        'scope':'Real source/state/parameter addresses and complete summands. A coordinate or MLP unit is not assigned a semantic label. This architecture-level path does not establish uniqueness, causality of a language feature or original pretraining formation.'}
