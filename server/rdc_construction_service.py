"""Read-only current construction atlas. No CUDA or inference in request paths."""
import gzip, importlib.util, json, sys, zipfile
from functools import lru_cache
from pathlib import Path
import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from server.rdc_feature_service import ROOT, read, decode
from server.rdc_relation_service import field_response
from server.rdc_joint_service import npz_headers

BASE = ROOT/'tests/glm5/result/rdc_query_construction_20260913'
router = APIRouter(prefix='/api/rdc-construction', tags=['RDC contextual query construction'])
MODELS = ['qwen4', 'qwen14', 'glm4']


def model_key(key):
    if key not in MODELS:
        raise HTTPException(422, 'Unknown registered model')
    return key


@lru_cache(maxsize=1)
def material():
    return json.loads(gzip.decompress((BASE/'material.json.gz').read_bytes()))


def rows(model):
    return material()['models'][model_key(model)]['rows']


def source(model, sid):
    row = next((r for r in rows(model) if r['sample_id'] == sid), None)
    if row is None:
        raise HTTPException(404, 'Unknown original sample ID')
    return row


def archive(path):
    if '\\' in path or '..' in path.split('/'):
        raise HTTPException(404, 'Outside registered archive')
    if path.startswith('phase2747/field_store/'):
        return formation_archive(path)
    file = BASE/path
    inside = file.resolve().is_relative_to(BASE.resolve())
    overflow_prefix = BASE/'phase2746/field_store'
    overflow_root = Path('C:/AI2050-RDC-Archive/rdc_query_construction_20260913/phase2746_fields')
    registered_overflow = (file.is_relative_to(overflow_prefix) and overflow_prefix.is_junction()
        and overflow_prefix.resolve() == overflow_root.resolve() and file.resolve().is_relative_to(overflow_root.resolve()))
    if (not inside and not registered_overflow) or not file.is_file() or file.suffix != '.npz' or '.tmp.' in file.name:
        raise HTTPException(404, 'Uncommitted or unknown numerical archive')
    if registered_overflow:
        relative = file.relative_to(overflow_prefix)
        if len(relative.parts) != 2:
            raise HTTPException(404, 'Unknown registered runtime field category')
        category=relative.parts[0]
        if category in ['pilot','main']:
            commit = BASE/'phase2746/runtime'/category/'commits'/(file.stem+'.json')
        elif category=='differential':
            commit=BASE/'phase2746/differential/commits'/(file.stem+'.json')
        elif category=='differential_derivatives':
            pieces=file.stem.split('_start')
            if len(pieces)!=2 or pieces[0] not in ['pilot','main'] or pieces[1] not in ['12','24','35']:
                raise HTTPException(404,'Unknown differential archive')
            commit=BASE/'phase2746/differential'/pieces[0]/('start_'+pieces[1]+'.json')
        elif category=='runtime_reuse':
            commit=BASE/'phase2746/runtime_reuse/commits'/(file.stem+'.json')
        elif category=='history_features' and file.stem=='points':
            commit=BASE/'phase2746/history_prediction/features/result.json'
        elif category=='history_fit' and file.stem in ['native_history','source_value_permuted']:
            commit=BASE/'phase2746/history_prediction/fit'/(file.stem+'.json')
        elif category=='history_validation' and file.stem=='selected':
            commit=BASE/'phase2746/history_prediction/validation/selected.json'
        elif category=='history_test' and file.stem=='heldout':
            commit=BASE/'phase2746/history_prediction/test/heldout.json'
        elif category=='history_deployment' and file.stem=='coefficients':
            commit=BASE/'phase2746/history_prediction/deployment/coefficients.json'
        elif category in ['history_confirmation_pilot','history_confirmation_main']:
            mode=category.removeprefix('history_confirmation_')
            commit=BASE/'phase2746/history_prediction/confirmation/native'/mode/'commits'/(file.stem+'.json')
        elif category=='history_confirmation_prediction' and file.stem=='points':
            commit=BASE/'phase2746/history_prediction/confirmation/prediction/result.json'
        elif category in ['history_autonomous_pilot','history_autonomous_main']:
            mode=category.removeprefix('history_autonomous_')
            commit=BASE/'phase2746/history_prediction/confirmation/autonomous'/mode/'commits'/(file.stem+'.json')
        else:
            raise HTTPException(404,'Unknown registered runtime field category')
        if not commit.exists() or read(commit, {}).get('field_path') != file.relative_to(BASE).as_posix():
            raise HTTPException(409, 'Runtime source field has not been committed')
    if 'fields' in file.parts and file.parent.parent.name in MODELS:
        commit = file.parent.parent/'commits'/(file.stem+'.json')
        if not commit.exists():
            raise HTTPException(409, 'Native row has not been committed')
    return file


def fields(model, sid):
    source(model, sid)
    return archive(f'capture/{model}/fields/{sid}.npz')


def show(value, labels, view='raw', **extra):
    value = np.asarray(value, float)
    if view == 'RMS':
        value = value/np.sqrt(np.mean(value*value, -1, keepdims=True)).clip(1e-12)
    elif view != 'raw':
        raise HTTPException(422, 'Unknown view')
    return field_response(value, labels, 0, value.shape[-1],
        view+'; complete original index order; no Top-K or component truncation.', **extra)


def native_state(z, boundary):
    if str(boundary) == 'postnorm' or str(boundary) == '-1':
        return decode(z['postnorm'])
    if not str(boundary).isdigit() or int(boundary) > 100:
        raise HTTPException(422, 'Invalid native boundary')
    ix = z['query_layer_indices'].tolist()
    if int(boundary) not in ix:
        raise HTTPException(409, 'This boundary was not retained for this row; no interpolated substitute')
    return decode(z['query_selected_states'][ix.index(int(boundary))])


def formation_archive(path):
    prefix=BASE/'phase2747/field_store'
    physical=Path('C:/AI2050-RDC-Archive/rdc_query_construction_20260913/phase2747_fields')
    file=BASE/path
    if (not file.is_relative_to(prefix) or not prefix.is_junction() or prefix.resolve()!=physical.resolve()
        or not file.resolve().is_relative_to(physical.resolve()) or not file.is_file() or file.suffix!='.npz'):
        raise HTTPException(404,'Outside registered formation archive')
    relative=file.relative_to(prefix)
    parts=relative.parts
    protocol=read(BASE/'phase2747/training/protocol.json',{})
    conditions=protocol.get('conditions',[])
    runs={c+'_'+str(s) for c in conditions for s in protocol.get('seeds',[])}
    radius_names={r['name'] for r in read(BASE/'phase2747/radius/protocol.json',{}).get('variants',[])}
    calibration_names={'native','bridge'}|{r+'_'+p for r in runs for p in ['FP32_bridge','native_BF16']}|radius_names
    transfer_directions={'python_to_en','en_to_python','zh_to_en','en_reordered_to_en'}
    allowed=(parts==('material','vocabulary.npz')
        or (len(parts)==3 and parts[0]=='training' and parts[1] in runs|{'baseline','draws','initial_gradients'})
        or (len(parts)==3 and parts[0]=='training_analysis' and parts[1] in runs)
        or (len(parts)==2 and parts[0]=='gradient' and file.stem in conditions)
        or (len(parts)==2 and parts[0]=='radius' and file.stem in radius_names)
        or (len(parts)==2 and parts[0]=='calibration' and file.stem in calibration_names)
        or (len(parts)==2 and parts[0]=='figures' and file.stem in {'training_full_coordinate_changes','parameter_history_complete_coordinates'})
        or (len(parts)==3 and parts[0] in {'transfer_prediction','transfer_readout'} and parts[1] in transfer_directions))
    if parts[0]=='own_history' and len(parts) in {4,5}:
        permitted=parts[1] in MODELS and (parts[2]=='native' or parts[1]=='qwen4' and parts[2] in runs)
        own_file=BASE/'phase2747/own_history/material.json.gz'
        own=json.loads(gzip.decompress(own_file.read_bytes())) if own_file.exists() else {}
        allowed=permitted and (len(parts)==4 or parts[3]=='pilot') and file.stem in {r['sample_id'] for r in own.get(parts[1],[])}
    if parts[0]=='program_own_history' and len(parts) in {3,4}:
        follow=json.loads(gzip.decompress((BASE/'phase2747/followup/material.json.gz').read_bytes()))
        allowed=parts[1] in {'native','code_identity','mapped_code','shuffled_map','mapped_digit_bias','mapped_letter_bias'} and (len(parts)==3 or parts[2]=='pilot') and file.stem in {r['sample_id'] for r in follow['program_own_targets']}
    if parts[0]=='parameter_propagation':
        spec=read(BASE/'phase2747/parameter_propagation/protocol.json',{})
        samples=set(spec.get('rows',[]))
        if len(parts)==4 and parts[1] in {'native','native_pilot'}:
            allowed=parts[2]=='baseline' and file.stem in samples
        elif len(parts)==5 and parts[1] in {'native','native_pilot'}:
            allowed=parts[2] in runs and parts[3] in {'s0p1','s0p3','s1p0'} and file.stem in samples
        elif len(parts)==4 and parts[1] in {'smooth','smooth_pilot'}:
            allowed=(parts[2]=='layers' and file.stem in {f'block_{i:02d}' for i in range(16,36)}) or (parts[2] in {'endpoints','adjoint'} and file.stem in samples|{'first_declared_sample_complete_parameter_VJP'})
    if not allowed:raise HTTPException(404,'Unknown formation archive category')
    commit=BASE/'phase2747'/relative.parent/'commits'/(file.stem+'.json')
    metadata=read(commit,{})
    if metadata.get('field_path')!=file.relative_to(ROOT).as_posix():
        raise HTTPException(409,'Formation field has no exact committed source')
    if Path(metadata.get('physical_path','')).resolve()!=file.resolve():
        raise HTTPException(409,'Formation physical source mismatch')
    return file


@router.get('/formation-progress')
def formation_progress():
    directory=BASE/'phase2747'
    protocol=read(directory/'training/protocol.json',{})
    runs=[]
    for seed in protocol.get('seeds',[]):
        for condition in protocol.get('conditions',[]):
            name=condition+'_'+str(seed)
            result=read(directory/'training'/name/'result.json',{})
            progress=result or read(directory/'training'/name/'progress.json',{})
            runs.append({'run':name,'seed':seed,'condition':condition,'complete':bool(result.get('all_passed')),
                'latest_committed_step':max([c['step'] for c in progress.get('checkpoints',[])],default=0),
                'checkpoints':[{k:c.get(k) for k in ['step','delta_FP32_L2','parameter_reconstruction_audit']} for c in progress.get('checkpoints',[])],
                'distinct_drawn_examples':result.get('distinct_drawn_examples'),
                'deployed_BF16_delta_L2':result.get('deployed_BF16_delta_L2')})
    def completed(relative):return bool(read(directory/relative,{}).get('all_passed'))
    scoring={(r['model'],r['variant']):r.get('scoring_audit')
             for r in read(directory/'own_history/analysis/result.json',{}).get('reports',[])}
    language_review=read(directory/'own_history/terminal_review/result.json',{})
    own_instances=[(m,'native') for m in MODELS]+[('qwen4',r['run']) for r in runs]
    tasks=[{'name':'真实参数形成、完整梯度与检查点分析','complete':all(completed(p) for p in ['training/result.json','training_analysis/result.json','gradient/result.json']),
            'committed':sum(r['complete'] for r in runs),'total':6},
        {'name':'两精度实际半径与方向对照','complete':completed('radius/result.json') and completed('radius_analysis/result.json'),
            'committed':read(directory/'radius/progress.json',{}).get('completed',0),'total':40},
        {'name':'词频、温度及总浓度对照','complete':completed('calibration/result.json'),
            'committed':read(directory/'calibration/progress.json',{}).get('complete',0),'total':54},
        {'name':'全部前缀位置的参数传播与伴随核对','complete':completed('parameter_propagation/native/result.json') and completed('parameter_propagation/smooth/result.json'),
            'committed':int(completed('parameter_propagation/native/result.json'))+int(completed('parameter_propagation/smooth/result.json')),'total':2},
        {'name':'文字代码映射、无gold读出与自身历史','complete':completed('transfer/readout_result.json') and completed('program_own_history/result.json'),
            'committed':read(directory/'transfer/readout_progress.json',{}).get('completed',0),'total':256},
        {'name':'三个原生模型及六训练部署的较长自身历史','complete':all(completed(f'own_history/{m}/{v}/result.json') for m,v in own_instances),
            'committed':sum(completed(f'own_history/{m}/{v}/result.json') for m,v in own_instances),'total':9},
        {'name':'三图谱、理论、客户端与完整证据审计','complete':completed('delivery_manifest.json'),
            'committed':int(completed('delivery_manifest.json')),'total':1}]
    return {'material':read(directory/'material/protocol.json',{}),'protocol':protocol,
        'pilot':read(directory/'training/pilot/result.json',{}),'runs':runs,
        'recovery':read(directory/'training/persistence_recovery.json',{}),
        'analysis':read(directory/'training_analysis/result.json',{}) or read(directory/'training_analysis/progress.json',{}),
        'result':read(directory/'training/result.json',{}),
        'followup':read(directory/'followup/protocol.json',{}),
        'tasks':tasks,'gradient':read(directory/'gradient/result.json',{}),
        'figures':formation_figures(),
        'program_own':{'complete':completed('program_own_history/result.json'),
            'progress':read(directory/'program_own_history/progress.json',{}),
            'summary':read(directory/'program_own_history/result.json',{}).get('summary',[]),
            'terminal_review':read(directory/'program_own_history/terminal_review/result.json',{})},
        'own_runs':[{'model':m,'variant':v,'complete':completed(f'own_history/{m}/{v}/result.json'),
            'progress':read(directory/'own_history'/m/v/'progress.json',{}),
            'wave_progress':read(directory/'own_history'/m/v/'wave_progress.json',{}),
            'scoring_audit':scoring.get((m,v)),
            'terminal_review':language_review.get('summary') if (m,v)==('glm4','native') else None,
            'summary':read(directory/'own_history'/m/v/'result.json',{}).get('summary',[])} for m,v in own_instances],
        'scope':'Only committed checkpoints shown. A completed training run is not the completed Phase or solved language mechanism.'}


def formation_figures():
    root=BASE/'phase2747/figures'
    indices=[]
    for pointer in ['current_training_figures.json','current_probability_figures.json','current_history_figures.json']:
        current=read(root/pointer,{})
        if current.get('index'):
            path=BASE/current['index']
            if path.resolve().is_relative_to(root.resolve()):indices.append(path)
    indices.append(root/'propagation_v1/index.json')
    return [figure for path in indices if read(path.parent/'visual_review.json',{}).get('all_passed')
        for figure in read(path,{}).get('figures',[])]


def formation_npy(file,key,index=None):
    # Read only the requested sample/step from a committed ZIP member.
    with zipfile.ZipFile(file) as packet:
        if key+'.npy' not in packet.namelist():raise HTTPException(409,'This axis was not collected')
        with packet.open(key+'.npy') as stream:
            version=np.lib.format.read_magic(stream)
            if version==(1,0):shape,fortran,dtype=np.lib.format.read_array_header_1_0(stream)
            elif version==(2,0):shape,fortran,dtype=np.lib.format.read_array_header_2_0(stream)
            else:raise HTTPException(409,'Unsupported NPY header')
            if fortran:raise HTTPException(409,'Unexpected array layout')
            if index is not None:
                if not 0<=index<shape[0]:raise HTTPException(422,'Index outside collected axis')
                count=int(np.prod(shape[1:]));stream.seek(stream.tell()+index*count*dtype.itemsize);shape=shape[1:]
            else:count=int(np.prod(shape))
            value=np.frombuffer(stream.read(count*dtype.itemsize),dtype=dtype).reshape(shape)
    return decode(value) if dtype==np.dtype('uint16') else value.astype(float)


def formation_own_run(model,variant):
    model_key(model)
    spec=read(BASE/'phase2747/training/protocol.json',{})
    allowed={'native'}|{c+'_'+str(s) for c in spec.get('conditions',[]) for s in spec.get('seeds',[])}
    if variant not in allowed or model!='qwen4' and variant!='native':raise HTTPException(422,'Unknown native deployment')
    return BASE/'phase2747/own_history'/model/variant


def formation_program_branch(branch):
    if branch not in {'native','code_identity','mapped_code','shuffled_map','mapped_digit_bias','mapped_letter_bias'}:
        raise HTTPException(422,'Unknown one-shot readout branch')
    return BASE/'phase2747/program_own_history/records'/branch


@router.get('/formation-program-samples')
def formation_program_samples(branch: str='native'):
    folder=formation_program_branch(branch)
    return [{k:r.get(k) for k in ['sample_id','source_group','target','depth','first_divergence_from_native']}
        for p in sorted(folder.glob('*.json')) if (r:=read(p,{}))]


@router.get('/formation-program-field')
def formation_program_field(branch: str='native',sample: str='',field: str='postnorm',view: str='raw'):
    folder=formation_program_branch(branch)
    if sample not in {r['sample_id'] for r in formation_program_samples(branch)}:
        raise HTTPException(404,'This main-run trajectory has not been committed')
    record=read(folder/(sample+'.json'),{})
    relative=(ROOT/record['field']['field_path']).relative_to(BASE).as_posix();file=archive(relative)
    if field=='postnorm':
        value=formation_npy(file,'native_postnorm_BF16');labels=[f'own step {i}' for i in range(len(value))]
    elif field=='first_readout':
        value=formation_npy(file,'first_actual_readout_BF16')[None];labels=['actual single readout']
    elif field in {'first_hidden','final_hidden'}:
        value=formation_npy(file,'first_final_all_hidden_BF16',int(field=='final_hidden'));labels=[f'H{i}' for i in range(len(value))]
    else:raise HTTPException(422,'Only first/final allH and every-step postnorm were collected')
    supplemental=next((r for r in read(BASE/'phase2747/program_own_history/terminal_review/result.json',{}).get('reviews',[])
        if r['branch']==branch and r['sample_id']==sample),None)
    return show(value,labels,view,record=record,archive=relative,
        supplemental_terminal_review=supplemental,
        axes='All2560native coordinates. Replacement readout is separate from the untouched native first-step hidden computation.',
        boundary='Exactly one readout replacement, then own native tokens. Initial originalKV is unchanged; equal first chosen tokens imply the same later deterministic history. Markdown prefixes are not literal-digit answers; use complete-answer/EOS/cap scores.')


@router.get('/formation-own-samples')
def formation_own_samples(model: str='qwen4',variant: str='native'):
    folder=formation_own_run(model,variant)
    return [{k:r.get(k) for k in ['sample_id','family','language','kind','full_hidden_collected','EOS','censored']}
        for p in sorted((folder/'records').glob('*.json')) if (r:=read(p,{}))]


@router.get('/formation-own-field')
def formation_own_field(model: str='qwen4',variant: str='native',sample: str='',field: str='postnorm',step: int=Query(0,ge=0,le=255),view: str='raw'):
    folder=formation_own_run(model,variant)
    known={r['sample_id'] for r in formation_own_samples(model,variant)}
    if sample not in known:raise HTTPException(404,'Own-history expression not committed')
    record=read(folder/'records'/(sample+'.json'),{})
    relative=(ROOT/record['field']['field_path']).relative_to(BASE).as_posix()
    file=archive(relative)
    if field=='postnorm':
        value=formation_npy(file,'postnorm_BF16');labels=[f'own step {i}' for i in range(len(value))]
    elif field=='all_hidden':
        value=formation_npy(file,'all_hidden_BF16',step);labels=[f'H{i}' for i in range(len(value))]
    else:raise HTTPException(422,'Unknown own-history field')
    supplemental=next((r for r in read(BASE/'phase2747/own_history/terminal_review/result.json',{}).get('reviews',[])
        if r['model']==model and r['variant']==variant and r['sample_id']==sample),None)
    return show(value,labels,view,record=record,archive=relative,supplemental_terminal_review=supplemental,
        axes='Every original native coordinate. Postnorm rows are actual own steps; allH rows are layers at a selected own step.',
        boundary='Only44frozenexpressions have every-step allH. All512 retain every-step postnorm. Natural completion is not scored against a unique gold continuation; firstB1 shape control is separate.')


@router.get('/formation-propagation-samples')
def formation_propagation_samples():
    follow=BASE/'phase2747/followup/material.json.gz'
    return [{k:r.get(k) for k in ['sample_id','family','source_group','original_text','kind']} for r in json.loads(gzip.decompress(follow.read_bytes()))['parameter_differential']] if follow.exists() else []


@router.get('/formation-propagation-field')
def formation_propagation_field(sample: str='',run: str='true_token_2747',field: str='hidden',route: str='full_prefix',view: str='raw'):
    spec=read(BASE/'phase2747/parameter_propagation/protocol.json',{})
    if sample not in spec.get('rows',[]):raise HTTPException(404,'Unknown propagation sample')
    if run not in spec.get('runs',[]):raise HTTPException(422,'Unknown propagation direction')
    if field not in ['hidden','Q','gate','up','product','MLP_input','MLP_write']:raise HTTPException(422,'Unknown native field')
    if route not in ['full_prefix','last_position_only','difference']:raise HTTPException(422,'Unknown tangent route')
    sample_i=spec['rows'].index(sample);run_i=spec['runs'].index(run);values=[];sources=[]
    for block in range(16,36):
        relative=f'phase2747/field_store/parameter_propagation/smooth/layers/block_{block:02d}.npz'
        a=formation_npy(archive(relative),'tangent_'+field,sample_i)
        values.append(a[2*run_i]-a[2*run_i+1] if route=='difference' else a[2*run_i+(route=='last_position_only')]);sources.append(relative)
    row=next(r for r in formation_propagation_samples() if r['sample_id']==sample)
    return show(np.stack(values),[f'block{i}' for i in range(16,36)],view,material=row,sources=sources,
        axes='Complete current-position FP32 tangent, blocks16..35; every original coordinate/unit. H is postblock, Q is head-major.',
        boundary='Actual whole-prefix derivative or same-current-position derivative omitting earlier-prefix initial tangents. Known original weights and prefix are used; no autonomous extraction or future gold input.')


@lru_cache(maxsize=1)
def formation_material():
    path=BASE/'phase2747/material/rows.json.gz'
    return json.loads(gzip.decompress(path.read_bytes())) if path.exists() else {}


@router.get('/formation-samples')
def formation_samples():
    ids=read(BASE/'phase2747/training/evaluation_rows.json',{}).get('fields',[])
    lookup={r['sample_id']:r for rows in formation_material().values() for r in rows}
    return [{k:lookup[sid].get(k) for k in ['sample_id','family','kind','language','split','source_group','target_text']} for sid in ids]


@router.get('/formation-field')
def formation_field(sample: str='',run: str='native',checkpoint: str='1',field: str='hidden',view: str='raw'):
    ids=read(BASE/'phase2747/training/evaluation_rows.json',{}).get('fields',[])
    if sample not in ids:raise HTTPException(404,'Unknown retained formation sample')
    index=ids.index(sample)
    if run in ['native','bridge']:
        relative=f'phase2747/field_store/training/baseline/{run}_fields.npz'
    else:
        known={r['run'] for r in formation_progress()['runs']}
        if run not in known:raise HTTPException(422,'Unknown formation run')
        if checkpoint not in ['1','8','32','128','deployed_BF16']:raise HTTPException(422,'Unknown formation checkpoint')
        name='deployed_BF16_fields' if checkpoint=='deployed_BF16' else 'fields_'+checkpoint.zfill(3)
        relative=f'phase2747/field_store/training/{run}/{name}.npz'
    if field not in ['hidden','Q','MLP_input','gate','up','product','MLP_write','postnorm_BF16']:
        raise HTTPException(422,'Unknown retained formation field')
    file=archive(relative)
    # ZIP stream skips preceding sample bytes; no unbounded complete-field cache.
    with zipfile.ZipFile(file) as packet,packet.open(field+'.npy') as stream:
        version=np.lib.format.read_magic(stream)
        if version==(1,0):shape,fortran,dtype=np.lib.format.read_array_header_1_0(stream)
        elif version==(2,0):shape,fortran,dtype=np.lib.format.read_array_header_2_0(stream)
        else:raise HTTPException(409,'Unsupported formation NPY header version')
        if fortran:raise HTTPException(409,'Unexpected formation layout')
        count=int(np.prod(shape[1:]))
        stream.seek(stream.tell()+index*count*dtype.itemsize)
        value=np.frombuffer(stream.read(count*dtype.itemsize),dtype=dtype).reshape(shape[1:])
    value=decode(value) if dtype==np.dtype('uint16') else value.astype(float)
    if value.ndim==1:value=value[None]
    if field=='Q':value=value.reshape(36,-1)
    rows=[r for rr in formation_material().values() for r in rr]
    row=next(r for r in rows if r['sample_id']==sample)
    return show(value,[('H' if field=='hidden' else 'block')+str(i) for i in range(len(value))],view,
        material=row,run=run,checkpoint=checkpoint,field=field,archive=relative,
        axes='Every retained native coordinate/unit in original order. Q=head-major4096; H=37x2560; gate/up/product=36x9728.',
        boundary='Trainableblock16MLP_write is FP32 before explicit bridge cast; H boundaries are actual BF16 residuals. Not every training token is retained.')


@router.get('/overview')
def overview():
    stages = {m: {name: read(BASE/name/m/'result.json', {}) for name in
                  ['capture', 'fit', 'diagonal', 'analysis', 'compilation', 'native_language']} for m in MODELS}
    for model, parts in stages.items():
        parts['captured_rows'] = sum(1 for _ in (BASE/'capture'/model/'commits').glob('*.json'))
        parts['native_language_review'] = read(BASE/'native_language'/model/'unparsed_EOS_adjudication.json', {})
        parts['native_language_progress'] = read(BASE/'native_language'/model/'progress.json', {})
    current = 'Phase2745: original-language completion and final delivery audit in progress'
    if all(parts['native_language'].get('all_passed') for parts in stages.values()):
        current = 'Phase2745 experiments complete; final delivery audit pending'
    if read(BASE/'delivery_manifest.json', {}).get('phase2745_complete'):
        current = 'Phase2745 verified; Phase2746 same-goal research in progress'
    if read(BASE/'phase2746/delivery_manifest.json', {}).get('phase2746_complete'):
        current = 'Phase2745/2746 verified; Phase2747 actual parameter formation in progress'
    if read(BASE/'phase2747/delivery_manifest.json', {}).get('phase2747_complete'):
        current = 'Phase2745/2746/2747 verified; same-goal continuation remains open, language mechanism not solved'
    return {'protocol': read(BASE/'protocol.json', {}), 'plan': read(BASE/'review/integrated_plan.json', {}),
        'review': read(BASE/'review/claim_audit.json', {}), 'queue': read(BASE/'queue/status.json', {}),
        'models': stages, 'norm': read(BASE/'norm_controls/analysis.json', {}),
        'norm_progress': read(BASE/'norm_controls/progress.json', {}),
        'norm_execution': read(BASE/'norm_controls/result.json', {}),
        'theory': read(BASE/'theory_snapshot.json', {}), 'verification': read(BASE/'verification/result.json', {}),
        'figures': read(BASE/'figures/index.json', {}),
        'current_status': current,
        'phase2747': formation_progress(),
        'phase2746': {'natural_interactions': read(BASE/'phase2746/natural_interactions/result.json', {}),
            'natural_scrutiny': read(BASE/'phase2746/natural_scrutiny/result.json', {}),
            'parameter_structure': read(BASE/'phase2746/parameter_structure/result.json', {}),
            'runtime_protocol': read(BASE/'phase2746/runtime/protocol.json', {}),
            'runtime_progress': read(BASE/'phase2746/runtime/progress.json', {}),
            'runtime_result': read(BASE/'phase2746/runtime/result.json', {}),
            'runtime_analysis':read(BASE/'phase2746/runtime_analysis/result.json',{}),
            'runtime_reuse':read(BASE/'phase2746/runtime_reuse/result.json',{}),
            'differential':read(BASE/'phase2746/differential/analysis/result.json',{}),
            'figures':read(BASE/'phase2746/figures/index.json',{}),
            'history_prediction':{'protocol':read(BASE/'phase2746/history_prediction/protocol.json',{}),
                'features':read(BASE/'phase2746/history_prediction/features/result.json',{}),
                'fit':read(BASE/'phase2746/history_prediction/fit/result.json',{}),
                'frozen':read(BASE/'phase2746/history_prediction/frozen.json',{}),
                'test':read(BASE/'phase2746/history_prediction/test/heldout.json',{}),
                'analysis':read(BASE/'phase2746/history_prediction/analysis/result.json',{}),
                'confirmation':{key:read(BASE/'phase2746/history_prediction/confirmation'/file,{}) for key,file in [
                    ('protocol','protocol.json'),('native','native/result.json'),('prediction','prediction/result.json'),
                    ('analysis','analysis/result.json'),('autonomous_progress','autonomous/progress.json'),
                    ('autonomous_result','autonomous/result.json'),('autonomous_analysis','autonomous/analysis.json')]}},
            'storage': read(BASE/'phase2746/storage.json', {})},
        'status': 'Observation, numerical reconstruction, heldout prediction and training/behavior evidence are distinct. Global mechanism is not claimed closed.'}


@lru_cache(maxsize=2)
def runtime_material(cohort='discovery'):
    if cohort not in ['discovery','confirmation']:raise HTTPException(422,'Unknown runtime cohort')
    path='phase2746/runtime/material.json.gz' if cohort=='discovery' else 'phase2746/history_prediction/confirmation/material.json.gz'
    return json.loads(gzip.decompress((BASE/path).read_bytes()))


def runtime_source(sid,cohort='discovery'):
    row=next((r for r in runtime_material(cohort) if r['sample_id']==sid),None)
    if row is None:raise HTTPException(404,'Unknown original runtime sample')
    prefix='phase2746/runtime/main/commits' if cohort=='discovery' else 'phase2746/history_prediction/confirmation/native/main/commits'
    path=BASE/prefix/(sid+'.json')
    if not path.is_file():raise HTTPException(409,'Runtime sample not committed')
    return row,read(path,{})


@router.get('/runtime-samples')
def runtime_samples(family: str='',cohort: str='discovery'):
    return [{k:r[k] for k in ['sample_id','family','kind','language','split','source_group']} for r in runtime_material(cohort)
        if not family or r['family']==family]


@router.get('/runtime')
def runtime_record(sample: str='',cohort: str='discovery'):
    row,record=runtime_source(sample,cohort)
    return {'material':row,'record':record,
        'scope':'Actual native own-history record. Natural32step trajectories are capped, not established complete answers. Discovery controlled histories replay prior runs; confirmation is newly generated.'}


@router.get('/runtime-field')
def runtime_field(sample: str='',step: int=Query(0,ge=0),mode: str='hidden',field: str='product',
                  block: int=Query(12,ge=0,le=35),view: str='raw',cohort: str='discovery'):
    row,record=runtime_source(sample,cohort)
    if step>=len(record['generated_ids']):raise HTTPException(422,'Step outside actual emitted-token history')
    if mode not in ['hidden','postnorm'] and step>=record['full_field_steps']:
        raise HTTPException(409,'Full unit/source fields were not retained at this step; no interpolated substitute')
    with np.load(archive(record['field_path'])) as z:
        if mode=='hidden':
            value=decode(z['hidden'][step]);labels=[f'H{i}' for i in range(len(value))]
            axes='All37original hidden boundaries at one actual current token; columns=all2560residual coordinates. Same index across layers is not assumed the same function.'
        elif mode=='postnorm':
            value=decode(z['postnorm'][step])[None];labels=['Final postnorm']
            axes='All2560original final-normalized coordinates, preceding the labeled emitted token.'
        elif mode in ['units','coordinates']:
            names=record['unit_field_names'] if mode=='units' else record['coordinate_field_names']
            if field not in names:raise HTTPException(422,'Unknown native field name')
            value=decode(z[mode][step,:,names.index(field)]);labels=[f'block{i} {field}' for i in range(36)]
            axes=('Every36native block; columns=all9728MLPunit indices within each block.' if mode=='units' else 'Every36native block; columns=all2560residual coordinates.')+' Different-layer indices are not functionally aligned.'
        elif mode=='attention':
            value=decode(z['attention_step'+str(step)][block]);labels=[f'head{i}' for i in range(len(value))]
            axes='All32query heads at selected block; columns=every actual source token position including current token, without padding or source ranking.'
        elif mode=='query':
            value=decode(z['Q_before_RoPE'][step,block]);labels=[f'head{i}' for i in range(len(value))]
            axes='All32query heads; columns=all128head components after native head norm and before RoPE, not residual coordinates.'
        else:raise HTTPException(422,'Unknown runtime field mode')
    return show(value,labels,view,axes=axes,sample_id=sample,model='qwen4',step=step,
        emitted_token_id=record['generated_ids'][step],full_field_steps=record['full_field_steps'],
        actual_steps=len(record['generated_ids']),source_group=row['source_group'])


@router.get('/self-history')
def self_history(sample: str=''):
    row,native=runtime_source(sample,'confirmation');directory=BASE/'phase2746/history_prediction/confirmation/autonomous/main/commits'
    records={name:read(directory/(sample+'__'+name+'.json'),{}) for name in ['native_B1_cache','frozen_direct','frozen_generalQ_native']}
    return {'material':row,'native_B8':native,'records':records,
        'scope':'Original-prefix initialization, then separate native-B1 or learned own-history branches. No teacher-state refresh; missing branches are not replaced by pilot results.'}


@router.get('/self-history-field')
def self_history_field(sample: str='',route: str='frozen_generalQ_native',field: str='postnorm',view: str='raw'):
    runtime_source(sample,'confirmation')
    if route not in ['native_B1_cache','frozen_direct','frozen_generalQ_native']:raise HTTPException(422,'Unknown self-fed route')
    path=BASE/'phase2746/history_prediction/confirmation/autonomous/main/commits'/(sample+'__'+route+'.json')
    if not path.is_file():raise HTTPException(409,'Self-fed row not committed; no pilot substitute')
    record=read(path,{})
    with np.load(archive(record['field_path'])) as z:
        if field=='postnorm':value=decode(z['postnorm'] if route=='native_B1_cache' else z['compiled_postnorm'])
        elif field in ['H0','H12','H35','H36']:
            boundary=int(field[1:])
            if route=='native_B1_cache':value=decode(z['hidden'][:,boundary])
            elif boundary<=12:value=decode(z['early_H0_H12'][:,boundary])
            else:value=z['predicted_H35_H36_Q35'][:,(boundary-35)*2560:(boundary-34)*2560]
        elif field=='Q35' and route!='native_B1_cache':value=z['predicted_H35_H36_Q35'][:,5120:]
        elif field=='candidate' and route!='native_B1_cache':value=z['available_candidate']
        else:raise HTTPException(409,'This field was not retained for this route')
    labels=[f'step {i} → token {t}' for i,t in enumerate(record['generated_ids'])]
    return show(value,labels,view,axes='Rows=every actual generated step, columns=complete original coordinates. Q35=32head blocks of128components. Later states of learned routes are predictions, not native hidden states.',
        route=route,field=field,sample_id=sample,actual_steps=len(labels),generated_text=record['generated_text'])


@router.get('/mechanism-case')
def mechanism_case(sample: str='',cohort: str='discovery',step: int=Query(0,ge=0),block: int=Query(35,ge=0,le=35)):
    row,record=runtime_source(sample,cohort)
    if step>=len(record['generated_ids']):raise HTTPException(422,'No actual generated step')
    root=BASE/'phase2746/history_prediction';stage=root/'confirmation/prediction' if cohort=='confirmation' else root/'test'
    point_id=sample+'_t'+str(step);predictions=[]
    if (stage/'records.json.gz').is_file():
        predictions=[r for r in json.loads(gzip.decompress((stage/'records.json.gz').read_bytes())) if r['point_id']==point_id]
    derivatives=[]
    if cohort=='discovery':
        for entrance in [12,24,35]:
            meta=read(BASE/'phase2746/differential/main'/('start_'+str(entrance)+'.json'),{})
            item=next((r for r in meta.get('records',[]) if r['sample_id']==sample and r['step']==step),None)
            if item is not None:derivatives.append({'record':item,'field_path':meta['field_path'],'start':entrance})
    return {'sample_id':sample,'point_id':point_id,'source_group':row['source_group'],'external_material':row,
        'native_runtime':{'field_path':record['field_path'],'step':step,'full_unit_field_retained':step<record['full_field_steps'],
            'prompt_and_past_generated_ids':record['actual_prompt_ids']+record['generated_ids'][:step],
            'actual_emitted_token_id':record['generated_ids'][step]},
        'native_parameter_addresses':[f'model.layers.{block}.{name}.weight' for name in ['mlp.gate_proj','mlp.up_proj','mlp.down_proj',
            'self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj','self_attn.q_norm','self_attn.k_norm']],
        'full_remaining_network_differentials':derivatives,'frozen_current_state_forecasts':predictions,
        'parameter_query':{'model':'qwen4','block':block,'MLP_unit_range':[0,9727],'residual_coordinate_range':[0,2559],'query_head_range':[0,31]},
        'scope':'Identity links between external expression, native all-unit state, fixed parameter addresses, scoped derivatives and frozen output forecasts. Missing derivative/prediction coverage is an empty list, not an inferred mechanism. Original native models and fixed architecture addresses do not establish semantic uniqueness.'}


@router.get('/samples')
def samples(model: str='qwen4', family: str='', split: str='', language: str=''):
    return [{'sample_id': r['sample_id'], 'pair_id': r['pair_id'], 'family': r['family'],
             'language': r['language'], 'case': r['case'], 'world': r['world'], 'split': r['split'],
             'tokens': len(r['prompt_ids']), 'captured': (BASE/'capture'/model/'commits'/(r['sample_id']+'.json')).exists()}
        for r in rows(model) if (not family or r['family'] == family) and (not split or r['split'] == split)
             and (not language or r['language'] == language)]


@router.get('/sample')
def sample(model: str='qwen4', sample: str=''):
    row = source(model, sample)
    return {'material': row, 'commit': read(BASE/'capture'/model/'commits'/(sample+'.json'), {}),
        'queries': material()['models'][model]['probes']}


@router.get('/field')
def field(model: str='qwen4', sample: str='', mode: str='queries', boundary: str='postnorm',
          query: int=Query(0, ge=0, le=99), block: int=Query(0, ge=0), view: str='raw'):
    row = source(model, sample)
    with np.load(fields(model, sample)) as z:
        if mode == 'queries':
            value = native_state(z, boundary)
            labels = [p['probe_id']+' '+p['text'] for p in material()['models'][model]['probes']]
            axes = 'Rows: fixed diagnostic queries. Columns: every residual coordinate.'
        elif mode == 'prefix':
            value = decode(z['prefix_layers'])
            labels = [f'prefix H{i}' for i in range(len(value))]
            axes = 'All layer boundaries at the final actual prefix token; not all prefix positions.'
        elif mode in ['q_before_rope', 'q_input', 'q_projected', 'attention', 'attention_output']:
            key = f'p{query}_L{block}_{mode}'
            if key not in z.files:
                raise HTTPException(409, 'This native block field was not retained')
            value = decode(z[key])
            if value.ndim == 1:
                value = value[None]
            labels = [f'{mode} row/head {h}' for h in range(len(value))]
            axes = 'Attention columns=all source positions (prefix then query); Q-before-RoPE columns=head components. Projected Q width need not equal residual width.'
        else:
            raise HTTPException(422, 'Unknown native field mode')
    return show(value, labels, view, axes=axes, sample_id=row['sample_id'], model=model)


@router.get('/pair')
def pair(model: str='qwen4', sample: str='', boundary: str='postnorm', query: int=Query(0, ge=0, le=99), view: str='raw'):
    chosen = source(model, sample)
    pair_rows = sorted([r for r in rows(model) if r['pair_id'] == chosen['pair_id']], key=lambda r:r['world'])
    value = []
    for r in pair_rows:
        with np.load(fields(model, r['sample_id'])) as z:
            value.append(native_state(z, boundary)[query].astype(float))
    return {'materials': pair_rows, 'field': show(np.stack([*value, value[1]-value[0]]),
        ['World A actual', 'World B actual', 'B minus A (measurement only)'], view),
        'scope': 'Exactly matched token multiset; numerical difference is not transported into a model. Position/order rules remain alternatives.'}


@router.get('/interaction')
def interaction(model: str='qwen4', sample: str='', boundary: str='postnorm', view: str='raw'):
    row = source(model, sample)
    if boundary != 'postnorm' and (not boundary.isdigit() or int(boundary) > 100):
        raise HTTPException(422, 'Invalid native boundary')
    path = BASE/'analysis'/model/'anova'/('boundary_'+boundary+'.npz')
    if not path.exists():
        raise HTTPException(409, 'ANOVA margins are not yet committed')
    index = next(i for i, r in enumerate(rows(model)) if r['sample_id'] == sample)
    v = 0 if view == 'raw' else 1 if view == 'RMS' else None
    if v is None:
        raise HTTPException(422, 'Unknown view')
    with np.load(fields(model, sample)) as z:
        actual = native_state(z, boundary).astype(float)
    if v:
        actual /= np.sqrt(np.mean(actual*actual, -1, keepdims=True)).clip(1e-12)
    with np.load(path) as z:
        mu, a, b = z['grand_mean'][v], z['prefix_effect'][v, index], z['query_effect'][v]
    residual = actual-mu-a-b
    labels = [q['probe_id'] for q in material()['models'][model]['probes']]
    return {'interaction': show(residual, labels), 'mean_terms': show(np.stack([mu, a]), ['Grand mean', 'Prefix main effect']),
        'view_before_decomposition': view, 'material': row,
        'scope': 'C=Y-mu-A-B in the observed balanced320by100grid; descriptive statistical interaction, not a unique semantic or causal component.'}


@router.get('/prediction')
def prediction(model: str='qwen4', sample: str='', query: int=Query(0, ge=0, le=99),
               target: str='postnorm', input: str='actual_query_H1'):
    row = source(model, sample)
    if target not in ['Hearly', 'postnorm'] or input not in ['actual_query_H1', 'available_prefix_ordered_candidate_early_output']:
        raise HTTPException(422, 'Unknown frozen operator input or target')
    path = BASE/'fit'/model/'operators'/(input+'__true_correspondence__'+target+'.npz')
    if not path.exists() or not (BASE/'fit'/model/'result.json').exists():
        raise HTTPException(409, 'This frozen operator is not committed')
    pairs = sorted([r for r in rows(model) if r['pair_id'] == row['pair_id']], key=lambda r:r['world'])
    early = read(BASE/'capture'/model/'result.json')['early']
    xx, yy = [], []
    for r in pairs:
        with np.load(fields(model, r['sample_id'])) as z:
            xx.append(native_state(z, '1')[query].astype(float) if input == 'actual_query_H1' else decode(z['candidate_early_output'][4, query]).astype(float))
            yy.append(native_state(z, str(early) if target == 'Hearly' else target)[query].astype(float))
    with np.load(path) as z:
        predicted = (xx[1]-xx[0])@z['operator']
    actual = yy[1]-yy[0]
    return show(np.stack([actual, predicted, predicted-actual]),
        ['Actual relation-pair change', 'Frozen predicted pair change', 'Prediction error'],
        MSE=float(np.mean((predicted-actual)**2)), zero_change_MSE=float(np.mean(actual*actual)),
        available_inputs=input+' only; later target used for scoring, never for this prediction. Pair subtraction is a measurement, not an injected activation.',
        model=model, target=target, source_split=row['split'],
        query_split=material()['models'][model]['probes'][query]['split'])


@router.get('/archives')
def archives(prefix: str='', offset: int=Query(0, ge=0), limit: int=Query(100, ge=1, le=1000)):
    if '\\' in prefix or '..' in prefix.split('/'):
        raise HTTPException(422, 'Invalid archive prefix')
    files = sorted(p for p in BASE.rglob('*.npz') if '.tmp.' not in p.name and p.relative_to(BASE).as_posix().startswith(prefix))
    return {'total': len(files), 'offset': offset,
        'rows': [{'path': p.relative_to(BASE).as_posix(), 'bytes': p.stat().st_size} for p in files[offset:offset+limit]]}


@router.get('/arrays')
def arrays(path: str=''):
    file = archive(path)
    return [{**r, 'name': r['array']} for r in npz_headers(str(file), file.stat().st_mtime_ns)]


@router.get('/array')
def array(path: str='', name: str='', row_start: int=Query(0, ge=0), row_count: int=Query(32, ge=1, le=128),
          start: int=Query(0, ge=0), count: int=Query(8192, ge=1, le=32768)):
    file = archive(path)
    with zipfile.ZipFile(file) as packet:
        if name+'.npy' not in packet.namelist():
            raise HTTPException(404, 'Unknown tensor')
        with packet.open(name+'.npy') as stream:
            version = np.lib.format.read_magic(stream)
            reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
            shape, fortran, dtype = reader(stream)
            if dtype.kind not in 'buif' or not shape:
                raise HTTPException(422, 'This tensor requires the exact original archive reader')
            width = shape[-1]
            total = int(np.prod(shape[:-1])) if len(shape) > 1 else 1
            if row_start >= total or start >= width:
                raise HTTPException(422, 'Page outside original axes')
            nrows = min(row_count, total-row_start)
            if fortran:
                if int(np.prod(shape)) > 262144:
                    raise HTTPException(422, 'Large Fortran-layout tensor requires its original reader; no incorrect C-order page is substituted')
                with packet.open(name+'.npy') as original:
                    value = np.lib.format.read_array(original,allow_pickle=False).reshape(-1,width)[row_start:row_start+nrows]
            else:
                stream.seek(stream.tell()+row_start*width*dtype.itemsize)
                value = np.frombuffer(stream.read(nrows*width*dtype.itemsize), dtype=dtype).reshape(nrows, width)
            value = decode(value) if dtype == np.dtype('uint16') else value.astype(float)
    if not np.isfinite(value).all():
        raise HTTPException(422, 'Undefined values; inspect validity mask in the original archive')
    labels = [str(tuple(map(int, np.unravel_index(i, shape[:-1])))) if len(shape) > 1 else 'vector' for i in range(row_start, row_start+nrows)]
    return field_response(value, labels, start, count,
        'Exact original tensor row/column window. ZIP data is streamed, not cached as a full tensor; no coordinate selection by magnitude.',
        tensor_shape=list(shape), row_start=row_start, row_end=row_start+nrows, total_rows=total,
        scale_scope='Maximum over the selected rows and all their native columns, not over the complete original tensor.')


@lru_cache(maxsize=1)
def native_reader():
    tests = str(ROOT/'tests/glm5')
    if tests not in sys.path:
        sys.path.insert(0, tests)
    import rdc_construction_parameters
    return rdc_construction_parameters


@router.get('/parameters')
def parameters(model: str='qwen4', block: int=Query(0, ge=0)):
    meta = native_reader().catalog(model_key(model))
    return {'model': model, 'config': meta['config'], 'registered_scalar_count': meta['registered_scalar_count'],
        'parameters': [r for r in meta['parameters'] if r['layer'] == block or r['layer'] is None],
        'scope': meta['coverage']}


@router.get('/parameter')
def parameter(model: str='qwen4', block: int=Query(0, ge=0), unit: int=Query(0, ge=0),
              input: int=Query(0, ge=0), input_r: int=Query(1, ge=0), output: int=Query(0, ge=0)):
    reader = native_reader()
    meta = reader.catalog(model_key(model))
    width, middle = meta['config']['hidden_size'], meta['config']['intermediate_size']
    if block >= meta['config']['num_hidden_layers'] or unit >= middle or max(input, input_r, output) >= width:
        raise HTTPException(422, 'Outside original parameter axes')
    gate, up, down = reader.mlp_factors(model, block, unit)
    return {'model': model, 'block': block, 'unit': unit, 'input_i': input, 'input_r': input_r, 'output_j': output,
        'W_gate_k_i': float(gate[input]), 'W_up_k_r': float(up[input_r]), 'W_down_j_k': float(down[output]),
        'Gamma_k_j_i_r': float(down[output]*gate[input]*up[input_r]),
        'factors': show(np.stack([gate, up, down]), ['All gate-read coordinates', 'All up-read coordinates', 'All down-write coordinates']),
        'scope': 'Exact product of three original BF16 scalar values evaluated in FP64; Gamma is an architecture factorization. These indices are not standalone semantic labels.'}


@router.get('/attention-parameter')
def attention_parameter(model: str='qwen4', block: int=Query(0, ge=0), head: int=Query(0, ge=0),
                        input: int=Query(0, ge=0), input_r: int=Query(1, ge=0), output: int=Query(0, ge=0),
                        query_position: int=Query(113, ge=0, le=1000000), key_position: int=Query(17, ge=0, le=1000000)):
    reader = native_reader()
    meta = reader.catalog(model_key(model)); config = meta['config']
    if block >= config['num_hidden_layers'] or head >= config['num_attention_heads'] or max(input,input_r,output) >= config['hidden_size']:
        raise HTTPException(422, 'Outside original attention parameter axes')
    import rdc_native_attention_parameters
    value = rdc_native_attention_parameters.entries(model,block,head,input,input_r,output,query_position,key_position)
    qk = value.pop('qk_all_head_component_terms'); ov = value.pop('ov_all_head_component_terms')
    factors = value.pop('read_write_factors')
    return {**value,'component_terms':show(np.stack([qk,ov]),['QK numerator: every head component','OV: every head component']),
        'factors':show(factors,['Q read at i','K read at r','V read at i','O write to j','Q norm gain','K norm gain','Q bias','K bias','V bias']),
        'axes':'Columns are all native head components, not residual coordinates. GQA key/value head mapping is explicit.'}


@router.get('/norm-variants')
def norm_variants():
    spec = read(BASE/'norm_controls/protocol.json', {})
    return [{'variant': v, 'execution': read(BASE/'norm_controls'/v['name']/'result.json', {}),
             'matching': read(BASE/'norm_controls'/v['name']/'parameter_matching.json', {})} for v in spec.get('variants', [])]


@router.get('/norm-behavior')
def norm_behavior(variant: str='', sample: str=''):
    source('qwen4', sample)
    spec = read(BASE/'norm_controls/protocol.json', {})
    selected = next((v for v in spec.get('variants', []) if v['name'] == variant), None)
    if selected is None:
        raise HTTPException(404, 'Unknown registered parameter condition')
    if selected['kind'] == 'reuse':
        file = ROOT/'tests/glm5/result/rdc_query_campaign_20260913/identifiability/behavior'/selected['old_variant']/'commits'/(sample+'.json')
        return {'record': read(file, {}), 'reused_prior_result': True}
    file = BASE/'norm_controls'/variant/'behavior.json.gz'
    if not file.exists():
        raise HTTPException(409, 'Own-history experiment not committed')
    data = json.loads(gzip.decompress(file.read_bytes()))
    return {'record': next(r for r in data if r['sample_id'] == sample), 'reused_prior_result': False}


@router.get('/figure')
def figure(name: str='',stage: str='2745'):
    if stage not in ['2745','2746','2747']:raise HTTPException(422,'Unknown figure stage')
    spec = formation_figures() if stage=='2747' else read(BASE/('figures/index.json' if stage=='2745' else 'phase2746/figures/index.json'), {}).get('figures', [])
    item = next((r for r in spec if r.get('name',Path(r['path']).stem) == name), None)
    if item is None:
        raise HTTPException(404, 'Unknown registered figure')
    file = BASE/item['path']
    if not file.resolve().is_relative_to(BASE.resolve()):
        raise HTTPException(404, 'Outside figure registry')
    return FileResponse(file)


@router.get('/native-language')
def native_language(model: str='qwen4', sample: str=''):
    row = source(model, sample)
    directory = BASE/'native_language'/model
    if model == 'qwen4':
        file = ROOT/'tests/glm5/result/rdc_query_campaign_20260913/identifiability/behavior/native/commits'/(sample+'.json')
    else:
        file = directory/'commits'/(sample+'.json')
    if not file.exists():
        raise HTTPException(409, 'Native own-history row not committed')
    return {'material': row, 'record': read(file), 'reused_prior_result': model == 'qwen4',
        'per_step_fields_available': model != 'qwen4',
        'scope': 'Original greedy model behavior; no learned predictor or external answer supplied. Q4 old trajectories reused, not new per-step capture.'}


@router.get('/native-language-field')
def native_language_field(model: str='glm4', sample: str='', boundary: int=-1, view: str='raw'):
    source(model, sample)
    if model == 'qwen4':
        raise HTTPException(409, 'Q4 behavior reused; no new complete per-step fields were collected')
    file = archive(f'native_language/{model}/fields/{sample}.npz')
    with np.load(file) as z:
        indices = z['selected_boundaries'].tolist()
        if boundary not in indices:
            raise HTTPException(409, 'Boundary not retained in this per-step field')
        value = decode(z['selected_hidden_states'][:, indices.index(boundary)])
        ids = z['generated_ids'].tolist()
    return show(value, [f'step {i}: next token ID {t}' for i, t in enumerate(ids)], view,
        model=model, sample_id=sample, boundary=boundary,
        axes='Rows are actual greedy prediction steps including EOS; columns are every residual coordinate. State precedes emission of its labeled token; final postnorm=-1. Prefix/KV fields are not all persisted here.')


from server.rdc_question_service import router as question_router
router.include_router(question_router)
