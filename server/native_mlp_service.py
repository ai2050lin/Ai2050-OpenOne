"""Versioned read-only Qwen3-4B/14B middle-layer native atlases. No GPU."""
from functools import lru_cache,partial
import json
import os
from pathlib import Path
from typing import Literal
import numpy as np
from fastapi import APIRouter,HTTPException,Query
from shared.dist_protocol.mechanism_trace import file_sha,unbits

ROOT=Path(__file__).resolve().parents[1]
SPECS={
 'qwen4':dict(run_id='QWEN4-GOODFIRE-B18-B28-20261113-v2',width=9728,dim=2560,
   snapshot='6be1034f0b55a6dfc9383b4f155a736433d822a21f3ab6178012c2c2e14739ad'),
 'qwen14':dict(run_id='QWEN14-GOODFIRE-B18-B28-20261113-v2',width=17408,dim=5120,
   snapshot='1e970d119494affd09689ec3dd25aa30587dfcbb16d873b40fb917016fe2a2ce')}
router=APIRouter(prefix='/api/research/native-mlp',tags=['native-mlp'])
_checked={}


def spec(run):
    for key,value in SPECS.items():
        if value['run_id']==run:return key,value
    raise HTTPException(404,'未知原生研究版本')


def folder(run):
    key,_=spec(run)
    if os.environ.get('AI2050_NATIVE_MLP_ROOT'):
        return (Path(os.environ['AI2050_NATIVE_MLP_ROOT'])/key).resolve()
    if os.environ.get('AI2050_DIST_DIR'):
        return (Path(os.environ['AI2050_DIST_DIR'])/'native_atlases'/run).resolve()
    return (ROOT/'tests/glm5/result/goodfire_middle_20261113_v2'/key).resolve()


def manifest(run):
    _,expected=spec(run);p=folder(run)/'manifest.json'
    if not p.is_file():raise HTTPException(404,'此模型的原生研究包尚未部署或尚未通过完整审计。')
    d=json.loads(p.read_text(encoding='utf-8'))
    if (d.get('schema')!='native_mlp_manifest.v1' or d.get('state')!='complete' or d.get('kind')!='real'
        or d.get('run_id')!=run or d.get('model_snapshot')!=expected['snapshot']
        or d.get('width')!=expected['width'] or d.get('hidden_size')!=expected['dim'] or d.get('blocks')!=list(range(18,29))):
        raise HTTPException(409,'原生研究包身份或完成状态不匹配')
    names=[r['path'] for r in d.get('files',[])]
    if len(names)!=len(set(names)):raise HTTPException(409,'研究清单含重复路径')
    return d


def path(name,run):
    d=manifest(run);record=next((r for r in d['files'] if r['path']==name),None)
    if record is None:raise HTTPException(404,'研究文件未登记')
    if Path(name).name!=name or '\\' in name or ':' in name:raise HTTPException(409,'非法研究路径')
    p=(folder(run)/name).resolve()
    if p.parent!=folder(run) or not p.is_file():raise HTTPException(404,'原生文件未安装：'+name)
    stamp=(p.stat().st_size,p.stat().st_mtime_ns,record['sha256'])
    if _checked.get(str(p))!=stamp:
        if p.stat().st_size!=record['bytes'] or file_sha(p)!=record['sha256']:
            raise HTTPException(409,'原生文件完整性校验失败：'+name)
        _checked[str(p)]=stamp
    return p


@lru_cache(maxsize=24)
def _read(p,stamp):return json.loads(Path(p).read_text(encoding='utf-8'))


def read(name,run):
    p=path(name,run);d=_read(str(p),p.stat().st_mtime_ns);_,expected=spec(run)
    if isinstance(d,dict) and (d.get('model_snapshot',expected['snapshot'])!=expected['snapshot'] or d.get('run_id',run)!=run):
        raise HTTPException(409,'研究文件内模型快照不一致')
    return d


def check(block=None,unit=None,snapshot=None,run=None):
    d=manifest(run)
    if snapshot and snapshot!=d['model_snapshot']:raise HTTPException(409,'选中模型与研究快照不一致')
    if block is not None and block not in d['blocks']:raise HTTPException(422,'此层未采集')
    if unit is not None and not 0<=unit<d['width']:raise HTTPException(422,'单元编号越界')
    return d


def access(meta):return partial(read,run=meta['run_id']),partial(path,run=meta['run_id'])
def legacy_name(meta,block,name):return name


def prediction(path,block):
    with np.load(path(f'B{block}-prediction.npz')) as z:
        result={}
        for f in ['G','V','A']:
            result[f]={'candidate_predictable':z[f+'_candidate'].tolist(),'splits':{}}
            for split in ['confirm','holdout']:
                values={k:z[f+'_'+split+'_'+v+'_mse'].tolist() for k,v in
                    [('mean_mse','mean'),('lexical_mse','baseline'),('conditional_mse','conditional'),
                     ('matched_current_mse','matched_current'),('shuffled_mse','shuffled'),
                     ('exact_token_background_mse','exact_token_background')]}
                values['conditional_over_lexical']=(z[f+'_'+split+'_conditional_mse']/np.maximum(z[f+'_'+split+'_baseline_mse'],1e-20)).tolist()
                if f=='A':values['composed_mse']=z['A_'+split+'_composed_mse'].tolist()
                result[f]['splits'][split]=values
        return result


@router.get('')
def overview(snapshot:str|None=None,run:str|None=None,model:str='qwen4'):
    if run is None:
        if model not in SPECS:raise HTTPException(404,'未知模型')
        run=SPECS[model]['run_id']
    meta=check(snapshot=snapshot,run=run);read,_=access(meta)
    return {**read('report.json'),'manifest_sha256':file_sha(folder(meta['run_id'])/'manifest.json')}


@router.get('/layers/{block}')
def layer(block:int,snapshot:str|None=None,run:str|None=None):
    meta=check(block,snapshot=snapshot,run=run);WIDTH=meta['width'];SNAPSHOT=meta['model_snapshot'];read,path=access(meta);d=read(f'B{block}-profile.json')
    extra={}
    if meta['schema']=='native_mlp_manifest.v1':
        with np.load(path(f'B{block}-regimes.npz')) as z:
            extra={'double_negative_mass':z['holdout_quadrant_abs_mass_fraction'][:,3].tolist(),
                   'negative_silu_derivative':z['holdout_negative_silu_derivative_fraction'].tolist()}
    return {'run_id':meta['run_id'],'model_snapshot':SNAPSHOT,'block':block,'width':WIDTH,
        'arrays':{**extra,**{f'{f}_{k}':d['fields'][f][k] for f in ['G','V','A'] for k in ['mean','std','min','max']},
                  **{f'sign_{i}':[r[i] for r in d['sign_regime_fractions']] for i in range(4)}},
        'semantic_confirmed':0,'scope':'原生编号全量观察；配色不是语义或因果强度。'}


@lru_cache(maxsize=2)
def tokenizer(p,stamp):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(p)


@router.get('/units/{block}/{unit}')
def unit(block:int,unit:int,split:Literal['discovery','confirm','holdout']='holdout',
         field:Literal['G','V','A']='A',pole:Literal['positive','negative']='positive',snapshot:str|None=None,run:str|None=None):
    meta=check(block,unit,snapshot,run);WIDTH=meta['width'];SNAPSHOT=meta['model_snapshot'];read,path=access(meta);p=read(f'B{block}-profile.json');docs=read('documents.json')
    tp=path('tokenizer.json');tok=tokenizer(str(tp),tp.stat().st_mtime_ns)
    ix=p['fields'][field][split][pole+'_indices'][unit];examples=[]
    maps={f:np.load(path(f'B{block}-{f}.npy'),mmap_mode='r') for f in ['G','V','A']}
    for idx in ix:
        di,slot=divmod(idx,len(docs['positions']));d=docs['docs'][di];pos=docs['positions'][slot]
        examples.append({'document_index':di,'token_position':pos,'token_id':d['tokens'][pos],
            **{f:float(unbits(m[di,slot,unit])) for f,m in maps.items()},
            'prefix':tok.decode(d['tokens'][:pos+1],skip_special_tokens=False),
            'following_not_causal':tok.decode(d['tokens'][pos+1:pos+33],skip_special_tokens=False),
            'source':{k:d.get(k) for k in ['doc_id','source','repo','revision','license','split']}})
    weights={}
    for f in ['gate','up','down']:
        w=np.load(path(f'B{block}-W-{f}.npy'),mmap_mode='r')
        weights[f]=unbits(w[:,unit] if f=='down' else w[unit]).tolist()
    di,slot=divmod(ix[0],len(docs['positions']))
    native_input=unbits(np.load(path(f'B{block}-input.npy'),mmap_mode='r')[di,slot])
    contribution=(native_input*0+examples[0]['A'])*np.asarray(weights['down'],dtype=np.float32)
    arithmetic={'document_index':di,'position':docs['positions'][slot],'normalized_input':native_input.tolist(),
        'native_G':examples[0]['G'],'native_V':examples[0]['V'],'native_A':examples[0]['A'],
        'fp64_gate_dot':float(native_input.astype(np.float64)@np.asarray(weights['gate'])),
        'fp64_up_dot':float(native_input.astype(np.float64)@np.asarray(weights['up'])),
        'native_column_contribution':contribution.tolist(),
        'scope':'FP64 scalar-dot audit differs from original BF16 matrix arithmetic; one down-column contribution is not the entire MLP output.'}
    with np.load(path(legacy_name(meta,block,'prefix-extrema.npz'))) as z:
        key=f'B{block}_{field}_{split}_{"high" if pole=="positive" else "low"}'
        idx=int(z[key+'_index'][unit]);di,pos=divmod(idx,1024);d=docs['docs'][di]
        full_pole={'document_index':di,'token_position':pos,'value':float(z[key+'_value'][unit]),
                   'prefix':tok.decode(d['tokens'][:pos+1],skip_special_tokens=False),'scope':'512–1023全部位置中的此端极值，不限于8个固定位置。'}
    pred=prediction(path,block)
    proxy=read(f'B{block}-readout.json')
    report=read('report.json');cases=[]
    if any(r['block']==block and r['unit']==unit for r in report.get('intervention',{}).get('selected_units',[])):
        effects=read(legacy_name(meta,block,'natural_exact-effects.json'))['rows'];branch=f'B{block}:U{unit}:0.0:native'
        relevant=[r for r in effects if r['branch']==branch]
        selected_cases=sorted(relevant,key=lambda r:abs(r['nll']-r['baseline_nll']),reverse=True)[:2]
        selected_cases += [r for r in relevant if r['argmax']==r['baseline_argmax']][:2]
        seen=set()
        for r in selected_cases:
            if r['sample'] in seen:continue
            seen.add(r['sample']);di=768+r['sample'];d=docs['docs'][di]
            cases.append({**r,'prefix':tok.decode(d['tokens'][:1007],skip_special_tokens=False),
                'target_token':tok.decode([r['target']],skip_special_tokens=False),
                'original_prediction':tok.decode([r['baseline_argmax']],skip_special_tokens=False),
                'edited_prediction':tok.decode([r['argmax']],skip_special_tokens=False)})
    extra={}
    if meta['schema']=='native_mlp_manifest.v1':
        with np.load(path(f'B{block}-regimes.npz')) as z:
            extra['regimes']={k:np.asarray(z[k][unit]).tolist() for k in z.files}
        with np.load(path(f'B{block}-prediction.npz')) as z:
            extra['standardized_prediction']={k:np.asarray(z[k][unit]).tolist() for k in z.files}
        if block>min(meta['blocks']):
            edge=read(f'B{block-1}-B{block}-prediction.json')
            extra['frozen_parent']={'block':block-1,'unit':edge['parents'][unit],
                'discovery_correlation':edge['discovery_correlation'][unit],
                'holdout_mse_ratio':edge['splits']['holdout']['ratio'][unit],
                'causal_confirmed':False}
    return {**extra,'run_id':meta['run_id'],'source_run_id':p.get('run_id'),'model_snapshot':SNAPSHOT,'block':block,'unit':unit,
        'statistics':{f:{k:p['fields'][f][k][unit] for k in ['mean','std','min','max']} for f in ['G','V','A']},
        'sign_regime_fractions':p['sign_regime_fractions'][unit],
        'candidate':read(f'B{block}-cues.json')['units'][unit],'examples':examples,'full_prefix_pole':full_pole,'weights':weights,
        'arithmetic':arithmetic,
        'intervention_cases':cases,
        'parameter_readout':{side:[{'token_id':i,'token':tok.decode([i],skip_special_tokens=False),'value':v}
            for i,v in zip(proxy[side+'_ids'][unit],proxy[side+'_values'][unit])] for side in ['positive','negative']},
        'prediction':{f:{'candidate_predictable':v['candidate_predictable'][unit],
                        'splits':{s:{k:a[unit] for k,a in values.items()} for s,values in v['splits'].items()}} for f,v in pred.items()},
        'semantic_confirmed':False,'scope':'候选语境与响应预测，不等于概念定义。原生参数按BF16实际值展示。'}


@router.get('/relations/{block}/{unit}')
def relations(block:int,unit:int,direction:Literal['incoming','outgoing']='outgoing',
              kind:Literal['rho-discovery','rho-confirm','rho-holdout','weight-gate','weight-up']='rho-holdout',snapshot:str|None=None,run:str|None=None):
    from threadpoolctl import threadpool_limits
    meta=check(block,unit,snapshot,run);WIDTH=meta['width'];SNAPSHOT=meta['model_snapshot'];read,path=access(meta)
    source,target=(block,block+1) if direction=='outgoing' else (block-1,block)
    if source not in meta['blocks'] or target not in meta['blocks']:raise HTTPException(422,'该相邻边界未采集')
    result=np.zeros(WIDTH,np.float64)
    with threadpool_limits(limits=4):
        if kind.startswith('rho'):
            a,b={'rho-discovery':(0,4096),'rho-confirm':(4096,6144),'rho-holdout':(6144,8192)}[kind]
            left=np.load(path(f'B{source}-A.npy'),mmap_mode='r').reshape(-1,WIDTH)
            right=np.load(path(f'B{target}-A.npy'),mmap_mode='r').reshape(-1,WIDTH)
            fixed,vary=(left,right) if direction=='outgoing' else (right,left)
            x=unbits(fixed[a:b,unit]).astype(np.float64);x-=x.mean();xn=np.linalg.norm(x)
            for lo in range(0,WIDTH,256):
                y=unbits(vary[a:b,lo:lo+256]).astype(np.float64);y-=y.mean(0)
                den=xn*np.linalg.norm(y,axis=0)
                result[lo:lo+256]=np.divide(x@y,den,out=np.zeros(y.shape[1]),where=den>1e-20)
        else:
            down=np.load(path(f'B{source}-W-down.npy'),mmap_mode='r')
            rd=np.load(path(f'B{target}-W-{kind.split("-")[1]}.npy'),mmap_mode='r')
            fixed=unbits(down[:,unit] if direction=='outgoing' else rd[unit]).astype(np.float64)
            for lo in range(0,WIDTH,256):
                y=unbits(rd[lo:lo+256] if direction=='outgoing' else down[:,lo:lo+256].T).astype(np.float64)
                den=np.linalg.norm(fixed)*np.linalg.norm(y,axis=1)
                result[lo:lo+256]=np.divide(y@fixed,den,out=np.zeros(len(y)),where=den>1e-20)
    return {'run_id':meta['run_id'],'model_snapshot':SNAPSHOT,'source':source,'target':target,'unit':unit,
        'direction':direction,'kind':'observation' if kind.startswith('rho') else 'parameter_alignment',
        'values':result.tolist(),'full_units':WIDTH,'scope':'全部相邻单元；相关和静态权重对齐均不是因果连接。'}


@router.get('/interventions/{stage}/{block}/{unit}')
def interventions(stage:Literal['natural','capability'],block:int,unit:int,
                  sample:int|None=Query(None,ge=0),snapshot:str|None=None,run:str|None=None):
    meta=check(block,unit,snapshot,run);WIDTH=meta['width'];SNAPSHOT=meta['model_snapshot'];read,path=access(meta)
    data=read(legacy_name(meta,block,f'{stage}_exact-effects.json'))
    if sample is not None and sample>=data['samples']:raise HTTPException(422,'样本编号越界')
    branch_ids={r['id'] for r in data['branches'] if r.get('block')==block and r.get('unit')==unit}
    if not branch_ids:raise HTTPException(404,'该单元已有普查档案，但尚未进行本项写入干预。')
    rows=[r for r in data['rows'] if r['branch'] in branch_ids and (sample is None or r['sample']==sample)]
    result={'run_id':meta['run_id'],'model_snapshot':SNAPSHOT,'stage':stage,'block':block,'unit':unit,'sample':sample,
        'samples':data['samples'],'rows':rows,
        'controls':[r for r in data['controls'] if r['branch'] in branch_ids and (sample is None or r['sample']==sample)],
        'replay':[r for r in data['replay'] if sample is None or r['sample']==sample],
        'scope':'全部指定范围的原始行，含失败对照；不是仅列显著或成功样本。'}
    if sample is not None:
        if stage=='natural':
            d=read('documents.json')['docs'][768+sample];ids=d['tokens'][:1007]
            result['input_identity']={k:d.get(k) for k in ['doc_id','source','repo','revision','license','split']}
        else:
            d=read(legacy_name(meta,block,'capability-inputs.json'))['rows'][sample];ids=d['tokens']
            result['input_identity']={k:v for k,v in d.items() if k!='tokens'}
        tp=path('tokenizer.json');tok=tokenizer(str(tp),tp.stat().st_mtime_ns)
        result['input_tokens']=ids;result['input_text']=tok.decode(ids,skip_special_tokens=False)
    return result
