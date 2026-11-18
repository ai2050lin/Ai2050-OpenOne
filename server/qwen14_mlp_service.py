"""Read-only, snapshot-bound Qwen3-14B native MLP atlas. No GPU or model load."""
from functools import lru_cache, partial
import json
import os
from pathlib import Path
from typing import Literal
import numpy as np
from fastapi import APIRouter, HTTPException, Query
from shared.dist_protocol.mechanism_trace import file_sha, unbits

ROOT=Path(__file__).resolve().parents[1]
from deploy.qwen14_mlp_pack import RUN, RUN_V2, validate
SNAPSHOT='1e970d119494affd09689ec3dd25aa30587dfcbb16d873b40fb917016fe2a2ce'
WIDTH=17408
router=APIRouter(prefix='/api/research/qwen14-mlp',tags=['qwen14-native-mlp'])
_checked={}


def folder(run=None):
    if run not in (None,RUN,RUN_V2):raise HTTPException(404,'未知的14B研究版本')
    explicit=os.environ.get('AI2050_QWEN14_MLP_DIR')
    if explicit:
        target=Path(explicit).resolve()
        if run is None:return target
        mp=target/'manifest.json'
        if mp.is_file() and json.loads(mp.read_text(encoding='utf-8')).get('run_id')==run:return target
    runs=[run] if run else [RUN_V2,RUN]
    if os.environ.get('AI2050_DIST_DIR'):
        base=Path(os.environ['AI2050_DIST_DIR']).resolve()/'native_atlases'
        for r in runs:
            if (base/r/'manifest.json').is_file():return base/r
        if not explicit:return base/runs[0]
    local={RUN:ROOT/'tests/glm5/result/qwen14_mlp_atlas_v1',RUN_V2:ROOT/'tests/glm5/result/qwen14_client_v2/native'}
    for r in runs:
        if (local[r]/'manifest.json').is_file():return local[r].resolve()
    return local[runs[0]].resolve()


def manifest(run=None):
    p=folder(run)/'manifest.json'
    if not p.is_file():raise HTTPException(404,'Qwen3-14B 原生研究包未部署；请安装所选版本原生包。')
    d=json.loads(p.read_text(encoding='utf-8'))
    try:validate(d)
    except (ValueError,KeyError,TypeError) as exc:raise HTTPException(409,'Qwen3-14B研究包身份、完整性或完成状态不符') from exc
    if run is not None and d['run_id']!=run:raise HTTPException(409,'研究版本不匹配')
    return d


def path(name,run=None):
    d=manifest(run);record=next((f for f in d['files'] if f['path']==name),None)
    if record is None:raise HTTPException(404,'研究文件未登记')
    if Path(name).name!=name or '\\' in name or ':' in name:raise HTTPException(409,'非法研究路径')
    p=(folder(run)/name).resolve()
    if p.parent!=folder(run) or not p.is_file():raise HTTPException(404,'原生文件未安装：'+name)
    stamp=(p.stat().st_size,p.stat().st_mtime_ns,record['sha256'])
    if _checked.get(str(p))!=stamp:
        if p.stat().st_size!=record['bytes'] or file_sha(p)!=record['sha256']:raise HTTPException(409,'原生文件完整性校验失败：'+name)
        _checked[str(p)]=stamp
    return p


@lru_cache(maxsize=18)
def _read(p,stamp):return json.loads(Path(p).read_text(encoding='utf-8'))


def read(name,run=None):
    p=path(name,run);doc=_read(str(p),p.stat().st_mtime_ns)
    if isinstance(doc,dict) and doc.get('model_snapshot',SNAPSHOT)!=SNAPSHOT:
        raise HTTPException(409,'研究文件内的模型快照不一致')
    return doc


def check(block=None,unit=None,snapshot=None,run=None):
    if snapshot and snapshot!=SNAPSHOT:raise HTTPException(409,'选中模型与Qwen3-14B研究快照不一致')
    meta=manifest(run)
    if block is not None and block not in meta['blocks']:raise HTTPException(422,'该版本没有此层的采集数据')
    if unit is not None and not 0<=unit<WIDTH:raise HTTPException(422,'单元范围是U0–U17407')
    return meta


def access(meta):
    return partial(read,run=meta['run_id']),partial(path,run=meta['run_id'])


def legacy_name(meta,block,name):
    return 'legacy-'+name if meta['run_id']==RUN_V2 and block<36 else name


@router.get('')
def overview(snapshot:str|None=None,run:str|None=None):
    meta=check(snapshot=snapshot,run=run);read,_=access(meta)
    return {**read('report.json'),'manifest_sha256':file_sha(folder(meta['run_id'])/'manifest.json')}


@router.get('/layers/{block}')
def layer(block:int,snapshot:str|None=None,run:str|None=None):
    meta=check(block,snapshot=snapshot,run=run);read,path=access(meta);d=read(f'B{block}-profile.json')
    extra={}
    if meta['run_id']==RUN_V2:
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
    meta=check(block,unit,snapshot,run);read,path=access(meta);p=read(f'B{block}-profile.json');docs=read('documents.json')
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
    pred=read(f'B{block}-prediction.json')['fields']
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
    if meta['run_id']==RUN_V2:
        with np.load(path(f'B{block}-regimes.npz')) as z:
            extra['regimes']={k:np.asarray(z[k][unit]).tolist() for k in z.files}
        with np.load(path(f'B{block}-scaled-prediction.npz')) as z:
            extra['secondary_standardized_prediction']={k:np.asarray(z[k][unit]).tolist() for k in z.files}
        if block>28:
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
    meta=check(block,unit,snapshot,run);read,path=access(meta)
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
        'values':result.tolist(),'full_units':WIDTH,'scope':'全17,408个相邻单元；相关和静态权重对齐均不是因果连接。'}


@router.get('/interventions/{stage}/{block}/{unit}')
def interventions(stage:Literal['natural','capability'],block:int,unit:int,
                  sample:int|None=Query(None,ge=0),snapshot:str|None=None,run:str|None=None):
    meta=check(block,unit,snapshot,run);read,path=access(meta)
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
