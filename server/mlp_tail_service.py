"""CPU-only public microscope for a verified, immutable native MLP tail atlas."""
from functools import lru_cache
import json
import os
from pathlib import Path
from typing import Literal
import numpy as np
from fastapi import APIRouter, HTTPException, Query
from shared.dist_protocol.mechanism_trace import file_sha, unbits

ROOT=Path(__file__).resolve().parents[1]
RUN='MLP-TAIL-GOODFIRE-20261108-v1'
EXTENSION_RUN='MLP-TAIL-B28-B35-ATLAS-v2'
EXTENSION_BLOCKS=[28,29,30,31]
router=APIRouter(prefix='/api/research/mlp-tail',tags=['native-mlp-tail'])
_checked={}


def folder():
    if os.environ.get('AI2050_MLP_TAIL_DIR'):return Path(os.environ['AI2050_MLP_TAIL_DIR']).resolve()
    if os.environ.get('AI2050_DIST_DIR'):
        return Path(os.environ['AI2050_DIST_DIR']).resolve()/'mlp_tail'/RUN
    return ROOT/'tests/glm5/result/mlp_tail_goodfire_v1'


def manifest():
    p=folder()/'manifest.json'
    if not p.is_file():raise HTTPException(404,'中心尚未导入最后四层 MLP 图谱；请部署研究数据，不能用演示替代。')
    d=json.loads(p.read_text(encoding='utf-8'))
    from deploy.mlp_tail_pack import validate_manifest
    try:validate_manifest(d)
    except (ValueError,KeyError,TypeError):raise HTTPException(409,'MLP图谱身份、必需字段或完成状态不符')
    return d


def extension_folder():
    if os.environ.get('AI2050_MLP_TAIL_EXTENSION_DIR'):
        return Path(os.environ['AI2050_MLP_TAIL_EXTENSION_DIR']).resolve()
    if os.environ.get('AI2050_MLP_TAIL_DIR'):
        return None
    if os.environ.get('AI2050_DIST_DIR'):
        return Path(os.environ['AI2050_DIST_DIR']).resolve()/'mlp_tail'/EXTENSION_RUN
    return ROOT/'tests/glm5/result/mlp_tail_b28_b31_v1'


def extension():
    root=extension_folder()
    if root is None or not (root/'manifest.json').is_file():return None
    base=manifest()
    expected={f'B{b}-{field}' for b in EXTENSION_BLOCKS for field in
              ('profile.json','cues.json','readout.json','pole-values.npy','W-gate.npy','W-up.npy','W-down.npy')}
    expected.update(f'B{b}-B{b+1}-{field}.npy' for b in EXTENSION_BLOCKS for field in
                    ('rho-discovery','rho-confirm','rho-holdout','weight-gate','weight-up'))
    expected.update(('plan.json','replay.json','report.json'))
    try:
        d=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
        expected_identity=('mlp_tail_extension.v1','real','complete',EXTENSION_RUN,EXTENSION_BLOCKS,
                           base['width'],base['positions'],base['model_snapshot'],file_sha(folder()/'manifest.json'))
        identity=tuple(d[key] for key in ('schema','kind','state','run_id','blocks','width','positions','model_snapshot','base_manifest_sha256'))
        names={item['path'] for item in d['files']}
        if identity!=expected_identity or not expected<=names or len(names)!=len(d['files']):
            raise ValueError('identity or files')
        for item in d['files']:
            name,checksum=item['path'],item['sha256']
            if not name or Path(name).name!=name or ':' in name or '\\' in name or name in ('.','..'):
                raise ValueError('filename')
            if item['bytes']<=0 or len(checksum)!=64 or any(c not in '0123456789abcdef' for c in checksum):
                raise ValueError('file metadata')
    except (OSError,ValueError,KeyError,TypeError):
        raise HTTPException(409,'B28–B31 扩展清单与原图谱不匹配或缺少必需产物')
    return root,d


def path(name,root=None,index=None):
    if root is None:
        extra=extension() if name.startswith(('B28-','B29-','B30-','B31-')) else None
        root,index=extra if extra else (folder(),manifest())
    record=next((f for f in index['files'] if f['path']==name),None)
    if not record:raise HTTPException(404,'所需字段未登记：'+name)
    p=(root/name).resolve()
    if not p.is_relative_to(root) or not p.is_file():raise HTTPException(404,'已登记数据尚未部署：'+name)
    stamp=(p.stat().st_size,p.stat().st_mtime_ns,record['sha256'])
    if _checked.get(str(p))!=stamp:
        if p.stat().st_size!=record['bytes'] or file_sha(p)!=record['sha256']:raise HTTPException(409,'图谱产物校验失败：'+name)
        _checked[str(p)]=stamp
    return p


@lru_cache(maxsize=16)
def _json(name,root,stamp):return json.loads(Path(root,name).read_text(encoding='utf-8'))


def read(name,root=None,index=None):
    p=path(name,root,index)
    return _json(name,str(p.parent),p.stat().st_mtime_ns)


def check(block,unit=None):
    first=28 if extension() else 32
    if not first<=block<=35 or (unit is not None and not 0<=unit<9728):
        raise HTTPException(422,f'本研究只覆盖B{first}–B35、U0–U9727')
    return first


@router.get('')
def overview():
    d=manifest();r=read('report.json')
    extra=extension()
    if extra:
        root,index=extra
        added=read('report.json',root,index)
        r={**r,'run_id':EXTENSION_RUN,'base_run_id':RUN,'blocks':index['blocks']+d['blocks'],
           'n_units':77824,'coverage':{**r['coverage'],'native_statistics':77824,
                     'native_parameter_locators':77824,'all_adjacent_pair_matrices':35},
           'replay':{**r['replay'],'run_id':EXTENSION_RUN,'checks':added['replay']['checks']+r['replay']['checks']},
           'extension':{**added,'manifest_sha256':file_sha(root/'manifest.json')},
           'prediction_blocks':d['blocks'],
           'conclusion':'B28–B35 全量原生观察与相邻层关联已接通。跨层递推、干预及能力检查沿用 B32–B35 的实测范围。',
           'limitations':[added['scope'],added['exposure'],*r['limitations']]}
    return {**r,'manifest_sha256':file_sha(folder()/'manifest.json'),'files':d['files'],
            'native_layout':'原生编号布局，不是语义空间距离；关联、权重对齐和原模型干预分别显示。'}


@router.get('/layers/{block}')
def layer(block:int):
    check(block);p=read(f'B{block}-profile.json');c=read(f'B{block}-cues.json')
    return {'run_id':p['run_id'],'model_snapshot':p['model_snapshot'],'block':block,'width':9728,
            'arrays':{**{f'{f}_{key}':p['fields'][f][key] for f in ['G','V','A'] for key in ['mean','std','min','max']},
                     **{f'sign_{i}':[row[i] for row in p['sign_regime_fractions']] for i in range(4)}},
            'cue_names':p['cue_names'],'cue_correlations':p['cue_correlations'],
            'candidate_labels':[r['description'] for r in c['units']],'semantic_confirmed':[False]*9728,'scope':p['scope']}


@lru_cache(maxsize=2)
def _tokenizer(root):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(path('tokenizer.json')))


@router.get('/units/{block}/{unit}')
def unit(block:int,unit:int,split:Literal['discovery','confirm','holdout']='holdout',
         field:Literal['G','V','A']='G',pole:Literal['positive','negative']='positive'):
    check(block,unit);p=read(f'B{block}-profile.json');d=read('documents.json')
    cue=read(f'B{block}-cues.json')['units'][unit]
    ids=p['fields'][field][split][pole+'_indices'][unit]
    tensor=np.load(path(f'B{block}-pole-values.npy'),mmap_mode='r')
    values=unbits(tensor[['G','V','A'].index(field),['discovery','confirm','holdout'].index(split),['positive','negative'].index(pole),unit])
    tok=_tokenizer(str(folder()));examples=[]
    for idx,value in zip(ids,values):
        di,slot=divmod(idx,16);doc=d['docs'][di];pos=d['positions'][slot]
        examples.append({'document_index':di,'token_position':pos,'token_id':doc['tokens'][pos],
            'G':float(value[0]),'V':float(value[1]),'A':float(value[2]),
            'regime':('+' if value[0]>=0 else '-')+('+' if value[1]>=0 else '-'),
            'prefix':tok.decode(doc['tokens'][:pos+1],skip_special_tokens=False),
            'following_not_causal':tok.decode(doc['tokens'][pos+1:min(1024,pos+33)],skip_special_tokens=False),
            'source':{k:doc.get(k) for k in ['doc_id','source','license','repo','revision','split']}})
    weights={}
    for key in ['gate','up','down']:
        arr=np.load(path(f'B{block}-W-{key}.npy'),mmap_mode='r')
        weights[key]=unbits(arr[:,unit] if key=='down' else arr[unit]).tolist()
    return {'run_id':p['run_id'],'model_snapshot':p['model_snapshot'],'block':block,'unit':unit,
            'statistics':{f:{k:p['fields'][f][k][unit] for k in ['mean','std','min','max']} for f in ['G','V','A']},
            'sign_regime_fractions':p['sign_regime_fractions'][unit],'candidate':cue,'examples':examples,'weights':weights,
            'cue_correlations':dict(zip(p['cue_names'],p['cue_correlations'][split][unit])),
            'parameter_readout':read(f'B{block}-readout.json')['units'][unit],
            'scope':'关键词是候选；原生参数与响应可复算，不等于语义解释已通过。'}


@router.get('/relations/{block}/{unit}')
def relations(block:int,unit:int,direction:Literal['incoming','outgoing']='outgoing',
              kind:Literal['rho-discovery','rho-confirm','rho-holdout','weight-gate','weight-up','causal']='rho-holdout',factor:float=2.):
    first=check(block,unit)
    source,target=(block,block+1) if direction=='outgoing' else (block-1,block)
    if source<first or target>35:raise HTTPException(422,'这条边界在当前研究范围之外')
    if kind=='causal':
        if direction!='outgoing':raise HTTPException(422,'因果响应按已干预的源单元查询')
        rows=read('path-cases.json')['rows'] if source>=32 else []
        all_indices=[i for i,r in enumerate(rows) if r['source']==source and r['unit']==unit and r['factor']==factor]
        valid_docs={rows[i]['doc_index'] for i in all_indices if rows[i]['condition']=='matched_noise' and rows[i]['norm_control_valid']}
        indices=[i for i in all_indices if rows[i]['condition']=='native_unit' and rows[i]['norm_control_valid'] and rows[i]['doc_index'] in valid_docs]
        paired_docs={rows[i]['doc_index'] for i in indices}
        control=[i for i in all_indices if rows[i]['condition']=='matched_noise' and rows[i]['doc_index'] in paired_docs]
        if not indices:return {'kind':'not_tested','source':source,'target':target,'unit':unit,'values':None,'n_valid':0,'scope':'本单元/剂量未完成有效匹配干预，不能从相关升级。'}
        arr=np.load(path('path-target-deltas.npy'),mmap_mode='r')
        values=np.asarray(arr[indices],dtype=np.float32).mean(0)
        return {'kind':'finite_native_intervention','source':source,'target':target,'unit':unit,'direction':direction,'factor':factor,'values':values.tolist(),
                'control_values':np.asarray(arr[control],dtype=np.float32).mean(0).tolist(),'n_valid':len(indices),
                'scope':'相同源单元、有限剂量、96既有自然语境中的有效匹配子集；目标为全部9728单元，非单一语义因果机制。'}
    arr=np.load(path(f'B{source}-B{target}-{kind}.npy'),mmap_mode='r')
    values=np.asarray(arr[unit] if direction=='outgoing' else arr[:,unit],dtype=np.float32)
    return {'kind':'observation' if kind.startswith('rho') else 'parameter_alignment','source':source,'target':target,
            'unit':unit,'direction':direction,'values':values.tolist(),'full_units':9728,
            'scope':'Pearson为自然共同响应；权重余弦不含归一化和注意力。二者都不是因果边。'}
