"""Node-local learned dictionaries and all native associations; never semantic labels."""
import json
from functools import lru_cache
from pathlib import Path
import numpy as np
from fastapi import APIRouter,HTTPException,Query
from shared.dist_protocol.mechanism_trace import file_sha,unbits
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'tests/glm5/result/mlp_atlas_dictionary_v1'
router=APIRouter(prefix='/mlp-dictionary',tags=['whole-layer-dictionaries'])
_verified={}

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def summary(block):
    if not 0<=block<36:raise HTTPException(422,'层块越界')
    p=OUT/f'dictionaries/B{block:02d}-result.json'
    if not p.exists():raise HTTPException(409,'当前层的字典训练和留出评估尚未提交')
    d=read(p)
    for suffix,key in [('transcoder','dictionary_sha256'),('associations','association_sha256')]:
        q=p.parent/f'B{block:02d}-{suffix}.npz';stamp=(q.stat().st_size,q.stat().st_mtime_ns)
        if _verified.get(str(q))!=stamp:
            if file_sha(q)!=d[key]:raise HTTPException(409,'字典产物哈希不符')
            _verified[str(q)]=stamp
    return d,p.parent

@router.get('/manifest')
def manifest():
    p=OUT/'dictionary-result.json'
    if p.exists():return read(p)
    layers=[read(p) for p in sorted((OUT/'dictionaries').glob('B??-result.json'))]
    return {'schema':'mlp_dictionary_progress.v1','state':'training','n_layers':36,'completed_layers':len(layers),'layers':layers}

@router.get('/layers/{block}')
def layer(block:int):
    d,p=summary(block)
    with np.load(p/f'B{block:02d}-associations.npz') as z:
        data={k:z[k].tolist() for k in ['discovery_partner','discovery_partner_correlation','confirm_partner_correlation','lockbox_partner_correlation']}
    return {**d,**data,'d_ff':9728,'feature_count':d['m'],'meaning':'correlation partner selected on discovery; not neuron semantic meaning'}

@router.get('/units/{block}/{unit}')
def unit(block:int,unit:int):
    if not 0<=unit<9728:raise HTTPException(422,'原生单元越界')
    d,p=summary(block)
    with np.load(p/f'B{block:02d}-associations.npz') as z:
        data={'all_discovery_correlations':z['all_discovery_correlations'][unit].tolist()}
        data.update({k:z[k][unit].item() for k in ['discovery_partner','discovery_partner_correlation','confirm_partner_correlation','lockbox_partner_correlation']})
    return {'block':block,'unit':unit,'model_snapshot':d['model_snapshot'],'source_sha256':d['association_sha256'],**data}

@lru_cache(maxsize=1)
def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained('D:/develop/model/Qwen/Qwen3-4B',local_files_only=True)

def contexts(indices,values,positions):
    source=ROOT/'tests/glm5/result/native_mlp_long_v1/input-tokens.npy';pr=read(ROOT/'research/mlp_atlas/dictionary_v1/prereg.research.json');stamp=(source.stat().st_size,source.stat().st_mtime_ns)
    if _verified.get(str(source))!=stamp:
        if file_sha(source)!=pr['source_tokens_sha256']:raise HTTPException(409,'自然原片的token源哈希不符')
        _verified[str(source)]=stamp
    tokens=np.load(source,mmap_mode='r');t=tokenizer();examples=[]
    for idx,value in zip(indices,values):
        if idx<0 or value<=0:continue
        doc,slot=divmod(int(idx),16);pos=positions[slot];start=max(0,pos-64);end=min(1024,pos+33)
        examples.append({'document_index':doc,'slot':slot,'token_position':pos,'activation':float(value),'prefix_context':t.decode(tokens[doc,start:pos+1].tolist()),'following_context_not_causal_input':t.decode(tokens[doc,pos+1:end].tolist())})
    return examples

@router.get('/features/{block}/{feature}')
def feature(block:int,feature:int):
    d,p=summary(block)
    if not 0<=feature<d['m']:raise HTTPException(422,'字典特征越界；特征不是原生单元')
    man=read(OUT/'bank/manifest.json')
    with np.load(p/f'B{block:02d}-transcoder.npz') as z:weights={'encoder':z['W_enc'][:,feature].tolist(),'decoder':z['W_dec'][feature].tolist(),'bias':float(z['b_enc'][feature]),'scales':z['scales'].tolist()}
    with np.load(p/f'B{block:02d}-associations.npz') as z:
        poles={split:contexts(z[split+'_feature_pole_indices'][:,feature],z[split+'_feature_pole_values'][:,feature],man['positions']) for split in ['discovery','confirm','lockbox']}
        correlations=z['all_discovery_correlations'][:,feature].tolist()
    return {'schema':'mlp_dictionary_feature.v1','model_snapshot':d['model_snapshot'],'block':block,'feature':feature,'source_sha256':d['dictionary_sha256'],'association_sha256':d['association_sha256'],'weights':weights,'natural_examples':poles,'all_native_unit_discovery_correlations':correlations,'scope':'Learned sparse feature; contexts and correlations are descriptive. No semantic label, individual necessity or causal edge asserted.'}

@router.get('/samples/{document}/{slot}')
def natural_sample(document:int,slot:int,block:int=Query(0,ge=0,le=35)):
    if not 0<=document<2048 or not 0<=slot<16:raise HTTPException(422,'自然样本越界')
    d,p=summary(block);man=read(OUT/'bank/manifest.json');out={}
    for kind in ['input','output']:
        path=OUT/f'bank/B{block:02d}-{kind}.npy';meta=next(f for f in man['files'] if f['path']==path.name);stamp=(path.stat().st_size,path.stat().st_mtime_ns)
        if _verified.get(str(path))!=stamp:
            if file_sha(path)!=meta['sha256']:raise HTTPException(409,'自然原场哈希不符')
            _verified[str(path)]=stamp
        out[kind]=unbits(np.load(path,mmap_mode='r')[document,slot]).tolist()
    # CPU evaluation of the frozen learned map. Original native G/V/A were not captured here.
    import torch
    from shared.dist_protocol.transcoder import TopKTranscoder
    with np.load(p/f'B{block:02d}-transcoder.npz') as z:t=TopKTranscoder.from_state(z)
    with torch.inference_mode():
        x=torch.tensor(out['input'])[None];a=t.encode(x);y=t.decode(a)
    return {'kind':'real','model_snapshot':d['model_snapshot'],'block':block,'document':document,'slot':slot,'token_position':man['positions'][slot],**out,'dictionary_activation':a[0].tolist(),'dictionary_prediction':y[0].tolist(),'source_dictionary_sha256':d['dictionary_sha256'],'scope':'input/output are native BF16; latent activation and prediction recomputed on CPU FP32, not original native G/V/A or attention'}
