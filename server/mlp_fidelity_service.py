"""Read-only versioned fidelity, sparse features and native tail evidence."""
import json
from pathlib import Path
from functools import lru_cache
import numpy as np
from fastapi import APIRouter,HTTPException,Query
from shared.dist_protocol.mechanism_trace import file_sha,unbits
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'tests/glm5/result/mlp_fidelity_v2';PUBLIC=ROOT/'research/mlp_fidelity/v2';router=APIRouter(prefix='/mlp-fidelity/v2',tags=['fidelity-natural-attributes-causal']);_verified={}
def read(p):
    if not p.exists():raise HTTPException(409,'该研究步骤尚未完成')
    return json.loads(p.read_text(encoding='utf-8'))
def verify(p,sha):
    stamp=(p.stat().st_size,p.stat().st_mtime_ns)
    if _verified.get(str(p))!=stamp:
        if file_sha(p)!=sha:raise HTTPException(409,'原始研究产物哈希不符')
        _verified[str(p)]=stamp
def native(p,meta):
    verify(p,meta['sha256']);a=np.load(p,mmap_mode='r')
    if a.dtype!=np.uint16 or list(a.shape)!=meta['shape']:raise HTTPException(409,'原生BF16数组格式或维度不符')
    return a
def trained(block):
    if not 0<=block<36:raise HTTPException(422,'原生层越界')
    d=read(OUT/f'trained/B{block:02d}-result.json');p=ROOT/d['state_path'];verify(p,d['state_sha256']);return d,p
def stats(block):
    d=read(OUT/f'fresh-evaluation/B{block:02d}-result.json');p=OUT/f'fresh-evaluation/B{block:02d}-statistics.npz';verify(p,d['statistics_sha256']);return d,p
@lru_cache(maxsize=1)
def tokenizer():
    from transformers import AutoTokenizer
    from shared.dist_protocol.natural_register_runner import snapshot
    model=Path('D:/develop/model/Qwen/Qwen3-4B')
    if snapshot(model)!=read(OUT/'plan.json')['model_snapshot']:raise HTTPException(409,'模型快照不符，停止原片解码')
    return AutoTokenizer.from_pretrained(model,local_files_only=True)

@router.get('/layers/{block}')
def layer(block:int):
    d,_=trained(block);fresh,p=stats(block)
    with np.load(p) as z:v={k:z[k].tolist() for k in ['confirm_feature_firings','lockbox_feature_firings']}
    sem=OUT/f'semantics-v2/B{block:02d}-result.json'
    return {'block':block,'model_snapshot':d['model_snapshot'],'candidate':d['candidate'],'fresh':fresh,'semantic':read(sem) if sem.exists() else None,**v}

def examples(indices,values):
    docs=read(OUT/'fresh-documents.json')['natural'];positions=read(OUT/'natural-bank/manifest.json')['positions'];t=tokenizer();result=[]
    for index,value in zip(indices,values):
        if index<0 or value<=0:continue
        doc,slot=divmod(int(index),8);pos=positions[slot];r=docs[doc];result.append({'document':doc,'slot':slot,'position':pos,'value':float(value),'source':{k:v for k,v in r.items() if k!='tokens'},'prefix':t.decode(r['tokens'][max(0,pos-64):pos+1]),'following_not_causal_input':t.decode(r['tokens'][pos+1:pos+33])})
    return result

@router.get('/features/{block}/{feature}')
def feature(block:int,feature:int):
    d,p=trained(block);m=d['candidate']['m']
    if not 0<=feature<m:raise HTTPException(422,'字典特征索引越界；不能与原生神经元混用')
    fresh,sp=stats(block)
    with np.load(p) as z:weights={'encoder':z['W_enc'][:,feature].tolist(),'decoder':z['W_dec'][feature].tolist(),'encoder_bias':float(z['b_enc'][feature]),'pre_bias':z['b_pre'].tolist(),'output_bias':z['b_dec'].tolist(),'scales':z['scales'].tolist()}
    with np.load(sp) as z:contexts={s:examples(z[s+'_feature_pole_indices'][:,feature],z[s+'_feature_pole_values'][:,feature]) for s in ['confirm','lockbox']}
    semfile=OUT/f'semantics-v2/B{block:02d}-result.json';response=None;correlation=None
    if semfile.exists():
        sem=read(semfile);cp=OUT/f'semantics-v2/B{block:02d}-native-feature-correlation.npz';verify(cp,sem['native_correlation_sha256'])
        with np.load(cp) as z:correlation=z['correlation'][:,feature].tolist()
        response=next((r for r in sem['features'] if r['feature']==feature),None)
    return {'schema':'mlp_fidelity_feature.v2','model_snapshot':d['model_snapshot'],'block':block,'feature':feature,'candidate':d['candidate'],'weights':weights,'natural_examples':contexts,'response':response,'native_correlations':correlation,'dictionary_sha256':d['state_sha256'],'fresh_fidelity_gate':fresh['both_fidelity_gates'],'scope':'SAE encoder observes native MLPoutput; sparse dictionary atom is not an original neuron. Correlations andstrongcontextsmustnotbe treated as unique semantic meaning.'}

@router.get('/units/{block}/{unit}')
def unit(block:int,unit:int):
    if not 0<=unit<9728:raise HTTPException(422,'原生单元越界')
    sem=read(OUT/f'semantics-v2/B{block:02d}-result.json');p=OUT/f'semantics-v2/B{block:02d}-native-feature-correlation.npz';verify(p,sem['native_correlation_sha256'])
    with np.load(p) as z:c=z['correlation'][unit].tolist()
    return {'block':block,'unit':unit,'correlations':c,'scope':'all4096discoveryassociations; not a causal map'}

def dictionary_values(block,x,y,latent=None):
    import torch
    from shared.dist_protocol.transcoder import TopKTranscoder
    d,p=trained(block)
    with np.load(p) as z:t=TopKTranscoder.from_state(z)
    with torch.inference_mode():
        a=t.encode(torch.tensor(y if d['candidate']['kind']=='sae_output' else x,dtype=torch.float32)[None]) if latent is None else torch.tensor(latent,dtype=torch.float32)[None]
        pred=t.decode(a)
    return a[0].tolist(),pred[0].tolist()

@router.get('/samples/natural/{document}/{slot}')
def natural(document:int,slot:int,block:int=Query(35,ge=0,le=35)):
    if not 0<=document<512 or not 0<=slot<8:raise HTTPException(422,'自然样本越界')
    man=read(OUT/'natural-bank/manifest.json');values={}
    for k in ['input','output']:
        p=OUT/f'natural-bank/B{block:02d}-{k}.npy';meta=next(r for r in man['files'] if r['path']==p.name);values[k]=unbits(native(p,meta)[document,slot]).tolist()
    a,pred=dictionary_values(block,values['input'],values['output']);doc=read(OUT/'fresh-documents.json')['natural'][document];pos=man['positions'][slot]
    return {'kind':'real','model_snapshot':man['model_snapshot'],'block':block,'document':document,'slot':slot,'position':pos,'input':values['input'],'output':values['output'],'latent':a,'prediction':pred,'prefix':tokenizer().decode(doc['tokens'][:pos+1]),'source':{k:v for k,v in doc.items() if k!='tokens'},'scope':'MLPinput/output nativeBF16; latent/predictionCPUFP32recomputed. No GVA/attention inthisnaturalbank.'}

@router.get('/samples/controlled/{index}')
def controlled(index:int,block:int=Query(35,ge=0,le=35),stage:int=Query(1,ge=0,le=1)):
    from server.mlp_bank_service import sample,native_field
    man,p,row=sample('MLP-FIDELITY-20261104-v2',index);y=native_field(man,p,index,block,stage,'mlp_output');sem=read(OUT/f'semantics-v2/B{block:02d}-result.json');p=OUT/f'semantics-v2/B{block:02d}-latent.npy';verify(p,sem['latent_sha256']);z=np.load(p,mmap_mode='r')
    if z.dtype!=np.float32 or z.shape!=(man['n_items'],2,4096):raise HTTPException(409,'登记的字典激活阵列格式不符')
    a,pred=dictionary_values(block,[],y,latent=z[index,stage]);return {'sample':row,'block':block,'stage':stage,'latent':a,'native_output':y.tolist(),'prediction':pred,'latent_source':'registered_FP32_array','scope':'actualnative query; SAEobservesactualMLPoutput; registeredlatent andCPUFP32reconstruction; not a functional label'}

@router.get('/samples/tail/{part}/{index}/{slot}')
def tail(part:str,index:int,slot:int):
    if part not in ['natural','generated','capability']:raise HTTPException(422,'未知尾部研究类型')
    folder=OUT/{'natural':'tail-natural','generated':'tail-generated-v2','capability':'tail-capability-v3'}[part];result=read(folder/'result.json')
    if result['status']!='complete':raise HTTPException(409,'本研究未满足运行门')
    if not 0<=index<len(result['rows']):raise HTTPException(422,'尾部样本越界')
    vals={}
    for meta in result['native_files']:
        if not 0<=slot<meta['shape'][1]:raise HTTPException(422,'尾部位置越界')
        p=folder/meta['path'];vals[meta['path'].split('.')[0]]=unbits(native(p,meta)[index,slot]).tolist()
    a,pred=dictionary_values(35,[],vals['output'])
    return {'kind':'real','model_snapshot':result['model_snapshot'],'block':35,'part':part,'index':index,'slot':slot,'row':result['rows'][index],'fields':vals,'latent':a,'prediction':pred,'final_hidden_source':'recomputed_from_native_r_plus_y_with_exact_logits_check' if part=='natural' else 'observed_final_norm_output','scope':'B35residual/A/output/finalh only; no inventedembedding/head/G/V'}
