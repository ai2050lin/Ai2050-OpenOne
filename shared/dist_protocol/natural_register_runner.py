"""natural_register_intervention.v1: frozen family shards, real CUDA only.

Standalone runner copied into a template bundle. Full native arrays remain on
the node. Uploaded execution checks do not establish style or mechanism claims.
"""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import json
import math
import os
import platform
import shutil
import time
from pathlib import Path

JOB_KIND='natural_register'
SCHEMA='natural_register_intervention.v1'
CONDITIONS=['base','target','random_direction','random_columns']
PATTERNS=('*.safetensors','*.bin','*.pt','*.gguf','config.json','generation_config.json',
    'tokenizer*','vocab*','merges.txt','special_tokens_map.json','*.model','*.tiktoken')


def canonical(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(x):return hashlib.sha256(canonical(x)).hexdigest()
def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def save(path,doc):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    raw=canonical(doc)
    if p.exists():
        if p.read_bytes()!=raw:raise ValueError('immutable checkpoint differs: '+p.name)
        return
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_bytes(raw);temp.replace(p)
def is_sha(x):return isinstance(x,str) and len(x)==64 and all(c in '0123456789abcdef' for c in x)


def validate_corpus(c):
    if c.get('schema')!=SCHEMA or c.get('job_kind')!=JOB_KIND:raise ValueError('invalid natural task schema')
    if not is_sha(c.get('model_snapshot')) or c.get('precision')!='bf16':raise ValueError('full bf16 snapshot required')
    if c.get('conditions')!=CONDITIONS:raise ValueError('four fixed weight conditions required')
    items=c.get('items');seen=set();pairs={};family_shards={}
    if not isinstance(items,list) or not 1<=len(items)<=10000:raise ValueError('invalid corpus size')
    if type(c.get('shards')) is not int or not 1<=c['shards']<=256:raise ValueError('invalid shard count')
    for r in items:
        if r['id'] in seen:raise ValueError('duplicate sample')
        seen.add(r['id'])
        if r['register'] not in ['acad','colloq'] or not isinstance(r['facts'],dict):raise ValueError('invalid sample factors')
        if type(r.get('shard_index')) is not int or not 0<=r['shard_index']<c['shards']:raise ValueError('invalid shard assignment')
        if not r.get('token_ids') or any(type(t) is not int or t<0 for t in r['token_ids']):raise ValueError('invalid token IDs')
        if len(r['token_ids'])!=r['tokens'] or not 512<=r['tokens']<=4096:raise ValueError('active token length differs')
        pairs.setdefault((r['family_id'],r['frame'],r['tokens']),[]).append(r)
        family_shards.setdefault(r['family_id'],set()).add(r['shard_index'])
    for group in pairs.values():
        if len(group)!=2 or {r['register'] for r in group}!={'acad','colloq'}:raise ValueError('incomplete style pair')
        a,b=group
        if a['facts']!=b['facts'] or a['text'].replace(a['style_request'],'')!=b['text'].replace(b['style_request'],''):raise ValueError('style pair changes other content')
    if any(len(x)!=1 for x in family_shards.values()):raise ValueError('a family must stay in one shard')
    if set(r['shard_index'] for r in items)!=set(range(c['shards'])):raise ValueError('empty shard')
    if not 32<=c.get('max_new_tokens',0)<=512 or not 1<=c.get('batch_size',0)<=4:raise ValueError('invalid runtime budget')
    e=c.get('edit',{})
    if e.get('alpha')!=6 or e.get('seed')!=27732026:raise ValueError('this runner only supports the inherited fixed edit')
    if sorted(map(int,e.get('directions',{})))!=list(range(18,28)):raise ValueError('ten pinned block directions required')
    for layer,u in e['directions'].items():
        ix=e['columns'].get(layer,[]);energy=e['energy'].get(layer,[])
        if len(u)!=2560 or len(energy)!=9728 or len(ix)!=512 or len(set(ix))!=512:raise ValueError('invalid edit shapes')
        if any(type(j) is not int or not 0<=j<9728 for j in ix):raise ValueError('column out of range')
        if any(not math.isfinite(float(x)) for x in u+energy) or abs(sum(x*x for x in u)-1)>.001:raise ValueError('invalid direction/energy')
    if c.get('source_edit_sha256')!=digest(e):raise ValueError('edit fingerprint differs')
    if c.get('canary_recipe_sha256')!=digest(c.get('canary_item')):raise ValueError('canary fingerprint differs')
    return {'families':len(family_shards),'samples':len(items),'pairs':len(pairs)}


def shard_items(c,shard):
    index,total=map(int,shard.split('/'))
    if total!=c['shards'] or not 0<=index<total:raise ValueError('shard differs from frozen job')
    return sorted([r for r in c['items'] if r['shard_index']==index],key=lambda r:(r['tokens'],r['id']))


def validate_result(c,shard,summary,samples,canary):
    expected={r['id']:r for r in shard_items(c,shard)}
    if summary.get('protocol')!=SCHEMA or summary.get('status')!='real':raise ValueError('real natural task result required')
    if summary.get('model_snapshot')!=c['model_snapshot'] or summary.get('precision')!='bf16':raise ValueError('result model mismatch')
    if summary.get('source_edit_sha256')!=c['source_edit_sha256'] or not summary.get('weights_restored'):raise ValueError('edit or restoration mismatch')
    if summary.get('corpus_sha256')!=digest(c) or summary.get('shard')!=shard:raise ValueError('result corpus/shard mismatch')
    if summary.get('canary_recipe_sha256')!=c['canary_recipe_sha256'] or not canary.get('entries'):raise ValueError('missing fixed canary')
    for layers in canary['entries'].values():
        for stats in layers.values():
            if not stats.get('head') or any(not math.isfinite(float(x)) for x in stats['head']+[stats['norm']]):raise ValueError('invalid canary values')
    outputs=samples.get('outputs',[]);keys=set()
    for x in outputs:
        key=(x.get('sample_id'),x.get('condition'))
        if key in keys or key[0] not in expected or key[1] not in CONDITIONS:raise ValueError('unexpected/duplicate output')
        keys.add(key);r=expected[key[0]]
        if x.get('input_sha256')!=digest(r['token_ids']):raise ValueError('input tokens differ')
        if not isinstance(x.get('text'),str) or not x.get('token_ids') or len(x['token_ids'])>c['max_new_tokens']:raise ValueError('invalid generation')
    if keys!={(sid,cond) for sid in expected for cond in CONDITIONS}:raise ValueError('incomplete four-condition shard')
    return {'execution_validated':True,'mechanism_evidence_promoted':False,'samples':len(expected),'outputs':len(keys),
        'review_status':'external_ai_calibration_pending','scope':'execution/identity checks only; no automatic style or causal grade'}


def snapshot(folder):
    root=Path(folder);files={}
    for p in sorted({p for pat in PATTERNS for p in root.glob(pat) if p.is_file()},key=lambda p:p.name):
        files[p.name]={'sha256':file_sha(p),'size':p.stat().st_size}
    if not files:raise ValueError('missing model files')
    return hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()


def run(c,shard,model_path,out,native_root,design_sha):
    import numpy as np
    import torch
    import transformers
    from transformers import AutoTokenizer,AutoModelForCausalLM
    validate_corpus(c);rows=shard_items(c,shard);started=time.time()
    if not torch.cuda.is_available():raise RuntimeError('real CUDA required; no synthetic fallback')
    if snapshot(model_path)!=c['model_snapshot']:raise ValueError('model snapshot differs')
    if shutil.disk_usage(native_root).free < 12*(1<<30):raise RuntimeError('less than12GiB free; native capture not started')
    torch.manual_seed(c['edit']['seed']);torch.set_num_threads(4)
    tok=AutoTokenizer.from_pretrained(model_path,local_files_only=True);tok.padding_side='left'
    model=AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True,dtype=torch.bfloat16,device_map='cuda:0',attn_implementation='sdpa').eval()
    if model.config.hidden_size!=2560 or model.config.intermediate_size!=9728:raise ValueError('runner edit requires pinned Qwen3-4B shape')
    def batch(ids):return {k:v.cuda() for k,v in tok.pad({'input_ids':ids},padding=True,return_tensors='pt').items()}
    # Re-tokenize all actual prompts, never trust a node-supplied token count.
    if any(tok.encode(r['prompt'],add_special_tokens=False)!=r['token_ids'] for r in rows):raise ValueError('tokenizer/input mismatch')
    blocks=list(range(18,28));base={l:model.model.layers[l].mlp.down_proj.weight.detach().clone() for l in blocks}
    env={'torch':torch.__version__,'transformers':transformers.__version__,'gpu':torch.cuda.get_device_name(),
        'precision':'bf16','attention':'sdpa','generation':'greedy_no_thinking','python':platform.python_version()}
    provenance={**c['provenance'],'model_path':str(model_path),'model_revision':c['model_snapshot'],
        'seed':c['edit']['seed'],'runner_sha256':file_sha(__file__)}
    native=Path(native_root)/design_sha/shard.replace('/','-');native.mkdir(parents=True,exist_ok=True)
    checkpoints=native/'checkpoints';outputs=[];native_files=[];edit_metadata={}
    can=c['canary_item']
    with torch.inference_mode():o=model(**batch([can['token_ids']]),output_hidden_states=True,logits_to_keep=1,use_cache=False)
    canary={'schema_version':'canary.v1','entries':{can['id']:{str(l):{'norm':float(h[0,-1].float().norm()),
        'head':h[0,-1,:16].float().cpu().tolist(),'dim':2560} for l,h in enumerate(o.hidden_states)}},'provenance':provenance}
    del o

    @contextlib.contextmanager
    def edited(condition):
        details=[]
        try:
            if condition!='base':
                with torch.no_grad():
                    for l in blocks:
                        u=torch.tensor(c['edit']['directions'][str(l)],device='cuda');ix=np.array(c['edit']['columns'][str(l)],int)
                        if condition=='random_columns':
                            rng=np.random.default_rng(c['edit']['seed']+500+l);energy=np.asarray(c['edit']['energy'][str(l)])
                            picked=[]
                            for bucket in np.array_split(np.argsort(energy,kind='stable'),16):
                                picked.extend(rng.choice(bucket,int(np.isin(bucket,ix).sum()),replace=False).tolist())
                            ix=np.asarray(picked,int)
                        idx=torch.tensor(ix,device='cuda');w=base[l][:,idx].float();target=u[:,None]*(u@w)[None,:]
                        if condition=='random_direction':
                            gen=torch.Generator(device='cuda').manual_seed(c['edit']['seed']+l)
                            v=torch.randn(u.shape,generator=gen,device='cuda');v/=v.norm()
                            delta=v[:,None]*(v@w)[None,:];delta*=target.norm()/delta.norm().clamp_min(1e-12)
                        else:delta=target
                        value=(w-c['edit']['alpha']*delta).to(torch.bfloat16)
                        model.model.layers[l].mlp.down_proj.weight[:,idx]=value
                        details.append({'block':l,'columns':ix.tolist(),'addressed_params':w.numel(),
                            'changed_params':int((value!=base[l][:,idx]).sum()),'relative_norm':float((value.float()-w).norm()/base[l].float().norm())})
                        details[-1]['edited_bits_sha256']=hashlib.sha256(value.contiguous().view(torch.uint16).cpu().numpy().tobytes()).hexdigest()
            yield details
        finally:
            with torch.no_grad():
                for l in blocks:model.model.layers[l].mlp.down_proj.weight.copy_(base[l])
            if not all(torch.equal(model.model.layers[l].mlp.down_proj.weight,base[l]) for l in blocks):raise RuntimeError('weight restoration failed')

    for condition in CONDITIONS:
        with edited(condition) as details:
            edit_metadata[condition]=details
            # Same lengths within a batch make all prefix positions unambiguous.
            for length in sorted({r['tokens'] for r in rows}):
                group=[r for r in rows if r['tokens']==length]
                for start in range(0,len(group),c['batch_size']):
                    if time.time()-started>c.get('timeout_seconds',5400):raise TimeoutError('frozen time budget exceeded; checkpoints retained')
                    if shutil.disk_usage(native).free < 4*(1<<30):raise RuntimeError('native disk reserve reached')
                    chunk=group[start:start+c['batch_size']];stem=f'{condition}-L{length}-B{start:04d}'
                    checkpoint=checkpoints/(stem+'.json');capture=condition in ['base','target']
                    if checkpoint.exists():
                        d=read(checkpoint)
                        if d['corpus_sha256']!=digest(c) or d['runner_sha256']!=file_sha(__file__):raise ValueError('checkpoint identity differs')
                        if d.get('native'):
                            if file_sha(native/d['native']['path'])!=d['native']['sha256']:raise ValueError('native checkpoint corrupted')
                            native_files.append(d['native'])
                        outputs.extend(d['outputs']);continue
                    positions=[0,length//4,length//2,length-2,length-1];calls={l:[] for l in blocks};handles=[]
                    if capture:
                        for l in blocks:
                            def hook(module,args,l=l):
                                x=args[0];chosen=x[:,positions,:] if x.shape[1]>1 else x[:,-1:,:]
                                calls[l].append(chosen.detach().contiguous().view(torch.uint16).cpu().numpy().copy())
                            handles.append(model.model.layers[l].mlp.down_proj.register_forward_pre_hook(hook))
                    try:
                        inputs=batch([r['token_ids'] for r in chunk])
                        with torch.inference_mode():g=model.generate(**inputs,max_new_tokens=c['max_new_tokens'],do_sample=False,use_cache=True,
                            pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id,return_dict_in_generate=True,
                            output_hidden_states=capture)
                    finally:
                        for handle in handles:handle.remove()
                    generated=g.sequences[:,length:];result=[];valid=[]
                    for r,t in zip(chunk,generated):
                        ids=t.tolist()
                        if tok.eos_token_id in ids:ids=ids[:ids.index(tok.eos_token_id)+1]
                        valid.append(len(ids));result.append({'sample_id':r['id'],'condition':condition,'family_id':r['family_id'],
                            'domain':r['domain'],'frame':r['frame'],'length':length,'input_sha256':digest(r['token_ids']),
                            'text':tok.decode(ids,skip_special_tokens=True),'token_ids':ids,
                            'hit_limit':len(ids)==c['max_new_tokens'] and ids[-1]!=tok.eos_token_id})
                    meta=None
                    if capture:
                        bits=lambda t:t.detach().contiguous().view(torch.uint16).cpu().numpy()
                        prefix=np.stack([bits(h[:,positions,:]) for h in g.hidden_states[0]],axis=1)
                        generation=np.stack([np.stack([bits(h[:,-1,:]) for h in step],axis=1) for step in g.hidden_states],axis=1)
                        mlp_prefix=np.stack([calls[l][0] for l in blocks],axis=1)
                        mlp_gen=np.stack([np.stack([x[:,-1,:] for x in calls[l]],axis=1) for l in blocks],axis=2)
                        if generation.shape[1]!=mlp_gen.shape[1]:raise ValueError('generation capture step mismatch')
                        p=native/(stem+'.npz');tmp=p.with_suffix('.npz.tmp')
                        with tmp.open('wb') as f:np.savez(f,prefix_hidden_bits=prefix,generation_hidden_bits=generation,
                            prefix_mlp_bits=mlp_prefix,generation_mlp_bits=mlp_gen,ids=np.asarray([r['id'] for r in chunk]),
                            prefix_positions=np.asarray(positions),valid_generation_steps=np.asarray(valid))
                        tmp.replace(p);meta={'path':p.name,'sha256':file_sha(p),'bytes':p.stat().st_size,'samples':len(chunk),
                            'generation_steps':generation.shape[1],'condition':condition}
                        native_files.append(meta)
                        del prefix,generation,mlp_prefix,mlp_gen
                    save(checkpoint,{'corpus_sha256':digest(c),'runner_sha256':file_sha(__file__),'outputs':result,'native':meta,'provenance':provenance})
                    outputs.extend(result);del g,inputs,calls,generated
                    print('@@progress '+json.dumps({'condition':condition,'samples':len(outputs),'total':len(rows)*4,'shard':shard}),flush=True)
    if snapshot(model_path)!=c['model_snapshot']:raise RuntimeError('original model files changed')
    native_manifest={'schema':'native_generation_field.v1','kind':'real','files':native_files,'model_snapshot':c['model_snapshot'],
        'locations':'node local: configured native root / design_sha / shard','precision':'BF16 bits in uint16',
        'hidden_boundary':'HF hidden_states0..36; final layer after norm','mlp_boundary':'down_proj.input B18..27',
        'generation_alignment':'step0 predicts first new token from prefix; arrays after EOS excluded by valid_generation_steps',
        'method':'actual cached autoregressive forward; not teacher-forced replay','provenance':provenance}
    samples={'schema':'natural_free_outputs.v1','kind':'real','outputs':outputs,'provenance':provenance}
    summary={'protocol':SCHEMA,'status':'real','job_kind':JOB_KIND,'run_id':c['run_id'],'model_snapshot':c['model_snapshot'],
        'precision':'bf16','source_edit_sha256':c['source_edit_sha256'],'canary_recipe_sha256':c['canary_recipe_sha256'],
        'corpus_sha256':digest(c),'shard':shard,'weights_restored':True,'n_items':len(rows),
        'seconds':time.time()-started,'peak_cuda_bytes':torch.cuda.max_memory_allocated(),'env':env,
        'native_manifest_sha256':digest(native_manifest),'evidence_status':'external_ai_calibration_pending',
        'edit_metadata':edit_metadata,'provenance':provenance}
    validate_result(c,shard,summary,samples,canary)
    for name,doc in [('summary.json',summary),('samples.json',samples),('canary.json',canary),('native-manifest.json',native_manifest)]:save(Path(out)/name,doc)
    print('@@result '+json.dumps({'kind':'real','shard':shard,'samples':len(rows),'outputs':len(outputs),'external_ai_review_pending':True}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args();here=Path(__file__).resolve().parent
    c=read(here/'corpus.json');contract=read(here/'contract.json')
    native=Path(os.environ.get('AI2050_NATIVE_DIR',str(Path.home()/'.ai2050/native_fields')));native.mkdir(parents=True,exist_ok=True)
    run(c,os.environ.get('AI2050_SHARD','0/1'),os.environ.get('AI2050_RUNNER_MODEL',''),Path(args.out),native,contract['design_sha'])
