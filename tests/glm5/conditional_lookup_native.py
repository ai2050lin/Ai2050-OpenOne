"""CLT v1: immutable eight-cell design and natural BF16 pilot; no donor patches."""
import os,sys,json,time,itertools,hashlib,gc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(Path(__file__).parent)]
OUT=ROOT/'tests/glm5/result/conditional_lookup_native_v1'
TMP=ROOT/'tests/glm5_temp/conditional_lookup_native_v1'
for key in ('TEMP','TMP','TMPDIR'): os.environ[key]=str(TMP)
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import numpy as np
import source_binding_design as D
from deploy.storage_policy import checked_dir,check_write
from tools.power_analysis import check_phase
SEED=202611170501
def save(name,obj):
    p=OUT/name
    assert not p.exists(),str(p)
    check_write(p,2**20,operation='CLT-v1')
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def freeze():
    from transformers import AutoTokenizer
    from shared.dist_protocol.natural_register_runner import snapshot
    checked_dir(OUT,2*2**30,operation='CLT-v1');checked_dir(TMP)
    assert not (OUT/'prereg.json').exists()
    tok=AutoTokenizer.from_pretrained(D.MODELS['qwen14'],local_files_only=True)
    ident=snapshot(D.MODELS['qwen14'])
    assert ident==D.read(D.OUT/'qwen14-identity.json')['model_snapshot']
    rng=np.random.default_rng(SEED);used=set();materials={}
    for split,n in [('pilot',16),('discovery',64),('confirm',128)]:
        rows=[]
        for f in range(n):
            names=[]
            while len(names)<2:
                name=''.join(rng.choice(list('bdfgklmnprstvz'))+rng.choice(list('aeiou')) for _ in range(3)).capitalize()
                if name not in used:used.add(name);names.append(name)
            values=list(map(str,rng.choice(D.WORDS,4,replace=False)))
            titles=['Orion','Vega'] if f%2==0 else ['Vega','Orion']
            for r0,r1,b,late in itertools.product((0,1),repeat=4):
                body='Two independent archives assign codes to the same entities. Return only the requested code word.\n'
                for block,r in enumerate((r0,r1)):
                    body+=f'Archive {titles[block]}:\n'
                    for line in (0,1):body+=f'{names[0] if line==r else names[1]} has code {values[2*block+line]}.\n'
                entity=f'Entity to look up: {names[0]}\n';source=f'Archive to use: {titles[b]}\n'
                content=body+(entity+source if late else source+entity)+'Return only the code word.'
                prompt=tok.apply_chat_template([{'role':'user','content':content}],tokenize=False,add_generation_prompt=True,enable_thinking=False)+'Answer:\n'
                enc=tok(prompt,add_special_tokens=False,return_offsets_mapping=True)
                start=prompt.index(content)+content.index('Entity to look up: ')+len('Entity to look up: ')
                pos=[i for i,(a,z) in enumerate(enc['offset_mapping']) if a<start+len(names[0]) and z>start]
                choice=[tok.encode(v,add_special_tokens=False) for v in values]
                assert all(len(c)==1 for c in choice),choice
                target=2*b+(r0,r1)[b]
                rows.append(dict(id=f'{split}-{f:03}-{r0}{r1}-{b}-{late}',family=f,split=split,r0=r0,r1=r1,b=b,late=late,names=names,values=values,titles=titles,content=content,prompt=prompt,tokens=enc['input_ids'],entity_pos=max(pos),choice_ids=[c[0] for c in choice],gold_id=choice[target][0],target_record=target))
        # Pairwise causal-prefix identity is checked BEFORE any GPU run.
        for row in rows:
            if row['late'] and row['b']==0:
                pair=next(x for x in rows if x['family']==row['family'] and x['r0']==row['r0'] and x['r1']==row['r1'] and x['late']==1 and x['b']==1)
                assert row['tokens'][:row['entity_pos']+1]==pair['tokens'][:pair['entity_pos']+1]
        save(split+'-materials.json',rows);materials[split]=dict(families=n,inputs=len(rows),sha256=D.sha(OUT/(split+'-materials.json')))
    power=check_phase(.15,128,p_d=.25,alpha=.025);assert power['pass']
    save('prereg.json',dict(run='CLT-20261117-v1',seed=SEED,model_snapshot=ident,model_path=str(D.MODELS['qwen14']),materials=materials,code={p.name:D.sha(p) for p in [Path(__file__),Path(__file__).with_name('source_binding_runtime.py'),Path(__file__).with_name('source_binding_design.py'),ROOT/'tests/glm5/conditional_lookup_prereg.md']},power=power,pilot_gate='strict margin>0 and full vocabulary top1 >=15/16 in EVERY r0,r1,b,late cell; no exclusion or repair after exposure',confirm_gate='strict margin and full vocabulary top1 >=.95 EACH cell',native_scope='pilot: all 41 residual boundaries at entity and final; native G/U/A at blocks20,21,22; raw BF16 bits',budget=dict(peak_gib=2,gpu_minutes_pilot=30,retained_gib_max=1.5),provenance=dict(git_sha=D.git('rev-parse','HEAD'),git_dirty=True,model_path=str(D.MODELS['qwen14']),model_revision=ident,seed=SEED)))
    print(json.dumps({'frozen':str(OUT),'materials':materials,'power':power}),flush=True)
def pilot():
    from source_binding_runtime import Runtime
    from shared.dist_protocol.mechanism_trace import GpuLease
    from shared.dist_protocol.natural_register_runner import snapshot
    from source_binding_resources import resources
    import torch
    pre=D.read(OUT/'prereg.json')
    for name,digest in pre['code'].items():assert D.sha(Path(__file__).parent/name)==digest
    assert snapshot(D.MODELS['qwen14'])==pre['model_snapshot']
    assert D.sha(OUT/'pilot-materials.json')==pre['materials']['pilot']['sha256']
    assert not (OUT/'pilot-result.json').exists()
    checked_dir(OUT,2*2**30,operation='CLT pilot');checked_dir(TMP)
    rows=D.read(OUT/'pilot-materials.json');start=time.time();arrays={};last=[-2];runtime=Runtime('qwen14')
    def progress(b):
        rec=dict(layer=b,completed=b+2,total=41,elapsed_seconds=time.time()-start,eta_seconds=(time.time()-start)/max(1,b+2)*(39-b),cuda_allocated=torch.cuda.memory_allocated(),cuda_peak=torch.cuda.max_memory_allocated(),**resources())
        with (OUT/'progress.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(rec)+'\n')
        print(json.dumps(rec),flush=True)
    def callback(b,layer,x,y,rr,length,offset=0,context=None):
        if last[0]!=b:progress(b);last[0]=b;check_write(OUT,2**28,operation='CLT layer checkpoint')
        h=x if b<0 else y
        ix=torch.arange(len(rr),device='cuda');ps=torch.tensor([length-len(r['tokens'])+r['entity_pos'] for r in rr],device='cuda')
        picked=torch.stack([h[ix,ps],h[:,-1]],dim=1)
        key=f'hs{b+1:02}'
        if key not in arrays:arrays[key]=np.empty((len(rows),2,runtime.dim),dtype=np.uint16)
        arrays[key][offset:offset+len(rr)]=picked.contiguous().view(torch.uint16).cpu().numpy()
    # Capture G/U/A from the ACTUAL model invocation, not a replay approximation.
    original=runtime.block
    def block(b):
        layer=original(b)
        if b in (20,21,22):
            counters={'g':0,'u':0,'a':0}
            def collect(kind,tensor):
                off=counters[kind];rr=rows[off:off+tensor.shape[0]];length=tensor.shape[1]
                ix=torch.arange(len(rr),device='cuda');ps=torch.tensor([length-len(r['tokens'])+r['entity_pos'] for r in rr],device='cuda')
                selected=torch.stack([tensor[ix,ps],tensor[:,-1]],dim=1)
                key=f'{kind}{b}'
                if key not in arrays:arrays[key]=np.empty((len(rows),2,runtime.width),dtype=np.uint16)
                arrays[key][off:off+len(rr)]=selected.contiguous().view(torch.uint16).cpu().numpy();counters[kind]+=len(rr)
            layer.mlp.gate_proj.register_forward_hook(lambda m,i,o:collect('g',o))
            layer.mlp.up_proj.register_forward_hook(lambda m,i,o:collect('u',o))
            layer.mlp.down_proj.register_forward_pre_hook(lambda m,i:collect('a',i[0]))
        return layer
    runtime.block=block
    with GpuLease(),torch.inference_mode():
        progress(-1)
        _,logits=runtime.forward(rows,callback)
        scores=runtime.score(logits,rows)
    cells=[]
    for a,c,b,l in itertools.product((0,1),repeat=4):
        ids=[i for i,r in enumerate(rows) if (r['r0'],r['r1'],r['b'],r['late'])==(a,c,b,l)]
        strict=sum(scores[i]['gold_margin']>0 for i in ids);top=sum(scores[i]['top1_correct'] for i in ids)
        cells.append(dict(r0=a,r1=c,b=b,late=l,n=len(ids),strict=strict,top1=top,pass_gate=min(strict,top)>=15))
    prefix=[]
    for r in rows:
        if r['late'] and r['b']==0:
            j=next(i for i,x in enumerate(rows) if (x['family'],x['r0'],x['r1'],x['b'],x['late'])==(r['family'],r['r0'],r['r1'],1,1));i=rows.index(r)
            prefix.append(all(np.array_equal(arrays[f'hs{k:02}'][i,0],arrays[f'hs{k:02}'][j,0]) for k in range(41)))
    np.savez_compressed(OUT/'pilot-native.npz',**arrays)
    result=dict(provenance=pre['provenance'],prereg_sha256=D.sha(OUT/'prereg.json'),elapsed_seconds=time.time()-start,cells=cells,scores=scores,behavior_pass=all(c['pass_gate'] for c in cells),prefix_pairs=len(prefix),prefix_bitwise_equal=sum(prefix),technical_pass=all(prefix),native_sha256=D.sha(OUT/'pilot-native.npz'),status='pilot only; no candidate selection, confirm exposure or causal claim')
    save('pilot-result.json',result)
    from tools.provenance import write_result
    os.environ['GIT_CONFIG_COUNT']='1';os.environ['GIT_CONFIG_KEY_0']='safe.directory';os.environ['GIT_CONFIG_VALUE_0']=str(ROOT)
    receipt=write_result('conditional_lookup_native_pilot',result,model_path=D.MODELS['qwen14'],seed=SEED,tool='conditional_lookup_native.py')
    print(json.dumps({'behavior_pass':result['behavior_pass'],'technical_pass':result['technical_pass'],'cells':cells,'receipt':str(receipt)}),flush=True)
if __name__=='__main__':
    {'freeze':freeze,'pilot':pilot}[sys.argv[1]]()
