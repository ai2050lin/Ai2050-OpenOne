"""CLT natural collection extension. Confirmation requires immutable candidate seal."""
import os,sys,json,time,gc,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(Path(__file__).parent)]
OUT=ROOT/'tests/glm5/result/conditional_lookup_native_v1'
TMP=ROOT/'tests/glm5_temp/conditional_lookup_discovery'
for k in ('TEMP','TMP','TMPDIR'):os.environ[k]=str(TMP)
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
import numpy as np
import source_binding_design as D
from deploy.storage_policy import checked_dir,check_write
def save(p,x):
    assert not p.exists(),str(p)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def freeze():
    checked_dir(OUT,3*2**30,operation='CLT natural');checked_dir(TMP)
    p=D.read(OUT/'prereg.json');assert D.read(OUT/'audit.json')['technical_pass']
    save(OUT/'natural-plan.json',dict(parent_sha=D.sha(OUT/'prereg.json'),provenance=p['provenance'],code={n:D.sha(ROOT/'tests/glm5'/n) for n in ['conditional_lookup_collect.py','conditional_lookup_analyze.py','source_binding_runtime.py']},analysis='all entity hs[0..40], entity G/U/A B20..22; final A and normalized MLP input at B20..22. Sparse supervised readouts are INFORMATION diagnostics only. Native source QK/OV accounting uses exact original weights, not a causal intervention.',positions=['entity_last','final'],native_attention_blocks=list(range(20,31)),group_order=['slot0','slot1','slot2','slot3','query_entity','other'],discovery_families=64,confirm_families=128,shard_families=16,numerical='calibrate on first 16 pilot rows BEFORE discovery; tolerance=max(1e-4,2*pilot max_abs). On violation stop scientific interpretation; observed computations preserved.',selection='5-fold family OOF with coordinate selection inside each fold. pool hs01..40, entity A20..22. max32 native coordinates selected by sum squared Pearson correlations with r0/r1. alpha .01,.1,1,10,100; choose maximum OOF joint then lower dimension/key order. random32 same pool 3 fixed seeds; nuisance token length/entity position/title order/code identities. b excluded.',confirm='candidate.json plus all model NPZ hashes frozen before ANY confirmation read. No retuning. All 16 behavior cells >=.95. Decode both bits >=.9 and joint>=.85, >=.1 advantage over strongest random/nuisance baseline, 97.5% family bootstrap lower bound >0. Failure remains failure. No causal promotion from natural accounting.',budget={'additional_disk_gib':3,'per_shard_peak_gib':.5,'gpu_minutes_max':45,'keep_free_gib':20},scope='All four table types already exposed in pilot; new-family confirmation only. Candidate may be frozen for a negative replication; no intervention until mechanism gates satisfied.'))
    print('natural plan frozen',flush=True)
def collect(split):
    import torch
    from source_binding_runtime import Runtime
    from source_binding_resources import resources
    from shared.dist_protocol.mechanism_trace import GpuLease
    from shared.dist_protocol.natural_register_runner import snapshot
    plan=D.read(OUT/'natural-plan.json');pre=D.read(OUT/'prereg.json')
    print(json.dumps(dict(stage='snapshot_hash_verification',split=split)),flush=True)
    for n,h in plan['code'].items():assert D.sha(ROOT/'tests/glm5'/n)==h
    assert snapshot(D.MODELS['qwen14'])==pre['model_snapshot']
    if split=='confirm':
        seal=D.read(OUT/'candidate.json')
        for n,h in seal['files'].items():assert D.sha(OUT/n)==h
        assert seal['analysis_sha']==D.sha(ROOT/'tests/glm5/conditional_lookup_analyze.py')
    source='pilot' if split=='calibrate' else split
    assert D.sha(OUT/(source+'-materials.json'))==pre['materials'][source]['sha256']
    rows=D.read(OUT/(source+'-materials.json'))
    if split=='calibrate':rows=rows[:16]
    else:assert (OUT/'calibration.json').exists()
    checked_dir(TMP);rt=Runtime('qwen14');tok=rt.tok
    # Add only token positions, never labels, to source masks.
    for row in rows:
        enc=tok(row['prompt'],add_special_tokens=False,return_offsets_mapping=True);assert enc['input_ids']==row['tokens']
        spans=[];at=row['prompt'].index(row['content'])
        for line in row['content'].splitlines(True):
            if ' has code ' in line:spans.append((at,at+len(line.rstrip('\n'))))
            at+=len(line)
        groups=[[i for i,(a,z) in enumerate(enc['offset_mapping']) if a<hi and z>lo] for lo,hi in spans]
        groups.append([row['entity_pos']]);row['_groups']=groups
    full_lm=rt.weight('lm_head.weight','cpu');vocab=sorted(set(v for r in rows for v in r['choice_ids']));small=full_lm[vocab].clone();del full_lm;gc.collect();lookup={v:i for i,v in enumerate(vocab)}
    start=time.time();summaries=[]
    with GpuLease(),torch.inference_mode():
        for shard,lo in enumerate(range(0,len(rows),256)):
            dest=OUT/f'{split}-{shard:02}'
            if dest.with_suffix('.json').exists():summaries.append(D.read(dest.with_suffix('.json')));continue
            rr=rows[lo:lo+256];n=len(rr);arrays={};errors=[];prefix=[];calls={};weights={}
            check_write(dest,2**29,operation='CLT shard')
            def log(b,done):
                rec=dict(split=split,shard=shard,layer=b,completed_layers=done,total_layers=40,completed_inputs=lo,elapsed=time.time()-start,eta=(time.time()-start)/max(1,shard*40+done)*(40*((len(rows)+255)//256)-shard*40-done),cuda_peak=torch.cuda.max_memory_allocated(),**resources())
                with (OUT/'natural-progress.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
                print(json.dumps(rec),flush=True)
            def put(key,v,off):
                data=v.contiguous().view(torch.uint16).cpu().numpy() if v.dtype==torch.bfloat16 else v.cpu().numpy()
                if key not in arrays:arrays[key]=np.empty((n,*data.shape[1:]),dtype=data.dtype)
                arrays[key][off:off+len(v)]=data
            def positions(batch,length):return torch.tensor([length-len(r['tokens'])+r['entity_pos'] for r in batch],device='cuda')
            def callback(b,layer,x,y,batch,length,offset=0,context=None):
                h=x if b<0 else y;ix=torch.arange(len(batch),device='cuda');pos=positions(batch,length)
                put(f'h{b+1:02}',h[ix,pos],offset)
                if b<0:log(b,0)
                elif offset+len(batch)==n:log(b,b+1)
            original=rt.__class__.block
            def block(b):
                layer=original(rt,b)
                if b in (20,21,22):
                    count={'g':0,'u':0,'a':0,'z':0}
                    def capture(kind,v):
                        off=count[kind];batch=rr[off:off+len(v)];pos=positions(batch,v.shape[1]);ix=torch.arange(len(v),device='cuda')
                        put(f'{kind}{b}',v[ix,pos],off)
                        if kind in ('a','z'):put(f'{kind}f{b}',v[:,-1],off)
                        count[kind]+=len(v)
                    layer.mlp.register_forward_pre_hook(lambda m,i:capture('z',i[0]))
                    layer.mlp.gate_proj.register_forward_hook(lambda m,i,o:capture('g',o))
                    layer.mlp.up_proj.register_forward_hook(lambda m,i,o:capture('u',o))
                    layer.mlp.down_proj.register_forward_pre_hook(lambda m,i:capture('a',i[0]))
                if 20<=b<=30:
                    state={'off':0}
                    def prepare(m,args,kw):
                        x=kw['hidden_states'];off=state['off'];batch=rr[off:off+len(x)];N,T,_=x.shape
                        q=m.q_norm(m.q_proj(x).view(N,T,rt.heads,rt.hd)).transpose(1,2)
                        k=m.k_norm(m.k_proj(x).view(N,T,rt.kv,rt.hd)).transpose(1,2)
                        v=m.v_proj(x).view(N,T,rt.kv,rt.hd).transpose(1,2)
                        q,k=rt.mod.apply_rotary_pos_emb(q,k,*kw['position_embeddings'])
                        k=rt.mod.repeat_kv(k,m.num_key_value_groups);v=rt.mod.repeat_kv(v,m.num_key_value_groups)
                        pp=positions(batch,T)
                        sites=[pp,torch.full_like(pp,T-1)] if b<=22 else [torch.full_like(pp,T-1)]
                        masses=[];writes=[];rebuild=[]
                        word=small[[lookup[t] for r in batch for t in r['choice_ids']]].reshape(N,4,rt.dim).to('cuda').float()
                        coef=(word @ m.o_proj.weight.float()).reshape(N,4,rt.heads,rt.hd)
                        group=torch.zeros((N,6,T),device='cuda',dtype=torch.bool)
                        for i,r in enumerate(batch):
                            pad=T-len(r['tokens'])
                            for g,idx in enumerate(r['_groups']):group[i,g,torch.tensor(idx,device='cuda')+pad]=True
                            group[i,5,pad:]=~group[i,:5,pad:].any(0)
                        for p in sites:
                            ix=torch.arange(N,device='cuda');qq=q[ix,:,p,:]
                            score=torch.einsum('nhd,nhtd->nht',qq.float(),k.float())*m.scaling
                            allowed=kw['attention_mask'][ix,0,p,:];score=score.masked_fill(~allowed[:,None,:],float('-inf'))
                            alpha=score.softmax(-1);out=torch.einsum('nht,nhtd->nhd',alpha,v.float());rebuild.append(out)
                            masses.append(torch.einsum('nht,ngt->nhg',alpha,group.float()))
                            bygroup=torch.einsum('nht,ngt,nhtd->nhgd',alpha,group.float(),v.float())
                            writes.append(torch.einsum('nhgd,nchd->nhgc',bygroup,coef))
                        state.update(rebuild=rebuild,sites=sites,masses=masses,writes=writes,N=N)
                    def audit(m,args):
                        v=args[0].reshape(len(args[0]),args[0].shape[1],rt.heads,rt.hd);N=state['N'];ix=torch.arange(N,device='cuda')
                        errors.extend(float((v[ix,p].float()-rec).abs().max()) for p,rec in zip(state['sites'],state['rebuild']))
                        put(f'mass{b}',torch.stack(state['masses'],1),state['off']);put(f'write{b}',torch.stack(state['writes'],1),state['off'])
                        state['off']+=N
                    layer.self_attn.register_forward_pre_hook(prepare,with_kwargs=True)
                    layer.self_attn.o_proj.register_forward_pre_hook(audit)
                return layer
            rt.block=block
            _,logits=rt.forward(rr,callback);scores=rt.score(logits,rr)
            # family-local positions and b variants stay in identical batch shapes.
            idx={(r['family'],r['r0'],r['r1'],r['b'],r['late']):i for i,r in enumerate(rr)}
            for r in rr:
                if r['late'] and r['b']==0:
                    i=idx[r['family'],r['r0'],r['r1'],0,1];j=idx[r['family'],r['r0'],r['r1'],1,1]
                    prefix.append(all(np.array_equal(arrays[f'h{k:02}'][i],arrays[f'h{k:02}'][j]) for k in range(41)))
            np.savez_compressed(dest.with_suffix('.npz'),**arrays)
            summary=dict(provenance=pre['provenance'],split=split,shard=shard,ids=[r['id'] for r in rr],scores=scores,source_audit_max=max(errors),prefix_pairs=len(prefix),prefix_equal=sum(prefix),native_sha=D.sha(dest.with_suffix('.npz')),plan_sha=D.sha(OUT/'natural-plan.json'),elapsed=time.time()-start,cuda_peak=torch.cuda.max_memory_allocated(),**resources())
            save(dest.with_suffix('.json'),summary);summaries.append(summary);del arrays,logits;gc.collect();torch.cuda.empty_cache()
    if split=='calibrate':save(OUT/'calibration.json',dict(max_abs=summaries[0]['source_audit_max'],tolerance=max(1e-4,2*summaries[0]['source_audit_max']),basis='pilot first family; exact BF16 q/k/v, FP32 attention accounting vs native SDPA',provenance=pre['provenance']))
    else:
        tolerance=D.read(OUT/'calibration.json')['tolerance'];scores=[s for part in summaries for s in part['scores']];cells=[]
        for r0,r1,b,l in __import__('itertools').product((0,1),repeat=4):
            ss=[s for r,s in zip(rows,scores) if (r['r0'],r['r1'],r['b'],r['late'])==(r0,r1,b,l)]
            cells.append(dict(cell=[r0,r1,b,l],n=len(ss),strict=sum(s['gold_margin']>0 for s in ss)/len(ss),top1=sum(s['top1_correct'] for s in ss)/len(ss)))
        save(OUT/(split+'-collection.json'),dict(provenance=pre['provenance'],cells=cells,behavior_pass=all(min(c['strict'],c['top1'])>=.95 for c in cells),technical_pass=all(p['prefix_pairs']==p['prefix_equal'] and p['source_audit_max']<=tolerance for p in summaries),shards={f'{split}-{i:02}.npz':p['native_sha'] for i,p in enumerate(summaries)},source_audit_max=max(p['source_audit_max'] for p in summaries),tolerance=tolerance,elapsed=time.time()-start))
    print('COMPLETE '+split,flush=True)
if __name__=='__main__':
    cmd=sys.argv[1]
    freeze() if cmd=='freeze' else collect(cmd)
