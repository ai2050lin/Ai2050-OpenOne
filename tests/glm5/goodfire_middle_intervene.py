"""Prospective B18/B23/B28 dose tests; adapted full-query prefix replay.

Port of qwen14_mlp_intervene.py with dynamic model depth, start boundary,
model-specific output weights and explicit discovery-only selection.
"""
import argparse
import gc
import time
import numpy as np
import goodfire_middle as M
C=M.C
save_result=M.write

def freeze():
    from transformers import AutoTokenizer
    from tools.power_analysis import check_phase
    assert not (C.OUT/'intervention-plan.json').exists()
    tokenizer=AutoTokenizer.from_pretrained(C.MODEL,local_files_only=True)
    source=C.ROOT/'tests/glm5/result/state_formation_v1/capability-materials.json'
    cap=C.read(source)['rows'];rows=[]
    for item in cap:
        text=item['question']+'\n'+'\n'.join(f'{letter}. {value}' for letter,value in zip('ABCDE',item['choices']))+'\n请选择一个正确选项字母。'
        prompt=tokenizer.apply_chat_template([{'role':'user','content':text}],tokenize=False,add_generation_prompt=True,enable_thinking=False)+'Answer:'
        rows.append({**item,'prompt':prompt,'tokens':tokenizer.encode(prompt,add_special_tokens=False)})
    natural=check_phase(.08,256,alpha=.05/6);capability=check_phase(.04,len(rows))
    assert natural['pass'] and capability['pass'] and len(rows)==512
    save_result(C.OUT/'capability-inputs.json',dict(rows=rows,source_sha256=C.file_sha(source),
        scope='Previously exposed keyed choices; separate fresh baseline for each model.'))
    save_result(C.OUT/'intervention-plan.json',dict(schema='goodfire_middle_interventions.v1',
        selection='At B18/B23/B28, highest discovery std(A)*norm(down), and distinct highest double-negative abs(A)*norm(down).',
        natural_documents=[768,1024],natural_readout=1006,natural_target=1007,doses=[0.,.5,2.],
        capability_inputs_sha256=C.file_sha(C.OUT/'capability-inputs.json'),
        control='Seeded random residual direction, actual BF16 norm matched within 2%; invalid rows retained.',
        restore='Two separate clean branches per intervention block; full-query prefix replay.',
        power=dict(next_token_large_effect=natural,capability_loss=capability),
        power_scope='At least eight-point effects per unit, six-unit correction within each model. Dose comparisons descriptive. No cross-model significance claim.',
        capability_preservation='One-sided 95% Clopper-Pearson upper bound on lost-originally-correct fraction <=.02 per edit; not universal ability.',
        norm_relative_error_max=.02,kv_replay_kl_max=.001,restore_bitwise_required=True,
        source_sha256=C.file_sha(__file__),formal_promotion=False,
        scope='Finite final-token down-column write scaling; no semantic or free-generation claim.'))
    selection=[]
    for b in C.BLOCKS:
        a=np.load(C.OUT/f'B{b}-A.npy',mmap_mode='r').reshape(-1,C.WIDTH)[:4096]
        g=np.load(C.OUT/f'B{b}-G.npy',mmap_mode='r').reshape(-1,C.WIDTH)[:4096]
        v=np.load(C.OUT/f'B{b}-V.npy',mmap_mode='r').reshape(-1,C.WIDTH)[:4096]
        down=C.unbits(np.load(C.OUT/f'B{b}-W-down.npy',mmap_mode='r'))
        norm=np.linalg.norm(down.astype(np.float64),axis=0);del down
        scale=np.empty(C.WIDTH);double_negative=np.empty(C.WIDTH)
        for lo in range(0,C.WIDTH,256):
            hi=min(C.WIDTH,lo+256);x=C.unbits(a[:,lo:hi])
            mask=(C.unbits(g[:,lo:hi])<0)&(C.unbits(v[:,lo:hi])<0)
            scale[lo:hi]=x.std(0,dtype=np.float64)*norm[lo:hi]
            double_negative[lo:hi]=(np.abs(x)*mask).mean(0,dtype=np.float64)*norm[lo:hi]
        first=int(np.argmax(scale));rank=np.argsort(-double_negative,kind='stable')
        second=next(int(u) for u in rank if int(u)!=first)
        selection.extend([dict(block=b,unit=first,criterion='discovery_std_A_times_down_norm',value=float(scale[first])),
            dict(block=b,unit=second,criterion='discovery_double_negative_abs_A_times_down_norm',value=float(double_negative[second]))])
    save_result(C.OUT/'intervention-selection.json',dict(units=selection,plan_sha256=C.file_sha(C.OUT/'intervention-plan.json')))
    save_result(C.OUT/'intervention-execution-binding.json',dict(source_sha256=C.file_sha(__file__),
        model=SPECS_NAME(),campaign_plan_sha256=C.file_sha(C.OUT/'plan.json')))


def SPECS_NAME():
    return M.SPEC['name']

def suffix(layer,states,keys,values,config,block,position_ids,pe,key_mask):
    """Batch-major branch ordering, using clean prefix keys and branch self key."""
    import torch
    from transformers.cache_utils import DynamicCache
    batch,branches,dim=states.shape
    cache=DynamicCache(config=config);slot=cache.layers[block]
    k=keys[:,:,:-1].repeat_interleave(branches,0);v=values[:,:,:-1].repeat_interleave(branches,0)
    slot.lazy_initialization(k,v);slot.keys=k;slot.values=v
    out=layer(states.reshape(batch*branches,1,dim),past_key_values=cache,use_cache=True,
        position_ids=position_ids[:,-1:].repeat_interleave(branches,0),
        position_embeddings=tuple(x[:,-1:].repeat_interleave(branches,0) for x in pe),
        attention_mask=key_mask.repeat_interleave(branches,0))
    return out[:,0].reshape(batch,branches,dim)


def norm_control(base,desired,noise):
    import torch
    target=desired.float().norm(dim=-1,keepdim=True)
    direction=noise/noise.norm(dim=-1,keepdim=True).clamp_min(1e-20)
    scale=target.clone();best=base.clone();error=torch.full_like(target,float('inf'))
    for _ in range(16):
        edited=(base.float()+direction*scale).to(base.dtype);actual=(edited.float()-base.float()).norm(dim=-1,keepdim=True)
        err=(actual-target).abs()/target.clamp_min(1e-20);choose=err<error
        best=torch.where(choose,edited,best);error=torch.minimum(error,err)
        scale*=target/actual.clamp_min(1e-20)
    valid=(error[:,0]<=.02)&(target[:,0]>0)
    return best,error[:,0],valid


def exact_suffix(layer,states,clean_input,keys,values,block,position_ids,pe,mask):
    """Use original full query shape and cached prefix KV, avoiding kernel-shape drift."""
    import torch
    class FrozenPrefix:
        def update(self,k,v,index):
            assert index==block
            return torch.cat([keys[:,:,:-1],k[:,:,-1:]],dim=-2),torch.cat([values[:,:,:-1],v[:,:,-1:]],dim=-2)
    out=torch.empty_like(states)
    for i in range(states.shape[1]):
        x=clean_input.clone();x[:,-1]=states[:,i]
        y=layer(x,past_key_values=FrozenPrefix(),use_cache=True,position_ids=position_ids,
                position_embeddings=pe,attention_mask=mask)
        out[:,i]=y[:,-1]
    return out


def run(stage):
    import torch
    from transformers import AutoConfig
    from transformers.cache_utils import DynamicCache
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
    from transformers import AutoTokenizer
    p=C.read(C.OUT/'intervention-plan.json');selection=C.read(C.OUT/'intervention-selection.json')['units']
    tok=AutoTokenizer.from_pretrained(C.MODEL,local_files_only=True)
    calibration=stage.startswith('calibrate')
    if not calibration:
        admission=C.read(C.OUT/'exact-calibration-admission.json')
        assert admission['passed'] and stage.endswith('_exact'), 'Stop uncalibrated or query-shape-changing replay'
        assert C.file_sha(__file__)==C.read(C.OUT/'intervention-execution-binding.json')['source_sha256']
    natural=stage.startswith('natural')
    if natural or calibration:
        lo,hi=(768,1024) if natural else (0,4)
        tokens=np.load(C.OUT/'tokens.npy')[lo:hi]
        rows=[{'id':f'document-{i+lo}','tokens':r[:1007].tolist(),'target':int(r[1007])} for i,r in enumerate(tokens)]
    else:rows=C.read(C.OUT/'capability-inputs.json')['rows']
    n=len(rows);start=time.monotonic();all_metrics=[];replay=[];controls=[]
    config=AutoConfig.from_pretrained(C.MODEL,local_files_only=True);config._attn_implementation='sdpa'
    branches=[{'id':'baseline'}]
    for b in C.BLOCKS:
        branches.append({'id':f'B{b}:restore','block':b,'condition':'restore'})
        branches.append({'id':f'B{b}:restore_repeat','block':b,'condition':'restore_repeat'})
        for item in [x for x in selection if x['block']==b]:
            for dose in p['doses']:
                for cond in ['native','norm_random']:
                    branches.append({'id':f'B{b}:U{item["unit"]}:{dose}:{cond}','block':b,'unit':item['unit'],'dose':dose,'condition':cond})
    save_result(C.OUT/f'{stage}-branches.json',{'branches':branches,'samples':n,'plan_sha256':C.file_sha(C.OUT/'intervention-plan.json')})
    final=np.lib.format.open_memmap(C.OUT/f'{stage}-final-states.npy',mode='w+',dtype=np.uint16,shape=(n,len(branches),C.DIM))
    reference=np.lib.format.open_memmap(C.OUT/f'{stage}-full-forward-final.npy',mode='w+',dtype=np.uint16,shape=(n,C.DIM))
    assert torch.cuda.is_available();torch.set_num_threads(4)
    if stage.endswith('_precise'):
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    with C.GpuLease(),torch.inference_mode():
        rotary=Qwen3RotaryEmbedding(config,device='cpu').cuda()
        for lo in range(0,n,32):
            batchrows=rows[lo:lo+32];size=len(batchrows);length=max(len(x['tokens']) for x in batchrows)
            ids=torch.full((size,length),tok.pad_token_id or 151643,device='cuda',dtype=torch.long)
            visible=torch.zeros((size,length),device='cuda',dtype=torch.bool)
            for j,r in enumerate(batchrows):ids[j,-len(r['tokens']):]=torch.tensor(r['tokens'],device='cuda');visible[j,-len(r['tokens']):]=True
            positions=(visible.long().cumsum(1)-1).clamp_min(0)
            emb=C.weight('model.embed_tokens.weight').cuda().to(torch.bfloat16)
            hidden=torch.nn.functional.embedding(ids,emb);del emb
            pe=rotary(hidden,positions)
            mask=torch.full((length,length),float('-inf'),device='cuda',dtype=torch.bfloat16).triu(1)[None,None].expand(size,1,length,length).clone()
            mask.masked_fill_(~visible[:,None,None,:],float('-inf'))
            key_mask=mask[:,:,-1:,:]
            states=None;active=[];capture={}
            for b in range(config.num_hidden_layers):
                layer=C.load_block(b,config);hooks=[]
                if b in C.BLOCKS:
                    hooks=[layer.mlp.down_proj.register_forward_pre_hook(lambda m,a:capture.update(A=a[0])),
                        layer.mlp.register_forward_hook(lambda m,a,y:capture.update(output=y)),
                        layer.post_attention_layernorm.register_forward_pre_hook(lambda m,a:capture.update(residual=a[0]))]
                new=[x for x in branches if x.get('block')==b]
                k=len(active)+len(new)
                if b==min(C.BLOCKS):k+=1
                next_states=torch.empty((size,k,C.DIM),dtype=torch.bfloat16,device='cuda') if k else None
                for j in range(0,size,2):
                    cache=DynamicCache(config=config)
                    original=layer(hidden[j:j+2],attention_mask=mask[j:j+2],position_ids=positions[j:j+2],
                        position_embeddings=tuple(x[j:j+2] for x in pe),past_key_values=cache,use_cache=True)
                    clean_last=original[:,-1].clone()
                    # Copy original capture before cached-query forwards trigger the same hooks.
                    saved={key:value for key,value in capture.items()};capture.clear()
                    if states is not None:
                        if stage.endswith('_exact'):
                            advanced=exact_suffix(layer,states[j:j+2],hidden[j:j+2],cache.layers[b].keys,cache.layers[b].values,b,
                                positions[j:j+2],tuple(x[j:j+2] for x in pe),mask[j:j+2])
                        else:
                            advanced=suffix(layer,states[j:j+2],cache.layers[b].keys,cache.layers[b].values,config,b,
                                positions[j:j+2],tuple(x[j:j+2] for x in pe),key_mask[j:j+2])
                        next_states[j:j+2,:len(active)]=advanced
                    elif b==min(C.BLOCKS):next_states[j:j+2,0]=clean_last
                    offset=len(active) if states is not None else (1 if b==min(C.BLOCKS) else 0)
                    cached_native={}
                    for col,condition in enumerate(new):
                        if condition['condition'] in ['restore','restore_repeat']:edit=clean_last.clone()
                        else:
                            u=condition['unit'];dose=condition['dose'];key=(u,dose)
                            if key not in cached_native:
                                write=(dose-1)*saved['A'][:,-1,u:u+1]*layer.mlp.down_proj.weight[:,u][None]
                                edit=saved['residual'][:,-1]+(saved['output'][:,-1]+write)
                                cached_native[key]=edit
                            edit=cached_native[key]
                            if condition['condition']=='norm_random':
                                noise=torch.from_numpy(np.stack([np.random.default_rng(M.SEED+(lo+j+t)*100000+b*20000+u).standard_normal(C.DIM) for t in range(len(clean_last))]).astype(np.float32)).cuda()
                                edit,err,valid=norm_control(clean_last,edit.float()-clean_last.float(),noise)
                                controls.extend({'sample':lo+j+t,'branch':condition['id'],'relative_norm_error':float(err[t]),'valid':bool(valid[t])} for t in range(len(clean_last)))
                        next_states[j:j+2,offset+col]=edit
                    hidden[j:j+2]=original
                    capture.clear();del original,cache,saved
                for hook in hooks:hook.remove()
                if b==min(C.BLOCKS):active=[branches[0]]+new
                elif new:active+=new
                states=next_states
                del layer;gc.collect();torch.cuda.empty_cache()
                if b%8==7:C.log(stage+'_kv',lo*config.num_hidden_layers+size*(b+1),n*config.num_hidden_layers,start,block=b,samples=lo+size)
            assert [x['id'] for x in active]==[x['id'] for x in branches]
            norm=C.weight('model.norm.weight').cuda().to(torch.bfloat16)
            def normalize(x):return (x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+config.rms_norm_eps)).to(torch.bfloat16)*norm
            states=normalize(states);ref=normalize(hidden[:,-1]);final[lo:lo+size]=states.contiguous().view(torch.uint16).cpu().numpy();reference[lo:lo+size]=ref.contiguous().view(torch.uint16).cpu().numpy()
            wu=C.weight('lm_head.weight').cuda().to(torch.bfloat16)
            choice_ids=[tok.encode(' '+s,add_special_tokens=False)[0] for s in 'ABCDE']
            for j,r in enumerate(batchrows):
                logits=torch.nn.functional.linear(states[j],wu).float()
                ref_logits=torch.nn.functional.linear(ref[j:j+1].expand(len(branches),-1).contiguous(),wu).float()
                logp=logits.log_softmax(-1);probs=logp.exp();ref_lp=ref_logits.log_softmax(-1)[0]
                restore_indices={b:next(i for i,c in enumerate(branches) if c['id']==f'B{b}:restore') for b in C.BLOCKS}
                restore_ok=all(torch.equal(states[j,i],states[j,restore_indices[c['block']]]) for i,c in enumerate(branches) if c.get('condition')=='restore_repeat')
                replay_kl=max(float((logp[i].exp()*(logp[i]-ref_lp)).sum()) for i in [0,*restore_indices.values()])
                replay.append({'sample':lo+j,'full_vs_kv_kl':replay_kl,'restore_bitwise':restore_ok})
                # Keep failed replay/control rows; prevent scientific admission later.
                for i,c in enumerate(branches):
                    base_index=restore_indices[c['block']] if c.get('block') is not None else 0
                    base=logp[base_index];kl=float((probs[i]*(logp[i]-base)).sum())
                    row={'sample':lo+j,'id':r['id'],'branch':c['id'],'baseline_branch':branches[base_index]['id'],'kl':kl,'argmax':int(logits[i].argmax()),'baseline_argmax':int(logits[base_index].argmax())}
                    if natural or calibration:row.update(target=r['target'],nll=float(-logp[i,r['target']]),baseline_nll=float(-base[r['target']]))
                    else:
                        scores=logits[i,choice_ids[:len(r['choices'])]].cpu().tolist()
                        base_scores=logits[base_index,choice_ids[:len(r['choices'])]].cpu().tolist()
                        row.update(gold='ABCDE'.index(r['answer']),scores=scores,baseline_scores=base_scores,correct=int(np.argmax(scores))=='ABCDE'.index(r['answer']))
                    all_metrics.append(row)
            final.flush();reference.flush()
            save_result(C.OUT/f'{stage}-progress.json',{'completed':lo+size,'total':n,'replay':replay,'controls':controls,'metrics':all_metrics})
            del hidden,states,ref,wu,norm,mask,pe,ids,visible;gc.collect();torch.cuda.empty_cache()
    save_result(C.OUT/f'{stage}-effects.json',{'schema':'qwen14_effects.v1','rows':all_metrics,'replay':replay,'controls':controls,
        'samples':n,'branches':branches,'source_sha256':C.file_sha(__file__),'seconds':time.monotonic()-start})



def main():
    parser=argparse.ArgumentParser();parser.add_argument('model',choices=M.SPECS)
    parser.add_argument('stage',choices=['freeze','calibrate','natural','capability'])
    args=parser.parse_args();M.configure(args.model);M.require_admission();C.BLOCKS=[18,23,28]
    if args.stage=='freeze':freeze()
    elif args.stage=='calibrate':
        run('calibrate_exact')
        rows=C.read(M.OUT/'calibrate_exact-effects.json')['replay']
        passed=all(x['restore_bitwise'] and x['full_vs_kv_kl']<=.001 for x in rows)
        save_result(M.OUT/'exact-calibration-admission.json',dict(passed=passed,rows=rows,threshold=.001))
        assert passed
    else:run(args.stage+'_exact')

if __name__=='__main__':main()
