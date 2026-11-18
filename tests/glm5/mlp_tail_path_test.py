"""Native suffix intervention/rescue, distinct from sparse correlation predictors.

Targets all units in the next MLP. Full attention/residual computation remains
active; zero-edit suffix is checked against the original model. Natural next-
token loss is a preservation proxy, not a general capability certificate.
"""
import argparse
import gc
import math
import time
import numpy as np
import mlp_tail_goodfire as C


def freeze():
    path=C.OUT/'path-plan.json';assert not path.exists()
    rng=np.random.default_rng(C.SEED+100)
    units={}
    for source,target in zip(C.BLOCKS,C.BLOCKS[1:]):
        with np.load(C.OUT/f'B{source}-B{target}-predictor.npz') as z:
            counts=np.bincount(z['parents'].reshape(-1),minlength=C.WIDTH)
        units[str(source)]=np.argsort(-counts,kind='stable')[:4].tolist()
    docs=[]
    for lo,hi in [(1024,1280),(1280,1536),(1536,1792),(1792,2048)]:
        docs.extend(rng.choice(np.arange(lo,hi),24,replace=False).tolist())
    C.save_json(path,{'schema':'mlp_tail_path_plan.v1','kind':'real','run_id':C.RUN,
       'model_snapshot':C.SNAPSHOT,'source_sha256':C.file_sha(__file__),'source_units':units,'document_indices':docs,
       'factors':[0.,.5,1.5,2.],'control':'seeded Gaussian direction matched to actual BF16 residual-change L2 within 1%; invalid matches retained, not promoted',
       'selection':'Top4 source parent usage counts per pair, discovery-only predictor, deterministic tie break.',
       'zero_edit':'Native suffix logits must match original final-position logits within max abs 0.001 before intervention.',
       'scope':'96 existing long natural contexts. Local next-MLP response and next-token loss/argmax; not semantic identification or complete ability-preservation proof.',
       'seed':C.SEED+100,'max_GPU_minutes':20})


def main():
    import torch
    from transformers import AutoModelForCausalLM
    from transformers.masking_utils import create_causal_mask
    p=C.read(C.OUT/'path-plan.json');assert p['source_sha256']==C.file_sha(__file__)
    tok=np.load(C.TOKENS,mmap_mode='r');started=time.monotonic();results=[];deltas=[]
    rng=np.random.default_rng(p['seed']);ctx={'mode':'none'};handles=[]
    with C.GpuLease(),torch.inference_mode():
        model=AutoModelForCausalLM.from_pretrained(C.MODEL,local_files_only=True,dtype=torch.bfloat16,
                    device_map='cuda',attn_implementation='sdpa').eval().requires_grad_(False)
        for b in C.BLOCKS:
            def after(_module,_args,output,b=b):
                if ctx['mode']=='base':ctx['hidden'][b]=output.detach().clone()
            def pre_down(_module,args,b=b):
                if ctx['mode']=='base':ctx['A'][b]=args[0][0,-1].float().detach().clone()
                elif ctx['mode']=='suffix' and b==ctx['target']:ctx['target_A']=args[0][0,-1].float().detach().clone()
            handles += [model.model.layers[b].register_forward_hook(after),model.model.layers[b].mlp.down_proj.register_forward_pre_hook(pre_down)]
        def suffix(hidden,source):
            pos=torch.arange(hidden.shape[1],device='cuda')[None]
            mask=create_causal_mask(config=model.config,inputs_embeds=hidden,attention_mask=None,past_key_values=None,position_ids=pos)
            emb=model.model.rotary_emb(hidden,pos)
            h=hidden
            for layer in model.model.layers[source+1:]:
                h=layer(h,attention_mask=mask,position_ids=pos,position_embeddings=emb,use_cache=False)
            return model.lm_head(model.model.norm(h[:,-1:])).float()[0,0]

        total=len(p['document_indices'])*sum(len(v) for v in p['source_units'].values())*len(p['factors'])*2
        completed=0
        for di,doc in enumerate(p['document_indices']):
            ids=torch.tensor(np.asarray(tok[doc,:1024]).copy(),device='cuda')[None]
            ctx.update(mode='base',hidden={},A={})
            original=model(input_ids=ids,use_cache=False,logits_to_keep=1).logits.float()[0,0]
            base_logp=original.log_softmax(0);base_p=base_logp.exp();gold=int(tok[doc,1024]);base_nll=float(-base_logp[gold])
            for source_key,units in p['source_units'].items():
                source=int(source_key);target=source+1;clean_hidden=ctx['hidden'][source]
                base_target=ctx['A'][target].clone();ctx.update(mode='suffix',target=target)
                restored=suffix(clean_hidden,source)
                zero_error=float((restored-original).abs().max())
                if zero_error>.001:raise RuntimeError(f'zero-edit suffix mismatch B{source}: {zero_error}; stop affected tests')
                for unit in units:
                    write=model.model.layers[source].mlp.down_proj.weight[:,unit].float()
                    activation=ctx['A'][source][unit].clone()
                    noise=torch.from_numpy(rng.normal(size=C.DIM).astype(np.float32)).cuda()
                    noise=noise/noise.norm().clamp_min(1e-20)
                    for factor in p['factors']:
                        delta=(factor-1)*activation*write
                        hidden=clean_hidden.clone();before=hidden[0,-1].float().clone()
                        hidden[0,-1]=(before+delta).to(torch.bfloat16)
                        actual=hidden[0,-1].float()-before;norm=float(actual.norm())
                        # Solve the matched control after BF16 addition, not just before quantization.
                        lo,hi=0.,max(norm*2,1e-8);control_vec=None
                        best=math.inf
                        for _ in range(24):
                            mid=(lo+hi)/2;candidate=(before+mid*noise).to(torch.bfloat16);cn=float((candidate.float()-before).norm())
                            if abs(cn-norm)<best:best=abs(cn-norm);control_vec=candidate.clone()
                            if cn<norm:lo=mid
                            else:hi=mid
                        for condition in ['native_unit','matched_noise']:
                            variant=hidden if condition=='native_unit' else clean_hidden.clone()
                            if condition=='matched_noise':variant[0,-1]=control_vec
                            logits=suffix(variant,source);change=ctx['target_A']-base_target
                            measured=float((variant[0,-1].float()-before).norm())
                            logp=logits.log_softmax(0)
                            row={'doc_index':int(doc),'source':source,'target':target,'unit':int(unit),'factor':factor,'condition':condition,
                                 'original_source_A':float(activation),'actual_residual_norm':measured,'reference_norm':norm,
                                 'norm_control_valid':bool(norm>0 and abs(measured-norm)/norm<=.01),
                                 'target_A_change_norm':float(change.norm()),'largest_changed_target':int(change.abs().argmax()),
                                 'next_token_KL':float((base_p*(base_logp-logp)).sum()),'next_token_nll_change':float(-logp[gold])-base_nll,
                                 'base_top1':int(original.argmax()),'edit_top1':int(logits.argmax()),'gold_next_token':gold,
                                 'zero_edit_max_abs_error':zero_error}
                            results.append(row);deltas.append(change.cpu().numpy().astype(np.float16));completed+=1
                            if completed%64==0:C.log('native_path_test',completed,total,started,doc_index=int(doc))
                            del logits,logp,change,variant
                        del hidden,actual,control_vec
                ctx['mode']='suffix'
            # An original-model repeat after all suffix edits verifies no persistent weight/state mutation.
            ctx['mode']='none';repeat=model(input_ids=ids,use_cache=False,logits_to_keep=1).logits.float()[0,0]
            if not torch.equal(repeat,original):raise RuntimeError('post-intervention original-model restoration failed')
            del repeat,original,base_p,base_logp,ids;ctx['hidden'].clear();ctx['A'].clear();torch.cuda.empty_cache()
            C.save_json(C.OUT/'path-progress.json',{'state':'running','documents':di+1,'total_documents':len(p['document_indices']),'rows':len(results)})
        for h in handles:h.remove()
        del model;gc.collect();torch.cuda.empty_cache()
    np.save(C.OUT/'path-target-deltas.npy',np.asarray(deltas))
    C.save_json(C.OUT/'path-cases.json',{'schema':'mlp_tail_path_cases.v1','kind':'real','run_id':C.RUN,'rows':results})
    summaries=[]
    for source_key,units in p['source_units'].items():
        for unit in units:
            for factor in p['factors']:
                rows=[r for r in results if r['source']==int(source_key) and r['unit']==unit and r['factor']==factor]
                for condition in ['native_unit','matched_noise']:
                    selected=[r for r in rows if r['condition']==condition]
                    valid=[r for r in selected if r['norm_control_valid']]
                    summaries.append({'source':int(source_key),'target':int(source_key)+1,'unit':unit,'factor':factor,'condition':condition,
                       'n':len(selected),'valid_norm_n':len(valid),'top1_changes':sum(r['base_top1']!=r['edit_top1'] for r in selected),
                       'mean_KL':float(np.mean([r['next_token_KL'] for r in selected])),
                       'mean_nll_change':float(np.mean([r['next_token_nll_change'] for r in selected])),
                       'mean_target_response_norm':float(np.mean([r['target_A_change_norm'] for r in selected]))})
    C.save_json(C.OUT/'path-report.json',{'schema':'mlp_tail_path_report.v1','kind':'real','run_id':C.RUN,'model_snapshot':C.SNAPSHOT,
       'documents':len(p['document_indices']),'rows':len(results),'summaries':summaries,'seconds':time.monotonic()-started,
       'restoration':'Every document original forward repeated bitwise after edits; every source zero-edit suffix max error <=.001.',
       'capability':'Natural next-token KL/NLL/argmax preservation proxy only; translation, programming and reasoning non-inferiority not established.',
       'scope':'Tested source-unit influence on all 9728 next-layer units; no unique semantic explanation, global necessity or all-unit causal proof.'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','run']);a=p.parse_args()
    freeze() if a.stage=='freeze' else main()
