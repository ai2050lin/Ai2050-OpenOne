"""Frozen native-coordinate MLP approximations compiled by the real remaining network.

Local query replacement and independent-history multi-step substitution are separate
experiments. These are partial network approximations, never a fully extracted LLM.
"""
import gc
import re
from collections import defaultdict
from rdc_operator_common import *
from rdc_native_conditional_operator import load_bank, apply_operator, weights
from phase2724_rdc_operator_material import lexical_features
from rdc_operator_qa import repeated_ngrams

BLOCKS=(6,16,34)
TEMPERATURES=(.5,.8,1.,1.25,2.)


def causal_meta(ids, language, tok):
    meta=[]
    for p,t in enumerate(ids):
        s=tok.decode([t],clean_up_tokenization_spaces=False).strip()
        piece=0 if p==0 else 1 if not s else 2 if all(not c.isalnum() for c in s) else 3 if s.isdecimal() else 4 if re.fullmatch(r'[A-Za-z]+',s) else 5 if any('\u4e00'<=c<='\u9fff' for c in s) else 6
        f=lexical_features(tok.decode(ids[:p+1],clean_up_tokenization_spaces=False))
        cue=sum((1<<j)*int(f[k]) for j,k in enumerate(('cause','contrast','negation','reference')))
        meta.append({'language':language,'piece':piece,'cue':cue,'token_id':t,'position_bin':min(3,p//32),'position':p})
    return meta


class Compiler:
    def __init__(self,model,tok):
        self.model,self.tok=model,tok
        self.active={};self.meta=[];self.local=True;self.handles=[]
        self.banks={b:load_bank(BASE/'operators'/f'L{b}_bank') for b in BLOCKS}
        self.w={b:weights(b) for b in BLOCKS}
        for b in BLOCKS:
            def hook(module,inp,out,b=b):
                if b not in self.active:
                    return out
                name=self.active[b]
                if self.local:
                    xx=inp[0][0,-1:].float();mm=self.meta[-1:];positions=[out.shape[1]-1]
                else:
                    n=out.shape[1];start=len(self.meta)-n
                    positions=[j for j in range(n) if start+j>0]
                    if not positions:
                        return out
                    xx=inp[0][0,positions].float();mm=[self.meta[start+j] for j in positions]
                import torch
                w=self.w[b]
                if name=='native32':
                    yy=(torch.nn.functional.silu(xx@w['g'].T)*(xx@w['u'].T))@w['d'].T
                else:
                    yy=apply_operator(name,xx,mm,self.banks[b],w)
                assert torch.isfinite(yy).all(),('nonfinite',b,name)
                result=out.clone();result[0,positions]=yy.to(out.dtype)
                return result
            self.handles.append(model.model.layers[b].mlp.register_forward_hook(hook))
    def close(self):
        for h in self.handles:h.remove()


def main():
    import torch
    from phase2721_rdc_joint_scale import load
    out=BASE/'compiled'
    if (out/'result.json').exists():return
    start=time.monotonic();guard(150*1024**2)
    frozen=read(BASE/'operators/frozen.json')
    configurations={}
    for b in BLOCKS:
        configurations[f'L{b}_selected']={b:frozen['choices'][str(b)]}
        configurations[f'L{b}_global']={b:'frozen_gate_global'}
    configurations['joint_selected']={b:frozen['choices'][str(b)] for b in BLOCKS}
    configurations['joint_global']={b:'frozen_gate_global' for b in BLOCKS}
    configurations['joint_native32']={b:'native32' for b in BLOCKS}
    rr=rows();validation=[]
    for lang in ('en','zh'):
        validation.extend([r for r in rr if r['split']=='validation' and r['language']==lang][:32])
    confirmation=[r for r in rr if r['split']=='confirmation']
    immutable(out/'protocol.json',{'timestamp':stamp(),'source':snapshot(Path(__file__)),
        'validation_sources':len(validation),'confirmation_sources':len(confirmation),'anchors_per_source':2,
        'frozen_choices':frozen['choices'],'configurations':configurations,'temperature_grid':TEMPERATURES,
        'local_mode':'Only last query position of chosen MLPs substituted; all attention and other native blocks unchanged. Prefix-only reference rerun with identical shape. Local x comes from current real computation; no target H/donor input.',
        'temperature_selection':'Each configuration chooses global positive logit temperature using validation mean full-vocabulary KL; fixed before confirmation. T cannot change greedy argmax.',
        'causal_labels':'Decode only actual known prefix IDs, including generated history; no future token or answer. Saved offset-based discovery labels are separately compared.',
        'autonomous':'64 confirmation source prefixes, native/joint_global/joint_selected, max48steps greedy; every branch has its own KV and immutable chosen-token history. Approximated blocks run on every noninitial position; first position remains native by explicit design. No reference-H refresh. Unmodified modules remain native, so this is not an extracted full model.',
        'limits':'Native32 is the numeric implementation floor, not a learned predictor; approximate local response accuracy need not imply output or long-horizon fidelity.'})
    model,tok=load('qwen4',out);device=model.get_input_embeddings().weight.device
    compiler=Compiler(model,tok);torch.set_num_threads(2)
    temperature={};records={};numeric=[];label_audit=[]
    try:
      with torch.inference_mode():
        for stage,selected in [('validation',validation),('confirmation',confirmation)]:
            stage_rows=[];temperature_scores=defaultdict(list)
            for i,r in enumerate(selected):
                cp=out/stage/'commits'/f'{r["sample_id"]}.json'
                if cp.exists():
                    saved=read(cp);stage_rows.extend(saved['rows']);label_audit.extend(saved['label_audit'])
                    for name,values in saved['temperature_scores'].items():temperature_scores[name].extend(values)
                    continue
                full_meta=causal_meta(r['prompt_ids'],r['language'],tok)
                mode='main' if stage=='validation' else 'confirmation'
                with np.load(BASE/'capture'/mode/'energies'/f'{r["sample_id"]}.npz') as z:
                    saved_cues=z['prefix_cue_mask'].copy()
                local_rows=[];local_scores=defaultdict(list);audit=[]
                for ai,p in enumerate(r['anchors']):
                    ids=torch.tensor([r['prompt_ids'][:p+1]],device=device)
                    compiler.meta=full_meta[:p+1];compiler.local=True;compiler.active={}
                    ref=model.model(input_ids=ids,use_cache=False).last_hidden_state[0,-1].float()
                    ref_logits=model.lm_head(ref.to(model.dtype)).float();lp=ref_logits.log_softmax(-1);prob=lp.exp()
                    if i==0 and ai==0:
                        repeat=model.model(input_ids=ids,use_cache=False).last_hidden_state[0,-1].float()
                        assert torch.equal(ref,repeat),'No-op same-shape reproducibility'
                        numeric.append({'stage':stage,'sample_id':r['sample_id'],'same_shape_noop_exact':True})
                    audit.append({'sample_id':r['sample_id'],'anchor':ai,'saved_offset_cue':int(saved_cues[p]),'causal_prefix_cue':full_meta[p]['cue']})
                    for name,config in configurations.items():
                        compiler.active=config
                        pred=model.model(input_ids=ids,use_cache=False).last_hidden_state[0,-1].float()
                        logits=model.lm_head(pred.to(model.dtype)).float()
                        klgrid=[float((prob*(lp-(logits/t).log_softmax(-1))).sum()) for t in TEMPERATURES]
                        local_scores[name].append(klgrid)
                        t=1. if stage=='validation' else temperature[name]
                        pp=(logits/t).log_softmax(-1)
                        row={'sample_id':r['sample_id'],'source_group':r['source_group'],'language':r['language'],'anchor':ai,'position':p,'name':name,
                            'raw_KL':klgrid[2],'temperature':t,'calibrated_KL':float((prob*(lp-pp)).sum()),
                            'argmax_agreement':bool(logits.argmax()==ref_logits.argmax()),'native_next_NLL':float(-lp[r['prompt_ids'][p+1]]),
                            'approx_next_NLL':float(-pp[r['prompt_ids'][p+1]]),'postnorm_relative_MSE':float((pred-ref).square().mean()/ref.square().mean()),
                            'native_entropy':float(-(prob*lp).sum()),'all_logits_finite':bool(torch.isfinite(logits).all())}
                        assert row['all_logits_finite'];local_rows.append(row)
                        if i==0 and ai==0:
                            npz(out/stage/'full_vocab'/f'{name}.npz',native_logits=ref_logits.cpu().numpy(),approx_logits=logits.cpu().numpy(),native_postnorm=ref.cpu().numpy(),approx_postnorm=pred.cpu().numpy())
                    del ids,ref,ref_logits,lp,prob,pred,logits,pp
                save(cp,{'rows':local_rows,'temperature_scores':dict(local_scores),'label_audit':audit})
                stage_rows.extend(local_rows);label_audit.extend(audit)
                for name,values in local_scores.items():temperature_scores[name].extend(values)
                if i<2 or (i+1)%16==0:print('COMPILE',stage,i+1,len(selected),'seconds',round(time.monotonic()-start,1),flush=True)
                assert time.monotonic()-start<7200
            if stage=='validation':
                temperature={name:TEMPERATURES[int(np.argmin(np.mean(values,0)))] for name,values in temperature_scores.items()}
                immutable(out/'frozen_temperatures.json',{'timestamp':stamp(),'temperature':temperature,'validation_mean_KL':{n:np.mean(v,0).tolist() for n,v in temperature_scores.items()},'grid':TEMPERATURES})
            records[stage]=stage_rows
        autoselected=[]
        for lang in ('en','zh'):autoselected.extend([r for r in confirmation if r['language']==lang][:32])
        eos=model.generation_config.eos_token_id or tok.eos_token_id
        stop=set(eos if isinstance(eos,list) else [eos]);autorecords=[]
        for i,r in enumerate(autoselected):
            cp=out/'autonomous'/f'{r["sample_id"]}.json'
            if cp.exists():autorecords.append(read(cp));continue
            initial=r['prompt_ids'][:r['anchors'][0]+1];branches={}
            for name in ('native','joint_global','joint_selected'):
                compiler.local=False;compiler.active={} if name=='native' else configurations[name]
                known=list(initial);generated=[];cache=None;first_log=None
                for step in range(48):
                    compiler.meta=causal_meta(known,r['language'],tok)
                    input_ids=torch.tensor([known if cache is None else [known[-1]]],device=device)
                    output=model.model(input_ids=input_ids,past_key_values=cache,use_cache=True)
                    cache=output.past_key_values
                    logits=model.lm_head(output.last_hidden_state[0,-1]).float()
                    if first_log is None:first_log=logits.cpu().numpy()
                    token=int(logits.argmax());known.append(token);generated.append(token)
                    if token in stop:break
                branches[name]={'generated_ids':generated,'generated_text':tok.decode(generated,skip_special_tokens=True),
                    'stopped_by_native_EOS':any(t in stop for t in generated),'hit_limit':len(generated)==48 and not any(t in stop for t in generated),
                    'repeated_4gram_fraction':repeated_ngrams(generated),'token_count':len(generated)}
                if i<2:npz(out/'autonomous_full_vocab'/f'{r["sample_id"]}_{name}.npz',first_logits=first_log)
                del cache,output,logits,input_ids
            for name in ('joint_global','joint_selected'):
                a,b=branches['native']['generated_ids'],branches[name]['generated_ids']
                common=next((j for j,(x,y) in enumerate(zip(a,b)) if x!=y),min(len(a),len(b)))
                branches[name]['identical_prefix_tokens']=common
                branches[name]['exact_generated_sequence']=a==b
                branches[name]['first_branch_step_1based']=common+1 if a!=b else None
            result={'sample_id':r['sample_id'],'source_group':r['source_group'],'language':r['language'],'initial_ids':initial,'initial_text':tok.decode(initial),'branches':branches}
            save(cp,result);autorecords.append(result)
            if i<2 or (i+1)%8==0:print('AUTONOMOUS',i+1,len(autoselected),'seconds',round(time.monotonic()-start,1),flush=True)
            guard();assert time.monotonic()-start<7200
    finally:
        compiler.close();del compiler,model;gc.collect();torch.cuda.empty_cache()
    summaries={}
    for stage,rec in records.items():
        summaries[stage]=[]
        for name in configurations:
            part=[r for r in rec if r['name']==name]
            summaries[stage].append({'name':name,'anchors':len(part),**{key:float(np.mean([r[key] for r in part])) for key in ('raw_KL','calibrated_KL','argmax_agreement','native_next_NLL','approx_next_NLL','postnorm_relative_MSE')},
                'raw_KL_cluster':clustered([r['raw_KL'] for r in part],[r['source_group'] for r in part])})
    result={'timestamp':stamp(),'source':snapshot(Path(__file__)),'local':summaries,'temperatures':temperature,'numeric_checks':numeric,
        'causal_cue_audit':{'anchors':len(label_audit),'offset_vs_known_prefix_disagreements':sum(r['saved_offset_cue']!=r['causal_prefix_cue'] for r in label_audit)},
        'autonomous_sources':len(autorecords),'autonomous':{name:{'mean_tokens':float(np.mean([r['branches'][name]['token_count'] for r in autorecords])),
            'EOS_fraction':float(np.mean([r['branches'][name]['stopped_by_native_EOS'] for r in autorecords])),
            'mean_repeated_4gram_fraction':float(np.mean([r['branches'][name]['repeated_4gram_fraction'] for r in autorecords])),
            **({} if name=='native' else {'mean_identical_prefix_tokens':float(np.mean([r['branches'][name]['identical_prefix_tokens'] for r in autorecords])),
                'exact_sequence_fraction':float(np.mean([r['branches'][name]['exact_generated_sequence'] for r in autorecords]))})} for name in ('native','joint_global','joint_selected')},
        'limits':'Partial native-network substitutions, not independent full-model extraction. Self-history departures are not evidence of a specific hidden reset or pulse mechanism. Positive output temperature cannot repair greedy token choices. Native noninitial approximation coverage and native first-position exception are explicit.'}
    save(out/'result.json',result);ledger('compiled_operator_probability_and_autonomous',time.monotonic()-start);guard()
    print('COMPILE_COMPLETE',result,flush=True)


if __name__=='__main__':main()
