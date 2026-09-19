"""Same-material sequential native three-model capability/coordinate replication."""
import argparse
import gc
from collections import defaultdict
from rdc_binding_common import *

def material():
    programs=gzread(BASE/'program_material.json.gz');natural=gzread(BASE/'natural_confirmation.json.gz')
    selected=[]
    for family in ('alias','conditional','mapping','addition'):
        for case in (24,25,32,33,40,41):
            selected.extend([r for r in programs if r['source_group']==f'program/{family}/{case:02d}'])
    passages=[]
    for cohort in ('gum','ewt'):
      for split in ('connected_test','matched_test'):
        passages.extend([r for r in natural if r['cohort']==cohort and r['split']==split][:8])
    assert len(selected)==96 and len(passages)==32
    return passages,selected

def main(key):
    import torch
    from rdc_operator_model import load,memory
    from phase2730_rdc_law_scale import Trace
    out=BASE/'scale'/key
    if (out/'result.json').exists():return
    passages,programs=material();rows=passages+programs
    immutable(BASE/'scale/protocol.json',{'natural_ids':[r['sample_id'] for r in passages],'program_ids':[r['sample_id'] for r in programs],
      'models':['qwen4','qwen14','glm4'],'native_program_generation_cap':8,'batch':1,
      'scope':'Same source text/semantic cases, each native tokenizer/template/width; full-coordinate fields and all finalMLPunits. No cross-model coordinate identification.',
      'pilot':'First two natural and first two program samples estimate remaining cost; enforce7200s process limit. Longer generation is a separate diagnostic, never replace8-token records.'})
    start=time.monotonic();guard(900*1024**2)
    model,tok=load(key,out/'residency',cpu_gib=11) if key=='qwen14' else load(key,out/'residency')
    torch.set_num_threads(2);device=model.get_input_embeddings().weight.device;trace=Trace(model)
    source_full={};active={'enabled':False}
    def sources(m,args,result):
        if active['enabled']:
            h=result[0] if isinstance(result,tuple) else result
            source_full['H_early_sources']=bits(h[0])
    handle=model.model.layers[trace.early-1].register_forward_hook(sources)
    runtime={'timestamp':stamp(),'source':snapshot(Path(__file__)),'model':key,'width':model.config.hidden_size,
      'units':model.config.intermediate_size,'depth':trace.depth,'early':trace.early,'last_block':trace.block,
      'dtype':str(model.dtype),'quantized':bool(getattr(model,'is_quantized',False)),
      'device_map':getattr(model,'hf_device_map',{'first_parameter':str(next(model.parameters()).device)}),
      'memory':memory(),'precision_scope':'Native BF16 fullmodel, no quantization; models execute in independent serial processes.'}
    save(out/'runtime.json',runtime);records=[];times=defaultdict(list)
    # Two samples of each type precede the remaining fixed material, no outcome-based selection.
    order=passages[:2]+programs[:2]+passages[2:]+programs[2:]
    try:
      with torch.inference_mode():
        for index,row in enumerate(order):
            sid=row['sample_id'];path=out/'fields'/f'{sid}.npz';commit=out/'commits'/f'{sid}.json'
            if commit.exists():
                rr=read(commit);assert sha(path)==rr['array_sha'];records.append(rr);times[rr['kind']].append(rr['seconds']);continue
            tick=time.monotonic();isprogram=row['kind']=='controlled_program'
            if isprogram:
                text=tok.apply_chat_template([{'role':'user','content':row['original_text']}],tokenize=False,add_generation_prompt=True,
                  **({'enable_thinking':False} if key.startswith('qwen') else {}))
            else:text=row['text']
            enc=tok(text,add_special_tokens=False,return_offsets_mapping=True)
            positions=[len(enc['input_ids'])-1] if isprogram else [max(i for i,(s,e) in enumerate(enc['offset_mapping']) if e>s and e<=row['token_offsets'][a][1]) for a in row['anchors']]
            if key=='qwen4':positions=row['anchors']
            trace.reset(positions);trace.enabled=True;active['enabled']=True
            ids=torch.tensor([enc['input_ids']],device=device)
            post=model.model(input_ids=ids,use_cache=False).last_hidden_state
            trace.enabled=False;active['enabled']=False
            field={**trace.fields,**source_full,'H':np.stack([trace.H[i] for i in range(trace.depth+1)]),'positions':np.array(positions),
              'postnorm':bits(post[0,positions]),'alltoken_layer_moments':np.stack([trace.moments[i] for i in range(trace.depth+1)]),
              'token_ids':np.array(enc['input_ids'],np.int32)}
            logits=model.lm_head(post[0,positions]).float();lp=logits.log_softmax(-1)
            field['argmax']=logits.argmax(-1).cpu().numpy()
            record={k:row[k] for k in ('sample_id','source_group','cohort','split','kind')}
            record.update(prompt_text=text,prompt_ids=enc['input_ids'],offsets=enc['offset_mapping'],positions=positions,
              all_token_layer_identities=trace.hashes,tokens=len(enc['input_ids']))
            if index==0:
                plain=model.model(input_ids=ids,use_cache=False).last_hidden_state
                assert torch.equal(plain,post);record['observer_same_shape_exact_noop']=True
                del plain
            if isprogram:
                target=tok(row['target'],add_special_tokens=False)['input_ids'];record['native_target_ids']=target
                field['first_target_nll']=(-lp[:,target[0]]).cpu().numpy()
                eos=model.generation_config.eos_token_id or tok.eos_token_id;stop=set(eos if isinstance(eos,list) else [eos])
                generated=model.generate(input_ids=ids,use_cache=True,do_sample=False,max_new_tokens=8,eos_token_id=eos,pad_token_id=tok.pad_token_id or tok.eos_token_id)
                seq=generated[0,len(enc['input_ids']):].tolist();textout=tok.decode(seq,skip_special_tokens=True)
                record.update(generated=textout,generated_ids=seq,target=row['target'],family=row['family'],representation=row['representation'],depth=row['depth'],
                  exact_correct=textout.strip()==row['target'],first_token_correct=bool(seq and seq[0]==target[0]),
                  eos=any(t in stop for t in seq),token_limit_censored=not any(t in stop for t in seq) and len(seq)==8)
                del generated
            if key=='qwen4':
                kind='program' if isprogram else 'natural'
                with np.load(BASE/'capture'/kind/f'{sid}.npz') as z:
                    assert np.array_equal(field['H'],z['H'])
                    assert np.array_equal(field['mlp'],z['L35_mlp'])
                record['independent_Q4_replay_exact']=True
                if isprogram:
                    prior=read(BASE/'capture/commits'/f'{sid}.json');assert prior['generated_ids']==record['generated_ids']
            npz(path,**field);record['array_sha']=sha(path);record['seconds']=time.monotonic()-tick
            save(commit,record);records.append(record);times[row['kind']].append(record['seconds'])
            trace.reset([]);source_full.clear();del field,post,ids,logits,lp
            gc.collect();torch.cuda.empty_cache()
            if index==3:
                estimate=30*np.mean(times['natural'])+94*np.mean(times['controlled_program'])+180
                pilot={'timestamp':stamp(),'elapsed':time.monotonic()-start,'estimated_remaining_seconds':float(estimate),
                  'natural_seconds':times['natural'],'program_seconds':times['controlled_program'],
                  'method':'First2of each type, remaining fixed counts,180sanalysis reserve; variable generation lengths remain a source of cost uncertainty.',
                  'passed':bool(estimate+time.monotonic()-start<7200)}
                save(out/'pilot.json',pilot);assert pilot['passed'],pilot
            guard(100*1024**2);assert memory()['host_available_bytes']>2*1024**3
            assert time.monotonic()-start<7200
            print('BINDING_SCALE',key,index+1,len(order),'seconds',round(time.monotonic()-start,1),flush=True)
    except Exception as exc:
        import traceback
        save(out/('failure_'+str(int(time.time()))+'.json'),{'error':str(exc),'traceback':traceback.format_exc(),'completed':len(records),'seconds':time.monotonic()-start})
        ledger('failed_binding_scale_'+key,time.monotonic()-start);raise
    finally:
        handle.remove();trace.close();del trace,model;gc.collect();torch.cuda.empty_cache()
    summaries=[]
    for split in ('validation','test','depth_test'):
      for rep in ('en','zh','python','en_reordered'):
        rr=[r for r in records if r['kind']=='controlled_program' and r['split']==split and r['representation']==rep]
        summaries.append({'split':split,'representation':rep,'rows':len(rr),**{k:float(np.mean([r[k] for r in rr])) for k in ('exact_correct','first_token_correct','eos','token_limit_censored')}})
    unitstats=[]
    for cohort in ('gum','ewt','program_en','program_zh','program_python','program_en_reordered'):
        pp=[];uu=[]
        for record in records:
            if record['cohort']!=cohort:continue
            with np.load(out/'fields'/f'{record["sample_id"]}.npz') as z:
                g=unbits(z['gate']).astype(float);u=unbits(z['up']).astype(float)
                pp.append(g/(1+np.exp(-np.clip(g,-80,80))));uu.append(u)
        phi=np.concatenate(pp);up=np.concatenate(uu);cov=(phi*up).mean(0)-phi.mean(0)*up.mean(0)
        npz(out/'all_unit_relations'/f'{cohort}.npz',phi_mean=phi.mean(0),up_mean=up.mean(0),product_mean=(phi*up).mean(0),covariance=cov)
        unitstats.append({'cohort':cohort,'anchors':len(phi),'units':len(cov),'covariance_l2':float(np.linalg.norm(cov))})
    result={'timestamp':stamp(),'source':snapshot(Path(__file__)),'model':key,'natural_rows':32,'program_rows':96,
      'runtime':runtime,'program_summary':summaries,'all_unit_statistics':unitstats,'seconds':time.monotonic()-start,
      'limits':['8-token censoring is not a proof of content failure.','Native chat/segmentation/architecture/training differ; not causal model-size isolation.','No new larger-model parameter training or cross-model physical coordinate projection.']}
    save(out/'result.json',result);ledger('binding_native_scale_'+key,result['seconds'])
    print('BINDING_SCALE_COMPLETE',key,summaries,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',required=True,choices=['qwen4','qwen14','glm4']);main(p.parse_args().model)
