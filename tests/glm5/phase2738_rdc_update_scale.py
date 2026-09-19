"""Serial original-checkpoint replication on the new mixed and bilingual materials."""
import argparse, gc
from collections import defaultdict
from rdc_update_common import *

def material():
    out=BASE/'scale';p=out/'protocol.json'
    if p.exists():return read(p),gzread(out/'material.json.gz')
    natural=gzread(BASE/'natural_material.json.gz');program=gzread(BASE/'program_material.json.gz');language=gzread(BASE/'language_material.json.gz')
    passages=[r for cohort in ('gum','ewt') for split in ('new_connected','new_matched') for r in [x for x in natural if x['cohort']==cohort and x['split']==split][:8]]
    groups=sorted({r['source_group'] for r in program if r['split']=='mixed_holdout'})[:14]
    programs=[r for r in program if r['source_group'] in groups];languages=[]
    for family in sorted({r['family'] for r in language}):
        gg=sorted({r['source_group'] for r in language if r['family']==family and r['split']=='language_test'})[:2]
        languages.extend(r for r in language if r['source_group'] in gg)
    rows=passages+programs+languages;assert len(rows)==128
    generation={r['sample_id'] for cohort in ('gum','ewt') for split in ('new_connected','new_matched') for r in [x for x in passages if x['cohort']==cohort and x['split']==split][:2]}
    generation.update(r['sample_id'] for r in programs if r['source_group'] in groups[:2])
    for family in sorted({r['family'] for r in languages}):
        group=min(r['source_group'] for r in languages if r['family']==family)
        generation.update(r['sample_id'] for r in languages if r['source_group']==group)
    assert len(generation)==36
    protocol={'timestamp':stamp(),'models':['qwen4','qwen14','glm4'],'source_ids':[r['sample_id'] for r in rows],
      'natural_rows':32,'mixed_rows':56,'language_rows':40,'generation_ids':sorted(generation),'generation_max_tokens':128,
      'precision':'Original BF16 local checkpoints; device_map auto/offload for larger models, one separate process at a time. No checkpoint mutation or quantization.',
      'fields':'Every coordinate at all layer anchors and every source at own early layer floor(depth/3); all last-MLP units; all-token all-layer moments aggregated by cohort and all tensor identities retained.',
      'alignment':'Natural sources match character endpoints; language body boundaries mapped within own chat template; model-specific tokenization is explicit. Equal indices across models do not denote equal functionality.',
      'pilot':'First natural/program/language prefill plus first2 of each generation type estimate fixed remaining workload before expansion. Native answer length remains uncertain; hard7200second process bound.',
      'retention':'Client-queryable full arrays, no all-layer/full-token vectors except early sources; no Top-K.',
      'scope':'Behavior and all-coordinate source/unit replication, not retraining the larger models or an isolated causal test of model size.'}
    compressed(out/'material.json.gz',rows);immutable(p,protocol);return protocol,rows

def native_prompt(tok,key,row):
    if row['kind']=='natural':return row['text']
    return tok.apply_chat_template([{'role':'user','content':row['original_text']}],tokenize=False,add_generation_prompt=True,
      **({'enable_thinking':False} if key.startswith('qwen') else {}))

def main(key):
    import torch
    import rdc_operator_model as loader
    from phase2730_rdc_law_scale import Trace
    from rdc_update_scoring import score as answer_score
    loader.snapshot=snapshot  # Original loader bookkeeping must not write its earlier campaign.
    out=BASE/'scale'/key;protocol,rows=material();start=time.monotonic()
    if (out/'result.json').exists():return
    guard(480*1024**2)
    model,tok=loader.load(key,out/'residency',cpu_gib=11) if key=='qwen14' else loader.load(key,out/'residency')
    model.eval();torch.set_num_threads(2);device=model.get_input_embeddings().weight.device;trace=Trace(model);source={}
    def early(m,a,o):
        if trace.enabled:source['H_early_sources']=bits((o[0] if isinstance(o,tuple) else o)[0])
    hook=model.model.layers[trace.early-1].register_forward_hook(early)
    runtime={'timestamp':stamp(),'source':snapshot(__file__),'width':model.config.hidden_size,'units':model.config.intermediate_size,
      'depth':trace.depth,'early':trace.early,'last_block':trace.block,'dtype':str(model.dtype),'quantized':bool(getattr(model,'is_quantized',False)),
      'device_map':getattr(model,'hf_device_map',{}),'model':key};save(out/'runtime.json',runtime)
    groups=defaultdict(list)
    for r in rows:groups[r['kind']].append(r)
    order=[r for kind in ('natural','controlled_program','controlled_language') for r in groups[kind][:2]]
    done={r['sample_id'] for r in order};order.extend(r for r in rows if r['sample_id'] not in done)
    records=[];moments={};tokens=defaultdict(int);timing=defaultdict(list)
    eos=model.generation_config.eos_token_id or tok.eos_token_id;stop=set(eos if isinstance(eos,list) else [eos])
    try:
      with torch.inference_mode():
        for i,row in enumerate(order):
            sid=row['sample_id'];tick=time.monotonic();text=native_prompt(tok,key,row);enc=tok(text,add_special_tokens=False,return_offsets_mapping=True)
            if row['kind']=='natural':ends=[row['token_offsets'][a][1] for a in row['anchors']]
            elif row['kind']=='controlled_language':ends=[text.index(row['body'])+len(row['body']),len(text)]
            else:ends=[len(text)]
            positions=[max(j for j,(a,b) in enumerate(enc['offset_mapping']) if b>a and b<=end) for end in ends]
            if key=='qwen4':
                assert enc['input_ids']==row['prompt_ids'];positions=row['anchors']
            trace.reset(positions);trace.enabled=True;source.clear();ids=torch.tensor([enc['input_ids']],device=device)
            post=model.model(input_ids=ids,use_cache=False).last_hidden_state;trace.enabled=False
            fields={**trace.fields,**source,'H':np.stack([trace.H[j] for j in range(trace.depth+1)]),'postnorm':bits(post[0,positions]),
              'token_ids':np.array(enc['input_ids'],np.int32),'positions':np.array(positions)}
            moment=np.stack([trace.moments[j] for j in range(trace.depth+1)]).astype(float)
            cohort=row['cohort'];moments[cohort]=moments.get(cohort,np.zeros_like(moment))+moment;tokens[cohort]+=len(enc['input_ids'])
            logits=model.lm_head(post[0,positions]).float();lp=logits.double().log_softmax(-1)
            rec={k:row[k] for k in ('sample_id','source_group','cohort','split','kind','language')}
            rec.update(prompt_text=text,prompt_ids=enc['input_ids'],offsets=enc['offset_mapping'],positions=positions,
              endpoint_exact=[enc['offset_mapping'][p][1]==end for p,end in zip(positions,ends)],
              alltoken_layer_identities=trace.hashes,tokens=len(enc['input_ids']),target=row.get('target'))
            if row['kind']=='natural':
                targets=[enc['input_ids'][p+1] for p in positions];fields['full_loss']=(-lp[torch.arange(len(positions),device=device),torch.tensor(targets,device=device)]).cpu().numpy()
                rec['first_accuracy']=float(np.mean(logits.argmax(-1).cpu().numpy()==targets))
            else:
                targets=tok(row['target'],add_special_tokens=False)['input_ids'];rec['native_target_ids']=targets
                rec['first_accuracy']=int(logits[-1].argmax())==targets[0];fields['full_loss']=np.array([float(-lp[-1,targets[0]])])
                candidate_texts=row.get('candidate_texts',[str(j) for j in range(1,9)])
                cc=[tok(s,add_special_tokens=False)['input_ids'] for s in candidate_texts]
                rec['native_candidates']=cc
                if len(targets)==1 and all(len(c)==1 for c in cc):
                    candidates=[c[0] for c in cc];clp=logits[-1,candidates].double().log_softmax(-1)
                    fields.update(content_loss=np.array([float(-clp[candidates.index(targets[0])])]),format_loss=np.array([float(-torch.logsumexp(lp[-1,candidates],0))]))
                    rec['conditional_accuracy']=candidates[int(clp.argmax())]==targets[0]
                else:rec['candidate_scope']='Multi-token target/candidate: no single-token content-format claim'
            fields['argmax']=logits.argmax(-1).cpu().numpy();rec['full_loss']=float(fields['full_loss'].mean())
            for name in ('content_loss','format_loss'):
                if name in fields:rec[name]=float(fields[name][0])
            if i==0:
                repeat=model.model(input_ids=ids,use_cache=False).last_hidden_state;assert torch.equal(repeat,post);rec['observer_noop_exact']=True;del repeat
            if key=='qwen4':
                with np.load(native_path(row)) as z:
                    assert np.array_equal(fields['H'],z['H']);assert np.array_equal(fields['mlp'],z['L35_mlp'])
                rec['original_capture_replay_exact']=True
            rec['prefill_seconds']=time.monotonic()-tick;timing[row['kind']].append(rec['prefill_seconds'])
            if sid in protocol['generation_ids']:
                gt=time.monotonic();prompt=enc['input_ids'][:positions[0]+1] if row['kind']=='natural' else enc['input_ids']
                gid=torch.tensor([prompt],device=device);cache=None;generated=[];genstats=[]
                for step in range(protocol['generation_max_tokens']):
                    value=model.model(input_ids=gid,past_key_values=cache,use_cache=True);cache=value.past_key_values
                    z=model.lm_head(value.last_hidden_state[0,-1]).float();chosen=int(z.argmax());generated.append(chosen)
                    genstats.append({'step':step,'chosen':chosen,'logprob':float(z.double().log_softmax(-1)[chosen])})
                    if chosen in stop:break
                    gid=torch.tensor([[chosen]],device=device)
                    if key!='qwen4' and (step+1)%16==0:
                        print('UPDATE_SCALE_GENERATING',key,i+1,step+1,round(time.monotonic()-gt,1),flush=True)
                generated_text=tok.decode(generated,skip_special_tokens=True);graded=answer_score(row,generated_text,generated,stop,protocol['generation_max_tokens']);answer=graded['conservative_final_answer']
                rec.update(generation_prompt_ids=prompt,generated_ids=generated,generated_text=generated_text,steps=genstats,
                  parsed_answer=answer,parsed_accuracy=bool(graded['conservative_final_correct']) if row['kind']!='natural' else None,
                  EOS=generated[-1] in stop,censored=generated[-1] not in stop,generation_seconds=time.monotonic()-gt,answer_scoring=graded)
                timing[row['kind']+'_generation'].append(rec['generation_seconds']);del cache,gid,value,z
            path=out/'fields'/f'{sid}.npz';npz(path,**fields);rec.update(array_sha256=sha(path),seconds=time.monotonic()-tick)
            save(out/'commits'/f'{sid}.json',rec);records.append(rec);trace.reset([]);source.clear();del post,fields,ids,logits,lp,moment
            if i==5:
                estimate=0.
                for kind,rs in groups.items():
                    estimate+=len(rs)*np.mean(timing[kind]);ng=sum(r['sample_id'] in protocol['generation_ids'] for r in rs)
                    if ng:estimate+=ng*np.mean(timing[kind+'_generation'])
                pilot={'timestamp':stamp(),'elapsed':time.monotonic()-start,'estimated_total_native_seconds':float(estimate),
                  'timings':dict(timing),'passed':bool(estimate+300<7200),'scope':'First2per kind; includes128-token cap, not only short first-token timing.'}
                save(out/'pilot.json',pilot);assert pilot['passed'],pilot
            assert loader.memory()['host_available_bytes']>2*1024**3;guard(8*1024**2);assert time.monotonic()-start<7200
            if i<6 or (i+1)%8==0:print('UPDATE_SCALE',key,i+1,len(rows),round(time.monotonic()-start,1),flush=True)
        for cohort,value in moments.items():npz(out/'alltoken_cohort_moments'/f'{cohort}.npz',sums=value,tokens=np.array(tokens[cohort]))
        summaries=[]
        for cohort in sorted({r['cohort'] for r in records}):
            rr=[r for r in records if r['cohort']==cohort];gg=[r for r in rr if 'generated_ids' in r]
            report={'cohort':cohort,'rows':len(rr),'generation_rows':len(gg)}
            for metric in ('full_loss','content_loss','format_loss','first_accuracy','conditional_accuracy'):
                vv=[float(r[metric]) for r in rr if metric in r]
                if vv:report[metric]=float(np.mean(vv))
            for metric in ('parsed_accuracy','EOS','censored'):
                vv=[float(r[metric]) for r in gg if r.get(metric) is not None]
                if vv:report[metric]=float(np.mean(vv))
            summaries.append(report)
        result={'timestamp':stamp(),'source':snapshot(__file__),'model':key,'rows':128,'generation_rows':36,'runtime':runtime,'summaries':summaries,
          'seconds':time.monotonic()-start,'retention':'All early-source and anchor coordinates, all final units, cohort full-coordinate moments, all-layer/token array identities. Per-token other-layer arrays are recomputable, not claimed retained.'}
        recovery=BASE/'runtime_recovery/scale_pilot_and_figure_serialization.json'
        if key=='qwen4' and recovery.exists():
            audit=read(recovery)
            for filename,digest in audit['array_sha256_before_rerun'].items():assert sha(BASE/filename)==digest
            for old in audit['commits_before_rerun']:
                new=read(out/'commits'/f'{old["sample_id"]}.json')
                assert new['prompt_ids']==old['prompt_ids'] and new.get('generated_ids')==old.get('generated_ids')
            result['pre_failure_six_arrays_and_generated_IDs_exact']=True
        backend_audit=BASE/'scale_recovery'/f'{key}_backend_audit.json'
        if backend_audit.exists():result['offload_backend_audit']=read(backend_audit)
        save(out/'result.json',result);ledger('new_three_model_replication_'+key,result['seconds']);print('UPDATE_SCALE_DONE',key,result['seconds'],flush=True)
    except Exception as exc:failure(out,start,exc);raise
    finally:
        hook.remove();trace.close();del model,trace;gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['qwen4','qwen14','glm4'],required=True);main(p.parse_args().model)
