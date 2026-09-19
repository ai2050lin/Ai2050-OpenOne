"""New native confirmation/program fields and candidate-free program behavior."""
import gc
from collections import Counter
from rdc_binding_common import *
from phase2728_rdc_law_capture import Observer

def main():
    import torch
    from phase2662_symmetric_mapping_contract import load_native
    out=BASE/'capture'
    if (out/'result.json').exists():print('BINDING_CAPTURE_EXISTS',flush=True);return
    assert (BASE/'prediction/frozen.json').exists() and (BASE/'verification/kernel_math.json').exists()
    natural=gzread(BASE/'natural_confirmation.json.gz');program=gzread(BASE/'program_material.json.gz')
    rows=natural+program;start=time.monotonic();guard(1800*1024**2)
    model,tok=load_native('qwen4');model.eval();torch.set_num_threads(2)
    observer=Observer(model);device=model.get_input_embeddings().weight.device
    fixtures={r['sample_id'] for cohort in sorted({r['cohort'] for r in rows}) for r in [s for s in rows if s['cohort']==cohort][:2]}
    save(out/'runtime.json',{'timestamp':stamp(),'source':snapshot(Path(__file__)),'model':'qwen3-4b',
      'dtype':str(model.dtype),'quantized':bool(getattr(model,'is_quantized',False)),
      'device_map':getattr(model,'hf_device_map',{'actual_first_parameter':str(next(model.parameters()).device)}),'batch':1,'padding':False,'attention':'native eager','cache_capture':False,
      'fixtures':sorted(fixtures),'capture':'All tokens/all coordinates scanned at embedding+36boundaries; completeH12 sources; complete anchor field and block6/16/35factors. Only predeclared fixtures persist all-layer/all-token fields.',
      'generation':'Unmodified model, greedy max8newtokens, own KV history. EOS and censoring reported separately; target never supplied.'})
    records=[];old_checks=[]
    try:
      with torch.inference_mode():
        for index,row in enumerate(rows):
            sid=row['sample_id'];kind='natural' if row['kind']=='natural' else 'program';file=out/kind/f'{sid}.npz';commit=out/'commits'/f'{sid}.json'
            if commit.exists():
                r=read(commit);assert sha(file)==r['array_sha'];records.append(r);continue
            tick=time.monotonic();ids=torch.tensor([row['prompt_ids']],device=device);p=row['anchors']
            assert tok(row['text'],add_special_tokens=False)['input_ids']==row['prompt_ids']
            observer.reset(p,sid in fixtures,index<2);observer.enabled=True
            post=model.model(input_ids=ids,use_cache=False).last_hidden_state;observer.enabled=False
            fields={'H':np.stack([observer.H[k] for k in range(37)]),'H12_sources':observer.source_H12,
              'embedding':observer.H[0],'postnorm':bits(post[0,p]),'positions':np.array(p),'token_ids':np.array(row['prompt_ids'],np.int32),
              **observer.factors,'last_residual':observer.training['residual'][p],'last_x':observer.training['x'][p],
              'H_full_sums':np.stack([observer.moments[k] for k in range(37)]),
              'unit_full_sums':np.stack([observer.unit_moments[k] for k in (6,16,35)])}
            logits=model.lm_head(post[0,p]).float();lp=logits.log_softmax(-1)
            fields['argmax']=logits.argmax(-1).cpu().numpy();fields['entropy']=(-(lp.exp()*lp).sum(-1)).cpu().numpy()
            r={k:row[k] for k in ('sample_id','source_group','cohort','split','kind')}
            r.update(tokens=len(row['prompt_ids']),checks=observer.checks,full_fixture=sid in fixtures)
            if kind=='program':
                target=row['target_ids'][0];fields['target_nll']=(-lp[:,-0+target]).cpu().numpy()
                generated=model.generate(input_ids=ids,use_cache=True,do_sample=False,max_new_tokens=8,pad_token_id=tok.pad_token_id)
                seq=generated[0,len(row['prompt_ids']):].tolist();text=tok.decode(seq,skip_special_tokens=True)
                r.update(target=row['target'],generated=text,generated_ids=seq,eos=tok.eos_token_id in seq,
                  exact_correct=text.strip()==row['target'],first_token_correct=bool(seq and seq[0]==target),
                  token_limit_censored=tok.eos_token_id not in seq and len(seq)==8,representation=row['representation'],family=row['family'],depth=row['depth'])
                del generated
            if observer.full:
                npz(out/'full_fields'/f'{sid}.npz',H=np.stack([observer.full_H[k] for k in range(37)]),postnorm=bits(post[0]))
            npz(file,**fields);r['array_sha']=sha(file);r['seconds']=time.monotonic()-tick
            save(commit,r);records.append(r)
            observer.reset([],False);del fields,post,logits,lp,ids
            guard(128*1024**2)
            assert time.monotonic()-start<read(BASE/'resources.json')['per_process_ceiling_seconds']
            if index<2 or (index+1)%32==0:print('BINDING_NATIVE_CAPTURE',index+1,len(rows),'seconds',round(time.monotonic()-start,1),flush=True)
    except Exception as exc:
        import traceback
        save(out/('failure_'+str(int(time.time()))+'.json'),{'timestamp':stamp(),'error':str(exc),'traceback':traceback.format_exc(),'completed':len(records),'seconds':time.monotonic()-start})
        ledger('failed_binding_capture',time.monotonic()-start);raise
    finally:
        observer.close();del observer,model;gc.collect();torch.cuda.empty_cache()
    result={'timestamp':stamp(),'rows':len(records),'natural_rows':len(natural),'program_rows':len(program),
      'tokens':sum(r['tokens'] for r in records),'seconds':time.monotonic()-start,'fixtures':sorted(fixtures),
      'program_results':[],'scope':'New natural graph confirmation and controlled program behavior, not all-natural semantic coverage.'}
    for key in sorted({r['split']+'/'+r['cohort'] for r in records if r['kind']=='controlled_program'}):
        rr=[r for r in records if r['split']+'/'+r['cohort']==key]
        result['program_results'].append({'group':key,'rows':len(rr),**{k:sum(r[k] for r in rr)/len(rr) for k in ('exact_correct','first_token_correct','eos','token_limit_censored')}})
    save(out/'result.json',result);ledger('new_native_qwen4_capture',result['seconds'],tokens=result['tokens'])
    print('BINDING_NATIVE_CAPTURE_COMPLETE',result,flush=True)

if __name__=='__main__':main()
