"""Serial unquantized Q4 collection at nonpunctuation and mixed-program boundaries."""
import argparse
import gc
from collections import Counter
from rdc_update_common import *

def main(pilot=False):
    import torch
    from phase2662_symmetric_mapping_contract import load_native
    from phase2728_rdc_law_capture import Observer
    out=BASE/'capture';result_path=out/('pilot.json' if pilot else 'result.json')
    if result_path.exists():return
    assert (BASE/'graph/frozen.json').exists(),'Freeze graph before new natural outcomes.'
    if not pilot:assert read(out/'pilot.json')['passed']
    natural=gzread(BASE/'natural_material.json.gz');program=gzread(BASE/'program_material.json.gz')
    order=[natural[0],natural[64],program[0],program[1]]
    first={r['sample_id'] for r in order};rows=order if pilot else order+[r for r in natural+program if r['sample_id'] not in first]
    fixtures={r['sample_id'] for cohort in sorted({r['cohort'] for r in natural+program}) for r in [s for s in natural+program if s['cohort']==cohort][:2]}
    immutable(out/'protocol.json',{'natural_rows':128,'program_rows':768,'all_layer_full_field_fixtures':sorted(fixtures),
      'batch':1,'dtype':'BF16 native, FP32/FP64 scored probabilities','quantized':False,'padding':False,'capture_cache':False,
      'natural': 'Three declared nonpunctuation-next-word boundaries; later tokens are not predictor inputs. Full-sentence execution has causal mask, but finite-precision prefix-shape differences are audited separately.',
      'program':'Native prefill; no generation during bulk capture. Separate own-history protocol handles answers and stopping.',
      'retention':'All H12 token coordinates, every layer anchor, block6/16/35 all units, all-token full-coordinate moments/hashes; full layer×token tensors for predeclared fixtures.'})
    start=time.monotonic();guard(2500*1024**2);model,tok=load_native('qwen4');model.eval();torch.set_num_threads(2)
    assert model.dtype==torch.bfloat16 and not getattr(model,'is_quantized',False)
    observer=Observer(model);device=model.get_input_embeddings().weight.device
    digits=[tok(str(i),add_special_tokens=False)['input_ids'][0] for i in range(1,9)];records=[]
    runtime={'timestamp':stamp(),'source':snapshot(__file__),'dtype':str(model.dtype),'quantized':False,
      'device_map':getattr(model,'hf_device_map',{}),'attention':'native eager','model_source':'Original local qwen3-4b; no checkpoint files written.'}
    save(out/('pilot_runtime.json' if pilot else 'runtime.json'),runtime)
    try:
      with torch.inference_mode():
        for index,row in enumerate(rows):
            sid=row['sample_id'];path=out/'qwen4'/f'{sid}.npz';commit=out/'commits'/f'{sid}.json'
            if commit.exists():
                r=read(commit);assert r['array_sha256']==sha(path);records.append(r);continue
            tick=time.monotonic();ids=torch.tensor([row['prompt_ids']],device=device);p=row['anchors']
            assert tok(row['text'],add_special_tokens=False)['input_ids']==row['prompt_ids']
            observer.reset(p,sid in fixtures,index<2);observer.enabled=True
            post=model.model(input_ids=ids,use_cache=False).last_hidden_state;observer.enabled=False
            z=model.lm_head(post[0,p]).float();lp=z.double().log_softmax(-1)
            fields={'H':np.stack([observer.H[k] for k in range(37)]),'H12_sources':observer.source_H12,'embedding':observer.H[0],
              'positions':np.array(p),'token_ids':np.array(row['prompt_ids'],np.int32),'postnorm':bits(post[0,p]),**observer.factors,
              'last_residual':observer.training['residual'][p],'last_x':observer.training['x'][p],
              'alltoken_layer_moments':np.stack([observer.moments[k] for k in range(37)]),
              'alltoken_unit_moments':np.stack([observer.unit_moments[k] for k in (6,16,35)]),
              'argmax':z.argmax(-1).cpu().numpy(),'entropy':(-(lp.exp()*lp).sum(-1)).cpu().numpy()}
            target=row['target_ids'] if row['kind']=='controlled_program' else [row['prompt_ids'][a+1] for a in p]
            tt=torch.tensor(target,device=device);fields['target_ids']=np.array(target)
            fields['full_loss']=(-lp[torch.arange(len(p),device=device),tt]).cpu().numpy()
            if row['kind']=='controlled_program':
                local=digits.index(target[0]);dlp=z[:,digits].double().log_softmax(-1);logmass=torch.logsumexp(lp[:,digits],-1)
                fields.update(content_loss=(-dlp[:,local]).cpu().numpy(),format_loss=(-logmass).cpu().numpy(),
                  digit_mass=logmass.exp().cpu().numpy(),conditional_argmax=np.array(digits)[dlp.argmax(-1).cpu().numpy()],
                  digit_logprobs=dlp.cpu().numpy())
                assert np.max(abs(fields['full_loss']-fields['content_loss']-fields['format_loss']))<1e-10
            record={k:row[k] for k in ('sample_id','source_group','cohort','kind','split')}
            record.update(tokens=len(row['prompt_ids']),anchors=p,checks=observer.checks,alltoken_layer_identities=observer.H_hashes)
            if index==0:
                plain=model.model(input_ids=ids,use_cache=False).last_hidden_state
                assert torch.equal(plain,post);record['observer_same_shape_exact_noop']=True
                # Re-run shorter prefixes: compare numerics, do not impose false
                # bitwise equality across different GEMM/attention shapes.
                shape=[]
                for a,position in enumerate(p):
                    short=model.model(input_ids=ids[:,:position+1],use_cache=False).last_hidden_state[0,-1]
                    slp=model.lm_head(short).float().double().log_softmax(-1)
                    shape.append({'position':position,'last_postnorm_relative_l2':float((short.float()-post[0,position].float()).norm()/post[0,position].float().norm()),
                      'full_vocab_KL_fullshape_to_prefix':float((lp[a].exp()*(lp[a]-slp)).sum()),
                      'argmax_same':int(slp.argmax())==int(z[a].argmax())})
                record['prefix_shape_audit']=shape;del plain
            if sid in fixtures:
                npz(out/'full_fields'/f'{sid}.npz',H=np.stack([observer.full_H[k] for k in range(37)]),postnorm=bits(post[0]),anchor_logprobs=lp.cpu().numpy())
            npz(path,**fields);record.update(array_sha256=sha(path),seconds=time.monotonic()-tick,full_fixture=sid in fixtures)
            save(commit,record);records.append(record);observer.reset([],False);del fields,post,z,lp,ids
            guard(64*1024**2);assert time.monotonic()-start<7200
            if index<4 or (index+1)%32==0:print('UPDATE_CAPTURE',index+1,len(rows),round(time.monotonic()-start,1),flush=True)
      report={'timestamp':stamp(),'source':snapshot(__file__),'rows':len(records),'tokens':sum(r['tokens'] for r in records),
        'counts':dict(Counter(r['kind'] for r in records)),'runtime':runtime,'seconds':time.monotonic()-start,
        'pilot':pilot,'passed':True,'estimated_full_native_seconds':sum(r['seconds'] for r in records)/len(records)*896 if pilot else None,
        'peak_cuda_bytes':torch.cuda.max_memory_allocated()}
      if pilot:assert report['estimated_full_native_seconds']<7200
      save(result_path,report);ledger('native_q4_capture_pilot' if pilot else 'native_q4_capture_remaining',report['seconds']);print('UPDATE_CAPTURE_DONE',report,flush=True)
    except Exception as exc:failure(out,start,exc);raise
    finally:
      observer.close();del observer,model;gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');main(p.parse_args().pilot)
