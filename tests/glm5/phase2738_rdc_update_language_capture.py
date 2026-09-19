"""All-width native bilingual relation fields with explicit answer-style boundaries."""
import gc
from rdc_update_common import *

def main():
    import torch
    from phase2662_symmetric_mapping_contract import load_native
    from phase2728_rdc_law_capture import Observer
    out=BASE/'language_capture';start=time.monotonic()
    if (out/'result.json').exists():return
    rows=gzread(BASE/'language_material.json.gz');model,tok=load_native('qwen4');model.eval();torch.set_num_threads(2)
    observer=Observer(model);device=model.get_input_embeddings().weight.device
    fixtures={next(r['sample_id'] for r in rows if r['family']==family and r['language']==lang and r['answer_style']=='direct')
      for family in sorted({r['family'] for r in rows}) for lang in ('en','zh')}
    immutable(out/'protocol.json',{'material_sha256':sha(BASE/'language_material.json.gz'),'rows':640,'model':'qwen3-4b','dtype':'BF16','quantized':False,
      'fixtures':sorted(fixtures),'anchors':'End of factual body and final generation prompt; full original token coordinates for every source and all layer anchors.',
      'scores':'Full CE and within declaredYes/No or是/否 candidate CE at final prompt. Explanation first-token score is not full-answer accuracy.',
      'capture':'Observer scans all coordinates at all tokens/layers; no PCA/TopK. Full layer×token archives at10predeclared fixtures.'})
    records=[];guard(1400*1024**2)
    try:
      with torch.inference_mode():
        for i,row in enumerate(rows):
            sid=row['sample_id'];path=out/'fields'/f'{sid}.npz';commit=out/'commits'/f'{sid}.json'
            if commit.exists():
                r=read(commit);assert sha(path)==r['array_sha256'];records.append(r);continue
            tick=time.monotonic();ids=torch.tensor([row['prompt_ids']],device=device);p=row['anchors']
            observer.reset(p,sid in fixtures,i<2);observer.enabled=True
            post=model.model(input_ids=ids,use_cache=False).last_hidden_state;observer.enabled=False
            z=model.lm_head(post[0,p]).float();lp=z.double().log_softmax(-1);candidates=row['candidate_ids'];clp=z[:,candidates].double().log_softmax(-1)
            target=row['target_ids'][0];local=candidates.index(target);logmass=torch.logsumexp(lp[:,candidates],-1)
            fields={'H':np.stack([observer.H[k] for k in range(37)]),'H12_sources':observer.source_H12,'embedding':observer.H[0],
              'positions':np.array(p),'token_ids':np.array(row['prompt_ids'],np.int32),'postnorm':bits(post[0,p]),**observer.factors,
              'last_residual':observer.training['residual'][p],'last_x':observer.training['x'][p],
              'alltoken_layer_moments':np.stack([observer.moments[k] for k in range(37)]),
              'alltoken_unit_moments':np.stack([observer.unit_moments[k] for k in (6,16,35)]),
              'argmax':z.argmax(-1).cpu().numpy(),'entropy':(-(lp.exp()*lp).sum(-1)).cpu().numpy(),
              'full_loss':(-lp[:,target]).cpu().numpy(),'content_loss':(-clp[:,local]).cpu().numpy(),
              'format_loss':(-logmass).cpu().numpy(),'candidate_mass':logmass.exp().cpu().numpy(),
              'conditional_argmax':np.array(candidates)[clp.argmax(-1).cpu().numpy()],'candidate_logprobs':clp.cpu().numpy()}
            assert np.max(abs(fields['full_loss']-fields['content_loss']-fields['format_loss']))<1e-10
            if sid in fixtures:npz(out/'full_fields'/f'{sid}.npz',H=np.stack([observer.full_H[k] for k in range(37)]),postnorm=bits(post[0]))
            if i==0:
                plain=model.model(input_ids=ids,use_cache=False).last_hidden_state;assert torch.equal(plain,post)
                del plain
            npz(path,**fields);record={k:row[k] for k in ('sample_id','source_group','family','language','answer_style','split','target')}
            record.update(array_sha256=sha(path),tokens=len(row['prompt_ids']),checks=observer.checks,alltoken_layer_identities=observer.H_hashes,
              first_token_correct=int(fields['argmax'][-1])==target,conditional_correct=int(fields['conditional_argmax'][-1])==target,
              full_loss=float(fields['full_loss'][-1]),content_loss=float(fields['content_loss'][-1]),format_loss=float(fields['format_loss'][-1]),
              seconds=time.monotonic()-tick,full_fixture=sid in fixtures)
            save(commit,record);records.append(record);observer.reset([],False);del fields,post,z,lp,ids
            if (i+1)%32==0:print('LANGUAGE_CAPTURE',i+1,len(rows),round(time.monotonic()-start,1),flush=True)
            guard(25*1024**2);assert time.monotonic()-start<7200
      summary=[]
      for family,language,style,split in sorted({(r['family'],r['language'],r['answer_style'],r['split']) for r in records}):
        rr=[r for r in records if (r['family'],r['language'],r['answer_style'],r['split'])==(family,language,style,split)]
        summary.append({'family':family,'language':language,'style':style,'split':split,'rows':len(rr),
          **{k:float(np.mean([r[k] for r in rr])) for k in ('full_loss','content_loss','format_loss','first_token_correct','conditional_correct')}})
      result={'timestamp':stamp(),'source':snapshot(__file__),'rows':len(records),'tokens':sum(r['tokens'] for r in records),'summary':summary,
        'seconds':time.monotonic()-start,'peak_cuda_bytes':torch.cuda.max_memory_allocated(),'scope':'Controlled multilingual language observation, not all authentic corpus and not free-generation performance.'}
      save(out/'result.json',result);ledger('native_bilingual_relation_capture',result['seconds']);print('LANGUAGE_CAPTURE_DONE',result['seconds'],flush=True)
    except Exception as exc:failure(out,start,exc);raise
    finally:
      observer.close();del observer,model;gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':main()
