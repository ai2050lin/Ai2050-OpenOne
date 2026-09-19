"""Deploy full-coordinate decoder outputs through the entire original vocabulary."""
from collections import defaultdict
from rdc_query_common import *
from phase2741_rdc_query_fit import rows_and_queries,packet,predict
from phase2741_rdc_query_rules import OUT,NAMES

def main():
    import torch
    file=OUT/'vocabulary_result.json'
    if file.exists():return
    assert read(OUT/'fit_result.json')['all_passed'];rows,probes,qi=rows_and_queries();rows=[r for r in rows if r['split']=='test']
    start=time.monotonic();model=None
    try:
      model,tok=load('qwen4',OUT/'vocabulary');records=[];maximum=0.;examples=[];example_cohorts=set()
      with np.load(BASE/'prototypes/qwen4.npz') as z:pr=np.stack([unbits(z[f'p{i}_H13'][-1]) for i in range(100)]).astype(float)
      with np.load(OUT/'decoder.npz') as z:beta=z['beta'].copy()
      groups=defaultdict(list)
      for i,p in enumerate(probes):groups[len(p['token_ids'])].append(i)
      with torch.inference_mode():
        for ri,row in enumerate(rows):
            fx,y=packet(row,pr);pred=np.stack([predict(fx(c),beta[c])[2] for c in range(5)])
            with np.load(BASE/'capture/fields'/f"{row['sample_id']}.npz") as z:reference=z['full_vocabulary_statistics'].copy()
            metrics=np.zeros((5,100,4));projection_rounding=np.zeros((5,100));predicted_argmax=np.zeros((5,100),np.int64)
            for length,indices in sorted(groups.items()):
              for j0 in range(0,len(indices),16):
                batch=indices[j0:j0+16]
                h=torch.tensor(y[2,batch],device='cuda',dtype=torch.bfloat16);z=model.lm_head(h).float();lp=z.double().log_softmax(-1);prob=lp.exp()
                entropy=-(prob*lp).sum(-1);maximum=max(maximum,float(np.max(np.abs(entropy.cpu().numpy()-reference[batch,0]))))
                assert np.array_equal(z.argmax(-1).cpu().numpy(),reference[batch,5].astype(int))
                for c in range(5):
                    ph=torch.tensor(pred[c,batch],device='cuda',dtype=torch.bfloat16);pl=model.lm_head(ph).float().double().log_softmax(-1)
                    predicted_argmax[c,batch]=pl.argmax(-1).cpu().numpy()
                    metrics[c,batch]=torch.stack([(prob*(lp-pl)).sum(-1),(pl.exp()*(pl-lp)).sum(-1),-(pl.exp()*pl).sum(-1),
                      (pl.argmax(-1)==lp.argmax(-1)).double()],-1).cpu().numpy()
                    projection_rounding[c,batch]=np.mean((ph.float().cpu().numpy()-pred[c,batch])**2,-1)
            npz(OUT/'vocabulary_fields'/f"{row['sample_id']}.npz",metrics=metrics,FP64_to_native_BF16_vector_rounding_mse=projection_rounding,predicted_argmax_token_ids=predicted_argmax)
            if row['cohort'] not in example_cohorts:
                example_cohorts.add(row['cohort'])
                for q in [0,40,99]:
                    examples.append({'sample_id':row['sample_id'],'cohort':row['cohort'],'prefix_text':row['text'],'probe_index':q,'probe_text':probes[q]['text'],
                      'native_next_token_id':int(reference[q,5]),'native_next_token_text':tok.decode([int(reference[q,5])]),
                      'predicted_next_token':{name:{'id':int(predicted_argmax[c,q]),'text':tok.decode([int(predicted_argmax[c,q])]),'KL_native_to_prediction':float(metrics[c,q,0])} for c,name in enumerate(NAMES)},
                      'scope':'First test sample per cohort and fixedqueryindices, not outcome-selected. Argmax is a single nexttoken after a diagnostic suffix, not a complete answer.'})
            for qs,ix in qi.items():records.append({'sample_id':row['sample_id'],'source_group':row['source_group'],'cohort':row['cohort'],'query_split':qs,'metrics':metrics[:,ix].mean(1).tolist()})
            if (ri+1)%12==0:print('QUERY_FULL_VOCABULARY',ri+1,len(rows),round(time.monotonic()-start,1),flush=True)
        compressed(OUT/'vocabulary_metrics.json.gz',records);summary=[]
        for qs in qi:
          for cohort in ['all','gum','ewt','cmrc']:
            rr=[r for r in records if r['query_split']==qs and (cohort=='all' or r['cohort']==cohort)];groups=[r['source_group'] for r in rr]
            summary.append({'query_split':qs,'cohort':cohort,'prefixes':len(rr),'KL_native_to_prediction':{
              name:clustered([r['metrics'][c][0] for r in rr],groups) for c,name in enumerate(NAMES)},
              'paired_KL_control_minus_ordered':{name:clustered([r['metrics'][c][0]-r['metrics'][4][0] for r in rr],groups) for c,name in enumerate(NAMES[:4])}})
        assert maximum<1e-7
        result={'timestamp':stamp(),'source':snapshot(__file__),'all_passed':True,'test_prefixes':len(rows),'query_endpoints':len(rows)*100,
          'candidate_full_vocabulary_distributions':len(rows)*100*5,'vocabulary':model.config.vocab_size,'native_recomputed_entropy_max_error':maximum,
          'columns':['KL_native_to_prediction','KL_prediction_to_native','prediction_entropy','argmax_matches_native'],
          'summary':summary,'seconds':time.monotonic()-start,'precision':'FP64 KL over all151936tokens after casting FP64 regression predictions to native BF16 before the original head; vector rounding MSE separately retained.',
          'predeclared_index_examples':examples,
          'scope':'Distribution prediction of fixed diagnostic suffixes, not free-generation accuracy or actual text semantics.'}
        save(file,result);ledger('full_vocabulary_query_prediction',result['seconds']);print('QUERY_VOCABULARY_DONE',result['seconds'],flush=True)
    except Exception as exc:failure(OUT/'vocabulary',start,exc);raise
    finally:
        if model is not None:del model
        gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':main()
