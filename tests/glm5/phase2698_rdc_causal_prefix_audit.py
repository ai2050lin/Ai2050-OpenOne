"""Same-shape falsification of apparent future-to-past response in causal prefill."""
import gc
from rdc_mechanism_common import *
from phase2693_rdc_language_capture import Capture
OUT=CAMPAIGN/'d_generalization'

def main():
    import torch
    from phase2662_symmetric_mapping_contract import load_native
    rows=read(CAMPAIGN/'b_relations/material.json');pairs=[]
    for r in rows:
        if r['unit']!=6 or r['negative_query']:continue
        other=next(s for s in rows if (s['family'],s['unit'],s['fact_truth'],s['language'],s['negative_query'])==(r['family'],r['unit'],r['fact_truth'],r['language'],True))
        pairs.append((r,other))
    assert len(pairs)==32
    immutable(OUT/'causal_prefix_protocol.json',{'source_sha':sha(Path(__file__)),'pairs':32,'new_forwards':64,
        'selection':'unit6,all8families,bothsupportstates,bothlanguages; positive/negativequery pair',
        'check':'Only common token prefix before first differing token. Native natural-length past difference compared with right-padded batch1 equal-length shape; masked pad tokens, unchanged real prefix and positions.',
        'scope':'Shape/rounding diagnostic, not semantic intervention or extension benchmark; no inverse temporal causation claim.',
        'resource':'One nonquant BF16 CUDA Qwen4 model;64short prefills; fullH compared in memory; only full-coordinate error summaries retained.'})
    model,tok=load_native('qwen4');cap=Capture(model);results=[];summaries={}
    try:
        with torch.inference_mode():
            for pi,(a,b) in enumerate(pairs):
                shared=next((i for i,(x,y) in enumerate(zip(a['prompt_ids'],b['prompt_ids'])) if x!=y),min(len(a['prompt_ids']),len(b['prompt_ids'])))
                raw=[]
                for r in (a,b):
                    with np.load(CAMPAIGN/'b_relations'/f'fields/{r["sample_id"]}.npz') as z:raw.append(unbits(z['h'][:,:shared]))
                natural=raw[1]-raw[0];del raw
                length=max(len(a['prompt_ids']),len(b['prompt_ids']));captured=[]
                for r in (a,b):
                    n=len(r['prompt_ids']);ids=torch.tensor([r['prompt_ids']+[tok.pad_token_id or tok.eos_token_id]*(length-n)],device='cuda:0')
                    mask=torch.tensor([[1]*n+[0]*(length-n)],device='cuda:0');cap.positions=[0];cap.arrays={};cap.enabled=True
                    model.model(input_ids=ids,attention_mask=mask,use_cache=False);cap.enabled=False
                    captured.append(np.stack([unbits(cap.arrays[f'H{l}'][:shared]) for l in range(37)]))
                matched=captured[1]-captured[0];del captured
                result={'a':a['sample_id'],'b':b['sample_id'],'shared_tokens':shared,'lengths':[len(a['prompt_ids']),len(b['prompt_ids'])],
                    'natural_nonzero_scalars':int(np.count_nonzero(natural)),'natural_max_abs':float(np.abs(natural).max()),
                    'same_shape_nonzero_scalars':int(np.count_nonzero(matched)),'same_shape_max_abs':float(np.abs(matched).max())}
                results.append(result);summaries[f'pair{pi}_natural_max_by_coordinate']=np.abs(natural).max(1);summaries[f'pair{pi}_same_shape_max_by_coordinate']=np.abs(matched).max(1)
                print('CAUSAL_PREFIX',pi+1,32,result['natural_max_abs'],result['same_shape_max_abs'],flush=True)
                del natural,matched;cap.arrays={};gc.collect()
    finally:cap.close()
    npz(OUT/'causal_prefix_all_coordinate_errors.npz',**summaries)
    save(OUT/'causal_prefix_result.json',{'timestamp':stamp(),'pairs':results,
        'natural_nonzero_pairs':sum(r['natural_nonzero_scalars']>0 for r in results),
        'same_shape_nonzero_pairs':sum(r['same_shape_nonzero_scalars']>0 for r in results),
        'limits':['Only32 unit6pairs audited; other tokenization/precision/device paths need separate checks.',
            'U/V precede the changed question: observed differences there are not evidence for response to unavailable future semantics.',
            'Last-query C has access to both facts and question; this diagnostic does not erase its heldout readable signal.']})
    del model;gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':main()
