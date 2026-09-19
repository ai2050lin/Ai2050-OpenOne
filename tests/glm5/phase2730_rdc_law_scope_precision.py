"""Diagnose scope-control execution shape without changing any saved rollout."""
import gc
from rdc_law_common import *
from rdc_law_live import Live,compare_caches
from phase2730_rdc_law_deployment import inputs,classifier


def main():
    import torch
    from rdc_operator_model import load
    start=time.monotonic();out=BASE/'deployment/scope_precision';guard(20*1024**2)
    immutable(out/'protocol.json',{'timestamp':stamp(),'source':snapshot(Path(__file__)),
        'live_source_before_correction':snapshot(ROOT/'tests/glm5/rdc_law_live.py'),
        'samples':['confirmation-gum-0960','confirmation-gum-0978'],
        'reason':'First scope row passed, second stopped at query-only/all-prefill output bitwise assertion after both same-history nativeKV checks passed.',
        'test':'Compare point/batch predictor inputs and outputs, then original query-only vs all-prefill writes vs query-only writes computed with IDENTICAL fullbatch predictor execution. No native reference quantities enter the predictor.',
        'scope':'Posthoc numerical implementation audit; not a new semantic confirmation or altered main rollout.'})
    rows={r['sample_id']:r for r in gzread(BASE/'confirmation_material.json.gz')}
    model,tok=load('qwen4',out/'native_load');model.eval();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    live=Live(model);reports=[]
    try:
      with torch.inference_mode():
        for sid in read(out/'protocol.json')['samples']:
            row=rows[sid];ids=torch.tensor([inputs(row)],device=live.device);n=ids.shape[1]
            live.set_branch('native');live.reset(classifier(row));native=model.model(input_ids=ids,use_cache=True)
            single=live.features([n-1]);batch=live.features(list(range(n)))
            feature_difference={k:float((single[k].double()-batch[k][-1:].double()).abs().max()) for k in single}
            ps=live.predict(35,[n-1]);pb=live.predict(35,list(range(n)))[-1:]
            prediction={'max_abs_float32':float((ps-pb).abs().max()),
                'relative_MSE_float32':float((ps-pb).square().mean()/pb.square().mean().clamp_min(1e-20)),
                'changed_BF16_coordinates':int((ps.bfloat16()!=pb.bfloat16()).sum())}
            original_predict=live.predict;outputs=[];writes=[];caches=[]
            for mode in ('query_point_shape','all_prefill_shape','query_matched_full_shape'):
                live.set_branch('early_prediction_L35',mode!='all_prefill_shape');live.reset(classifier(row))
                if mode=='query_matched_full_shape':
                    def matched(b,positions):
                        whole=original_predict(b,list(range(len(live.current_h12))))
                        return whole[positions]
                    live.predict=matched
                result=model.model(input_ids=ids,use_cache=True)
                outputs.append(model.lm_head(result.last_hidden_state[0,-1]).float())
                writes.append(live.predicted_writeback[35]);caches.append(result.past_key_values)
                live.predict=original_predict
            comparisons=[]
            for index,mode in ((0,'point_vs_all'),(2,'matched_shape_query_vs_all')):
                x,y=outputs[index],outputs[1];xp,yp=x.log_softmax(-1),y.log_softmax(-1)
                comparisons.append({'mode':mode,'logits_bitwise_equal':torch.equal(x,y),
                    'changed_logits':int((x!=y).sum()),'max_logit_abs':float((x-y).abs().max()),
                    'logit_relative_MSE':float((x-y).square().mean()/y.square().mean().clamp_min(1e-20)),
                    'native_direction_KL_between_scopes':float((yp.exp()*(yp-xp)).sum()),
                    'argmax_agrees':int(x.argmax())==int(y.argmax()),
                    'changed_MLP_BF16_coordinates':int(np.count_nonzero(writes[index]!=writes[1])),
                    'all_KV_bitwise_equal':compare_caches(caches[index],caches[1])['all_bitwise_equal']})
            assert comparisons[1]['logits_bitwise_equal'] and comparisons[1]['all_KV_bitwise_equal']
            assert all(compare_caches(c,native.past_key_values)['all_bitwise_equal'] for c in caches)
            npz(out/f'{sid}.npz',point_prediction=ps.cpu().numpy(),batch_prediction=pb.cpu().numpy(),
                actual_MLP_writes=np.stack(writes),actual_fullvocab_logits=np.stack([x.cpu().numpy() for x in outputs]))
            reports.append({'sample_id':sid,'tokens':n,'all_feature_max_abs_differences':feature_difference,
                'prediction_arithmetic_difference':prediction,'comparisons':comparisons})
            print('LAW_SCOPE_PRECISION',sid,prediction,comparisons,flush=True)
            del native,result,ids,ps,pb,single,batch,outputs,writes,caches;live.reset(0);gc.collect();torch.cuda.empty_cache()
    finally:
        live.close();del live,model;gc.collect();torch.cuda.empty_cache()
    result={'timestamp':stamp(),'source':snapshot(Path(__file__)),'passed':True,'reports':reports,
        'finding':'Matching predictor execution shape makes query-only/all-prefill finalMLP logits and allKV bitwise identical in both prescribed diagnostic cases. Original main all-prefill rollout is unchanged.',
        'seconds':time.monotonic()-start}
    save(out/'result.json',result);ledger('scope_execution_shape_diagnosis',result['seconds']);guard()


def readout_precision():
    from scipy.special import logsumexp
    start=time.monotonic();out=BASE/'deployment/scope_precision';records=[]
    for path in sorted(out.glob('confirmation-*.npz')):
        with np.load(path) as z:logits=z['actual_fullvocab_logits'].astype(np.float64)
        lp=logits-logsumexp(logits,axis=1,keepdims=True)
        for i,mode in ((0,'point_vs_all'),(2,'matched_shape_query_vs_all')):
            records.append({'sample_id':path.stem,'mode':mode,
                'KL_float64':float(np.sum(np.exp(lp[1])*(lp[1]-lp[i]))),
                'actual_logits_sha256':identity(logits)['sha256']})
    save(out/'readout_precision.json',{'timestamp':stamp(),'source':snapshot(Path(__file__)),'records':records,
        'scope':'Reevaluate saved fullvocabulary logits using float64 logsumexp. Original signedfloat32 nearzero KL is retained, not interpreted as negative mathematical divergence.'})
    ledger('scope_saved_readout_precision',time.monotonic()-start);print('LAW_SCOPE_READOUT_PRECISION',records,flush=True)


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--readout-only',action='store_true');args=ap.parse_args()
    if args.readout_only:readout_precision()
    else:main()
