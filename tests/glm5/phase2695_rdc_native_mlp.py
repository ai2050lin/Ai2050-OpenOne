"""Full native-unit ledger: actual gate/up/down weights, no donor activation or parameter edits."""
import gc,time
from rdc_mechanism_common import *

OUT=CAMPAIGN/'a_native'
def main():
    import torch
    torch.backends.cuda.matmul.allow_tf32=False
    rows=read(OUT/'material.json');assert len(rows)==1024
    immutable(OUT/'native_protocol.json',{'source_sha':sha(Path(__file__)),'layers':[11,23,35],
        'cases':1024,'widths':{'hidden':2560,'mlp':9728},'checkpoint':'qwen3-4b BF16 actual weights; no parameter edits',
        'ledger':'beta[b,j,a]=sum_k Wdown[k,j]*reader[b,k,a]; all j,k supported by lazy source queries',
        'arithmetic':'Gate/up full FP32 CUDA products with TF32 disabled; independent FP64 checks first32 native positions. FP64 down-to-reader composite and observed BF16 remainder.',
        'rounding':'Stored native down and gate/up/a are observed BF16; ideal sums must retain explicit numerical residual.',
        'not_a_discovery_claim':'Algebraic decomposition and candidate condition maps; not unique native semantic units.'})
    results=[];catalog=[]
    for r in rows:catalog.append(dict(sample_id=r['sample_id'],origin_run=r['origin_run'],family=r['family'],language=r['language'],spans=r['spans'],tokens=r['tokens'],prompt_ids=r['prompt_ids']))
    immutable(OUT/'native_catalog.json',catalog)
    for layer in (11,23,35):
        start=time.monotonic();keys=('a','gate','up','mlp_x','down');values={k:[] for k in keys};bounds=[];pos=[];cursor=0
        for i,r in enumerate(rows):
            path=PREVIOUS/r['origin_run']/f'fields/{r["sample_id"]}.npz'
            with np.load(path) as z:
                positions=z['native_positions'].tolist();pos.append(positions)
                for key in keys:values[key].append(unbits(z[f'L{layer}_{key}']))
            bounds.append((cursor,cursor+len(positions)));cursor+=len(positions)
            if i%256==0:print('NATIVE_READ',layer,i,1024,flush=True)
        arrays={k:np.concatenate(v) for k,v in values.items()};del values
        wg=checkpoint(f'model.layers.{layer}.mlp.gate_proj.weight').float().numpy()
        wu=checkpoint(f'model.layers.{layer}.mlp.up_proj.weight').float().numpy()
        wd=checkpoint(f'model.layers.{layer}.mlp.down_proj.weight').float().numpy()
        with np.load(PREVIOUS/f's2pilot/coordinate_ledgers/word__family__H{layer+1}__A1_linear.npz') as z:reader=z['weights'].reshape(3,2560,8)
        beta=np.stack([wd.astype(np.float64).T@reader[b] for b in range(3)])
        npz(OUT/f'ledgers/L{layer}_native_coefficients.npz',beta=beta,reader=reader)
        project={};checks={}
        x=torch.from_numpy(arrays['mlp_x']).to('cuda')
        with torch.inference_mode():
            for name,w in [('gate',wg),('up',wu)]:
                weight=torch.from_numpy(w).to('cuda');pred=(x@weight.T).cpu().numpy();del weight
                delta=pred-arrays[name]
                fp64=arrays['mlp_x'][:32].astype(np.float64)@w.astype(np.float64).T
                checks[name]={'max_abs_observed_bf16_remainder':float(np.abs(delta).max()),
                    'mean_abs_observed_bf16_remainder':float(np.abs(delta).mean()),
                    'sum_abs_observed_bf16_remainder':float(np.abs(delta).sum(dtype=np.float64)),
                    'fp32_vs_fp64_first32_max':float(np.abs(pred[:32]-fp64).max())}
                del pred,delta,fp64
            g=torch.from_numpy(arrays['gate']).to('cuda',dtype=torch.bfloat16);u=torch.from_numpy(arrays['up']).to('cuda',dtype=torch.bfloat16)
            activation=(torch.nn.functional.silu(g)*u).float().cpu().numpy()
            checks['native_activation']={'compared_scalars':int(activation.size),'bitwise_equal_values':int(np.count_nonzero(activation==arrays['a'])),
                                        'max_abs_error':float(np.abs(activation-arrays['a']).max())}
            del g,u,activation,x
        # Aggregate only for statistics; all individual original native positions stay referenced in catalog.
        means=np.empty((1024,3,9728),dtype=np.float32);scores=np.empty((1024,3,8));reconstructed=np.empty_like(scores)
        for i,r in enumerate(rows):
            lo,hi=bounds[i]
            for b in range(3):
                wanted=r['spans']['u' if b==0 else 'v']['positions'] if b<2 else [len(r['prompt_ids'])-1]
                indices=[lo+pos[i].index(t) for t in wanted]
                means[i,b]=arrays['a'][indices].mean(0)
                scores[i,b]=arrays['down'][indices].mean(0).astype(np.float64)@reader[b]
                reconstructed[i,b]=means[i,b].astype(np.float64)@beta[b]
        remainder=scores-reconstructed
        family_means=np.stack([means[[r['family_index']==f for r in rows]].mean(0,dtype=np.float64) for f in range(8)])
        npz(OUT/f'ledgers/L{layer}_unit_conditions.npz',mean_native_a=means,observed_down_reader_score=scores,
            ideal_native_reader_score=reconstructed,rounding_remainder=remainder,family_mean_a=family_means,
            mean_signed_unit_contribution=family_means[:,:,:,None]*beta[None],
            mean_abs_unit_contribution=np.stack([np.abs(means[[r['family_index']==f for r in rows]]).mean(0,dtype=np.float64) for f in range(8)])[:,:,:,None]*np.abs(beta[None]))
        conditions=[]
        for origin in ('s1','s2pilot'):
            for language in ('en','zh'):
                for family in sorted({r['family'] for r in rows}):
                    ids=[i for i,r in enumerate(rows) if (r['origin_run'],r['language'],r['family'])==(origin,language,family)]
                    f=rows[ids[0]]['family_index'];contrib=means[ids].astype(np.float64)*beta[:, :,f][None]
                    conditions.append(dict(origin_run=origin,language=language,family=family,n=len(ids),
                        mean_observed_block_score=scores[ids,:,f].mean(0).tolist(),
                        positive_total=float(np.maximum(contrib,0).sum()),negative_total=float(np.minimum(contrib,0).sum()),
                        nonzero_unit_count=int(np.count_nonzero(np.any(contrib!=0,axis=(0,1))))))
        result={'layer':layer,'reader_checkpoint':layer+1,'cases':1024,'native_positions':cursor,'all_units':9728,
            'projection_checks':checks,'reader_rounding_remainder':{'max_abs':float(np.abs(remainder).max()),
                'mean_abs':float(np.abs(remainder).mean()),'sum_abs':float(np.abs(remainder).sum())},
            'conditions':conditions,'elapsed_seconds':time.monotonic()-start}
        results.append(result);save(OUT/'native_result.json',{'status':'partial','layers':results})
        events('a_native','native_layer_complete',layer=layer,cases=1024);print('NATIVE_LAYER_DONE',layer,result['elapsed_seconds'],flush=True)
        del arrays,wg,wu,wd,beta,means,reader,scores,reconstructed;gc.collect();torch.cuda.empty_cache()
    save(OUT/'native_result.json',{'timestamp':stamp(),'status':'native_analysis_complete','layers':results,
        'limits':['beta is a composition of real Wdown and fitted reader, not another native parameter.',
           'All-unit mean diagrams are navigation, not a causal semantic dictionary.',
           'FP32 CUDA gate/up reconstructions and ideal FP64 reader sums retain measured BF16 remainders.',
           'No full source-attention path was inferred from these partial historical captures; next-stage explicit capture required.']})
    announce('a_native',state='science_complete_client_pending',completed=1024,total=1024)
    print('NATIVE_DONE',flush=True)

if __name__=='__main__':main()
