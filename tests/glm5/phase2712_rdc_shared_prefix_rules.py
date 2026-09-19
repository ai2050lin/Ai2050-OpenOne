"""One shared library, fixed algorithms, joint layer targets and next-input-available updates."""
import argparse
from rdc_prefix_estimators import *
OUT=CAMPAIGN/'shared_rules'


def prepare():
    immutable(OUT/'protocol.json',{'phase':2712,'source_sha':sha(Path(__file__)),
      'estimator_sha':sha(ROOT/'tests/glm5/rdc_prefix_estimators.py'),'material_sha':sha(CAMPAIGN/'material_stratified.json'),
      'selection':'All512 units; both quantile anchors stay with their sentence/source group. 640 train/192 validation/192 test anchor states. No correctness filter.',
      'inputs':'Current H12 full2560, all-seen-token meanH12 full2560, actual currentH0 full2560; visible-prefix graph descriptor. No future state, emitted future token, full-sentence UD tag or answer label.',
      'descriptor':'21 cue/count/open-state features, all16x16 ordered adjacent cue-type pairs, all16x16 same-clause type co-presence counts,16 last-visible cue positions. 549 entries, not a full semantic parser or latent-coordinate compression.',
      'kernels':list(KERNELS),'controls':'Current H12; embedding+history; full linear/quadratic; graph only/additive/product; same-dimensional deterministic random-prefix-hash product. Also graph/hash fixed effective-df128. No per-template functions.',
      'joint_targets':'H23,H24,H36 all native2560. One ridge per shared multitask rule, chosen on equal-training-RMS-normalized target blocks, full coordinates. All residuals preserved.',
      'temporal':'After seeing the newly appended token, previousH36 + previous prefix meanH12 + newH0 + newly visible prefix graph predict newH12 and newH36. Neither the next emitted token nor actual newH12 is input. This conditional teacher-forced update is not a complete Markov state or autonomous generator.',
      'temporal_kernels':['early_linear','embedding_history','full_linear','graph_interaction','hash_interaction'],
      'evaluation':'All test native coordinates and training-energy quartiles, sentence grouped errors; full-vocabulary current-output assessment through actual finalnorm/unembedding in a separate serial GPU audit.',
      'native_constraint':'Predicted H23 may be compiled with real L23 inputnorm/QKV/head norms and observed past layer23 K/V. This adds deeper historical information; do not equate it with H12-only direct prediction.',
      'freeze':'Serialize every rule, training inputs, scales and hashes before official heldout capture. Confirmation will not refit.',
      'limits':['Graph descriptors are incomplete lexical/order features, not human-level meaning/role binding.','History mean is a baseline candidate and is not a claim that all history can be losslessly compressed.',
        'Native coordinates and outputs stay full; statistical models may fail on rare words, grammar or new lengths.','Test errors are conditional forecasts, not proof of causality or task competence.']})


def main():
    prepare();start=time.monotonic();rows=build_features();tr,va,te=splits(rows)
    assert tuple(map(len,(tr,va,te)))==(640,192,192)
    with np.load(OUT/'qwen4/features.npz') as z:data={k:z[k] for k in z.files}
    bank=KernelBank(data,tr);save(OUT/'input_scales.json',bank.serial_scales())
    y=np.concatenate([data[f'h{l}'] for l in TARGET_LAYERS],1);blocks=[(i*2560,(i+1)*2560) for i in range(3)]
    reports=[];grams={};testrows=[rows[i] for i in te]
    for name in KERNELS:
        gram=bank.gram(name,np.arange(len(rows)),np.arange(len(rows)));grams[name]=gram.astype(np.float32)
        choices=[(name,None)]+([(name+'_df128',128)] if name in ('graph_interaction','hash_interaction') else [])
        for mid,df in choices:
            pred,val,meta=fit(gram,tr,va,te,y,blocks,OUT/f'models/{mid}.npz',df)
            layer_reports={};arrays={}
            for i,l in enumerate(TARGET_LAYERS):
                report,a=errors(y[te,i*2560:(i+1)*2560],pred[:,i*2560:(i+1)*2560],y[tr,i*2560:(i+1)*2560],testrows)
                layer_reports[f'H{l}']=report;arrays.update({f'H{l}_{k}':v for k,v in a.items()})
            reports.append({'model':mid,'kernel':name,**meta,'layers':layer_reports})
            npz(OUT/f'predictions/{mid}.npz',prediction=pred,validation_prediction=val,test=te,validation=va,**arrays)
            print('SHARED_RULE',mid,layer_reports['H36']['mse'],meta['effective_df'],flush=True)
            assert time.monotonic()-start<7200;guard()
    npz(OUT/'input_grams.npz',**grams,train=tr,validation=va,test=te)
    # Pure copying and train-mean baselines expose large common-background effects.
    for name,pred in [('copy_H12',np.tile(data['h12'][te],(1,3))),('train_mean',np.broadcast_to(y[tr].mean(0),(len(te),y.shape[1])))]:
        layer_reports={}
        for i,l in enumerate(TARGET_LAYERS):layer_reports[f'H{l}']=errors(y[te,i*2560:(i+1)*2560],pred[:,i*2560:(i+1)*2560],y[tr,i*2560:(i+1)*2560],testrows)[0]
        reports.append({'model':name,'layers':layer_reports,'baseline':True});npz(OUT/f'predictions/{name}.npz',prediction=pred.astype(np.float32),test=te)
    temporal=KernelBank(data,tr,temporal=True);save(OUT/'temporal_scales.json',temporal.serial_scales())
    ty=np.concatenate([data['next_h12'],data['next_h36']],1);treports=[]
    for name in ('early_linear','embedding_history','full_linear','graph_interaction','hash_interaction'):
        gram=temporal.gram(name,np.arange(len(rows)),np.arange(len(rows)))
        pred,val,meta=fit(gram,tr,va,te,ty,[(0,2560),(2560,5120)],OUT/f'models/temporal_{name}.npz')
        report,a=errors(ty[te,2560:],pred[:,2560:],ty[tr,2560:],testrows)
        treports.append({'model':'temporal_'+name,'kernel':name,**meta,'new_H36':report})
        npz(OUT/f'predictions/temporal_{name}.npz',prediction=pred,validation_prediction=val,test=te,validation=va,**a)
        print('TEMPORAL_RULE',name,report['mse'],flush=True)
    best=min((r for r in reports if not r.get('baseline')),key=lambda r:r['normalized_validation_mse'])['model']
    best_temporal=min(treports,key=lambda r:r['normalized_validation_mse'])['model']
    result={'phase':2712,'timestamp':stamp(),'units':512,'anchors':1024,'train':640,'validation':192,'test':192,
      'reports':reports,'temporal_reports':treports,'selected_by_validation':best,'selected_temporal_by_validation':best_temporal,
      'model_selection_criterion':'joint full-coordinate training-RMS-normalized target error; not test scores or future labels',
      'elapsed_seconds':time.monotonic()-start,'full_vocabulary_audit':'pending separate native readout audit',
      'new_mathematical_theorem':False,'mechanism_closed':False}
    save(OUT/'result.json',result)
    frozen={str(p.relative_to(OUT)):sha(p) for p in (OUT/'models').glob('*.npz')}
    for rel in ('input_scales.json','temporal_scales.json','qwen4/features.npz','qwen4/rows.json','protocol.json','result.json'):frozen[rel]=sha(OUT/rel)
    immutable(OUT/'frozen_models.json',{'timestamp':stamp(),'files':frozen,'chosen_by_validation':best,
      'temporal_chosen_by_validation':best_temporal,'confirmation_fitting_allowed':False,
      'scope':'Frozen shared-kernel rules and their inputs. Later native probability evaluation does not alter these fitted rules.'})
    status('shared_rules',state='kernel_rules_frozen',models=len(frozen),anchors=len(rows));print('SHARED_RULES_FROZEN',best,best_temporal,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');a=p.parse_args();prepare() if a.prepare else main()
