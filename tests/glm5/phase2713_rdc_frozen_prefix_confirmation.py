"""Official held-out sentence units, frozen rules; no refitting on confirmation."""
from rdc_prefix_estimators import *
OUT=CAMPAIGN/'confirmation'


def main():
    src=CAMPAIGN/'shared_rules';frozen=read(src/'frozen_models.json')
    for rel,digest in frozen['files'].items():assert sha(src/rel)==digest,(rel,'frozen rule changed')
    assert read(CAMPAIGN/'qwen4_confirmation/status.json')['state']=='captured'
    rows=build_features('qwen4_confirmation');trainrows=read(src/'qwen4/rows.json');tr,va,te=splits(trainrows)
    with np.load(src/'qwen4/features.npz') as z:old={k:z[k] for k in z.files}
    with np.load(src/'qwen4_confirmation/features.npz') as z:new={k:z[k] for k in z.files}
    bank=KernelBank(new,[],read(src/'input_scales.json'));trainbank=KernelBank(old,tr,read(src/'input_scales.json'))
    temporal=KernelBank(new,[],read(src/'temporal_scales.json'),temporal=True);train_temporal=KernelBank(old,tr,read(src/'temporal_scales.json'),temporal=True)
    indices=np.arange(len(rows));reports=[]
    for path in sorted((src/'models').glob('*.npz')):
        mid=path.stem;is_temporal=mid.startswith('temporal_')
        name=mid.removeprefix('temporal_').removesuffix('_df128')
        gram=(temporal.gram(name,indices,tr,train_temporal) if is_temporal else bank.gram(name,indices,tr,trainbank))
        with np.load(path) as z:pred=(gram@z['alpha'])*z['target_scales']+z['means']
        layerreports={};arrays={};layers=('next_h12','next_h36') if is_temporal else ('h23','h24','h36')
        for i,k in enumerate(layers):
            report,a=errors(new[k],pred[:,i*2560:(i+1)*2560],old[k][tr],rows)
            layerreports[k]=report;arrays.update({k+'_'+j:v for j,v in a.items()})
        reports.append({'model':mid,'source_model_sha':sha(path),'layers':layerreports,'fitted_here':False})
        npz(OUT/f'predictions/{mid}.npz',prediction=pred.astype(np.float32),test=indices,**arrays)
        print('FROZEN_CONFIRMATION',mid,layerreports[layers[-1]]['mse'],flush=True)
    # Frozen train means / raw H12 baselines have the same information restrictions.
    for name in ('train_mean','copy_H12'):
        pred=np.concatenate([np.broadcast_to(old[k][tr].mean(0),new[k].shape) if name=='train_mean' else new['h12'] for k in ('h23','h24','h36')],1)
        layerreports={k:errors(new[k],pred[:,i*2560:(i+1)*2560],old[k][tr],rows)[0] for i,k in enumerate(('h23','h24','h36'))}
        reports.append({'model':name,'layers':layerreports,'fitted_here':False});npz(OUT/f'predictions/{name}.npz',prediction=pred.astype(np.float32),test=indices)
    # Candidate-level paired source-unit comparison; no individual token treated as an independent document.
    reportmap={r['model']:r for r in reports};comparisons=[]
    for a,b in [('early_linear','full_quadratic'),('early_linear','graph_interaction'),('graph_interaction','hash_interaction'),('graph_interaction_df128','hash_interaction_df128')]:
        aa=reportmap[a]['layers']['h36']['by_source_group'];bb=reportmap[b]['layers']['h36']['by_source_group']
        ids=sorted(set(aa)&set(bb));d=np.array([aa[k]['mse']-bb[k]['mse'] for k in ids]);rng=np.random.default_rng(2713)
        means=d[rng.integers(0,len(d),size=(2000,len(d)))].mean(1)
        comparisons.append({'a':a,'b':b,'source_groups':len(ids),'mean_mse_a_minus_b':float(d.mean()),
          'source_unit_bootstrap_CI95':np.quantile(means,[.025,.975]).tolist(),'fraction_a_better':float(np.mean(d<0))})
    save(OUT/'rows.json',rows)
    save(OUT/'result.json',{'phase':2713,'timestamp':stamp(),'units':128,'anchors':256,'reports':reports,'paired_comparisons':comparisons,
      'selected_before_confirmation':frozen['chosen_by_validation'],'selected_temporal_before_confirmation':frozen['temporal_chosen_by_validation'],
      'frozen_models_sha':sha(src/'frozen_models.json'),'fitting_in_confirmation':False,
      'new_mathematical_theorem':False,'mechanism_closed':False,
      'limits':['Independent held-out source units relative to extraction fit, not verified absent from pretrained LLM data.',
        'English source document IDs disjoint; Chinese document membership unknown, normalized-content grouping only.',
        'Natural Chinese Wikipedia and English web genres remain confounded; no unseen language or unrestricted task claim.']})
    status('confirmation',state='complete',units=128,anchors=256);guard();print('FROZEN_CONFIRMATION_COMPLETE',flush=True)


if __name__=='__main__':main()
