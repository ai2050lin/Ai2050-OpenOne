"""Frozen finite parameter tests and calibrated, actually matched BF16 deltas."""
import gc
from rdc_update_common import *

def actual_delta(original,unit,scale):
    import torch
    return {k:(original[k]-scale*unit[k]).to(torch.bfloat16).float()-original[k] for k in original}

def calibrate(original,unit,target,tolerance=.005):
    from rdc_law_native import parameter_norm
    lo=0.;hi=target
    for _ in range(16):
        delta=actual_delta(original,unit,hi);norm=float(parameter_norm(delta))
        if norm>=target:break
        hi*=2
    else:raise AssertionError('BF16 norm bracketing failed')
    best=(abs(norm-target),hi,norm,delta)
    for _ in range(30):
        mid=(lo+hi)/2;delta=actual_delta(original,unit,mid);norm=float(parameter_norm(delta))
        if abs(norm-target)<best[0]:best=(abs(norm-target),mid,norm,delta)
        if norm<target:lo=mid
        else:hi=mid
        if best[0]/target<tolerance*.1:break
    assert best[0]/target<tolerance,(target,best[:3])
    return best[3],{'target_actual_BF16_norm':target,'pre_round_scale':best[1],'actual_BF16_norm':best[2],
      'relative_norm_error':best[0]/target,'tolerance':tolerance,
      'rule':'Monotone bisection uses only original BF16 weights and fixed direction. No loss, labels, validation or test outcomes enter calibration.'}

def main():
    import torch
    from transformers import AutoTokenizer
    from rdc_law_native import Tail,parameter_norm
    from phase2735_rdc_binding_decomposition import collect_parts,describe
    from phase2737_rdc_update_directions import material_arrays,PARTS
    out=BASE/'learning';start=time.monotonic()
    if (out/'finite_result.json').exists():return
    frozen=read(out/'frozen.json');protocol=read(out/'protocol.json');rows=gzread(BASE/'program_material.json.gz')
    assert sha(BASE/'program_material.json.gz')==frozen['material_sha256'] and sha(out/'frozen_forecast.npz')==frozen['forecast_sha256']
    for name,digest in frozen['direction_sha256'].items():assert sha(out/'directions'/name)==digest
    guard(600*1024**2);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    tok=AutoTokenizer.from_pretrained(ROOT/'models/hf/qwen3-4b',local_files_only=True);digits=[tok(str(i),add_special_tokens=False)['input_ids'][0] for i in range(1,9)]
    x,r,tt,_=material_arrays(rows);tail=Tail();x=torch.tensor(x,device='cuda');r=torch.tensor(r,device='cuda');targets=torch.tensor(tt,device='cuda')
    original={k:v.detach().clone() for k,v in tail.w.items()}
    with np.load(out/'initial_scores.npz') as z:baseline={k:z[k] for k in z.files}
    with np.load(out/'oracle_derivatives.npz') as z:oracle=z['derivatives']
    with np.load(out/'frozen_forecast.npz') as z:forecast=z['derivatives']
    _,replay=collect_parts(tail,x,r,targets,digits,False)
    assert all(np.array_equal(replay[k],baseline[k]) for k in replay),'Same-shape preupdate score replay changed'
    updates=[];deployments=[]
    for j,name in enumerate(frozen['direction_order']):
        with np.load(out/'directions'/f'{name}.npz') as z:unit={k:torch.tensor(z[k],device='cuda') for k in ('g','u','d')}
        steps=protocol['step_norms']+([-.02] if name=='content_format_constrained_1e-6' else [])
        for step in steps:
            with torch.no_grad():
                for k,v in tail.w.items():v.copy_(original[k]);v.add_(unit[k],alpha=-step)
            _,stats=collect_parts(tail,x,r,targets,digits,False);delta=np.stack([stats[p+'_loss']-baseline[p+'_loss'] for p in PARTS],-1)
            label=f'{name}_{step}'
            npz(out/'finite_updates'/f'{label}.npz',**stats,loss_delta=delta,oracle_taylor=-step*oracle[:,j],H12_forecast=-step*forecast[:,j])
            diagnostics=[]
            for split in ('validation','test','mixed_holdout'):
              for rep in ('en','zh','python','en_reordered'):
                ix=[i for i,r0 in enumerate(rows) if r0['split']==split and r0['representation']==rep]
                for part,p in enumerate(PARTS):
                    actual=delta[ix,part];taylor=-step*oracle[ix,j,part];pred=-step*forecast[ix,j,part]
                    diagnostics.append({'split':split,'representation':rep,'part':p,'rows':len(ix),'actual_delta':float(actual.mean()),
                      'oracle_taylor_delta':float(taylor.mean()),'H12_forecast_delta':float(pred.mean()),
                      'oracle_mean_absolute_error':float(np.mean(abs(actual-taylor))),'forecast_mean_absolute_error':float(np.mean(abs(actual-pred))),
                      'actual_cluster':clustered(actual,[rows[i]['source_group'] for i in ix]),
                      'forecast_cluster_error':clustered(pred-actual,[rows[i]['source_group'] for i in ix])})
            updates.append({'direction':name,'FP32_parameter_step':step,'reports':describe(rows,stats,baseline),'forecast_diagnostics':diagnostics})
            print('FINITE_PARAMETER_UPDATE',label,flush=True)
        # Exactly the same declared target0.02 actual norm for every deployed
        # direction, including random; differs from previous pre-round matching.
        deployed_name=name
        if name=='rademacher':
            assert (out/'norm_recovery/failed_attempt.json').exists()
            generator=torch.Generator(device='cuda').manual_seed(273700)
            unit={k:torch.randn(v.shape,device='cuda',generator=generator) for k,v in original.items()}
            norm=parameter_norm(unit);unit={k:v/norm for k,v in unit.items()}
            deployed_name='gaussian_native_control'
            npz(out/'native_control_gaussian_unit.npz',**{k:v.cpu().numpy() for k,v in unit.items()})
        realized,report=calibrate(original,unit,.02,protocol['native_norm_relative_tolerance'])
        for k,v in realized.items():report[k]={'actual_norm':float(v.norm()),'changed_scalar_fraction':float((v!=0).float().mean())}
        report['direction']=deployed_name;report['FP32_finite_control']=name
        report['original_parameter_identity']={k:identity(v.cpu().numpy()) for k,v in original.items()}
        npz(out/'native_deltas'/f'{deployed_name}.npz',**{k:v.cpu().numpy() for k,v in realized.items()})
        report['delta_sha256']=sha(out/'native_deltas'/f'{deployed_name}.npz');deployments.append(report);del unit,realized
        guard(50*1024**2)
    with torch.no_grad():
        for k,v in tail.w.items():v.copy_(original[k])
    _,after=collect_parts(tail,x,r,targets,digits,False);assert all(np.array_equal(after[k],baseline[k]) for k in after)
    preserved=read(out/'norm_recovery/failed_attempt.json')['preserved_artifacts']
    assert all(sha(BASE/p)==digest for p,digest in preserved.items()),'Prior finite tests changed during norm-only recovery'
    result={'timestamp':stamp(),'source':snapshot(__file__),'frozen_sha256':sha(out/'frozen.json'),'preupdate_replay_exact':True,
      'post_reset_replay_exact':True,'updates':updates,'matched_native_deployments':deployments,'seconds':time.monotonic()-start,
      'preserved_failed_artifact_hashes_unchanged':len(preserved),'norm_recovery':'norm_recovery/failed_attempt.json; FP32 random is Rademacher, native matched random is Gaussian seed273700.',
      'scope':'Finite FP32 local parameter effects, with all upstream inputs frozen at native values. BF16 calibrated deltas are only prepared here; natural full-model own-history deployment is a separate experiment.'}
    save(out/'finite_result.json',result);ledger('finite_parameter_tests_and_BF16_norm_calibration',result['seconds']);print('FINITE_UPDATES_DONE',result['seconds'],flush=True)
    del tail,original;gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':
    begin=time.monotonic()
    try:main()
    except Exception as exc:failure(BASE/'learning/finite_failure',begin,exc);raise
