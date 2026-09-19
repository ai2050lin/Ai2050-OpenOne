"""A simple shared-layer-update control prevents over-crediting boundary affine complexity."""
from rdc_relation_common import *


def main():
    out=BASE/'boundary_compilation';guard(180*1024)
    with np.load(out/'first_source_affine.npz') as z:a=z['a'];b=z['b'];common=(z['training_mean_h23']-z['training_mean_h12']).astype(np.float32)
    records=[];errors=[];precision=[];firsttrain={r['prompt_ids'][0] for r in rows() if r['split']=='train'}
    for r in rows(True):
        z=load_field(r,True);x=unbits(z['h12'][0]);y=unbits(z['h23'][0]);p=x+common;ap=x*a+b;hi=x.astype(float)*a+b;ee=(p.astype(float)-y)**2;errors.append(ee);precision.append(np.max(np.abs(ap-hi)))
        records.append({'sample_id':r['sample_id'],'first_token_seen':r['prompt_ids'][0] in firsttrain,'common_increment_MSE':float(ee.mean()),'affine_FP32_MSE':float(np.mean((ap.astype(float)-y)**2))})
    with np.load(BASE/'energy_audit/first_boundary_coordinate_errors.npz') as z:groups=z['training_energy_quartile'];energy=z['fresh_target_energy']
    mse=np.mean(errors,0);npz(out/'constant_update_control.npz',training_common_update=common,all_coordinate_fresh_MSE=mse)
    report={'timestamp':stamp(),'source':snapshot(Path(__file__)),'scope':'Posthoc simple baseline for the special first position only. Shared average layer update fitted only on320 training first states; not semantic difference transport and not a new native compilation experiment.',
      'formula':'h23_first = h12_first + mean_training(h23_first-h12_first); all2560 coordinates. Compare the already fitted per-coordinate affine complexity.',
      'fresh_sources':len(records),'common_increment_MSE':float(mse.mean()),'affine_FP32_MSE':float(np.mean([r['affine_FP32_MSE'] for r in records])),
      'by_first_token_seen':{str(v):{'sources':len(rr),'common_increment_MSE':float(np.mean([r['common_increment_MSE'] for r in rr]))} for v in (False,True) if (rr:=[r for r in records if r['first_token_seen']==v])},
      'energy_quartiles':[{'quartile':q,'coordinates':int((groups==q).sum()),'MSE':float(mse[groups==q].mean()),'relative_to_target_energy':float(mse[groups==q].sum()/energy[groups==q].sum())} for q in range(4)],
      'affine_FP32_vs_high_precision_coordinate_reconstruction_max_abs_difference':float(max(precision)),
      'limits':'Constant-update success would mean the affine gain is largely a shared numerical layer increment, not evidence of semantic coordinate gears. It does not test origin-layer emergence, other positions or other models.'}
    save(out/'constant_update_result.json',report);save(out/'constant_update_source_rows.json',records);print('CONSTANT_BOUNDARY_CONTROL',report,flush=True)


if __name__=='__main__':main()
