"""Independent retained-array checks and bounded same-goal continuation handoff."""
from rdc_relation_common import *


def main():
    guard(160*1024);checks={};tr=[r for r in rows() if r['split']=='train'];xx=[];yy=[]
    for r in tr:
        z=load_field(r);xx.append(unbits(z['h12'][0]));yy.append(unbits(z['h23'][0]))
    x=np.stack(xx).astype(float);y=np.stack(yy).astype(float);mx=x.mean(0);my=y.mean(0)
    a=(np.mean((x-mx)*(y-my),0)/(1.01*np.mean((x-mx)**2,0)+1e-8)).astype(np.float32);b=(my-a*mx).astype(np.float32)
    with np.load(BASE/'boundary_compilation/first_source_affine.npz') as z:
        for k,v in {'a':a,'b':b,'training_mean_h12':mx,'training_mean_h23':my}.items():assert np.array_equal(v,z[k])
    common=(my-mx).astype(np.float32);ee=[];h24=[]
    for r in rows(True):
        z=load_field(r,True);p=unbits(z['h12'][0])+common;target=unbits(z['h23'][0]);ee.append((p.astype(float)-target)**2)
        with np.load(BASE/f'boundary_compilation/fields/{r["sample_id"]}.npz') as z2:
            assert z2['h24'].shape==(2,2560);h24.append(np.mean((unbits(z2['h24']).astype(float)-unbits(z['h24'][[0,3]]))**2))
    with np.load(BASE/'boundary_compilation/constant_update_control.npz') as z:
        assert np.array_equal(common,z['training_common_update']) and np.array_equal(np.mean(ee,0),z['all_coordinate_fresh_MSE'])
    br=read(BASE/'boundary_compilation/result.json');assert abs(np.mean(h24)-br['native']['first_source_affine_else_frozen_quadratic']['H24_MSE'])<1e-12
    checks['all320_training_boundary_coefficients_and128_constant_predictions_recomputed']=True
    checks['all128_retained_boundary_H24_arrays_reproduce_summary']=True
    geometry={}
    for scope in ('current','temporal','self_generation'):
        rr=read(BASE/f'output_geometry/{scope}_rows.json');assert len(rr)==(1024 if scope=='self_generation' else 256)
        kl=np.array([r['KL_same_FP32'] for r in rr]);integ=np.array([r['exact_curvature_integral'] for r in rr]);assert np.max(np.abs(integ-kl))<2e-4
        if scope!='self_generation':
            with np.load(BASE/f'output_geometry/{scope}_all_coordinate_attributions.npz') as z:
                v=z['attribution'];assert v.shape==(256,2560);err=np.max(np.abs(v.sum(1,dtype=float)-kl));assert err<1e-5
        else:
            with np.load(BASE/'output_geometry/generation_all_coordinate_attributions.npz') as z:
                assert z['signed_sum'].shape==(16,2560) and np.array_equal(z['counts'],np.full(16,64))
                perstep=np.array([sum(r['KL_same_FP32'] for r in rr if r['step']==s) for s in range(16)])
                err=np.max(np.abs(z['signed_sum'].sum(1)-perstep));assert err<1e-3
        with np.load(BASE/f'output_geometry/{scope}_coordinate_profiles.npz') as z:assert abs(z['signed'].sum()-kl.mean())<1e-5
        geometry[scope]={'observations':len(rr),'retained_attribution_sum_max_error':float(err),'integral_max_error':float(np.max(np.abs(integ-kl)))}
    checks['complete_probability_accounting']=geometry
    sr=read(BASE/'surrogate_stability/result.json');assert sr['candidate_cycles']==10
    for c in sr['cycles']:
        with np.load(BASE/f'surrogate_stability/cycles/{c["cycle_id"]:03d}.npz') as z:
            spec=z['spectrum_real']+1j*z['spectrum_imaginary'];assert spec.shape==(2560,) and np.array_equal(spec[640:],np.zeros(1920))
            assert abs(np.max(np.abs(spec))-c['spectral_radius'])<1e-12 and z['fixed_cycle_states'].shape==(1,2560)
    trace=read(BASE/'surrogate_stability/trajectory_result.json');assert trace['sources_with_monotone_suffix_distance']==39
    checks['all10_surrogate_spectra_and39_observed_suffix_trajectories']=True
    recon=read(BASE/'source_normalization/recomputation_audit.json');assert recon['passed'] and recon['all_fitted_arrays_hash_equal'] and len(recon['fits'])==8
    checks['source_normalization_recomputation_report']=recon
    client=read(BASE/'verification/client_extended_api.json');assert client['passed'] and client['extended']
    checks['extended_client_checks']=client['check_count']
    figures=read(BASE/'figures/index.json');assert len(figures)==7 and all((BASE/'figures'/r['file']).exists() for r in figures)
    sources=[snapshot(p) for p in sorted((ROOT/'tests/glm5').glob('phase2718*.py'))]
    sources += [snapshot(ROOT/p) for p in ('server/rdc_relation_service.py','frontend/src/components/app/RdcRelationStudy.jsx','tests/glm5/phase2717_rdc_relation_client_checks.py')]
    save(BASE/'verification/extension_integrity.json',{'timestamp':stamp(),'passed':True,'checks':checks,'sources':sources,'figures':figures,
      'validation':{'component_ESLint':'passed22:32; captured actual tool result','Vite_build':'passed22:32,2839 modules; existing large-chunk warning retained','browser_QA':'not performed or claimed','science_figures':'seven PNGs inspected as static scientific figures; first five in2717, last two22:28'},
      'limits':'Independent arithmetic and stored-array checks, not a GPU rerun of every diagnostic. Scale models and normalized-source fits have separate exact coefficient reconstruction reports. Full NPZ hash/finite audit follows after MEMO append.'})
    current=usage();free=shutil.disk_usage(ROOT).free
    # Lower bound covers ONLY two retained all-token layers, not all proposed work.
    minimum_field_bytes=512*38*2*2560*2
    decision={'timestamp':stamp(),'completed_phases':[2715,2716,2717,2718],'same_scientific_goal':True,'automatic_extension_executed':2718,
      'status':'Bounded integrated delivery complete. Next independent broad stage planned, not executed; resource-limited continuation boundary, not AGI completion.',
      'resource_snapshot':{'current_run_bytes':current,'registered_ceiling_bytes':CEILING,'remaining_run_bytes':CEILING-current,'physical_free_bytes':free,'physical_floor_bytes':FLOOR,'physical_headroom_bytes':free-FLOOR,
        'engineering_limits_not_user_supplied_numeric_budget':True,'next_independent_two_layer_raw_field_lower_bound_bytes':minimum_field_bytes,'lower_bound_formula':'512 sources * planning38 tokens *2 layers *2560 coords *2 BF16 bytes; not measured future corpus size'},
      'reason':'All planned2718 reuse diagnostics completed. New independent material is needed to upgrade posthoc findings; merely retesting reused sources would not establish confirmation. Proposed broad phase exceeds both remaining run allocation and physical8GiB headroom. No raising ceiling, no deleting fields used by client.',
      'next_phase':{'number':2719,'status':'not executed','question':'Which position-conditioned full-coordinate updates survive independent natural material and jointly improve complete-vocabulary prediction and history-dependent continuation?',
        'work_packages':['Freeze independently sourced bilingual natural documents, controlling first-token identity, position, genre and reused skeletons; expand only after reliable pilot.',
          'Capture embedding and all-layer first-position/ordinary-position full-coordinate trajectories in blocks; compare identity, common update and affine boundary baselines, then freeze and confirm on unused sources.',
          'Compare raw and RMS source organization with equal-capacity probability-aware/full-coordinate objectives; source-position controls and weak noninitial relation replication.',
          'Test explicit available history plus state/input update on same native prefixes and self prefixes separately; longer generation, fixed-point guards and readout errors; no actual hidden refresh in autonomous branch.',
          'Follow replicated coordinate interactions through original Q/K/MLP scalar paths before cross-model sequence, retaining each model own indices and numerical floor.'],
        'entry_criteria':'New destination or deliberate resource allocation with >=256MiB additional result capacity above physical reserve; recompute pilot estimate before executing. This is a conservative planning target, not a measured future cost.'},
      'retention':'All current H12/H23 and selected later full-coordinate fields retained and exposed in local client or exact reconstruction recipes; no original HiddenState, model weights or unrelated user data deleted.',
      'resume_files':['plan.json','frozen.json','continuation_decision.json','verification/final_integrity.json','verification/extension_integrity.json'],
      'code_entrypoints':['tests/glm5/phase2717_rdc_relation_integrity.py --final','tests/glm5/phase2717_rdc_relation_client_checks.py --extended','tests/glm5/phase2718_rdc_source_normalization.py --verify']}
    assert minimum_field_bytes>CEILING-current and minimum_field_bytes>free-FLOOR
    save(BASE/'continuation_decision.json',decision);print('EXTENSION_DELIVERY_AUDIT_PASS',checks,decision['resource_snapshot'],flush=True)


if __name__=='__main__':main()
