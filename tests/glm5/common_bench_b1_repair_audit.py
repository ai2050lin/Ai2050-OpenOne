"""End-to-end identity, admission and result audit; no model fitting."""
import os
os.environ['MPLCONFIGDIR']=str(__import__('pathlib').Path(__file__).resolve().parent/'result/common_bench_b1_repair_v2/matplotlib-cache')
import sys,json,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import common_bench_b1_repair as R

def main():
    checks=[];start=time.monotonic();pre=R.checked()
    def check(name,ok,**details):checks.append({'name':name,'passed':bool(ok),**details})
    check('old_D_freeze_unchanged',R.sha(R.OLD/'staged-frozen.json')==pre['old_d_frozen_sha256'])
    oldpre=R.read(R.OLD/'staged-prereg.json')
    check('old_D_source_unchanged',R.sha(R.ROOT/'tests/glm5/common_bench_b1_stage_d.py')==oldpre['code_sha256'])
    for split in ['discovery','lockbox','lockbox_frame']:
        data=R.read(R.OUT/f'{split}-items.json');man=R.read(R.OUT/f'{split}-manifest.json');rows=data['rows']
        check(split+'/unique_complete_rows',len({x['id'] for x in rows})==len(rows)==(256 if split=='discovery' else 720))
        check(split+'/row_order',man['row_ids']==[x['id'] for x in rows])
        check(split+'/calibration',all(all(v.values()) for v in data['calibration']))
        seeds=[R.seed_fingerprint(x['id'],f'random{j}') for x in rows for j in range(3)]
        check(split+'/unique_full_id_random_streams',len(seeds)==len(set(seeds)),n=len(seeds))
        folder=R.OUT/f'{split}-native'
        for name,meta in man['files'].items():
            path=folder/name
            check(split+'/native/'+name,R.sha(path)==meta['sha256'] and path.stat().st_size==meta['bytes'])
        a=np.load(folder/'hidden.npy',mmap_mode='r');b=np.load(folder/'final_states.npy',mmap_mode='r')
        check(split+'/final_norm_capture_identity',np.array_equal(a[:,36,2],b[:,0]))
        del a,b
        report=R.read(R.OUT/f'{split}-report.json')
        check(split+'/norm_matching',all(x['valid_fraction']>=.99 for x in report['norm_checks'].values()))
        if not report['gate']['passed']:
            check(split+'/failed_behavior_not_promoted',all(v=='behavior_ineligible' for v in report['decisions'].values()) and all(not v['prediction_gate'] and v['status']=='behavior_ineligible' for v in report['path_predictions'].values()))
    check('no_original_lockbox_report_overwritten',not (R.OLD/'staged-lockbox-report.json').exists() and not (R.OLD/'staged-lockbox_frame-report.json').exists())
    fz=R.read(R.OUT/'path-frozen.json')
    check('path_selection_frozen_before_lockbox',fz['provenance']['created_at']<R.read(R.OUT/'lockbox-started.json')['provenance']['created_at'])
    gen=R.read(R.OUT/'generation.json')['rows'];cap=R.read(R.OUT/'capability.json')['rows']
    check('generation_complete',len(gen)==len({(x['split'],x['id'],x['arm']) for x in gen})==1024)
    check('capability_complete',len(cap)==len({x['id'] for x in cap})==512 and all(len(x['arms'])==4 for x in cap))
    restore=R.read(R.OUT/'restore-check.json');check('nonzero_edit_restoration',restore['all_passed'] and len(restore['checks'])==17)
    path=R.read(R.OUT/'path-account.json');check('residual_identity',max(x['identity_max_abs_error'] for x in path['rows'])<1e-4)
    check('no_extra_failed_panel_internal_replay',set(x['split'] for x in path['rows'])=={'lockbox'})
    for split,data in path['native'].items():
        for name,meta in data['files'].items():check('path/'+split+'/'+name,R.sha(R.OUT/f'{split}-path-account'/name)==meta['sha256'])
    ro=R.read(R.OUT/'rollout-output.json');check('output_readout_replay_calibration',ro['validation']['passed'])
    # Rehash original native arrays used for the retrospective state analysis.
    import common_bench_b1_stage_c as C
    for split in ['discovery','confirm']:
        oldman=R.read(R.OLD/f'stagec-native-{split}.json')
        for name,meta in oldman['files'].items():check('historical_native/'+split+'/'+name,R.sha(C.store(split)/name)==meta['sha256'])
    report=R.read(R.OUT/'report.json');check('no_cross_frame_confirmation',not report['both_panel_admissibility'] and report['cross_frame_mechanism_confirmation']=='not_admissible')
    compact={}
    goodrows=R.read(R.OUT/'lockbox-items.json')['rows'];byid={r['id']:r for r in R.read(R.OLD/'lockbox-materials.json')}
    _,mask=R.behavior_gate([byid[x['id']] for x in goodrows],goodrows);good={x['id'] for x,v in zip(goodrows,mask) if v}
    for arm in ['pooled','cross_language','cross_source','unembed']:
        compact[arm]={}
        for scope,pred in [('all',lambda x:True),('primary',lambda x:x['id'] in good)]:
            rr=[x for x in path['rows'] if x['arm']==arm and pred(x)]
            compact[arm][scope]={'n':len(rr),'actual_margin_change':float(np.mean([x['actual_fixed_token_margin_change'] for x in rr])),
                'direct_attribution':float(np.mean([x['direct_margin_attribution'] for x in rr])),
                'attention_attribution':float(np.mean([sum(x['attention_margin_attribution']) for x in rr])),
                'MLP_attribution':float(np.mean([sum(x['MLP_margin_attribution']) for x in rr])),
                'MLP_by_block':np.mean([x['MLP_margin_attribution'] for x in rr],axis=0).tolist(),
                'rounding_and_quadrature_residual_mean':float(np.mean([x['secant_and_BF16_residual'] for x in rr]))}
    footprint=sum(p.stat().st_size for p in R.OUT.rglob('*') if p.is_file())
    R.save('audit.json',{'checks':checks,'all_checks_passed':all(x['passed'] for x in checks),
        'path_accounting_summary':compact,'bytes_before_cleanup':footprint,'elapsed_seconds':time.monotonic()-start,
        'runtime_limitations':{'shared_GPU_telemetry':'Windows GPU Process Memory performance-counter object is unavailable. shared_gpu_bytes=null; not a measured zero. Dedicated GPU and host RAM were logged.',
            'admission_order':'D v2 collects baseline and edits family by family, so complete split behavior eligibility is determined after collection. The failed new-frame panel is barred from confirmation; subsequent internal replay excluded it. Future runner should perform a complete baseline pass first.',
            'reporter_fix':'Initial derived path report exposed a raw numerical threshold as prediction_gate even on failed behavior; before delivery it was separated into numerical_prediction_threshold_met and admission-aware prediction_gate. Frozen runner and thresholds unchanged.'},
        'code_sha256':{p.name:R.sha(p) for p in (R.ROOT/'tests/glm5').glob('common_bench_b1_*.py') if p.name.startswith(('common_bench_b1_repair','common_bench_b1_path_account','common_bench_b1_rollout','common_bench_b1_restore','common_bench_b1_state_analysis'))},
        'formal_evidence_promotion':0})
    assert all(x['passed'] for x in checks),[x for x in checks if not x['passed']]
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(13,4.5),layout='constrained')
    methods=['constant','decoded_A2','linear15'];colors=['#64748b','#df8120','#148477']
    for j,(method,color) in enumerate(zip(methods,colors)):
        ax[0].bar(np.arange(3)+(j-1)*.23,[ro['summary'][f'hs{s}/{method}']['primary']['kl'] for s in [14,27,32]],width=.23,color=color,label=method)
    ax[0].axhline(ro['summary']['mean_normalized_state']['primary']['kl'],color='black',ls='--',lw=1,label='mean-state baseline')
    ax[0].set(xticks=np.arange(3),xticklabels=['hs14','hs27','hs32'],ylabel='KL(true model || predicted output)',title='Autonomous surrogate: output fidelity\nExposed confirm, 3,604 behavior-correct items')
    ax[0].legend(fontsize=8)
    names=['pooled','unembed'];keys=['direct_attribution','attention_attribution','MLP_attribution','actual_margin_change']
    for j,(key,color) in enumerate(zip(keys,['#3b82b5','#b678b1','#148477','#273345'])):
        ax[1].bar(np.arange(2)+(j-1.5)*.18,[compact[a]['primary'][key] for a in names],width=.18,label=key.replace('_attribution','').replace('_margin_change',' total'),color=color)
    ax[1].axhline(0,color='black',lw=.5)
    ax[1].set(xticks=np.arange(2),xticklabels=['Class-centre shift','Output-weight shift'],ylabel='Fixed target-minus-original logit change',title='Residual accounting: direct and downstream\nEligible lockbox, 630 primary items; path attribution')
    ax[1].legend(fontsize=8)
    fig.savefig(R.OUT/'mechanism-summary.png',dpi=150);plt.close(fig)
    print(json.dumps({'checks':len(checks),'all_passed':True,'GiB':footprint/2**30,'path':compact}),flush=True)

if __name__=='__main__':main()
