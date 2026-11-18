"""Independent artifact, admission-order and implementation identity checks."""
import argparse,json,sys,time
from pathlib import Path
from datetime import datetime
sys.path.insert(0,str(Path(__file__).parent))
import source_binding_design as D

def environment():
    import platform,torch,transformers,numpy,scipy
    from transformers.models.qwen3 import modeling_qwen3
    from transformers.models.glm import modeling_glm
    from transformers.integrations import sdpa_attention
    modules=[modeling_qwen3,modeling_glm,sdpa_attention]
    D.save('runtime-environment.json',{'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,
        'transformers':transformers.__version__,'numpy':numpy.__version__,'scipy':scipy.__version__,
        'CUDA_build':torch.version.cuda,'library_files':{m.__file__:D.sha(m.__file__) for m in modules},
        'recording_boundary':'during discovery collection recovery, before any candidate fit or heldout run'})

def audit():
    import numpy as np
    from source_binding_report import text_baselines
    from source_binding_analysis import verify_code
    verify_code();checks=[];start=time.monotonic()
    def check(label,value):
        checks.append({'check':label,'passed':bool(value)})
        assert value,label
    addendum=D.read(D.OUT/'conditional-update-addendum.json')
    check('conditional update code identity',addendum['code_sha256']==D.sha(D.ROOT/'tests/glm5/source_binding_update.py'))
    for name in ['native-v2-code-freeze.json','causal-admission-v2.json','runtime-environment.json']:
        p=D.OUT/name
        if p.exists():
            x=D.read(p)
            for f,h in x.get('code_sha256',x.get('library_files',{})).items():check(f'{name}:{f}',D.sha(D.ROOT/f)==h)
    plan=D.read(D.OUT/'plan.json')
    for model in D.MODELS:
        D.checked(model);root=D.OUT/model;mp=D.OUT/f'{model}-materials.json'
        check(f'{model}:materials',D.sha(mp)==plan['materials'][model]);mat=D.read(mp)['splits'];names={}
        for split,rows in mat.items():
            names[split]={name for r in rows for name in r['names']}
            check(f'{model}/{split}:unique_ids',len({r['id'] for r in rows})==len(rows))
            check(f'{model}/{split}:factor_product',len(rows)==8*D.COUNTS[split])
            check(f'{model}/{split}:all_frozen_frames',{r['frame'] for r in rows}==set(range(4,8) if split=='lockbox_frame' else range(4)))
            check(f'{model}/{split}:parser',text_baselines(rows)['text_parser']==1.)
            for r in rows:
                check_ok=len(set(r['choice_ids']))==4 and len(r['positions'])==5 and r['positions'][-1]==len(r['tokens'])-1
                assert check_ok and all(max(v['span'])<r['positions'][-1] for v in r['records'].values())
            check(f'{model}/{split}:visible_and_token_aligned',True)
            gp=root/f'{split}-gate.json';manpath=root/f'{split}-native-manifest.json'
            if gp.exists():
                g=D.read(gp);b=D.read(root/f'{split}-behavior.json')
                check(f'{model}/{split}:gate_recompute',D.behavior(rows,b['rows'])['passed']==g['passed'])
                check(f'{model}/{split}:gate_hash',D.sha(root/f'{split}-behavior.json')==g['behavior_sha256'])
                if g['passed']:D.admission(model,split)
                else:check(f'{model}/{split}:failed_gate_no_internal',not manpath.exists() and not(root/f'{split}-native').exists())
            if manpath.exists():
                m=D.read(manpath);s=D.read(root/f'{split}-capture-started.json');g=D.read(gp)
                check(f'{model}/{split}:behavior_precedes_internals',g['provenance']['created_at']<s['provenance']['created_at'])
                check(f'{model}/{split}:complete_native_rows',m['row_ids']==[r['id'] for r in rows])
                check(f'{model}/{split}:baseline_replay',m['base_replay']['passed'])
                check(f'{model}/{split}:all41_native_boundaries',np.load(root/f'{split}-native/hidden.npy',mmap_mode='r').shape[1:3]==(41,5))
                h0=np.load(root/f'{split}-native/hidden.npy',mmap_mode='r')[:,0,-1]
                check(f'{model}/{split}:shared_final_token_embedding_baseline',bool(np.all(h0==h0[:1])))
                qkv=np.load(root/f'{split}-native/qkv_error.npy',mmap_mode='r');check(f'{model}/{split}:QKV_replay',float(qkv.max())<.02)
                check(f'{model}/{split}:SwiGLU_Down_replay',all(x['SwiGLU_bitwise_equal'] and x['MLP_down_relative_error']<.02 for x in m['checks']))
                for filename,info in m['files'].items():
                    p=root/f'{split}-native'/filename;check(f'{model}/{split}/{filename}:bytes',p.stat().st_size==info['bytes']);check(f'{model}/{split}/{filename}:sha256',D.sha(p)==info['sha256'])
                    D.log('artifact-audit',len(checks),len(checks)+100,start,last_file=filename)
                cp=root/'candidates.json'
                if split!='discovery':
                    c=D.read(cp);check(f'{model}/{split}:candidate_frozen_before_behavior',c['provenance']['created_at']<D.read(root/f'{split}-behavior-started.json')['provenance']['created_at'])
                pp=root/f'{split}-prediction.json'
                if pp.exists():
                    p=D.read(pp);check(f'{model}/{split}:prediction_candidate_identity',p['candidates_sha256']==D.sha(cp))
                    check(f'{model}/{split}:prediction_array_identity',p['predictions_sha256']==D.sha(root/f'{split}-predictions.npz'))
                cp2=root/f'{split}-causal.json'
                if cp2.exists():
                    c2=D.read(cp2);check(f'{model}/{split}:causal_baseline_replay',c2['baseline_replay_max_choice_error']==0)
                    check(f'{model}/{split}:causal_noop_restore',all(x.get('noop_logits_bitwise_equal',x.get('restored_original_backend_bitwise_equal',False)) for x in c2['calibration']))
                    check(f'{model}/{split}:causal_final_identity',c2['final_state_sha256']==D.sha(root/f'{split}-causal-final.npy'))
                    if not c2['mlp_controls_valid']:check(f'{model}/{split}:invalid_MLP_control_not_executed',all(a not in c2['summary'] for a in ['mlp_selected','mlp_matched']))
                cp3=root/f'{split}-conditional-update.json'
                if cp3.exists():
                    u=D.read(cp3);check(f'{model}/{split}:conditional_fit_identity',u['fit_sha256']==D.sha(root/'conditional-update-fit.npz'))
                    check(f'{model}/{split}:conditional_final_identity',u['final_sha256']==D.sha(root/f'{split}-conditional-update-final.npz'))
                cp4=root/f'{split}-conditional-output.json'
                if cp4.exists():check(f'{model}/{split}:conditional_readout_calibration',all(x['raw_final_replay_exact'] for x in D.read(cp4)['checks']))
                cp5=root/f'{split}-parameter-replay.json'
                if cp5.exists():check(f'{model}/{split}:native_parameter_lineage',D.read(cp5)['passed'])
        splits=list(names)
        check(f'{model}:disjoint_entity_names',all(not(names[a]&names[b]) for i,a in enumerate(splits) for b in splits[i+1:]))
        pilot=root/'pilot-gate.json'
        if pilot.exists():
            pg=D.read(pilot);pb=D.read(root/'pilot-behavior.json');pr=mat['discovery'][:128]
            check(f'{model}:pilot_gate_recomputed',D.behavior(pr,pb['rows'])['passed']==pg['passed'])
            check(f'{model}:pilot_behavior_hash',D.sha(root/'pilot-behavior.json')==pg['behavior_sha256'])
            if not pg['passed']:check(f'{model}:failed_pilot_no_internal_work',not list(root.glob('*-native-manifest.json')) and not(root/'candidates.json').exists())
        diagnostic=root/'pilot-output-diagnostic.json'
        if diagnostic.exists():
            d=D.read(diagnostic);check(f'{model}:diagnostic_not_override',not d['gate_override'] and not d['internal_arrays_written'] and d['first_step_replay_exact'])
    decision=D.read(D.OUT/'qwen14/replication-decision.json')
    if (D.OUT/'glm9/pilot-behavior.json').exists():check('GLM conditional entry authorized',decision['glm_eligible'])
    if (D.OUT/'glm9/pilot-behavior-started.json').exists():check('conditional update recipe fixed before GLM',addendum['provenance']['created_at']<D.read(D.OUT/'glm9/pilot-behavior-started.json')['provenance']['created_at'])
    check('campaign_disk_budget',sum(p.stat().st_size for p in D.OUT.rglob('*') if p.is_file())<32*2**30)
    D.save('audit.json',{'checks':checks,'passed':True,'count':len(checks),'native_files_rehashed':True,'formal_evidence_promotion':0})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['environment','audit']);a=ap.parse_args();globals()[a.stage]()
