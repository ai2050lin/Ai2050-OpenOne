"""Explicit post-gate-failure diagnostic, never a successful confirmation.

The original runner stopped before any confirm intervention. This separate
protocol retains all 768 frozen cases, edits and predictors without re-selection.
Filename continuity serves the unchanged analyzer, whose mandatory behavior gate
keeps EVERY semantic-use and prediction acceptance flag false.
"""
import argparse,time
import numpy as np
import state_formation_core as C
import state_formation_runner as R
from shared.dist_protocol.mechanism_trace import GpuLease

def freeze():
    p=C.checked();gate=C.read(C.OUT/'confirm-behavior.json');assert not gate['passed']
    assert not list(C.OUT.glob('causal-[0-9]*.json'))
    assert 'Confirm behavior gate failed' in (C.OUT/'causal.log').read_text(encoding='utf-8-sig')
    C.save('diagnostic-amendment.json',{'schema':'state_formation_diagnostic_amendment.v1',
        'created_at':__import__('datetime').datetime.now().astimezone().isoformat(),
        'original_plan_sha256':C.sha(C.OUT/'plan.json'),'failed_behavior_sha256':C.sha(C.OUT/'confirm-behavior.json'),
        'original_failed_log_sha256':C.sha(C.OUT/'causal.log'),
        'diagnostic_source_sha256':C.sha(__file__),'effects_frozen_sha256':C.sha(C.OUT/'effects-frozen.json'),
        'purpose':'diagnose response to original frozen edits after the aggregate semantic behavior gate failed; no successful semantic confirmation',
        'exposure':'all confirmation behavior is now exposed; no new blinded causal-confirmation claim',
        'cases':768,'selection':'unchanged one prespecified condition per family, no failed cells removed',
        'operations':'same four candidates, doses, norm-matched controls, source exclusions, seeds and restoration checks as frozen plan',
        'outputs':'causal-NNNN.json and logits retain analyzer compatibility; every record explicitly marked diagnostic',
        'acceptance':'original confirm-behavior remains false; all scope_use_supported and effect_prediction_supported flags must remain false',
        'new_semantic_claims':False,'resource_budget':'within original 32 GiB result budget, C reserve >30 GiB, serial CUDA only',
        'next_confirmatory_recovery':'new materials and new frozen protocol, never relabel this exposed set as fresh'})

def run():
    import torch
    p=C.checked();amend=C.read(C.OUT/'diagnostic-amendment.json')
    assert amend['diagnostic_source_sha256']==C.sha(__file__)
    assert amend['failed_behavior_sha256']==C.sha(C.OUT/'confirm-behavior.json')
    assert not C.read(C.OUT/'confirm-behavior.json')['passed']
    effects=C.read(C.OUT/'effects-frozen.json');assert effects['effects_sha256']==C.sha(C.OUT/'effects.npz')
    selection=C.read(C.OUT/'selection.json');assert effects['selection_sha256']==C.sha(C.OUT/'selection.json')
    material=C.read(C.OUT/'confirm-materials.json')['rows'];amend_sha=C.sha(C.OUT/'diagnostic-amendment.json')
    with GpuLease():
        model=C.load_model('diagnostic_causal');cap=C.Capture(model);start=time.monotonic();C.log('diagnostic_causal',0,768,start)
        try:
            for f in range(768):
                path=C.OUT/f'causal-{f:04d}.json'
                if path.exists():
                    saved=C.read(path);assert saved['diagnostic_amendment_sha256']==amend_sha;continue
                rs=material[f*8:f*8+8];row=rs[R.condition_for(rs[0],16)]
                C.save(f'commit-{f:04d}.json',{'sample_id':row['id'],'effects_frozen_sha256':C.sha(C.OUT/'effects-frozen.json'),
                    'stratum':[row['color'],row['language']],'diagnostic_amendment_sha256':amend_sha})
                original,base=cap.forward(row,full=False);original=original.clone();base_scores=C.scores(original,row,p)
                records=[];kept={'base_logits':C.bits(original)}
                for g,candidate in enumerate(selection['candidates']):
                    state=C.boundary(base,candidate)
                    if f<16:
                        with C.residual_replay(model,candidate,state):zero,replayed=cap.forward(row,full=False)
                        assert torch.equal(original,zero) and torch.equal(C.boundary(replayed,candidate),state)
                        with C.ablated(model,candidate,0):zero,_=cap.forward(row,full=False)
                        assert torch.equal(original,zero)
                    for dose_index,dose in enumerate(p['intervention']['doses']):
                        with C.ablated(model,candidate,dose):edited,edited_values=cap.forward(row,full=False)
                        actual=C.boundary(edited_values,candidate);norm=float((actual.float()-state.float()).norm())
                        edit_metric=R.metrics(edited,original,row,p)
                        if dose==1:kept[candidate['id']+'_logits']=C.bits(edited)
                        random_state,error,random_norm=C.matched_state(state,norm,C.SEED+f*4096+g*16+dose_index+10000000)
                        with C.residual_replay(model,candidate,random_state):control,control_values=cap.forward(row,full=False)
                        assert torch.equal(C.boundary(control_values,candidate),random_state)
                        records.append({'candidate':candidate['id'],'dose':dose,'method':'ablation','edited':edit_metric,
                            'norm_control':R.metrics(control,original,row,p),'realized_delta_norm':norm,'control_delta_norm':random_norm,
                            'norm_relative_error':error,'norm_valid':error<=.02})
                    if candidate['module']=='attention':
                        key='instruction' if candidate['factor']=='language' else 'record';indices=row['spans'][key];neutral=row['spans']['neutral']
                        begin=(len(neutral)-len(indices))//2;control_indices=neutral[begin:begin+len(indices)];assert len(control_indices)==len(indices)
                        with C.exclude_source(model,candidate,indices):edited,_=cap.forward(row,full=False)
                        with C.exclude_source(model,candidate,control_indices):control,_=cap.forward(row,full=False)
                        records.append({'candidate':candidate['id'],'dose':1.,'method':'source_exclusion','source':key,'masked_positions':indices,'neutral_positions':control_indices,
                            'edited':R.metrics(edited,original,row,p),'norm_control':R.metrics(control,original,row,p),'norm_valid':None,
                            'scope':'equal token count, not equal attention mass or equal residual norm'})
                restored,_=cap.forward(row,full=False);assert torch.equal(original,restored)
                digest=C.save_npz(f'causal-logits-{f:04d}.npz',**kept)
                C.save(path.name,{'selection_sha256':C.sha(C.OUT/'selection.json'),'plan_sha256':C.sha(C.OUT/'plan.json'),'sample':row,
                    'baseline':base_scores,'records':records,'native_logits_sha256':digest,'restoration_bitwise':True,'identity_replay_checked':f<16,
                    'diagnostic_amendment_sha256':amend_sha,'protocol_scope':'post_behavior_gate_failure_diagnostic','formal_confirmation':False})
                C.reserve()
                if (f+1)%8==0 or f==767:C.log('diagnostic_causal',f+1,768,start)
                assert time.monotonic()-start<120*60
        finally:cap.close()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','run']);args=parser.parse_args()
    {'freeze':freeze,'run':run}[args.stage]()
