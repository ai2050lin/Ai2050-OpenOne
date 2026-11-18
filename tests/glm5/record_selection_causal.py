"""Native head/unit edits with same-layer groups and realized-norm controls."""
import time
import record_selection_core_v3 as R
from record_selection_core_v3 import C
from shared.dist_protocol.mechanism_trace import GpuLease
from record_selection_runner_v3 import check_analysis

def scored(logits,baseline,row,p):
    import torch
    result=R.score(logits,row,p);lp=torch.log_softmax(baseline.float(),-1);lq=torch.log_softmax(logits.float(),-1)
    result['KL_edit_base']=float((lq.exp()*(lq-lp)).sum());return result

def run():
    import torch
    p=C.checked();check_analysis();selection=C.read(C.OUT/'selection.json');gate=C.read(C.OUT/'confirm-behavior.json')
    rows=C.read(C.OUT/'confirm-materials.json')['rows'];candidates=selection['candidates'];start=time.monotonic()
    scope='frozen_semantic_test' if gate['passed'] and selection.get('semantic_discovery_admitted',True) else 'observed_choice_diagnostic_after_behavior_failure'
    C.save('causal-admission.json',{'schema':'record_selection_causal_admission.v1','behavior_sha256':C.sha(C.OUT/'confirm-behavior.json'),
        'selection_sha256':C.sha(C.OUT/'selection.json'),'scope':scope,'failed_cells':[k for k,v in gate['cells'].items() if v['strict_accuracy']<.9],
        'semantic_positive_allowed':scope=='frozen_semantic_test','counterexamples_kept':True})
    with GpuLease():
        model=C.load_model('causal');cap=C.Capture(model);C.log('causal',0,384,start)
        try:
            for f in range(384):
                dest=C.OUT/f'causal-{f:04d}.json'
                if dest.exists():
                    assert C.read(dest)['selection_sha256']==C.sha(C.OUT/'selection.json');continue
                rs=rows[f*8:(f+1)*8];i=1+2*((rs[0]['frame']+rs[0]['local_family'])%4);row=rs[i];agree=rs[i-1]
                original,base=cap.forward(row,full=False);original=original.clone();baseline=R.score(original,row,p);records=[];keep={'base':C.bits(original)}
                for j,candidate in enumerate(candidates):
                    state=C.boundary(base,candidate)
                    if f<8:
                        with R.group_edit(model,candidate,0):zero,_=cap.forward(row,full=False)
                        assert torch.equal(original,zero)
                        with C.residual_replay(model,candidate,state):zero,_=cap.forward(row,full=False)
                        assert torch.equal(original,zero)
                    for dose in [.5,1.]:
                        with R.group_edit(model,candidate,dose):edited,ev=cap.forward(row,full=False)
                        norm=float((C.boundary(ev,candidate).float()-state.float()).norm())
                        rec={'candidate':candidate['id'],'method':'ablation','dose':dose,'edited':scored(edited,original,row,p),'realized_norm':norm}
                        if dose==1:
                            keep[candidate['id']]=C.bits(edited);controls=[]
                            for control in candidate['controls']:
                                cc={**candidate,'indices':control['indices']}
                                with R.group_edit(model,cc,1.):value,cv=cap.forward(row,full=False)
                                controls.append({**scored(value,original,row,p),'indices':cc['indices'],
                                    'realized_norm':float((C.boundary(cv,candidate).float()-state.float()).norm())})
                            random_state,error,actualnorm=C.matched_state(state,norm,C.SEED+f*32+j)
                            with C.residual_replay(model,candidate,random_state):value,cv=cap.forward(row,full=False)
                            assert torch.equal(C.boundary(cv,candidate),random_state)
                            assert error<=.02,('norm match failed',f,candidate['id'],error)
                            rec.update(group_controls=controls,norm_control=scored(value,original,row,p),norm_relative_error=error,control_norm=actualnorm)
                        records.append(rec)
                    if candidate['kind']=='heads':
                        valid='record1' if row['valid_position']==-1 else 'record2';old='record2' if valid=='record1' else 'record1'
                        for label,key in [('valid_source',valid),('obsolete_source',old)]:
                            with R.group_source_exclusion(model,candidate,row['spans'][key]):value,_=cap.forward(row,full=False)
                            records.append({'candidate':candidate['id'],'method':label,'dose':1.,'edited':scored(value,original,row,p),
                                'positions':row['spans'][key],'scope':'selected heads final-query direct edges only; attention renormalizes; source sizes may differ'})
                restored,_=cap.forward(row,full=False);assert torch.equal(original,restored)
                agreement,_=cap.forward(agree,full=False);agreement=agreement.clone();agreement_edits={}
                for candidate in candidates:
                    with R.group_edit(model,candidate,1.):value,_=cap.forward(agree,full=False)
                    agreement_edits[candidate['id']]=scored(value,agreement,agree,p)
                restored,_=cap.forward(agree,full=False);assert torch.equal(agreement,restored)
                digest=C.save_npz(f'causal-logits-{f:04d}.npz',**keep)
                C.save(dest.name,{'selection_sha256':C.sha(C.OUT/'selection.json'),'scope':scope,'sample':row,'baseline':baseline,'records':records,
                    'agreement_sample':agree,'agreement_baseline':R.score(agreement,agree,p),'agreement_edits':agreement_edits,
                    'restoration_bitwise':True,'zero_identity_checked':f<8,'native_logits_sha256':digest})
                C.reserve()
                if (f+1)%8==0 or f==383:C.log('causal',f+1,384,start)
                assert time.monotonic()-start<90*60
        finally:cap.close()

if __name__=='__main__':run()
