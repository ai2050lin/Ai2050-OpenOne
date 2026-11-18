"""Predeclared final-query edits of the previous run's frozen native groups."""
import argparse,contextlib,hashlib,time
import numpy as np
import source_output_core as S
from source_output_core import C,R
from source_output_runner import admitted,check_analysis
from record_selection_match import matched_direction
from shared.dist_protocol.mechanism_trace import GpuLease
from state_formation_math import bootstrap

def selected(rows,active):
    result=[]
    for row in rows:
        j=row['local_family'];f=row['frame'];alphabet='digits' if j<8 else 'letters';s=1 if (f+j+j//2+j//4)%2 else -1
        if S.branch(row) in active and row['alphabet']==alphabet and row['status_position']==s:result.append(row)
    return result

def scored(logits,base,row,p):
    import torch
    lp=torch.log_softmax(base.float(),-1);lq=torch.log_softmax(logits.float(),-1)
    return {**S.score(logits,row,p),'KL_edit_base':float((lq.exp()*(lq-lp)).sum()),'logits_sha256':hashlib.sha256(C.bits(logits).tobytes()).hexdigest()}

def run():
    import torch
    p=C.checked();check_analysis();active=admitted('confirm');rows=selected(C.read(C.OUT/'confirm-materials.json')['rows'],active)
    C.save('causal-admission.json',{'active_branches':active,'samples':len(rows),'selection':'Frozen metadata-only rule; includes every chosen error and tie',
        'analysis_plan_sha256':C.sha(C.OUT/'analysis-plan.json'),'stopped_branches':C.read(C.OUT/'admission-confirm.json')['stopped_branches']})
    if not rows:C.save('causal-complete.json',{'n':0,'reason':'No behavior-admitted branch'});return
    start=time.monotonic();candidates=p['fixed_candidates']
    with GpuLease():
        model=C.load_model('causal');cap=C.Capture(model);C.log('causal',0,len(rows),start)
        try:
            for i,row in enumerate(rows):
                name=f'causal-{i:04d}.json';path=C.OUT/name
                if path.exists():assert C.read(path)['analysis_plan_sha256']==C.sha(C.OUT/'analysis-plan.json');continue
                original,base=cap.forward(row,full=False);original=original.clone();baseline=scored(original,original,row,p)
                recorded=C.read(C.OUT/f"behavior-confirm-{row['family']:04d}.json")['scores'][row['variant']]
                for k,v in S.score(original,row,p).items():assert recorded[k]==v,(i,k)
                keep={'base':C.bits(original)} if i%16==0 else {};edits=[]
                for j,candidate in enumerate(candidates):
                    state=C.boundary(base,candidate)
                    if i<4:
                        with R.group_edit(model,candidate,0):zero,_=cap.forward(row,full=False)
                        assert torch.equal(original,zero)
                        with C.residual_replay(model,candidate,state):zero,_=cap.forward(row,full=False)
                        assert torch.equal(original,zero)
                    with R.group_edit(model,candidate,1):value,values=cap.forward(row,full=False)
                    norm=float((C.boundary(values,candidate).float()-state.float()).norm())
                    edit={'candidate':candidate['id'],'edited':scored(value,original,row,p),'norm':norm,'controls':[]}
                    if i%16==0:keep[candidate['id']]=C.bits(value)
                    for control in candidate['controls'][:2]:
                        cc={**candidate,'indices':control['indices']}
                        with R.group_edit(model,cc,1):value,values=cap.forward(row,full=False)
                        other=C.boundary(values,candidate);initial=float((other.float()-state.float()).norm())
                        rec={'indices':cc['indices'],'original':scored(value,original,row,p),'original_norm':initial}
                        if initial==0 and norm>0:
                            rec.update(valid=False,reason='Zero native direction cannot match nonzero candidate; retained, not filtered')
                        else:
                            match,error,actual,_=matched_direction(state,other,norm);scale=norm/initial if initial else 0.
                            # Retain the invalid scale, but do not execute an extrapolation above the frozen bound.
                            rec.update(scale=scale,norm_error=error,actual_norm=actual,valid=bool(error<=.02 and scale<=4))
                            if rec['valid']:
                                with C.residual_replay(model,candidate,match):value,values=cap.forward(row,full=False)
                                assert torch.equal(C.boundary(values,candidate),match);rec['matched']=scored(value,original,row,p)
                            else:rec['reason']='Frozen norm tolerance or maximum scale exceeded; this control claim stops'
                        edit['controls'].append(rec)
                    match,error,actual=C.matched_state(state,norm,C.SEED+i*32+j)
                    with C.residual_replay(model,candidate,match):value,values=cap.forward(row,full=False)
                    assert torch.equal(C.boundary(values,candidate),match)
                    edit.update(random=scored(value,original,row,p),random_error=error,random_norm=actual,random_valid=error<=.02)
                    edits.append(edit)
                with contextlib.ExitStack() as stack:
                    for c in candidates:stack.enter_context(R.group_edit(model,c,1))
                    value,_=cap.forward(row,full=False)
                joint=scored(value,original,row,p)
                if i%16==0:keep['joint']=C.bits(value)
                restored,_=cap.forward(row,full=False);assert torch.equal(original,restored)
                digest=C.save_npz(f'causal-logits-{i:04d}.npz',**keep) if keep else None
                C.save(name,{'sample_id':row['id'],'family':row['family'],'frame':row['frame'],'branch':S.branch(row),'variant':row['variant'],
                    'baseline':baseline,'edits':edits,'joint':joint,'restoration_bitwise':True,'zero_identity_checked':i<4,'audit_logits_sha256':digest,
                    'analysis_plan_sha256':C.sha(C.OUT/'analysis-plan.json')})
                C.reserve()
                if (i+1)%8==0 or i==len(rows)-1:C.log('causal',i+1,len(rows),start)
                assert time.monotonic()-start<60*60
        finally:cap.close()
    C.save('causal-complete.json',{'n':len(rows),'restored_all':True})

def aggregate():
    p=C.checked();check_analysis();active=admitted('confirm');rows=selected(C.read(C.OUT/'confirm-materials.json')['rows'],active)
    records=[C.read(C.OUT/f'causal-{i:04d}.json') for i in range(len(rows))];results=[];bybranch={};specificity=[]
    for branch in active:
        part=[r for r in records if r['branch']==branch];families=sorted({r['family'] for r in part});frames=np.array([next(r['frame'] for r in part if r['family']==f) for f in families]);bdata={}
        def family_mean(values):return np.array([np.mean([v for r,v in zip(part,values) if r['family']==f]) for f in families])
        for candidate in p['fixed_candidates']:
            cid=candidate['id'];ed=[next(e for e in r['edits'] if e['candidate']==cid) for r in part];loss=[r['baseline']['target_margin']-e['edited']['target_margin'] for r,e in zip(part,ed)]
            values=family_mean(loss);bdata[cid]={'values':values,'families':families,'frames':frames}
            random=family_mean([r['baseline']['target_margin']-e['random']['target_margin'] for r,e in zip(part,ed)])
            valid=all(e['random_valid'] and all(c['valid'] for c in e['controls']) for e in ed);control=None
            if valid:control=family_mean([r['baseline']['target_margin']-np.mean([c['matched']['target_margin'] for c in e['controls']]) for r,e in zip(part,ed)])
            excess_random=bootstrap(values-random,frames,alpha=p['alpha']);excess_native=bootstrap(values-control,frames,alpha=p['alpha']) if valid else None
            results.append({'branch':branch,'candidate':cid,'n':len(part),'families':len(families),'margin_loss':bootstrap(values,frames,alpha=p['alpha']),
                'mean_loss':float(values.mean()),'random_excess':excess_random,'native_excess':excess_native,'controls_valid':valid,
                'invalid_native_controls':sum(not c['valid'] for e in ed for c in e['controls']),'max_scale':max((c.get('scale',0) for e in ed for c in e['controls']),default=0),
                'mean_kl':float(np.mean([e['edited']['KL_edit_base'] for e in ed])),
                'top1_changes':sum(e['edited']['argmax']!=r['baseline']['argmax'] for r,e in zip(part,ed)),
                'correct_to_wrong':sum(r['baseline']['strict_correct'] and not e['edited']['strict_correct'] for r,e in zip(part,ed)),
                'supported':bool(valid and values.mean()>=.2 and excess_random['frame_interval'][0]>0 and excess_native['frame_interval'][0]>0),
                'scope':'fixed native group final-query full ablation; all chosen families retained'})
        joint=family_mean([r['baseline']['target_margin']-r['joint']['target_margin'] for r in part]);interaction=joint-sum(d['values'] for d in bdata.values())
        bdata['joint']={'values':joint,'families':families,'frames':frames};bybranch[branch]=bdata
        results.append({'branch':branch,'candidate':'joint','n':len(part),'families':len(families),'mean_loss':float(joint.mean()),
            'margin_loss':bootstrap(joint,frames,alpha=p['alpha']),'interaction':bootstrap(interaction,frames,alpha=p['alpha']),
            'top1_changes':sum(r['joint']['argmax']!=r['baseline']['argmax'] for r in part),'supported':False,'scope':'joint effect only; no directed mediation claim'})
    for alphabet in S.ALPHABETS:
        for task in ['status','time']:
            source=f'{task}:{alphabet}';control='explicit:'+alphabet
            if source not in bybranch or control not in bybranch:
                specificity.append({'branch':source,'status':'not_tested','reason':'One whole branch failed the behavior gate'});continue
            for cid,d in bybranch[source].items():
                other=bybranch[control][cid];assert d['families']==other['families'];delta=d['values']-other['values'];interval=bootstrap(delta,d['frames'],alpha=p['alpha'])
                lo,hi=interval['frame_interval'];specificity.append({'branch':source,'candidate':cid,'difference_vs_explicit':interval,
                    'equivalent_within_0.1':bool(lo>=-.1 and hi<=.1),'status':'tested','scope':'Conditional task effect, not proof of a unique selector'})
    C.save('effects.json',{'schema':'source_output_effects.v1','results':results,'specificity':specificity,'causal_samples':len(rows),'formal_promotion':False})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['run','aggregate']);globals()[parser.parse_args().stage]()
