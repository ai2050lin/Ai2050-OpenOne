"""Sequential CUDA measurement; all edits restored within each original forward."""
import argparse,contextlib,time
import numpy as np
import state_formation_core as C
from shared.dist_protocol.mechanism_trace import GpuLease

def behavior(scores,p):
    cells={}
    for row in scores:cells.setdefault(f"{row['frame']}:{row['history']}",[]).append(row)
    result={k:{'n':len(v),'four_choice_accuracy':float(np.mean([r['four_correct'] for r in v])),
        'color_accuracy':float(np.mean([r['color_correct'] for r in v])),
        'language_accuracy':float(np.mean([r['language_correct'] for r in v]))} for k,v in cells.items()}
    top=float(np.mean([r['first_top1'] for r in scores]))
    return {'cells':result,'first_top1':top,'rows':len(scores),
        'passed':min(v['four_choice_accuracy'] for v in result.values())>=p['behavior_gate']['four_choice_per_frame_history_min'] and top>=p['behavior_gate']['first_top1_overall_min'],
        'scope':'four explicit first-token alternatives, not unrestricted free generation or full answer formatting'}

def collect(model,p,split):
    if split=='discovery':assert C.read(C.OUT/'pilot-behavior.json')['passed'],'Pilot gate failed'
    if split=='confirm':
        frozen=C.read(C.OUT/'effects-frozen.json');assert frozen['confirm_unopened'];assert C.sha(C.OUT/'effects.npz')==frozen['effects_sha256']
    material=C.read(C.OUT/('discovery-materials.json' if split=='pilot' else split+'-materials.json'))['rows']
    families=p['pilot_families'] if split=='pilot' else list(range(len(material)//8))
    cap=C.Capture(model);start=time.monotonic();scores=[];C.log(split,0,len(families),start)
    try:
        for n,f in enumerate(families):
            destination=C.OUT/f'{split}-{f:04d}.json'
            if destination.exists():
                old=C.read(destination);assert old['plan_sha256']==C.sha(C.OUT/'plan.json')
                if split!='pilot':assert C.sha(destination.with_suffix('.npz'))==old['native_sha256']
                scores.extend(old['scores']);continue
            arrays={};family_scores=[];rs=material[f*8:f*8+8]
            for row in rs:
                logits,values=cap.forward(row);native=cap.arrays(values)
                for key,value in native.items():arrays.setdefault(key,[]).append(value)
                family_scores.append({**C.scores(logits,row,p),'id':row['id'],'family':f,'frame':row['frame'],'history':row['history'],'noun':row['noun']})
            arrays={k:np.stack(v) for k,v in arrays.items()}
            negative={}
            for key in ['hidden','post_attention','attn_write','mlp_write']:
                maximum=0.;exact=True
                for indices in [[0,1,4,5],[2,3,6,7]]:
                    a=C.unbits(arrays[key][indices,:,0]);b=a[0]
                    maximum=max(maximum,float(np.max(np.linalg.norm(a-b,axis=-1)/np.maximum(np.linalg.norm(b,axis=-1),1e-8))))
                    exact=exact and bool(np.all(arrays[key][indices,:,0]==arrays[key][indices[0],:,0]))
                negative[key]={'future_factors_bitwise_equal':exact,'max_relative_l2':maximum}
            assert max(v['max_relative_l2'] for v in negative.values())<.001,'Prefix measurement negative control failed'
            digest=C.save_npz(f'{split}-{f:04d}.npz',**arrays) if split!='pilot' else None
            C.save(destination.name,{'plan_sha256':C.sha(C.OUT/'plan.json'),'native_sha256':digest,'scores':family_scores,'negative_controls':negative,
                'capture_identity':'BF16 residual+attention==post_attention;post_attention+MLP==raw block output, every layer/position/sample verified'})
            scores.extend(family_scores);C.reserve()
            if (n+1)%8==0 or n+1==len(families):C.log(split,n+1,len(families),start)
            assert time.monotonic()-start<120*60,'Stage budget exhausted; partial files preserved'
        result=behavior(scores,p);C.save(split+'-behavior.json',result);C.log(split+'_complete',len(families),len(families),start,passed=result['passed'],top1=result['first_top1'])
    finally:cap.close()

def condition_for(row,nframes):
    j=row['local_family'];return (row['frame']+j+2*row['history']+j//16)%8

def metrics(logits,baseline,row,p):
    import torch
    result=C.scores(logits,row,p)
    lp=torch.log_softmax(baseline.float(),-1);lq=torch.log_softmax(logits.float(),-1)
    result['KL_from_base']=float((lq.exp()*(lq-lp)).sum())
    return result

def causal(model,p,calibrate=False):
    import torch
    selection=C.read(C.OUT/'selection.json');assert selection['estimator_sha256']==C.sha(C.OUT/'estimator.npz')
    split='discovery' if calibrate else 'confirm';material=C.read(C.OUT/(split+'-materials.json'))['rows']
    families=p['intervention']['calibration_families'] if calibrate else list(range(768));stage='calibration' if calibrate else 'causal'
    if calibrate:assert not list(C.OUT.glob('confirm-*.npz')),'Confirm already exposed before effect freeze'
    else:
        assert C.read(C.OUT/'confirm-behavior.json')['passed'],'Confirm behavior gate failed: stop semantic causal experiment'
        effect_freeze=C.read(C.OUT/'effects-frozen.json');assert effect_freeze['effects_sha256']==C.sha(C.OUT/'effects.npz')
        assert effect_freeze['selection_sha256']==C.sha(C.OUT/'selection.json')
    cap=C.Capture(model);start=time.monotonic();C.log(stage,0,len(families),start);checks=[]
    try:
        for n,f in enumerate(families):
            path=C.OUT/f'{stage}-{f:04d}.json'
            if path.exists():
                saved=C.read(path);assert saved['selection_sha256']==C.sha(C.OUT/'selection.json');continue
            rs=material[f*8:f*8+8];row=rs[condition_for(rs[0],8 if calibrate else 16)]
            if not calibrate:C.save(f'commit-{f:04d}.json',{'sample_id':row['id'],'effects_frozen_sha256':C.sha(C.OUT/'effects-frozen.json'),'stratum':[row['color'],row['language']]})
            original,base=cap.forward(row,full=False);original=original.clone();base_scores=C.scores(original,row,p);records=[];kept={'base_logits':C.bits(original)}
            for g,candidate in enumerate(selection['candidates']):
                state=C.boundary(base,candidate)
                # Matching exact residual states avoids a second BF16 subtraction/addition.
                if n<16:
                    with C.residual_replay(model,candidate,state):zero,replayed=cap.forward(row,full=False)
                    assert torch.equal(original,zero),('identity replay logits',stage,f,candidate['id'])
                    assert torch.equal(C.boundary(replayed,candidate),state),('identity replay state',candidate['id'])
                    with C.ablated(model,candidate,0):zero,_=cap.forward(row,full=False)
                    assert torch.equal(original,zero),('zero-dose replay',candidate['id'])
                for dose_index,dose in enumerate(p['intervention']['doses']):
                    with C.ablated(model,candidate,dose):edited,edited_values=cap.forward(row,full=False)
                    actual_state=C.boundary(edited_values,candidate);norm=float((actual_state.float()-state.float()).norm())
                    edit_metric=metrics(edited,original,row,p)
                    if dose==1:kept[candidate['id']+'_logits']=C.bits(edited)
                    random_state,error,random_norm=C.matched_state(state,norm,C.SEED+f*4096+g*16+dose_index+(0 if calibrate else 10000000))
                    with C.residual_replay(model,candidate,random_state):control,control_values=cap.forward(row,full=False)
                    assert torch.equal(C.boundary(control_values,candidate),random_state),('realized control state',candidate['id'])
                    records.append({'candidate':candidate['id'],'dose':dose,'method':'ablation','edited':edit_metric,
                        'norm_control':metrics(control,original,row,p),'realized_delta_norm':norm,'control_delta_norm':random_norm,
                        'norm_relative_error':error,'norm_valid':error<=.02})
                if candidate['module']=='attention':
                    key='instruction' if candidate['factor']=='language' else 'record';indices=row['spans'][key];neutral=row['spans']['neutral'];start_index=(len(neutral)-len(indices))//2
                    control_indices=neutral[start_index:start_index+len(indices)];assert len(control_indices)==len(indices)
                    with C.exclude_source(model,candidate,indices):edited,_=cap.forward(row,full=False)
                    with C.exclude_source(model,candidate,control_indices):control,_=cap.forward(row,full=False)
                    records.append({'candidate':candidate['id'],'dose':1.,'method':'source_exclusion','source':key,'masked_positions':indices,'neutral_positions':control_indices,
                        'edited':metrics(edited,original,row,p),'norm_control':metrics(control,original,row,p),
                        'norm_valid':None,'scope':'equal token count, not equal attention mass or equal residual norm'})
            restored,_=cap.forward(row,full=False);assert torch.equal(original,restored),'Restoration after native edits failed'
            digest=C.save_npz(f'{stage}-logits-{f:04d}.npz',**kept)
            C.save(path.name,{'selection_sha256':C.sha(C.OUT/'selection.json'),'plan_sha256':C.sha(C.OUT/'plan.json'),'sample':row,
                'baseline':base_scores,'records':records,'native_logits_sha256':digest,'restoration_bitwise':True,'identity_replay_checked':n<16})
            C.reserve()
            if (n+1)%8==0 or n+1==len(families):C.log(stage,n+1,len(families),start)
            assert time.monotonic()-start<120*60
    finally:cap.close()
    if calibrate:
        keys=[];sums={};counts={}
        for f in families:
            doc=C.read(C.OUT/f'calibration-{f:04d}.json');row=doc['sample'];stratum=(row['color']==1)*2+(row['language']==1)
            for record in doc['records']:
                key=f"{record['candidate']}|{record['method']}|{record['dose']}"
                if key not in sums:keys.append(key);sums[key]=np.zeros((4,4));counts[key]=np.zeros(4,int)
                sums[key][stratum]+=np.array(record['edited']['four_logits'])-doc['baseline']['four_logits'];counts[key][stratum]+=1
        assert all(np.min(v)>0 for v in counts.values())
        changes=np.stack([sums[k]/counts[k][:,None] for k in keys]);global_mean=np.stack([sums[k].sum(0)/counts[k].sum() for k in keys])
        digest=C.save_npz('effects.npz',keys=np.array(keys),changes=changes,global_mean=global_mean,counts=np.stack([counts[k] for k in keys]))
        C.save('effects-frozen.json',{'schema':'state_formation_effects.v1','selection_sha256':C.sha(C.OUT/'selection.json'),
            'effects_sha256':digest,'confirm_unopened':not list(C.OUT.glob('confirm-*.npz')),'families':families,
            'predictor_inputs':'actual baseline four logits plus discovery-fitted change for parsed color/language; not input-only output prediction'})

def capability(model,tokenizer,p):
    candidates=C.read(C.OUT/'selection.json')['candidates'];items=C.read(C.OUT/'capability-materials.json')['rows'];cap=C.Capture(model)
    choices=[tokenizer.encode(' '+k,add_special_tokens=False)[0] for k in 'ABCDE'];result=[];start=time.monotonic();C.log('capability',0,len(items),start)
    try:
        for f,item in enumerate(items):
            path=C.OUT/f'capability-{f:04d}.json'
            if path.exists():result.append(C.read(path));continue
            text=item['question']+'\n'+'\n'.join(f'{letter}. {value}' for letter,value in zip('ABCDE',item['choices']))+'\n请选择一个正确选项字母。'
            prompt=tokenizer.apply_chat_template([{'role':'user','content':text}],tokenize=False,add_generation_prompt=True,enable_thinking=False)+'Answer:'
            row={'tokens':tokenizer.encode(prompt,add_special_tokens=False)};base,_=cap.forward(row,full=False)
            scores={'base':base[choices[:len(item['choices'])]].float().cpu().tolist()}
            for candidate in candidates:
                with C.ablated(model,candidate,1.):edit,_=cap.forward(row,full=False)
                scores[candidate['id']]=edit[choices[:len(item['choices'])]].float().cpu().tolist()
            doc=C.save(path.name,{'id':item['id'],'gold':'ABCDE'.index(item['answer']),'scores':scores,'prompt':prompt,'selection_sha256':C.sha(C.OUT/'selection.json')})
            result.append(doc)
            if (f+1)%32==0:C.log('capability',f+1,len(items),start)
        C.save('capability.json',{'rows':result,'scope':'512 previously exposed keyed multiple-choice questions; no new general capability guarantee'})
    finally:cap.close()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',required=True,choices=['pilot','discovery','calibrate','confirm','causal','capability']);args=parser.parse_args()
    p=C.checked();C.reserve()
    with GpuLease():
        model=C.load_model(args.stage)
        if args.stage in ['pilot','discovery','confirm']:collect(model,p,args.stage)
        elif args.stage in ['calibrate','causal']:causal(model,p,calibrate=args.stage=='calibrate')
        else:
            from transformers import AutoTokenizer
            capability(model,AutoTokenizer.from_pretrained(C.MODEL,local_files_only=True),p)

if __name__=='__main__':main()
