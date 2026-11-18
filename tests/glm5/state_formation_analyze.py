"""Frozen reporting of confirmation, baselines, original-model use and failures."""
import time
import numpy as np
import state_formation_core as C
import state_formation_math as M

def state_name(index):
    if index==0:return 'embedding'
    if index==73:return 'final_RMSNorm'
    return f'B{(index-1)//2}:'+('post_attention' if index%2 else 'post_MLP')

def analyze():
    p=C.checked();selection=C.read(C.OUT/'selection.json');freeze=C.read(C.OUT/'effects-frozen.json');gate=C.read(C.OUT/'confirm-behavior.json')
    assert selection['estimator_sha256']==C.sha(C.OUT/'estimator.npz');assert freeze['effects_sha256']==C.sha(C.OUT/'effects.npz')
    estimator=np.load(C.OUT/'estimator.npz');beta=estimator['beta'].astype(float);var=estimator['variance'].astype(float)
    rows=C.read(C.OUT/'confirm-materials.json')['rows'];x=C.design(rows[:8]);frames=np.array([rows[f*8]['frame'] for f in range(768)])
    predictions={name:M.predict(beta,x,cols) for name,cols in [('constant',[0]),('additive',[0,1,2,3]),('four_state',[0,3,6,7]),('eight_cell',list(range(8)))]}
    center_names=['fixed','oracle_center','decoded_center','pooled','norm','wrong_center','permuted_labels']
    center_acc=np.zeros((768,7,74));probes=np.zeros((768,74,2));mse=np.zeros((768,74,4));observed=np.zeros_like(beta)
    module_coef=np.zeros((2,36,8,2560));control_max=0.;start=time.monotonic();C.log('confirm_analysis',0,768,start)
    for f in range(768):
        with M.load_family('confirm',f) as native:
            h=M.states(native);observed+=M.contrast(h,x)
            for k,key in enumerate(['attn_write','mlp_write']):module_coef[k]+=M.contrast(C.unbits(native[key])[:,:,3].transpose(1,0,2),x)
        for j,factor in enumerate([3,6]):probes[f,:,j]=M.accuracy(M.project(h,*M.direction(beta,var,factor)),x[:,factor])
        centers=M.center_predictions(h,beta,var,estimator['norm_coeff'],estimator['permuted_direction'],x[:,6])
        for j,name in enumerate(center_names):center_acc[f,j]=M.accuracy(centers[name],x[:,3])
        for j,pred in enumerate(predictions.values()):mse[f,:,j]=np.mean((h-pred)**2,(1,2))
        meta=C.read(C.OUT/f'confirm-{f:04d}.json');control_max=max(control_max,max(v['max_relative_l2'] for v in meta['negative_controls'].values()))
        if (f+1)%32==0:C.log('confirm_analysis',f+1,768,start)
    observed/=768;module_coef/=768
    primary_index=2*p['center_test']['primary_hidden_index'];gain=center_acc[:,2,primary_index]-center_acc[:,0,primary_index]
    center_ci=M.bootstrap(gain,frames);center_pass=bool(center_ci['mean']>=.05 and center_ci['frame_interval'][0]>0 and gate['passed'])
    center_rows=[]
    for hidden in [33,34,35]:
        index=2*hidden
        center_rows.append({'hidden_index':hidden,'state_index':index,'accuracies':dict(zip(center_names,center_acc[:,:,index].mean(0).tolist())),
            'gain':M.bootstrap(center_acc[:,2,index]-center_acc[:,0,index],frames),
            'history':{str(history):{name:float(center_acc[[rows[f*8]['history']==history for f in range(768)],j,index].mean()) for j,name in enumerate(center_names)} for history in [0,1]},
            'by_frame':{str(frame):dict(zip(center_names,center_acc[frames==frame,:,index].mean(0).tolist())) for frame in range(16)},
            'primary':hidden==34,'supported':center_pass if hidden==34 else None})
    formation=[]
    for candidate in selection['candidates']:
        field=3 if candidate['factor']=='language' else 6;pre=candidate['before_state'];post=candidate['after_state'];w=beta[post,field]/np.maximum(var[post],1e-6);w/=max(float(beta[post,field]@w),1e-12)
        signed_before=float(observed[pre,field]@w);signed_after=float(observed[post,field]@w)
        formation.append({**candidate,'confirmation_probe_accuracy_before':float(probes[:,pre,0 if field==3 else 1].mean()),
            'confirmation_probe_accuracy_after':float(probes[:,post,0 if field==3 else 1].mean()),
            'same_post_decoder_signed_contrast_before':signed_before,'same_post_decoder_signed_contrast_after':signed_after,
            'same_post_decoder_signed_increment':signed_after-signed_before,
            'raw_write_contrast_norm':float(np.linalg.norm(module_coef[0 if candidate['module']=='attention' else 1,candidate['block'],field]))})
    # Read the entire frozen intervention set, without dropping controls/failures.
    effects=np.load(C.OUT/'effects.npz');effect_keys=effects['keys'].tolist();causal=[C.read(C.OUT/f'causal-{f:04d}.json') for f in range(768)]
    for item in causal:
        assert item['restoration_bitwise'] and item['selection_sha256']==C.sha(C.OUT/'selection.json')
    effect_rows=[]
    for candidate in selection['candidates']:
        for method in ['ablation']+(['source_exclusion'] if candidate['module']=='attention' else []):
            for dose in p['intervention']['doses'] if method=='ablation' else [1.]:
                key=f"{candidate['id']}|{method}|{dose}";j=effect_keys.index(key);field=candidate['factor']+'_margin';losses=[];nulls=[];valid=[];errors=[];nochange=[];global_errors=[];counter=[]
                flip=lost=gained=other_lost=0;kl=[]
                for f,item in enumerate(causal):
                    rec=next(r for r in item['records'] if r['candidate']==candidate['id'] and r['method']==method and r['dose']==dose)
                    base=item['baseline'];edit=rec['edited'];loss=base[field]-edit[field];null=base[field]-rec['norm_control'][field]
                    losses.append(loss);nulls.append(null);valid.append(rec['norm_valid'] if method=='ablation' else True);kl.append(edit['KL_from_base'])
                    c=item['sample']['color'];lang=item['sample']['language'];stratum=(c==1)*2+(lang==1)
                    gold=(0 if c==1 else 1)+(0 if lang==1 else 2);other=gold^(1 if candidate['factor']=='color' else 2)
                    change=effects['changes'][j,stratum];global_change=effects['global_mean'][j]
                    predicted_loss=-(change[gold]-change[other]);global_loss=-(global_change[gold]-global_change[other])
                    errors.append((loss-predicted_loss)**2);nochange.append(loss*loss);global_errors.append((loss-global_loss)**2)
                    flip+=base['argmax']!=edit['argmax'];lost+=bool(base['first_top1'] and not edit['first_top1']);gained+=bool(not base['first_top1'] and edit['first_top1'])
                    other_factor='language' if candidate['factor']=='color' else 'color';other_lost+=bool(base[other_factor+'_correct'] and not edit[other_factor+'_correct'])
                    counter.append({'family':f,'target_margin_loss':loss,'matched_excess':loss-null,'norm_valid':valid[-1],'id':item['sample']['id'],'prompt':item['sample']['prompt']})
                excess=M.bootstrap(np.asarray(losses)-nulls,frames);fraction=float(np.mean(valid));baseline=min(np.mean(nochange),np.mean(global_errors));ratio=float(np.mean(errors)/max(baseline,1e-15))
                supported=bool(np.mean(losses)>=.2 and excess['frame_interval'][0]>0 and fraction>=.95 and gate['passed'])
                failures=[]
                if not gate['passed']:failures.append('确认行为门未全过，语义机制主张不晋级')
                if np.mean(losses)<.2:failures.append('目标优势下降未达到冻结0.2门')
                if excess['frame_interval'][0]<=0:failures.append('按句式聚类，作用未稳定超过匹配对照')
                if fraction<.95:failures.append('实际BF16范数对照合格率不足95%')
                if ratio>.8:failures.append('作用预测未稳定优于无变化/全局均值强基线')
                effect_rows.append({'candidate':candidate['id'],'factor':candidate['factor'],'module':candidate['module'],'block':candidate['block'],'method':method,'dose':dose,
                    'n':768,'mean_target_margin_loss':float(np.mean(losses)),'mean_control_margin_loss':float(np.mean(nulls)),'matched_excess':excess,
                    'control_valid_fraction':fraction,'prediction_mse':float(np.mean(errors)),'nochange_mse':float(np.mean(nochange)),'global_effect_mse':float(np.mean(global_errors)),
                    'prediction_to_best_baseline':ratio,'scope_use_supported':supported,'effect_prediction_supported':bool(ratio<=.8 and gate['passed']),
                    'argmax_changed':int(flip),'first_correct_to_wrong':int(lost),'first_wrong_to_correct':int(gained),'other_factor_correct_to_wrong':int(other_lost),'mean_KL':float(np.mean(kl)),
                    'failures':failures,'counterexamples':sorted(counter,key=lambda r:r['matched_excess'])[:3]})
    capability=C.read(C.OUT/'capability.json')['rows'];capstats={}
    for candidate in selection['candidates']:
        base=np.array([np.argmax(r['scores']['base'])==r['gold'] for r in capability]);edit=np.array([np.argmax(r['scores'][candidate['id']])==r['gold'] for r in capability])
        capstats[candidate['id']]={'n':len(base),'base_correct':int(base.sum()),'edited_correct':int(edit.sum()),'lost':int((base&~edit).sum()),'gained':int((~base&edit).sum()),'scope':'historical keyed first-letter choice, descriptive'}
    curve=[]
    for index in range(74):
        errors=mse[:,index].mean(0)
        curve.append({'state_index':index,'label':state_name(index),'language_accuracy':float(probes[:,index,0].mean()),'color_accuracy':float(probes[:,index,1].mean()),
            'state_prediction_mse':dict(zip(predictions,errors.tolist())),
            'eight_over_four':float(errors[3]/errors[2]) if errors[2]>1e-20 else None,
            'center_accuracies':dict(zip(center_names,center_acc[:,:,index].mean(0).tolist()))})
    worst=np.argsort(gain)[:6];examples=[{'family':int(f),'gain':float(gain[f]),'fixed_accuracy':float(center_acc[f,0,primary_index]),'corrected_accuracy':float(center_acc[f,2,primary_index]),'prompt':rows[int(f)*8]['prompt']} for f in worst]
    digest=C.save_npz('analysis-arrays.npz',evaluation_beta=observed.astype('<f4'),module_coefficients=module_coef.astype('<f4'),
        center_accuracy=center_acc.astype('<f4'),probe_accuracy=probes.astype('<f4'),state_prediction_mse=mse.astype('<f4'))
    report={'schema':'state_formation_report.v1','plan_sha256':C.sha(C.OUT/'plan.json'),'selection_sha256':C.sha(C.OUT/'selection.json'),
        'effects_frozen_sha256':C.sha(C.OUT/'effects-frozen.json'),'arrays_sha256':digest,
        'families':{'discovery':256,'confirm':768},'inputs':8192,'behavior':gate,'center_rows':center_rows,'center_primary_supported':center_pass,
        'formation':formation,'curves':curve,'effects':effect_rows,'capability':capstats,'counterexamples':examples,
        'negative_controls':{'maximum_future_factor_relative_l2':control_max,'native_residual_additions_verified':True,'all_768_restorations_bitwise':True,'identity_and_zero_dose_replay_families_per_stage':16},
        'limitations':p['limitations']+['center selection uses decoded color; oracle comparison explicitly uses supplied color','center corrections fitted using both discovery colors, not unseen-color training',
            'pre/post formation accuracies use separate decoders; same-post-decoder signed contrasts are also shown',
            'source masking changes attention normalization and all heads; count matching is not attention-mass matching',
            'component ablation is broad, not a feature-specific or single-neuron edit','single-token color choices and limited keyed side-effect checks','no independent-node replication'],
        'formal_evidence_promotion':False}
    C.save('report.json',report);C.log('analysis_complete',768,768,start,center_primary_supported=center_pass,behavior_passed=gate['passed'])

if __name__=='__main__':analyze()
