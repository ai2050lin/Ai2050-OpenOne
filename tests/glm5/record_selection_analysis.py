"""Frozen discovery selection, family nulls, held-out readout and reporting."""
import argparse,itertools,time
import numpy as np
import record_selection_core_v3 as R
from record_selection_core_v3 import C
from state_formation_math import bootstrap
from record_selection_collect import eligible

def admission(split):
    rows=C.read(C.OUT/(split+'-materials.json'))['rows']
    return np.array(eligible(rows,C.read(C.OUT/(split+'-behavior.json'))),bool)

def freeze():
    p=C.checked();cal=C.read(C.OUT/'calibration-full-behavior.json');assert not list(C.OUT.glob('discovery-[0-9]*.npz'))
    names=['tests/glm5/record_selection_analysis.py','tests/glm5/record_selection_causal.py','tests/glm5/record_selection_collect.py','tests/glm5/state_formation_math.py']
    C.save('analysis-plan.json',{'schema':'record_selection_analysis_plan.v1','base_plan_sha256':C.sha(C.OUT/'plan.json'),
        'sources':{n:C.sha(C.ROOT/n) for n in names},'candidate_head_blocks':list(range(36)),
        'calibration_sha256':C.sha(C.OUT/'calibration-full-behavior.json'),'whole_discovery_gate_passed':cal['passed'],
        'scope_amendment':'Original full-scope pilot and expanded calibration failed. Freeze restricted conditional candidate study before any native collection. Exclude whole paired families touching a failed cell, in both splits, without re-ranking on confirm. Full-scope semantic claim remains failed. Failed fields archived but excluded from feature summaries/prediction metrics. Causal failures retained as diagnostic only.',
        'discovery_eligible_families':[i for i,k in enumerate(eligible(C.read(C.OUT/'discovery-materials.json')['rows'],cal)) if k],
        'candidate_unit_blocks':list(range(12,27)),'discovery_train_frames':[0,1,2,3],'discovery_validation_frames':[4,5,6,7],
        'selection_head':'per-block top2 of min(position-specific valid-minus-old QK mass) on train; choose block by same validation min',
        'selection_units':'per-block top32 positive paired source-choice DLA contrasts on train; choose block by validation sum',
        'negative_control':'1024 family-level Rademacher swaps; absolute self-normalized statistic; max over train-selected candidate groups on independent validation',
        'predictor':'fixed standardized ridge lambda1 on actual native head contexts, physical-record mass differences and MLP group readout; no gold source index as predictor input',
        'baselines':['global mean by query language','last-record number always2','first-record number always1','explicit status parser','32-cell mean given parsed experimental factors'],
        'prediction_gate':'family-paired template CI of structural MSE minus native MSE >0; descriptive if any behavior gate fails',
        'controls':'four same-layer equal-size groups chosen by discovery diagonal write-energy matching; one realized-residual-norm random direction for each primary full-dose edit',
        'effect_gate':p['effect_gate'],'effect_alpha':.0125,'interventions':'final-position head o_proj input / MLP down_proj input; doses .5,1; valid/obsolete direct sources for selected heads',
        'failure_action':'no semantic promotion if any frozen behavior gate fails; raw observations and counterexamples retained',
        'scope':'valid-record number task, not color binding or independent-node scientific replication'})
    export_weights(p)

def export_weights(p):
    import torch
    torch.set_num_threads(4)
    from safetensors import safe_open
    index=C.read(C.MODEL/'model.safetensors.index.json')['weight_map']
    def tensor(name):
        with safe_open(C.MODEL/index[name],framework='pt',device='cpu') as f:return f.get_tensor(name)
    embedding=tensor('model.embed_tokens.weight');w=(embedding[p['record_ids']['1']].float()-embedding[p['record_ids']['2']].float());del embedding
    heads=[];units=[];hnorm=[];unorm=[]
    for b in range(36):
        wo=tensor(f'model.layers.{b}.self_attn.o_proj.weight').float();down=tensor(f'model.layers.{b}.mlp.down_proj.weight').float()
        heads.append((w@wo).reshape(32,128).numpy());units.append((w@down).numpy())
        hnorm.append(wo.reshape(2560,32,128).square().sum((0,2)).numpy()/128);unorm.append(down.square().sum(0).numpy())
        del wo,down
    C.save_npz('write-weights.npz',heads=np.stack(heads),units=np.stack(units),head_norm=np.stack(hnorm),unit_norm=np.stack(unorm))
    C.save('weights.json',{'schema':'record_selection_weights.v1','sha256':C.sha(C.OUT/'write-weights.npz'),
        'definition':'FP32 dot of actual o_proj/down_proj columns with embedding[1]-embedding[2], before final RMSNorm and downstream layers',
        'scope':'exact local linear write reader; not actual final causal contribution'})

def checked():
    p=C.checked();a=C.read(C.OUT/'analysis-plan.json')
    for name,digest in a['sources'].items():assert C.sha(C.ROOT/name)==digest,name
    assert C.sha(C.OUT/'write-weights.npz')==C.read(C.OUT/'weights.json')['sha256'];return p,a

def null_test(x,draws=1024,seed=C.SEED):
    x=np.asarray(x,float);n=len(x);x=np.divide(x,np.sqrt(np.mean(x*x,axis=0)),out=np.zeros_like(x),where=np.mean(x*x,axis=0)>1e-20)
    observed=np.sqrt(n)*np.abs(x.mean(0));rng=np.random.default_rng(seed)
    null=np.max(np.abs(rng.choice([-1.,1.],size=(draws,n))@x/np.sqrt(n)),axis=1)
    p=(1+(null[:,None]>=observed).sum(0))/(draws+1)
    return p,observed,null

def native(split,f,i):
    meta=C.read(C.OUT/f'{split}-{f:04d}.json')['native'][i];path=C.OUT/meta['path'];assert C.sha(path)==meta['sha256'];return np.load(path)

def summaries(split):
    p,a=checked();rows=C.read(C.OUT/(split+'-materials.json'))['rows'];n=len(rows)//8;w=np.load(C.OUT/'write-weights.npz')
    keep=admission(split);route=np.zeros((n,2,36,32),np.float32);unit=np.zeros((n,36,9728),np.float32);unit_raw=np.zeros_like(unit)
    head_energy=np.zeros((36,32));unit_energy=np.zeros((36,9728));max_error=0.;start=time.monotonic()
    for f in range(n):
        if not keep[f]:continue
        for i,row in enumerate(rows[f*8:(f+1)*8]):
            with native(split,f,i) as z:
                masses=z['source_mass'];h=C.unbits(z['head_context']);u=C.unbits(z['units']);max_error=max(max_error,float(z['qkv_relative_error'].max()))
                head_energy+=(h*h).sum(-1)*w['head_norm'];unit_energy+=u*u*w['unit_norm']
                if row['conflict']:
                    position=0 if row['valid_position']==-1 else 1;sign=-row['valid_position']
                    route[f,position]+=(masses[:,:,position]-masses[:,:,1-position])/2
                    unit_raw[f]+=sign*u/4;unit[f]+=sign*u*w['units']/4
        if (f+1)%16==0:C.log('summarize_'+split,f+1,n,start)
    return dict(route=route,unit=unit,unit_raw=unit_raw,head_energy=head_energy/(8*keep.sum()),unit_energy=unit_energy/(8*keep.sum()),max_qkv_error=max_error)

def controls(indices,energy,count,seed):
    rng=np.random.default_rng(seed);available=np.array([i for i in range(len(energy)) if i not in indices]);size=len(indices);target=float(energy[indices].sum())
    sets=list(itertools.combinations(available.tolist(),size)) if size==2 else [tuple(sorted(rng.choice(available,size,replace=False).tolist())) for _ in range(512)]
    sets=list(set(sets));sets.sort(key=lambda ids:(abs(float(energy[list(ids)].sum())-target)/max(target,1e-12),ids))
    return [{'indices':list(v),'discovery_diagonal_energy_ratio':float(energy[list(v)].sum()/max(target,1e-12))} for v in sets[:count]]

def row_features(z,candidates,w,row):
    head,unit=candidates;h=C.unbits(z['head_context'])[head['block']];u=C.unbits(z['units'])[unit['block']];mass=z['source_mass'][head['block']]
    ids=head['indices'];head_values=(h[ids]*w['heads'][head['block'],ids]).sum(-1);deltas=mass[ids,0]-mass[ids,1]
    value=float(u[unit['indices']]@w['units'][unit['block'],unit['indices']])
    return np.array([*head_values,*deltas,value,row['language'],row['conflict']],float)

def cell(row):return ((row['query']==1)*16+(row['language']==1)*8+(row['valid_position']==1)*4+(row['mapping']==1)*2+row['conflict'])

def ridge_fit(x,y):
    mu=x.mean(0);scale=x.std(0);scale[scale<1e-8]=1.;v=(x-mu)/scale;v=np.column_stack([np.ones(len(v)),v]);penalty=np.eye(v.shape[1]);penalty[0,0]=0
    return mu,scale,np.linalg.solve(v.T@v+penalty,v.T@y)

def ridge_predict(x,mu,scale,coef):return np.column_stack([np.ones(len(x)),(x-mu)/scale])@coef

def select():
    p,a=checked();assert not list(C.OUT.glob('confirm-[0-9]*.npz'))
    rows=C.read(C.OUT/'discovery-materials.json')['rows'];frames=np.array([rows[f*8]['frame'] for f in range(128)]);keep=admission('discovery');train=(frames<4)&keep;val=(frames>=4)&keep
    assert np.where(keep)[0].tolist()==a['discovery_eligible_families'],'Behavior changed under capture'
    s=summaries('discovery');head_candidates=[];unit_candidates=[]
    for b in a['candidate_head_blocks']:
        rating=s['route'][train,:,b].mean(0).min(0);ids=np.argsort(-rating,kind='stable')[:2].tolist()
        measure=s['route'][val,:,b][:,:,ids].mean(2)
        head_candidates.append({'kind':'heads','module':'attention','block':b,'indices':ids,'validation_score':float(measure.mean(0).min()),
            'validation_by_position':measure.mean(0).tolist()})
    for b in a['candidate_unit_blocks']:
        ids=np.argsort(-s['unit'][train,b].mean(0),kind='stable')[:32].tolist()
        unit_candidates.append({'kind':'units','module':'mlp','block':b,'indices':ids,'validation_score':float(s['unit'][val,b][:,ids].sum(1).mean())})
    head=max(head_candidates,key=lambda x:(x['validation_score'],-x['block']));unit=max(unit_candidates,key=lambda x:(x['validation_score'],-x['block']))
    hv=np.column_stack([s['route'][val,:,c['block']][:,:,c['indices']].mean((1,2)) for c in head_candidates])
    uv=np.column_stack([s['unit'][val,c['block']][:,c['indices']].sum(1) for c in unit_candidates])
    hp,ht,hn=null_test(hv);up,ut,un=null_test(uv)
    candidates=[]
    for group,allgroups,pvalues,energy in [(head,head_candidates,hp,s['head_energy']),(unit,unit_candidates,up,s['unit_energy'])]:
        index=allgroups.index(group);group={**group,'id':f"{group['kind']}-B{group['block']}",'validation_maxT_p':float(pvalues[index]),
            'discovery_support':bool(pvalues[index]<=.0125 and group['validation_score']>0)}
        group['controls']=controls(group['indices'],energy[group['block']],4,C.SEED+group['block']);candidates.append(group)
    C.save_npz('discovery-atlas.npz',head_validity_by_position=s['route'][keep].mean(0),unit_choice_contrast=s['unit_raw'][keep].mean(0),unit_dla_contrast=s['unit'][keep].mean(0),
        head_null_max=hn,unit_null_max=un,head_null_observed=ht,unit_null_observed=ut)
    w=np.load(C.OUT/'write-weights.npz');x=[];y=[];codes=[];languages=[]
    for f in range(128):
        if not keep[f]:continue
        scores=C.read(C.OUT/f'discovery-{f:04d}.json')['scores']
        for i,row in enumerate(rows[f*8:(f+1)*8]):
            with native('discovery',f,i) as z:x.append(row_features(z,candidates,w,row))
            y.append(scores[i]['record_margin']);codes.append(cell(row));languages.append(row['language'])
    x=np.array(x);y=np.array(y);mu,scale,coef=ridge_fit(x,y);cells=np.array([y[np.array(codes)==i].mean() for i in range(32)])
    globalmean=np.array([y[np.array(languages)==l].mean() for l in [-1,1]])
    C.save_npz('predictor.npz',mu=mu,scale=scale,coef=coef,cells=cells,global_mean=globalmean)
    C.save('selection.json',{'schema':'record_selection_selection.v1','plan_sha256':C.sha(C.OUT/'plan.json'),'candidates':candidates,
        'all_head_candidates':head_candidates,'all_unit_candidates':unit_candidates,'confirm_unopened':True,'predictor_sha256':C.sha(C.OUT/'predictor.npz'),
        'semantic_discovery_admitted':False,'eligible_families':int(keep.sum()),'scope':'conditional candidates only; failed full-scope discovery behavior cannot be repaired by this subset',
        'discovery_atlas_sha256':C.sha(C.OUT/'discovery-atlas.npz'),'max_qkv_error':s['max_qkv_error'],
        'exposure':'unit/head ranking uses frames0-3; group/block choice uses frames4-7; maxT null on validation groups; confirm untouched'})
    print(candidates,flush=True)

def report():
    p,a=checked();selection=C.read(C.OUT/'selection.json');candidates=selection['candidates'];gate=C.read(C.OUT/'confirm-behavior.json')
    assert selection['predictor_sha256']==C.sha(C.OUT/'predictor.npz');pred=np.load(C.OUT/'predictor.npz');w=np.load(C.OUT/'write-weights.npz')
    rows=C.read(C.OUT/'confirm-materials.json')['rows'];s=summaries('confirm');frames=np.array([rows[f*8]['frame'] for f in range(384)])
    keep=admission('confirm');x=[];y=[];truth=[];codes=[];languages=[];failed=[];start=time.monotonic()
    for f in range(384):
        scores=C.read(C.OUT/f'confirm-{f:04d}.json')['scores']
        for i,row in enumerate(rows[f*8:(f+1)*8]):
            if keep[f]:
                with native('confirm',f,i) as z:x.append(row_features(z,candidates,w,row))
            else:x.append(np.full(7,np.nan))
            y.append(scores[i]['record_margin']);truth.append(-row['valid_position']);codes.append(cell(row));languages.append(row['language'])
            if not scores[i]['strict_color_correct']:failed.append({'sample':row,'score':scores[i]})
    x=np.array(x);y=np.array(y);truth=np.array(truth);actual=ridge_predict(x,pred['mu'],pred['scale'],pred['coef'])
    predictions={'global_mean':pred['global_mean'][(np.array(languages)==1).astype(int)],'parsed_32cell':pred['cells'][codes],'native_group_readout':actual}
    errors={k:((v-y)**2).reshape(384,8).mean(1)[keep] for k,v in predictions.items()};metric_frames=frames[keep];rowkeep=np.repeat(keep,8)
    baseline={'mse':{k:float(v.mean()) for k,v in errors.items()},'native_over_structural':float(errors['native_group_readout'].mean()/errors['parsed_32cell'].mean()),
        'native_gain_ci':bootstrap(errors['parsed_32cell']-errors['native_group_readout'],metric_frames,alpha=.0125),'eligible_families':int(keep.sum()),
        'gold_accuracy':{'always_first':float(np.mean(truth==1)),'always_last':float(np.mean(truth==-1)),'explicit_status_parser':1.,
            'native_group_readout':float(np.mean(np.where(actual[rowkeep]>=0,1,-1)==truth[rowkeep]))},'scope':'Conditional eligible subset, not whole-scope validation. Native predictor requires actual original-model activations; structural parser is an externally supplied symbolic rule, not a discovered neural mechanism'}
    observations=[]
    for c in candidates:
        value=s['route'][:,:,c['block']][:,:,c['indices']].mean((1,2)) if c['kind']=='heads' else s['unit'][:,c['block']][:,c['indices']].sum(1)
        value=value[keep];pv,stat,null=null_test(value[:,None],seed=C.SEED+9)
        observations.append({'id':c['id'],'mean':float(value.mean()),'cluster_ci':bootstrap(value,metric_frames,alpha=.0125),'two_sided_family_swap_p':float(pv[0]),
            'head_positions':s['route'][keep,:,c['block']][:,:,c['indices']].mean((0,2)).tolist() if c['kind']=='heads' else None})
    causal=[C.read(C.OUT/f'causal-{f:04d}.json') for f in range(384)];effects=[]
    for c in candidates:
        for method,dose in [('ablation',.5),('ablation',1.),*([('valid_source',1.),('obsolete_source',1.)] if c['kind']=='heads' else [])]:
            losses=[];control=[];norm=[];interaction=[];lost=gained=0;kl=[];examples=[]
            for f,item in enumerate(causal):
                if not keep[f]:continue
                rec=next(z for z in item['records'] if z['candidate']==c['id'] and z['method']==method and z['dose']==dose)
                loss=item['baseline']['target_margin']-rec['edited']['target_margin'];losses.append(loss);kl.append(rec['edited']['KL_edit_base'])
                lost+=bool(item['baseline']['first_top1'] and not rec['edited']['first_top1']);gained+=bool(not item['baseline']['first_top1'] and rec['edited']['first_top1'])
                if method=='ablation' and dose==1:
                    control.append(float(np.mean([item['baseline']['target_margin']-z['target_margin'] for z in rec['group_controls']])))
                    norm.append(item['baseline']['target_margin']-rec['norm_control']['target_margin'])
                    interaction.append(loss-(item['agreement_baseline']['target_margin']-item['agreement_edits'][c['id']]['target_margin']))
                examples.append({'family':f,'loss':loss,'prompt':item['sample']['prompt']})
            excess=bootstrap(np.array(losses)-control,metric_frames,alpha=.0125) if control else None
            normex=bootstrap(np.array(losses)-norm,metric_frames,alpha=.0125) if norm else None
            support=bool(gate['passed'] and selection['semantic_discovery_admitted'] and c['discovery_support'] and control and np.mean(losses)>=.2 and excess['frame_interval'][0]>0 and normex['frame_interval'][0]>0)
            effects.append({'candidate':c['id'],'method':method,'dose':dose,'n':int(keep.sum()),'mean_margin_loss':float(np.mean(losses)),
                'native_group_control_loss':float(np.mean(control)) if control else None,'matched_norm_control_loss':float(np.mean(norm)) if norm else None,
                'excess_native':excess,'excess_norm':normex,'conflict_minus_agreement':bootstrap(interaction,metric_frames,alpha=.0125) if interaction else None,
                'mean_KL':float(np.mean(kl)),'first_correct_to_wrong':int(lost),'first_wrong_to_correct':int(gained),'scope_use_supported':support,
                'failures':['full-scope discovery behavior failed; conditional diagnostic only']+(['behavior gate failed'] if not gate['passed'] else [])+(['discovery selectivity maxT gate failed'] if not c['discovery_support'] else [])+
                    (['mean loss below 0.2'] if np.mean(losses)<.2 else [])+(['native-group control not exceeded'] if excess and excess['frame_interval'][0]<=0 else [])+
                    (['matched-norm control not exceeded'] if normex and normex['frame_interval'][0]<=0 else [])+
                    (['source masks have no matched-norm/control sufficiency test; descriptive'] if not control else []),
                'counterexamples':sorted(examples,key=lambda r:r['loss'])[:3]})
    C.save_npz('confirm-atlas.npz',head_validity_by_position=s['route'][keep].mean(0),unit_choice_contrast=s['unit_raw'][keep].mean(0),unit_dla_contrast=s['unit'][keep].mean(0),
        prediction_truth=y,prediction_native=actual,mse_global=errors['global_mean'],mse_structural=errors['parsed_32cell'],mse_native=errors['native_group_readout'])
    C.save('report.json',{'schema':'record_selection_report.v1','behavior':gate,'candidates':candidates,'observations':observations,'baselines':baseline,
        'effects':effects,'failures':failed,'families':p['families'],'unique_inputs':p['unique_inputs'],'plan_sha256':C.sha(C.OUT/'plan.json'),
        'admission':C.read(C.OUT/'confirm-admission.json'),'discovery_calibration':C.read(C.OUT/'calibration-full-behavior.json'),
        'selection_sha256':C.sha(C.OUT/'selection.json'),'atlas_sha256':C.sha(C.OUT/'confirm-atlas.npz'),'max_qkv_error':s['max_qkv_error'],
        'limitations':p['limitations']+['source QK decomposition is FP32 diagnostic with measured BF16 SDPA error','same-layer diagonal-energy controls need not be semantically irrelevant',
            'head and MLP groups tested separately; no causal mediation or complete circuit closure','signed local write readout omits later nonlinearities','four query frames per language, sixteen record frames; correlated wording uncertainty remains'],
        'formal_evidence_promotion':False})
    C.log('report_complete',384,384,start)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','select','report']);args=parser.parse_args();globals()[args.stage]()
