"""Primitive-only fits and prospective unseen-composition state predictions."""
import argparse
import gc
import time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
import b36_composition14 as B
import b36_composition14_collect as C
from branch_reuse14_analysis import fit,predict,lexical
from recursive_branch14_analysis import inference

METHODS=['selectedA','hidden','hidden_prefix','lexical','embedding']


def load(split,all_rows=False):
    meta=B.read(B.OUT/split/'collection.json');allrows=B.read(B.OUT/'materials.json')['rows']
    rows=[allrows[i] for i in meta['row_indices']];assert [r['id'] for r in rows]==meta['row_ids']
    if not all_rows and split in ['confirm','format','order']:
        scope=B.read(B.OUT/(split+'-admissibility.json'));assert scope['collection_sha256']==B.file_sha(B.OUT/split/'collection.json')
        ids=set(scope['eligible_row_ids']);meta=dict(meta,active_indices=[i for i,r in enumerate(rows) if r['id'] in ids]);rows=[r for r in rows if r['id'] in ids]
    return rows,meta


def field(split,block,key,readout=1,units=None):
    _,meta=load(split);values=[]
    for item in meta['chunks']:
        with np.load(Path(meta.get('native_origin',B.OUT/split))/item['name']/f'B{block}.npz') as data:
            value=data[key] if readout==1 else data[key+'_content']
            if units is not None:value=value[:,units if readout==1 and block in [36,39] else [B.tracked().index(u) for u in units]]
            values.append(B.unbits(value))
    out=np.concatenate(values);return out[meta.get('active_indices',list(range(len(out))))].astype(np.float64)


def state(split,boundary,readout=1):
    _,meta=load(split);values=[]
    for item in meta['chunks']:
        with np.load(Path(meta.get('native_origin',B.OUT/split))/item['name']/'states.npz') as data:values.append(B.unbits(data['hidden'][:,boundary,readout]))
    out=np.concatenate(values);return out[meta.get('active_indices',list(range(len(out))))].astype(np.float64)


def meanstate(split,key):
    _,meta=load(split);values=[]
    for item in meta['chunks']:
        with np.load(Path(meta.get('native_origin',B.OUT/split))/item['name']/'states.npz') as data:values.append(data[key])
    out=np.concatenate(values);return out[meta.get('active_indices',list(range(len(out))))].astype(np.float64)


def targets(split,readout=1):
    return np.stack([field(split,39,f,readout,B.roots()) for f in 'GVA'],-1).reshape(-1,len(B.roots())*3)


def inputs(split,readout):
    rows,_=load(split);h=state(split,37,readout)
    # prefix_h37 includes question/options. It is available at answer readout only.
    result=dict(selectedA=field(split,36,'A',readout,B.group()),hidden=h,lexical=lexical(rows,readout),embedding=meanstate(split,'prefix_embedding'))
    if readout==1:result['hidden_prefix']=np.column_stack([h,meanstate(split,'prefix_h37')])
    return result


def get(name):
    with np.load(B.OUT/'fits'/(name+'.npz')) as f:return {k:f[k] for k in f.files}


def put(name,model,x):
    dest=B.OUT/'fits';dest.mkdir(exist_ok=True);p=dest/(name+'.npz');assert not p.exists()
    compact={k:v.astype(np.float32) if np.issubdtype(v.dtype,np.floating) else v for k,v in model.items()}
    check_x=x[:min(32,len(x))]
    error=float(np.max(np.abs(predict(model,check_x)-predict(compact,check_x))/model['ys']))
    assert error<=2e-4,(name,error)
    np.savez_compressed(p,**compact)
    return dict(name=name,stored_dtype='float32',standardized_prediction_max_error_first32=error,limit=2e-4)


def grouped_ratio(rows,loss,base):
    families=sorted({r['family'] for r in rows})
    aa=np.array([np.mean(loss[[i for i,r in enumerate(rows) if r['family']==f]]) for f in families])
    bb=np.array([np.mean(base[[i for i,r in enumerate(rows) if r['family']==f]]) for f in families])
    rng=np.random.default_rng(B.SEED);ix=rng.integers(len(families),size=(2000,len(families)))
    boot=aa[ix].mean(1)/np.maximum(bb[ix].mean(1),1e-14)
    return dict(ratio=float(aa.mean()/max(bb.mean(),1e-14)),ci95=np.quantile(boot,[.025,.975]).tolist(),families=len(families))


def plan():
    B.configure();C.admit();assert not (B.OUT/'analysis-plan.json').exists()
    from scipy.stats import norm
    from tools.power_analysis import check_phase
    binary_power=check_phase(.10,160,p_d=.15);assert binary_power['pass']
    B.save('analysis-plan.json',dict(source_sha256=B.file_sha(__file__),pure_fit_helper_sha256=B.file_sha(B.ROOT/'tests/glm5/branch_reuse14_analysis.py'),
        trial_sha256=B.file_sha(B.OUT/'trial-plan.json'),group=B.group(),roots=B.roots(),matched_group=B.matched(),
        fits='Only48 discovery families per branch, primitives A/B in v3 admitted source strata: number/program, number/archive, label/archive. Three family-grouped folds select ridge alpha1/10/100/1000. Pilot excluded. Fit float64, frozen coefficients float32 after first32 prediction check <=2e-4 target SD. No confirm/format/order fitting.',
        natural_prediction=dict(targets=['B39 fixed roots G/V/A at content and answer','raw h40 minus raw h37 at answer'],
            inputs='B36 selected32 A; full h37; h37 plus mean historical h37 prefix (answer only); token hashes; prefix mean embedding.',
            available_information='All source h37 components are after B36, before B37. No actual downstream state or answer label is input to these fits. Prefix mean never used at earlier content readout.',
            gate='Each eligible branch x depth x source: hidden or hidden_prefix root MSE <=.9 selectedA and <=.9 lexical, and <=.5 mean. State completion claim is conditional, with unequal input budgets.',
            incremental_context='hidden_prefix MSE <=.9 hidden in each branch x depth, otherwise no demonstrated gain from this particular prefix mean.',
            future_update='h40-h37 error <=.9 zero-update and <=.9 discovery-mean update; predictions compose with the same measured h37 for output. This is conditional late-state fidelity.',
            output='Whole-vocabulary KL teacher||predicted; compare all predictors, mean update and identity h37 carry. Do not call this text-only atlas fidelity.'),
        interventions=dict(families_per_branch=80,inputs='All behavior-eligible confirmation families; ABC variants, and separate reused-family XYZ format panel.',
            binary_power=dict(**binary_power,alpha=.05,scope='One prespecified archive/depth1/plain-title item per family, pooled160 families. Target10 percentage-point paired accuracy change with15% discordance; smaller branch effects not powered. Margin primary tests have separate continuous power.',
                warning='Planning effect is a detectable large change, not an estimate or a claim that group32 must cause it.'),
            primary_effect='B36 group0 minus same-layer matched32 group0 margin change, paired titles per family, branch x depth1/2/3 x source (12 tests, alpha=.05/12 including failed/missing tests); require negative difference and upper corrected CI<0. Report dose2, KL and strict accuracy separately.',
            complexity='For each branch, [delta_margin(program,d3)-delta_margin(archive,d3)] minus same difference at depth2; average titles within family. Expected negative, dz<=-.5, p<.025. Negative result does not prove absence of composition.',
            approximate_power_dz_point5_n80_alpha_point025=float(norm.cdf(.5*np.sqrt(80)-norm.ppf(1-.025/2))),
            controls='Frozen matched32, random direction and negative radial direction matched to actual BF16 post-B36 residual displacement, <=2% norm error. Frozen group, no-op and dose0/2. Both paths and scalar members remain distinct.',
            paths='For group0, freeze natural attention outputs, MLP outputs, both, or restore only B39 roots; same random-root restoration control. Final-token edits cannot change earlier prefix states.',
            native_math='Residual carry plus smooth FP32 RMS/QK-softmax/V/O/gated-MLP Jacobian evaluated on native baseline; compare full, attention-frozen, MLP-frozen and residual-only predictions. Not a BF16 rounding derivative.',
            counterfactual_gate='Full local predictor root-effect MSE <=.25 zero-effect and <=.9 residual-only in every branch/depth/source at dose0 and dose2. Root standardized effect RMS>=.05 or indeterminate. Final raw-state effect, output KL and exact path-clamp tests separately reported.',
            interpretation='A known native-computation derivative can explain local causal propagation without identifying shared semantic variables. Baseline natural late fields are permitted inputs for conditional counterfactual prediction, never edited downstream targets.'),
        complexity_limits='Depth is explicit operation composition, but variable lexical/operator state and answer margins are not automatically higher-complexity internal features. No automatic mechanism upgrade from larger depth effects.',
        title_control='Paired semantic-neutral technical/nontechnical cover titles have identical token length; target positions match. Modes share content except source cue; heldout wording frames differ from fitting.',
        multiple_testing='Fixed primary families of comparisons. All additional profiles, path shares and per-unit attributions descriptive unless separately specified.',
        order_panel='Same80 families per branch, opposite orders at two/three steps, program and plain title only; combine with matching main-confirm inputs. Two-step AB/BA has identical word multiset and identical option list. Three-step ABA/BAB changes operation counts and is a sequence contrast, not a pure order control. Reuse means no extra independent N. Strict behavior>=.9 per branch/depth/order before internal comparison.',
        stopping='No affected internal confirmation or intervention after behavior failure. Automatic predeclared branch/depth/source admission requires BOTH titles pass pilot and confirmation, recorded before internal results opened. Missing joint cells always make full joint claim fail. No post-hoc changing indices, thresholds, fit inputs or depth labels.'))


def admit():
    C.admit();p=B.read(B.OUT/'analysis-plan.json')
    assert p['source_sha256']==B.file_sha(__file__)
    assert p['pure_fit_helper_sha256']==B.file_sha(B.ROOT/'tests/glm5/branch_reuse14_analysis.py')
    assert p['trial_sha256']==B.file_sha(B.OUT/'trial-plan.json')
    if (B.OUT/'selection.json').exists():
        s=B.read(B.OUT/'selection.json')
        for f,sha in s['fits'].items():assert B.file_sha(B.OUT/'fits'/f)==sha
    return p


def select():
    B.configure();admit();rows,meta=load('discovery');assert meta['behavior']['passed']
    assert not (B.OUT/'selection.json').exists();start=time.monotonic();count=0;storage_checks=[]
    with threadpool_limits(limits=4):
      for readout in [0,1]:
        xx=inputs('discovery',readout);yy=targets('discovery',readout)
        for method,x in xx.items():
            storage_checks.append(put(f'roots-r{readout}-{method}',fit(x,yy,rows),x));count+=1;B.log('fit_primitive_models',count,14,start,readout=readout,method=method)
        if readout==1:
            update=state('discovery',40)-state('discovery',37)
            for method,x in xx.items():
                storage_checks.append(put('update-'+method,fit(x,update,rows),x));count+=1;B.log('fit_primitive_models',count,14,start,target='update',method=method)
        del xx,yy;gc.collect()
    # A same-layer root restoration control selected only by primitive response scale.
    a=field('discovery',39,'A');sd=a.std(0);rng=np.random.default_rng(B.SEED+39);used=set(B.roots());controls=[]
    for u in B.roots():
        ranked=np.argsort(np.abs(np.log(np.maximum(sd,1e-7))-np.log(max(sd[u],1e-7))))
        pool=[int(j) for j in ranked if j not in used][:64];j=int(rng.choice(pool));used.add(j);controls.append(j)
    B.save('selection.json',dict(kind='frozen_primitive_fits',fits={p.name:B.file_sha(p) for p in sorted((B.OUT/'fits').glob('*.npz'))},
        group=B.group(),matched_group=B.matched(),roots=B.roots(),root_control=controls,analysis_plan_sha256=B.file_sha(B.OUT/'analysis-plan.json'),
        rows=len(rows),families=len({r['family'] for r in rows}),seen_sequences=sorted({r['sequence'] for r in rows}),fit_storage_checks=storage_checks,source_sha256=B.file_sha(__file__)))


def eligible(split):
    rows,meta=load(split);assert rows,'No behavior-eligible scope'
    return rows,meta


def scope(split='confirm'):
    B.configure();admit();rows,meta=load(split,True);assert not (B.OUT/(split+'-admissibility.json')).exists()
    pilot=B.read(B.OUT/'pilot/collection.json')['behavior']['cells'];formal=meta['behavior']['cells'];admitted=[];strata=[]
    main=None if split=='confirm' else B.read(B.OUT/'confirm-admissibility.json')['eligible_strata']
    for key in sorted({(r['branch'],r['depth'],r['mode']) for r in rows}):
        p=[c for c in pilot if (c['branch'],c['depth'],c['mode'])==key];f=[c for c in formal if (c['branch'],c['depth'],c['mode'])==key]
        if split=='order':
            mainrows,mainmeta=load('confirm',True)
            pairs=[(r,s) for r,s in zip(rows,meta['scores']) if (r['branch'],r['depth'],r['mode'])==key]
            pairs += [(r,s) for r,s in zip(mainrows,mainmeta['scores']) if (r['branch'],r['depth'],r['mode'])==key and r['title']==0]
            for sequence in sorted({r['sequence'] for r,s in pairs}):
                rr=[(r,s) for r,s in pairs if r['sequence']==sequence]
                correct=sum(s[r['gold']]>max(v for j,v in enumerate(s) if j!=r['gold']) for r,s in rr)
                f.append(dict(sequence=sequence,n=len(rr),accuracy=correct/len(rr),passed=correct/len(rr)>=.9))
        passed=len(p)==2 and all(c['passed'] for c in p) and bool(f) and all(c['passed'] for c in f)
        if main is not None:passed=passed and list(key) in main
        if passed:admitted.append(list(key))
        strata.append(dict(branch=key[0],depth=key[1],mode=key[2],pilot=p,formal=f,passed=passed))
    ids=[r['id'] for r in rows if [r['branch'],r['depth'],r['mode']] in admitted]
    B.save(split+'-admissibility.json',dict(kind='behavior_only_predeclared_partial_admission',collection_sha256=B.file_sha(B.OUT/split/'collection.json'),
        eligible_strata=admitted,eligible_row_ids=ids,strata=strata,joint_passed=all(s['passed'] for s in strata),source_sha256=B.file_sha(__file__),
        rule='Declared before confirmation. Both pilot title cells and every formal title cell must pass; no internal response/selection used.'))


def analyze(split='confirm'):
    B.configure();admit();rows,_=eligible(split);results=[];saved={};start=time.monotonic()
    with threadpool_limits(limits=4):
      for readout in [0,1]:
        xx=inputs(split,readout);yy=targets(split,readout);base=get(f'roots-r{readout}-selectedA');scale=base['ys']
        pred={key:predict(get(f'roots-r{readout}-{key}'),x) for key,x in xx.items()};loss={key:np.mean(((p-yy)/scale)**2,1) for key,p in pred.items()}
        loss['mean']=np.mean(((base['ym']-yy)/scale)**2,1)
        for branch,depth,mode in sorted({(r['branch'],r['depth'],r['mode']) for r in rows}):
          ix=np.array([i for i,r in enumerate(rows) if (r['branch'],r['depth'],r['mode'])==(branch,depth,mode)]);rr=[rows[i] for i in ix]
          if not len(ix):continue
          for method in pred:
            ratios={name:grouped_ratio(rr,loss[method][ix],value[ix]) for name,value in loss.items() if name!=method}
            passed=method in ['hidden','hidden_prefix'] and ratios['selectedA']['ratio']<=.9 and ratios['lexical']['ratio']<=.9 and ratios['mean']['ratio']<=.5
            results.append(dict(target='roots',readout=readout,branch=branch,depth=depth,mode=mode,method=method,ratios=ratios,state_completion_passed=passed))
        saved.update({f'root-r{readout}-{name}':value.astype(np.float32) for name,value in pred.items()})
        if readout==1:
            yy=state(split,40)-state(split,37)
            base=get('update-selectedA');scale=base['ys'];loss={name:np.mean(((predict(get('update-'+name),x)-yy)/scale)**2,1) for name,x in xx.items()}
            loss['zero']=np.mean((yy/scale)**2,1);loss['mean']=np.mean(((base['ym']-yy)/scale)**2,1)
            for branch,depth,mode in sorted({(r['branch'],r['depth'],r['mode']) for r in rows}):
              ix=np.array([i for i,r in enumerate(rows) if (r['branch'],r['depth'],r['mode'])==(branch,depth,mode)]);rr=[rows[i] for i in ix]
              if not len(ix):continue
              for method in xx:
                ratios={name:grouped_ratio(rr,loss[method][ix],loss[name][ix]) for name in ['zero','mean']}
                results.append(dict(target='residual_update',readout=1,branch=branch,depth=depth,mode=mode,method=method,ratios=ratios,state_completion_passed=all(r['ratio']<=.9 for r in ratios.values())))
        B.log('unseen_composition_prediction',readout+1,2,start,split=split)
    B.arrays(B.OUT/f'{split}-root-predictions.npz',saved)
    B.save(f'{split}-state-report.json',dict(kind='real',results=results,selection_sha256=B.file_sha(B.OUT/'selection.json'),source_sha256=B.file_sha(__file__),
        warning='Full residual and prefix means have larger information budgets. This is conditional prediction from native late states, not extraction of the complete language algorithm.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['plan','select','analyze','scope']);p.add_argument('--split',default='confirm',choices=['confirm','format','order']);a=p.parse_args()
    globals()[a.stage](a.split) if a.stage in ['analyze','scope'] else globals()[a.stage]()
