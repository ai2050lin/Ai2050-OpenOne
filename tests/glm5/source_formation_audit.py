"""Independent artifact, order, identity and metric checks. No evidence auto-promotion."""
import argparse,hashlib,json,sys,time,zipfile
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import source_formation_design as D
from source_binding_native_v2 import unbits

def audit():
    from source_formation_native import verify
    from shared.dist_protocol.natural_register_runner import snapshot
    from transformers import AutoTokenizer
    verify();assert D.read(D.OUT/'campaign-complete.json')['completed'];checks=[];timer=time.monotonic()
    def check(name,ok):
        checks.append({'check':name,'passed':bool(ok)})
        assert ok,name
    def when(obj):return obj['provenance']['created_at']
    plan=D.read(D.OUT/'plan.json')
    revision=D.read(D.OUT/'internal-execution-v2.json')
    with zipfile.ZipFile(D.OUT/'pre-index-fix-sources.zip') as z:
        for path,sha in D.read(D.OUT/'internal-freeze.json')['code_sha256'].items():
            check('original frozen archive '+path,hashlib.sha256(z.read(path.replace('\\','/'))).hexdigest()==sha)
    for name in ['native-parameter-addendum','content-control-addendum','formation-error-split-addendum']:
        p=D.read(D.OUT/(name+'.json'))
        source={'native-parameter-addendum':'source_formation_parameters.py','content-control-addendum':'source_formation_content_controls.py','formation-error-split-addendum':'source_formation_error_split.py'}[name]
        check(name+' code',D.sha(D.ROOT/'tests/glm5'/source)==p['code_sha256'])
    runtime=D.read(D.OUT/'runtime-environment.json')
    for path,sha in runtime['library_files'].items():check('runtime code '+path,D.sha(path)==sha)
    complete=D.read(D.OUT/'campaign-complete.json')
    queue=D.read(D.OUT/'execution-queue-protocol.json')
    check('queue runner identity',D.sha(D.ROOT/'tests/glm5/source_formation_queue.py')==queue['queue_sha256'])
    check('campaign runner identity',D.sha(D.ROOT/'tests/glm5/source_formation_campaign.py')==queue['campaign_sha256'])
    check('retained discovery fit identity',revision['existing_fit_sha256']==D.sha(D.OUT/'qwen14/formation-fit.npz'))
    for model,path in D.MODELS.items():
        root=D.OUT/model;allrows=D.read(D.OUT/f'{model}-materials.json')['splits']
        check(model+' material hash',D.sha(D.OUT/f'{model}-materials.json')==plan['materials'][model])
        check(model+' model/tokenizer actual snapshot',snapshot(path)==plan['models'][model])
        namesets=[{name for r in rs for name in r['names']+r['sources']} for rs in allrows.values()]
        check(model+' disjoint name families',sum(map(len,namesets))==len(set.union(*namesets)))
        tok=AutoTokenizer.from_pretrained(path,local_files_only=True);cfg=D.read(path/'config.json');eos=cfg['eos_token_id'];eos=set(eos if isinstance(eos,list) else [eos])
        if (root/'discovery-behavior.json').exists() and model=='glm9':
            decision=D.read(D.OUT/'qwen14/replication-decision.json')
            check('GLM admitted after Qwen mechanism outcome',decision['glm_eligible'] and when(decision)<when(D.read(root/'discovery-behavior-v2-started.json')))
        for split,rows in allrows.items():
            check(model+'/'+split+' canonical sample count',len(rows)==12*D.COUNTS[split] and len({r['id'] for r in rows})==len(rows))
            label_recoverable=True;disjoint_spans=True
            for r in rows:
                question=r['content'].split('Question: ',1)[1]
                src=[i for i,x in enumerate(r['sources']) if x in question];ent=[i for i,x in enumerate(r['names']) if x in question]
                label_recoverable &= src==[r['source']] and ent==[r['entity']]
                spans=[set(r['records'][str(j)]['span']) for j in range(6)]
                disjoint_spans &= sum(map(len,spans))==len(set.union(*spans))
            check(model+'/'+split+' query relations recoverable from text',label_recoverable)
            check(model+'/'+split+' disjoint record token spans',disjoint_spans)
            if not (root/f'{split}-behavior.json').exists():
                check(model+'/'+split+' no unadmitted internal fields',not (root/f'{split}-native').exists())
                continue
            b=D.read(root/f'{split}-behavior.json');g=D.read(root/f'{split}-gate.json')
            check(model+'/'+split+' complete behavior identity',b['complete'] and g['behavior_sha256']==D.sha(root/f'{split}-behavior.json') and g['plan_sha256']==D.sha(D.OUT/'plan.json'))
            check(model+'/'+split+' complete gate recomputation',D.gate(rows,b['rows'])=={k:g[k] for k in D.gate(rows,b['rows'])})
            correct=True
            for r,a in zip(rows,b['rows']):
                decoded=tok.decode(a['generated_ids'],skip_special_tokens=True).strip()
                correct &= decoded==a['generated_text'] and (decoded==r['gold'])==a['strict_correct']
                correct &= (D.normalized(decoded)==D.normalized(r['gold']))==a['normalized_correct']
                correct &= (any(i in eos for i in a['generated_ids']))==a['terminated'] and len(a['generated_ids'])<=4
                correct &= int(np.argmax(a['scores']))==a['pred_record'] and (a['pred_record']==r['target_record'])==a['forced_correct']
                correct &= a['generated_ids'][0]==a['top1'] and (a['top1']==r['gold_id'])==a['top1_correct']
            check(model+'/'+split+' token decoding and six-choice scores',correct)
            mp=root/f'{split}-native-manifest.json'
            if not g['passed']:
                check(model+'/'+split+' stopped failed behavior',not (root/f'{split}-native').exists() and not mp.exists())
                continue
            check(model+'/'+split+' admitted full native manifest exists',mp.exists())
            manifest=D.read(mp);start=D.read(root/f'{split}-capture-started.json')
            check(model+'/'+split+' baseline precedes internal capture',when(g)<when(start) and start['admission']['gate_sha256']==D.sha(root/f'{split}-gate.json'))
            check(model+'/'+split+' natural replay exact',manifest['replay']['passed'] and manifest['replay']['choice_logits_max_difference']==0)
            for filename,info in manifest['files'].items():
                actual=root/f'{split}-native'/filename
                check(model+'/'+split+'/'+filename+' size and SHA256',actual.resolve().is_relative_to((root/f'{split}-native').resolve()) and actual.stat().st_size==info['bytes'] and D.sha(actual)==info['sha256'])
            D.log(model+'/'+split+'/audit-native',1,1,timer)
            fit=D.read(root/'formation-fit.json')
            check(model+' fitted coefficients unchanged',fit['fit_sha256']==D.sha(root/'formation-fit.npz'))
            check(model+'/'+split+' candidate before holdout capture',split=='discovery' or when(fit)<when(start))
            pred=D.read(root/f'{split}-formation.json');store=np.load(root/f'{split}-formation-predictions.npz')
            check(model+'/'+split+' prediction artifact hash',pred['predictions_sha256']==D.sha(root/f'{split}-formation-predictions.npz'))
            for c in pred['candidates']:
                truth=store[c['id']+'_truth']
                for method,result in c['methods'].items():
                    x=store[c['id']+'_'+method];mse=float(np.square(truth-x).sum(1).mean())
                    check(model+'/'+split+'/'+c['id']+'/'+method+' direct MSE',np.isfinite(x).all() and np.isclose(mse,result['MSE'],rtol=1e-5,atol=1e-5))
                expected=c['methods']['native']['improvement']>=.1 and c['methods']['native']['MSE']<c['methods']['embedding']['MSE']
                check(model+'/'+split+'/'+c['id']+' frozen gate',c['passes']==expected)
                check(model+'/'+split+'/'+c['id']+' QKV/WO replay',c['methods']['full_native']['relative_error']<.02)
            store.close()
            if (root/f'{split}-update.json').exists():
                update=D.read(root/f'{split}-update.json');check(model+'/'+split+' recurrent final hash',update['final_sha256']==D.sha(root/f'{split}-update-final.npz'))
                for step in update['steps']:
                    check(model+'/'+split+f'/update{step["block"]} finite',all(np.isfinite(x['rollout_MSE']) for x in step['methods'].values()))
            if split!='discovery':
                co=D.read(root/f'{split}-causal.json')
                check(model+'/'+split+' causal final hash',co['final_sha256']==D.sha(root/f'{split}-causal-final.npy'))
                check(model+'/'+split+' causal natural replay exact',co['natural_logits_replay_max_error']==0)
                for kind in ['head','mlp']:
                    errors=[];bounded=True
                    for x in co['local']:
                        if x['kind']==kind:
                            bounded &= all(0<=v<=1 for v in x['scale'])
                            errors.extend((np.abs(np.array(x['actual_norm'])-x['target_norm'])/np.maximum(x['target_norm'],1e-8)).tolist())
                    frac=float(np.mean(np.array(errors)<=.05))
                    check(model+'/'+split+f'/{kind}/all damping bounded',bounded)
                    check(model+'/'+split+'/'+kind+' matching gate recomputed',np.isclose(frac,co['norm_matching'][kind]['fraction_within_5pct']) and co['norm_matching'][kind]['admitted']==(frac>=.99))
                check(model+'/'+split+' update true-output replay',D.read(root/f'{split}-update-output.json')['actual_final_logits_replay_exact'])
                err=D.read(root/f'{split}-error-split.json')
                check(model+'/'+split+' diagnostic code identity',err['addendum_sha256']==D.sha(D.OUT/'formation-error-split-addendum.json'))
                check(model+'/'+split+' bilinear score identity',all(c['algebra_identity_max_abs_error']<1e-8 for c in err['results']))
                if split=='lockbox_frame' or model=='glm9':
                    check(model+'/'+split+' diagnostic recipe before heldout behavior',when(D.read(D.OUT/'formation-error-split-addendum.json'))<when(D.read(root/f'{split}-behavior-v2-started.json')))
            for suffix in ['native-parameters','content-controls']:
                check(model+'/'+split+'/'+suffix+' secondary complete',(root/f'{split}-{suffix}.json').exists())
            pp=D.read(root/f'{split}-native-parameters.json')
            check(model+'/'+split+' original parameter full replay',pp['calibration_passed'])
            for suffix,source in [('matching-numerics','source_formation_matching_diagnostic.py'),('native39-matching-numerics','source_formation_matching_native39.py'),('matching-coordinate-lineage','source_formation_matching_lineage.py')]:
                path=root/f'{split}-{suffix}.json'
                if path.exists():
                    extra=D.read(path);check(model+'/'+split+'/'+suffix+' posthoc source hash',extra['source_sha256']==D.sha(D.ROOT/'tests/glm5'/source))
                    if 'arrays_sha256' in extra:check(model+'/'+split+'/'+suffix+' array hash',extra['arrays_sha256']==D.sha(root/f'{split}-{suffix}.npz'))
            for kind in ['native-parameter','content-control']:
                fit2=D.read(root/f'{kind}-fit.json')
                check(model+'/'+split+'/'+kind+' fit identity',fit2['fit_sha256']==D.sha(root/f'{kind}-fit.npz'))
                if model=='qwen14' and split=='confirm' and kind=='content-control':
                    check(model+'/'+split+'/'+kind+' fit before completed heldout artifact (capture already started)',when(fit2)<when(manifest))
                else:
                    check(model+'/'+split+'/'+kind+' fit before heldout capture',split=='discovery' or when(fit2)<when(start))
        D.log(model+'/audit',1,1,timer)
    D.save('audit.json',{'passed':True,'count':len(checks),'checks':checks,'formal_evidence_promotion':0,
        'scope':'Artifact identity, execution gates, numerical calibrations, metric recomputation and exposure order; not independent-node replication or scientific proof.'})
    print('AUDIT_PASSED',len(checks),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['audit']);ap.parse_args();audit()
