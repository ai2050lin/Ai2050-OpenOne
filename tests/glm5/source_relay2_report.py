"""Read-only scientific aggregation and identity audit for the locked relay campaign."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import source_relay2_design as D
import source_relay2_math as M

def reporting_log(stage,done,total,start,**kw):
    """Read-only closeout remains available after the measurement stop marker."""
    from datetime import datetime
    entry={'time':datetime.now().astimezone().isoformat(),'stage':stage,'done':done,'total':total,
           'elapsed_seconds':time.monotonic()-start,**kw}
    with (D.OUT/'closeout-progress.jsonl').open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(entry)+'\n');stream.flush()
    print(json.dumps(entry),flush=True)

def report():
    D.verify_internal();root=D.OUT/D.MODEL;mat=D.read(D.OUT/'qwen14-materials.json');panels={}
    for split,rows in mat['splits'].items():
        if not (root/f'{split}-gate.json').exists():
            stop=D.read(D.OUT/'measurement-stop.json')
            panels[split]={'status':'incomplete_stopped','expected_rows':len(rows),'reason':stop['reason'],
                'complete_behavior_rows_logged':stop['partial_frame_behavior_complete_rows_logged'],
                'row_results_persisted':False,'internal_measurement_started':False}
            continue
        gate=D.read(root/f'{split}-gate.json');behavior=D.read(root/f'{split}-behavior.json');failures=[]
        for r,b in zip(rows,behavior['rows']):
            if not b['normalized_correct'] or not b['terminated']:
                label='format_or_termination'
                if b['pred_record']!=r['target_record']:
                    s,e=divmod(b['pred_record'],2)
                    label='wrong_entity_same_source' if s==r['source'] else 'wrong_source_same_entity' if e==r['entity'] else 'wrong_source_and_entity'
                failures.append({'id':r['id'],'family':r['family'],'frame':r['frame'],'gold':r['gold'],'prediction':b['generated_text'],'failure':label})
        panel={'behavior':gate,'failures':failures,'failure_counts':{k:sum(x['failure']==k for x in failures) for k in sorted({x['failure'] for x in failures})}}
        for name,file in [('forecast','forecast'),('parameters','parameter-ledger'),('update','update-output'),('causal','causal'),('output_diagnostic','output-diagnostic')]:
            p=root/f'{split}-{file}.json'
            if p.exists():panel[name]=D.read(p)
        if 'forecast' in panel:
            z=np.load(root/f'{split}-forecast.npz');prob=z['actual_p'];panel['actual_attention_groups']=[]
            for bi,c in enumerate(D.HEADS):
                masses={group:[] for group in rows[0]['groups']}
                for i,r in enumerate(rows):
                    for group,ix in r['groups'].items():masses[group].append(float(prob[i,bi,np.array(ix,dtype=int)+prob.shape[-1]-len(r['tokens'])].sum()))
                panel['actual_attention_groups'].append({**c,'mean_mass':{g:float(np.mean(v)) for g,v in masses.items()},'per_row_mass':masses})
            z.close()
        panels[split]=panel
    shared=[]
    if all('forecast' in panels[s] for s in ['confirm','lockbox_frame']):
        for i,c in enumerate(D.HEADS):
            if all(panels[s]['forecast']['heads'][i]['passes'] for s in ['confirm','lockbox_frame']):shared.append(c)
    progress=[json.loads(line) for line in (D.OUT/'progress.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    resource={'peak_cuda_GiB':max((r.get('cuda_peak',0) for r in progress),default=0)/2**30,
        'peak_shared_MiB':max((r.get('shared_gpu_bytes') or 0 for r in progress),default=0)/2**20,
        'min_available_RAM_GiB':min(r['ram_available_bytes'] for r in progress)/2**30,
        'note':'Peaks across measured process telemetry; stage wall time includes IO and any lease waiting, not GPU active-compute time.'}
    D.save('report.json',{'panels':panels,'dual_panel_primary_pass_heads':shared,'resources':resource,
        'unique_materials_planned':3072,'families_planned':256,
        'complete_behavior_rows':sum(p['behavior']['n'] for p in panels.values() if 'behavior' in p),
        'internal_rows':sum(len(mat['splits'][s]) for s,p in panels.items() if 'forecast' in p),
        'execution_status':'stopped_for_missing_prospective_power_preflight',
        'scientific_admissibility':'Not formally admissible under AGENTS section13; completed measurements remain exploratory. Dual-panel criterion not established.',
        'formal_evidence_promotion':0,'scope':'Qwen3-14B, constructed 3-source 2-entity task, known answer vocabulary; heldout names and query forms. Native source ledger is algebra, not causal uniqueness. Original-model context replacement edits two head contexts and retains every other original-model component; recurrence is fitted and evaluated independently.',
        'limitations':['No natural-corpus SAE comparison, independent physical node, cross-language, long reasoning, or unrelated-task side-effect panel.',
            'All primary forecast methods observe full hs20 prefix, not only text or embeddings; learned coefficients are not native parameters.',
            'PCA128 has a dense learned input transform and higher total compute/parameter cost than native128 despite equal latent width.',
            'Path masks alter attention renormalization and do not equate perturbation size; they support bounded path effects, not pure semantic necessity.',
            'Original-model causal outputs are next-token distributions, not free-generation tests under intervention.',
            'No automatic GLM replication; prior GLM frame panel failed its frozen behavior gate.']})

def audit():
    start=time.monotonic();D.verify_internal();root=D.OUT/D.MODEL;mat=D.read(D.OUT/'qwen14-materials.json');checks=[]
    def check(name,condition,detail=None):
        checks.append({'name':name,'passed':bool(condition),'detail':detail});assert condition,(name,detail)
    check('engineering',D.read(D.OUT/'engineering.json')['passed'])
    plan=D.read(D.OUT/'plan.json');cal=D.read(D.OUT/'measurement-calibration.json')
    check('calibration identity',D.sha(D.OUT/'measurement-calibration.json')==plan['calibration_sha256'])
    check('runtime environment identity',D.sha(D.OUT/'runtime-environment.json')==plan['environment_sha256'])
    check('calibration source identity',D.sha(D.ROOT/cal['source'])==cal['source_sha256'])
    diagnostic=D.read(D.ROOT/cal['source'])
    check('canonical numerical calibration passed',cal['canonical_test_passed'] and diagnostic['canonical_test_passed'])
    check('calibration raw arrays identity',D.sha((D.ROOT/cal['source']).with_suffix('.npz'))==cal['arrays_sha256'])
    import torch,transformers,platform
    from transformers.models.qwen3 import modeling_qwen3
    from transformers.integrations import sdpa_attention
    env=D.read(D.OUT/'runtime-environment.json')
    for key,value in [('python',platform.python_version()),('torch',torch.__version__),('transformers',transformers.__version__),
                      ('numpy',np.__version__),('cuda_build',torch.version.cuda),('modeling_source_sha256',D.sha(Path(modeling_qwen3.__file__))),
                      ('sdpa_source_sha256',D.sha(Path(sdpa_attention.__file__)))] :
        check('current runtime/'+key,env[key]==value)
    fit=D.read(root/'fit.json');check('fit hash',D.sha(root/'fit.npz')==fit['fit_sha256']);check('fit freeze',fit['internal_freeze_sha256']==D.sha(D.OUT/'internal-freeze.json'))
    check('new names separated',not ({x for r in mat['splits']['discovery'] for x in r['names']+r['sources']} & {x for s in ['confirm','lockbox_frame'] for r in mat['splits'][s] for x in r['names']+r['sources']}))
    check('heldout panels name disjointness',not ({x for r in mat['splits']['confirm'] for x in r['names']+r['sources']} & {x for r in mat['splits']['lockbox_frame'] for x in r['names']+r['sources']}))
    old=D.read(D.ROOT/'tests/glm5/result/source_relay_v1/qwen14-materials.json')
    oldnames={x for rs in old['splits'].values() for row in rs for x in row['names']+row['sources']}
    newnames={x for rs in mat['splits'].values() for row in rs for x in row['names']+row['sources']}
    check('v1 exposed names excluded',not (oldnames & newnames))
    check('fixed width capacity',plan['fixed_length']==max(len(r['tokens']) for rs in mat['splits'].values() for r in rs)+8)
    for split,rows in mat['splits'].items():
        check(split+'/family count',len(set(r['family'] for r in rows))==D.COUNTS[split])
        check(split+'/query count',len(rows)==12*D.COUNTS[split])
        check(split+'/group partition',all(sorted(t for ix in r['groups'].values() for t in ix)==list(range(len(r['tokens']))) for r in rows))
        check(split+'/real final readout',all(r['positions'][-1]==len(r['tokens'])-1 for r in rows))
        if not (root/f'{split}-gate.json').exists():
            check(split+'/stopped panel documented',(D.OUT/'measurement-stop.json').exists())
            check(split+'/stopped panel not measured internally',not (root/f'{split}-native').exists())
            check(split+'/no invented behavior aggregate',not (root/f'{split}-behavior.json').exists())
            continue
        ids=[r['id'] for r in rows];behavior=D.read(root/f'{split}-behavior.json');gate=D.read(root/f'{split}-gate.json')
        check(split+'/complete behavior order',ids==[b['id'] for b in behavior['rows']])
        check(split+'/gate hash',gate['behavior_sha256']==D.sha(root/f'{split}-behavior.json'))
        recompute=D.gate(rows,behavior['rows']);check(split+'/gate recomputed',all(recompute[k]==gate[k] for k in recompute))
        if not gate['passed']:
            check(split+'/failed gate stopped internal',not (root/f'{split}-native').exists());continue
        D.admission(split);manifest=D.read(root/f'{split}-native-manifest.json')
        check(split+'/native baseline exact',manifest['replay']['passed'] and manifest['replay']['choice_logits_max_difference']==0)
        check(split+'/identical-prefix K exact',manifest['prefix_invariance']['passed'] and manifest['prefix_invariance']['different']==0)
        for name,info in manifest['files'].items():
            p=root/f'{split}-native'/name;check(split+'/'+name,p.stat().st_size==info['bytes'] and D.sha(p)==info['sha256'])
        check(split+'/capture after gate',D.read(root/f'{split}-native-started.json')['provenance']['created_at']>gate['provenance']['created_at'])
        if split!='discovery':check(split+'/capture after fit freeze',D.read(root/f'{split}-native-started.json')['provenance']['created_at']>fit['provenance']['created_at'])
        pred=D.read(root/f'{split}-forecast.json');check(split+'/forecast hash',D.sha(root/f'{split}-forecast.npz')==pred['arrays_sha256'])
        arr=np.load(root/f'{split}-forecast.npz');p=np.load(root/'fit.npz');ap=arr['aware_native_p']
        for bi in range(2):
            truth=arr['truth'][:,bi].astype(np.float64);wo=p[f'wo_{bi}'].astype(np.float64)
            for method in ['native','random','pca','embedding','aware_native','aware_pca']:
                # Independently use all residual output coordinates, not the evaluation Gram matrix.
                error=np.square((truth-arr[method+'_trueV'][:,bi])@wo.T).sum(-1).mean()
                claimed=pred['heads'][bi]['methods'][method]['write_MSE']
                check(split+f'/full-coordinate-MSE/{bi}/{method}',abs(error-claimed)<=max(1e-5,abs(error)*2e-5),{'recomputed':float(error),'stored':claimed})
            check(split+f'/probability-mask/{bi}',all(np.all(ap[i,bi,:ap.shape[-1]-len(r['tokens'])]==0) for i,r in enumerate(rows)))
        arr.close();p.close()
        pars=D.read(root/f'{split}-parameter-ledger.json');check(split+'/parameter ledger hash',D.sha(root/f'{split}-parameter-ledger.npz')==pars['arrays_sha256'])
        upd=D.read(root/f'{split}-update.json');check(split+'/recurrence final hash',D.sha(root/f'{split}-update-final.npz')==upd['final_sha256'])
        output=D.read(root/f'{split}-update-output.json');check(split+'/recurrence output baseline',output['actual_final_logits_replay_exact'])
        for method,items in output['rows'].items():check(split+'/recurrence row IDs/'+method,[r['id'] for r in items]==ids)
        if split!='discovery':
            diag=D.read(root/f'{split}-output-diagnostic.json');protocol=D.read(D.OUT/'output-diagnostic-protocol.json')
            check(split+'/output diagnostic source',diag['input_sha256']==D.sha(root/f'{split}-update-output.json') and diag['protocol_sha256']==D.sha(D.OUT/'output-diagnostic-protocol.json'))
            check(split+'/output diagnostic script',protocol['script_sha256']==D.sha(D.ROOT/'tests/glm5/source_relay2_output_diagnostic.py'))
            for method,entry in diag['methods'].items():
                check(split+'/output diagnostic count/'+method,entry['n']==len(rows) and sum(x['n'] for x in entry['token_counts'])==len(rows))
            if split=='lockbox_frame':check('output diagnostic recipe before new-frame behavior',protocol['provenance']['created_at']<D.read(root/'lockbox_frame-behavior-v2-started.json')['provenance']['created_at'])
            causal=D.read(root/f'{split}-causal.json');check(split+'/causal output baseline',causal['natural_logits_replay_max_error']==0 and causal['natural_top1_replay_exact'])
            check(split+'/causal final hash',D.sha(root/f'{split}-causal-final.npy')==causal['final_sha256'])
            for arm,items in causal['rows'].items():check(split+'/causal row IDs/'+arm,[r['id'] for r in items]==ids)
        reporting_log('audit',len(checks),len(checks),start,panel=split)
    from shared.dist_protocol.natural_register_runner import snapshot
    check('current model/tokenizer snapshot',snapshot(D.MODELS[D.MODEL])==D.read(D.OUT/'plan.json')['model_snapshot'])
    D.save('audit.json',{'passed':False,'technical_identity_checks_passed':True,'checks':checks,'count':len(checks),'elapsed_seconds':time.monotonic()-start,
        'governance_passed':False,'missing_requirements':['Prospective effect-size, required n and power>=0.8 calculation was absent before measurement.',
            'Frozen result provenance uses model/model_snapshot rather than the exact model_path/model_revision fields required in section13; originals preserved, no retroactive rewrite.'],
        'scope':'Completed-data identity and numeric checks passed. Overall research admission failed due to missing prospective preflight. Not independent scientific replication or evidence promotion.'})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['report','audit']);a=ap.parse_args();globals()[a.stage]()
