"""Read-only scientific aggregation and identity audit for the locked relay campaign."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import source_relay_design as D
import source_relay_math as M

def report():
    D.verify_internal();root=D.OUT/D.MODEL;mat=D.read(D.OUT/'qwen14-materials.json');panels={}
    for split,rows in mat['splits'].items():
        gate=D.read(root/f'{split}-gate.json');behavior=D.read(root/f'{split}-behavior.json');failures=[]
        for r,b in zip(rows,behavior['rows']):
            if not b['normalized_correct'] or not b['terminated']:
                label='format_or_termination'
                if b['pred_record']!=r['target_record']:
                    s,e=divmod(b['pred_record'],2)
                    label='wrong_entity_same_source' if s==r['source'] else 'wrong_source_same_entity' if e==r['entity'] else 'wrong_source_and_entity'
                failures.append({'id':r['id'],'family':r['family'],'frame':r['frame'],'gold':r['gold'],'prediction':b['generated_text'],'failure':label})
        panel={'behavior':gate,'failures':failures,'failure_counts':{k:sum(x['failure']==k for x in failures) for k in sorted({x['failure'] for x in failures})}}
        for name,file in [('forecast','forecast'),('parameters','parameter-ledger'),('update','update-output'),('causal','causal')]:
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
        'unique_materials':3072,'families':256,'internal_rows':sum(len(mat['splits'][s]) for s,p in panels.items() if 'forecast' in p),
        'formal_evidence_promotion':0,'scope':'Qwen3-14B, constructed3-source2-entity task, known answer vocabulary; heldout names and query forms. Native source ledger is algebra, not causal uniqueness. Original-model context replacement retains the other38 blocks/heads; recurrence is fitted and evaluated independently.',
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
    fit=D.read(root/'fit.json');check('fit hash',D.sha(root/'fit.npz')==fit['fit_sha256']);check('fit freeze',fit['internal_freeze_sha256']==D.sha(D.OUT/'internal-freeze.json'))
    check('new names separated',not ({x for r in mat['splits']['discovery'] for x in r['names']+r['sources']} & {x for s in ['confirm','lockbox_frame'] for r in mat['splits'][s] for x in r['names']+r['sources']}))
    for split,rows in mat['splits'].items():
        ids=[r['id'] for r in rows];behavior=D.read(root/f'{split}-behavior.json');gate=D.read(root/f'{split}-gate.json')
        check(split+'/complete behavior order',ids==[b['id'] for b in behavior['rows']])
        check(split+'/gate hash',gate['behavior_sha256']==D.sha(root/f'{split}-behavior.json'))
        recompute=D.gate(rows,behavior['rows']);check(split+'/gate recomputed',all(recompute[k]==gate[k] for k in recompute))
        if not gate['passed']:
            check(split+'/failed gate stopped internal',not (root/f'{split}-native').exists());continue
        D.admission(split);manifest=D.read(root/f'{split}-native-manifest.json')
        check(split+'/native baseline exact',manifest['replay']['passed'] and manifest['replay']['choice_logits_max_difference']==0)
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
            causal=D.read(root/f'{split}-causal.json');check(split+'/causal output baseline',causal['natural_logits_replay_max_error']==0 and causal['natural_top1_replay_exact'])
            check(split+'/causal final hash',D.sha(root/f'{split}-causal-final.npy')==causal['final_sha256'])
            for arm,items in causal['rows'].items():check(split+'/causal row IDs/'+arm,[r['id'] for r in items]==ids)
        D.log('audit',len(checks),len(checks),start,panel=split)
    from shared.dist_protocol.natural_register_runner import snapshot
    check('current model/tokenizer snapshot',snapshot(D.MODELS[D.MODEL])==D.read(D.OUT/'plan.json')['model_snapshot'])
    D.save('audit.json',{'passed':True,'checks':checks,'count':len(checks),'elapsed_seconds':time.monotonic()-start,
        'scope':'Technical identity, chronology, complete behavior admission, native hashes, full-coordinate numeric recomputation. Not an independent scientific replication or automatic evidence promotion.'})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['report','audit']);a=ap.parse_args();globals()[a.stage]()
