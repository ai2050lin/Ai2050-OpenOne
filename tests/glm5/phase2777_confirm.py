"""Prospective one-way natural-generation sufficiency test after calibration.

Written-source behavior passed discovery calibration. Spoken instruction
compliance did not. Therefore NO new contrast axis is fitted from invalid
spoken cells. Test the PREEXISTING edit's ability to induce natural speech,
with immutable new records/frames, exact length pairs and matched controls.
"""
from pathlib import Path
import argparse
import json
import sys
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import phase2773_w6_loop as W
import phase2777_materials_v3 as M
import phase2777_natural_loop_v3 as D

ROOT=W.ROOT;OLD=ROOT/'tests/glm5/result/phase2773_w6_v2'
OUT=ROOT/'tests/glm5/result/phase2777_confirm'
PUBLIC=ROOT/'research/phase2777/confirm'
RUN='P2777-W6-NATURAL-CONFIRM-v1';CFG={'k':512,'alpha':6.,'control':'target'}


def status(stage,state='done',**kw):
    from deploy.research_library import Library
    p=PUBLIC/'00-progress.research.json'
    d=W.read(p) if p.exists() else {'schema':'research_workpackage.v1','kind':'real','run_id':RUN,
        'title':'W6 · 等长度、多句式自由生成的正式充分性检验','stages':{},
        'model_snapshot':W.read(OLD/'freeze.json')['model_snapshot']['sha256']}
    d['stages'][stage]={'state':state,'time':time.time(),**kw};d['updated_at']=time.time()
    W.save(p,d);Library(ROOT/'distributed_data/research_library').ingest(d,str(p.relative_to(ROOT)),kind='local_report')
    W.log(stage,run_id=RUN,state=state,**kw)


def prepare():
    from transformers import AutoTokenizer
    from tools.power_analysis import check_phase
    pilots=[]
    for suffix in ['', '_v2','_v3','_v4','_v5']:
        p=ROOT/f'tests/glm5/result/phase2777_natural{suffix}/pilot-result.json';d=W.read(p)
        pilots.append({'run_id':d['run_id'],'path':str(p.relative_to(ROOT)),'sha256':W.sha(p),
            'passed':d['passed'],'style_accuracy':d['style_accuracy'],'content_accuracy':d['content_accuracy']})
    eligible=[p for p in pilots if p['style_accuracy']['acad']>=.8 and p['content_accuracy']>=.8]
    best=max(eligible,key=lambda p:sum(p['style_accuracy'].values()))
    assert best['run_id']=='P2777-W6-NATURAL-v3'
    b=M.Builder(AutoTokenizer.from_pretrained(W.MODEL,local_files_only=True));rows=[]
    for i in range(3000,3128):
        for frame in [2,3]:rows+=b.pair(i,frame,384 if i%2==0 else 512,'lockbox_frame_and_record')
    assert len(rows)==512
    power=check_phase(.20,128,p_d=.5);assert power['pass']
    design={'schema':'natural_sufficiency_prereg.v1','run_id':RUN,'kind':'real','registered_at':time.time(),
        'question':'Can the preexisting weight edit induce qualified colloquial free completions from stable written-source long prompts?',
        'prior_exposure':pilots,'material_selection':'discovery only: maximize balanced style agreement among written>=.8 and content>=.8; chosen user-request v3',
        'scope_change':'two-way style contrast failed discovery gate; this is ONE-WAY edit sufficiency, not confirmation of a natural preexisting spoken-vs-written mechanism',
        'matched_axis':'NOT_RUN: spoken conditions failed behavior calibration, so no new mean-difference axis is extracted',
        'families':128,'frames':[2,3],'lengths':[384,512],'n_prompts':512,
        'pairs':'same facts and all input text outside equal-token style request identical; 64 independent families per length, both frames per family',
        'heldout':'new records3000..3127 and new source/request frames2/3, not exposed in pilots',
        'main_edit':CFG,'axis_source':'P2773 frozen discovery direction and actual-contribution columns, no selection on new data',
        'controls':['same-column norm-matched random direction seed0','activation-energy-stratified random columns seed0'],
        'generation':'greedy,batch8,max160 new tokens,no thinking; full independent outputs, no answer pool',
        'style_evaluation':'unchanged v3 original-base-model blinded judge in both A/B letter orders plus frozen linguistic-family rubric; agreement and>=45chars/2sentences',
        'judge_limit':'same-model rater and authored calibration, not independent human validation',
        'baseline_gate':'written-source classification>=.70 pooled and>=.60 each frame-length; provided-content agreement>=.75 pooled',
        'primary':'qualified written->colloquial fraction over ALL written-source inputs >=.20 and 95% family CI lower>.10; positive lower difference vs both own controls; content loss CI upper<=.05; original colloquial-response loss CI upper<=.05; keyed capability NI passes',
        'qualified':'unedited output written, edited output colloquial under both evaluators, edited content correct, not truncated',
        'spoken_monitor':'relative retention of whatever spoken outputs baseline actually produced; no absolute70% condition and no use as valid internal-style positive cells',
        'statistics':'128 family clusters, preserve frames/styles,6000 bootstrap,.025/.975 quantiles; only one primary hypothesis',
        'power':power,'capability':{'formal_n':2200,'reference_n':140,'margin':.02,
            'method':'paired simultaneous Clopper-Pearson losses/gains,95% union-bound coverage',
            'scope':'same frozen monitoring mixture as previous run; repeats disclosed; not a new general ability benchmark'},
        'raw_fields':'all37 layers/all9728 MLP inputs B18..27, last two visible positions, original BF16 bits; no new feature selection',
        'resource_budget_seconds':3600,'software_dependencies':{str(Path(p).relative_to(ROOT)):W.sha(p)
            for p in [__file__,W.__file__,D.__file__,M.__file__,M.V2.__file__,M.V2.V1.__file__]},
        'not_measured':['new matched axis','cross-language','other models','graph-only prediction','multi-node replication']}
    W.save(OUT/'corpus.json',rows,True);W.save(PUBLIC/'prereg.json',design,True)
    f={'script_sha256':W.sha(__file__),'dependencies':design['software_dependencies'],'model_snapshot':W.read(OLD/'freeze.json')['model_snapshot'],
        'directions_sha256':W.sha(OLD/'directions.npz'),'corpus_sha256':W.sha(OUT/'corpus.json'),
        'capability_sha256':W.sha(OLD/'capability-inputs.json'),'design_sha256':W.sha(PUBLIC/'prereg.json')}
    W.save(OUT/'freeze.json',f,True);status('freeze',families=128,samples=512,power=power)


def ci(v):
    v=np.asarray(v,float);rng=np.random.default_rng(M.SEED+99)
    means=v[rng.integers(0,len(v),size=(6000,len(v)))].mean(1)
    return {'n_families':len(v),'mean':float(v.mean()),'ci':[float(np.quantile(means,.025)),float(np.quantile(means,.975))]}


def run():
    from deploy.parameter_reader import model_snapshot
    f=W.read(OUT/'freeze.json')
    assert all(W.sha(ROOT/p)==h for p,h in f['dependencies'].items())
    assert W.sha(OUT/'corpus.json')==f['corpus_sha256'] and W.sha(PUBLIC/'prereg.json')==f['design_sha256']
    assert W.sha(OLD/'directions.npz')==f['directions_sha256'] and W.sha(OLD/'capability-inputs.json')==f['capability_sha256']
    assert model_snapshot(W.MODEL)['sha256']==f['model_snapshot']['sha256']
    e=D.make_engine();rows=W.read(OUT/'corpus.json');W.OUT=OUT;z=np.load(OLD/'directions.npz');D.set_axis(e,z)
    status('collection','running');e.capture(rows,OUT/'native');status('collection',samples=len(rows))
    cache=OUT/'judge-cache.json';conditions=[('base',None),('legacy512_a6','target'),
        ('legacy_random_direction','random_direction'),('legacy_energy_columns','random_columns')]
    for name,ctl in conditions:
        p=OUT/f'{name}-assessed.json'
        if p.exists():continue
        if ctl is None:a=D.generate(e,rows,OUT/'generation'/name)
        else:
            with e.edit({**CFG,'control':ctl}) as tensors:a=D.generate(e,rows,OUT/'generation'/name)
            W.save(OUT/f'{name}-edit.json',{'tensors':tensors,'config':{**CFG,'control':ctl}},True)
        W.save(p,D.assess(e,a,cache),True);status('validation','running',condition=name)
        if time.time()-e.started>3600:raise TimeoutError('Frozen one-hour budget; all batches retained for recovery')
    base=W.read(OUT/'base-assessed.json');edited=W.read(OUT/'legacy512_a6-assessed.json')
    cells={}
    for frame in [2,3]:
        for length in [384,512]:
            for reg in ['acad','colloq']:
                a=[r for r in base if r['frame']==frame and r['length']==length and r['register']==reg]
                cells[f'F{frame}/L{length}/{reg}']=float(np.mean([r['style']==reg for r in a]))
    content_base=float(np.mean([r['content']['pass'] for r in base]))
    baseline_gate=min(v for k,v in cells.items() if k.endswith('acad'))>=.60 and \
        np.mean([r['style']=='acad' for r in base if r['register']=='acad'])>=.70 and content_base>=.75
    def qual(a):return np.array([b['style']=='acad' and x['style']=='colloq' and x['content']['pass'] and not x['hit_limit'] for b,x in zip(base,a)])
    q=qual(edited);effect=ci(D.family_mean(rows,q,'acad'));comparisons={}
    for name in ['legacy_random_direction','legacy_energy_columns']:
        a=W.read(OUT/f'{name}-assessed.json');comparisons[name]=ci(D.family_mean(rows,q.astype(float)-qual(a).astype(float),'acad'))
    loss=ci(D.family_mean(rows,[float(b['content']['pass'])-float(a['content']['pass']) for b,a in zip(base,edited)]))
    spoken_loss=ci(D.family_mean(rows,[float(b['style']=='colloq' and b['content']['pass'])-float(a['style']=='colloq' and a['content']['pass']) for b,a in zip(base,edited)],'colloq'))
    qs=W.read(OLD/'capability-inputs.json');status('capability','running',n=len(qs))
    if not (OUT/'capability-base.json').exists():W.save(OUT/'capability-base.json',e.capability(qs),True)
    if not (OUT/'capability-edited.json').exists():
        with e.edit(CFG):W.save(OUT/'capability-edited.json',e.capability(qs),True)
    bc=W.read(OUT/'capability-base.json');ec=W.read(OUT/'capability-edited.json')
    cap=W.paired_interval([x['correct'] for x in bc],[x['correct'] for x in ec],np.array([x['gate_eligible'] for x in bc]))
    passed=baseline_gate and effect['mean']>=.2 and effect['ci'][0]>.1 and all(v['ci'][0]>0 for v in comparisons.values()) \
        and loss['ci'][1]<=.05 and spoken_loss['ci'][1]<=.05 and cap['status']=='noninferior'
    counter=[]
    for b,a,v in zip(base,edited,q):
        if (a['register']=='acad' and not v) or not a['content']['pass']:
            counter.append({'sample_id':a['id'],'condition':'legacy512_a6','reason':'natural_style_or_content_not_supported',
                'base_style':b['style'],'edited_style':a['style'],'base':b['text'],'edited':a['text'],
                'content':a['content'],'judge':a['blind_judge'],'rubric':a['rubric']})
    counter += [{'sample_id':b['id'],'condition':'legacy512_a6','reason':'keyed capability regression',
        'base':b['prediction'],'edited':a['prediction'],'answer':b['answer'],'source':b['source']}
        for b,a in zip(bc,ec) if b['correct'] and not a['correct']]
    h={'qualified_flip':effect,'comparisons':comparisons,'content_loss':loss,'colloquial_loss':spoken_loss,
        'colloquial_source_preserved':float(np.mean([r['style']=='colloq' and r['content']['pass'] for r in edited if r['register']=='colloq'])),
        'capability':cap,'supported':bool(passed)}
    result={'schema':'natural_register_result.v1','kind':'real','run_id':RUN,'model_snapshot':f['model_snapshot']['sha256'],
        'baseline_gate':bool(baseline_gate),'baseline_cells':cells,'baseline_content':content_base,'hypotheses':{'legacy512_a6':h},
        'baseline_by_register':{reg:float(np.mean([r['style']==reg for r in base if r['register']==reg])) for reg in ['acad','colloq']},
        'matched_axis_status':'not_run_due_to_failed_spoken_behavior_calibration','counterexamples':counter,
        'seconds':time.time()-e.started,'peak_cuda_bytes':int(e.torch.cuda.max_memory_allocated()),'env':e.env,
        'design_sha256':f['design_sha256'],'scope':'one-way sufficiency only; no successful preexisting two-way register contrast claim',
        'original_model_snapshot_after':model_snapshot(W.MODEL)['sha256']}
    assert result['original_model_snapshot_after']==result['model_snapshot']
    W.save(OUT/'result.json',result,True);status('capability',**cap)
    status('validation','done' if passed else 'not_supported',hypothesis=h,baseline_gate=bool(baseline_gate))
    W.log('natural_confirmation_complete',hypothesis=h,baseline_gate=bool(baseline_gate),seconds=result['seconds'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run']);a=p.parse_args();{'prepare':prepare,'run':run}[a.stage]()
