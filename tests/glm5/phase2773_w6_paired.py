"""W6 v3 prospectively frozen paired-context follow-up to the v2 readout failure.

Candidate fixed from v2 CONFIRM (not its lockbox or capability outcomes).
New record families 300..459 are never used to select an edit. Reuse original
native discovery directions, unedited capability baseline, and read-only model.
"""
from pathlib import Path
import argparse
import json
import sys
import time
import hashlib
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import phase2773_w6_loop as W
from phase2773_resume import ids_only

ROOT=W.ROOT
OLD=W.OUT
OUT=ROOT/'tests/glm5/result/phase2773_w6_v3'
PUBLIC=ROOT/'research/phase2773/v3'
RUN='P2773-W6-REGISTER-v3'


def prepare():
    from transformers import AutoTokenizer
    from tools.power_analysis import check_phase
    if (OUT/'freeze.json').exists():raise ValueError('Already frozen')
    old=W.read(OLD/'result.json');curve=W.read(OLD/'dose-curve.json')['rows']
    # Selected BEFORE any newly generated family has been executed.
    selected=max((r for r in curve if r['k']<9728),key=lambda r:(r['shift'],-r['addressed_params'],-abs(r['alpha'])))
    tok=AutoTokenizer.from_pretrained(W.MODEL,local_files_only=True)
    rows=[]
    for i in range(300,460):
        for reg in ['acad','colloq']:
            text,fact=W.render(i,reg)
            prompt=tok.apply_chat_template([{'role':'user','content':text}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
            ids=tok.encode(prompt,add_special_tokens=False)
            rows.append({'id':f'evt-{i:03d}-{reg}','family_id':f'evt-{i:03d}','split':'lockbox',
                'register':reg,'text':text,'prompt':prompt,'token_ids':ids,'tokens':len(ids),'fact':fact})
    assert len({r['family_id'] for r in rows})==160 and min(r['tokens'] for r in rows)>=256
    # Audit of old measurement is explicitly post hoc, not retroactive prereg.
    oldbase=W.read(OLD/'baseline-scores.json');lookup=dict(zip(oldbase['ids'],oldbase['scores']))
    oldrows=W.read(OLD/'corpus.json');disc=[r for r in oldrows if r['split']=='discovery' and r['register']=='acad']
    gaps=[]
    for r in disc:
        a=lookup[r['id']];c=lookup[r['id'].replace('acad','colloq')]
        gaps.append((c[0]-c[1])-(a[0]-a[1]))
    audit={'kind':'posthoc_measurement_audit','source_run':W.RUN,
        'baseline_absolute_classifier':{'acad_accuracy':1.0,'colloquial_accuracy':0.0,'balanced_accuracy':.5},
        'discovery_paired_context_gap_mean_nats':float(np.mean(gaps)),
        'discovery_positive_paired_gap_fraction':float(np.mean(np.array(gaps)>0)),
        'full_mask_random_columns_degenerate':True,
        'interpretation':'Absolute switch lacks calibration; all-column mask control is identical. Neither failure proves absence of conditional control.',
        'source_scores_sha256':W.sha(OLD/'baseline-scores.json')}
    design={'schema':'w6_prereg.v3','run_id':RUN,'kind':'real','seed':W.SEED,
        'registered_at':time.time(),'prior_results_exposed':'Complete v2 and its failures; discovery/confirm directions reused with explicit provenance',
        'hypothesis':'Editing a preselected sparse MLP writing direction attenuates paired-context register preference in NEW records',
        'primary_candidate':{k:selected[k] for k in ['k','alpha','control']},
        'candidate_source':'maximum mean margin shift among sparse k<9728 in v2 confirm; no selection on new families',
        'paired_metric':'gap = [logP(colloq_completion)-logP(formal_completion)]_colloq_context - same_academic_context',
        'effect':'attenuation = gap_base - gap_edited (nats), per entire record family',
        'alternatives':{'colloquial':W.COLLOQ,'formal':W.FORMAL},
        'baseline_measurement_gate':'On new records: paired gap mean CI lower>0 and fraction(gap>0)>=0.80. Otherwise no positive interpretation.',
        'primary_criterion':{'min_mean_attenuation_nats':.5,'require_mean_ci95_lower_above':.5,
            'min_fraction_attenuation_ge_half_nat':.70,'require_vs_each_control_ci95_lower_above':0},
        'controls':['three same-column norm-matched random directions','activation-energy-stratified random columns',
                    'wrong layers 0..9, intervention magnitude reported separately'],
        'dose':{'fixed_k':selected['k'],'alphas':[-1.,.5,1.,2.,4.,6.],
                'secondary_only':'all doses published; no selection of a replacement primary after new results'},
        'power':check_phase(.20,160,p_d=.5),
        'power_scope':'paired binary directional-win planning (0.70 vs 0.50); continuous-effect CI may be inconclusive',
        'capability':W.read(ROOT/'research/phase2773/W6-v2-prereg.json')['capability'],
        'capability_reuse':'same frozen 2340-question monitor and unedited base scores from v2; no claim this is a newly untouched benchmark',
        'natural_generation':{'n_families':16,'registers':['acad','colloq'],'max_new_tokens':96,'do_sample':False,
            'selection':'first 16 new family IDs; paired base and selected edit; literal and lexical proxies descriptive'},
        'raw_fields':'all37 layers and all9728 MLP units in layers18..27 at last two positions, original BF16 bits, unedited new records',
        'no_claims':['universal low-rank gates','capacity not locally editable','true minimum parameter count','general natural style control','cross-model replication'],
        'time_budget_seconds':1500,'native_discovery_source_sha256':W.sha(OLD/'directions.npz'),
        'model_snapshot':old['model_snapshot'],'repair_note':'BatchEncoding container adapter from recorded v2 execution repair; token IDs identical',
        'natural_markers':W.read(ROOT/'research/phase2773/W6-v2-prereg.json')['natural_markers']}
    assert design['power']['pass']
    W.save(OUT/'corpus.json',rows,True);W.save(PUBLIC/'W6-v3-prereg.json',design,True)
    W.save(PUBLIC/'v2-measurement-audit.research.json',audit,True)
    freeze={'run_id':RUN,'model_snapshot':W.read(OLD/'freeze.json')['model_snapshot'],
        'script_sha256':W.sha(__file__),'engine_sha256':W.sha(W.__file__),
        'design_sha256':W.sha(PUBLIC/'W6-v3-prereg.json'),'corpus_sha256':W.sha(OUT/'corpus.json'),
        'directions_sha256':W.sha(OLD/'directions.npz'),'capability_inputs_sha256':W.sha(OLD/'capability-inputs.json'),
        'capability_base_sha256':W.sha(OLD/'capability-base.json'),'created_at':time.time()}
    W.save(OUT/'freeze.json',freeze,True)
    status('freeze',families=160,samples=320,primary=design['primary_candidate'])


def status(stage,state='done',**kw):
    from deploy.research_library import Library
    p=PUBLIC/'00-progress.research.json'
    d=W.read(p) if p.exists() else {'schema':'research_workpackage.v1','kind':'real','run_id':RUN,
        'title':'W6 v3 · 成对语域条件效应与能力保持','stages':{},'model_snapshot':W.read(OUT/'freeze.json')['model_snapshot']['sha256']}
    d['stages'][stage]={'state':state,'time':time.time(),**kw};d['updated_at']=time.time()
    W.save(p,d);Library(ROOT/'distributed_data/research_library').ingest(d,str(p.relative_to(ROOT)),kind='local_report')
    W.log(stage,run_id=RUN,state=state,**kw)


def gap(scores):
    margin=np.asarray(scores)[:,0]-np.asarray(scores)[:,1]
    # Frozen row order is family then academic/colloquial.
    return margin[1::2]-margin[0::2]


def run():
    from deploy.parameter_reader import model_snapshot
    f=W.read(OUT/'freeze.json');design=W.read(PUBLIC/'W6-v3-prereg.json')
    for path,key in [(Path(__file__),'script_sha256'),(Path(W.__file__),'engine_sha256'),
                     (PUBLIC/'W6-v3-prereg.json','design_sha256'),(OUT/'corpus.json','corpus_sha256'),
                     (OLD/'directions.npz','directions_sha256'),(OLD/'capability-inputs.json','capability_inputs_sha256'),
                     (OLD/'capability-base.json','capability_base_sha256')]:assert W.sha(path)==f[key]
    assert model_snapshot(W.MODEL)['sha256']==f['model_snapshot']['sha256']
    if (OUT/'result.json').exists():return
    e=W.Engine();original=e.tok.apply_chat_template
    def adapter(*a,**kw):
        value=original(*a,**kw);return ids_only(value) if kw.get('tokenize',True) else value
    e.tok.apply_chat_template=adapter
    # Use the unchanged pre-recorded discovery tensors, never refit on new data.
    z=np.load(OLD/'directions.npz');e.u={};e.ranking={};e.static={};e.energy={}
    for l in W.LAYERS:
        e.u[l]=e.torch.tensor(z[f'u_{l}'],device='cuda');e.ranking[l]=z[f'order_{l}'];e.energy[l]=z[f'energy_{l}']
    W.OUT=OUT  # only edit artifact destination; algorithm remains byte-for-byte frozen.
    rows=W.read(OUT/'corpus.json');status('collection','running');e.capture(rows,OUT/'native')
    if not (OUT/'base.json').exists():W.save(OUT/'base.json',{'scores':e.scores(rows).tolist()},True)
    base=gap(W.read(OUT/'base.json')['scores']);baseline_gate=W.bootstrap_mean(base)
    baseline_ok=baseline_gate['ci95'][0]>0 and float((base>0).mean())>=.8
    status('collection',baseline_gate=baseline_gate,baseline_measurement_passed=baseline_ok)
    chosen=design['primary_candidate'];dose=[];primary_scores=None
    for alpha in design['dose']['alphas']:
        cfg={**chosen,'alpha':alpha};p=OUT/'dose'/f'a{alpha}.json'
        if not p.exists():
            with e.edit(cfg,store=alpha==chosen['alpha']) as tensors:s=e.scores(rows)
            W.save(p,{'config':cfg,'scores':s.tolist(),'tensors':tensors},True)
        d=W.read(p);atten=base-gap(d['scores'])
        dose.append({'id':f'k{chosen["k"]}-a{alpha}','k':chosen['k'],'alpha':alpha,
            'shift':float(atten.mean()),'flip_fraction':float((atten>=.5).mean()),
            'addressed_params':sum(t['addressed_params'] for t in d['tensors']),
            'changed_params':sum(t['changed_params'] for t in d['tensors']),
            'max_relative_frobenius':max(t['relative_frobenius'] for t in d['tensors'])})
        if alpha==chosen['alpha']:primary_scores=d['scores'];selected=dose[-1]
        W.log('paired_dose',**dose[-1])
    status('discovery',note='候选继承 v2 确认集；本轮剂量仅描述，不重新选择',config=chosen)
    target=base-gap(primary_scores);comparisons={};lockbox={'target':{'config':chosen,'shift':W.bootstrap_mean(target),
        'flip_fraction':float((target>=.5).mean()),'metric':'fraction of families with attenuation >=0.5 nats'}}
    configs=[(f'random-direction-{s}',{**chosen,'control':'random_direction','seed':s}) for s in range(3)]
    configs += [('energy-matched-columns',{**chosen,'control':'random_columns'}),('wrong-layers',{**chosen,'control':'wrong_layers'})]
    for name,cfg in configs:
        p=OUT/'controls'/f'{name}.json'
        if not p.exists():
            with e.edit(cfg) as tensors:s=e.scores(rows)
            W.save(p,{'config':cfg,'scores':s.tolist(),'tensors':tensors},True)
        attenuation=base-gap(W.read(p)['scores'])
        lockbox[name]={'config':cfg,'shift':W.bootstrap_mean(attenuation),'flip_fraction':float((attenuation>=.5).mean())}
        comparisons[name]=W.bootstrap_mean(target-attenuation)
    natural=rows[:32]
    # Existing marker definitions are unchanged; use v2 engine for identical generation settings.
    if not (OUT/'generation-base.json').exists():W.save(OUT/'generation-base.json',e.generate(natural),True)
    if not (OUT/'generation-edited.json').exists():
        with e.edit(chosen):W.save(OUT/'generation-edited.json',e.generate(natural),True)
    status('capability','running',n=2340,reused_unedited_baseline=True)
    if not (OUT/'capability-edited.json').exists():
        with e.edit(chosen):W.save(OUT/'capability-edited.json',e.capability(W.read(OLD/'capability-inputs.json')),True)
    bc=W.read(OLD/'capability-base.json');ec=W.read(OUT/'capability-edited.json')
    cap=W.paired_interval([r['correct'] for r in bc],[r['correct'] for r in ec],np.array([r['gate_eligible'] for r in bc]))
    sources={s:W.paired_interval([r['correct'] for r in bc],[r['correct'] for r in ec],np.array([r['source']==s for r in bc])) for s in sorted({r['source'] for r in bc})}
    passed=baseline_ok and lockbox['target']['shift']['ci95'][0]>.5 and lockbox['target']['flip_fraction']>=.7 \
        and all(v['ci95'][0]>0 for v in comparisons.values()) and cap['status']=='noninferior'
    counter=[{'sample_id':r['family_id'],'reason':'paired attenuation below preregistered 0.5 nats',
        'base_context_gap':float(base[i]),'edited_context_gap':float(gap(primary_scores)[i]),'attenuation':float(target[i])}
        for i,r in enumerate(rows[::2]) if target[i]<.5]
    counter += [{'sample_id':b['id'],'reason':'keyed capability regression','source':b['source'],
        'base':b['prediction'],'edited':a['prediction'],'answer':b['answer']} for b,a in zip(bc,ec) if b['correct'] and not a['correct']]
    counter += [{'sample_id':a['id'],'reason':'generated literal-content proxy failed','base':b['text'],
        'edited':a['text'],'expected':a['expected']} for b,a in zip(W.read(OUT/'generation-base.json'),W.read(OUT/'generation-edited.json')) if not a['literal_content_ok']]
    result={'schema':'w6_result.v3','kind':'real','run_id':RUN,'model_snapshot':f['model_snapshot']['sha256'],
        'design_sha256':f['design_sha256'],'env':e.env,'selected':selected,'lockbox':lockbox,'control_comparisons':comparisons,
        'baseline_readout_gate':baseline_ok,'baseline_paired_gap':baseline_gate,'positive_baseline_gap_fraction':float((base>0).mean()),
        'capability':cap,'capability_sources':sources,'target_supported_in_scope':passed,
        'verdict':'paired_context_attenuation_supported' if passed else 'paired_context_attenuation_not_supported',
        'counterexamples':counter,'compactness':{'status':'observed_upper_bound' if passed else 'not_found',
            'smallest_tested_feasible_params':selected['addressed_params'] if passed else None,
            'scope':'one prespecified sparse candidate; dose series not used for replacement selection'},
        'seconds':time.time()-e.started,'peak_cuda_bytes':int(e.torch.cuda.max_memory_allocated()),
        'provenance':{**W.read(OLD/'result.json')['provenance'],'script_sha256':f['script_sha256'],
            'model_revision':f['model_snapshot']['sha256'],'created_at':time.time()}}
    W.save(OUT/'result.json',result,True)
    W.save(OUT/'dose-curve.json',{'split':'new record families; descriptive doses, primary fixed in advance',
        'rows':dose,'selected':selected},True)
    status('capability',**cap);status('validation','done' if passed else 'not_supported',verdict=result['verdict'],counterexamples=len(counter))
    W.log('paired_complete',verdict=result['verdict'],primary=lockbox['target'],capability=cap)


def publish():
    from deploy.research_library import Library
    r=W.read(OUT/'result.json');lib=Library(ROOT/'distributed_data/research_library')
    rows=W.read(OUT/'corpus.json');native=W.read(OUT/'native/manifest.json');edit=W.read(OUT/'selected-edit-manifest.json')
    units={t['layer']:t['columns'][0] for t in edit['tensors']}
    preview_ids={rows[0]['id'],rows[1]['id']};preview=[];byid={s['id']:s for s in rows}
    # Two complete original-model responses at every edited block; no edited-state claim.
    for batch in native['files']:
        p=OUT/'native'/batch['path'];assert W.sha(p)==batch['sha256'];z=np.load(p)
        for j,key in enumerate(z['ids'].tolist()):
            if key not in preview_ids:continue
            row=byid[key];rec={'id':key,'entity':row['family_id'],'split':row['split'],'text':row['text'],
                'tokens':row['tokens'],'responses':{},'activation_by_layer':{}}
            for n,l in enumerate(W.LAYERS):
                values=(z['mlp_bits'][j,n,-1].astype(np.uint32)<<16).view(np.float32)
                field={'schema':'research_mlp_response.v1','kind':'real','run_id':RUN,'model_snapshot':r['model_snapshot'],
                    'block':l,'boundary':'mlp.down_proj.input','readout':'final','selected_unit':units[l],
                    'sample_id':key,'text':row['text'],'values':values.tolist(),
                    'response_state':'unedited baseline','reconstruction':'lossless BF16 bits expanded to float32; no recomputed SwiGLU approximation',
                    'native_manifest_sha256':W.sha(OUT/'native/manifest.json')}
                fp=PUBLIC/'responses'/f'{key}-B{l}.research.json';W.save(fp,field,True)
                rec['responses'][str(l)]=lib.ingest(field,str(fp.relative_to(ROOT)),kind='local_report')['sha256']
                rec['activation_by_layer'][str(l)]=float(values[units[l]])
            preview.append(rec)
    report={'schema':'research_intervention_report.v1','kind':'real','run_id':RUN,
        'title':'W6 v3 · 成对语域条件效应与能力保持','model_snapshot':r['model_snapshot'],
        'primary_label':'最终留出：衰减≥0.5 nats 的家族比例','primary_name':'成对语域条件效应',
        'dose_description':'新记录家族上的成对条件差衰减（nats）；正值为条件差减小。主编辑已在运行前固定，其余剂量只作描述。',
        'result':r,'dose_curve':W.read(OUT/'dose-curve.json'),'preregistration':W.read(PUBLIC/'W6-v3-prereg.json'),
        'freeze':W.read(OUT/'freeze.json'),'raw_manifest':native,'edit_manifest':edit,'sample_preview':preview,
        'inputs':[{k:v for k,v in x.items() if k!='token_ids'} for x in rows],
        'generations':{'base':W.read(OUT/'generation-base.json'),'edited':W.read(OUT/'generation-edited.json')},
        'prior_measurement_audit':W.read(PUBLIC/'v2-measurement-audit.research.json'),
        'limitations':['成对续写偏好是语域条件效应代理，不是完整自然风格分类',
            '没有独立人工评审，自然生成保义仅作字面代理',
            '能力保持是固定混合题池的提供答案口径，重复使用原模型监测基线；M3Exam仅参考',
            '只测一个快照；没有图谱独立预测、SAE发现或第二节点复算'],
        'artifact_files':{str(p.relative_to(OUT)):W.sha(p) for p in OUT.glob('*.json')}}
    rp=PUBLIC/'report.json';W.save(rp,report,True);ref=lib.ingest(report,str(rp.relative_to(ROOT)),kind='local_report')['sha256']
    prior=W.read(ROOT/'research/phase2773/graph.research.json');cards=[];edges=[]
    level='intervention' if r['target_supported_in_scope'] else 'negative'
    for t in edit['tensors']:
        l=t['layer'];fid=f'{r["model_snapshot"]}:B{l}:W6v3:{RUN}'
        cards.append({'id':fid,'label':f'W6 v3 · B{l} 成对条件效应（'+('范围内支持' if level=='intervention' else '未通过主判')+'）',
            'layer':l,'boundary':'mlp.down_proj.input','index':{'space':'mlp','unit':units[l]},'run_id':RUN,
            'method':'inherited discovery-conditioned write direction; preregistered sparse parameter edit',
            'scope':{'readout':'final','edited_columns':t['columns'],'representative_unit_only':True},
            'evidence':{'level':level,'artifact_sha256':ref},'counterexamples':r['counterexamples'],
            'compactness':r['compactness'],'causal_tests':'实际参数编辑与预注册留出对照；原始响应来自未编辑模型',
            'summary':{'limitations':report['limitations']}})
        edges.append({'id':fid+':edge','source':'language:register-conditional-readout','target':fid,
            'relation':'tested_paired_context_attenuation' if level=='intervention' else 'unsupported_paired_context_attenuation',
            'run_id':RUN,'evidence':{'level':level,'artifact_sha256':ref}})
    graph={'schema':'research_graph.v1','kind':'real','run_id':RUN,'model_snapshot':r['model_snapshot'],
        'feature_cards':cards,'graph_edges':edges,'prior_run_graph_sha256':W.read(OLD/'registration.json')['graph_sha256']}
    gp=PUBLIC/'graph.research.json';W.save(gp,graph,True);g=lib.ingest(graph,str(gp.relative_to(ROOT)),kind='registered')['sha256']
    registration={'report_sha256':ref,'graph_sha256':g,'feature_id':cards[0]['id'],'features':10,'edges':10,
        'client_url':f'http://127.0.0.1:5173/workbench?lens=process&station=plan&run={RUN}'}
    W.save(OUT/'registration.json',registration,True);status('registration_validated',**registration)
    W.log('paired_published',**registration)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','run','publish']);a=parser.parse_args()
    {'prepare':prepare,'run':run,'publish':publish}[a.stage]()
