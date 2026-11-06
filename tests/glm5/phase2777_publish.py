"""Verify real immutable data, archive natural-generation evidence and graph links.

No inference, discovery selection, threshold changes, or evidence promotion.
"""
from pathlib import Path
import collections
import copy
import json
import subprocess
import sys
import time
import numpy as np
from scipy.stats import beta
sys.path.insert(0,str(Path(__file__).parent))
import phase2777_confirm as C
import phase2773_w6_loop as W


def main():
    from deploy.research_library import Library
    from deploy.parameter_reader import model_snapshot
    from tools import provenance as prov
    root=W.ROOT;out=C.OUT;public=C.PUBLIC;lib=Library(root/'distributed_data/research_library')
    r=W.read(out/'result.json');freeze=W.read(out/'freeze.json');rows=W.read(out/'corpus.json')
    assert all(W.sha(root/p)==h for p,h in freeze['dependencies'].items())
    assert W.sha(out/'corpus.json')==freeze['corpus_sha256']
    assert W.sha(public/'prereg.json')==freeze['design_sha256']
    assert model_snapshot(W.MODEL)['sha256']==r['model_snapshot']==r['original_model_snapshot_after']
    assert len(rows)==512 and len({x['family_id'] for x in rows})==128
    assert {x['record'] for x in rows}==set(range(3000,3128))
    pairs={}
    for x in rows:pairs.setdefault((x['family_id'],x['frame']),[]).append(x)
    mismatch=0
    for group in pairs.values():
        assert len(group)==2
        a,b=group
        mismatch+=int(len(a['token_ids'])!=len(b['token_ids']))
        assert a['tokens']==b['tokens']==a['length']==len(a['token_ids'])
        assert a['fact']==b['fact']
        assert a['text'].replace(a['matching']['register_request'],'')==b['text'].replace(b['matching']['register_request'],'')
        assert a['matching']['common_neutral_prefix']==b['matching']['common_neutral_prefix']
    assert not mismatch
    pilot_refs=[];pilots=[]
    for suffix in ['', '_v2','_v3','_v4','_v5']:
        folder=root/f'tests/glm5/result/phase2777_natural{suffix}'
        p=W.read(folder/'pilot-result.json');pilots.append({k:p[k] for k in ['run_id','passed','style_accuracy','content_accuracy','calibration']})
        assert not p['passed']
        assert {x['family_id'] for x in W.read(folder/'pilot-corpus.json')}.isdisjoint(x['family_id'] for x in rows)
        pilot_doc={'schema':'research_natural_register_report.v1','kind':'real','run_id':p['run_id'],
            'title':'自然语域校准未通过 · 失败历史','model_snapshot':r['model_snapshot'],
            'result':None,'pilots':[pilots[-1]],'pilot_full_result':p,'preregistration':W.read(root/f'research/phase2777/{"v1" if not suffix else suffix[1:]}/pilot-prereg.json')
                if (root/f'research/phase2777/{"v1" if not suffix else suffix[1:]}/pilot-prereg.json').exists() else W.read(root/'research/phase2777/pilot-prereg.json'),
            'freeze':W.read(folder/'pilot-freeze.json'),'source_result_sha256':W.sha(folder/'pilot-result.json'),
            'inputs':W.read(folder/'pilot-corpus.json'),
            'limitations':['发现集试跑，不能作为正式推广结论；两向语域行为门未通过。']}
        path=public/'pilot-history'/f'{p["run_id"]}.research.json';W.save(path,pilot_doc,True)
        pilot_ref=lib.ingest(pilot_doc,str(path.relative_to(root)),kind='local_report')['sha256']
        pilot_refs.append({'run_id':p['run_id'],'sha256':pilot_ref})
        progress_path=root/f'research/phase2777/{"" if not suffix else suffix[1:]+"/"}00-progress.research.json'
        progress=W.read(progress_path)
        progress.update(parent_run_id=C.RUN,title=f'W6 校准 {"v1" if not suffix else suffix[1:]} · 双向语域行为门未通过',updated_at=time.time())
        progress['stages']['registration_validated']={'state':'done','time':time.time(),'report_sha256':pilot_ref}
        W.save(progress_path,progress);lib.ingest(progress,str(progress_path.relative_to(root)),kind='local_report')
    native=W.read(out/'native/manifest.json');native_ids=[];native_bytes=0;preview=[]
    edit=W.read(out/'legacy512_a6-edit.json');units={x['layer']:x['columns'][0] for x in edit['tensors']}
    byid={x['id']:x for x in rows};preview_ids={rows[0]['id'],rows[1]['id']}
    for obj in native['files']:
        path=out/'native'/obj['path'];assert W.sha(path)==obj['sha256'];native_bytes+=path.stat().st_size
        z=np.load(path);native_ids.extend(z['ids'].tolist())
        assert z['hidden_bits'].dtype==np.uint16 and z['hidden_bits'].shape[1:]==(37,2,2560)
        assert z['mlp_bits'].dtype==np.uint16 and z['mlp_bits'].shape[1:]==(10,2,9728)
        for j,key in enumerate(z['ids'].tolist()):
            if key not in preview_ids:continue
            row=byid[key];record={k:row[k] for k in ['id','text','tokens','split']}
            record.update(entity=row['family_id'],responses={},activation_by_layer={})
            for n,l in enumerate(W.LAYERS):
                values=(z['mlp_bits'][j,n,-1].astype(np.uint32)<<16).view(np.float32)
                doc={'schema':'research_mlp_response.v1','kind':'real','run_id':C.RUN,'model_snapshot':r['model_snapshot'],
                    'block':l,'boundary':'mlp.down_proj.input','readout':'final','selected_unit':units[l],
                    'sample_id':key,'text':row['text'],'tokens':row['tokens'],'values':values.tolist(),
                    'response_state':'unedited baseline','reconstruction':'lossless BF16 bits expanded to float32',
                    'native_manifest_sha256':W.sha(out/'native/manifest.json')}
                path=public/'responses'/f'{key}-B{l}.research.json';W.save(path,doc,True)
                record['responses'][str(l)]=lib.ingest(doc,str(path.relative_to(root)),kind='local_report')['sha256']
                record['activation_by_layer'][str(l)]=float(values[units[l]])
            preview.append(record)
    assert len(native_ids)==len(set(native_ids))==512 and set(native_ids)==set(byid)
    assessed={name:W.read(out/f'{name}-assessed.json') for name in ['base','legacy512_a6','legacy_random_direction','legacy_energy_columns']}
    for name,a in assessed.items():assert [x['id'] for x in a]==[x['id'] for x in rows]
    base=assessed['base'];edited=assessed['legacy512_a6']
    qualified=np.array([b['style']=='acad' and x['style']=='colloq' and x['content']['pass'] and not x['hit_limit'] for b,x in zip(base,edited)])
    effect=C.ci(C.D.family_mean(rows,qualified,'acad'))
    assert effect==r['hypotheses']['legacy512_a6']['qualified_flip']
    bc=W.read(out/'capability-base.json');ec=W.read(out/'capability-edited.json')
    qs=W.read(C.OLD/'capability-inputs.json')
    assert [x['id'] for x in bc]==[x['id'] for x in ec]==[x['id'] for x in qs]
    cap=W.paired_interval([x['correct'] for x in bc],[x['correct'] for x in ec],np.array([x['gate_eligible'] for x in bc]))
    assert cap==r['hypotheses']['legacy512_a6']['capability']
    source_cap={s:W.paired_interval([x['correct'] for x in bc],[x['correct'] for x in ec],np.array([x['source']==s for x in bc])) for s in sorted({x['source'] for x in bc})}
    matching={'pairs':len(pairs),'mismatched_pairs':mismatch,'lengths':[384,512],'families':128,
        'heldout_frames':[2,3],'families_per_length':64,'common_facts_and_other_text_identical':True,
        'equal_active_tokens_not_padding':True,'scope':'384/512-token Chinese factual restatements, not multi-thousand-token natural documents',
        'length_crossing':'length between families, frame within family; no inference of a pure within-family length effect'}
    inputs={'schema':'research_frozen_inputs.v1','kind':'real','run_id':C.RUN,
        'model_snapshot':r['model_snapshot'],'design_sha256':r['design_sha256'],
        'local_file_sha256':freeze['corpus_sha256'],'items':rows}
    path=public/'frozen-inputs.research.json';W.save(path,inputs,True)
    input_ref=lib.ingest(inputs,str(path.relative_to(root)),kind='local_report')['sha256']
    source_bundle={'schema':'research_source_bundle.v1','kind':'real','run_id':C.RUN,
        'model_snapshot':r['model_snapshot'],'execution':'data for review/reproduction; never auto-execute downloaded code',
        'files':[{'path':p,'sha256':h,'text':(root/p).read_bytes().decode('utf-8')}
            for p,h in freeze['dependencies'].items()],
        'runtime_adapter':{'path':'tests/glm5/phase2773_resume.py','sha256':W.sha(root/'tests/glm5/phase2773_resume.py'),
            'text':(root/'tests/glm5/phase2773_resume.py').read_bytes().decode('utf-8'),
            'scope':'ids_only helper imported by the frozen engine; extra archive, not a preregistered dependency hash'},
        'env':r['env']}
    path=public/'sources.research.json';W.save(path,source_bundle,True)
    source_ref=lib.ingest(source_bundle,str(path.relative_to(root)),kind='local_report')['sha256']
    monitor={'schema':'research_capability_monitor.v1','kind':'real','run_id':C.RUN,
        'model_snapshot':r['model_snapshot'],'scope':'repeated fixed monitor; provided keys; M3Exam reference only',
        'input_local_sha256':W.sha(C.OLD/'capability-inputs.json'),'questions':qs,'base':bc,'edited':ec,
        'aggregate':cap,'by_source':source_cap}
    path=public/'capability-monitor.research.json';W.save(path,monitor,True)
    cap_ref=lib.ingest(monitor,str(path.relative_to(root)),kind='local_report')['sha256']
    generation_refs=[]
    for name,a in assessed.items():
        doc={'schema':'research_free_generation.v1','kind':'real','run_id':C.RUN,'condition':name,
            'model_snapshot':r['model_snapshot'],'design_sha256':r['design_sha256'],'outputs':a,
            'source_sha256':W.sha(out/f'{name}-assessed.json')}
        path=public/'generations'/f'{name}.research.json';W.save(path,doc,True)
        generation_refs.append({'condition':name,'sha256':lib.ingest(doc,str(path.relative_to(root)),kind='local_report')['sha256'],'n':len(a)})
    examples=[]
    for frame in [2,3]:
        for length in [384,512]:
            for reg in ['acad','colloq']:
                i=next(i for i,x in enumerate(rows) if (x['frame'],x['length'],x['register'])==(frame,length,reg))
                examples.append({**{k:rows[i][k] for k in ['id','register','frame','length','prompt']},
                    'outputs':{name:a[i] for name,a in assessed.items()}})
    diagnostics=[]
    for name,a in assessed.items():
        for reg in ['acad','colloq']:
            group=[x for x in a if x['register']==reg]
            diagnostics.append({'condition':name,'input_register':reg,'n':len(group),
                'styles':dict(collections.Counter(x['style'] for x in group)),
                'rubric_labels':dict(collections.Counter(x['rubric']['label'] for x in group)),
                'judge_labels':dict(collections.Counter(x['blind_judge']['label'] for x in group)),
                'rater_disagreement':sum(x['rubric']['label']!=x['blind_judge']['label'] for x in group),
                'content_pass':sum(x['content']['pass'] for x in group),'truncated':sum(x['hit_limit'] for x in group),
                'mean_generated_tokens':float(np.mean([x['generated_tokens'] for x in group]))})
    zero_bound=None
    if not qualified.any():zero_bound={'n_families':128,'events':0,'upper95':float(beta.ppf(.95,1,128)),
        'scope':'supplementary exact binomial bound on any qualified flip per independent family in this material population; not a replacement primary test'}
    git=lambda *a:subprocess.check_output(['git','-c',f'safe.directory={root.as_posix()}',*a],cwd=root,text=True).strip()
    provenance={'git_sha':git('rev-parse','HEAD'),'git_dirty':bool(git('status','--porcelain')),
        'model_path':str(W.MODEL),'model_revision':r['model_snapshot'],'seed':W.SEED,
        'analysis_bootstrap_seed':C.M.SEED+99,'control_seed':0,
        'tool':'phase2777_publish.py; full-snapshot provenance adapter','created_at':time.time()}
    assert not prov.check_provenance({'provenance':provenance})
    audit={'kind':'software_and_research_recalculation','passed':True,'run_id':C.RUN,
        'native_samples':512,'native_batches':len(native['files']),'native_bytes':native_bytes,
        'hidden_shape_per_sample':[37,2,2560],'mlp_shape_per_sample':[10,2,9728],
        'original_weights_unchanged':True,'formal_source_hashes_unchanged':True,'matching':matching,
        'capability_recalculated':True,'primary_recalculated':True,'heldout_records_disjoint_from_pilots':True,
        'original_model_snapshot':r['model_snapshot'],'provenance':provenance}
    W.save(out/'verification.json',audit,True)
    review=[{'sample_id':base[0]['id'],'issue':'规则对正式书面输出的假阴性','text':base[0]['text'],
            'description':'原模型盲评为书面，但冻结词表要求至少两个指定学术词，给出unknown；完整文本呈现核验报告语气。这是联合评分器不能跨句式推广的具体反例。'},
        {'sample_id':base[1]['id'],'issue':'温和口语未被规则识别','text':base[1]['text'],
            'description':'这事儿挺顺利的，数据都对得上等口语表达获盲评口语标签，却未达到冻结语言特征组数。不得事后改词表将它变为正式成功。'},
        {'sample_id':edited[2]['id'],'issue':'字面内容代理可能漏掉语义错误','text':edited[2]['text'],
            'description':'材料物品为记录册、地点为图书室；输出新增“记录中所涉物品均为图书”。即便同时命中记录册与151，代理也不能证明完整保义；须额外核查实体、角色及增减关系。'}]
    limits=['正式留出的联合评分器行为门失败；原模型盲评认为256/256书面条件输出为书面，联合规则仅77/256通过，不能将本轮失败解释成不存在语域控制。',
        '旧方向继承自句式和长度混杂的上一轮，仅检验该具体编辑能否推广到等长度新句式，不从失败条件提取新方向。',
        '两向语域校准未通过；本轮是单向充分性检验，不证明模型原本存在可分离的自然语域开关。',
        '语域规则与同一未编辑Qwen盲评，不是独立人工或跨模型判据；unknown不算成功。',
        '内容检查是物体与最终数量的自动代理，不能覆盖语义、语用和所有事实错误。',
        '384/512有效token含共同中性背景，只覆盖中文物资记录；不代表跨领域、跨语言或更长篇章。',
        '128新记录家族、两种新句式；词汇仍可能与发现集重叠，不声称未见词汇推广。',
        '能力题池重复使用，医学题占比较大；2200题为提供答案口径，140道M3Exam镜像答案仅参考。',
        '10层联合参数编辑，单层、单元或单参数的独立因果作用未测；展示单元只是定位代表。',
        '原生全坐标仅采集未编辑提示末尾两个位置，不含完整生成轨迹，也不是所有token的全场。',
        '记录由冻结文法确定性展开；家族bootstrap依赖家族可交换近似，不代表对全部自然语言总体的随机抽样。',
        '未执行多节点复算、字典发现或仅靠图谱预测；单个快照的范围内失败不证明不存在门控。']
    report={'schema':'research_natural_register_report.v1','kind':'real','run_id':C.RUN,
        'title':'W6 · 等长度留出句式与自然生成充分性检验','model_snapshot':r['model_snapshot'],
        'result':r,'pilots':pilots,'pilot_artifacts':pilot_refs,'generation_artifacts':generation_refs,
        'preregistration':W.read(public/'prereg.json'),'freeze':freeze,'matching_audit':matching,
        'verification':audit,'examples':examples,'sample_preview':preview,'raw_manifest':native,
        'edit_manifest':edit,'zero_event_bound':zero_bound,'capability_sources':source_cap,'secondary_diagnostics':diagnostics,
        'qualitative_review':review,'claim_status':'inconclusive_due_to_measurement_gate' if not r['baseline_gate'] else 'scoped_primary_test',
        'counterexample_inputs':{x['id']:x for x in qs if x['id'] in {c['sample_id'] for c in r['counterexamples']}},
        'scope':'one joint 10-block edit; primary natural-generation outcome distinct from previous fixed-candidate preference',
        'prior_preference_report_sha256':W.read(root/'tests/glm5/result/phase2773_w6_v3/client-registration.json')['report_sha256'],
        'limitations':limits,'provenance':provenance,
        'inputs_artifact_sha256':input_ref,'source_bundle_sha256':source_ref,'capability_artifact_sha256':cap_ref,
        'inputs_local_sha256':freeze['corpus_sha256'],
        'source_files':{str(p.relative_to(out)):W.sha(p) for p in out.glob('*.json')}}
    path=public/'report.research.json';W.save(path,report,True)
    ref=lib.ingest(report,str(path.relative_to(root)),kind='local_report')['sha256']
    cards=[];edges=[];level='intervention' if r['hypotheses']['legacy512_a6']['supported'] else 'negative'
    for t in edit['tensors']:
        l=t['layer'];fid=f'{r["model_snapshot"]}:B{l}:W6natural:{C.RUN}'
        cards.append({'id':fid,'label':f'W6 自然生成 · B{l} '+('范围内支持' if level=='intervention' else '留出测量门失败，未验收' if not r['baseline_gate'] else '范围未支持'),
            'layer':l,'boundary':'mlp.down_proj.input','index':{'space':'mlp','unit':units[l]},'run_id':C.RUN,
            'method':'preexisting write-direction edit; equal active tokens; new records and frames; free generation',
            'scope':{'intervention_group':C.RUN,'readout':'final','edited_columns':t['columns'],
                'representative_unit_only':True,'individual_layer_causality':'not_tested','joint_tests':1,
                'baseline_measurement_passed':r['baseline_gate'],'claim_status':report['claim_status']},
            'evidence':{'level':level,'artifact_sha256':ref},
            'counterexamples':[{'sample_id':x['sample_id'],'reason':x['reason']} for x in r['counterexamples'][:20]],
            'counterexamples_total':len(r['counterexamples']),
            'counterexamples_full_report_sha256':ref,'summary':{'limitations':limits,'matching':matching},
            'related_prior_preference_report_sha256':report['prior_preference_report_sha256']})
        edges.append({'id':fid+':natural-generation','source':'language:register-free-generation','target':fid,
            'relation':'tested_natural_register_sufficiency' if level=='intervention' else 'unsupported_natural_register_sufficiency',
            'run_id':C.RUN,'evidence':{'level':level,'artifact_sha256':ref}})
    graph={'schema':'research_graph.v1','kind':'real','run_id':C.RUN,'model_snapshot':r['model_snapshot'],
        'feature_cards':cards,'graph_edges':edges,'independent_intervention_tests':1,'provenance':provenance}
    path=public/'graph.research.json';W.save(path,graph,True)
    gr=lib.ingest(graph,str(path.relative_to(root)),kind='registered')['sha256']
    registration={'report_sha256':ref,'graph_sha256':gr,'feature_id':cards[0]['id'],'features':10,'edges':10,
        'independent_tests':1,'response_arrays':len(preview)*10,'counterexamples':len(r['counterexamples']),
        'client_url':f'http://127.0.0.1:5173/workbench?lens=process&station=plan&run={C.RUN}'}
    W.save(out/'client-registration.json',registration,True);C.status('registration_validated',**registration)
    print(json.dumps({'verification':audit,'registration':registration},ensure_ascii=False))


if __name__=='__main__':main()
