"""Register a complete exploratory study, preserving blind disagreement and failures."""
import collections
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import phase2790_research as P
from deploy.research_library import Library
from shared.dist_protocol.research_registry import Registry


def interval(values):
    x=np.asarray(values,float);rng=np.random.default_rng(27901004)
    means=x[rng.integers(0,len(x),(10000,len(x)))].mean(1)
    return {'mean':float(x.mean()),'ci95':np.quantile(means,[.025,.975]).tolist(),'n_families':len(x),'inference':'descriptive discovery-family bootstrap, not confirmatory'}


def archive(lib,name,doc):
    path=P.PUBLIC/(name+'.research.json');P.W.save(path,doc,True)
    return lib.ingest(doc,str(path.relative_to(P.W.ROOT)),kind='local_report')['sha256']


def main():
    W=P.W;lib=Library(W.ROOT/'distributed_data/research_library');reg=Registry(lib);c=W.read(P.OUT/'corpus.json')
    private=W.read(P.OUT/'generation-private-map.json');byid={x['id']:x for x in c['items']};raters={};rating_refs=[]
    blind=W.read(P.OUT/'generation-blind.json');blind_ref=archive(lib,'generation-blind',{'kind':'real',**blind,'provenance':c['provenance']})
    cid='P2790-AI-generation-v1';reg.create_ai_review({'schema':'external_ai_review_campaign.v1','kind':'real','id':cid,
        'rater_models':['gpt-6-astra','gpt-6.1-sol'],'input_sha256':W.sha(P.OUT/'generation-blind.json'),
        'blind_artifact_sha256':blind_ref,'n_items':len(blind['records']),'human_validation':False})
    for suffix in ['astra','sol']:
        path=P.OUT/f'generation-{suffix}.json';doc=W.read(path)
        assert doc['input_sha256']==W.sha(P.OUT/'generation-blind.json')
        assert len(doc['ratings'])==len({x['id'] for x in doc['ratings']})==len(blind['records'])
        raters[suffix]={x['id']:x for x in doc['ratings']}
        doc={**doc,'kind':'real','provenance':{**c['provenance'],'model_path':'collaboration:'+doc['rater_model'],
            'model_revision':'provider_alias:'+doc['rater_model'],'seed':None},'rated_model_snapshot':c['model_snapshot'],
            'rater_limit':'distinct provider model aliases; exact checkpoint/seed unavailable; correlated AI error remains possible',
            'source_rating_file_sha256':W.sha(path)}
        rating_refs.append({'rater_model':doc['rater_model'],**reg.submit_ai_review(cid,doc)})
    rows=[]
    for r in private:
        a,b=[raters[k][r['blind_id']] for k in ['astra','sol']]
        style=a['style'] if a['style']==b['style'] and min(a['naturalness'],b['naturalness'])>=3 else 'uncertain'
        meaning=a['meaning']==b['meaning']=='preserved'
        rows.append({**r,'consensus_style':style,'consensus_meaning':meaning,'rater_disagreement':(a['style'],a['meaning'])!=(b['style'],b['meaning']),
            'ratings':{'astra':a,'sol':b}})
    lookup={(r['sample_id'],r['condition']):r for r in rows};base=[r for r in rows if r['condition']=='base']
    stats=[];cells=[]
    for cond in P.N.CONDITIONS:
        for requested in ['acad','colloq']:
            group=[r for r in rows if r['condition']==cond and r['register']==requested];desired={'acad':'formal','colloq':'spoken'}[requested]
            stats.append({'condition':cond,'requested_register':requested,'n':len(group),'style_accuracy':sum(r['consensus_style']==desired for r in group)/len(group),
                'meaning_preserved':sum(r['consensus_meaning'] for r in group)/len(group),
                'styles':dict(collections.Counter(r['consensus_style'] for r in group)),'disagreements':sum(r['rater_disagreement'] for r in group)})
    for domain in P.M.DOMAINS:
        for frame in [0,1]:
            for length in [512,1024,2048]:
                for requested,desired in [('acad','formal'),('colloq','spoken')]:
                    group=[r for r in base if (r['domain'],r['frame'],r['length'],r['register'])==(domain,frame,length,requested)]
                    assert len(group)==2
                    cells.append({'domain':domain,'frame':frame,'length':length,'register':requested,'n':2,
                        'style_passed':sum(r['consensus_style']==desired for r in group),'meaning_passed':sum(r['consensus_meaning'] for r in group)})
    baseline_stats=[s for s in stats if s['condition']=='base'];meaning_rate=np.mean([r['consensus_meaning'] for r in base])
    gate=all(s['style_accuracy']>=.7 for s in baseline_stats) and meaning_rate>=.7 and all(x['style_passed']>0 for x in cells)
    controls=[]
    gold={x['id']:x for x in W.read(P.OUT/'semantic-controls-private.json')}
    for suffix in ['astra','sol']:
        labels=W.read(P.OUT/f'semantic-controls-{suffix}.json')['ratings']
        acc=np.mean([x['meaning']==gold[x['id']]['meaning'] for x in labels])
        negative=[x for x in labels if gold[x['id']]['meaning']=='changed']
        sensitivity=np.mean([x['meaning']=='changed' for x in negative])
        controls.append({'rater':suffix,'n':len(labels),'meaning_accuracy':float(acc),'corruption_sensitivity':float(sensitivity),
            'known_corruptions':len(negative),'passed':bool(acc>=.9 and sensitivity>=.9)})
    families=sorted({r['family_id'] for r in base});effects=[]
    for cond in ['target','random_direction','random_columns']:
        values=[]
        for family in families:
            g=[r for r in base if r['family_id']==family and r['register']=='acad']
            values.append(np.mean([r['consensus_style']=='formal' and r['consensus_meaning'] and lookup[(r['sample_id'],cond)]['consensus_style']=='spoken'
                and lookup[(r['sample_id'],cond)]['consensus_meaning'] for r in g]))
        effects.append({'condition':cond,'qualified_formal_to_spoken':interval(values)})
    native=W.read(P.OUT/'native-audit.json');assert native['edit_hashes_match_previous_capability_run']
    old=W.read(W.ROOT/'tests/glm5/result/phase2777_confirm/result.json');cap=old['hypotheses']['legacy512_a6']['capability']
    capability={'reused':True,'not_fresh':True,'exact_bf16_hashes_match':True,'matched_block_instances':80,
        'source_run_id':old['run_id'],'source_file_sha256':W.sha(W.ROOT/'tests/glm5/result/phase2777_confirm/result.json'),
        'result':cap,'scope':'Same fixed 2200 keyed MCQs, same full model snapshot, BF16 and exact target edited blocks; 140 unverified reference questions excluded. Does not cover new domains or open-ended preservation.'}
    errors=[]
    for r in rows:
        desired={'acad':'formal','colloq':'spoken'}[r['register']]
        if not r['consensus_meaning'] or r['rater_disagreement'] or r['consensus_style']!=desired:
            errors.append({k:r[k] for k in ['sample_id','condition','family_id','domain','frame','length','register','text','consensus_style','consensus_meaning','rater_disagreement','ratings']})
    raw_ref=archive(lib,'generations',{'schema':'research_ai_assessed_generations.v1','kind':'real','run_id':P.RUN,
        'model_snapshot':c['model_snapshot'],'outputs':rows,'provenance':c['provenance']})
    input_ref=archive(lib,'inputs',{'schema':'research_frozen_inputs.v1','kind':'real','run_id':P.RUN,
        'model_snapshot':c['model_snapshot'],'items':c['items'],'provenance':c['provenance']})
    native_ref=archive(lib,'native-audit',native)
    execution=W.read(P.OUT/'execution.json');execution_ref=archive(lib,'execution',execution)
    limitations=['本轮8个内容家族是发现/评分校准，未达到预注册后续128家族确认集规模；区间只作描述。',
        'AI评审不等于人工金标准。模型别名不同、任务独立，但供应商相关偏差和版本不可完全控制；分歧不强行裁决。',
        '语义破坏对照仅覆盖负责人、时间和限制反转，不能证明全部保义错误可检测。',
        '材料是四领域人工设计的受控长文档，使用重复篇章框架；不是自然语料。长度增加也增加背景内容，不能断言纯长度效应。',
        '语域对在相同家族/句式/长度内，核心事实、其他输入和有效token数完全相等；句式、长度与领域仍仅在有限范围内覆盖。',
        '被测方向继承自旧实验，未从本轮失败重新选参；一次alpha6联合编辑不是最小编辑或自然生成剂量曲线。旧固定候选剂量不冒充新自然生成剂量。',
        '能力保持复用同一精确BF16编辑的2200道固定题监控，不是新盲测，也不等于开放式生成能力保持。',
        '真实中心与节点完成8分片，但仅一台物理节点；独立第二节点复算和异构硬件一致性仍待测。',
        '原场含37层HiddenState、B18..27全部9728MLP单元及每个实际生成步；提示只保存五个位置，不是所有提示token。',
        '生成分叉后base/target前缀不同；同序号步状态比较是观察，不能直接当同条件因果效应。',
        '10个层卡是同一次联合操作的定位入口，不是10项独立机制。未完成字典发现、归因计算边或仅图谱预测。']
    report={'schema':'research_external_ai_register_report.v1','kind':'real','run_id':P.RUN,'title':'四领域等长度长文本 · 双AI盲评与生成全场',
        'model_snapshot':c['model_snapshot'],'baseline_gate':bool(gate),'baseline_meaning':float(meaning_rate),'baseline_cells':cells,
        'stats':stats,'effects':effects,'n_inputs':96,'n_outputs':384,'n_families':8,'unique_blind_records':len(blind['records']),
        'rater_models':['gpt-6-astra','gpt-6.1-sol'],'ratings':rating_refs,'semantic_controls':controls,
        'unique_style_agreement':sum(raters['astra'][k]['style']==raters['sol'][k]['style'] for k in raters['astra'])/len(raters['astra']),
        'unique_meaning_agreement':sum(raters['astra'][k]['meaning']==raters['sol'][k]['meaning'] for k in raters['astra'])/len(raters['astra']),
        'capability':capability,'counterexamples':errors,'claim_status':'exploratory_gate_passed' if gate else 'inconclusive_baseline_gate_failed',
        'preregistration':W.read(P.PUBLIC/'prereg.research.json'),'freeze':W.read(P.OUT/'freeze.json'),
        'execution_artifact_sha256':execution_ref,'native_audit_sha256':native_ref,'generations_sha256':raw_ref,'inputs_sha256':input_ref,
        'sample_preview':native['sample_preview'],'traces':native['traces'],'native_summary':{k:v for k,v in native.items() if k not in ['files','sample_preview','traces']},
        'next_decision':'Proceed to a newly frozen independent confirmation set only after measurement gaps are fixed; do not select edits on these outcomes.',
        'limitations':limitations,'provenance':c['provenance']}
    report_ref=archive(lib,'report',report);canonical=W.read(P.OUT/'registry-registration.json')['canonical_operation_id'];cards=[];edges=[]
    for l in range(18,28):
        fid=f'{c["model_snapshot"]}:B{l}:AI-register:{P.RUN}'
        cards.append({'id':fid,'label':f'自然语域盲评 · B{l} · 探索候选','layer':l,'boundary':'mlp.down_proj.input',
            'index':{'space':'mlp','unit':c['edit']['columns'][str(l)][0]},'run_id':P.RUN,'method':'frozen prior rank-one write edit; external AI consensus; actual autoregressive capture',
            'scope':{'readout':'generation','intervention_group':canonical,'edited_columns':c['edit']['columns'][str(l)],'representative_unit_only':True,
                'individual_layer_causality':'not_tested','claim_status':report['claim_status']},'evidence':{'level':'observation','artifact_sha256':report_ref},
            'counterexamples':[{'sample_id':e['sample_id'],'reason':'AI style/content disagreement or preservation failure'} for e in errors[:20]],
            'counterexamples_total':len(errors),'counterexamples_full_report_sha256':report_ref})
        edges.append({'id':fid+':ai-scoped','source':'language:register','target':fid,'relation':'exploratory_generation_assessment',
            'run_id':P.RUN,'evidence':{'level':'observation','artifact_sha256':report_ref}})
    graph_ref=archive(lib,'graph',{'schema':'research_graph.v1','kind':'real','run_id':P.RUN,'model_snapshot':c['model_snapshot'],
        'feature_cards':cards,'graph_edges':edges,'independent_intervention_tests':1,'provenance':c['provenance']})
    data=next(x['id'] for x in reg.catalog()['datasets'] if x['body']['dataset_id']=='multidomain-long-v1')
    claim={'schema':'research_claim.v1','kind':'real','id':'claim:'+P.N.digest({'op':canonical,'run':P.RUN,'metric':'external_ai_natural_register'}),
        'canonical_feature_id':canonical,'model_snapshot':c['model_snapshot'],'external_pattern_id':'language:register',
        'outcome_metric':'external_ai_natural_register_discovery','scope':'Four controlled Chinese domains,8 discovery families,F0/F1,512/1024/2048 equal-token pairs; two independent AI aliases, not human gold.',
        'status':'inconclusive','dataset_versions':[data],'artifact_sha256':report_ref,'run_id':P.RUN,'human_validated':False,'provenance':c['provenance']}
    reg.register(claim)
    for card in cards:
        reg.register({'schema':'research_feature_alias.v1','kind':'real','id':'alias:'+P.N.digest({'graph':graph_ref,'card':card['id']}),
            'canonical_feature_id':canonical,'model_snapshot':c['model_snapshot'],'legacy_feature_id':card['id'],'artifact_sha256':graph_ref,
            'relation':'member_locator','run_id':P.RUN,'layer':card['layer'],'boundary':card['boundary'],'provenance':c['provenance']})
    registration={'report_sha256':report_ref,'graph_sha256':graph_ref,'feature_id':cards[0]['id'],'independent_tests':1,
        'client_url':f'http://127.0.0.1:5173/workbench?lens=process&station=plan&run={P.RUN}',
        'parameter_url':f'http://127.0.0.1:5173/workbench?lens=spatial&view=parameters&artifact={graph_ref}&feature={cards[0]["id"]}&pmodel=9a784350de7f8d690b04ffef'}
    W.save(P.OUT/'client-registration.json',registration,True);P.status('ai_review',models=report['rater_models'],unique_records=len(blind['records']))
    P.status('capability',state=cap['status'],reused=True,n=cap['n']);P.status('registration_validated',**registration)
    print({'baseline_gate':bool(gate),'baseline_stats':baseline_stats,'baseline_meaning':float(meaning_rate),'effects':effects,'counterexamples':len(errors),'registration':registration})


if __name__=='__main__':main()
