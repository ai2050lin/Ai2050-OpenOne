"""Publish scoped real evidence, failures, native locators and MLP-first plan."""
import sys,json,collections
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import phase2794_materials as P
from phase2794_run import status
from shared.dist_protocol.mechanism_trace import TraceStore,digest,file_sha,unbits
from shared.dist_protocol.research_registry import Registry
from deploy.research_library import Library

PLAN=[
 {'title':'原生单元优先，作为研究主线','detail':'Qwen3-4B共36×9728=350,208个原生MLP单元。先分层普查和分组解释，保留SAE/transcoder作同预算对照；不把“原生一定干净”或“一单元一概念”作为结论。已有GF-L30-U9517/U80使用256token窗口且未通过项目全部判据，不能算本轮长文本复现。'},
 {'title':'自然长语料两极、弱响应与门控分解','detail':'按独立文档划分发现/确认/锁箱；512–2048token多领域长文采集G、V、A=SiLU(G)×V及写出。记录正负两极、G/V四象限、范数/RMS、弱响应、频率、近零与背景。原生全阵列不截断；Top例子仅作检索。先估计稀有强响应率，再按每极有效事件和独立文档数分配语料预算，不假定2Mtoken足够。'},
 {'title':'解释必须能预测新例与近似反例','detail':'从自然上下文提出单元功能假设；AI可写标签但不能自动升级证据。冻结新句式、新实体、包含关键词但不满足功能的近似反例，以及不含关键词的语义正例；独立划分检验两极和读入/写出。未知、粗粒度、多用途和无法复现都合法保留。'},
 {'title':'从读入走到真实写出和上下游','detail':'区分gate/up读取、down写出、残差/RMS、Attention提供的信息和后续神经元的读入。读写方向不必相同，不把它们不平行单独当作失败。沿参数乘积和实际响应构建类型化边，再用剂量、同层/同能量随机对照与交互消融检验作用。'},
 {'title':'用预测保真度和能力保持验收','detail':'先以可解释且可重复的小型MLP子图为交付，不承诺先破解全部MLP。比较native与同k字典的激活预测、完整词表KL/损失恢复和未解释残差；预测时不能读取目标真值状态。持续做能力保持和生成保义检查，向Attention/残差追踪缺失来源，最后顺序跨模型与独立节点复算。'}]

def main():
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    c=read(P.OUT/'formal-corpus.json');obs=read(P.OUT/'observations.json');a=read(P.OUT/'analysis.json');cap=read(P.OUT/'capability-result.json')
    freeze=read(P.OUT/'execution-freeze.json');assert digest(c)==freeze['corpus_sha256']
    assert all(file_sha(P.ROOT/n)==h for n,h in freeze['dependencies'].items())
    lib=Library(P.ROOT/'distributed_data/research_library');reg=Registry(lib);store=TraceStore();prov=obs['provenance'];refs={}
    def archive(name,doc):
        path=P.PUBLIC/(name+'.research.json');P.save(path,doc,True);return lib.ingest(doc,str(path.relative_to(P.ROOT)),kind='local_report')['sha256']
    for key,name in [('observations','observations.json'),('interventions','interventions.json'),('source_free','source-free-priors.json')]:
        refs[key]=archive(key,{'schema':'research_raw_observations.v1','kind':'real','run_id':P.RUN,'model_snapshot':c['model_snapshot'],'data':read(P.OUT/name),'provenance':prov})
    refs['inputs']=[]
    for start in range(0,len(c['items']),64):
        refs['inputs'].append(archive(f'inputs-{start//64:02d}',{'schema':'research_frozen_inputs.v1','kind':'real','run_id':P.RUN,'model_snapshot':c['model_snapshot'],'items':c['items'][start:start+64],'provenance':prov}))
    files=total=0;maxerr=0.;allmanifests=[];byid={r['id']:r for r in obs['rows']};byitem={r['id']:r for r in c['items']}
    for row in obs['rows']:
        for identity in [row['trace_id']]+row['generation_trace_ids']:
            d,_=store.manifest(identity);allmanifests.append(d)
            for f in d['files']:
                store.native_path(identity,f['path']);files+=1;total+=f['bytes']
            maxerr=max(maxerr,max(v['attention_relative_l2'] for v in d['reconstruction'].values()))
    assert maxerr<.05
    refs['manifests']=[]
    for i in range(0,len(allmanifests),32):
        refs['manifests'].append(archive(f'manifests-{i//32:02d}',{'schema':'research_native_manifests.v1','kind':'real','run_id':P.RUN,'model_snapshot':c['model_snapshot'],'manifests':allmanifests[i:i+32],'provenance':prov}))
    native={'files':files,'bytes':total,'all_file_hashes_verified':True,'prefix_traces':704,'generation_steps':len(allmanifests)-704,'max_attention_relative_l2':maxerr,
        'local_tensor_files_only':True,'scope':'all37 hidden layers/all36 blocks complete readout fields; deep8 full-token prefix; current-token cached generation traces'}
    refs['capability']=archive('capability',{'schema':'research_capability_monitor.v1','kind':'real','run_id':P.RUN,'model_snapshot':c['model_snapshot'],
        'base':read(P.OUT/'capability-base.json'),'edited':read(P.OUT/'capability-selected_units.json'),'result':cap,'contract':read(P.OUT/'capability-contract.json'),'provenance':prov})
    nc=read(P.OUT/'numeric-control.json')
    refs['numeric_control']=archive('numeric-control',{'schema':'research_numeric_control.v1','kind':'real','run_id':P.RUN,'model_snapshot':c['model_snapshot'],
        'result':nc,'contract':read(P.OUT/'numeric-control-contract.json'),'provenance':prov})
    versions=[]
    for version,folder in [(1,'phase2794_arbitration'),(2,'phase2794_arbitration_v2'),(3,'phase2794_arbitration_v3'),(4,'phase2794_arbitration_v4')]:
        p=P.ROOT/'tests/glm5/result'/folder;d=read(p/'pilot-result.json')
        versions.append({'version':version,'passed':d['passed'],'cells':d['cells'],'scope':'trusted-reference-assisted' if version==4 else 'no real-reference card',
            'artifact_sha256':archive(f'pilot-v{version}',{'schema':'research_calibration.v1','kind':'real','run_id':P.RUN,'model_snapshot':c['model_snapshot'],
                'version':version,'source_file_sha256':file_sha(p/'pilot-corpus.json'),'corpus':read(p/'pilot-corpus.json'),'result':d,'provenance':prov})})
    # The one examined block retains full signed scope-response profiles.
    block=a['profile_block'];train=[r for r in obs['rows'] if r['split']=='discovery'];held=[r for r in obs['rows'] if r['split']=='unseen_combination']
    profile_arrays={};unitcards=[]
    for split,rows in [('discovery',train),('heldout',held)]:
        v={s:[] for s in ['文档','现实']};h={s:[] for s in v}
        for row in rows:
            z=store.block(row['trace_id'],block);v[row['scope']].append(unbits(z['activation'][-1]));h[row['scope']].append(store.hidden(row['trace_id'],block+1,row['length']-1)['values'])
        for space,data,dim in [('mlp',v,9728),('hidden',h,2560)]:
            arr=np.zeros((36,dim),np.float32);arr[block]=np.mean(data['文档'],0)-np.mean(data['现实'],0);profile_arrays[space+'_'+split+'_delta']=arr
    np.savez(P.OUT/'native-profiles.npz',**profile_arrays,profile_block=block)
    selection=read(P.OUT/'selection.json');counter=[]
    from server.mechanism_trace_service import contributions
    example_row=next(r for r in obs['rows'] if r['split']=='unseen_combination' and r['frame']==2 and r['scope']=='文档' and byitem[r['id']]['id'] in read(P.PUBLIC/'prereg.research.json')['deep_ids'])
    parameter_refs={};parameter_heads={}
    for l,u in selection['units']:
        h=next((h for a,h in selection['heads'] if a==l),0)
        connection=contributions(example_row['trace_id'],l,example_row['length']-1,h,u,0)
        assert connection['verified_full_snapshot']
        parameter_refs[(l,u)]=archive(f'connection-B{l}-U{u}',{**connection,'run_id':P.RUN,'provenance':prov})
        parameter_heads[(l,u)]=int(np.argmax(np.abs(connection['head_gate_terms'])))
    for r in obs['rows']:
        if not r['correct']:counter.append({'sample_id':r['id'],'reason':'complete_integer_incorrect','trace_id':r['trace_id'],'expected':r['expected'],'generated':r['generated'],'scope':r['scope']})
    for r in read(P.OUT/'interventions.json')['rows']:
        if byid[r['sample_id']]['correct'] and not r['correct']:counter.append({'reason':'intervention_regression','trace_id':byid[r['sample_id']]['trace_id'],**r})
    for r in a['output_prediction']['rows']:
        if r['predicted_first_token_id']!=r['actual_first_token_id']:counter.append({'reason':'graph_first_token_mismatch','trace_id':byid[r['sample_id']]['trace_id'],**r})
    cb=read(P.OUT/'capability-base.json');ce=read(P.OUT/'capability-selected_units.json')
    for b,e in zip(cb,ce):
        if b['correct'] and not e['correct']:counter.append({'sample_id':b['id'],'reason':'capability_regression' if b['gate_eligible'] else 'reference_only_change','base':b,'edited':e})
    ncb={r['sample_id']:r for r in nc['rows'] if r['condition']=='base'}
    for r in nc['rows']:
        if r['condition']!='base' and ncb[r['sample_id']]['correct'] and not r['correct']:counter.append({'reason':'no_conflict_numeric_regression',**r})
    examples=[]
    unit=next(u for l,u in selection['units'] if l==block)
    for item in c['items']:
        if item['id'] not in read(P.PUBLIC/'prereg.research.json')['deep_ids']:continue
        r=byid[item['id']];examples.append({'entity':r['entity'],'scope':r['scope']+'（有参考）' if r['scope']=='现实' else r['scope'],'length':r['length'],'trace_id':r['trace_id'],
            'url':f'?lens=spatial&view=live&trace={r["trace_id"]}&tblock={block}&thead={selection["heads"][0][1]}&tunit={unit}'})
    source_names=list(freeze['dependencies'])+['tests/glm5/phase2794_freeze.py','tests/glm5/phase2794_pilot.py','tests/glm5/phase2794_capability.py',str(Path(__file__).relative_to(P.ROOT))]
    refs['sources']=archive('sources-'+file_sha(Path(__file__))[:10],{'schema':'research_source_bundle.v1','kind':'real','run_id':P.RUN,'model_snapshot':c['model_snapshot'],
        'files':[{'path':n,'sha256':file_sha(P.ROOT/n),'text':(P.ROOT/n).read_bytes().decode('utf-8')} for n in source_names],
        'large_local_arrays':[{'path':str(p.relative_to(P.ROOT)),'sha256':file_sha(p),'bytes':p.stat().st_size} for p in [P.OUT/'next-logits.npy',P.OUT/'source-free-logits.npy',P.OUT/'graph-native-prediction.npz']],
        'reference_data':{'url':P.SOURCE,'sha256':c['reference_file_sha256']},'provenance':prov})
    limits=['v1/v2/v3无现实参考卡的校准失败；v4的通过只属于有参考来源仲裁，不属于闭卷知识记忆。','主行为是完整贪心生成整数；DLA和图谱输出预测仅下一token（首位数字），两者分开。',
        '未见组合是熟悉实体的新虚构值和新句式；新实体仅测模型行为，未宣称图谱知识泛化。','预测器是条件统计代理，不是所有边都经参数/因果闭合的完整机制图。',
        '系统指令先于正文，不沿用P2793的早位置未来线索不可见负对照。','背景为作者受控重复篇章，长度增加伴随内容增加。','32单元按答案首位贡献选择；可能涉及数字/格式而不是范围选择，未证明单个单元必要或语义单一。',
        '能力检验为旧题池新操作重测、医学占比较高；140参考题不入门。','仅一物理节点和一个模型；无跨模型与独立节点复算。','未完成自然长语料神经元全体语义普查，本轮不是Goodfire全部结果复现。']
    stronger=read(P.OUT/'strong-baselines.json')
    limits.append('强外部来源规则基线是看到主分析后增加的审查；其训练只用发现集，但比较属于事后审查，不改主判、不冒充新的确认测试。')
    report={'schema':'research_atomic_arbitration_report.v1','kind':'real','run_id':P.RUN,'model_snapshot':c['model_snapshot'],'n_inputs':704,'strong_baselines':stronger,
        'analysis':a,'capability':cap,'numeric_control':nc,'calibration_versions':versions,'counterexamples':counter,'examples':examples,'native':native,'selection':selection,
        'predictor':read(P.OUT/'graph-predictor.json'),'mlp_plan':PLAN,'limitations':limits,'reference_source':P.SOURCE,
        'preregistration':read(P.PUBLIC/'prereg.research.json'),'freeze':freeze,'execution':read(P.OUT/'execution.json'),'artifact_refs':refs,'provenance':prov}
    rr=archive('report',report);cards=[];edges=[]
    for l,u in selection['units']:
        fid=f'{c["model_snapshot"]}:B{l}:U{u}:P2794-answer-support'
        cards.append({'id':fid,'label':f'首位数字答案贡献候选 B{l}/U{u}','layer':l,'boundary':'mlp.down_proj.input','index':{'space':'mlp','unit':u},
            'run_id':P.RUN,'method':'discovery native G/V/A and direct first-digit contribution, joint32-unit doses in heldout',
            'scope':{'representative_unit_only':True,'joint_operation':P.RUN+':MLP32','individual_necessity':'not_tested','reference_assisted':True,
                'trace_id':example_row['trace_id'],'position':example_row['length']-1,'parameter_connection_sha256':parameter_refs[(l,u)]},
            'evidence':{'level':'intervention' if a['H2_supported'] else 'observation','artifact_sha256':rr},'counterexamples':counter[:8]})
        edges.append({'id':fid+':scope','source':'language:reference-assisted-source-arbitration','target':fid,'relation':'joint32_operation_not_individual_semantics',
            'run_id':P.RUN,'evidence':{'level':'intervention' if a['H2_supported'] else 'negative','artifact_sha256':rr}})
        head=parameter_heads[(l,u)];hid=f'{c["model_snapshot"]}:B{l}:Head{head}:P2794-parameter-source'
        if not any(x['id']==hid for x in cards):
            cards.append({'id':hid,'label':f'样本局部Gate来源 B{l}/Head{head}','layer':l,'boundary':'self_attn.o_proj.input',
                'index':{'space':'attention','head':head},'run_id':P.RUN,'method':'actual weights and native response with sample RMS fixed; all32 terms retained, displayed largest absolute term',
                'scope':{'trace_id':example_row['trace_id'],'position':example_row['length']-1,'individual_necessity':'not_tested'},
                'evidence':{'level':'parameter_source','artifact_sha256':parameter_refs[(l,u)]},'counterexamples':[]})
        edges.append({'id':fid+':parameter','source':hid,'target':fid,'relation':'actual_parameter_local_gate_contribution_not_total_causality',
            'run_id':P.RUN,'evidence':{'level':'parameter_source','artifact_sha256':parameter_refs[(l,u)]}})
    gr=archive('graph',{'schema':'research_graph.v1','kind':'real','run_id':P.RUN,'model_snapshot':c['model_snapshot'],'feature_cards':cards,'graph_edges':edges,
        'independent_primary_joint_tests':1,'individual_neuron_mechanisms_proven':0,'provenance':prov})
    did=P.RUN+':corpus';reg.register({'schema':'dataset_version.v1','kind':'real','id':did,'dataset_id':'reference-assisted-atomic-arbitration','version':'4',
        'content_sha256':freeze['corpus_sha256'],'source':P.SOURCE+' + authored long conflict documents','license':'PubChem US government factual reference + project-authored text',
        'label_status':'unique atomic number with explicit trusted reference; not colour andnot closed-book recall','role':'controlled_mechanism',
        'source_groups':[e['id'] for e in c['entities']],'sample_count':704,'provenance':prov})
    for metric,supported in [('reference_assisted_behavior',a['behavior_gate']),('joint32_MLP',a['H2_supported']),('next_token_vs_scope_blind',a['output_prediction']['supported'])]:
        reg.register({'schema':'research_claim.v1','kind':'real','id':P.RUN+':'+metric,'canonical_feature_id':P.RUN+':MLP32' if metric=='joint32_MLP' else P.RUN+':'+metric,
            'model_snapshot':c['model_snapshot'],'external_pattern_id':'language:reference-assisted-source-arbitration','outcome_metric':metric,
            'scope':'reference-assisted atomic-number single-entity documents; separate full-integer behavior andfamiliar-entity first-token prediction',
            'status':'supported_in_scope' if supported else 'not_supported','dataset_versions':[did],'artifact_sha256':rr,'run_id':P.RUN,'provenance':prov})
    for card in [c for c in cards if c['index']['space']=='mlp']:reg.register({'schema':'research_feature_alias.v1','kind':'real','id':'alias:'+digest({'graph':gr,'card':card['id']}),
        'canonical_feature_id':P.RUN+':MLP32','legacy_feature_id':card['id'],'model_snapshot':c['model_snapshot'],'artifact_sha256':gr,'relation':'member_locator'})
    original=read(P.ROOT/'research/phase2793/unified-evidence.research.json')
    acc={s:float(np.mean([r['correct'] for r in obs['rows'] if r['split']=='unseen_combination' and r['scope']==s])) for s in ['文档','现实']}
    context={'schema':'unified_research_context.v1','kind':'real','run_id':P.RUN,'model_snapshot':c['model_snapshot'],'report_sha256':rr,'graph_sha256':gr,
        'report_summary':{'behavior_gate':a['behavior_gate'],'scope_accuracy':acc,'H2_supported':a['H2_supported']},'historical_evidence':original['historical_evidence'],
        'reference_mode':'现实范围提供独立参考卡；不等于闭卷知识回忆','profiles':{str(block):{'unit':unit}},'examples':examples,
        'n_discovery_entities':64,'n_heldout_entities':64,'heldout_profile_label':'熟悉实体的新组合/新句式','provenance':prov}
    archive('unified-evidence',context)
    registration={'report_sha256':rr,'graph_sha256':gr,'client_url':f'http://127.0.0.1:5173/workbench?lens=process&station=plan&run={P.RUN}',
        'trace_url':'http://127.0.0.1:5173/workbench'+examples[0]['url'],'native':native}
    P.save(P.OUT/'client-registration.json',registration,True)
    status('validation',behavior_gate=a['behavior_gate'],H2_supported=a['H2_supported'],state='done' if a['H2_supported'] else 'not_supported')
    status('capability',state=cap['status'],n=cap['n'],net_loss=cap['net_loss']);status('registration_validated',**registration)
    print(json.dumps({'behavior_gate':a['behavior_gate'],'H2':a['H2_supported'],'output_prediction':{k:v for k,v in a['output_prediction'].items() if k!='rows'},'internal_prediction':a['internal_prediction'],'capability':cap,'registration':registration},ensure_ascii=False))
if __name__=='__main__':main()
