"""Entity-cluster inference, immutable negative findings, native/history registration."""
from pathlib import Path
import collections
import json
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import phase2793_materials as P
from shared.dist_protocol.mechanism_trace import TraceStore,digest,file_sha,unbits
from deploy.research_library import Library
from shared.dist_protocol.research_registry import Registry
from phase2793_run import status

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def interval(x):
    x=np.asarray(x,float);rng=np.random.default_rng(P.SEED)
    means=x[rng.integers(0,len(x),(20000,len(x)))].mean(1)
    return {'mean':float(x.mean()),'ci95':np.quantile(means,[.025,.975]).tolist(),'n_entities':len(x),'unit':'entity family, paired conditions kept together'}

def main():
    lib=Library(P.ROOT/'distributed_data/research_library');reg=Registry(lib);store=TraceStore()
    c=read(P.OUT/'corpus.json');raw=read(P.OUT/'observations.json');pr=read(P.PUBLIC/'prereg.research.json');rows=raw['rows'];prov=raw['provenance'];snap=c['model_snapshot']
    def archive(name,d):
        p=P.PUBLIC/(name+'.research.json');P.save(p,d,True)
        return lib.ingest(d,str(p.relative_to(P.ROOT)),kind='local_report')['sha256']
    # Large inputs are chunked by explicit order, retaining original IDs and full inputs.
    inputrefs=[]
    for start in range(0,len(c['items']),48):
        part={'schema':'research_frozen_inputs.v1','kind':'real','run_id':P.RUN,'model_snapshot':snap,'items':c['items'][start:start+48],'provenance':prov}
        inputrefs.append(archive(f'inputs-{start//48:02d}',part))
    frozen=archive('frozen-design',{'schema':'research_frozen_design.v1','kind':'real','run_id':P.RUN,'model_snapshot':snap,
        'preregistration':pr,'execution_freeze':read(P.OUT/'execution-freeze.json'),'original_corpus_file_sha256':file_sha(P.OUT/'corpus.json'),
        'input_chunk_sha256':inputrefs,'provenance':prov})
    held=[r for r in rows if r['split']=='heldout_entity_frame'];families=sorted({r['family_id'] for r in held})
    by={(r['family_id'],r['frame'],r['length'],r['scope']):r for r in rows}
    effects=[np.mean([int(by[e,f,l,'文档']['document_choice'])-int(by[e,f,l,'现实']['document_choice']) for f in [2,3] for l in [512,1024,2048]]) for e in families]
    h1=interval(effects);h1['threshold_met']=h1['mean']>=.5 and h1['ci95'][0]>.2
    cells=[]
    for split in ['discovery','heldout_entity_frame']:
        for scope in ['文档','现实']:
            for f in ([0,1] if split=='discovery' else [2,3]):
                for l in [512,1024,2048]:
                    g=[r for r in rows if (r['split'],r['scope'],r['frame'],r['length'])==(split,scope,f,l)]
                    cells.append({'split':split,'scope':scope,'frame':f,'length':l,'n':len(g),'correct':sum(r['correct'] for r in g),'accuracy':float(np.mean([r['correct'] for r in g]))})
    scope_stats=[{'scope':s,'n':len(g),'accuracy':float(np.mean([r['correct'] for r in g]))} for s in ['文档','现实'] for g in [[r for r in held if r['scope']==s]]]
    gate=all(s['accuracy']>=.8 for s in scope_stats)
    h1['supported']=bool(gate and h1['threshold_met'])
    inter=read(P.OUT/'head-interventions.json');tests=[]
    for cond in ['selected_heads','matched_random_heads']:
        g=[r for r in inter['rows'] if r['condition']==cond]
        d=[int(by[r['family_id'],2,1024,'文档']['correct'])-int(r['correct']) for r in g]
        tests.append({'condition':cond,'n':len(g),'accuracy':float(np.mean([r['correct'] for r in g])),'drop':interval(d)})
    h2=bool(tests[0]['drop']['mean']>=.2 and tests[0]['drop']['ci95'][0]>0 and tests[0]['drop']['mean']>tests[1]['drop']['mean'])
    priors=read(P.OUT/'source-free-priors.json');prior_by={r['family_id']:r for r in priors['rows']}
    errors=[{**{k:r[k] for k in ['id','entity','family_id','scope','split','frame','length','expected','palette_choice','prior','document','trace_id']},
        'reason':'scope answer does not match registered reference','source_free_prior_correct':prior_by[r['family_id']]['correct']} for r in rows if not r['correct']]
    for r in inter['rows']:
        if not r['correct']:errors.append({**r,'reason':'head intervention answer failure','trace_id':by[r['family_id'],2,1024,'文档']['trace_id']})
    if not h1['supported']:errors.append({'reason':'H1 not supported or behavior gate failed','statistic':h1,'scope_accuracy':scope_stats})
    if not h2:errors.append({'reason':'H2 preregistered head effect not supported','interventions':tests})
    observation_ref=archive('observations',raw)
    prior_ref=archive('source-free-priors',priors)
    inter_ref=archive('head-interventions',inter)
    execution_ref=archive('execution',read(P.OUT/'execution.json'))
    trace_index=[];totalbytes=0;filecount=0;maxerr=0.
    for row in rows:
        d,folder=store.manifest(row['trace_id']);native=[]
        for f in d['files']:
            p=store.native_path(d['trace_id'],f['path']);totalbytes+=p.stat().st_size;filecount+=1
        maxerr=max(maxerr,max(v['attention_relative_l2'] for v in d['reconstruction'].values()))
        trace_index.append({k:d.get(k) for k in ['trace_id','sample_id','entity','scope','split','deep','n_tokens','readout_positions','generation_step']})
    genids=[t for g in raw['generations'] for t in g['trace_ids'][1:]]
    for identity in genids:
        d,_=store.manifest(identity)
        for f in d['files']:
            p=store.native_path(identity,f['path']);totalbytes+=p.stat().st_size;filecount+=1
        maxerr=max(maxerr,max(v['attention_relative_l2'] for v in d['reconstruction'].values()))
    native_ref=archive('native-index',{'schema':'research_native_trace_index.v1','kind':'real','run_id':P.RUN,'model_snapshot':snap,
        'traces':trace_index,'generation_trace_ids':genids,'files':filecount,'bytes':totalbytes,'all_file_sha256_rechecked':True,
        'max_attention_relative_l2':maxerr,'scope':'all37 hidden layers and all36 blocks full MLP G/V/A and HeadZ at three registered positions; deep8 also all-prefix hidden and postRoPE QKV; generated steps at current position',
        'local_store':'distributed_data/mechanism_traces/index.sqlite3; native tensors retained under tests/glm5/result/phase2793_arbitration/native', 'provenance':prov})
    history=[]
    for group in ['entity_atlas','weight_space','transcoder','assembly']:
        for path in sorted((P.ROOT/'research'/group).glob('*/report.json')):
            source=read(path);identity=source.get('model_snapshot') or source.get('model_rev') or source.get('model')
            # Legacy values are preserved by the library scanner; use exact source byte hash as well.
            lib.scan(path.parent)
            with lib.db() as db:found=db.execute('SELECT sha FROM file_cache WHERE path=?',(str(path.resolve()),)).fetchone()
            if not found:continue
            history.append({'source':str(path.relative_to(P.ROOT)),'artifact_sha256':found['sha'],'source_file_sha256':file_sha(path),
                'model_snapshot':identity,'family':group,'native_trace_available':False,
                'evidence_boundary':'historical report; native arrays not migrated by this index; parameters/coordinates cannot be inferred from a label'})
    history_ref=archive('history-index',{'schema':'research_evidence_index.v1','kind':'real','run_id':P.RUN,'entries':history,
        'identity_contract':['model_snapshot','precision','run_id','sample_id','position','generation_step','block','boundary','method','evidence','counterexamples'],
        'note':'historical reports remain in their own model/precision/method buckets; this is a provenance migration, not a claim that all old raw fields exist'})
    limits=['长文本为受控人工篇章框架，非自然语料；增加长度同时增加背景内容。','主要指标限于11个单token颜色候选；8条实际自由生成只作描述。',
        '现实范围行为门未通过时，不将该格内部差异解释为已成功执行范围仲裁的机制。','Head DLA固定本样本最终RMS，是直接读出归因，不等于通过全部后续层的总效应。',
        '被选8头仅代表此发现算法的候选；组消融失败不否定冗余、其他头或其他读出位置。','只有一台物理节点；没有独立节点复算或跨模型复制。',
        '全单元指声明读出位置的完整阵列；除8条deep样本外，不宣称所有提示位置均有MLP或QKV原场。','没有得到图谱独立预测全词表或连续生成的结果；参数乘积对账不是语义机制闭合。']
    report={'schema':'research_entity_arbitration_report.v1','kind':'real','run_id':P.RUN,'title':'实体颜色属性与文档/现实范围：长文本三模块研究闭环','model_snapshot':snap,
        'n_inputs':len(rows),'n_entities':48,'heldout_entities':32,'scope_stats':scope_stats,'behavior_gate':gate,'H1':h1,'H2_supported':h2,'head_tests':tests,
        'selected_heads':inter['selection'],'cells':cells,'source_free_prior_accuracy':float(np.mean([r['correct'] for r in priors['rows']])),
        'counterexamples':errors,'counterexamples_count':len(errors),'generations':raw['generations'],'trace_preview':[r for r in trace_index if r['deep']],
        'native':{'files':filecount,'bytes':totalbytes,'generation_steps':len(genids),'max_attention_relative_l2':maxerr,'all_file_sha256_rechecked':True},
        'capability':{'status':'not_tested','reason':'H2未支持，不触发预注册的2200题新Head干预能力保持检验；旧W6权重编辑能力结果不复用到此操作。'},
        'references':{'design':frozen,'observations':observation_ref,'priors':prior_ref,'interventions':inter_ref,'native':native_ref,'execution':execution_ref,'history':history_ref},
        'limitations':limits,'preregistration':pr,'provenance':prov}
    if h2:raise RuntimeError('H2 supported: actual new capability monitor required before report publication')
    report_ref=archive('report',report)
    # Native locators are deliberately not named semantic atoms.
    example=next(r for r in trace_index if r['deep'] and r['split']=='heldout_entity_frame' and r['scope']=='文档');identity=example['trace_id'];d,_=store.manifest(identity);pos=d['readout_positions'][-1]
    cards=[];edges=[]
    embid=f'{snap}:embedding:{P.RUN}'
    cards.append({'id':embid,'label':'实体仲裁 · 实际输入词嵌入','layer':0,'boundary':'hidden[0]','index':{'space':'residual','coordinate':0},'run_id':P.RUN,
        'method':'actual input lookup, all native coordinates retained','scope':{'trace_id':identity,'position':pos,'semantic_interpretation':'not established'},'evidence':{'level':'observation','artifact_sha256':native_ref},'counterexamples':[]})
    for l,h in inter['selection']['heads']:
        z=store.block(identity,l);ri=d['readout_positions'].index(pos);a=unbits(z['activation'][ri]);u=int(np.abs(a).argmax())
        hid=f'{snap}:B{l}:H{h}:{P.RUN}';mid=f'{snap}:B{l}:U{u}:{P.RUN}'
        for fid,label,bound,index in [(hid,f'B{l} H{h} · 仲裁候选（组消融未支持）','self_attn.sdpa.output',{'space':'attention','head':h}),
            (mid,f'B{l} MLP{u} · 原场定位','mlp.down_proj.input',{'space':'mlp','unit':u})]:
            if any(x['id']==fid for x in cards):continue
            cards.append({'id':fid,'label':label,'layer':l,'boundary':bound,'index':index,'run_id':P.RUN,
                'method':'discovery head DLA selection; native MLP representative by example activation, not a semantic selection',
                'scope':{'trace_id':identity,'position':pos,'representative_unit_only':True,'group_intervention_supported':False},
                'evidence':{'level':'observation','artifact_sha256':report_ref},'counterexamples':[{'reason':'scope behavior gate and selected head group effect did not pass','report_sha256':report_ref}]})
        edges.append({'id':hid+':parameter-source','source':hid,'target':mid,'relation':'local_parameter_connection_not_causal','run_id':P.RUN,
            'evidence':{'level':'parameter_source','artifact_sha256':native_ref},'scope':{'trace_id':identity,'position':pos,'formula':'W_gate[j] diag(gamma_M) W_O[h] / actual_RMS; signed sample contribution queried through connections endpoint'}})
        edges.append({'id':hid+':scope-test','source':'language:entity-property-scope','target':hid,'relation':'candidate_group_intervention_not_supported','run_id':P.RUN,'evidence':{'level':'negative','artifact_sha256':report_ref}})
    graph_ref=archive('graph',{'schema':'research_graph.v1','kind':'real','run_id':P.RUN,'model_snapshot':snap,'feature_cards':cards,'graph_edges':edges,'semantic_mechanisms_claimed':0,'provenance':prov})
    dataset={'schema':'dataset_version.v1','kind':'real','id':P.RUN+':corpus','dataset_id':'entity-colour-scope-long','version':'v1','source':'ENT-XF_nouns.v2.tsv plus authored paired scope documents',
        'license':'project-authored templates; lexicon provenance in repository','label_status':'reference lexical colours and explicit fictional facts; model prior errors retained','role':'controlled_mechanism',
        'content_sha256':digest(c),'source_groups':[e['id'] for e in c['entities']],'sample_count':576,'frozen_design_sha256':frozen}
    reg.register(dataset)
    for metric in ['paired_scope_behaviour','selected_head_group_effect']:
        reg.register({'schema':'research_claim.v1','kind':'real','id':P.RUN+':'+metric,'canonical_feature_id':P.RUN+':scope-route','model_snapshot':snap,
            'external_pattern_id':'language:entity-property-scope','outcome_metric':metric,'scope':'48 nouns;32 heldout entities and2 unseen frames;512/1024/2048 Chinese tokens;11 colour candidates',
            'status':'not_supported','dataset_versions':[dataset['id']],'artifact_sha256':report_ref,'run_id':P.RUN})
    registration={'report_sha256':report_ref,'graph_sha256':graph_ref,'history_sha256':history_ref,'trace_id':identity,
        'client_url':f'http://127.0.0.1:5173/workbench?lens=process&station=plan&run={P.RUN}',
        'space_url':f'http://127.0.0.1:5173/workbench?lens=spatial&view=live&trace={identity}'}
    P.save(P.OUT/'client-registration.json',registration,True)
    status('validation',state='not_supported',behavior_gate=gate,H1=h1,H2_supported=h2)
    status('native_audit',files=filecount,bytes=totalbytes,max_attention_relative_l2=maxerr)
    status('registry',history_reports=len(history),feature_locators=len(cards),semantic_mechanisms=0)
    status('capability',state='not_tested',reason=report['capability']['reason'])
    status('registration_validated',**registration)
    print(json.dumps({'H1':h1,'scope_stats':scope_stats,'head_tests':tests,'native':report['native'],'history_reports':len(history),'registration':registration},ensure_ascii=False))

if __name__=='__main__':main()
