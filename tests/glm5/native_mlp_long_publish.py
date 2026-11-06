"""Immutable corpus, cards, typed graph and honest client research closure."""
import sys,json,collections
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import native_mlp_long_materials as P
from native_mlp_long_run import status
from shared.dist_protocol.mechanism_trace import TraceStore,file_sha,digest
from shared.dist_protocol.research_registry import Registry
from deploy.research_library import Library

def main():
    from transformers import AutoTokenizer
    lib=Library(P.ROOT/'distributed_data/research_library');reg=Registry(lib);store=TraceStore()
    data=P.read(P.OUT/'documents.json');a=P.read(P.OUT/'analysis.json');col=P.read(P.OUT/'collection.json');dose=P.read(P.OUT/'dose-result.json');cap=P.read(P.OUT/'capability-result.json');prov=col['provenance'];snap=col['model_snapshot'];freeze=P.read(P.OUT/'execution-freeze.json')
    assert all(file_sha(P.ROOT/n)==h for n,h in freeze['dependencies'].items())
    def archive(name,doc):
        body={'schema':doc.get('schema','research_native_evidence.v1'),'kind':'real','run_id':P.RUN,'model_snapshot':snap,'provenance':prov,**doc}
        path=P.PUBLIC/(name+'.research.json');P.save(path,body);return lib.ingest(body,str(path.relative_to(P.ROOT)),kind='local_report')['sha256']
    import torch,transformers
    refs={};diagnostics=P.read(P.OUT/'diagnostics.json');refs['diagnostics']=archive('diagnostics',diagnostics)
    refs['execution']=archive('execution',{'torch':torch.__version__,'transformers':transformers.__version__,'cuda':torch.version.cuda,'attention':'sdpa','dtype':'bfloat16',
        'collection_seconds':col['seconds'],'collection_peak_cuda_bytes':col['peak_cuda_bytes'],'collection_batch':2,'dose_batch':1,'snapshot_before_after_equal':col['snapshot_unchanged']})
    tok=AutoTokenizer.from_pretrained(P.MODEL,local_files_only=True);raw=np.load(P.OUT/'input-tokens.npy',mmap_mode='r');positions=collections.defaultdict(set)
    for c in a['cards']:
        for p in c['poles'].values():
            for ex in p['examples']:positions[ex['doc_index']].add(ex['position'])
        for e in c['counterexamples']:
            for side in ['a','b']:positions[e['pair'][side]].add(e['pair']['position'])
    refs['documents']=[]
    for start in range(0,2048,32):
        docs=[]
        for d in data['docs'][start:start+32]:
            di=d['doc_index'];docs.append({k:v for k,v in d.items() if k!='tokens'}|{'token_ids':raw[di,:1024].tolist(),'text':tok.decode(raw[di,:1024]),
                'prefix_texts':{str(p):tok.decode(raw[di,:p+1]) for p in sorted(positions[di])}})
        refs['documents'].append(archive(f'documents-{start//32:02d}',{'schema':'research_natural_documents.v1','documents':docs}))
    stats=np.load(P.OUT/'population-statistics.npz');cardrefs=[];features=[];edges=[]
    refs['population']=[]
    for l in range(36):
        # Lossless aggregate NPZ remains authoritative; this compact viewer is derived.
        n=stats['count'][:,None,None];mean=stats['sums'][:,l]/n;rms=np.sqrt(stats['squares'][:,l]/n)
        refs['population'].append(archive(f'population-B{l:02d}',{'schema':'research_native_population_layer.v1','layer':l,'count':stats['count'].tolist(),
            'GVA_mean':mean.tolist(),'GVA_RMS':rms.tolist(),'GV_sign_partition_counts':stats['regimes'][:,l].tolist(),
            'note':'G>0/V>0 partitions, zeros belong to nonpositive; not a pure-negative quadrant. All extrema/moments losslessly retained in NPZ.'}))
    for c0 in a['cards']:
        c=dict(c0);l,u=c['layer'],c['unit'];population=[]
        c['reader_formula_audit']=next(r for r in diagnostics['reconstruction'] if r['layer']==l and r['unit']==u)
        selected_index=[r['unit'] for r in data['units'] if r['layer']==l].index(u)
        fields={k:np.load(P.OUT/f'L{l:02d}-{k}.npy',mmap_mode='r') for k in ['G','V','A']}
        from shared.dist_protocol.mechanism_trace import unbits
        for pole in c['poles'].values():
            for ex in pole['examples']:
                ex['GVA']={k:float(unbits(v[ex['doc_index'],ex['position'],selected_index])) for k,v in fields.items()}
        for f in range(3):
            population.append({'split':['discovery','confirm','lockbox'][f],'count':int(stats['count'][f]),
                'GVA_mean':(stats['sums'][f,l,:,u]/stats['count'][f]).tolist(),'GVA_RMS':np.sqrt(stats['squares'][f,l,:,u]/stats['count'][f]).tolist(),
                'GVA_min':stats['mins'][f,l,:,u].tolist(),'GVA_max':stats['maxs'][f,l,:,u].tolist(),'GV_sign_partition_counts':stats['regimes'][f,l,u].tolist()})
        c['population']=population;ref=archive(f'card-B{l:02d}-U{u:04d}',{'schema':'research_native_neuron_card.v1','card':c})
        cardrefs.append({**{k:c[k] for k in ['layer','unit','selection','replicated']},'sha256':ref,'confirm_delta':c['confirm']['delta'],'lockbox_delta':c['lockbox']['delta']})
        fid=f'{snap}:B{l}:U{u}:{P.RUN}'
        features.append({'id':fid,'label':f'自然前缀响应 B{l}/U{u}','layer':l,'boundary':'mlp.down_proj.input','index':{'space':'mlp','unit':u},'run_id':P.RUN,
            'method':'discovery text-only Ridge with token/position baseline, document-disjoint matched-token+position confirm/lockbox; natural G/V/A poles',
            'scope':{'semantic_function':'unlabelled','card_sha256':ref,'sampling':c['selection'],'individual_necessity':'not_tested'},
            'evidence':{'level':'heldout' if c['replicated'] else 'observation','artifact_sha256':ref},'counterexamples':c['counterexamples'][:2]})
        edges.append({'id':fid+':text','source':'language:prefix-token-category-context','target':fid,'relation':'heldout_context_response_prediction' if c['replicated'] else 'not_replicated_context_candidate',
            'run_id':P.RUN,'evidence':{'level':'heldout' if c['replicated'] else 'negative','artifact_sha256':ref}})
    traces=P.read(P.OUT/'trace-examples.json')['rows'];manifests=[];bytes_native=0;tracefiles=0;examples=[]
    from server.mechanism_trace_service import contributions
    refs['parameters']=[]
    t=traces[0]
    for l,u in a['selection']['units']:
        r=contributions(t['trace_id'],l,t['position'],0,u,0);assert r['verified_full_snapshot'];head=int(np.argmax(np.abs(r['head_gate_terms'])))
        ref=archive(f'parameter-B{l:02d}-U{u:04d}',r);refs['parameters'].append(ref);hid=f'{snap}:B{l}:Head{head}:{P.RUN}'
        features.append({'id':hid,'label':f'自然样本Gate来源 B{l}/H{head}','layer':l,'boundary':'self_attn.o_proj.input','index':{'space':'attention','head':head},'run_id':P.RUN,
            'method':'actual weights, fixed actual sample RMS; all32 source terms retained','scope':{'trace_id':t['trace_id'],'position':t['position'],'individual_necessity':'not_tested'},
            'evidence':{'level':'parameter_source','artifact_sha256':ref},'counterexamples':[]})
        edges.append({'id':hid+':to-mlp','source':hid,'target':f'{snap}:B{l}:U{u}:{P.RUN}','relation':'actual_parameter_gate_source_not_total_causality','run_id':P.RUN,'evidence':{'level':'parameter_source','artifact_sha256':ref}})
    for t in traces:
        d,_=store.manifest(t['trace_id']);manifests.append(d)
        for f in d['files']:store.native_path(t['trace_id'],f['path']);bytes_native+=f['bytes'];tracefiles+=1
        examples.append({**t,'url':f'?lens=spatial&view=live&trace={t["trace_id"]}&tblock={t["layer"]}&thead=0&tunit={t["unit"]}'})
    refs['manifests']=archive('manifests',{'schema':'research_native_manifests.v1','manifests':manifests})
    refs['capability']=archive('capability',{'base':P.read(P.OUT/'capability-base.json'),'edited':P.read(P.OUT/'capability-selected_units.json'),'result':cap,'contract':P.read(P.OUT/'capability-contract.json')})
    refs['doses']=archive('doses',{'result':dose,'contract':P.read(P.OUT/'dose-contract.json')})
    refs['numeric']=archive('numeric',{'result':P.read(P.OUT/'numeric-result.json'),'contract':P.read(P.OUT/'numeric-contract.json')})
    names=[str(p.relative_to(P.ROOT)) for p in sorted((P.ROOT/'tests/glm5').glob('native_mlp_long_*.py'))]
    refs['sources']=archive('sources',{'schema':'research_source_bundle.v1','files':[{'path':n,'sha256':file_sha(P.ROOT/n),'text':(P.ROOT/n).read_text(encoding='utf-8')} for n in names]})
    native_files=[]
    for p in sorted(P.OUT.iterdir()):
        if p.is_file() and p.suffix in ['.npy','.npz']:native_files.append({'path':str(p.relative_to(P.ROOT)),'sha256':file_sha(p),'bytes':p.stat().st_size})
    native={'selected_units':466,'all_unit_statistics':350208,'raw_files':native_files,'trace_files':tracefiles,'trace_bytes':bytes_native,'bytes':sum(p['bytes'] for p in native_files)+bytes_native,
        'all_file_hashes_verified':True,'complete_all_module_per_token':False,'snapshot_unchanged':col['snapshot_unchanged'],'max_attention_relative_l2':max(t['max_attention_relative_l2'] for t in traces)}
    refs['native']=archive('native-files',native)
    audit={'instrumentation_bitwise_equal_all':all(r['hook_logits_bitwise_equal'] for r in dose['instrumentation']),
        'zero_dose_clone_sham_all_equal':P.read(P.OUT/'numeric-result.json')['all_equal'],
        'batch2_to_batch1_max_selected_A_error':max(r['batch2_to_batch1_selected_A_max_absolute'] for r in dose['instrumentation']),
        'A_formula_exact_fraction':diagnostics['A_exact_fraction'],'A_formula_max_abs':diagnostics['A_max_abs'],
        'warning':'Primary raw collection uses batch2; dose uses batch1. Captured logits are independently compared bitwise with no hooks at the same batch1 shape.'}
    limits=['2M token仅覆盖中文/英文教育网页和百科，未覆盖代码、对话或全部语言；每篇只用首1024 token。',
        '发现语料曾被历史普查汇总统计使用；确认和锁箱来自旧普查未使用的维基行组，模型预训练曝光未知。',
        '近重复仅做精确前缀与MinHash候选5-token重合检查，不能排除跨语言或更高层语义重复。',
        '配对偏向发现语料中常见的共享token；控制当前token与位置，不代表控制了词汇领域、语种等所有因素。',
        '预测器仅用前缀字符类别，未得到独立语义功能命名；不是完整Goodfire复现，也未完成同预算SAE/transcoder比较。',
        '432随机单元与34历史定向单元分开；统计覆盖不是破解进度，不能外推为350208单元全部解释。',
        '只有466单元保存逐token G/V/A；全单元保存汇总矩，不能回查任意单元的任意token原值。',
        'G/V符号分区将零归非正侧；读写余弦和两极一致度仅描述，不是单义性判据。',
        '16单元为跨层联合当前query操作；负结果不等于单元无用，也不证明分布式容量定律；过去位置未被移除。',
        '能力保持使用已曝光的混合选择题池，单模型单节点；没有独立节点复算、自然生成保义或跨模型复现。']
    summary={k:a[k] for k in ['random_units','random_replicated','random_shuffle_double_pass']}
    report={'schema':'research_native_mlp_long_report.v1','analysis':summary,'cards':cardrefs,'dose':{k:v for k,v in dose.items() if k not in ['rows','instrumentation']},
        'capability':cap,'numeric_audit':audit,'selection':a['selection'],'examples':examples,'native':native,'preregistration':P.read(P.PUBLIC/'prereg.research.json'),
        'freeze':freeze,'artifact_refs':refs,'limitations':limits}
    rr=archive('report',report);gr=archive('graph',{'schema':'research_graph.v1','feature_cards':features,'graph_edges':edges,'individual_neuron_mechanisms_proven':0})
    did=P.RUN+':natural-documents';reg.register({'schema':'dataset_version.v1','kind':'real','id':did,'dataset_id':'native-mlp-natural-long','version':'1','content_sha256':freeze['documents_sha256'],
        'source':'HuggingFaceFW/fineweb-edu;opencsg/Fineweb-Edu-Chinese-V2.1;wikimedia/wikipedia pinned revisions','license':'Per-document source license retained; Wikipedia CC-BY-SA3.0/GFDL',
        'label_status':'natural text; character-category features operational; neuron semantic labels unvalidated','role':'natural_discovery','source_groups':[r['doc_id'] for r in data['docs']],
        'sample_count':2048,'provenance':prov})
    for name,passed,metric in [('context_prediction',summary['random_replicated']>0,'number of 432 units passing both heldout fixed thresholds'),('joint16_query_effect',dose['supported'],'natural next-token accuracy loss>=.10 and specificity>=.05 with positive95lower bounds')]:
        reg.register({'schema':'research_claim.v1','kind':'real','id':P.RUN+':'+name,'canonical_feature_id':P.RUN+':'+name,'model_snapshot':snap,'external_pattern_id':'language:prefix-token-category-context',
            'outcome_metric':metric,'scope':'1024token en/zh web/encyclopedia; matched current token andabsolute position; no semantic label or individual necessity',
            'status':'supported_in_scope' if passed else 'not_supported','dataset_versions':[did],'artifact_sha256':rr,'run_id':P.RUN,'provenance':prov})
    archive('unified-evidence',{'schema':'unified_research_context.v1','study_type':'native_mlp_long','report_sha256':rr,'graph_sha256':gr,
        'unit_cards':{f'{r["layer"]}:{r["unit"]}':r['sha256'] for r in cardrefs},'document_refs':refs['documents'],'report_summary':summary,'examples':examples})
    registration={'report_sha256':rr,'graph_sha256':gr,'client_url':f'http://127.0.0.1:5173/workbench?lens=process&station=plan&run={P.RUN}',
        'trace_url':'http://127.0.0.1:5173/workbench'+examples[0]['url'],'feature_cards':len(features),'graph_edges':len(edges),'native_bytes':native['bytes']}
    P.save(P.OUT/'client-registration.json',registration);status('prediction',**summary);status('registration_validated',**registration)
    print(json.dumps(registration,ensure_ascii=False))
if __name__=='__main__':main()
