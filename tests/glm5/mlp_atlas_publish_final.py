"""Immutable source/case/array registration for whole MLP research; no evidence promotion."""
import sys,json,os,sqlite3,argparse
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import mlp_atlas_code_materials_v2 as P
from shared.dist_protocol.mechanism_trace import file_sha,digest
from deploy.research_library import Library

class Archive:
    def __init__(self,folder,run,man):
        self.folder=folder;self.folder.mkdir(parents=True,exist_ok=True);self.run=run;self.snap=man['model_snapshot'];self.prov=man['provenance'];self.lib=Library(P.ROOT/'distributed_data/research_library')
    def put(self,name,doc):
        body={'schema':'research_native_evidence.v1','kind':'real','run_id':self.run,'model_snapshot':self.snap,'provenance':self.prov,**doc};path=self.folder/(name+'-'+digest(body)[:12]+'.research.json');P.save(path,body)
        return self.lib.ingest(body,str(path.relative_to(P.ROOT)),kind='local_report')['sha256']
    def native(self,path,sha=None):
        sha=sha or file_sha(path);dest=P.ROOT/'distributed_data/research_native_blobs'/sha[:2]/(sha+path.suffix);dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():os.link(path,dest)
        assert dest.stat().st_size==path.stat().st_size
        return {'sha256':sha,'bytes':dest.stat().st_size,'path':str(dest.relative_to(P.ROOT)),'access':'node-local immutable tensor; no raw tensor in JSON offline pack'}
    def bind(self,report,graph,context,title,stages):
        self.put('binding',{'schema':'research_space_binding.v1','report_sha256':report,'graph_sha256':graph,'traces':[],'context':context})
        self.put('workpackage',{'schema':'research_workpackage.v1','title':title,'stages':{**stages,'registration_validated':{'status':'done','report_sha256':report,'graph_sha256':graph}}})
    def sources(self,names):
        return self.put('sources',{'schema':'research_source_bundle.v1','files':[{'path':name,'sha256':file_sha(P.ROOT/name),'text':(P.ROOT/name).read_text(encoding='utf-8')} for name in names]})

def code():
    man=P.read(P.OUT/'bank/manifest.json');a=P.read(P.OUT/'analysis-result.json');assert man['state']=='complete';A=Archive(P.PUBLIC,P.RUN,man);refs={}
    for key,path in [('prereg','prereg.research.json'),('analysis_plan','analysis-prereg.research.json'),('power_clarification','power-clarification.research.json')]:refs[key]=A.put(key,P.read(P.PUBLIC/path))
    refs['pilot']=A.put('pilot',P.read(P.OUT/'pilot-result.json'));refs['execution']=A.put('execution',P.read(P.OUT/'bank-execution-freeze.json'));refs['analysis_execution']=A.put('analysis-execution',P.read(P.OUT/'analysis-execution-freeze.json'))
    refs['sources']=A.sources(['tests/glm5/'+n for n in ['mlp_atlas_code_materials.py','mlp_atlas_code_materials_v2.py','mlp_atlas_code_pilot_v2.py','mlp_atlas_code_collect.py','mlp_atlas_code_analyze.py']]+['shared/dist_protocol/native_mlp_atlas.py'])
    tensors=[A.native(P.OUT/'bank'/f['path'],f['sha256']) for f in man['files']];refs['bank_manifest']=A.put('bank-manifest',{**man,'native_content_addressed_files':tensors})
    db=sqlite3.connect(P.OUT/'bank/samples.sqlite3');cursor=db.execute('SELECT idx,body FROM samples ORDER BY idx');chunks=[]
    while True:
        rows=cursor.fetchmany(64)
        if not rows:break
        chunks.append(A.put('cases-'+str(rows[0][0]),{'schema':'mlp_atlas_cases.v1','first_index':rows[0][0],'cases':[json.loads(body) for idx,body in rows]}))
    db.close();refs['cases']=A.put('case-index',{'schema':'mlp_atlas_case_index.v1','n_items':9984,'chunk_refs':chunks,'all_tokens_and_actual_inputs_preserved':True})
    layer_refs={}
    for meta in a['layers']:
        b=meta['block'];path=P.OUT/f'analysis/B{b:02d}.npz';tensor=A.native(path,meta['sha256'])
        with np.load(path) as z:values={k:z[k].tolist() for k in ['sign','response_replicated','eligible_supported','shuffled_replicated','confirm_fvu','lockbox_fvu','confirm_successes','lockbox_successes']}
        layer_refs[str(b)]=A.put(f'analysis-B{b:02d}',{'schema':'mlp_atlas_layer_analysis.v1','block':b,'d_ff':9728,'native_index_order':True,'behavior_gate':a['behavior_gate'],'full_tensor':tensor,'tensor_sha256':meta['sha256'],**values})
    refs['all_layer_arrays']=layer_refs;refs['analysis']=A.put('analysis-result',a)
    conclusion=f"9,984条采集及350,208单元检验完成；两个读出响应复现{a['response_replicated_by_readout']}，但96个留出行为格有{sum(not c['passed'] for c in a['behavior_cells'])}格失败，功能解释资格为0。响应与功能明确分开。"
    report={'schema':'mlp_atlas_code_report.v1','state':'complete','bank_available':True,'conclusion':conclusion,'analysis':a,'preregistration':P.read(P.PUBLIC/'prereg.research.json'),'analysis_plan':P.read(P.PUBLIC/'analysis-prereg.research.json'),
        'pilot':{'passed':True,'n_inputs':384},'artifact_refs':refs,'limitations':[a['scope'],a['prediction_scope'],'This is registered code selection, not free generation.','Behavior gate failed: no atom/function promotion.','Scope intersection rule may miss conditional/moderate coding.','Raw large tensors remain on collecting node; CAS hashes and all evidence metadata persist.']}
    rr=A.put('report-final',report);cards=[]
    for b,u in [(24,8443),(28,4371),(32,5610),(32,3952),(29,8463),(34,446),(27,2658)]:
        with np.load(P.OUT/f'analysis/B{b:02d}.npz') as z:rep=z['response_replicated'][:,u].tolist()
        cards.append({'id':f'{A.snap}:B{b}:U{u}:{P.RUN}','layer':b,'boundary':'mlp.down_proj.input','index':{'space':'mlp','unit':u},'method':'frozen factorial decision ordering across neutral code and mapping; allnativeunit arrays retained','run_id':P.RUN,
            'label':f'B{b}/U{u} · 响应{rep}；行为门阻断功能升级','evidence':{'level':'negative','artifact_sha256':layer_refs[str(b)]},'counterexamples':[{'reason':'globalheldout behavior gate failed; not evidence of no function','cases_ref':refs['cases']}],'scope':{'response_by_readout':rep,'functional_eligibility':False,'sample_bank':P.RUN}})
    gr=A.put('graph-final',{'schema':'research_graph.v1','feature_cards':cards,'graph_edges':[],'note':'Seven historical unit anchors, not seven confirmed functions. Full350208records in layer arrays.'})
    A.bind(rr,gr,{'study_type':'mlp_atlas_code','native_bank':True,'layer_refs':layer_refs,'case_index':refs['cases']},'全MLP答码解耦 · 全量检验完成，功能门未过',{'collection':{'status':'done','inputs':9984,'native_units':350208},'analysis':{'status':'done','unique_response_units':a['unique_response_units']},'functional_gate':{'status':'behavior_failed','eligible_units':0}})
    P.save(P.OUT/'final-registration.json',{'report_sha256':rr,'graph_sha256':gr,'layer_refs':layer_refs,'case_index':refs['cases'],'run_id':P.RUN,'model_snapshot':A.snap,'provenance':A.prov},False);print(json.dumps({'code_report':rr,'graph':gr}))

def dictionary():
    import mlp_atlas_natural_collect as D
    man=D.read(D.OUT/'bank/manifest.json');result=D.read(D.OUT/'dictionary-result.json');A=Archive(D.PUBLIC,D.RUN,man);refs={}
    refs['prereg']=A.put('prereg-proof',D.read(D.PUBLIC/'prereg.research.json'));refs['execution']=A.put('execution',D.read(D.OUT/'training-execution-freeze.json'))
    refs['sources']=A.sources(['tests/glm5/mlp_atlas_natural_collect.py','tests/glm5/mlp_atlas_dictionary_train.py','shared/dist_protocol/transcoder.py'])
    refs['native_bank']=A.put('native-bank',{**man,'native_content_addressed_files':[A.native(D.OUT/'bank'/f['path'],f['sha256']) for f in man['files']]})
    layer_refs={}
    for row in result['layers']:
        b=row['block'];folder=D.OUT/'dictionaries';weights=A.native(folder/f'B{b:02d}-transcoder.npz',row['dictionary_sha256']);associations=A.native(folder/f'B{b:02d}-associations.npz',row['association_sha256'])
        with np.load(folder/f'B{b:02d}-associations.npz') as z:values={k:z[k].tolist() for k in ['discovery_partner','discovery_partner_correlation','confirm_partner_correlation','lockbox_partner_correlation']}
        layer_refs[str(b)]=A.put(f'layer-B{b:02d}',{'schema':'mlp_dictionary_layer.v1','block':b,'dictionary':weights,'associations':associations,'results':row,**values})
    refs['all_layer_arrays']=layer_refs
    report={**result,'preregistration':D.read(D.PUBLIC/'prereg.research.json'),'artifact_refs':refs,'conclusion':'36层无标注字典与350,208个原生单元关联评估完成；重建、相关与语义/因果破解分别记账。'}
    rr=A.put('report-final',report);gr=A.put('graph-final',{'schema':'research_graph.v1','feature_cards':[],'graph_edges':[],'native_unit_records':350208,'dictionary_feature_records':36864,'note':'Full native/dictionary arrays queryable; latentfeatures are not native neurons and are not asserted semantic nodes.'})
    A.bind(rr,gr,{'study_type':'mlp_dictionary_atlas','native_bank':True,'layer_refs':layer_refs,'dictionary_features':36864},'全36层自然MLP字典 · 重建、关联与原片',{'collection':{'status':'done','natural_documents':2048,'positions_per_layer':32768},'dictionary':{'status':'done','layers':36,'features':36864},'semantic_causal_closure':{'status':'not_complete'}})
    D.save(D.OUT/'final-registration.json',{'report_sha256':rr,'graph_sha256':gr,'layer_refs':layer_refs,'run_id':D.RUN,'model_snapshot':A.snap,'provenance':A.prov},False);print(json.dumps({'dictionary_report':rr}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('kind',choices=['code','dictionary']);args=parser.parse_args();code() if args.kind=='code' else dictionary()
