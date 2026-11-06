"""Resume metadata registration after tensor verification; no repeated inference."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import native_mlp_function_materials_v4 as P
from native_mlp_function_publish import register_binding
from shared.dist_protocol.mechanism_trace import TraceStore,digest
from shared.dist_protocol.research_registry import Registry
from deploy.research_library import Library

def main():
    lib=Library(P.ROOT/'distributed_data/research_library');reg=Registry(lib);store=TraceStore();report=P.read(P.PUBLIC/'report.research.json');graph=P.read(P.PUBLIC/'graph.research.json');prov=report['provenance'];snap=report['model_snapshot']
    def archive(name,doc,run=P.RUN):
        d={'schema':'research_native_evidence.v1','kind':'real','run_id':run,'model_snapshot':snap,'provenance':prov,**doc};p=P.PUBLIC/(name+'.research.json');P.save(p,d)
        return lib.ingest(d,str(p.relative_to(P.ROOT)),kind='local_report')['sha256']
    rr=lib.ingest(report,'research/native_mlp_function/v4/report.research.json')['sha256'];gr=lib.ingest(graph,'research/native_mlp_function/v4/graph.research.json')['sha256']
    datasets=[]
    for name,path,role in [('controlled-long-text',P.OUT/'corpus.json','controlled_mechanism'),('universal-polarity',P.OUT/'polarity-corpus.json','counterexample'),('place-polarity',P.OUT/'place-polarity/polarity-corpus.json','counterexample')]:
        c=P.read(path);items=c['items'];groups=sorted({str(r.get('background_doc_id',r.get('background_doc_index'))) for r in items});did=P.RUN+':'+name
        reg.register({'schema':'dataset_version.v1','kind':'real','id':did,'dataset_id':'native-mlp-'+name,'version':'4','content_sha256':digest(c),'source':'frozen generated task over pinned natural backgrounds; exact token IDs and source lineage archived','license':'generated material plus original natural source licenses','label_status':'external operational labels; no internal ontology assumed','role':role,'source_groups':groups,'sample_count':len(items),'provenance':prov});datasets.append(did)
    for name,status,scope in [('conditioned_answer_response','supported_in_scope','six selected coordinates under two fresh fact×question cohorts; readout roles separately'),('fixed_property_neuron','not_supported','polarity/name-balanced counterexample narrows fixed-property interpretation'),('joint_query_switch','not_supported','six/eight units finalassistantquery only; no correctness changes')]:
        reg.register({'schema':'research_claim.v1','kind':'real','id':P.RUN+':'+name,'canonical_feature_id':P.RUN+':'+name,'model_snapshot':snap,'external_pattern_id':'language:conditioned_yes_no_answer','outcome_metric':name,'scope':scope,'status':status,'dataset_versions':datasets,'artifact_sha256':rr,'run_id':P.RUN,'provenance':prov})
    migrated=[]
    for rel in ['research/phase2793/unified-evidence.research.json','research/phase2794/unified-evidence.research.json','research/native_mlp_long/v1/unified-evidence.research.json']:
        p=P.ROOT/rel
        if not p.exists():continue
        d=P.read(p);examples=d.get('examples',[])[:6]
        if not examples:continue
        register_binding(archive,store,d['run_id'],d['report_sha256'],d['graph_sha256'],examples,d,'历史原生研究 · '+d['run_id']);migrated.append(d['run_id'])
    examples=report['examples'];out={'report_sha256':rr,'graph_sha256':gr,'case_index_sha256':report['case_index_sha256'],'feature_cards':len(graph['feature_cards']),'edges':len(graph['graph_edges']),'native':report['native'],
        'space_url':f'http://127.0.0.1:5173/workbench?lens=spatial&view=research&run={P.RUN}','trace_url':'http://127.0.0.1:5173/workbench'+examples[0]['url'],'legacy_native_bindings':migrated,
        'registration_note':'The initial metadata publisher used an unsupported dataset role. This resume uses controlled_mechanism, retains real observations and prior CAS versions, and does not rerun any model or alter a scientific verdict.'}
    archive('registration-recovery',{'datasets':datasets,'migrated':migrated,'note':out['registration_note']});P.save(P.OUT/'client-registration.json',out)
    print(json.dumps({k:v for k,v in out.items() if k!='native'},ensure_ascii=False))
if __name__=='__main__':main()
