"""Supplement immutable primary findings with integration/monitor audit; no promotion."""
from pathlib import Path
import json,sys
sys.path.insert(0,str(Path(__file__).parent))
import phase2793_materials as P
import phase2793_run as R
from shared.dist_protocol.mechanism_trace import TraceStore,file_sha,digest
from shared.dist_protocol.research_registry import Registry
from deploy.research_library import Library

def main():
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    lib=Library(P.ROOT/'distributed_data/research_library');reg=Registry(lib);store=TraceStore()
    report=read(P.PUBLIC/'report-unified-v2.research.json');registration=read(P.OUT/'client-registration-v2.json')
    freeze=read(P.OUT/'execution-freeze.json');assert all(file_sha(P.ROOT/p)==h for p,h in freeze['dependencies'].items())
    for name,expected in [('trace-tests.log','Ran 11 tests'),('protocol-tests-final.log','Ran 261 tests'),('client-tests-final.log','39 passed'),('frontend-build-final.log','built in')]:
        assert expected in (P.OUT/name).read_text(encoding='utf-8',errors='replace'),name
    sources=['shared/dist_protocol/mechanism_trace.py','server/mechanism_trace_service.py','server/live_analysis_service.py','server/live_only_app.py',
        'deploy/center_app.py','frontend/src/components/v8/MechanismScene.jsx','frontend/src/components/v8/MechanismTraceExplorer.jsx',
        'frontend/src/components/v8/EntityArbitrationReport.jsx','frontend/src/components/v8/ResearchExecution.jsx',
        'tests/glm5/phase2793_publish.py','tests/glm5/phase2793_capability.py','tests/glm5/phase2793_verify_system.py',
        'tests/glm5/test_phase2793_trace.py','tests/glm5/phase2793_explorer.test.jsx',str(Path(__file__).relative_to(P.ROOT))]
    bundle={'schema':'research_source_bundle.v1','kind':'real','run_id':P.RUN,'model_snapshot':report['model_snapshot'],
        'scope':'post-run integration code; collection dependencies still match prospective execution freeze',
        'files':[{'path':n,'sha256':file_sha(P.ROOT/n),'text':(P.ROOT/n).read_bytes().decode('utf-8')} for n in sources],
        'execution_freeze':freeze,'provenance':report['provenance']}
    def archive(name,doc):
        p=P.PUBLIC/(name+'.research.json');P.save(p,doc,True);return lib.ingest(doc,str(p.relative_to(P.ROOT)),kind='local_report')['sha256']
    source_ref=archive('integration-sources-'+digest(bundle)[:12],bundle)
    native=read(P.PUBLIC/'native-index.research.json');extra_files=extra_bytes=0
    for identity in native['generation_trace_ids']:
        d,_=store.manifest(identity)
        for f in d['files']:
            p=store.native_path(identity,f['path']);assert file_sha(p)==f['sha256'];extra_files+=1;extra_bytes+=f['bytes']
    assert extra_files+report['native_audit']['native_files_verified']==native['files']
    assert extra_bytes+report['native_audit']['native_bytes']==native['bytes']
    graph=lib.artifact(registration['graph_sha256']);alias_refs=[]
    for card in graph['feature_cards']:
        mlp=card['index']['space']=='mlp'
        canonical=P.RUN+':scope-response-profile' if mlp else f"native:{report['model_snapshot']}:B{card['layer']}:Head{card['index']['head']}"
        alias_refs.append(reg.register({'schema':'research_feature_alias.v1','kind':'real','id':'alias:'+digest({'graph':registration['graph_sha256'],'card':card['id']}),
            'canonical_feature_id':canonical,'model_snapshot':report['model_snapshot'],'legacy_feature_id':card['id'],
            'relation':'member_locator' if mlp else 'same_extracted_feature','artifact_sha256':registration['graph_sha256'],
            'note':'Native locator identity or member of one descriptive scope profile; never semantic equivalence or a new independent mechanism test.'}))
    acceptance={'schema':'research_integration_acceptance.v1','kind':'real','run_id':P.RUN,'model_snapshot':report['model_snapshot'],
        'report_sha256':registration['report_sha256'],'graph_sha256':registration['graph_sha256'],'sources_sha256':source_ref,
        'system':read(P.OUT/'system-acceptance.json'),'tests':{'protocol':261,'native_trace':11,'client':39,'production_build':'passed'},
        'generation_native_files_rehashed':extra_files,'all_native_files':native['files'],'all_native_bytes':native['bytes'],
        'cached_position_reader_fix':'absolute current token reads native hidden_readouts; uncaptured historical query rejected',
        'canonical_aliases_registered':len(alias_refs),'unchanged_primary_status':'H1 and H2 not supported',
        'scope':'one physical local node; historical index21 reports,9 core wrappers; no multi-node replication or full graph-only prediction claim',
        'provenance':report['provenance']}
    ref=archive('integration-acceptance',acceptance)
    R.status('capability',state=report['capability']['status'],n=2200,net_loss=report['capability']['net_loss'],
        loss_interval=report['capability']['loss_interval'],scope='additional exploratory monitor, after arbitration outcomes; primary status unchanged')
    R.status('interface',schema='mechanism_trace.v1',long_text_limit=2048,cached_generation_reader_fixed=True,artifact_sha256=ref)
    R.status('registry',datasets_added=1,primary_tests=2,latest_locators=44,independent_profiles=1,independent_joint_interventions=1,
        semantic_mechanisms_claimed=0,duplicate_dataset_or_claim_added=False)
    P.save(P.OUT/'integration-acceptance.json',acceptance,True)
    print(json.dumps({'acceptance_sha256':ref,'native_files':native['files'],'native_GiB':native['bytes']/2**30,'aliases':len(alias_refs)},ensure_ascii=False))
if __name__=='__main__':main()
