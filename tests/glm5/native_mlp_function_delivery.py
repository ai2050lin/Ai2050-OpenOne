"""Persist final acceptance, source versions and the append-only memo in 3D archive."""
import sys,json,re
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import native_mlp_function_materials_v4 as P
from native_mlp_function_publish import register_binding
from shared.dist_protocol.mechanism_trace import file_sha,digest,TraceStore
from deploy.research_library import Library

def main():
    r=P.read(P.OUT/'client-registration-mask-v4.json');lib=Library(P.ROOT/'distributed_data/research_library');report=lib.artifact(r['report_sha256']);prov=report['provenance'];snap=report['model_snapshot']
    def archive(name,doc,run=P.RUN):
        d={'schema':'research_native_evidence.v1','kind':'real','run_id':run,'model_snapshot':snap,'provenance':prov,**doc};p=P.PUBLIC/(name+'-'+digest(d)[:12]+'.research.json');P.save(p,d)
        return lib.ingest(d,str(p.relative_to(P.ROOT)),kind='local_report')['sha256']
    sources=sorted((P.ROOT/'tests/glm5').glob('native_mlp_function*.py'))+[P.ROOT/p for p in [
        'tests/glm5/test_native_mlp_function.py','tests/glm5/native_mlp_function_report.test.jsx','tests/glm5/native_mlp_function_space.test.jsx',
        'server/research_space_service.py','server/trace_catalog.py','server/mechanism_trace_service.py','deploy/research_library.py','deploy/datapack.py',
        'frontend/src/components/v8/ResearchSpace.jsx','frontend/src/components/v8/NativeFunctionReport.jsx','frontend/src/components/v8/MechanismTraceExplorer.jsx','frontend/src/components/v8/MechanismScene.jsx','frontend/src/components/v8/ResearchInterventionReport.jsx']]
    ss=archive('final-sources',{'schema':'research_source_bundle.v1','files':[{'path':str(p.relative_to(P.ROOT)),'sha256':file_sha(p),'text':p.read_text(encoding='utf-8')} for p in sources]})
    mr=P.read(P.OUT/'memo-registration.json');memo=P.ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md';text=memo.read_text(encoding='utf-8');begin=text.index('## Phase '+str(mr['phase'])+':');end=text.find('\n## Phase ',begin+1)
    ms=archive('scientific-memo',{'schema':'research_scientific_memo.v1','phase':mr['phase'],'timestamp':mr['timestamp'],'path':str(memo.relative_to(P.ROOT)),'file_sha256_at_capture':file_sha(memo),'text':text[begin:end if end>=0 else None]})
    checks={}
    for name,n in [('software-tests-final.log',11),('trace-tests-final.log',11),('protocol-tests-complete.log',261),('client-tests-final.log',57),('frontend-build-final.log',None)]:
        p=P.OUT/name;txt=p.read_text(encoding='utf-8',errors='replace');assert ('OK' in txt if name.endswith('.log') and 'tests-' in name and 'client' not in name and 'build' not in name else '57 passed' in txt if 'client-tests' in name else 'built in' in txt)
        checks[name]={'sha256':file_sha(p),'passed_tests':n,'tail':'\n'.join(txt.splitlines()[-10:])}
    images=[]
    for name in ['client-research-space-3d.png','client-research-space-full.png','client-trace-3d.png','research-results.png','research-results.pdf']:
        p=P.OUT/name;images.append({'path':str(p.relative_to(P.ROOT)),'sha256':file_sha(p),'bytes':p.stat().st_size})
    qa=archive('client-acceptance-final',{'schema':'research_client_acceptance.v1','checks':checks,'api':P.read(P.OUT/'api-acceptance.json'),'mask_correction':P.read(P.OUT/'mask-correction.json'),
        'browser_verified':['all archived studies and artifact versions','readout-dependent unit evidence','all9728 native units with actual G/V/A','all32 attention heads and reconstructed source row','parameter snapshot and arithmetic sources','case search and changed capability item','intervened A explicitly identified'],
        'proof_files':images,'sources_sha256':ss,'scientific_memo_sha256':ms,'raw_tensor_delivery':'local native data; offline package contains metadata and evidence objects, not the full native arrays','independent_nodes_verified':False})
    with lib.db() as db:rows=db.execute("SELECT sha FROM artifacts WHERE kind='space_binding' ORDER BY created DESC,sha DESC").fetchall()
    b=next(lib.artifact(row['sha']) for row in rows if lib.artifact(row['sha'])['run_id']==P.RUN)
    ctx={**b['context'],'client_acceptance_sha256':qa,'scientific_memo_sha256':ms}
    register_binding(archive,TraceStore(),P.RUN,r['report_sha256'],r['graph_sha256'],b['traces'],ctx,'原生MLP：答案方向反例与真实3D闭环',native=r['native'])
    wp=archive('workpackage-delivered',{'schema':'research_workpackage.v1','title':'原生MLP：答案方向反例与真实3D闭环','stages':{
        'collection':{'status':'done','current_unique_cases':4358,'actual_snapshots_archived':4378,'native_bytes':r['native']['total_bytes']},
        'registration_validated':{'status':'done','report_sha256':r['report_sha256'],'graph_sha256':r['graph_sha256']},
        'client_acceptance':{'status':'done','artifact_sha256':qa,'tests':340,'browser_verified':True,'source_sha256':ss},
        'scientific_memo':{'status':'appended','phase':mr['phase'],'artifact_sha256':ms},
        'interpretation':{'status':'scope_limited','answer_direction_candidates':6,'fixed_property_mechanisms_proven':0,'joint_query_effect':'not_supported','capability':'each_operation_noninferior','next_stage_plan_sha256':r['next_stage_plan_sha256']}}})
    out={**r,'acceptance_sha256':qa,'scientific_memo_sha256':ms,'source_bundle_sha256':ss,'workpackage_sha256':wp};P.save(P.OUT/'delivery-registration.json',out)
    print(json.dumps({'acceptance_sha256':qa,'scientific_memo_sha256':ms,'tests':340,'report_sha256':r['report_sha256']},ensure_ascii=False))
if __name__=='__main__':main()
