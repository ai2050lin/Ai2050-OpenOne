"""Expose actual inventory and calibrated/native-bank progress; never promote evidence."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import mlp_atlas_code_materials_v2 as P
from shared.dist_protocol.mechanism_trace import digest,file_sha
from deploy.research_library import Library

def main():
    lib=Library(P.ROOT/'distributed_data/research_library');inv=P.read(P.ROOT/'tests/glm5/result/mlp_atlas_inventory_v1/registration-v2.json');snap=inv['model_snapshot'];prov=inv['provenance']
    def archive(folder,name,doc,run):
        body={'schema':'research_native_evidence.v1','kind':'real','run_id':run,'model_snapshot':snap,'provenance':prov,**doc};path=folder/(name+'-'+digest(body)[:12]+'.research.json');P.save(path,body);return lib.ingest(body,str(path.relative_to(P.ROOT)),kind='local_report')['sha256']
    run=inv['run_id'];folder=P.ROOT/'research/mlp_atlas/inventory_v1';gr=archive(folder,'graph',{'schema':'research_graph.v1','feature_cards':[],'graph_edges':[],'note':'Native unit records are array entries, not350208assertedsemanticfeatures.'},run)
    archive(folder,'binding',{'schema':'research_space_binding.v1','report_sha256':inv['report_sha256'],'graph_sha256':gr,'traces':[],'context':{'study_type':'mlp_reverse_inventory','layer_refs':inv['layer_refs'],'all_units':350208,'default_statement':'full native coverage, functional closure not established'}},run)
    for version in [1,2]:
        out=P.ROOT/f'tests/glm5/result/mlp_atlas_code_v{version}';pub=P.ROOT/f'research/mlp_atlas/code_v{version}';run=f'MLP-ATLAS-CODE-20261104-v{version}';pilot=P.read(out/'pilot-result.json');pr=P.read(pub/'prereg.research.json');refs={}
        refs['pilot']=archive(pub,'pilot-raw',pilot,run);refs['prereg']=archive(pub,'prereg-proof',pr,run)
        bank=out/'bank/manifest.json';state='collecting' if bank.exists() else 'calibration_failed';bankstate=P.read(bank) if bank.exists() else None
        if version==2:refs['analysis_plan']=archive(pub,'analysis-plan-proof',P.read(pub/'analysis-prereg.research.json'),run)
        conclusion='语义中性答码48格发现校准全过门，正式全阵列采集中；留出判据尚未执行。' if version==2 else '反转Yes/No与是/否含义未过行为门，正式采集未启动；这些失败不得用来否定内部相关机制。'
        report={'schema':'mlp_atlas_code_report.v1','state':state,'bank_available':bankstate is not None,'conclusion':conclusion,'preregistration':pr,'analysis_plan':P.read(pub/'analysis-prereg.research.json') if version==2 else None,
            'pilot':{'passed':pilot['passed'],'n_inputs':len(pilot['rows']),'cells':pilot['cells'],'seconds':pilot['seconds']},'artifact_refs':refs,'bank_status_at_capture':{k:bankstate[k] for k in ['completed','n_items','state']} if bankstate else None,
            'limitations':['This is one localQwen3-4B snapshot.','Pilot usesdiscoveryonly; no failedheldoutcell is silentlydropped.','Native capture/coverage isnotfunctionaldecoding or individualcausalclosure.','Rawbank stays onthecollectingnode; unrecordedattention rows remainabsent.']}
        rr=archive(pub,'report-progress',report,run);gr=archive(pub,'graph-progress',{'schema':'research_graph.v1','feature_cards':[],'graph_edges':[],'mechanisms_closed':0},run)
        archive(pub,'binding-progress',{'schema':'research_space_binding.v1','report_sha256':rr,'graph_sha256':gr,'traces':[],'context':{'study_type':'mlp_atlas_code','native_bank':bankstate is not None,'inventory_run_id':inv['run_id']}},run)
        archive(pub,'workpackage-progress',{'schema':'research_workpackage.v1','title':f'全MLP事实/问法/答码解耦v{version} · '+('正式采集中' if version==2 else '校准失败保留'),
            'stages':{'pilot':{'status':'done' if pilot['passed'] else 'behavior_failed','n':len(pilot['rows']),'passed':pilot['passed']},'collection':{'status':'running' if bankstate else 'not_started','completed_at_capture':bankstate['completed'] if bankstate else 0,'total':9984},'registration_validated':{'status':'done','report_sha256':rr,'graph_sha256':gr}}},run)
        P.save(out/'progress-registration.json',{'report_sha256':rr,'graph_sha256':gr,'run_id':run,'provenance':prov},False)
    print(json.dumps({'inventory_report':inv['report_sha256'],'code_runs_visible':[1,2],'all_native_units':350208}))
if __name__=='__main__':main()
