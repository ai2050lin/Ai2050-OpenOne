"""Bounded-entry validation of the actual offline package and current study proofs."""
import sys,json,hashlib,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import mlp_atlas_code_materials_v2 as P
from deploy.research_pack_stream import iter_records
from deploy.research_library import canonical

def main():
    inv=P.read(P.ROOT/'tests/glm5/result/mlp_atlas_inventory_v1/registration-v5.json');code=P.read(P.OUT/'final-registration.json');dic=P.read(P.ROOT/'tests/glm5/result/mlp_atlas_dictionary_v1/interpretation-registration-v2.json');audit=P.read(P.ROOT/'tests/glm5/result/mlp_atlas_parameter_audit_v1/final-registration.json')
    wanted={inv['report_sha256'],code['report_sha256'],dic['report_sha256'],audit['report_sha256'],inv['research_plan_sha256'],inv['phase_summary_sha256'],inv['acceptance_sha256'],*inv['layer_refs'].values(),*code['layer_refs'].values(),*dic['layer_refs'].values(),*audit['layer_refs'].values()};found=set();paths=set();n=0;hashed=0;start=time.time();tail=None
    path=P.ROOT/'distributed_data/datapack/extra/research-library.json.gz'
    for r in iter_records(path):
        if r.get('complete'):tail=r;continue
        if 'path' not in r:continue
        key=r['path'];assert key not in paths;paths.add(key);n+=1
        if key.startswith('/api/research-library/artifacts/'):
            sha=key.rsplit('/',1)[-1];assert hashlib.sha256(canonical(r['data'])).hexdigest()==sha;hashed+=1
            if sha in wanted:found.add(sha)
    assert tail and tail['n_entries']==n and wanted==found,(n,len(wanted-found))
    P.save(P.OUT/'offline-pack-acceptance.json',{'schema':'mlp_atlas_pack_acceptance.v1','kind':'real','run_id':P.RUN,'provenance':inv['provenance'],'model_snapshot':inv['model_snapshot'],'entries':n,'hashed_artifacts':hashed,'required_current_artifacts':len(wanted),'all_current_artifacts_found':True,'version':tail['version'],'compressed_bytes':path.stat().st_size,'seconds':time.time()-start,'scope':'offlinepackage integrity, notscientificpromotion'})
    print(json.dumps({'entries':n,'hashed_artifacts':hashed,'current_proofs':len(found),'seconds':time.time()-start}),flush=True)
if __name__=='__main__':main()
