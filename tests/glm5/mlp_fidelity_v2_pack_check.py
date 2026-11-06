"""Bounded-memory validation of the new offline pack against real registrations."""
import hashlib
import re
import time

import mlp_fidelity_v2_plan as P
from deploy.research_library import canonical
from deploy.research_pack_stream import iter_records


def refs(value):
    if isinstance(value,dict):
        for v in value.values():yield from refs(v)
    elif isinstance(value,list):
        for v in value:yield from refs(v)
    elif isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value):yield value


def main():
    start=time.monotonic()
    reg=P.read(P.OUT/'registration-delivery.json')
    phase=P.read(P.OUT/'phase-registration.json')
    inv=P.read(P.ROOT/'tests/glm5/result/mlp_atlas_inventory_v1/registration-v6.json')
    case_sha=reg['artifact_refs']['case_index']
    case_index=P.read(P.PUBLIC/f'case-index-{case_sha[:12]}.research.json')
    expected={reg['report_sha256'],reg['graph_sha256'],inv['report_sha256'],phase['phase_summary_sha256'],phase['acceptance_sha256'],*refs(reg['artifact_refs']),*case_index['chunks']}
    pack=P.ROOT/'distributed_data/datapack/extra/research-library.json.gz'
    found=set();paths=set();hashed=0;tail=None;checked_index=False
    for rec in iter_records(pack):
        if rec.get('complete'):tail=rec;continue
        path=rec.get('path')
        if not path:continue
        assert path not in paths,path
        paths.add(path)
        if path=='/api/traces/research-space':
            study=next(s for s in rec['data']['studies'] if s['run_id']==P.RUN)
            assert study['report_sha256']==reg['report_sha256']
            checked_index=True
        if path.startswith('/api/research-library/artifacts/'):
            sha=path.rsplit('/',1)[1]
            assert hashlib.sha256(canonical(rec['data'])).hexdigest()==sha,path
            hashed+=1
            if sha in expected:found.add(sha)
    assert tail and tail['n_entries']==len(paths) and checked_index
    assert found==expected,sorted(expected-found)
    P.save(P.OUT/'offline-pack-acceptance.json',{'schema':'mlp_fidelity_offline_acceptance.v1','run_id':P.RUN,'version':tail['version'],'entries':len(paths),'hashed_artifacts':hashed,'verified_current_refs':len(found),'verified_case_chunks':len(case_index['chunks']),'compressed_bytes':pack.stat().st_size,'seconds':time.monotonic()-start,'scope':'offline reports and evidence indexes; native arrays stay node-local'})
    print(f'{len(paths)} entries, {hashed} content hashes and {len(found)} current references verified',flush=True)


if __name__=='__main__':
    main()
