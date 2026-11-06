"""Check the final binary package after the acceptance/memo revision was added."""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import native_mlp_function_materials_v4 as P
from deploy.research_pack_stream import iter_records
from deploy.research_library import canonical
from shared.dist_protocol.mechanism_trace import file_sha

def main():
    r=P.read(P.OUT/'delivery-registration-stream.json');meta=P.read(P.ROOT/'distributed_data/datapack/extras.json')['research-library'];path=P.ROOT/'distributed_data/datapack/extra/research-library.json.gz'
    assert path.stat().st_size==meta['bytes'];expected={r[k] for k in ['report_sha256','graph_sha256','case_index_sha256','acceptance_sha256','scientific_memo_sha256','stream_source_bundle_sha256']};found=set();header=None;footer=None;count=0;space=False
    for row in iter_records(path):
        if header is None:header=row;assert row['version']==meta['version'];continue
        if row.get('complete'):footer=row;continue
        count+=1;k=row['path'];v=row['data']
        if k.startswith('/api/research-library/artifacts/'):
            sha=k.rsplit('/',1)[-1]
            if sha in expected:assert hashlib.sha256(canonical(v)).hexdigest()==sha;found.add(sha)
        if k=='/api/traces/research-space':
            s=next(s for s in v['studies'] if s['run_id']==P.RUN);assert s['report_sha256']==r['report_sha256'];space=True
    assert footer and count==footer['n_entries']==meta['n_entries'] and found==expected and space
    out={'kind':'real','run_id':P.RUN,'package_path':str(path),'package_sha256':file_sha(path),'version':meta['version'],'bytes':meta['bytes'],'entries':count,'all_latest_proofs_present':True,'research_space_current':True,'native_tensors_included':False,'browser_full_indexeddb_ingestion_tested':False}
    P.save(P.OUT/'offline-pack-final-inventory.json',out);print(json.dumps(out))
if __name__=='__main__':main()
