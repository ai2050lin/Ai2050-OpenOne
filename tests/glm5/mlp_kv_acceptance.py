"""Validate the live KV artifacts and their full offline archive by content hash."""
import argparse
import hashlib
import json
from urllib.request import urlopen

import numpy as np

import mlp_kv_common as C
from deploy.research_library import canonical
from deploy.research_pack_stream import iter_records


def live():
    reg=C.read(C.OUT/'registration-final.json');checks=[]
    def get(path):
        with urlopen('http://127.0.0.1:5001/api'+path,timeout=45) as r:answer=json.load(r)
        checks.append(path);return answer
    def art(sha):return get('/research-library/artifacts/'+sha)
    study=next(r for r in get('/traces/research-space')['studies'] if r['run_id']==C.RUN)
    assert study['report_sha256']==reg['report_sha256'] and study['model_snapshot']==reg['model_snapshot']
    doc=art(reg['report_sha256']);graph=art(reg['graph_sha256'])
    assert doc['state']=='complete' and doc['mediation']['checks']['replay_logits_bitwise_equal']==256
    assert len(graph['feature_cards'])==reg['cards'] and len(graph['graph_edges'])==reg['edges']
    for name,info in reg['native'].items():
        folder=C.OUT/'targeted' if name=='targeted' else C.OUT
        p=folder/info['provenance']['path'];assert C.file_sha(p)==info['provenance']['sha256']
        with np.load(p) as raw:
            for index in [0,info['n']-1]:
                size=info['chunk_size'];entry=art(info['chunks'][index//size])['cases'][index%size]
                assert entry['index']==index
                for key,array in entry['fields'].items():np.testing.assert_array_equal(np.asarray(array,dtype=np.uint16),raw[key][index])
                size2=doc['cases'][name]['chunk_size']
                case=art(doc['cases'][name]['chunks'][index//size2])['cases'][index%size2]
                assert case['id']==entry['id']
    for key,info in reg['cases'].items():
        if not info['n']:continue
        page=art(info['chunks'][-1]);assert page['collection']==key and page['start']+len(page['cases'])==info['n']
    assert all(r['n']==2200 for r in doc['capability']['summary'].values())
    C.save(C.OUT/'live-acceptance.json',{**C.identity(),'status':'passed','checks':checks,'report_sha256':reg['report_sha256']},False)
    print(len(checks),'live real-artifact checks passed',flush=True)


def offline():
    reg=C.read(C.OUT/'registration-final.json');phase=C.read(C.OUT/'phase-registration.json')
    expected={reg['report_sha256'],reg['graph_sha256'],*reg['refs'].values(),phase['summary_sha256'],phase['acceptance_sha256']}
    for k in ['cases','native']:
        for entry in reg[k].values():expected.update(entry['chunks'])
    pack=C.ROOT/'distributed_data/datapack/extra/research-library.json.gz';tail=None;paths=set();found=set();count=0;current=False
    for record in iter_records(pack):
        if record.get('complete'):tail=record;continue
        path=record.get('path')
        if path is None:continue
        assert path not in paths;paths.add(path)
        if path=='/api/traces/research-space':current=any(r['run_id']==C.RUN and r['report_sha256']==reg['report_sha256'] for r in record['data']['studies'])
        if path.startswith('/api/research-library/artifacts/'):
            sha=path.rsplit('/',1)[1];assert hashlib.sha256(canonical(record['data'])).hexdigest()==sha,path
            count+=1
            if sha in expected:found.add(sha)
    assert current and tail and tail['n_entries']==len(paths) and expected==found,(current,len(expected-found))
    C.save(C.OUT/'offline-acceptance.json',{**C.identity(),'status':'passed','entries':len(paths),'artifact_hashes':count,
          'current_refs':len(found),'version':tail['version'],'bytes':pack.stat().st_size},False)
    print(len(paths),'entries',count,'artifact hashes',len(found),'current refs passed',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('part',choices=['live','offline']);{'live':live,'offline':offline}[p.parse_args().part]()
