"""Verify live research data and the offline pack without allocating the whole archive."""
import argparse
import hashlib
import json
from urllib.error import HTTPError
from urllib.request import urlopen

import numpy as np

import mlp_functional_common as C
from deploy.research_library import canonical
from deploy.research_pack_stream import iter_records
from shared.dist_protocol.mechanism_trace import unbits


def live(final):
    checks=[]
    def get(path,status=200):
        try:
            with urlopen('http://127.0.0.1:5001/api'+path,timeout=90) as response:code,value=response.status,json.load(response)
        except HTTPError as exc:code,value=exc.code,json.load(exc)
        assert code==status,(path,code,status)
        checks.append({'path':path,'status':code})
        return value
    reg=C.read(C.OUT/f'registration-{"final" if final else "observed"}.json')
    report=get('/research-library/artifacts/'+reg['report_sha256'])
    study=next(r for r in get('/traces/research-space')['studies'] if r['run_id']==C.RUN)
    assert study['report_sha256']==reg['report_sha256']
    assert report['state']==('complete' if final else 'running') and report['native_count']==1792
    assert report['dictionary']['both_passed'] is False and report['behavior']['passed'] is False
    graph=get('/research-library/artifacts/'+reg['graph_sha256'])
    assert len(graph['feature_cards'])==reg['cards'] and len(graph['graph_edges'])==reg['edges']
    for key,info in reg['case_index'].items():
        if info['chunks']:
            last=get('/research-library/artifacts/'+info['chunks'][-1])
            assert last['collection']==key and last['start']+len(last['cases'])==info['n']
    base='/traces/mlp-functional/v1/samples/'
    for index,block in [(0,0),(1791,34),(1791,35)]:
        row=get(f'{base}{index}?block={block}')
        assert row['kind']=='real' and row['block']==block and row['model_snapshot']==reg['model_snapshot']
        assert len(row['embedding'])==len(row['incoming_b35'])==2560 and np.shape(row['head_z'])==(32,128)
        assert row['queries'][0]['q']==0 and row['queries'][1]['q']==1
        assert row['queries'][0]['b']==row['queries'][1]['b']==index//4
        head=np.load(C.OUT/'trace/head.npy',mmap_mode='r')[index,block]
        np.testing.assert_array_equal(np.asarray(row['head_z'],dtype=np.float32),unbits(head))
        assert (row['mlp_output'] is None) == (block==35)
    get(base+'1792?block=35',422)
    get(base+'0?block=36',422)
    if final:
        rev=report['revision'];assert rev['native']['fields']['incoming'][0]==1280
        r=get(base+'1279?block=34&cohort=revision');assert np.shape(r['head_z'])==(32,128) and len(r['mlp_output'])==2560
        assert r['queries'][0]['split']=='lockbox' and r['queries'][0]['b']==351 and r['queries'][1]['q']==1
        with np.load(C.OUT/'revision/observed-native.npz') as z:
            np.testing.assert_array_equal(np.asarray(r['head_z'],dtype=np.float32),unbits(z['head34'][1279]))
        get(base+'1280?block=34&cohort=revision',422)
        get(base+'0?block=0&cohort=revision',404)
        assert report['capability']['summary']['functional_sae']['n']==2200
        if rev['passed']:assert report['generated']['summary']['industry']['n']==160
    path=C.OUT/f'live-{"final" if final else "observed"}-acceptance.json'
    C.save(path,{**C.identity(),'status':'passed','report_sha256':reg['report_sha256'],'checks':checks},False)
    print(len(checks),'live real checks passed',flush=True)


def offline():
    reg=C.read(C.OUT/'registration-final.json')
    required={reg['report_sha256'],reg['graph_sha256'],*reg['refs'].values()}
    for entry in reg['case_index'].values():required.update(entry['chunks'])
    phase=C.read(C.OUT/'phase-registration.json');required.update([phase['summary_sha256'],phase['acceptance_sha256']])
    pack=C.ROOT/'distributed_data/datapack/extra/research-library.json.gz'
    seen=set();found=set();hashed=0;complete=None;current=False
    for record in iter_records(pack):
        if record.get('complete'):complete=record;continue
        path=record.get('path')
        if path is None:continue
        assert path not in seen;seen.add(path)
        if path=='/api/traces/research-space':
            current=any(r['run_id']==C.RUN and r['report_sha256']==reg['report_sha256'] for r in record['data']['studies'])
        if path.startswith('/api/research-library/artifacts/'):
            sha=path.rsplit('/',1)[1];assert hashlib.sha256(canonical(record['data'])).hexdigest()==sha,path
            hashed+=1
            if sha in required:found.add(sha)
    assert complete and len(seen)==complete['n_entries'] and current
    assert required==found,sorted(required-found)
    C.save(C.OUT/'offline-acceptance.json',{**C.identity(),'status':'passed','n_entries':len(seen),'verified_artifacts':hashed,
          'required_refs':len(found),'version':complete['version'],'compressed_bytes':pack.stat().st_size},False)
    print(len(seen),'entries,',hashed,'hashes,',len(found),'current references verified',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('part',choices=['observed','final','offline']);part=p.parse_args().part
    offline() if part=='offline' else live(part=='final')
