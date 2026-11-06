"""Live proof for the early read-route case index and bounded offline hash verification."""
import argparse
import hashlib
import json
from urllib.request import urlopen

import mlp_read_route_common as C
from deploy.research_library import canonical
from deploy.research_pack_stream import iter_records


def live():
    out=C.OUT;reg=C.read(out/'registration-final.json');checks=[]
    def get(path):
        with urlopen('http://127.0.0.1:5001/api'+path,timeout=45) as r:value=json.load(r)
        checks.append(path);return value
    def artifact(sha):return get('/research-library/artifacts/'+sha)
    study=next(r for r in get('/traces/research-space')['studies'] if r['run_id']==C.RUN)
    assert study['report_sha256']==reg['report_sha256'] and study['model_snapshot']==reg['model_snapshot']
    report=artifact(reg['report_sha256']);graph=artifact(reg['graph_sha256'])
    assert report['state']=='complete' and len(graph['feature_cards'])==reg['cards'] and len(graph['graph_edges'])==reg['edges']
    for key,entry in reg['cases'].items():
        if not entry['n']:continue
        last=artifact(entry['chunks'][-1]);assert last['collection']==key and last['start']+len(last['cases'])==entry['n']
    native=reg['native'];rows=C.read_gz(out/'holdout-native.json.gz')['rows']
    assert C.file_sha(out/'holdout-native.json.gz')==native['provenance']['raw_slots_sha256']
    assert C.file_sha(out/'holdout-weights.npz')==native['provenance']['full_weight_sha256']
    for i in [0,native['n']-1]:
        size=native['chunk_size'];page=artifact(native['chunks'][i//size]);r=page['cases'][i%size]
        assert r==rows[i] and all(p<991 for p in r['positions'])
        n=reg['cases']['causal']['chunk_size'];case=artifact(reg['cases']['causal']['chunks'][i//n])['cases'][i%n]
        assert case['id']==r['id'] and case['replay_logits_bitwise_equal'] and case['rescue_logits_bitwise_equal']
    assert all(v['n']==2200 for v in report['capability']['summary'].values())
    assert report['causal']['checks']['strict_matched_control']['max_pre_rounding_relative_error']<1e-5
    C.save(out/'live-acceptance.json',{**C.identity(),'status':'passed','checks':checks,'report_sha256':reg['report_sha256']},False)
    print(len(checks),'live artifact checks passed',flush=True)


def offline():
    out=C.OUT;reg=C.read(out/'registration-final.json');phase=C.read(out/'phase-registration.json')
    required={reg['report_sha256'],reg['graph_sha256'],*reg['refs'].values(),phase['summary_sha256'],phase['acceptance_sha256'],*reg['native']['chunks']}
    for v in reg['cases'].values():required.update(v['chunks'])
    pack=C.ROOT/'distributed_data/datapack/extra/research-library.json.gz';seen=set();found=set();hashed=0;tail=None;current=False
    for r in iter_records(pack):
        if r.get('complete'):tail=r;continue
        path=r.get('path')
        if not path:continue
        assert path not in seen;seen.add(path)
        if path=='/api/traces/research-space':current=any(v['run_id']==C.RUN and v['report_sha256']==reg['report_sha256'] for v in r['data']['studies'])
        if path.startswith('/api/research-library/artifacts/'):
            sha=path.rsplit('/',1)[1];assert hashlib.sha256(canonical(r['data'])).hexdigest()==sha,path
            hashed+=1
            if sha in required:found.add(sha)
    assert tail and tail['n_entries']==len(seen) and current and found==required,(current,sorted(required-found))
    C.save(out/'offline-acceptance.json',{**C.identity(),'status':'passed','entries':len(seen),'hashed_artifacts':hashed,
          'current_refs':len(found),'version':tail['version'],'compressed_bytes':pack.stat().st_size},False)
    print(len(seen),'entries',hashed,'hashes',len(found),'current references passed',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('part',choices=['live','offline']);{'live':live,'offline':offline}[p.parse_args().part]()
