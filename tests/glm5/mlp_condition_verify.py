"""Read-only, independent checks of frozen raw evidence and remote CAS publication."""
import argparse
import base64
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import tarfile
from urllib.request import urlopen

import numpy as np
from deploy.research_library import canonical
import mlp_condition_v2 as V
import mlp_condition_design as D


def local():
    plan=D.checked();frozen=D.READ(D.OUT/'frozen-predictor.json')
    assert frozen['frozen_before_holdout']
    assert frozen['calibration_sha256']==D.HASH(D.OUT/'discovery-calibration.npz')
    assert frozen['selection_sha256']==D.HASH(D.OUT/'selection.json')
    assert D.READ(D.OUT/'selection.json')['behavior_gate_passed']
    families={};identities=set();norms=[];replays=0
    for split,n in [('discovery',128),('holdout',512)]:
        fs=D.families(plan,split);assert len(fs)==n;families[split]=n
        for f in fs:
            assert len(f['rows'])==16
            assert len({len(r['tokens']) for r in f['rows']})==1
            assert len({tuple(r[k] for k in ('identity','role','late','mapping')) for r in f['rows']})==16
            for r in f['rows']:
                key=hashlib.sha256(canonical(r['tokens'])).hexdigest()
                assert key not in identities;identities.add(key)
                assert len(r['positions'])==6 and r['positions']==sorted(r['positions'])
            meta=D.READ(D.OUT/f'{split}-{f["family"]:04d}.json')
            assert meta['sha256']==D.HASH(D.OUT/f'{split}-{f["family"]:04d}.npz')
            if split=='holdout':assert meta['prediction_frozen_sha256']==D.HASH(D.OUT/'frozen-predictor.json')
    for f in range(512):
        r=D.READ(D.OUT/f'causal-{f:04d}.json');assert r['restore_bitwise'];replays+=1
        for label,sha in r['prediction_hashes'].items():
            assert sha==D.HASH(D.OUT/f'prediction-{f:04d}-{label}.json')
        assert len(r['curves'])==6
        for c in r['curves']:
            error=abs(c['norm']-c['control_norm'])/max(c['norm'],1e-12)
            if c['norm_valid']:assert error<=.010001
            norms.append(error)
    return {'families':families,'unique_prompts':len(identities),'restored':replays,
            'controls':len(norms),'max_relative_norm_error':max(norms)}


def bundle(remote=None):
    reg=D.READ(D.OUT/'registration.json');path=D.ROOT/reg['bundle']
    assert D.HASH(path)==reg['bundle_sha256'];objects={}
    with tarfile.open(path) as archive:
        manifest=json.load(archive.extractfile('MANIFEST.json'))
        for o in manifest['objects']:
            sha=o['sha256'];raw=archive.extractfile(f'objects/{sha}.json').read()
            assert hashlib.sha256(raw).hexdigest()==sha
            objects[sha]=json.loads(raw)
    report=objects[reg['report_sha256']]
    assert report['formal_evidence_promotion'] is False
    for entry in report['raw_manifest']:
        raw_path=D.OUT/entry['path']
        assert raw_path.stat().st_size==entry['bytes'] and D.HASH(raw_path)==entry['sha256'],entry['path']
    for b,sha in reg['condition_layers'].items():
        layer=objects[sha];assert layer['block']==int(b)
        assert all(len(a)==9728 and np.isfinite(a).all() for a in layer['arrays'].values())
    assert len(reg['condition_layers'])==8
    weights=objects[report['native_parameter_sha256']]['parameters']
    assert len(weights)==4
    assert all(len(v['values'])==2560 for p in weights.values() for v in p.values())
    for sha in report['output_predictor_artifacts']:
        v=objects[sha]['logit_slope'];array=np.frombuffer(base64.b64decode(v['data']),dtype=v['dtype'])
        assert list(array.shape)==v['shape'] and np.isfinite(array).all()
    result={'objects':len(objects),'bundle_bytes':path.stat().st_size,'condition_units':77824,
            'controlled_units':4,'full_parameter_values':4*3*2560,'remote':remote}
    if remote:
        def read(route):
            with urlopen(remote+route,timeout=90) as response:return json.load(response)
        index=read('/api/traces/research-space')
        study=next(s for s in index['studies'] if s['run_id']==D.RUN and s['model_snapshot']==D.SNAPSHOT)
        assert study['report_sha256']==reg['report_sha256']
        def verify(sha):
            doc=read('/api/research-library/artifacts/'+sha)
            assert hashlib.sha256(canonical(doc)).hexdigest()==sha,sha
            return sha
        with ThreadPoolExecutor(max_workers=3) as pool:
            verified=list(pool.map(verify,objects))
        result.update(remote_verified=len(verified),study=study)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--remote');args=parser.parse_args();V.configure()
    record={'provenance':D.checked()['provenance'],'raw_checks':local(),'bundle_checks':bundle(args.remote)}
    D.SAVE(D.OUT/('server-acceptance.json' if args.remote else 'local-acceptance.json'),record)
    print(record,flush=True)
