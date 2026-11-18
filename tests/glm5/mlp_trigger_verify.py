"""Independent raw-byte, design-custody, control and remote checks."""
import argparse
import base64
import hashlib
import io
import json
import tarfile
import time
from http.client import IncompleteRead
from urllib.error import HTTPError, URLError
from concurrent.futures import ThreadPoolExecutor
from urllib.request import urlopen
import numpy as np
from deploy.research_library import canonical
from shared.dist_protocol.mechanism_trace import unbits
import mlp_trigger_design as D

def local():
    p=D.checked();frozen=D.READ(D.OUT/'frozen.json');assert frozen['heldout_unopened']
    assert frozen['selection_sha256']==D.HASH(D.OUT/'selection.json') and frozen['calibration_sha256']==D.HASH(D.OUT/'calibration.npz')
    audit_root=D.ROOT/p['discovery_reused_from'] if p.get('discovery_reused_from') else D.OUT
    assert D.READ(audit_root/'implementation-audit.json')['heldout_unopened']
    assert all(v['pass'] for v in p['power'].values())
    seen=set();files=[];peaks=0
    for split,n in [('discovery',D.N_DISC),('holdout',D.N_HOLD)]:
        rows=D.materials(p,split);assert len(rows)==8*n
        raw_root=audit_root if split=='discovery' else D.OUT
        for f in range(n):
            path=raw_root/f'{split}-{f:04d}.npz';meta=D.READ(path.with_suffix('.json'));assert meta['native_sha256']==D.HASH(path)
            assert meta['plan_sha256']==D.HASH(raw_root/'prereg.json')
            z=np.load(path);a=unbits(z['prefix']);r4=unbits(z['R4']);res=unbits(z['residual']);chunk=rows[8*f:8*f+8]
            assert z['prefix'].dtype==np.uint16 and z['residual'].shape==(8,2,2560)
            for long in [0,1]:assert len({len(r['tokens']) for r in chunk if r['long']==long})==1
            assert len({(r['long'],r['repetition'],r['past']) for r in chunk})==8
            for k,(r,m) in enumerate(zip(chunk,meta['rows'])):
                h=hashlib.sha256(canonical(r['tokens'])).hexdigest();assert h not in seen;seen.add(h)
                length=len(r['tokens']);pos=64+int(abs(r4[k,2,64:length]).argmax())
                assert pos==m['positions']['rare'] and res[k,1,4]==r4[k,2,pos]
                assert np.array_equal(np.flatnonzero(abs(a[k,2,2,64:length])>10)+64,m['rare_positions'])
                assert not a[k,:,:,length:].any() and not r4[k,:,length:].any();peaks+=len(m['rare_positions'])
            files.append({'path':path.name,'sha256':D.HASH(path),'bytes':path.stat().st_size})
    controls=[];restored=0;prediction_commits=0
    for f in range(D.N_HOLD):
        doc=D.READ(D.OUT/f'causal-{f:04d}.json');path=D.OUT/f'prediction-{f:04d}.json';prediction=D.READ(path)
        assert doc['prediction_sha256']==D.HASH(path) and prediction['frozen_sha256']==D.HASH(D.OUT/'frozen.json')
        assert path.stat().st_mtime_ns<=(D.OUT/f'causal-{f:04d}.json').stat().st_mtime_ns
        assert len(doc['records'])==8 and len(prediction['predictions'])==8
        assert doc['bitwise_restored'];restored+=1;prediction_commits+=1
        for c in doc['records']:
            e=abs(c['norm']-c['control_norm'])/max(c['norm'],1e-30)
            assert c['norm_valid']==(c['norm']>0 and e<=.01)
            assert c['effect']['kl_to_baseline']>=-1e-12 and c['effect']['kl_to_prediction']>=-1e-12
            controls.append({'valid':c['norm_valid'],'error':e})
    cap=D.READ(D.OUT/'capability.json')['rows'];assert len(cap)==2200 and len({v['id'] for v in cap})==2200
    rep=D.READ(D.OUT/'repeatability.json')['rows'];assert all(v['repeat_bitwise'] for v in rep)
    result={'unique_inputs':len(seen),'raw_files':files,'detected_prefix_events':peaks,'prediction_commits':prediction_commits,
       'restoration_checks':restored,'controls':len(controls),'valid_controls':sum(c['valid'] for c in controls),
       'maximum_valid_norm_error':max(c['error'] for c in controls if c['valid']),'capability_items':len(cap),
       'batch_shape_changed_peaks':sum(c['batch1_peak']!=c['batch4_peak'] for c in rep),
       'five_fields_complete':all(k in p['provenance'] for k in ['git_sha','git_dirty','model_path','model_revision','seed']),
       'scientific_promotion':False,'reason':'Single physical node; automatic checks are technical, not independent scientific replication.'}
    D.save('verification.json',result,p);print({k:v for k,v in result.items() if k!='raw_files'},flush=True)

def remote(center):
    p=D.checked();reg=D.READ(D.OUT/'registration.json')
    def get(route):
        for attempt in range(5):
            try:
                with urlopen(center+route,timeout=120) as res:return json.load(res)
            except (IncompleteRead,URLError,TimeoutError,ConnectionError) as exc:
                if isinstance(exc,HTTPError) and exc.code<500 and exc.code!=429:raise
                if attempt==4:raise
                print(f'remote retry {attempt+1}: {route} ({type(exc).__name__})',flush=True)
                time.sleep(1+attempt)
    index=get('/api/traces/research-space');study=next(s for s in index['studies'] if s['run_id']==D.RUN and s['model_snapshot']==D.SNAPSHOT)
    assert all(study[k]==reg[k] for k in ['report_sha256','binding_sha256','workpackage_sha256'])
    def verify(sha):
        doc=get('/api/research-library/artifacts/'+sha);assert hashlib.sha256(canonical(doc)).hexdigest()==sha
        native_files=0
        if doc.get('schema')=='mlp_trigger_native.v1':
            for file in doc['files']:
                raw=base64.b64decode(file['data']);assert hashlib.sha256(raw).hexdigest()==file['sha256']
                with np.load(io.BytesIO(raw)) as z:assert z['prefix'].dtype==np.uint16 and z['residual'].shape==(8,2,2560)
                native_files+=1
        return {'sha256':sha,'native_files':native_files}
    verified=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for obj in pool.map(verify,reg['all_objects']):
            verified.append(obj)
            if len(verified)%5==0 or len(verified)==len(reg['all_objects']):
                print(f'remote verified {len(verified)}/{len(reg["all_objects"])}',flush=True)
    result={'center':center,'study':study,'verified_objects':len(verified),'verified_native_families':sum(v['native_files'] for v in verified),'objects':verified}
    assert result['verified_native_families']==reg.get('native_family_count',480)
    D.save('remote-verification.json',result,p);print({k:v for k,v in result.items() if k!='objects'},flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--remote');args=parser.parse_args()
    if args.remote:remote(args.remote.rstrip('/'))
    else:local()
