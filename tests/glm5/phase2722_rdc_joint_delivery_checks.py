"""Independent arithmetic, provenance, original-array and extended query acceptance."""
import hashlib,sys,urllib.request
from collections import Counter
from scipy.special import expit
from rdc_joint_common import *
from rdc_joint_capture import array_identity


def main():
    start=time.monotonic();guard(4*1024**2);checks=[];tail=BASE/'extension/tail_confirmation'
    frozen=read(BASE/'frozen.json')
    for name,digest in frozen['files'].items():assert sha(BASE/name)==digest,name
    for item in read(BASE/'review.json')['evidence']:assert sha(ROOT/item['path'])==item['sha256']
    checks.append({'name':'all125_frozen_and21_prior_artifacts_unchanged','passed':True})
    prefix=read(BASE/'memo_prefix.json');memo=(ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md').read_bytes()
    assert hashlib.sha256(memo[:prefix['bytes']]).hexdigest()==prefix['sha256']
    checks.append({'name':'original_MEMO_byte_prefix_unchanged','passed':True,'bytes':prefix['bytes']})
    for fresh in (False,True):
        scope='fresh' if fresh else 'main';count=0
        for r in rows(fresh):
            z=field(r,fresh);commit=read(BASE/scope/'commits'/f'{r["sample_id"]}.json')
            assert {k:array_identity(v) for k,v in z.items()}==commit['arrays'],r['sample_id'];count+=1
        checks.append({'name':scope+'_every_original_array_SHA_dtype_shape','sources':count,'passed':True})
    for key in ('qwen4','qwen14','glm4'):
        rr=read(BASE/'scale'/key/'result.json');assert rr['sources']==224 and rr['dtype']=='torch.bfloat16' and not rr['quantized']
        for p in (BASE/'scale'/key/'rows').glob('*.json'):
            r=read(p);assert sha(BASE/'scale'/key/'fields'/f'{r["sample_id"]}.npz')==r['field_sha']
        checks.append({'name':key+'_all224_archive_files_SHA','width':rr['width'],'passed':True})
    meta=json.loads(gzip.decompress((tail/'all_token_results.json.gz').read_bytes()));threshold=read(BASE/'extension/event_threshold.json');non=[r for r in meta if r['position']>0]
    energy=np.array([r['energy_H12_H23_H36'] for r in non]);ev=(energy[:,1]>threshold['H23_energy_threshold'])&(energy[:,1]/np.maximum(energy[:,0],1e-20)>=threshold['energy_ratio_min'])
    assert np.array_equal(ev,[r['event'] for r in non]) and len(non)==45113 and ev.sum()==19
    with np.load(tail/'full_coordinate_moments.npz') as z:
        assert np.allclose(z['pooled_all_noninitial'][1].mean(-1),energy.mean(0),rtol=3e-7)
        assert np.allclose(z['pooled_event'][1].mean(-1),energy[ev].mean(0),rtol=3e-7)
        stored_event=z['pooled_event']
    event_values=[]
    for p in sorted((tail/'events').glob('*.npz')):
        with np.load(p) as z:event_values.append(np.stack([unbits(z[k]) for k in ('h12','h23','h36')],1).astype(float))
    event_values=np.concatenate(event_values)
    assert np.allclose(event_values.mean(0),stored_event[0],rtol=3e-7,atol=1e-6)
    assert np.allclose(np.mean(event_values*event_values,0),stored_event[1],rtol=3e-7,atol=1e-6)
    checks.append({'name':'all_new_tokens_threshold_and_full_coordinate_event_energy_accounting','all_tokens':len(meta),'noninitial':len(non),'events':int(ev.sum()),'passed':True})
    with np.load(BASE/'extension/event_forecast.npz') as z:fit={k:z[k] for k in z.files}
    lookup={(r['sample_id'],r['position']):r for r in meta};maximum=0.;n=0
    for area in ('fields','events'):
      for p in (tail/area).glob('*.npz'):
        with np.load(p) as z:
            x=unbits(z['h12']);positions=z['positions'] if 'positions' in z.files else range(len(x))
            predictions=expit(((x-fit['mean'])/fit['standard_deviation']).astype(float)@fit['coefficient']+fit['intercept'][0])
        expected=np.array([lookup[p.stem,int(i)]['prediction']['full_H12_linear_logistic'] for i in positions]);maximum=max(maximum,float(np.max(np.abs(expected-predictions))));n+=len(x)
    assert maximum<1e-12,maximum
    checks.append({'name':'frozen_predictor_matches_every_retained_new_event_and_fixture_token','tokens':n,'max_abs_difference':maximum,'passed':True})
    for name,digest in read(tail/'protocol.json')['frozen_files'].items():assert sha(BASE/name)==digest
    sys.path.insert(0,str(ROOT))
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from server.rdc_joint_service import router,extension_index
    app=FastAPI();app.include_router(router);client=TestClient(app);items=extension_index();kinds=sorted({r['kind'] for r in items})
    assert len(kinds)==8,kinds
    for kind in kinds:
        eligible=[r for r in items if r['kind']==kind]
        for item in (eligible[0],eligible[-1]):
            with np.load(BASE/item['file']) as z:a=z[item['array']];a=unbits(a) if a.dtype==np.uint16 else a
            a=a.reshape(-1,a.shape[-1]);row=max(0,len(a)-2);startcol=max(0,a.shape[1]-7)
            response=client.get('/api/rdc-joint/extension-field',params={'id':item['id'],'row_start':row,'row_count':2,'start':startcol,'count':7})
            assert response.status_code==200,response.text;r=response.json()
            assert np.array_equal(np.array(r['values']),a[row:row+2,startcol:]) and r['native_width']==a.shape[1]
            assert r['source']['sample_id']==item['sample_id']
        checks.append({'name':'extension_original_full_tail_columns_'+kind,'registered_arrays':len(eligible),'cases':2,'passed':True})
    assert client.get('/api/rdc-joint/extension-field',params={'id':'../../models'}).status_code==404
    assert client.get('/api/rdc-joint/extension-field',params={'id':items[0]['id'],'row_start':1000000}).status_code==422
    checks.append({'name':'extension_scope_and_row_rejection','passed':True})
    registered=client.get('/api/rdc-joint/analysis-index').json()
    for width in (4096,5120):
        item=next(r for r in registered if r['shape'][-1]==width and 'all_layer' in r['id'])
        r=client.get('/api/rdc-joint/analysis-field',params={'id':item['id'],'row_start':0,'row_count':1,'count':width}).json()
        assert len(r['values'][0])==width
    checks.append({'name':'larger_model_all_layer_full_coordinate_summaries','passed':True})
    live={}
    for path in ('/api/rdc-joint/overview','/api/rdc-relation/overview','/api/rdc-prefix/overview'):
        with urllib.request.urlopen('http://127.0.0.1:5001'+path,timeout=20) as response:live[path]=response.status;assert response.status==200
    checks.append({'name':'restored_full_service_including_old_atlas_routes','statuses':live,'passed':True})
    scale={}
    for model in ('qwen4','qwen14','glm4'):
        scale[model]={}
        for a,b in [('current_linear','current_quadratic'),('temporal_linear','temporal_bilinear')]:
            aa=json.loads(gzip.decompress((BASE/'scale'/model/f'{a}_fresh_rows.json.gz').read_bytes()));bb=json.loads(gzip.decompress((BASE/'scale'/model/f'{b}_fresh_rows.json.gz').read_bytes()))
            assert [(r['sample_id'],r['anchor']) for r in aa]==[(r['sample_id'],r['anchor']) for r in bb]
            scale[model][a+'_minus_'+b]={metric:paired_summary([x[metric]-y[metric] for x,y in zip(aa,bb)],[r['source_group'] for r in aa]) for metric in ('MSE','KL')}
    save(BASE/'verification/scale_paired_differences.json',{'timestamp':stamp(),'comparisons':scale,'limits':'Within-model common-source differences, not raw-MSE comparisons across models;49 declared source clusters, tokenizer and architectures differ.'})
    snapshots=[snapshot(p) for p in (ROOT/'tests/glm5').glob('phase272[12]_rdc_joint*.py')]
    snapshots.extend(snapshot(ROOT/p) for p in ['server/rdc_joint_service.py','frontend/src/components/app/RdcJointAtlas.jsx','frontend/src/components/app/RdcJointAtlas.css'])
    report={'timestamp':stamp(),'passed':True,'checks':checks,'source_snapshots':snapshots,'result_usage_bytes':usage(),'seconds':time.monotonic()-start,
        'scope':'Scientific arithmetic/provenance and API acceptance. Browser-visible acceptance is separately recorded; no claim of full language mechanism or full model-weight-file SHA verification.'}
    prior=BASE/'verification/phase2722_delivery.json';archived=BASE/'verification/phase2722_delivery_before_regimes.json'
    if prior.exists() and not archived.exists():shutil.copyfile(prior,archived)
    save(prior,report);print('JOINT_EXTENDED_DELIVERY_PASS',len(checks),'checks',report['seconds'],'seconds',flush=True)


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):main()
