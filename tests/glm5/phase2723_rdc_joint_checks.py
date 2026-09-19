"""Independent original-array and corrected-signature checks, plus native parameter API."""
import sys,hashlib
from rdc_joint_common import *
from rdc_joint_capture import array_identity
from rdc_relation_native_parameters import parameter,decode

OUT=BASE/'extension/native_regimes'

def main():
    start=time.monotonic();checks=[];audit=read(OUT/'normalization_audit.json')
    for name,digest in audit['original_hashes'].items():assert sha(OUT/'pre_correction'/name)==digest
    assert sha(OUT/'frozen.json')==audit['original_hashes']['frozen.json']
    assert sha(OUT/'frozen_training_signatures.npz')==audit['original_hashes']['frozen_training_signatures.npz']
    assert sha(OUT/'result.json')==audit['corrected_result_sha']
    checks.append('originals_and_corrected_result_SHA')
    identities=json.loads(gzip.decompress((OUT/'array_identities.json.gz').read_bytes()))
    for r in identities:
        for key in ('fields','factors'):
            with np.load(OUT/key/f'{r["sample_id"]}.npz') as z:assert {k:array_identity(z[k]) for k in z.files}==r[key]
    checks.append('all44_original_full_coordinate_and_unit_packets')
    with np.load(OUT/'corrected_training_signatures.npz') as z:sigs={k:z[k] for k in z.files}
    with np.load(BASE/'native_factors/train_L6_all_unit_and_coordinate_profiles.npz') as z:act=z['activation_sum'][0].astype(float)
    w=decode(parameter(ROOT,'model.layers.6.mlp.down_proj.weight')).astype(float)
    assert np.array_equal(w@act,sigs['L6_first']);del w
    records=json.loads(gzip.decompress((OUT/'rows_corrected.json.gz').read_bytes()))
    for r in records:
        with np.load(OUT/'factors'/f'{r["sample_id"]}.npz') as z:
            j=z['positions'].tolist().index(r['position'])
            for b in (6,16,34):
                m=unbits(z[f'L{b}_mlp'][j]).astype(float);sig=sigs['L6_first' if b==6 else f'L{b}_event']
                assert abs(np.mean((m-sig)**2)-r['blocks'][str(b)]['signature_constant_MSE'])<1e-9
    checks.append('all90_corrected_prototypes_MSE_recomputed_from_native_arrays')
    sys.path.insert(0,str(ROOT));from server.rdc_joint_service import router,extension_index
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    app=FastAPI();app.include_router(router);client=TestClient(app)
    item=next(i for i in extension_index() if i['kind']=='regime_factors' and i['sample_id'].startswith('tail-'))
    with np.load(BASE/item['file']) as z:
        pos=len(z['positions'])-1
        for b in (6,16,34):
            params={'id':item['id'],'block':b,'position_index':pos,'unit':9727,'input_coordinate':2559,'output_coordinate':2559}
            response=client.get('/api/rdc-joint/extension-scalar',params=params);assert response.status_code==200,response.text;r=response.json()
            x=unbits(z[f'L{b}_mlp_input'][pos]).astype(float);a=unbits(z[f'L{b}_activation'][pos]).astype(float)
            down=decode(parameter(ROOT,f'model.layers.{b}.mlp.down_proj.weight')[2559]).astype(float)
            assert np.array_equal(r['unit_terms']['values'][0],a*down)
            assert len(r['input_terms']['values'][0])==2560 and len(r['unit_terms']['values'][0])==9728
            assert abs(r['chain']['all_unit_sum']-np.dot(a,down))<1e-10
            assert r['native_position']==int(z['positions'][pos])
        assert client.get('/api/rdc-joint/extension-scalar',params={'id':item['id'],'block':35}).status_code==422
        assert client.get('/api/rdc-joint/extension-scalar',params={'id':item['id'],'position_index':9999}).status_code==422
    assert client.get('/api/rdc-joint/extension-scalar',params={'id':'../../models'}).status_code==404
    checks.append('3_native_blocks_arbitrary_last_scalar_and_ALL9728_terms_and_bounds')
    registry=client.get('/api/rdc-joint/analysis-index').json()
    assert any('corrected_training_signatures' in x['id'] for x in registry)
    assert not any('/frozen_training_signatures' in x['id'] for x in registry)
    checks.append('active_corrected_signature_only_in_client_index')
    save(BASE/'verification/phase2723_delivery.json',{'timestamp':stamp(),'passed':True,'checks':checks,'seconds':time.monotonic()-start,'source':snapshot(Path(__file__))})
    print('NATIVE_REGIME_DELIVERY_PASS',checks,flush=True)

if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):main()
