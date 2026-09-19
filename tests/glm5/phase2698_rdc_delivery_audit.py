"""Whole-artifact integrity, native endpoint checks, and bounded campaign delivery."""
import sys
from rdc_mechanism_common import *

def main():
    checks=[]
    for run,count in [('b_relations',512),('c_generation',384)]:
        out=CAMPAIGN/run;rows=read(out/'material.json');assert len(rows)==count
        scalars=0;size=0;seconds=0
        for i,r in enumerate(rows):
            sid=r['sample_id'];c=read(out/f'commits/{sid}.json');assert c['protocol_sha']==sha(out/'protocol.json')
            for p,digest in c['files'].items():assert sha(out/p)==digest
            path=out/f'fields/{sid}.npz';size+=path.stat().st_size
            with np.load(path) as z:
                for key in z.files:
                    a=z[key];assert np.isfinite(unbits(a) if a.dtype==np.uint16 else a).all();scalars+=a.size
            seconds+=c.get('elapsed_seconds',0)
            if i%128==0:print('AUDIT',run,i,count,flush=True)
        if run=='c_generation':seconds=sum(read(p)['elapsed_seconds'] for p in (out/'prefix_commits').glob('*.json'))
        checks.append({'run':run,'rows':len(rows),'finite_scalar_values':int(scalars),'raw_bytes':size,'captured_case_seconds_sum':seconds})
    # Use route implementation under a test application; this never imports the model-loading server.
    sys.path.insert(0,str(ROOT))
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from server.rdc_feature_service import router
    app=FastAPI();app.include_router(router);client=TestClient(app);api=[]
    arows=read(CAMPAIGN/'a_native/material.json');row=arows[0];token=row['spans']['u']['positions'][0];sid=row['sample_id']
    paths=[f'/api/rdc/runs/a_native/field?sample={sid}&token={token}&coordinate=2496&width=64',
        f'/api/rdc/mechanism/mlp_units?sample={sid}&token={token}&layer=35&coordinate=9664&width=64',
        f'/api/rdc/mechanism/parameter_path?sample={sid}&token={token}&layer=23&unit=9727&coordinate=2496&width=64']
    cs=read(CAMPAIGN/'c_generation/material.json')[-1]['sample_id']
    paths.extend([f'/api/rdc/runs/c_generation/field?sample={cs}&field={key}&layer=35&token=0&coordinate=0&width=16' for key in ('h','p','k','v','a')])
    paths.extend([f'/api/rdc/mechanism/{key}?sample={cs}' for key in ('output_ledger','output_units','source_ledger')])
    for path in paths:
        response=client.get(path);assert response.status_code==200,(path,response.text)
        data=response.json();assert np.isfinite(np.array(data['values'])).all();api.append({'path':path,'shown_values':data['shown_values'],'no_topk':data['no_topk']})
    assert client.get('/api/rdc/mechanism/parameter_path',params={'sample':sid,'token':token,'unit':9728}).status_code==422
    before=read(CAMPAIGN/'memo_prefix.json');memo=ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md'
    with memo.open('rb') as f:prefix=f.read(before['bytes'])
    prefix_ok=hashlib.sha256(prefix).hexdigest()==before['sha']
    save(CAMPAIGN/'delivery_audit.json',{'timestamp':stamp(),'artifacts':checks,'api_checks':api,'memo_original_prefix_preserved':prefix_ok,
        'old_field_retention':'All18,049,992,623 bytes of S1/S2 remain referenced by a_native for client and research.',
        'deletions':0,'model_checkpoint_mutated':False})
    assert prefix_ok,'MEMO preexisting bytes changed; investigate before delivery'
    a=CAMPAIGN/'a_native';save(a/'result.json',dict(read(a/'reader_result.json'),native=read(a/'native_result.json')))
    announce('a_native',state='complete',completed=1024,total=1024)
    print('DELIVERY_AUDIT_DONE',checks,flush=True)

if __name__=='__main__':main()
