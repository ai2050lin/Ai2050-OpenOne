"""Check the live client API against the completed real CUDA evidence."""
import json
from urllib.error import HTTPError
from urllib.request import urlopen

import numpy as np
import mlp_fidelity_v2_plan as P


def main():
    checks = []

    def get(path, status=200):
        try:
            with urlopen('http://127.0.0.1:5001/api'+path, timeout=90) as r:
                code, value = r.status, json.load(r)
        except HTTPError as e:
            code, value = e.code, json.load(e)
        assert code == status, (path, code, value if code != 200 else 'unexpected success')
        checks.append({'path':path, 'status':code})
        return value

    reg = P.read(P.OUT/'registration-delivery.json')
    space = get('/traces/research-space')
    study = next(s for s in space['studies'] if s['run_id']==P.RUN)
    assert study['report_sha256']==reg['report_sha256'] and study['model_snapshot']==reg['model_snapshot']
    report = get('/research-library/artifacts/'+reg['report_sha256'])
    assert report['state']=='complete' and report['fresh_evaluation']['passed_layers']==[35]
    assert report['semantics']['behavior_gate'] and not report['tails']['generated']['summary']['effect_supported']
    graph = get('/research-library/artifacts/'+reg['graph_sha256'])
    assert len(graph['feature_cards'])==51 and len(graph['graph_edges'])==50 and graph['semantic_mechanisms_closed']==0
    bank = f'/traces/mlp-bank/{P.RUN}'
    man = get(bank+'/manifest')
    assert man['completed']==4096 and man['state']=='complete'
    row = get(bank+'/samples/4095')
    assert row['input_tokens']==1034 and row['bank_index']==4095
    for stage in [0,1]:
        for block, boundary, shape in [(35,'mlp_activation',[9728]),(35,'head_z',[32,128]),(0,'embedding',[2560]),(36,'hidden_state',[2560])]:
            v=get(f'{bank}/samples/4095/field?block={block}&stage={stage}&boundary={boundary}')
            assert v['shape']==shape and v['position']==row['analysis_readouts'][stage] and v['source']=='native_BF16_bits'
    get(bank+'/samples/4095/field?block=0&boundary=mlp_activation',404)
    get(bank+'/samples/4095/field?block=35&boundary=attention_rows',404)
    get(bank+'/samples/4096',422)
    conn=get(bank+'/samples/4095/connections?block=35&stage=0&unit=9727&head=31')
    assert conn['verified_full_snapshot'] and not conn['attention_rows_captured']
    fid='/traces/mlp-fidelity/v2'
    assert len(get(fid+'/layers/35')['lockbox_feature_firings'])==4096
    f=get(fid+'/features/35/4095')
    assert len(f['weights']['encoder'])==len(f['weights']['decoder'])==2560 and len(f['native_correlations'])==9728
    get(fid+'/features/35/4096',422)
    assert len(get(fid+'/units/35/9727')['correlations'])==4096
    get(fid+'/units/0/0',409)
    n=get(fid+'/samples/natural/511/7?block=35')
    assert n['prefix'] and n['source'] and len(n['latent'])==4096 and len(n['input'])==len(n['output'])==2560
    s=get(fid+'/samples/controlled/4095?block=35&stage=0')
    z=np.load(P.OUT/'semantics-v2/B35-latent.npy',mmap_mode='r')
    assert s['latent_source']=='registered_FP32_array' and np.array_equal(np.asarray(s['latent'],np.float32),z[4095,0])
    for part,index,slot in [('natural',511,7),('generated',383,0),('capability',2339,0)]:
        t=get(f'{fid}/samples/tail/{part}/{index}/{slot}')
        assert t['model_snapshot']==reg['model_snapshot']
        assert {k:len(v) for k,v in t['fields'].items()}=={'activation':9728,'output':2560,'residual':2560,'final-hidden':2560}
        if part=='generated':
            parent=get(f'{bank}/samples/{t["row"]["bank_index"]}')
            assert parent['expected']==t['row']['expected'] and parent['background_index']==t['row']['background_index']
        if part=='capability':
            assert isinstance(t['row']['gate_eligible'], bool)
    get(fid+'/samples/tail/generated/384/0',422)
    P.save(P.OUT/'live-api-acceptance.json',{'schema':'mlp_fidelity_api_acceptance.v1','run_id':P.RUN,'report_sha256':reg['report_sha256'],'checks':checks,'status':'passed'})
    print(f'{len(checks)} real API routes and expected error responses passed',flush=True)


if __name__=='__main__':
    main()
