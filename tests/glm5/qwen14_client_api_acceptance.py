"""Read every layer and both endpoint units via HTTP against frozen native arrays."""
import json
from pathlib import Path
import sys
import time
import urllib.request
import urllib.error
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from deploy.qwen14_mlp_pack import RUN,RUN_V2,sha
from shared.dist_protocol.mechanism_trace import unbits
from deploy.storage_policy import check_write
OUT=ROOT/'tests/glm5/result/qwen14_client_v2';NATIVE=OUT/'native'
BASE='http://127.0.0.1:5034/api/research/qwen14-mlp'
def get(route):
    with urllib.request.urlopen(BASE+route,timeout=120) as r:return json.load(r)
def main():
    start=time.monotonic();report=get('');assert report['blocks']==list(range(28,40)) and report['n_units']==208896
    assert get('?run='+RUN)['n_units']==139264
    layers=[];units=[]
    for b in range(28,40):
        d=get(f'/layers/{b}');assert d['run_id']==RUN_V2
        assert all(len(a)==17408 and np.isfinite(a).all() for a in d['arrays'].values())
        p=json.loads((NATIVE/f'B{b}-profile.json').read_text(encoding='utf-8'))
        for field in ['G','V','A']:
            for k in ['mean','std','min','max']:assert d['arrays'][field+'_'+k]==p['fields'][field][k]
        with np.load(NATIVE/f'B{b}-regimes.npz') as z:
            np.testing.assert_array_equal(d['arrays']['double_negative_mass'],z['holdout_quadrant_abs_mass_fraction'][:,3])
        layers.append(b)
        for u in [0,17407]:
            card=get(f'/units/{b}/{u}');assert card['run_id']==RUN_V2
            for f in ['gate','up','down']:
                raw=np.load(NATIVE/f'B{b}-W-{f}.npy',mmap_mode='r')
                np.testing.assert_array_equal(card['weights'][f],unbits(raw[:,u] if f=='down' else raw[u]))
            for example in card['examples']:
                di=example['document_index'];slot=report['positions'].index(example['token_position'])
                for f in ['G','V','A']:
                    raw=np.load(NATIVE/f'B{b}-{f}.npy',mmap_mode='r');assert example[f]==float(unbits(raw[di,slot,u]))
            units.append([b,u])
        elapsed=time.monotonic()-start;print(f'API layers={b-27}/12 units={len(units)}/24 elapsed={elapsed:.1f}s eta={elapsed/(b-27)*(39-b):.1f}s',flush=True)
    selected=report['intervention']['selected_units'];assert len(selected)==24
    for row in selected:
        b,u=row['block'],row['unit']
        for stage,n in [('natural',256),('capability',512)]:
            d=get(f'/interventions/{stage}/{b}/{u}?sample={n-1}')
            assert len(d['rows'])==6 and len(d['controls'])==3 and len(d['replay'])==1
            assert d['input_text'] and d['input_tokens']
        print(f'intervention B{b}/U{u} natural and capability source checked',flush=True)
    relations=[]
    for b in range(28,39):
        a=get(f'/relations/{b}/0?kind=rho-holdout');assert len(a['values'])==17408 and a['target']==b+1
        relations.append([b,b+1])
    for kind in ['rho-holdout','weight-gate','weight-up']:
        a=get('/relations/38/0?kind='+kind);b=get('/relations/39/17407?direction=incoming&kind='+kind)
        assert abs(a['values'][17407]-b['values'][0])<1e-6
    guards=[]
    for route,code in [('/layers/39?run='+RUN,422),('/layers/40',422),('/units/39/17408',422),('/relations/39/0',422),('?run=unknown',404),('?snapshot=wrong',409),('/interventions/natural/39/0?sample=0',404),('/interventions/natural/39/9191?sample=256',422)]:
        try:get(route);raise AssertionError(route)
        except urllib.error.HTTPError as error:assert error.code==code;guards.append({'route':route,'status':code})
    result={'passed':True,'layers':layers,'endpoint_units':units,'intervened_units':selected,'adjacent_boundaries':relations,'guards':guards,
        'manifest_sha256':sha(NATIVE/'manifest.json'),'seconds':time.monotonic()-start,'scope':'Same-node real-file readback; no scientific promotion or new model computation.'}
    check_write(OUT/'api-acceptance.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('API acceptance passed',flush=True)
if __name__=='__main__':main()
