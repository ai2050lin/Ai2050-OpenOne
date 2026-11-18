"""HTTP readback against every real native layer and model-specific parameter width."""
import argparse
import json
import time
import urllib.request
import urllib.error
import numpy as np
import goodfire_middle as M
C=M.C
BASE='http://127.0.0.1:5035/api/research/native-mlp'


def get(route='',**query):
    from urllib.parse import urlencode
    query={'run':M.RUN,**query}
    with urllib.request.urlopen(BASE+route+'?'+urlencode(query),timeout=120) as r:return json.load(r)


def main():
    p=argparse.ArgumentParser();p.add_argument('model',choices=M.SPECS);a=p.parse_args();M.configure(a.model)
    start=time.monotonic();report=get();assert report['blocks']==M.BLOCKS and report['n_units']==11*C.WIDTH
    layers=[];units=[]
    for b in M.BLOCKS:
        d=get(f'/layers/{b}');assert d['run_id']==M.RUN
        assert all(len(v)==C.WIDTH and np.isfinite(v).all() for v in d['arrays'].values())
        profile=C.read(M.OUT/f'B{b}-profile.json')
        for f in ['G','V','A']:
            for k in ['mean','std','min','max']:assert d['arrays'][f+'_'+k]==profile['fields'][f][k]
        with np.load(M.OUT/f'B{b}-regimes.npz') as z:
            np.testing.assert_array_equal(d['arrays']['double_negative_mass'],z['holdout_quadrant_abs_mass_fraction'][:,3])
        layers.append(b)
        for u in [0,C.WIDTH-1]:
            card=get(f'/units/{b}/{u}');assert card['run_id']==M.RUN and not card['semantic_confirmed']
            for f in ['gate','up','down']:
                raw=np.load(M.OUT/f'B{b}-W-{f}.npy',mmap_mode='r')
                np.testing.assert_array_equal(card['weights'][f],C.unbits(raw[:,u] if f=='down' else raw[u]))
            for e in card['examples']:
                di=e['document_index'];slot=report['positions'].index(e['token_position'])
                for f in ['G','V','A']:
                    raw=np.load(M.OUT/f'B{b}-{f}.npy',mmap_mode='r');assert e[f]==float(C.unbits(raw[di,slot,u]))
            with np.load(M.OUT/f'B{b}-prediction.npz') as z:
                assert card['standardized_prediction']['A_holdout_composed_mse']==float(z['A_holdout_composed_mse'][u])
            units.append([b,u])
        M.log('http_native_readback',b-17,11,start)
    selected=report['intervention']['selected_units'];assert len(selected)==6
    for row in selected:
        b,u=row['block'],row['unit']
        for stage,n in [('natural',256),('capability',512)]:
            d=get(f'/interventions/{stage}/{b}/{u}',sample=n-1)
            assert len(d['rows'])==6 and len(d['controls'])==3 and len(d['replay'])==1
            assert d['input_text'] and d['input_tokens']
    relations=[]
    for b in M.BLOCKS[:-1]:
        d=get(f'/relations/{b}/0',kind='rho-holdout');assert len(d['values'])==C.WIDTH and d['target']==b+1
        relations.append([b,b+1])
    for kind in ['rho-holdout','weight-gate','weight-up']:
        x=get('/relations/27/0',kind=kind);y=get(f'/relations/28/{C.WIDTH-1}',direction='incoming',kind=kind)
        assert abs(x['values'][C.WIDTH-1]-y['values'][0])<1e-6
    guards=[];b,u=selected[0]['block'],selected[0]['unit']
    for route,params,code in [('/layers/17',{},422),('/layers/29',{},422),('/units/28/'+str(C.WIDTH),{},422),
        ('/relations/28/0',{},422),('',dict(run='unknown'),404),('',dict(snapshot='wrong'),409),
        ('/interventions/natural/19/0',dict(sample=0),404),(f'/interventions/natural/{b}/{u}',dict(sample=256),422)]:
        try:get(route,**params);raise AssertionError(route)
        except urllib.error.HTTPError as error:assert error.code==code;guards.append(dict(route=route,status=code))
    M.write(M.OUT/'api-acceptance.json',dict(passed=True,layers=layers,endpoint_units=units,intervened_units=selected,
        adjacent_boundaries=relations,guards=guards,seconds=time.monotonic()-start,
        manifest_sha256=C.file_sha(M.OUT/'manifest.json'),scope='Readback of real frozen artifacts; no new model computation.'))
    print('API acceptance passed',M.MODEL_KEY,flush=True)


if __name__=='__main__':main()
