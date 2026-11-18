"""Verify live HTTP responses against the archived native 14B arrays."""
import json
import urllib.request
import urllib.error
import numpy as np
import qwen14_mlp_atlas as C
from qwen14_mlp_provenance import save_result

BASE='http://127.0.0.1:5033/api/research/qwen14-mlp'
def get(route):
    with urllib.request.urlopen(BASE+route,timeout=120) as response:return json.load(response)

def main():
    report=get('');assert report['model_snapshot']==C.read(C.OUT/'plan.json')['model_snapshot']
    assert report['n_units']==139264 and report['coverage']['semantic_confirmed']==0
    layers=[]
    for b in C.BLOCKS:
        d=get(f'/layers/{b}');assert d['width']==17408
        assert all(len(v)==17408 and np.isfinite(v).all() for v in d['arrays'].values())
        native=C.read(C.OUT/f'B{b}-profile.json')
        assert d['arrays']['A_std']==native['fields']['A']['std']
        layers.append(b);print(f'API checked B{b} all 17,408 units',flush=True)
    units=[]
    for b,u in [(28,0),(28,17407),(35,0),(35,17407),(28,5921)]:
        d=get(f'/units/{b}/{u}');assert (d['block'],d['unit'])==(b,u)
        assert len(d['arithmetic']['normalized_input'])==5120
        for field in ['gate','up','down']:
            raw=np.load(C.OUT/f'B{b}-W-{field}.npy',mmap_mode='r')
            np.testing.assert_array_equal(d['weights'][field],C.unbits(raw[:,u] if field=='down' else raw[u]))
        for example in d['examples']:
            di=example['document_index'];slot=C.POSITIONS.index(example['token_position'])
            for field in ['G','V','A']:
                raw=np.load(C.OUT/f'B{b}-{field}.npy',mmap_mode='r')
                assert example[field]==float(C.unbits(raw[di,slot,u]))
        if (b,u)==(28,5921):assert d['intervention_cases']
        units.append([b,u]);print(f'API checked B{b}/U{u} full parameters and native examples',flush=True)
    relation_counts=[]
    for kind in ['rho-holdout','weight-gate','weight-up']:
        a=get(f'/relations/28/0?kind={kind}')
        b=get(f'/relations/29/17407?direction=incoming&kind={kind}')
        assert len(a['values'])==len(b['values'])==17408
        assert abs(a['values'][17407]-b['values'][0])<1e-6
        relation_counts.append({'kind':kind,'all_units':len(a['values']),'reverse_consistent':True})
    errors=[]
    for stage,n in [('natural',256),('capability',512)]:
        rows=get(f'/interventions/{stage}/28/5921')
        assert len(rows['rows'])==n*6 and len(rows['controls'])==n*3 and len(rows['replay'])==n
        sample=get(f'/interventions/{stage}/28/5921?sample={n-1}')
        assert len(sample['rows'])==6 and sample['input_tokens'] and sample['input_text']
    for route,status in [('/units/35/17408',422),('/layers/27',422),('?snapshot=wrong',409),('/relations/35/0',422)]:
        try:get(route);raise AssertionError('Guard did not reject '+route)
        except urllib.error.HTTPError as e:assert e.code==status;errors.append({'route':route,'status':e.code})
    save_result(C.OUT/'api-acceptance.json',{'passed':True,'base':BASE,'layers':layers,'units':units,
        'relation_checks':relation_counts,'guard_checks':errors,'report_manifest_sha256':report['manifest_sha256'],
        'scope':'Live local center HTTP vs real native BF16 archive; not a remote deployment or independent scientific replication.'})

if __name__=='__main__':main()
