"""HTTP-only client and exact-value regression checks; no browser or model execution."""
import urllib.request,urllib.error,urllib.parse,argparse
from rdc_relation_common import *
from phase2715_rdc_prefix_relations import PrefixRelations


def main(extended=False,phase2717=False):
    root='http://127.0.0.1:5001';checks=[]
    def get(path,expected=200):
        try:
            with urllib.request.urlopen(root+path,timeout=90) as r:code=r.status;body=r.read()
        except urllib.error.HTTPError as e:code=e.code;body=e.read()
        assert code==expected,(path,code,body[:150]);checks.append({'path':path,'status':code});return json.loads(body) if body[:1] in (b'{',b'[') else body
    def query(path,**kw):return get('/api/rdc-relation/'+path+'?'+urllib.parse.urlencode(kw))
    overview=get('/api/rdc-relation/overview');assert overview['capture']['main']['units']==512 and overview['native']['queries']==256
    parser=PrefixRelations()
    for scope,fresh in [('main',False),('fresh',True)]:
        rr=rows(fresh);listed=query('samples',scope=scope);assert len(listed)==len(rr)
        for r in [rr[0],rr[1],rr[-1]]:
            sid=r['sample_id'];z=load_field(r,fresh)
            for layer in ('h12','h23','h24','h36','postnorm'):
                d=query('field',scope=scope,sample=sid,layer=layer,start=2557,count=3);assert d['native_width']==2560 and d['end']==2560;assert np.array_equal(d['values'],unbits(z[layer])[:,2557:])
            p=r['anchors'][0];g=query('sample',scope=scope,sample=sid,position=p)['prefix_graph'];pr=parser.graph(r['prompt_ids'][:p+1],r['language']);assert np.max(np.abs(np.asarray(g['all_relation_probabilities'])-pr['all_relation_probabilities']),initial=0)<1e-12
        d=query('field',scope=scope,sample=rr[0]['sample_id'],layer='h12',normalized='true',start=0,count=2560);assert len(d['values'][0])==2560
    for kind in ('current','temporal','native_all_sources','native_hybrid'):
        d=query('prediction',scope='fresh',sample=rows(True)[0]['sample_id'],kind=kind,anchor=1,start=0,count=2560);v=np.array(d['values']);assert v.shape==(3,2560) and np.allclose(v[2],v[1]-v[0],atol=1e-5);assert abs(d['MSE']-np.mean((v[1]-v[0])**2))<1e-5
    d=query('units',start=0,count=9728);assert np.array(d['values']).shape==(3,9728)
    for field in ('native_h12','native_h36','self_predicted_h36','native_h36_on_self_prefix'):
        d=query('generation',sample=rows(True)[0]['sample_id'],field=field,start=0,count=2560);assert np.array(d['field']['values']).shape==(16,2560);assert d['record']['first_token_divergence_zero_based']==1
    # All native indices including far corner are accessible; train-z matrix diagonal matches saved complete profile.
    for rel in ('nmod','conj','compound'):
        for split in ('train','test'):
            d=query('matrix',relation=rel,control='distance_pos',pair='H12_H23',split=split,view='train_z',row_start=2528,column_start=2528,count=32)
            with np.load(BASE/f'relation_atlas/profiles/{rel}_distance_pos_H12_H23_train_z.npz') as z:expected=z[split+'_diagonal'][2528:]
            assert np.allclose(np.diag(d['values']),expected,atol=2e-5,rtol=2e-5),(rel,split)
    for path in ['/field?scope=bad&sample=x','/field?scope=fresh&sample=../x','/field?scope=fresh&sample='+rows(True)[0]['sample_id']+'&start=2560','/units?start=9728','/matrix?pair=H0_H36']:
        get('/api/rdc-relation'+path,404 if '../' in path else 422)
    get('/api/rdc-relation/matrix?relation=conj&control=distance_pos_same_dependent_id&split=test',409)
    get('/api/rdc-prefix/overview');get('/api/rdc-prefix/runs/qwen4/samples');get('/api/rdc/runs')
    r=rows(True)[0];blob=query('download',scope='fresh',sample=r['sample_id']);assert __import__('hashlib').sha256(blob).hexdigest()==sha(BASE/f'fresh/fields/{r["sample_id"]}.npz')
    if extended or phase2717:
        for model in ('qwen4','qwen14','glm4'):
            folder=BASE/'scale'/model;mm=query('scale-samples',model=model);assert len(mm)==224;width=read(folder/'runtime.json')['width']
            for r in [mm[0],mm[-1]]:
                m=read(folder/f'rows/{r["sample_id"]}.json')
                for layer in ('early','late'):
                    v=query('scale-field',model=model,sample=r['sample_id'],field=layer,start=width-3,count=3)['field'];assert v['native_width']==width and np.array(v['values']).shape==(4,3)
                    if model=='qwen4':
                        with np.load(BASE/m['origin']/f'fields/{r["sample_id"]}.npz') as z:expected=unbits(z['h12'][m['positions']] if layer=='early' else z['h36'][[0,1,3,4]])
                    else:
                        with np.load(folder/f'fields/{r["sample_id"]}.npz') as z:expected=unbits(z[layer])
                    assert np.array_equal(v['values'],expected[:,-3:])
            sid=rows(True)[0]['sample_id']
            for layer in ('generation_early','generation_late'):
                v=query('scale-field',model=model,sample=sid,field=layer,start=width-3,count=3)['field']
                with np.load(folder/f'generation_fixtures/{sid}.npz') as z:expected=unbits(z[layer.removeprefix('generation_')])
                assert np.array_equal(v['values'],expected[:,-3:])
            v=query('scale-field',model=model,sample=sid,field='generation_errors',start=0,count=width)['field'];assert np.array(v['values']).shape==(16,width)
        from rdc_relation_native_parameters import parameter,decode
        for r in rows(True)[:4]:
            for anchor in (0,1):
                d=query('scalar-parameters',sample=r['sample_id'],anchor=anchor,unit=9727,input_coordinate=2559,output_coordinate=2559)
                assert np.array(d['input_terms']['values']).shape==(2,2560) and np.array(d['unit_terms']['values']).shape==(1,9728)
                assert np.allclose(np.array(d['input_terms']['values']).sum(1),d['all_input_coordinate_sums'])
                assert abs(np.array(d['unit_terms']['values']).sum()-d['all_unit_output_sum'])<1e-9
                w=decode(parameter(ROOT,'model.layers.23.mlp.down_proj.weight')[2559]);assert d['scalar_chain']['Wdown_j_k']==float(w[9727])
        idx=get('/api/rdc-relation/analysis-index')
        if not phase2717:assert any(r['area']=='output_geometry' for r in idx)
        for r in idx:
            width=r['shape'][-1];d=query('analysis-field',area=r['area'],file=r['file'],array=r['array'],start=width-3,count=3)
            with np.load(BASE/r['area']/(r['file']+'.npz')) as z:v=z[r['array']];v=v[None] if v.ndim==1 else v
            assert np.array_equal(d['values'],v[:,-3:])
        if not phase2717:
            assert overview['constant_boundary_control']['fresh_sources']==128
            r=rows(True)[0];d=query('prediction',scope='fresh',sample=r['sample_id'],kind='boundary_affine',anchor=1,start=0,count=2560)
            with np.load(BASE/f'boundary_compilation/fields/{r["sample_id"]}.npz') as z:expected=unbits(z['h24'][1])
            assert np.array_equal(d['values'][1],expected)
            for rel in ('nmod','conj','compound'):
                for split in ('train','test'):
                    d=query('matrix',relation=rel,control='distance_pos_noninitial',pair='H12_H23',split=split,view='train_z',row_start=2528,column_start=2528,count=32)
                    with np.load(BASE/f'boundary_relations/{rel}_profiles.npz') as z:expected=z[split+'_diagonal'][2528:]
                    assert np.allclose(np.diag(d['values']),expected,atol=2e-5,rtol=2e-5)
            sr=read(BASE/'surrogate_stability/result.json')
            for token in [sr['token_ids'][0],sr['token_ids'][-1]]:
                d=query('temporal-operator',token_id=token,input_start=2559,output_start=2559,count=1);assert np.array(d['values']).shape==(1,1) and np.isfinite(d['values']).all()
        get('/api/rdc-relation/scalar-parameters?sample='+rows(True)[0]['sample_id']+'&unit=9728',422)
        get('/api/rdc-relation/analysis-field?area=../&file=x&array=x',404)
    save(BASE/'verification'/('client_phase2717_api.json' if phase2717 else 'client_extended_api.json' if extended else 'client_api.json'),{'timestamp':stamp(),'passed':True,'extended':extended,'phase2717_only':phase2717,'checks':checks,'check_count':len(checks),'source':snapshot(Path(__file__)),
      'scope':'HTTP status, array identity, complete dimensions, prefix probability identity, full-matrix far-corner numeric reconstruction, input validation and legacy read-only endpoints. No browser visual QA claimed.'});print('RELATION_CLIENT_API_PASS',len(checks),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--extended',action='store_true');p.add_argument('--phase2717',action='store_true');a=p.parse_args();main(a.extended,a.phase2717)
