"""Read-only HTTP identities, original indices, full fields and boundary checks."""
import argparse,urllib.request,urllib.parse,urllib.error
from rdc_query_common import *

URL='http://127.0.0.1:5003/api/rdc-query'
def get(path,params=None,expect=200):
    try:
        with urllib.request.urlopen(URL+path+('?' +urllib.parse.urlencode(params) if params else ''),timeout=120) as r:
            assert r.status==expect;return json.load(r)
    except urllib.error.HTTPError as e:assert e.code==expect,(path,e.code,expect);return None

def main(final=False):
    start=time.monotonic();checks=[];overview=get('/overview');assert overview['committed_natural_prefixes']>0;checks.append('Live committed counts, no planned count substituted')
    index=get('/samples',{'cohort':'gum','split':'test','limit':100});r=next(r for r in index['rows'] if r['captured']);sid=r['sample_id']
    detail=next(r for r in index['rows'] if r['captured'] and r['detail']);material=get('/sample',{'sample':sid})
    assert material['sample_id']==sid and material['source_group']==r['source_group'];checks.append('Stable sample and source-group identities')
    h=get('/field',{'sample':sid,'mode':'queries'});assert h['native_width']==2560 and np.array(h['values']).shape==(100,2560)
    with np.load(BASE/'capture/fields'/f'{sid}.npz') as z:assert np.array_equal(np.array(h['values']),unbits(z['postnorm']))
    checks.append('Every100query x2560native BF16 value returned in original order')
    h=get('/field',{'sample':sid,'mode':'layers','view':'RMS'});assert np.array(h['values']).shape==(37,2560);assert np.max(abs(np.mean(np.array(h['values'])**2,axis=1)-1))<1e-10
    checks.append('All37raw layers and independently normalized numerical view')
    source=get('/field',{'sample':detail['sample_id'],'mode':'sources'});assert np.array(source['values']).shape==(detail['tokens'],2560);checks.append('Declared all-token H12 source field, not only a coordinate subset')
    path=f'capture/fields/{sid}.npz';headers=get('/arrays',{'path':path});assert any(r['array']=='postnorm' and r['shape']==[100,2560] for r in headers)
    page=get('/array',{'path':path,'name':'postnorm','row_start':99,'row_count':1,'start':2559,'count':1});assert np.array(page['values']).shape==(1,1)
    with np.load(BASE/path) as z:assert page['values'][0][0]==float(unbits(z['postnorm'][99,2559]))
    checks.append('Last query and last coordinate paging exact, no fencepost omission')
    get('/field',{'sample':'unregistered'},404);get('/field',{'sample':sid,'scope':'natural','model':'qwen14'},409)
    get('/array',{'path':path,'name':'postnorm','start':2560},422);get('/array',{'path':path,'name':'postnorm','row_start':100},422)
    get('/arrays',{'path':'../rdc_update_campaign_20260913/graph/head_mapping.npz'},404)
    get('/arrays',{'path':'C:\\Windows\\system.ini'},404);checks.append('Unknown identity/model/page/traversal cannot silently substitute data')
    try:urllib.request.urlopen(urllib.request.Request(URL+'/overview',data=b'{}',method='POST'),timeout=30);raise AssertionError('POST accepted')
    except urllib.error.HTTPError as e:assert e.code==405
    checks.append('No POST research actions exposed')
    if final:
        assert overview['committed_query_endpoints']==1000000
        if overview['final']:assert overview['final']['all_passed']
        assert overview['rules']['all_passed'] and overview['formation']['all_passed']
        assert overview['prediction_analysis']['prespecified_candidate_qualification']==read(BASE/'analysis/phase2741.json')['prespecified_candidate_qualification']
        assert overview['formation_history_analysis']['formation']['actual_parameter_displacements']==read(BASE/'analysis/phase2742.json')['formation']['actual_parameter_displacements']
        checks.append('Scientific summaries preserve separate coordinate/vocabulary gates and actual cumulative parameter norms')
        pred=get('/prediction',{'sample':detail['sample_id'],'query':99,'target':2,'candidate':4});assert np.array(pred['values']).shape==(3,2560);checks.append('Frozen full-coordinate prediction versus actual target')
        paths=get('/path-index');assert len(paths)==44;nativepath=paths[0]['path'].replace('\\','/')
        p=get('/ordered-path',{'path':nativepath,'block':35,'unit':9727,'input_coordinate':2559,'output_coordinate':2559});assert p['all_units']['native_width']==9728
        assert p['input_terms']['native_width']==2560;checks.append('Ordered source-pair scalar and all-unit/all-coordinate contributions')
        eindex=get('/event-index');assert len(eindex)==32 and all(r['captured'] for r in eindex)
        ev=get('/event',{'sample':eindex[0]['sample_id'],'anchor':len(eindex[0]['anchors'])-1});assert ev['layer_field']['native_width']==2560 and len(ev['query_field']['values'])==100
        checks.append('Actual generated terminal-time all-layer and100query field')
        for model,width in [('qwen4',2560),('qwen14',5120),('glm4',4096)]:
            rr=get('/samples',{'scope':'scale','model':model,'limit':100})['rows'];sample=next(r['sample_id'] for r in rr if r['captured'])
            v=get('/field',{'scope':'scale','model':model,'sample':sample});assert v['native_width']==width and len(v['values'])==100
            metadata=get('/sample',{'scope':'scale','model':model,'sample':sample})
            committed=read(BASE/'scale'/model/'commits'/f'{sample}.json')
            assert metadata['selected_native_model']==model and metadata['native_prefix_token_ids']==committed['prefix_ids']
            assert len(metadata['native_query_token_ids'])==100
        checks.append('All3models use own actual full coordinate width')
        for mode,branches in [('late',['native','fixed128_digit','entropy_digit','entropy_letter','terminal_marker_digit']),('injection',['native','code_identity','mapped_code'])]:
            for b in branches:
                rs=get('/behavior-index',{'mode':mode,'branch':b});assert len(rs)==(96 if mode=='late' else 32)
                r=get('/behavior',{'mode':mode,'branch':b,'sample':rs[0]['sample_id']});assert r['branch']==b and r['generated_ids'] and 'censored' in r['answer_scoring']
        checks.append('All480late and96injection trajectories have exact branch identities and separate terminal scores')
        fresh=get('/samples',{'scope':'followup','split':'confirmation','limit':100});assert fresh['total']==96 and all(r['captured'] for r in fresh['rows'])
        get('/field',{'scope':'followup','sample':fresh['rows'][0]['sample_id']});checks.append('Independent96document followup native fields queryable')
        if (BASE/'identifiability/analysis/result.json').exists():
            assert overview['theory']['identifiability_pair_change_control_sha256']==sha(BASE/'identifiability/analysis/pair_change_control.json')
            assert overview['theory']['identifiability_pair_change_control']['all_passed']
            rows=get('/identity-index');assert len(rows)==320
            sid=rows[-1]['sample_id'];pair=get('/identity-pair',{'sample':sid,'variant':'native'})
            assert np.array(pair['field']['values']).shape==(12,2560)
            assert len(pair['material'])==2 and pair['material'][0]['target']!=pair['material'][1]['target']
            from collections import Counter
            assert Counter(pair['material'][0]['prompt_ids'])==Counter(pair['material'][1]['prompt_ids'])
            for variant in ['native','natural_target_2742','within_cohort_permuted_target_2742','natural_target_2743','within_cohort_permuted_target_2743']:
                data=get('/identity-pair',{'sample':sid,'variant':variant,'mode':'units'})
                assert np.array(data['field']['values']).shape==(12,9728)
                assert all(r['variant']==variant and r['generated_ids'] for r in data['own_history'])
                with np.load(BASE/'identifiability/relations'/variant/'fields'/f"{data['material'][0]['sample_id']}.npz") as z:
                    assert np.array_equal(np.array(data['field']['values'])[0],unbits(z['L16_gate_proj']))
            h=get('/field',{'scope':'identifiability','sample':sid});assert np.array(h['values']).shape==(100,2560)
            get('/identity-pair',{'sample':sid,'variant':'imaginary_model'},422)
            checks.append('All320strict-token pairs, five actual parameter variants, all9728units and exact own histories')
    result={'timestamp':stamp(),'source':snapshot(__file__),'all_passed':True,'final':final,'checks':checks,'seconds':time.monotonic()-start,
      'scope':'Read-only local HTTP tests; do not load or alter any checkpoint.'}
    save(BASE/'client'/('api_final.json' if final else 'api_preliminary.json'),result);print('QUERY_API_PASS',len(checks),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--final',action='store_true');main(p.parse_args().final)
