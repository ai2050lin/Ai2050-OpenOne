"""HTTP read-only, identity, boundary, full-coordinate and terminal-score regression."""
import argparse,sys
from rdc_update_common import *


def main(final=False):
    import requests
    start=time.monotonic();checks=[];host='http://127.0.0.1:5002/api/rdc-update';session=requests.Session()
    def get(endpoint,expected=200,**params):
        r=session.get(host+endpoint,params=params,timeout=120);assert r.status_code==expected,(endpoint,params,r.status_code,r.text[:300])
        return r.json() if r.headers.get('content-type','').startswith('application/json') else r.content
    samples=get('/samples');assert len(samples)==2176 and len({r['sample_id'] for r in samples})==2176
    checks.append('2176unique stable material identities')
    lookup={r['sample_id']:r for r in samples};overview=get('/overview')
    # Final.json includes this HTTP receipt, so it is the one deliberately
    # non-circular completion item checked by the later final-delivery audit.
    if final:assert all(v for k,v in overview['completion'].items() if k!='integrity'),overview['completion']
    sid=gzread(BASE/'natural_material.json.gz')[0]['sample_id']
    for model,width in [('qwen4',2560)]+([('qwen14',5120),('glm4',4096)] if final else []):
        rr=get('/samples',model=model);assert len(rr)==(2176 if model=='qwen4' else 128)
        if model!='qwen4':assert all(r['captured'] for r in rr)
        pick=next(r for r in rr if r['sample_id']==sid)
        raw=get('/field',sample=sid,model=model);norm=get('/field',sample=sid,model=model,view='RMS')
        assert raw['native_width']==norm['native_width']==width and raw['model']==norm['model']==model
        assert raw['sample_id']==norm['sample_id']==sid
        a=np.array(raw['values']);b=np.array(norm['values']);assert a.shape==b.shape and a.shape[-1]==width
        assert np.allclose(b,a/np.sqrt(np.mean(a*a,-1,keepdims=True)).clip(1e-8),rtol=2e-6,atol=2e-6)
        sources0=get('/field',sample=sid,model=model,mode='sources');assert len(sources0['values'])==pick['tokens']
        checks.append(f'{model} actual all-layer and all-source full{width}raw/RMS arrays')
    graph=get('/graph',sample=sid);a=np.array(graph['values']);assert a.shape==(18,18) and np.allclose(a.sum(-1),1,rtol=1e-5) and np.allclose(a.diagonal(),0)
    scalar=get('/scalar',sample=sid,unit=9727,input_coordinate=2559,output_coordinate=2559)
    assert scalar['sample_id']==sid;checks.append('Complete source graph and extreme scalar coordinates')
    gradients=[get('/gradient',query=767,part=p,unit=9727,input_coordinate=2559,output_coordinate=2559) for p in ('full','content','format')]
    for key in ('gate','up','down'):
        a,c,f=[x['scalars'][key] for x in gradients];assert abs(a-c-f)<=1e-5*max(abs(a),abs(c)+abs(f),1e-7)
    checks.append('Full/content/format extreme-index scalar gradient sum')
    pathrow=next(r for r in samples if r['native_path']);p=pathrow['anchors'][-1]
    native=get('/native-path',sample=pathrow['sample_id'],block=35,anchor=len(pathrow['anchors'])-1,source_position=p,unit=9727,output_coordinate=2559)
    assert native['unit_reads']['native_width']==9728 and len(native['chain']['attention_weights_all32heads'])==32
    checks.append('Committed native source ledger, all32heads and9728units')
    arrays=get('/arrays',area='language_analysis',file='all_condition_profiles.npz');assert {a['array'] for a in arrays}=={'raw_H','RMS_H','raw_MLP_activation'}
    v=get('/array',area='language_analysis',file='all_condition_profiles.npz',name='raw_MLP_activation',row_start=359,row_count=1,start=9700,count=28)
    assert v['tensor_shape']==[60,3,2,9728] and v['start']==9700
    with np.load(BASE/'language_analysis/all_condition_profiles.npz') as z:assert np.array_equal(np.array(v['values'])[0],z['raw_MLP_activation'].reshape(-1,9728)[359,9700:])
    checks.append('Original archive last-row/last-unit pagination exact')
    if final:
        replay=read(BASE/'causal_replay/result.json');assert overview['causal_replay']['unique_prefixes']==80
        prefix=replay['records'][0]['sample_id']
        value=get('/array',area='causal_replay',file=f'fields/{prefix}.npz',name='H',row_start=36,row_count=1,start=0,count=2560)
        with np.load(BASE/'causal_replay/fields'/f'{prefix}.npz') as z:assert np.array_equal(np.array(value['values'])[0],unbits(z['H'])[36,0])
        checks.append('Prefix-only native recovery full-coordinate array and overview')
        manual=overview['manual_terminal_audit'];assert manual['all_passed']
        for r in manual['adjudications']:
            raw=read(BASE/r['raw_record'])
            value=get('/behavior',mode='long_answers',sample=raw['sample_id'],branch=raw['branch'])
            assert value['manual_terminal_adjudication']==r
        checks.append('Every residual manual terminal annotation exposed separately without replacing automated scores')
    for endpoint,params,code in [('/field',{'sample':'not-a-source'},404),('/field',{'sample':sid,'model':'wrong'},422),('/field',{'sample':sid,'anchor':3},422),
      ('/scalar',{'sample':sid,'unit':9728},422),('/gradient',{'query':768},422),('/array',{'area':'../','file':'contract.json','name':'x'},404),
      ('/arrays',{'area':'language_analysis','file':'../contract.json'},404),('/array',{'area':'language_analysis','file':'all_condition_profiles.npz','name':'raw_H','start':2560},422),
      ('/native-path',{'sample':pathrow['sample_id'],'source_position':p+1},422),('/behavior',{'mode':'own_history','sample':sid,'branch':'../native'},404)]:get(endpoint,code,**params)
    absent=next(r['sample_id'] for r in samples if r['sample_id'] not in read(BASE/'scale/protocol.json')['source_ids'])
    get('/field',409,sample=absent,model='qwen14');assert session.post(host+'/overview',timeout=20).status_code==405
    checks.append('Unknown identity/bounds/traversal/mismatched model rejected; no POST mutation')
    scoring_meta=read(BASE/'behavior_analysis'/('result.json' if final else 'preliminary.json'))
    formal=gzread(BASE/'behavior_analysis'/scoring_meta.get('records_file','records.json.gz' if final else 'preliminary_records.json.gz'))
    for mode,n in [('own_history',1056),('same_history',108)]+([('long_answers',192),('qwen4',36),('qwen14',36),('glm4',36)] if final else []):
        rr=get('/behavior-index',mode=mode);assert len(rr)==n,(mode,len(rr))
        pick=next(r for r in formal if r['mode']==mode and r['kind']!='natural') if any(r['mode']==mode for r in formal) else rr[0]
        response=get('/behavior',mode=mode,sample=pick['sample_id'],branch=pick['branch'])
        if 'answer_scoring' in pick:assert response['answer_scoring']==pick['answer_scoring']
        if 'format_aware_scoring' in pick:assert response['format_aware_scoring']==pick['format_aware_scoring']
        assert response['material']['sample_id']==pick['sample_id']
        checks.append(f'{mode} exact trajectory identity/count and conservative terminal scoring')
    result={'timestamp':stamp(),'source':snapshot(__file__),'all_passed':True,'final':final,'checks':checks,'seconds':time.monotonic()-start,
      'scope':'Live read-only HTTP data and mathematical consistency; browser-layout regression recorded separately.'}
    save(BASE/'client'/('api_final.json' if final else 'api_preliminary.json'),result);print('UPDATE_API_PASS',len(checks),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--final',action='store_true');main(p.parse_args().final)
