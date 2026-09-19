"""Independent artifact/HTTP numerical audit; never launches a model or overwrites science results."""
import argparse, sys, urllib.request, urllib.parse, urllib.error
from collections import Counter
from rdc_feature_common import *
from phase2693_rdc_feature_benchmark import partitions

API='http://127.0.0.1:5001/api/rdc'
def get(path,**query):
    url=API+path+('?' + urllib.parse.urlencode(query) if query else '')
    with urllib.request.urlopen(url,timeout=60) as r:return json.load(r)

def main(run='s1'):
    out=CAMPAIGN/run;rows=read(out/'material.json');protocol_sha=sha(out/'protocol.json')
    checks={};times=[];bytes_total=0;scalar_count=0;finite_count=0;repeated=0
    for i,r in enumerate(rows):
        c=read(out/f'commits/{r["sample_id"]}.json');assert c['sample_id']==r['sample_id'] and c['protocol_sha']==protocol_sha
        for rel,digest in c['files'].items():assert sha(out/rel)==digest
        times.append(c['elapsed_seconds'])
        path=out/f'fields/{r["sample_id"]}.npz';bytes_total+=path.stat().st_size
        with np.load(path) as z:
            h=z['h'];assert h.shape==(37,len(r['prompt_ids']),2560) and h.dtype==np.uint16
            assert np.isfinite(unbits(h)).all();scalar_count+=h.size;finite_count+=h.size
            assert all(h[0,p].shape==(2560,) for p in r['spans']['u']['positions'])
            for field in z.files:
                if field=='h' or field=='native_positions':continue
                assert np.isfinite(unbits(z[field])).all();finite_count+=z[field].size
        if i%128==0:print('AUDIT',run,i,len(rows),flush=True)
    checks['all_commit_files_sha']=True;checks['all_full_fields_finite']=True
    checks['all_37_layers_all_tokens_all_2560_coordinates']=True
    r=rows[0];sample=r['sample_id'];token=r['spans']['u']['positions'][0]
    with np.load(out/f'fields/{sample}.npz') as z:
        for field,layer,coord in [('h',0,0),('h',36,2496),('postnorm',36,2496),('q',0,4032),('a',0,9664)]:
            reply=get(f'/runs/{run}/field',sample=sample,field=field,layer=layer,layers=1,token=token,tokens=1,coordinate=coord,width=64)
            if field=='h':truth=unbits(z['h'][layer,token,coord:coord+64])
            elif field=='postnorm':truth=unbits(z[field][token,coord:coord+64])
            else:
                pos=z['native_positions'].tolist().index(token);truth=unbits(z[f'L{layer}_{field}'][pos].reshape(-1)[coord:coord+64])
            assert np.array_equal(truth,np.array(reply['values']).reshape(-1));repeated+=1
        low=unbits(z['h'][0,token]);assert np.any((np.abs(low)>0)&(np.abs(low)<.01))
    checks['http_slices_exact_including_last_coordinates']=repeated
    for path,query,code in [(f'/runs/{run}/field',dict(sample='../bad'),404),
        (f'/runs/{run}/field',dict(sample=sample,width=257),422),
        (f'/runs/{run}/field',dict(sample=sample,coordinate=999999),422),
        ('/runs/not_real/status',{},404),('/parameter',dict(component='bad'),422)]:
        try:get(path,**query);raise AssertionError('invalid query unexpectedly accepted')
        except urllib.error.HTTPError as e:assert e.code==code
    checks['invalid_queries_rejected']=5
    cursors=[];cursor=0
    while True:
        e=get(f'/runs/{run}/events',after=cursor);cursors.extend(x['cursor'] for x in e['events']);cursor=e['cursor']
        if not e['has_more']:break
    assert cursors==sorted(set(cursors));assert get(f'/runs/{run}/events',after=cursor)['events']==[]
    checks['event_cursor_idempotent']=True
    # Independent safetensors value and native-bit check, CPU row window only.
    import torch
    from safetensors import safe_open
    param=get('/parameter',component='gate',layer=11,row=9727,start=2544,count=16)
    model=ROOT/'models/hf/qwen3-4b';index=read(model/'model.safetensors.index.json')['weight_map']
    with safe_open(str(model/index[param['key']]),framework='pt',device='cpu') as f:truth=f.get_slice(param['key'])[9727:9728,2544:2560].clone()
    assert param['values']==truth.float()[0].tolist() and param['native_bits']==truth.view(torch.uint16)[0].tolist()
    checks['checkpoint_parameter_exact']=True
    details={}
    if run=='s1':
        result=read(out/'result.json');assert len(result['results'])==198 and len(result['random_word_controls'])==72 and len(result['future_field_prediction'])==16
        distributions={}
        for split,(tr,va,te) in partitions(rows).items():
            distributions[split]={name:dict(Counter(str(rows[i]['expected_yes']) for i in ids)) for name,ids in [('train',tr),('validation',va),('test',te)]}
            if split in ('word','joint'):
                types=[{(rows[i]['family_index'],rows[i]['unit']) for i in ids}|{(rows[i]['partner_family_index'],rows[i]['partner_unit']) for i in ids} for ids in (tr,va,te)]
                assert not(types[0]&types[1] or types[0]&types[2] or types[1]&types[2])
        mid='word__family__H0__A2_quadratic';reply=get('/coefficient',model_id=mid,j=2559,start=2550,count=16,target=0)
        with np.load(out/f'models/{mid}.npz') as z:
            # Explicit sample-wise sum, not the server matrix multiply implementation.
            truth=[sum(z['alpha'][i,0]*z['z_train'][i,2559]*z['z_train'][i,k] for i in range(len(z['alpha'])))/(z['raw_scale_vector'][2559]*z['raw_scale_vector'][k]) for k in range(2550,2566)]
            assert np.allclose(reply['values'],truth,rtol=1e-10,atol=1e-10)
        test=next(r for r in rows if r['word_split']=='test')
        pred=get('/prediction',sample=test['sample_id'],coordinate=2496,width=64)
        with np.load(out/'predictions/word__future_H36__A1_linear.npz') as z:
            assert np.array_equal(np.array(pred['values'])[1,0],z['prediction'][0,2496:2560])
            assert np.array_equal(np.array(pred['values'])[2,0],(z['prediction'][0]-z['target'][0])[2496:2560])
        checks['lazy_quadratic_raw_coordinate_coefficients_exact']=True;checks['future_prediction_error_field_exact']=True
        rec=read(CAMPAIGN/'reconciliation/result.json')
        for rel,digest in rec['archive_hashes'].items():assert sha(CAMPAIGN/'reconciliation'/rel)==digest
        checks['old_archive_unchanged']=len(rec['archive_hashes'])
        details={'answer_label_distributions':distributions,
          'correction':'Original joint external-answer test has only No; label polarity is confounded with word block/form. Do not interpret this result as general binary reasoning failure or mechanism closure failure. Original files retained.',
          'grouped_cases':'512 realizations of64 lexical-use entries, not512 independent concepts; joint test32 cases /16 bilingual lexical entries.',
          'pooling':'All native coordinates retained, but multi-token term spans are averaged for extractors; original all-token fields retained separately.'}
    result={'timestamp':stamp(),'run_id':run,'status':'data_and_api_audited','cases':len(rows),'checks':checks,'details':details,
        'native_hidden_scalars':scalar_count,'total_finite_scalars_checked':finite_count,'field_bytes_retained':bytes_total,
        'case_seconds_sum':sum(times),'case_seconds_mean':sum(times)/len(times),'case_seconds_max':max(times),
        'retention':'All raw fields retained for working native-coordinate client and next analyses; deleted0. Lossless BF16bit NPZ; no coordinate truncation.',
        'source_sha':sha(Path(__file__)),'protocol_sha':protocol_sha}
    save(out/'delivery_audit.json',result);print('AUDIT_DONE',run,checks,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',default='s1',choices=('s1','s2pilot'));main(p.parse_args().run)
