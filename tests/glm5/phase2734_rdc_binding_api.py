"""Actual read-only API checks; no CUDA or demo outputs."""
import urllib.request
import urllib.error
import urllib.parse
from rdc_binding_common import *

def main():
    start=time.monotonic();records=[]
    def get(route,params=None,status=200):
        url='http://127.0.0.1:5001'+route+('?' + urllib.parse.urlencode(params) if params else '')
        try:
            with urllib.request.urlopen(url,timeout=60) as response:code=response.status;body=response.read()
        except urllib.error.HTTPError as exc:code=exc.code;body=exc.read()
        assert code==status,(url,code,body[:200]);records.append({'route':route,'params':params,'status':code,'bytes':len(body)})
        return json.loads(body)
    overview=get('/api/rdc-binding/overview');samples=get('/api/rdc-binding/samples')
    assert len(samples)==len({r['sample_id'] for r in samples})==1664
    correction=read(BASE/'signed_source/identity_recovery/result.json')
    for occurrence in correction['aliases']:
        row=get('/api/rdc-binding/sample',{'sample':occurrence['resolved_sample_id']})
        assert row['component_ids']==occurrence['component_ids'] and row['frozen_sample_id']==occurrence['frozen_sample_id']
        assert all(r['sample_id']==row['sample_id'] for r in row['connected_visibility'])
    paths=[('gum','connected_test'),('ewt','matched_test'),('program_en','train'),('program_zh','depth_test'),('gum','signed_connected'),('program_en','prospective_depth6')]
    for cohort,split in paths:
        row=next(r for r in samples if r['cohort']==cohort and r['split']==split);sid=row['sample_id']
        source=get('/api/rdc-binding/sample',{'sample':sid});assert len(source['prompt_ids'])==row['tokens']
        if not row['captured']:
            get('/api/rdc-binding/field',{'sample':sid},409)
            get('/api/rdc-binding/roles',{'sample':sid},409)
            continue
        for mode in ('all_layers','H12_sources'):
          for view in ('raw','RMS'):
            result=get('/api/rdc-binding/field',{'sample':sid,'mode':mode,'view':view})
            a=np.array(result['values']);assert a.shape[1]==2560 and np.isfinite(a).all()
        result=get('/api/rdc-binding/roles',{'sample':sid});a=np.array(result['values']);assert a.shape[1]==6 and np.max(abs(a.sum(1)-1))<1e-5
    sid=next(r['sample_id'] for r in samples if r['cohort']=='gum' and r['split']=='connected_test')
    for block in (6,16,35):
        result=get('/api/rdc-binding/scalar',{'sample':sid,'block':block,'unit':9727,'input_coordinate':2559,'output_coordinate':2559})
        assert result['input_terms']['native_width']==2560 and result['unit_terms']['native_width']==9728
    gradients={}
    for part in ('full','content','format'):
        ready=(BASE/'format_content'/f'{part}_gradient_factors.npz').exists()
        result=get('/api/rdc-binding/gradient',{'query':767,'part':part,'unit':9727,'input_coordinate':2559,'output_coordinate':2559},200 if ready else 409)
        if ready:
            assert result['input_terms']['native_width']==2560 and result['output_terms']['native_width']==9728
            assert result['factor_archive'].replace('\\','/').startswith('format_content/')
            gradients[part]=result
    if len(gradients)==3:
        for key in ('input_terms','output_terms'):
            f,c,s=[np.array(gradients[part][key]['values']) for part in ('full','content','format')]
            assert np.max(np.abs(f-c-s))/max(np.max(abs(f)),np.max(abs(c)),np.max(abs(s)),1e-12)<1e-5
    areas=get('/api/rdc-binding/areas');registered=set()
    for area in areas:
        listing=get('/api/rdc-binding/files',{'area':area['area']})
        assert len(listing)==area['files']
        registered.update(area['area']+'/'+r['file'] for r in listing)
    on_disk={p.relative_to(BASE).as_posix() for p in BASE.rglob('*.npz')}
    assert registered==on_disk,(len(registered),len(on_disk))
    for area in ('atlas/condition_profiles','confirmation','native_bilinear','middle_training/coherent_2733','gradient_span'):
        files=get('/api/rdc-binding/files',{'area':area})
        if not files:continue
        file=files[0]['file'];headers=get('/api/rdc-binding/arrays',{'area':area,'file':file});h=headers[0]
        result=get('/api/rdc-binding/array',{'area':area,'file':file,'name':h['array'],'row_count':3,'count':16})
        assert result['tensor_shape']==h['shape'] and np.isfinite(result['values']).all()
    for mode in ('qwen4','qwen14','glm4','binding','long_native','autonomous'):
        rows=get('/api/rdc-binding/behavior-index',{'mode':mode})
        if rows:
            r=rows[0];result=get('/api/rdc-binding/behavior',{'mode':mode,'sample':r['sample_id'],'branch':r['branch']})
            assert result['material']['sample_id']==r['sample_id'] and 'generated_ids' in result
    get('/api/rdc-binding/array',{'area':'../..','file':'secret.npz','name':'x'},404)
    get('/api/rdc-binding/field',{'sample':'not_registered'},404)
    get('/api/rdc-binding/scalar',{'sample':sid,'unit':9728},422)
    get('/api/rdc-binding/gradient',{'query':768},422)
    for name in ('rdc-joint','rdc-operator','rdc-law'):get('/api/'+name+'/overview')
    save(BASE/'verification/api.json',{'timestamp':stamp(),'source':snapshot(Path(__file__)),'checks':records,'count':len(records),
      'all_passed':True,'samples':len(samples),'unique_sample_ids':len({r['sample_id'] for r in samples}),'array_areas':len(areas),'registered_npz_files':len(registered),
      'every_saved_npz_registered':True,'same_precision_gradient_parts_checked':len(gradients)==3,'seconds':time.monotonic()-start,
      'scope':'Actual HTTP GET endpoints and parameter bounds; pending content/format returns409 until committed, not substituted.'})
    print('BINDING_API_PASS',len(records),'samples',len(samples),flush=True)

if __name__=='__main__':main()
