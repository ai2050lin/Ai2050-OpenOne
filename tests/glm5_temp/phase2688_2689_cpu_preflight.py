"""CPU-only scientific/pretrained-input previews, not completed phases."""
import sys,itertools
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'glm5'))
import numpy as np
import torch
import phase2688_native_qkv_terms as t
import phase2689_native_qkv_scalar as s
from phase2685_native_attention_math import self_test,local_scalar_path


def main():
    meta,primary,windows,weights=t.load_weights()
    r=t.read(t.MATERIAL)[0];source=t.FIELD/'source/case_0000.npz'
    tested=0;largest=0
    with np.load(source) as z:
        for w in windows:
            x=t.unbits(z[f'L{w["layer"]}__upstream_attention_x']).astype(np.float64)
            products=x*weights[w['key']];t.exact32(products)
            actual=t.unbits(z[f'L{w["layer"]}__upstream_linear_{w["kind"]}'])[:,w['output_row']]
            largest=max(largest,float(np.abs(actual-products.sum(-1)).max()));tested+=1
    # Exhaustive deterministic sign-mapping sanity checks including zero weights.
    d=np.array([[0.,-2.,3.],[4.,0.,-5.],[-6.,7.,0.]])
    for w in (np.array([0.,-2.,4.]),np.array([1.,1.,1.]),np.array([-1.,0.,-1.])):
        for name in ('sum','sumabs'):
            x=d.sum(0) if name=='sum' else np.abs(d).sum(0)
            expected=(d*w).sum(0) if name=='sum' else np.abs(d*w).sum(0)
            assert np.array_equal(t.transform_map(x,w,name),expected)
    controls=meta['controls'];deltas=[]
    for c in controls:
        a=torch.tensor(c['original_weight'],dtype=torch.bfloat16);original=a.clone()
        assert float(a)==c['original_weight']
        for dose,sign in itertools.product(s.DOSES,s.SIGNS):
            try:
                a.copy_(original.float()+sign*dose*c['row_RMS']);effective=float(a)-float(original)
                assert np.isfinite(effective);deltas.append(effective)
            finally:a.copy_(original)
            assert torch.equal(a,original)
    rows=[r for r in t.read(t.MATERIAL) if r['unit']==r['form']==r['roster_order']==r['mention_order']==0 and r['output_function'] in ('truth','name')]
    assert len(rows)==128 and sum(r['parameter_published'] for r in rows)==16
    m=s.map_empty();a=np.arange(2*32*7).reshape(2,32,7)/1024
    s.add_response(m,0,0,'probability',a,'native',7)
    assert np.array_equal(m['probability__native__sum'][0,0,:,:,:7],a) and not m['probability__native__sum'][0,0,:,:,7:].any()
    report={'all_checks_passed':True,'preview_only':True,'whole_phase_completion':False,'model_loaded':False,'cuda_initialized':torch.cuda.is_initialized(),
        'actual_first_published_source_windows_checked':tested,'actual_first_source_projection_max_abs_residual':largest,
        'synthetic_scalar_restore_checks':len(deltas),'synthetic_effective_weight_zero_count':sum(d==0 for d in deltas),
        'frozen_math_72synthetic':self_test(),'material128_prefixes_verified':True,'all_coordinate_map_shape_checks':True}
    assert not report['cuda_initialized'];t.save(t.OUT/'analysis/cpu_preflight.json',report)
    t.save(s.OUT/'analysis/cpu_preflight.json',report)
    print({k:v for k,v in report.items() if k!='frozen_math_72synthetic'},flush=True)


if __name__=='__main__':main()
