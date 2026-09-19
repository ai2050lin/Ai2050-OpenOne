"""CPU production-path tests with a tiny synthetic BF16 head, not language-model evidence."""
from types import SimpleNamespace
from rdc_operator_common import *
from phase2727_rdc_operator_metric import complete_readout_geometry


def main():
    import torch
    start=time.monotonic();torch.set_num_threads(2)
    generator=torch.Generator(device='cpu').manual_seed(272701)
    w=torch.randn(37,7,generator=generator).bfloat16();h=torch.randn(7,generator=generator).bfloat16()
    fake=SimpleNamespace(config=SimpleNamespace(hidden_size=7),lm_head=SimpleNamespace(weight=w))
    results=[];directory=BASE/'metric_followup/math_geometry'
    for name,delta in [('nonzero',torch.randn(7,generator=generator).bfloat16()),('zero',torch.zeros(7,dtype=torch.bfloat16))]:
        other=h+delta;z=w@h;zz=w@other
        packet={'native_postnorm':bits(h),'joint_global_postnorm':bits(other)}
        r=complete_readout_geometry(fake,z,zz,packet,directory,{'id':'synthetic37x7_'+name,'scope':'CPU_synthetic_calibration_NOT_language_evidence'})
        with np.load(directory/'readout_geometry'/f'synthetic37x7_{name}.npz') as archive:
            g=archive['G_full_native_coordinates'];assert g.shape==(7,7)
            assert np.linalg.eigvalsh(g).min()>-1e-10
        if name=='zero':assert r['full_matrix_quadratic_form']==0 and r['observed_BF16_head_delta_variance']==0
        results.append(r)
    save(BASE/'metric_followup/readout_production_CPU_test.json',{'timestamp':stamp(),'source':snapshot(Path(__file__)),'passed':True,'cases':results,
        'scope':'Actual full-coordinate implementation exercised on a synthetic37-category/7-coordinate BF16 head, including zero displacement and PSD check. This is not a native LLM task, no CUDA model was loaded.'})
    ledger('complete_readout_production_path_CPU_test',time.monotonic()-start)
    print('COMPLETE_READOUT_CPU_PRODUCTION_TEST_PASS',flush=True)


if __name__=='__main__':main()
