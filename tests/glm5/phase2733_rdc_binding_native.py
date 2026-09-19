"""All-unit exact gate/up bilinear parameter geometry, no huge dense Kronecker."""
from rdc_binding_common import *

def main():
    import torch
    from rdc_law_native import parameter
    out=BASE/'native_bilinear'
    if (out/'result.json').exists():return
    start=time.monotonic();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;guard(1400*1024**2)
    reports=[]
    for b in (6,16,35):
        wg=parameter(f'model.layers.{b}.mlp.gate_proj.weight');wu=parameter(f'model.layers.{b}.mlp.up_proj.weight')
        m,d=wg.shape
        # C[k,i,j] = Wg[k,i] Wu[k,j]. Every i,j pair is represented exactly.
        gram=(wg@wg.T)*(wu@wu.T);den=gram.diag().clamp_min(1e-30).sqrt()
        cosine=gram/den[:,None]/den[None,:]
        npz(out/f'block{b}_all_units.npz',bilinear_gram=gram.cpu().numpy(),bilinear_cosine=cosine.cpu().numpy(),
            gate=wg.cpu().numpy(),up=wu.cpu().numpy())
        rng=np.random.default_rng(2733+b);a=torch.tensor(rng.normal(size=(7,d)),device='cuda',dtype=torch.float32)
        g=a@wg.T;u=a@wu.T
        explicit=(g.sigmoid()*g*u)
        native=torch.nn.functional.silu(g)*u
        error=float((explicit-native).abs().max());relative=float((explicit-native).square().sum().sqrt()/native.square().sum().sqrt())
        reports.append({'block':b,'units':m,'coordinates':d,'dense_same_unit_bilinear_entries':m*d*d,
          'dense_same_unit_FP32_bytes':4*m*d*d,'dense_full_weight_Kronecker_FP32_bytes':4*(m*d)**2,
          'explicit_factor_entries':2*m*d,'all_unit_pairs':m*m,'identity_absolute_error':error,'identity_relative_error':relative,
          'offdiagonal_cosine_rms':float(((cosine.square().sum()-cosine.diag().square().sum())/(m*m-m)).sqrt()),
          'diagonal_min':float(gram.diag().min()),'diagonal_max':float(gram.diag().max()),'array_sha':sha(out/f'block{b}_all_units.npz')})
        del wg,wu,gram,cosine,den
        print('ALL_UNIT_BILINEAR',reports[-1],flush=True)
    save(out/'result.json',{'timestamp':stamp(),'source':snapshot(Path(__file__)),'blocks':reports,'seconds':time.monotonic()-start,
      'definition':'C_kij = Wg_ki Wu_kj; <C_k,C_l> = <Wg_k,Wg_l><Wu_k,Wu_l>; activation_k(x) = sigmoid(Wg_k x) * x^T C_k x.',
      'scope':'Exact native parameter factorization/Gram for every unit and coordinate, not a learned universal language law or original training history.',
      'limits':['Sigmoid gate and contextual input remain essential; C alone is not the MLP.','Parameter tensor similarity is not functional identity or a knowledge localization.','Different layer indices/units are not assumed semantically aligned.']})
    ledger('native_bilinear_all_units',time.monotonic()-start)

if __name__=='__main__':main()

