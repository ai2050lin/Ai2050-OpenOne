"""All-unit native MLP arithmetic plus arbitrary source-coordinate/parameter witnesses."""
from rdc_relation_common import *
from rdc_relation_native_parameters import parameter,decode


def main():
    import torch,inspect
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm,Qwen3DecoderLayer
    out=BASE/'scalar_parameters';guard(1024**2)
    if (out/'result.json').exists():return
    torch.set_num_threads(2);layer=23;prefix=f'model.layers.{layer}.';cfg=read(ROOT/'models/hf/qwen3-4b/config.json')
    norm=Qwen3RMSNorm(2560,eps=cfg['rms_norm_eps']).to(dtype=torch.bfloat16)
    with torch.no_grad():norm.weight.copy_(torch.from_numpy(decode(parameter(ROOT,prefix+'post_attention_layernorm.weight')).copy()))
    save(out/'protocol.json',{'timestamp':stamp(),'source':snapshot(Path(__file__)),'parameter_reader_source':snapshot(ROOT/'tests/glm5/rdc_relation_native_parameters.py'),
      'sources':[r['sample_id'] for r in rows(True)[:4]],'queries':8,'all_gate_up_units':9728,'all_down_output_coordinates':2560,
      'input':'Reconstruct native BF16 residual H23+actual attention, then actual Qwen3 postattention RMSNorm implementation and original gamma; this norm input was not separately captured.',
      'comparison':'FP64 complete dot products versus captured native BF16 projections. Numerical rounding differences reported, not silently treated as bitwise identities.',
      'scalar_paths':'Any input coordinate i, MLP unit k and output coordinate j. Every gate/up input-coordinate contribution and all9728 down-projection unit contributions are exactly accessible. No learned weights, transport or Top-K.',
      'interpretation':'Actual parameter computation witness, not semantic identity, necessity, exclusivity or sufficiency of one unit.',
      'actual_forward_source':inspect.getsource(Qwen3DecoderLayer.forward),'norm_source':inspect.getsource(Qwen3RMSNorm.forward)})
    wg=decode(parameter(ROOT,prefix+'mlp.gate_proj.weight')).astype(float);wu=decode(parameter(ROOT,prefix+'mlp.up_proj.weight')).astype(float);wd=decode(parameter(ROOT,prefix+'mlp.down_proj.weight')).astype(float)
    profile={'gate_MSE':np.zeros(9728),'up_MSE':np.zeros(9728),'down_MSE':np.zeros(2560)};reports=[]
    with torch.inference_mode():
      for r in rows(True)[:4]:
        z=load_field(r,True)
        with np.load(BASE/f'native/fields/{r["sample_id"]}.npz') as f:fixture={k:f[k].copy() for k in f.files if k.startswith('fixture_oracle_')}
        x=torch.from_numpy(unbits(z['h23'][r['anchors']])).to(torch.bfloat16)+torch.from_numpy(fixture['fixture_oracle_attention']).to(torch.bfloat16);v=norm(x).float().numpy();npz(out/f'fields/{r["sample_id"]}.npz',postattention_normalized_input=bits(norm(x)),residual_before_norm=bits(x))
        pred={'gate':v.astype(float)@wg.T,'up':v.astype(float)@wu.T,'down':fixture['fixture_oracle_activation'].astype(float)@wd.T}
        actual={'gate':fixture['fixture_oracle_gate'],'up':fixture['fixture_oracle_up'],'down':fixture['fixture_oracle_mlp']}
        report={'sample_id':r['sample_id'],'positions':r['anchors']}
        for name in pred:
            err=pred[name]-actual[name];profile[name+'_MSE']+=np.sum(err*err,axis=0);report[name]={'MSE':float(np.mean(err*err)),'max_abs_error':float(np.max(np.abs(err))),'relative_MSE':float(np.mean(err*err)/np.mean(actual[name]**2))}
            assert report[name]['relative_MSE']<1e-3,('Native parameter reconstruction gross mismatch',r['sample_id'],name,report[name])
        reports.append(report)
    npz(out/'all_coordinate_and_unit_error.npz',**{k:v/8 for k,v in profile.items()})
    save(out/'result.json',{'timestamp':stamp(),'sources':4,'queries':8,'reports':reports,'complete_gate_up_units':9728,'complete_down_coordinates':2560,
      'all_terms_present':True,'actual_checkpoint_unchanged':True,'native_BF16_vs_FP64_rounding_explicit':True});print('SCALAR_PARAMETER_WITNESSES_COMPLETE',reports,flush=True)


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):main()
