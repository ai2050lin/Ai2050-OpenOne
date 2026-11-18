"""Read-only native artifact accounting and parameter extraction for the two-route study."""

from pathlib import Path
from hashlib import sha256
import json
import time

import numpy as np

import paradigm_routes as X
import paradigm_routes_report as V
from deploy.storage_policy import check_write,staging_file


R,C,OUT=X.R,X.C,X.OUT


def extract_parameters(frozen):
    import torch
    from safetensors import safe_open

    model=R.MODEL
    index=json.loads((model/'model.safetensors.index.json').read_bytes())['weight_map']
    def tensor(name):
        with safe_open(model/index[name],framework='pt',device='cpu') as file:
            value=file.get_tensor(name)
            assert value.dtype==torch.bfloat16
            return value
    coordinates=frozen['units']+frozen['random_units']
    arrays={key:np.zeros((len(coordinates),5120),np.int16) for key in ('gate_rows','up_rows','down_columns','post_norm')}
    arrays['unit_indices']=np.asarray(coordinates,np.int64)
    for layer in sorted({l for l,u in coordinates}):
        print('Reading native MLP rows',layer,flush=True)
        at=[i for i,(l,u) in enumerate(coordinates) if l==layer]
        units=[coordinates[i][1] for i in at]
        prefix=f'model.layers.{layer}.'
        for key,weight in (('gate_rows','mlp.gate_proj.weight'),('up_rows','mlp.up_proj.weight'),('down_columns','mlp.down_proj.weight')):
            original=tensor(prefix+weight)
            selected=original[:,units].T if key=='down_columns' else original[units]
            arrays[key][at]=selected.contiguous().view(torch.int16).numpy()
        norm=tensor(prefix+'post_attention_layernorm.weight').view(torch.int16).numpy()
        arrays['post_norm'][at]=norm
    layer,head=frozen['head']
    for name,h in (('target',head),('control',frozen['control_head'])):
        prefix=f'model.layers.{layer}.self_attn.'
        for key,width in (('q_proj',128),('k_proj',128),('v_proj',128),('o_proj',128)):
            original=tensor(prefix+key+'.weight')
            group=h//5 if key in ('k_proj','v_proj') else h
            chosen=original[:,group*width:(group+1)*width] if key=='o_proj' else original[group*width:(group+1)*width]
            arrays[name+'.'+key]=chosen.contiguous().view(torch.int16).numpy().copy()
        for key in ('q_norm','k_norm'):
            arrays[name+'.'+key]=tensor(prefix+key+'.weight').view(torch.int16).numpy().copy()
    arrays['head_input_norm']=tensor(f'model.layers.{layer}.input_layernorm.weight').view(torch.int16).numpy().copy()
    for name in ('print','The'):
        anchors=R.read('plan.json')['anchors']
        arrays['readout.'+name]=tensor('lm_head.weight')[anchors[name]].view(torch.int16).numpy().copy()
    digest=X.write_np('native-parameters.npz',arrays)
    return arrays,digest


def run():
    import torch

    torch.set_num_threads(4)
    print('CPU audit started',C.stamp(),flush=True)
    plan=X.verify()
    frozen=R.read('candidate-freeze.json')
    params,param_sha=extract_parameters(frozen)
    direction=V.unbits(params['readout.print'])-V.unbits(params['readout.The'])
    records=[]
    started=time.monotonic()
    files=[*sorted(OUT.glob('pilot-???.npz')),*sorted(OUT.glob('discovery-???.npz')),*sorted(OUT.glob('confirm-???.npz'))]
    for i,file in enumerate(files):
        z=X.read_np(file.name)
        alpha=z['alpha'];scores=z['scores']
        normalizer=np.logaddexp.reduce(scores,axis=-1,keepdims=True)
        implied=np.exp(scores-normalizer)
        attention_error=float(abs(alpha-implied).max())
        assert attention_error<2e-5 and np.allclose(alpha.sum(-1),1,atol=2e-5)
        active=V.unbits(z['A'])
        local=(active*z['down'][None]).sum(-1)+z['head_effect'].sum(-1)
        hidden=V.unbits(z['field'])
        delta=(hidden[:,1:]-hidden[:,:-1])@direction
        residual_error=delta-local
        # Recompute the native elementwise gate multiplication using stored BF16 inputs.
        with torch.no_grad():
            gate=torch.as_tensor(z['G']).view(torch.bfloat16)
            up=torch.as_tensor(z['U']).view(torch.bfloat16)
            activation=(torch.nn.functional.silu(gate)*up).float().numpy()
        gate_error=float(abs(active-activation).max())
        family=(str(z['ids'][0]).split('-')[0],str(z['ids'][-1]).split('-')[0])
        item=dict(file=file.name,n=len(z['ids']),families=len(z['ids'])//8,first_last=family,
                  attention_reconstruction_max_abs=attention_error,GUA_native_formula_max_abs=gate_error,
                  raw_write_sum_vs_residual_max_abs=float(abs(residual_error).max()),
                  raw_write_sum_vs_residual_mean_abs=float(abs(residual_error).mean()))
        print(json.dumps(item),flush=True)
        records.append(item)
        del z,alpha,scores,active,hidden,delta,local,activation,gate,up
    result=dict(at=C.stamp(),plan=C.A.digest(OUT/'plan.json'),parameters=param_sha,records=records,
                provenance=plan['provenance'],seconds=time.monotonic()-started,
                script=Path(__file__).relative_to(R.ROOT).as_posix(),script_sha=sha256(Path(__file__).read_bytes()).hexdigest(),
                sum_scope='Raw residual print-The projection equals all native heads plus all native MLP writes up to native BF16 rounding. This local accounting includes all coordinates and all units.',
                norm_scope='Attention source softmax reconstruction is independent of the final normalization and task labels.')
    R.write('native-audit.json',result)
    print('Native audit finished',len(records),flush=True)


if __name__=='__main__':
    R.OUT,R.TEMP=OUT,X.TEMP
    run()
