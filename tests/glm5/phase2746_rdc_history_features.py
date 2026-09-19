"""Available-history native candidate features; future states kept only as targets."""
from rdc_construction_common import *
from rdc_construction_storage import FIELD_STORE,verify_storage
from rdc_native_tail import loaded_block,checkpoint_tensor,final_norm,config,cuda_singleton,CUDA_TASKS
from rdc_history_prediction import past_cache,tensor,position_factors,complete_block,source_permutation
from phase2746_rdc_history_prediction_contract import OUT,freeze


def main():
    import torch
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
    cuda_singleton(CUDA_TASKS|{Path(__file__).name});verify_storage(512*1024**2)
    start=time.monotonic();protocol,rows=freeze();n=len(rows)
    directory=OUT/'features';file=FIELD_STORE/'history_features/points.npz'
    if (directory/'result.json').exists():
        rec=read(directory/'result.json');assert sha(BASE/rec['field_path'])==rec['field_sha256'];return
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    try:
        c=config();rotary=Qwen3RotaryEmbedding(c,device='cuda')
        layer=loaded_block(35,torch.float32);native=loaded_block(35,torch.bfloat16)
        norm=checkpoint_tensor('model.norm.weight',torch.float32)
        result={k:np.empty((n,2560),np.uint16) for k in ['H0','H12','target_H35','target_H36','target_postnorm','oracle_BF16_H36','oracle_BF16_postnorm']}
        result.update({k:np.empty((n,2560),np.float32) for k in ['candidate_native','candidate_source_value_shuffled','oracle_FP32_H36','oracle_FP32_postnorm']})
        result['target_Q35']=np.empty((n,32,128),np.uint16)
        diagnostics=[]
        with torch.no_grad():
            for begin in range(0,n,3):
                group=rows[begin:begin+3];assert len({r['sample_id'] for r in group})==1
                source=BASE/group[0]['field_path'];assert sha(source)==group[0]['field_sha256']
                with np.load(source) as z:
                    h=z['hidden'][:3];post=z['postnorm'][:3];q=z['Q_before_RoPE'][:3,35]
                    for offset,row in enumerate(group):
                        index=begin+offset;step=row['step'];position=len(row['prompt_ids'])-1
                        kk,vv=past_cache(z,step)
                        assert kk.shape==vv.shape==(8,position,128)
                        key,value=tensor(kk)[None],tensor(vv)[None]
                        cos,sin=position_factors(rotary,position)
                        # Only H0/H12/pastKV/position enter these feature calls.
                        early=tensor(h[step,12])[None,None]
                        candidate=complete_block(layer,early,key,value,cos,sin)
                        permutation=source_permutation(row['sample_id'],step,position)
                        perm=torch.tensor(permutation,device='cuda',dtype=torch.long)
                        shuffled=complete_block(layer,early,key,value.index_select(-2,perm),cos,sin)
                        result['H0'][index]=h[step,0];result['H12'][index]=h[step,12]
                        result['candidate_native'][index]=candidate[0,0].cpu().numpy()
                        result['candidate_source_value_shuffled'][index]=shuffled[0,0].cpu().numpy()
                        # Future fields are attached only after all predictor
                        # feature calls above. They never construct X.
                        result['target_H35'][index]=h[step,35];result['target_H36'][index]=h[step,36]
                        result['target_postnorm'][index]=post[step];result['target_Q35'][index]=q[step]
                        actual=tensor(h[step,35])[None,None]
                        oracle=complete_block(layer,actual,key,value,cos,sin)
                        original=complete_block(native,actual.bfloat16(),key.bfloat16(),value.bfloat16(),cos.bfloat16(),sin.bfloat16())
                        smoothpost=final_norm(oracle,norm,c.rms_norm_eps)
                        nativepost=final_norm(original,norm.bfloat16(),c.rms_norm_eps)
                        result['oracle_FP32_H36'][index]=oracle[0,0].cpu().numpy()
                        result['oracle_FP32_postnorm'][index]=smoothpost[0,0].cpu().numpy()
                        result['oracle_BF16_H36'][index]=bits(original[0,0])
                        result['oracle_BF16_postnorm'][index]=bits(nativepost[0,0])
                        target=unbits(post[step]).astype(float)
                        diagnostics.append({'point_id':row['point_id'],'sample_id':row['sample_id'],'step':step,
                            'available_source_positions':position,
                            'BF16_B1_vs_observed_B8_postnorm_MSE':float(np.mean((unbits(result['oracle_BF16_postnorm'][index]).astype(float)-target)**2)),
                            'BF16_B1_vs_observed_B8_H36_bit_equal':bool(np.array_equal(result['oracle_BF16_H36'][index],h[step,36])),
                            'same_valued_FP32_vs_observed_BF16_postnorm_MSE':float(np.mean((result['oracle_FP32_postnorm'][index].astype(float)-target)**2))})
                if (begin+3)%96==0:print('HISTORY_FEATURES',begin+3,n,round(time.monotonic()-start,1),flush=True)
        assert all(np.isfinite(unbits(a) if a.dtype==np.uint16 else a).all() for a in result.values())
        verify_storage(sum(a.nbytes for a in result.values()));npz(file,**result)
        compressed(directory/'numerical_diagnostics.json.gz',diagnostics)
        meta={'timestamp':stamp(),'all_passed':True,'source':snapshot(__file__),
            'helper':snapshot(Path(__file__).with_name('rdc_history_prediction.py')),
            'native_helper':snapshot(Path(__file__).with_name('rdc_native_tail.py')),
            'protocol_sha256':sha(OUT/'protocol.json'),'points':n,'source_rows':n//3,
            'field_path':file.relative_to(BASE).as_posix(),'field_sha256':sha(file),'field_bytes':file.stat().st_size,
            'axes':{k:list(a.shape) for k,a in result.items()},'row_ids':[r['point_id'] for r in rows],
            'input_arrays':['H0','H12','candidate_native','candidate_source_value_shuffled'],
            'target_or_oracle_arrays':[k for k in result if k.startswith('target_') or k.startswith('oracle_')],
            'same_valued_FP32_native_block':True,'maximum_BF16_B1_B8_postnorm_MSE':max(r['BF16_B1_vs_observed_B8_postnorm_MSE'] for r in diagnostics),
            'numerical_diagnostics_sha256':sha(directory/'numerical_diagnostics.json.gz'),
            'seconds':time.monotonic()-start,'scope':protocol['scope']}
        save(directory/'result.json',meta);ledger('phase2746_history_features',meta['seconds'])
        print('HISTORY_FEATURES_COMPLETE',n,meta['field_bytes'],flush=True)
    except Exception as exc:
        failure(directory,start,exc);raise


if __name__=='__main__':main()
